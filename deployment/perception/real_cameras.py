# -*- coding: utf-8 -*-
"""真机相机标定：内参 + 刚性外参 → 每帧相机内外参。

=== 数据流 ===
    configs/real_cameras.yaml        现场填写（本模块只读）
        ↓ load_cameras()
    RealBackend.camera_info()        30 Hz：底盘位姿 + 关节角 → 世界位姿
        ↓ L3 状态帧广播（control/server.py）
    LocalBackend.camera_model(name)  L2 拿 CameraModel 做反投影

=== 外参的两种 parent ===
  base_link   固定安装（front_cam）。世界位姿 = 底盘位姿 ∘ mount。
  grasp_site  眼在手（arm_cam）。世界位姿 =
              底盘 ∘ arm_mount ∘ [T_arm_nom]⁻¹ ∘ T_site(q) ∘ mount
              其中 T_site(q) 由存储好的 MJCF 做正解（与 eef_9d 同源），
              再换回臂座系，所以标定值不随姿态变化。

=== 为什么外参要经过「臂座系」而不是直接用仿真世界系 ===
仿真正解给出的 grasp_site 位姿是**模型世界系**（臂座固定在 mount.yaml
标称位置）。真机上底盘会走、会转，臂座跟着动。先把 site 位姿换到臂座系
（常量变换只算一次），再乘真机臂座世界位姿，底盘运动只进入最后一乘。

=== 没标定时必须显式失败 ===
任何一台相机 calibrated=false 或内参为 null，camera_info 返回 {} 而不是
编一组默认值 —— 反投影错了不会抛异常，只会把目标估偏几十厘米。
"""
import os
import threading

import numpy as np
import yaml

from ..spec import SIM_DIR, config_path

_LOCK = threading.Lock()          # MJCF 正解共用一份 MjData，必须串行
_ARM_NOM = {}                     # scene -> (pos, rot)
_SITE_CACHE = {}


def _H(pos, rot):
    h = np.eye(4)
    h[:3, :3] = np.asarray(rot, float).reshape(3, 3)
    h[:3, 3] = np.asarray(pos, float).reshape(3)
    return h


def _inv(h):
    r = h[:3, :3]
    out = np.eye(4)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ h[:3, 3]
    return out


def _rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _bad(msg):
    raise ValueError("real_cameras: " + msg)


def _check_camera(name, c):
    for k in ("width", "height", "fx", "fy", "cx", "cy", "parent",
              "pos", "rot"):
        if k not in c:
            _bad("%s 缺字段 %s" % (name, k))
    if not c.get("calibrated"):
        _bad("%s 尚未标定（calibrated=false）" % name)
    for k in ("fx", "fy", "cx", "cy"):
        if c[k] is None:
            _bad("%s 的 %s 为空（先用棋盘格标定内参）" % (name, k))
    if float(c["fx"]) <= 0 or float(c["fy"]) <= 0:
        _bad("%s 的 fx/fy 必须为正" % name)
    if c["parent"] not in ("base_link", "grasp_site"):
        _bad("%s 的 parent=%r，只支持 base_link / grasp_site"
             % (name, c["parent"]))
    if np.asarray(c["pos"], float).shape != (3,):
        _bad("%s 的 pos 必须是 3 个数" % name)
    if np.asarray(c["rot"], float).shape != (3, 3):
        _bad("%s 的 rot 必须是 3x3" % name)


def load_cameras(path=None):
    """读标定文件并校验；任何一台没标定都会抛 ValueError。"""
    path = path or config_path("real_cameras.yaml")
    if not os.path.isfile(path):
        _bad("找不到标定文件 %s" % path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if raw.get("base_link_z") is None:
        _bad("缺 base_link_z（站立时 base_link 世界高度，实测）")
    cams = raw.get("cameras") or {}
    if not cams:
        _bad("cameras 段为空")
    for name, c in cams.items():
        _check_camera(name, c)
    if any(c["parent"] == "grasp_site" for c in cams.values()):
        am = raw.get("arm_mount") or {}
        if "pos" not in am or "rot" not in am:
            _bad("有 grasp_site 相机时 arm_mount(pos/rot) 必填")
    raw["base_link_z"] = float(raw["base_link_z"])
    return raw


# ---------- 仿真正解（只用来把标定值换到臂座系） ----------
def _ik(scene=None):
    import sys
    # kinematics.get_ik 按裸模块名 import ik_core；入口脚本平时由
    # bootstrap.setup() 把 SIM_DIR 放进 sys.path，这里不假定调用顺序。
    if SIM_DIR not in sys.path:
        sys.path.insert(0, SIM_DIR)
    from ..skills.kinematics import get_ik
    scene = scene or os.path.join(SIM_DIR, "p30_rm65_scene.xml")
    return scene, get_ik(scene)


def _arm_base_pose(scene=None):
    """臂座在**模型世界系**的标称位姿（常量，缓存）。"""
    scene, ik = _ik(scene)
    if scene not in _ARM_NOM:
        import mujoco
        with _LOCK:
            mujoco.mj_forward(ik.m, ik.d)
            bid = ik.m.body("arm_base_link").id
            _ARM_NOM[scene] = (
                np.array(ik.d.xpos[bid], float).copy(),
                np.array(ik.d.xmat[bid], float).reshape(3, 3).copy())
    p, r = _ARM_NOM[scene]
    return p.copy(), r.copy()


def _site_pose(q, scene=None):
    """grasp_site 在模型世界系的位姿（随关节角变化）。

    注意：这里会改共享 MjData 的 qpos，必须持锁；返回前立刻 copy。
    """
    scene, ik = _ik(scene)
    import mujoco
    q = np.asarray(q, float).reshape(-1)
    if q.size != 6:
        _bad("关节角应为 6 维，收到 %d" % q.size)
    with _LOCK:
        ik.d.qpos[ik.qadr] = q
        mujoco.mj_kinematics(ik.m, ik.d)
        pos = np.array(ik.d.site_xpos[ik.site], float).copy()
        rot = np.array(ik.d.site_xmat[ik.site],
                       float).reshape(3, 3).copy()
    return pos, rot


def camera_info(calib, base_pose, joint_rad):
    """每帧内外参。base_pose=(x, y, yaw) 或 {"pos","yaw"}。

    返回 {name: {w, h, fovy, fx, fy, cx, cy, pos, rot}}，pos/rot 为世界系。
    base_pose 不可用时返回 {}（上层会显式报「无视觉」）。
    """
    if base_pose is None:
        return {}
    if isinstance(base_pose, dict):
        pos_xy, yaw = base_pose.get("pos"), base_pose.get("yaw")
        if pos_xy is None or yaw is None:
            return {}
    else:
        # 元组口径是 (x, y, yaw)；早先这里把整个元组当 pos_xy、yaw 丢了，
        # 表现为「车转了相机不转」——0.4 rad 时前端偏 8 cm。
        bp = np.asarray(base_pose, float).reshape(-1)
        pos_xy = bp[:2]
        yaw = float(bp[2]) if bp.size >= 3 else 0.0
    h_base = _H([float(pos_xy[0]), float(pos_xy[1]),
                 float(calib["base_link_z"])], _rot_z(float(yaw)))

    arm_chain = {}
    out = {}
    for name, c in calib["cameras"].items():
        if c["parent"] == "base_link":
            h_cam = h_base @ _H(c["pos"], c["rot"])
        else:
            if not arm_chain:
                ap, ar = _arm_base_pose()
                h_arm_nom = _H(ap, ar)
                am = calib["arm_mount"]
                h_arm = h_base @ _H(am["pos"], am["rot"])
                arm_chain = {"h_arm": h_arm,
                             "h_arm_nom_inv": _inv(h_arm_nom)}
            sp, sr = _site_pose(joint_rad)
            h_cam = (arm_chain["h_arm"]
                     @ arm_chain["h_arm_nom_inv"]
                     @ _H(sp, sr) @ _H(c["pos"], c["rot"]))
        hgt = float(c["height"])
        out[name] = {
            "w": int(c["width"]),
            "h": int(c["height"]),
            "fx": float(c["fx"]),
            "fy": float(c["fy"]),
            "cx": float(c["cx"]),
            "cy": float(c["cy"]),
            "fovy": float(np.degrees(2.0 * np.arctan(
                (hgt / 2.0) / float(c["fy"])))),
            "pos": h_cam[:3, 3].copy(),
            "rot": h_cam[:3, :3].copy(),
        }
    return out
