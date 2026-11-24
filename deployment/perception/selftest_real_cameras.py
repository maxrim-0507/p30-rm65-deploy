# -*- coding: utf-8 -*-
"""真机相机标定链路自检：不需要硬件、不需要渲染。

做法：从导航场景反解出「标称安装位姿」写进临时标定文件，再让
real_cameras.camera_info() 在**随机的底盘位姿 + 关节角**下重算相机
世界位姿，与 MuJoCo 真值逐项比对。这同时验证了：

  - 内参（fx/fy/cx/cy）读写与 CameraModel 构造；
  - base_link 固定相机（front_cam）的链式变换；
  - 眼在手相机（arm_cam）的「臂座系 → 正解 → mount」链路；
  - RealBackend.camera_info/camera_model 在无硬件时的装配。

用法：python -m deployment.perception.selftest_real_cameras
"""
import os
import tempfile

import numpy as np
import yaml

from ..nav.presets import FWD2_Q_DEG, SEARCH_Q_DEG
from ..spec import CAM_H, CAM_W, SIM_DIR
from .color_grounding import CameraModel
from .real_cameras import camera_info, load_cameras

SCENE = os.path.join(SIM_DIR, "p30_rm65_nav.xml")
REF_Q = np.radians(FWD2_Q_DEG)


def _H(pos, rot):
    h = np.eye(4)
    h[:3, :3] = np.asarray(rot, float).reshape(3, 3)
    h[:3, 3] = np.asarray(pos, float).reshape(3)
    return h


def _inv(h):
    out = np.eye(4)
    out[:3, :3] = h[:3, :3].T
    out[:3, 3] = -h[:3, :3].T @ h[:3, 3]
    return out


def _angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


class _Sim(object):
    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint("arm_joint%d" % i).qposadr[0]
                              for i in range(1, 7)])
        self.jb = {n: self.m.joint(n).qposadr[0] for n in
                   ("base_slide_x", "base_slide_y", "base_hinge_z")}

    def set(self, x=0.0, y=0.0, yaw=0.0, q=None):
        self.d.qpos[:] = 0.0
        self.d.qpos[self.jb["base_slide_x"]] = x
        self.d.qpos[self.jb["base_slide_y"]] = y
        self.d.qpos[self.jb["base_hinge_z"]] = yaw
        if q is not None:
            self.d.qpos[self.qadr] = np.asarray(q, float)
        # 必须 mj_forward：cam_xpos/cam_xmat 只有 mj_camlight/mj_forward 会
        # 刷新，mj_kinematics 之后读到的还是上一帧（本项目踩过）。
        self.mj.mj_forward(self.m, self.d)

    def body(self, name):
        bid = self.m.body(name).id
        return (np.array(self.d.xpos[bid], float).copy(),
                np.array(self.d.xmat[bid], float).reshape(3, 3).copy())

    def cam(self, name):
        cid = self.m.camera(name).id
        return (np.array(self.d.cam_xpos[cid], float).copy(),
                np.array(self.d.cam_xmat[cid], float).reshape(3, 3).copy())


def _derive(sim):
    """标称安装位姿（与 real_cameras.yaml 的注释同源）。"""
    sim.set(q=REF_Q)
    pb, rb = sim.body("base_link")
    pa, ra = sim.body("arm_base_link")
    pf, rf = sim.cam("front_cam")
    sid = sim.m.site("grasp_site").id
    ps = np.array(sim.d.site_xpos[sid], float)
    rs = np.array(sim.d.site_xmat[sid], float).reshape(3, 3)
    pc, rc = sim.cam("arm_cam")
    hb = _H(pb, rb)
    return {
        "base_link_z": float(pb[2]),
        "arm_mount": {"pos": (_inv(hb) @ _H(pa, ra))[:3, 3].tolist(),
                      "rot": (_inv(hb) @ _H(pa, ra))[:3, :3].tolist()},
        "front": {"pos": (_inv(hb) @ _H(pf, rf))[:3, 3].tolist(),
                  "rot": (_inv(hb) @ _H(pf, rf))[:3, :3].tolist()},
        "arm": {"pos": (_inv(_H(ps, rs)) @ _H(pc, rc))[:3, 3].tolist(),
                "rot": (_inv(_H(ps, rs)) @ _H(pc, rc))[:3, :3].tolist()},
    }


def _write_calib(d, fx_ratio=1.0):
    fy = float((CAM_H / 2.0) / np.tan(np.radians(85.0) / 2.0))   # arm 85
    front_fy = float((CAM_H / 2.0) / np.tan(np.radians(70.0) / 2.0))
    cfg = {
        "base_link_z": float(d["base_link_z"]),
        "arm_mount": d["arm_mount"],
        "cameras": {
            "front_cam": {
                "width": CAM_W, "height": CAM_H,
                "fx": front_fy, "fy": front_fy,
                "cx": CAM_W / 2.0 + 3.0,        # 故意偏心 3 px
                "cy": CAM_H / 2.0 - 2.0,
                "parent": "base_link",
                "pos": d["front"]["pos"], "rot": d["front"]["rot"],
                "calibrated": True,
            },
            "arm_cam": {
                "width": CAM_W, "height": CAM_H,
                "fx": fy * fx_ratio, "fy": fy,
                "cx": CAM_W / 2.0, "cy": CAM_H / 2.0,
                "parent": "grasp_site",
                "pos": d["arm"]["pos"], "rot": d["arm"]["rot"],
                "calibrated": True,
            },
        },
    }
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                    encoding="utf-8")
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    f.close()
    return f.name


def t_roundtrip(sim, calp):
    cal = load_cameras(calp)
    rng = np.random.RandomState(0)
    worst_p = worst_a = 0.0
    qs = [np.radians(SEARCH_Q_DEG), np.radians(FWD2_Q_DEG),
          np.radians([10.0, -5.0, 40.0, 20.0, 10.0, 30.0])]
    for i in range(9):
        x, y = rng.uniform(-1.5, 0.5), rng.uniform(-1.0, 1.0)
        yaw = rng.uniform(-1.0, 1.0)
        q = qs[i % len(qs)]
        sim.set(x, y, yaw, q)
        info = camera_info(cal, (x, y, yaw), q)
        for name in ("front_cam", "arm_cam"):
            tp, tr = sim.cam(name)
            ep = np.linalg.norm(info[name]["pos"] - tp)
            ea = _angle(info[name]["rot"], tr)
            worst_p, worst_a = max(worst_p, ep), max(worst_a, ea)
            assert ep < 1e-6, (i, name, ep)
            assert ea < 1e-5, (i, name, np.degrees(ea))
    return "9 组底盘位姿×关节角：相机位姿误差 pos<%.1e m / rot<%.1e°" \
        % (worst_p, np.degrees(worst_a))


def t_intrinsics(calp):
    cal = load_cameras(calp)
    c = cal["cameras"]["front_cam"]
    cam = CameraModel(CAM_W, CAM_H, None, fx=c["fx"], fy=c["fy"],
                      cx=c["cx"], cy=c["cy"])
    assert abs(cam.cx - (CAM_W / 2.0 + 3.0)) < 1e-9
    assert abs(cam.fx - cam.fy) < 1e-9
    # 射线 → 与 z=0 平面求交 → 反投影回原像素
    cam.pos[:] = [0.0, 0.0, 1.0]
    # 相机 z 轴朝上 = 光轴（-z_cam）朝下
    cam.rot[:] = np.diag([1.0, -1.0, 1.0])
    for uv in [(10.0, 20.0), (CAM_W / 2.0, CAM_H / 2.0), (300.0, 200.0)]:
        p, ok = cam.hit_plane(uv[0], uv[1], 0.0)
        u, v, z = cam.project(p)
        assert ok and abs(u - uv[0]) < 1e-6 and abs(v - uv[1]) < 1e-6, \
            (uv, u, v)
    return "fx≠fovy 折算、偏心主点：hit_plane/project 往返 < 1e-6 px"


def t_refuse(calp):
    cal = load_cameras(calp)
    bad = dict(cal)
    bad["cameras"] = {k: dict(v) for k, v in cal["cameras"].items()}
    bad["cameras"]["arm_cam"]["calibrated"] = False
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                    encoding="utf-8")
    yaml.safe_dump(bad, f, allow_unicode=True, sort_keys=False)
    f.close()
    try:
        load_cameras(f.name)
    except ValueError as e:
        assert "尚未标定" in str(e), e
    else:
        raise AssertionError("未标定相机没有被拒绝")
    finally:
        os.unlink(f.name)
    return "calibrated=false → ValueError（不会静默编默认外参）"


def t_template_refused():
    """仓库里的模板必须保持「未标定」状态，避免被误当成已标定使用。"""
    try:
        load_cameras()
    except ValueError as e:
        assert "尚未标定" in str(e), e
    else:
        raise AssertionError("configs/real_cameras.yaml 居然通过了校验："
                             "模板不应带 calibrated=true")
    return "仓库模板 calibrated=false → 明确拒绝"


def t_backend(calp):
    from ..hal.real_backend import RealBackend

    class Drv(object):
        def __init__(self):
            self.q_deg = np.degrees(FWD2_Q_DEG)
            self.grip = 1.0

        def connect(self):
            pass

        def close(self):
            pass

        def get_joint_deg(self):
            return self.q_deg.copy()

        def set_gripper(self, x):
            self.grip = x

        def get_gripper(self):
            return self.grip

        def get_base_state(self):
            return {"pos": (0.25, -0.15), "yaw": 0.35, "vel": (0, 0, 0)}

    be = RealBackend(allow_base_motion=True, driver=Drv(),
                     camera_calib=calp)
    be.connect()
    info = be.camera_info()
    assert set(info) == {"front_cam", "arm_cam"}, list(info)
    cm = be.camera_model("arm_cam")
    assert cm is not None and cm.fx > 0
    assert abs(cm.cy - CAM_H / 2.0) < 1e-9
    be.close()
    return "RealBackend 装配：无硬件给出双相机内外参 + CameraModel"


def main():
    sim = _Sim()
    d = _derive(sim)
    calp = _write_calib(d)
    try:
        cases = [t_roundtrip(sim, calp), t_intrinsics(calp),
                 t_refuse(calp), t_template_refused(), t_backend(calp)]
    finally:
        os.unlink(calp)
    for i, msg in enumerate(cases, 1):
        print("  %d. %s" % (i, msg))
    print("real_cameras selftest: %d/%d OK" % (len(cases), len(cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
