# -*- coding: utf-8 -*-
"""REALMAN RM65 官方 IK 封装：把 `RM_API2` 的 `Algo` 算法库当厂商 IK 用。

=== 为什么需要这一层 ===
仿真里原本用 `ik_core`（scipy 最小二乘、5 自由度、随机多起点）充当
「厂商 IK 替身」。它能解、位置也准，但和真机控制器不是同一套算法：
  - 它约束 5 维（yaw 自由），真机 `rm_movej_p/movel` 收完整 6D；
  - 它靠最小运动正则挑分支，真机控制器按自己的规则挑；
  - 同一 eef 命令，两边的 joint2/joint5 可能落在不同构型（实测一支
    joint2=+123.5°，另一支 +84.1°）。
后果是仿真采到的腕部相机画面对应真机上不存在的关节构型，VLA 学到的
视觉—动作映射在真机上失真。

改用官方算法库后，仿真与真机是**同一份 IK**，构型分支自然一致。

=== 已实测的事实（2026-11-09）===
- `Algo` 不连机械臂即可用（`handle.id=0`），算法库版本 1.6.0-701ee1e6；
- 关节点位/限位与 MuJoCo 模型一致（SDK `[178,130,135,178,128,360]`）；
- 单步解 0.006 ms，全解 0.035 ms（比 scipy 快三个数量级）；
- 反解 + 正解闭环误差 < 1e-7。

=== 坐标系（三处都不能省）===
1. **基座系**：SDK 的位姿在机器人基座系，而我们的 eef_9d 是「世界系，
   臂座 yaw=180°」。实测同一关节角下 SDK 正解 x 与我们反号，即
       p_sdk = Rz(180°) · (p_world - arm_base_world)
   漏掉它会让目标整体跑到基座背后。
2. **欧拉角顺序**：SDK 头文件只写 "rad" 没写顺序。用
   `rm_algo_euler2matrix` 做单轴 0.5 rad 自检，六种组合里只有
   `Rz·Ry·Rx`（RPY 外旋）误差 5.96e-08，其余都在 2e-2 以上。
3. **工具坐标系**：`grasp_site` 相对法兰是 pos=(0,0,0.1113)、
   R=[[1,0,0],[0,0,-1],[0,1,0]]，对应 RPY=(π/2,0,0)。注册后
   `rm_algo_inverse_kinematics` 的目标位姿**直接就是夹持面**，
   不需要在调用方做矩阵乘（少一处符号错的机会）。

=== Python 3.8 兼容 ===
SDK 的 `rm_robot_interface.py` / `rm_ctypes_wrap.py` 用了 `tuple[...]`
这类 PEP 585 注解（3.9+），在 3.8 下 import 直接 TypeError。这里在加载
时读源码、前置 `from __future__ import annotations` 再 exec 进自建
module —— 语义等价，且不改 `third_party/` 里的文件。

注意 `site-packages` 里可能装了同名 `Robotic_Arm`（内容不同），所以
必须按显式路径加载，不能裸 `import Robotic_Arm`。

=== 失败语义 ===
官方 IK 返回 int：0 成功、1 失败、-1 上一时刻角超限、-2 四元数不合法。
失败时**没有残差数值**，无法像 scipy 那样区分「不可达」与「一步跨太大」。
按部署约定：先用种子做单步解；失败再调 `inverse_kinematics_all` 拿 8 组
全解，挑离种子最近的一组。全解也失败才算不可达。
"""
import os
import sys
import types

import numpy as np

# 机械臂安装偏航（mount.yaml）。SDK 基座系与我们的世界系差这个角度。
MOUNT_YAW_DEG = 180.0
# grasp_site 相对法兰的位姿（MuJoCo 实测，见 p30_rm65_scene.xml）
TOOL_POS = (0.0, 0.0, 0.1113)
TOOL_RPY = (np.pi / 2.0, 0.0, 0.0)

_SDK_LOADED = {}


def _sdk_dir():
    """定位 `third_party/RM_API2/Python/Robotic_Arm`。

    从本文件向上找仓库根（含 deployment/ 与 third_party/ 的那一级）。
    不依赖 cwd，因为 Thor 上的部署副本目录结构不同。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    for _ in range(6):
        cand = os.path.join(cur, "third_party", "RM_API2", "Python",
                            "Robotic_Arm")
        if os.path.isdir(cand):
            return cand
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    raise FileNotFoundError(
        "找不到 RM_API2 的 Robotic_Arm 包；期望 <repo>/third_party/"
        "RM_API2/Python/Robotic_Arm")


# 合成包名。**不能**叫 "Robotic_Arm"：site-packages 里有同名包，注册成同名
# 会让 `from .rm_ctypes_wrap import *` 解析到那份未加固的 3.9+ 副本，
# 在 py3.8 下直接 TypeError（`tuple[...]` 不可下标）。
_PKG = "_rm_sdk_shim"


def _ensure_pkg(sdk_dir):
    """建一个合成父包，使 SDK 内部的相对导入命中我们的 shim 而非 site-packages。"""
    mod = sys.modules.get(_PKG)
    if mod is not None:
        return mod
    mod = types.ModuleType(_PKG)
    mod.__file__ = os.path.join(sdk_dir, "__init__.py")
    mod.__path__ = [sdk_dir]
    mod.__package__ = _PKG
    sys.modules[_PKG] = mod
    return mod


def _load_module(path, name):
    """以 `from __future__ import annotations` 方式加载 3.9+ 写法的 SDK 模块。"""
    full = _PKG + "." + name
    if full in _SDK_LOADED:
        return _SDK_LOADED[full]
    src = io_read(path)
    src = "from __future__ import annotations\n" + src
    mod = types.ModuleType(full)
    mod.__file__ = path
    mod.__package__ = _PKG
    sys.modules[full] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    _SDK_LOADED[full] = mod
    return mod


def io_read(path):
    if sys.version_info >= (3, 9):
        with open(path, encoding="utf-8") as f:
            return f.read()
    import io
    with io.open(path, encoding="utf-8") as f:
        return f.read()


def load_sdk():
    """返回 (rm_robot_interface module, rm_ctypes_wrap module)。幂等。"""
    if "ri" in _SDK_LOADED:
        return _SDK_LOADED["ri"], _SDK_LOADED["ctypes"]
    d = _sdk_dir()
    _ensure_pkg(d)
    # 先装 ctypes 封装，再装接口层（后者 import 前者）
    cw = _load_module(os.path.join(d, "rm_ctypes_wrap.py"), "rm_ctypes_wrap")
    ri = _load_module(os.path.join(d, "rm_robot_interface.py"),
                      "rm_robot_interface")
    _SDK_LOADED["ri"], _SDK_LOADED["ctypes"] = ri, cw
    return ri, cw


def rpy_from_matrix(R):
    """旋转矩阵 → RPY（Rz·Ry·Rx 外旋），与 SDK 的 euler 约定一致。"""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    sp = float(np.clip(-R[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sp))
    # 万向锁时 roll 与 yaw 只有组合有意义，取 roll=0
    if abs(sp) > 1.0 - 1e-9:
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw])


def matrix_from_rpy(rpy):
    """RPY（Rz·Ry·Rx 外旋）→ 旋转矩阵。rpy_from_matrix 的逆。"""
    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


class RmAlgoIK:
    """RM65 官方 IK/FK，接口与 ik_core.IK 的常用部分对齐。

    所有公开方法的位姿都在**我们的世界系**（与 eef_9d 同口径）：
    pos 相对 `arm_base`（**xyz 三个分量都相对臂座**），R 是 3×3 旋转矩阵。
    """

    def __init__(self, arm_base, mount_yaw_deg=MOUNT_YAW_DEG,
                 tool_pos=TOOL_POS, tool_rpy=TOOL_RPY, arm_model="RM_65"):
        ri, _ = load_sdk()
        self.ri = ri
        self.arm_base = np.asarray(arm_base, dtype=float).reshape(3)
        self.mount_yaw = float(np.radians(mount_yaw_deg))
        self.algo = ri.Algo(
            getattr(ri.rm_robot_arm_model_e, "RM_MODEL_%s_E" % arm_model),
            ri.rm_force_type_e.RM_MODEL_RM_B_E)
        self.arm_dof = int(self.algo.arm_dof)
        # SDK 限位是度（实测 -178..178），这里统一转 rad 供调用方比较
        self._lo = np.radians(np.array(self.algo.rm_algo_get_joint_min_limit(),
                                      float))
        self._hi = np.radians(np.array(self.algo.rm_algo_get_joint_max_limit(),
                                      float))
        # 注册夹持面为工具坐标系：之后 IK 目标 / FK 输出都是 grasp_site 位姿
        self.algo.rm_algo_set_toolframe(ri.rm_frame_t(
            "grasp_site", tuple(tool_pos) + tuple(tool_rpy),
            0.0, 0.0, 0.0, 0.0))
        # 连续控制场景用单步模式（比遍历模式快，且不会跨分支乱跳）
        self.algo.rm_algo_set_redundant_parameter_traversal_mode(False)

    # ---------- 坐标换算 ----------
    def _world_to_sdk(self, p_world, R_world=None):
        """世界系（相对臂座）→ SDK 基座系。p 的 xyz 都相对臂座。"""
        p = np.asarray(p_world, dtype=float).reshape(3)
        # 我们的「相对臂座」已经是 xy 减过臂座的；这里把 z 也拿回臂座系
        rel = np.array([p[0] - self.arm_base[0], p[1] - self.arm_base[1],
                        p[2] - self.arm_base[2]], dtype=float)
        Rz = np.array([[np.cos(self.mount_yaw), -np.sin(self.mount_yaw), 0.0],
                       [np.sin(self.mount_yaw), np.cos(self.mount_yaw), 0.0],
                       [0.0, 0.0, 1.0]])
        p_sdk = Rz @ rel
        R_sdk = None if R_world is None else Rz @ np.asarray(R_world, float)
        return p_sdk, R_sdk

    def _sdk_to_world(self, p_sdk, R_sdk=None):
        Rz = np.array([[np.cos(self.mount_yaw), -np.sin(self.mount_yaw), 0.0],
                       [np.sin(self.mount_yaw), np.cos(self.mount_yaw), 0.0],
                       [0.0, 0.0, 1.0]])
        p = Rz.T @ np.asarray(p_sdk, float).reshape(3)
        p_world = np.array([p[0] + self.arm_base[0],
                            p[1] + self.arm_base[1],
                            p[2] + self.arm_base[2]])
        R_world = None if R_sdk is None else Rz.T @ np.asarray(R_sdk, float)
        return p_world, R_world

    # ---------- 求解 ----------
    def _ik_once(self, p_sdk, rpy, q_seed_deg):
        tgt = [float(p_sdk[0]), float(p_sdk[1]), float(p_sdk[2]),
               float(rpy[0]), float(rpy[1]), float(rpy[2])]
        params = self.ri.rm_inverse_kinematics_params_t(
            [float(v) for v in q_seed_deg], tgt, 1)      # 1 = 欧拉角
        ret, q = self.algo.rm_algo_inverse_kinematics(params)
        return int(ret), np.radians(np.asarray(q, dtype=float))

    def _ik_all(self, p_sdk, rpy, q_ref_deg):
        tgt = [float(p_sdk[0]), float(p_sdk[1]), float(p_sdk[2]),
               float(rpy[0]), float(rpy[1]), float(rpy[2])]
        params = self.ri.rm_inverse_kinematics_params_t(
            [float(v) for v in q_ref_deg], tgt, 1)
        allsol = self.algo.rm_algo_inverse_kinematics_all(params)
        n = int(allsol.num)
        return [np.radians(np.asarray(allsol.q_solve[i][:self.arm_dof], float))
                for i in range(n)]

    def _wrap_to_seed(self, q, q_seed):
        """把 2π 等价的关节角折到离种子最近且**在本体限位内**的那个。

        === 为什么必须做 ===
        SDK 的 joint6 限位是 ±360°，`_ik_once` 实测会直接返回 -360.000°。
        而 MuJoCo 模型和真机的关节量程都略小于 360°（MJCF 359.82°，
        官方 `[±178,±130,±135,±178,±128,±360]`），-360° 下发会被位置
        驱动器/限位裁掉，导致该帧末端停在别处。折到等价角（0°）即可：
        物理姿态完全相同，只是写法不同。

        同理 joint6 从 +123° 一路插值到 -180° 会绕 300°，折算后是 0°。
        """
        seed = np.asarray(q_seed, float).reshape(self.arm_dof)
        out = np.array(q, dtype=float).reshape(self.arm_dof)
        for i in range(out.size):
            cand = out[i] + 2 * np.pi * np.round((seed[i] - out[i]) / (2 * np.pi))
            lo, hi = self._lo[i], self._hi[i]
            if lo - 1e-9 <= cand <= hi + 1e-9:
                out[i] = float(np.clip(cand, lo, hi))
        return out

    def _angdiff_deg(self, q, q_seed):
        """按 2π 等价角算最大单关节差（度）。joint6 行程 ±2π 必须这样比。"""
        d = (np.asarray(q, float) - np.asarray(q_seed, float) + np.pi) \
            % (2 * np.pi) - np.pi
        return float(np.degrees(np.abs(d).max()))

    def solve(self, pos_world, R_world, q_seed, return_diag=False):
        """种子局部解；失败则从 8 组全解里挑离种子最近的。

        pos_world: 3，grasp_site 目标（xyz 均相对臂座）
        R_world:  3×3，世界系姿态
        q_seed:   6×rad，上一帧关节角（保持分支连续）

        返回 (ok, q_rad, info)。info = {"err": 0.0|1.0, "ret": int,
        "n_all": int, "jump_deg": float}

        **任何路径返回的 q 都先过 `_wrap_to_seed`**：SDK 会给出
        joint6=-360° 这种「物理等价但超本体限位」的写法，不折会让
        下游 ctrl 被裁。
        """
        p_sdk, R_sdk = self._world_to_sdk(pos_world, R_world)
        rpy = rpy_from_matrix(R_sdk)
        seed = np.asarray(q_seed, float).reshape(self.arm_dof)
        seed_deg = np.degrees(seed)

        ret, q = self._ik_once(p_sdk, rpy, seed_deg)
        if ret == 0:
            q = self._wrap_to_seed(q, seed)
            info = {"err": 0.0, "ret": 0, "n_all": 0,
                    "jump_deg": self._angdiff_deg(q, seed)}
            return True, q, info

        # 单步解失败 → 全解里挑最接近种子的
        sols = self._ik_all(p_sdk, rpy, seed_deg)
        if not sols:
            return False, seed.copy(), {
                "err": 1.0, "ret": ret, "n_all": 0, "jump_deg": 0.0}
        sols = [self._wrap_to_seed(s, seed) for s in sols]
        diffs = [self._angdiff_deg(s, seed) for s in sols]
        k = int(np.argmin(diffs))
        return True, sols[k], {"err": 0.0, "ret": ret, "n_all": len(sols),
                               "jump_deg": diffs[k]}

    def solve_all(self, pos_world, R_world):
        """返回全部解析解（诊断 / 可达性分析用）。"""
        p_sdk, R_sdk = self._world_to_sdk(pos_world, R_world)
        rpy = rpy_from_matrix(R_sdk)
        return self._ik_all(p_sdk, rpy, np.zeros(self.arm_dof))

    def fk(self, q_rad):
        """正解：关节角 → (pos_world, R_world)，输出在 grasp_site（工具系）。"""
        q_deg = np.degrees(np.asarray(q_rad, float).reshape(self.arm_dof))
        pose = self.algo.rm_algo_forward_kinematics([float(v) for v in q_deg], 1)
        p_sdk = np.asarray(pose[:3], float)
        R_sdk = matrix_from_rpy(pose[3:6])
        return self._sdk_to_world(p_sdk, R_sdk)

    def limits_rad(self):
        """官方关节限位（rad），用于越界判断/夹持。"""
        return self._lo.copy(), self._hi.copy()

    def tool_tilt_deg(self, q_rad):
        """工具轴（工具系 z）与世界 -z 的夹角，诊断用。"""
        _, R = self.fk(q_rad)
        z = R[:, 2]
        return float(np.degrees(np.arccos(np.clip(z @ np.array([0.0, 0.0, -1.0]),
                                                  -1.0, 1.0))))
