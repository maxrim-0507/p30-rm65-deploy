# -*- coding: utf-8 -*-
"""L2 的运动学接口：包一层已固化的 ik_core，只暴露任务需要的三件事。

=== 为什么直接复用 ik_core 而不重写 ===
ik_core 里的每条规则都是踩坑得出的（5 自由度而非 6、内死区外退、
零空间正则、joint6 对齐方块 yaw）。重写一遍等于重踩一遍，而且
仿真与部署会算出不同解 —— 那是最难查的一类不一致。

=== 分工（2026-11-06 调整）===
机械臂只接受末端位姿，IK 由机械臂控制器内部算。所以本模块不再是
部署执行路径的一部分，只剩两个用途：
  - 仿真：`MujocoBackend.actuate_pose` 用 ik_core 充当「厂商 IK 替身」；
  - L1 可达性预检 / Pregrasp 经典闭环：只求解、用于拒绝或生成关节链。
VLA 的动作执行（ManipulateSkill → L3 → 真机）不再调用 solve_near。

=== 幂等 ===
同一个进程反复构造 IK 会重复加载 MJCF（~30 ms）。这里做进程内单例，
按场景路径缓存。
"""
import os

import numpy as np

from ..spec import CUBE_HALF, SIM_DIR, TABLE_TOP_Z

_CACHE = {}


def get_ik(scene=None):
    """返回进程内共享的 IK 实例。"""
    scene = scene or os.path.join(SIM_DIR, "p30_rm65_scene.xml")
    if scene not in _CACHE:
        from ik_core import IK                    # simulation/mujoco 在 sys.path
        _CACHE[scene] = IK(scene)
    return _CACHE[scene]


def world_target_to_ik_xy(xy_world, base_xy, base_yaw):
    """世界目标 → ik_core 静态模型的世界坐标。

    === 为什么需要这一步 ===
    ik_core 的 MJCF 里臂基座固定在 ARM_BASE、底盘偏航为 0。底盘开走以后
    （运动学挂载场景），同一个世界目标在**臂基座系**里的坐标变了；把目标
    换算回「模型世界」再解 IK，得到的关节角才是相对当前臂基座正确的解。

    推导：真实臂基座世界位姿 = p_c + Rz(θ)·(mount_offset)，朝向 Rz(θ+π)；
    静态模型朝向 Rz(π)。两者相差 Rz(θ)，于是
        p' = ARM_BASE + Rz(-θ) · (p_world - p_armbase)
    其中 p_armbase 用同一个 mount_offset 从底盘位姿推出来。

    xy_world: (x, y) 世界系
    base_xy:  底盘（base_planar）世界 (x, y)
    base_yaw: 底盘偏航（rad）
    """
    from ..spec import ARM_BASE
    base_xy = np.asarray(base_xy, float).reshape(2)
    c, s = np.cos(float(base_yaw)), np.sin(float(base_yaw))
    ox, oy = float(ARM_BASE[0]), float(ARM_BASE[1])
    abx = base_xy[0] + c * ox - s * oy
    aby = base_xy[1] + s * ox + c * oy
    d = np.asarray(xy_world, float).reshape(2) - np.array([abx, aby])
    bx = c * d[0] + s * d[1]
    by = -s * d[0] + c * d[1]
    return np.array([ox + bx, oy + by])


def grasp_targets(pos_xy, table_top_z=None):
    """方块 (x,y) → (pre, hover, grasp, lift) 四个目标点。

    高度、内死区外退全部由 ik_core 决定（其中 grasp_z 依赖台上高度与
    grasp_site 的 3.6 cm 偏移，这两个数改过一次，别在这里复算）。
    """
    from ik_core import approach_points, grasp_z_for, lift_point
    from ..spec import ARM_BASE

    table_top_z = TABLE_TOP_Z if table_top_z is None else table_top_z
    gz = grasp_z_for(table_top_z)
    pre, hov, grp = approach_points(tuple(pos_xy), gz, np.asarray(ARM_BASE))
    up = lift_point(tuple(pos_xy), gz, np.asarray(ARM_BASE))
    return pre, hov, grp, up


def solve_approach(pos_xy, q_seed=None, cube_yaw=None, scene=None):
    """求「预抓 → 悬停 → 抓取 → 抬升」整链的关节角。

    返回 {"ok": bool, "q": {pre,hover,grasp,lift}, "err": float, "reason": str}

    === 种子必须喂给抓取点 ===
    `solve_chain` 的 seed 作用在 pos_list[0] 上。把抓取点的解当整链的
    首点种子，构型差得很远，实测成功率 0%。正确顺序是：先单独解抓取点，
    再以它为种子向两侧展开。
    """
    ik = get_ik(scene)
    pre, hov, grp, up = grasp_targets(pos_xy)
    rng = np.random.default_rng(20261104)

    if q_seed is None:
        e_g, q_g = ik.solve(grp, rng, n_restart=32)
    else:
        # 实测关节角可能比 MJCF 限位多出 ~1e-7（FWD2 的 joint3 正好压在
        # 134.9° 上限上，位置伺服会有微小过冲）。scipy 的 least_squares
        # 遇到越界初值直接抛 ValueError，而不是返回大残差，所以先夹回域内。
        q0 = np.clip(np.asarray(q_seed, float), ik.lo, ik.hi)
        e_g, q_g = ik.solve_near(grp, q0, reg=0.02)
        if e_g >= 1e-3:
            e_g, q_g = ik.solve(grp, rng, n_restart=24)
    if e_g >= 1e-3:
        return {"ok": False, "q": None, "err": e_g, "reason": "抓取点不可达"}
    if ik.bad_contacts(q_g):
        return {"ok": False, "q": None, "err": e_g, "reason": "抓取点碰撞"}

    pre_h = ik.solve_chain([grp, hov, pre], rng, n_restart=1, seed=q_g)
    up_h = ik.solve_chain([grp, up], rng, n_restart=1, seed=q_g)
    cand = [pre_h[2], pre_h[1], pre_h[0], up_h[1]]
    worst = max(c[0] for c in cand)
    if worst >= 1e-3:
        # 退回完整链的随机多起点（边缘格点偶尔需要）
        chain = ik.solve_chain([pre, hov, grp, up], rng, n_restart=32)
        cand = chain
        worst = max(c[0] for c in cand)
        if worst >= 1e-3:
            return {"ok": False, "q": None, "err": worst,
                    "reason": "整链 IK 失败 err=%.2e" % worst}

    if cube_yaw is not None:
        cand = ik.align_yaw(cand, float(cube_yaw))

    q = {"pre": cand[0][1], "hover": cand[1][1],
         "grasp": cand[2][1], "lift": cand[3][1]}
    jump = float(np.degrees(np.abs(q["hover"] - q["grasp"])).max())
    if jump > 60:
        return {"ok": False, "q": q, "err": worst,
                "reason": "跨 IK 分支 jump=%.1f°" % jump}
    return {"ok": True, "q": q, "err": worst, "reason": ""}


def fk_grasp_pos(q, scene=None):
    """正解：给 6 关节角，返回 grasp_site 的世界坐标。"""
    ik = get_ik(scene)
    ik.d.qpos[ik.qadr] = np.asarray(q, float)
    import mujoco
    mujoco.mj_kinematics(ik.m, ik.d)
    return ik.d.site_xpos[ik.site].copy()


def tool_tilt_deg(q, scene=None):
    return get_ik(scene).tool_tilt_deg(np.asarray(q, float))
