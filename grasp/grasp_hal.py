# -*- coding: utf-8 -*-
"""抓取闭环 HAL：策略 / 闭环 / 可插拔后端 三层。

 策略(不换)  vla.LocalGr00tPolicy —— 同一份 VLA（仿真/真机共用）
 闭环(不换)  run_grasp(_loop) —— 读观测→VLA→执行16步，同步逐帧（勿用异步服务）
 后端(可换)  真机 = RealGraspBackend（实现下方6接口）
             仿真 = simulation 的 ColorGraspEnv（作为回归，见 examples/）

后端接口契约（real backend 必须实现，语义同 ColorGraspEnv 对应方法）：
  reset(layout, q_init)       机体摆到抓取起点
  set_base_pose(x, y, yaw)    设底盘位姿 -> {"ok":bool}
  observe()                   -> {"front","arm","state"(16维),"eef_pos","base_xy","base_yaw"}
  step(a13)                   执行13维动作(eef9+grip+base_vel)，推进16控制步
                              -> {"ik_err","tilt","rejected","base_cmd"}
  judge(color)                -> {"success","lift","picked"}
  solve_pregrasp(layout,color)-> (q0,pre,hov,grp,p) 或 None（预备抓取位，真机可查表/真机IK）

注意：抓取闭环必须同步（读当前相机→推理→发16步→再读），不要把物理推进交给后台异步线程
+读快照（测过会系统偏、抓空）。
"""

ROOT = __import__("os").path.dirname(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import os
import sys

sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from vla.local_gr00t_policy import LocalGr00tPolicy  # noqa: E402

CKPT_DEFAULT = os.path.join(ROOT, "checkpoints", "groot_rel_v36", "020000")
FWD2 = np.radians([0.0, -13.6, 134.9, 0.0, -16.2, -90.0])


class GraspBackend:
    """后端接口契约（真机/仿真均需满足）。"""
    def reset(self, layout, q_init):
        raise NotImplementedError

    def set_base_pose(self, x, y, yaw):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    def step(self, a13):
        raise NotImplementedError

    def judge(self, color):
        raise NotImplementedError

    def solve_pregrasp(self, layout, color):
        raise NotImplementedError


class RealGraspBackend(GraspBackend):
    """真机后端骨架 —— 上真机时按接口实现，逻辑照 ColorGraspEnv 对应方法。

    实现指引见 docs/real_robot_deploy.md。
    """
    def __init__(self, camera_calib=None):
        raise NotImplementedError("真机后端待实现：相机流 + RM65/P30 SDK + FK/标定")

    def reset(self, layout, q_init):
        raise NotImplementedError

    def set_base_pose(self, x, y, yaw):
        raise NotImplementedError

    def observe(self):
        # 实现要点：
        #  front/arm = 真机相机读图，尺寸/内参/颜色序与训练一致(320x240)
        #  eef9 = 真机 FK 算 grasp_site 在臂座系位姿(pos3 + rot6d6)
        #  state = np.concatenate([eef9, [grip_norm], joint6])  # grip_norm=(0.0245-q)/(0.0245+0.020)
        raise NotImplementedError

    def step(self, a13):
        # 实现要点：a13[:3]末端xyz, a13[3:9]rot6d, a13[9]夹爪[0,1], a13[10:13]底盘速度
        #  -> 真机控制器(rm_movej_p/movel + 底盘速度)，等 ~16*32ms
        raise NotImplementedError

    def judge(self, color):
        raise NotImplementedError

    def solve_pregrasp(self, layout, color):
        raise NotImplementedError


def make_sim_backend():
    """仿真后端（回归用，需 simulation/mujoco 资产）。"""
    try:
        from rollout_color_grasp import ColorGraspEnv
    except Exception as e:  # noqa
        raise RuntimeError("仿真后端不可用（需 simulation/mujoco + mesh 资产）：%r" % e)
    from rollout_color_grasp import HERE
    import os as _os
    return ColorGraspEnv(view=False, scene=_os.path.join(HERE, "p30_rm65_nav.xml"))


def run_grasp_loop(backend, policy, layout, color, instruction,
                   base=None, q_init=FWD2, max_steps=600, receding=40):
    """同步逐帧闭环（后端无关）。仿真/真机同一份逻辑。

    receding: 每块执行步数（40=整块，对齐已验证基线；过小反而抖）。
    """
    info = backend.solve_pregrasp(layout, color)
    if info is None:
        return {"success": False, "reason": "pregrasp IK 不可达"}
    q0, pre, hov, grp, p = info
    backend.reset(layout, q0 if q_init is None else q_init)
    if base is not None:
        r = backend.set_base_pose(*base)
        if not r.get("ok"):
            return {"success": False, "reason": r.get("err")}

    chunk = []
    done = 0
    while done < max_steps:
        if not chunk:
            obs = backend.observe()
            instruction_ = instruction
            chunk = policy.infer(obs, instruction_, frame_t=done)
            if not chunk:
                return {"success": False, "reason": "empty chunk"}
        for a in chunk[:max(1, receding)]:
            a = np.asarray(a, float)
            backend.step(a)
            done += 1
        chunk = []
    return backend.judge(color)
