# -*- coding: utf-8 -*-
"""Pregrasp 技能：从当前位置走到「抓取点上方」，夹爪张开，目标在视野内。

=== 为什么它是独立技能而不是 Manipulate 的第一步 ===
它是**经典闭环**（IK + 标定），不依赖任何学习模型。分开之后：
  - VLA 还没训出来时，Pregrasp + 脚本合爪就能跑通全链路；
  - VLA 训好之后，替换的只是 Manipulate，Pregrasp 一行不改；
  - 失败可归因（IK 无解 vs 策略乱动），这是调 VLA 时最关键的信息。

=== 入口/出口约定（与 docs/cargobot/04 §3.3 一致）===
入口：任意姿态
出口：位于 grasp 点正上方 hover 高度，夹爪张开，工具轴朝下
"""
import time

import numpy as np

from . import kinematics
from .base import Skill, SkillResult


class PregraspSkill(Skill):
    name = "pregrasp"

    def __init__(self, backend, pre_height=0.20, hover_height=0.10,
                 **kw):
        super().__init__(backend, **kw)
        self.pre_height = float(pre_height)
        self.hover_height = float(hover_height)

    def run(self, ctx):
        """ctx 需要：
            target_xy  (x, y) 世界系（方块中心在地面的投影）
            target_yaw 方块 yaw（rad），用于 joint6 对齐；可为 None
            base_pose  (x, y, yaw) 底盘当前位姿；给了就换算到臂基座系
            task_offset_z  方块中心相对台面的高度（默认 cube_half）
        """
        t0 = time.monotonic()
        xy = ctx.get("target_xy")
        if xy is None:
            return SkillResult(False, "FAULT", "缺少 target_xy")

        cube_yaw = ctx.get("target_yaw")
        bp = ctx.get("base_pose")
        if bp is not None:
            # 底盘不在标称位：世界目标 → 臂基座系，否则 IK 会算成
            # 「底盘没动」时的解，执行端会整体偏出停车误差那么多。
            xy = kinematics.world_target_to_ik_xy(xy, bp[:2], bp[2])
            if cube_yaw is not None:
                cube_yaw = float(cube_yaw) - float(bp[2])

        st = self.backend.get_state()
        q_now = st["joint"]

        sol = kinematics.solve_approach(xy, q_seed=q_now, cube_yaw=cube_yaw)
        if not sol["ok"]:
            return SkillResult(False, "FAULT", "IK: " + sol["reason"],
                               {"err": sol["err"]})

        q = sol["q"]
        grip_open = 1.0
        deadline = self._deadline()

        # 直接去抓取点上方：先到 pre，再到 hover。
        # 中间不经过「当前点 → pre」的直线笛卡尔，避免扫过台面边缘。
        self.move_joints(q_now, q["pre"], st["gripper"], grip_open,
                         deadline=deadline)
        self.move_joints(q["pre"], q["hover"], grip_open, grip_open,
                         deadline=deadline)
        # 到位稳定：经 L3 控服时指令要过 100 Hz 插值与限速层，最后一帧
        # 推送完的瞬间实测角还落后几度，直接判工具轴会误报 10°+。
        self.hold_for(0.3, grip=grip_open, q=q["hover"])

        # 校验：末端朝下、悬停高度合理。这条要在推送之后重读状态，
        # 不能用 IK 的理论值 —— 真机有重力静差。
        st2 = self.backend.get_state()
        tilt = kinematics.tool_tilt_deg(st2["joint"])
        if tilt > 8.0:
            return SkillResult(False, "FAULT",
                               "工具轴偏 %.1f°（>8°）" % tilt)

        return SkillResult(True, "PREGRASP_DONE", "到达预抓取位",
                           {"q": q, "tilt_deg": tilt,
                            "elapsed_s": time.monotonic() - t0})
