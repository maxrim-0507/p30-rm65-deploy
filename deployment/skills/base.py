# -*- coding: utf-8 -*-
"""技能基类 + 关节轨迹执行器。

=== 技能是什么 ===
一个技能 = 「一串关节目标 + 判定成功的条件」，在固定频率下把目标
推给后端，直到完成或超时。技能里**不碰硬件**，只调 HAL 的
actuate/get_state，所以同一套技能在仿真和真机上跑的是同一份代码。

=== 三段时间参数 ===
  rate_hz      技能自身循环频率（30 Hz）
  seg_speed    关节空间插值的角速度（度/秒），按最长那一关节算段时长
  timeout_s    整个技能的兜底超时

=== 为什么用关节空间插值而不是笛卡尔 ===
IK 已经保证整条链在同一分支上（kinsematics.solve_approach）。关节
线性插值一定落在可达域内，且不会在腕部奇异点附近产生巨大角速度。
笛卡尔直线插值需要逐步 IK，误差会累积，而且我们本来就是 5-DOF
（yaw 自由），笛卡尔姿态没有唯一解。
"""
import time

import numpy as np


class SkillResult:
    def __init__(self, ok, state, message="", data=None):
        self.ok = bool(ok)
        self.state = state
        self.message = message
        self.data = data or {}

    def __repr__(self):
        return "SkillResult(ok=%s, state=%s, %s)" % (
            self.ok, self.state, self.message)


class Skill:
    name = "skill"

    def __init__(self, backend, rate_hz=30.0, speed_deg_s=45.0, timeout_s=20.0):
        self.backend = backend
        self.rate_hz = float(rate_hz)
        self.speed_deg_s = float(speed_deg_s)
        self.timeout_s = float(timeout_s)

    # ---------- 轨迹执行 ----------
    def _advance(self):
        """推进一个控制周期的物理时间。

        不能直接 step() —— 直连 MuJoCo 时一个物理步只有 2 ms，30 Hz 的
        控制周期是 33 ms，差 17 倍。换算由后端负责（steps_per_control）。
        """
        spc = getattr(self.backend, "steps_per_control", None)
        self.backend.step(spc(self.rate_hz) if spc is not None else 1)

    def move_joints(self, q_from, q_to, grip_from, grip_to, deadline=None):
        """关节空间线性插值并推送。返回实际到达的 (q, grip)。"""
        qa = np.asarray(q_from, float).reshape(6)
        qb = np.asarray(q_to, float).reshape(6)
        dist_deg = float(np.degrees(np.abs(qb - qa)).max())
        dur = max(0.15, dist_deg / max(self.speed_deg_s, 1e-3))
        n = max(2, int(round(dur * self.rate_hz)))
        dt = 1.0 / self.rate_hz

        q = qa.copy()
        g = float(grip_from)
        for k in range(1, n + 1):
            t = k / float(n)
            # 余弦缓动：起停平滑，避免位置伺服在段首段尾冲击
            s = 0.5 - 0.5 * np.cos(np.pi * t)
            q = qa + (qb - qa) * s
            g = float(grip_from) + (float(grip_to) - float(grip_from)) * s
            self.backend.actuate(q, g)
            self._advance()
            time.sleep(dt)
            if deadline is not None and time.monotonic() > deadline:
                return q, g
        return q, g

    def hold_for(self, seconds, grip=None, q=None):
        """保持若干秒（等物理稳定 / 等夹爪闭合并建立接触）。

        === q 必须能指定，不能总用实测值 ===
        经 L3 控制服务时，指令要过安全层的速度/加速度限幅，实际关节角
        会滞后目标几度。若合爪阶段用 get_state() 的实测角去 hold，等于
        把「还差几度」固化下来 —— 夹爪在离方块 1~2 cm 的高处闭合，
        抬升时滑脱（实测抬升 +0.011 m）。所以抓取接触阶段要传入目标角，
        让伺服在这 0.8 s 里继续收敛。
        """
        if q is None:
            q = self.backend.get_state()["joint"]
        else:
            q = np.asarray(q, float).reshape(6)
        st = self.backend.get_state()
        g = st["gripper"] if grip is None else float(grip)
        n = max(1, int(round(seconds * self.rate_hz)))
        dt = 1.0 / self.rate_hz
        for _ in range(n):
            self.backend.actuate(q, g)
            self._advance()
            time.sleep(dt)

    # ---------- 子类实现 ----------
    def run(self, ctx):
        raise NotImplementedError

    # ---------- 工具 ----------
    def _deadline(self):
        return time.monotonic() + self.timeout_s
