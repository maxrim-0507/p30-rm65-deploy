# -*- coding: utf-8 -*-
"""L3 安全层：限位 / 限速 / 工作空间 / 心跳看门狗。

=== 为什么安全层单独一个文件 ===
它要能在**没有硬件、没有模型**的环境里被单测。真机上这几十行是
唯一挡住「模型的荒谬输出直接驱动电机」的东西，不能埋在大循环里。

=== 四道闸，顺序不能换 ===
1. 数值有效性（NaN/inf）—— 后面任何比较遇到 NaN 都返回 False，
   会让闸门全部「放过」，必须先挡。
2. 关节限位 —— 硬约束，来自模型或厂商规格。
3. 速度限幅 —— 按 dt 折算最大跳变，超了就截断（不是拒绝）。
   截断而不是拒绝，是因为 30 Hz 上层给出 200 ms 前的目标时，
   差值本来就会很大，拒绝会让机器人卡住不动。
4. 工作空间盒 —— 末端位置越界就拒绝。这条**拒绝**不截断：
   夹爪跑到台面下或货架里，任何「就近修正」都可能撞上去。

=== 心跳门在最后 ===
心跳超时优先级最高：直接 hold，连新指令都不接受。

=== 位姿链路是另一组闸 ===
只控末端时 L3 拿不到关节目标（IK 在厂商控制器里），所以 filter_pose()
单独做「数值 + 工作空间 + 单步位移」。关节限位交给厂商控制器；
仿真里由 ik_core.solve_near 保证解在限位内。

=== 底盘是第三组闸（2026-11-09 接入 VLA）===
filter_base_vel() 只管「数值 + 速度上限 + 加速度上限」，不含位置闸：
底盘要走到哪里是任务层的事（nav 目标点在 L1），L3 只保证「不会因为
模型输出一条 3 m/s 的速度就冲出去」。三轴独立限幅，与 P30 SendVelocity
的 ctrlrange（vx ±0.6 / vy ±0.4 / wz ±0.8）同量级，配置值不应超过它。
"""
import time

import numpy as np

from ..spec import ARM_BASE, MOUNT_YAW_DEG

DEG = np.pi / 180.0


class SafetyLimiter:

    def __init__(self, joint_lo, joint_hi, max_speed_deg_s=90.0,
                 max_acc_deg_s2=360.0, workspace=None, watchdog_ms=200.0,
                 max_eef_speed_m_s=0.25, base_vel_max=None,
                 base_acc_max=None):
        self.lo = np.asarray(joint_lo, dtype=float)
        self.hi = np.asarray(joint_hi, dtype=float)
        self.max_speed = float(max_speed_deg_s) * DEG      # rad/s
        self.max_acc = float(max_acc_deg_s2) * DEG          # rad/s²
        self.max_eef_speed = float(max_eef_speed_m_s)       # m/s
        # 底盘：[vx, vy, wz] 的上限。缺省与 p30_rm65_nav.xml 的执行器
        # ctrlrange 一致；超了按比例缩到边界上，不整条拒绝。
        self.base_vel_max = np.asarray(
            base_vel_max if base_vel_max is not None else [0.6, 0.4, 0.8],
            dtype=float)
        self.base_acc_max = np.asarray(
            base_acc_max if base_acc_max is not None else [1.0, 1.0, 2.0],
            dtype=float)
        self.ws = workspace or {}
        self.watchdog_s = float(watchdog_ms) / 1000.0
        # 工作空间盒按「静态系」（名义臂座朝向）标定；位姿链路的 p 现在
        # 是**臂座系**（get_state/actuate_pose 口径）。比较前先把 p 旋转到
        # 静态系：p_static = R_base_nom · p_base。
        _yaw = MOUNT_YAW_DEG * DEG
        _c, _s = float(np.cos(_yaw)), float(np.sin(_yaw))
        self._R_base_nom = np.array([[_c, -_s, 0.0],
                                     [_s, _c, 0.0],
                                     [0.0, 0.0, 1.0]])

        self._q_prev = None          # 上一次**已下发**的目标
        self._qd_prev = np.zeros(6)
        self._t_prev = 0.0
        self._p_prev = None          # 位姿链路：上一次已下发的末端位置
        self._t_prev_p = 0.0
        self._last_hb = 0.0
        self._last_reject = ""
        # 底盘链路：上一次已下发的速度指令，用于加速度限幅
        self._v_prev = np.zeros(3)
        self._t_prev_v = 0.0
        self.stats = {"clamped_limit": 0, "clamped_speed": 0,
                      "clamped_pose_step": 0,
                      "reject_ws": 0, "reject_bad": 0, "held_hb": 0,
                      "clamped_base_speed": 0, "clamped_base_acc": 0}

    # ---------- 心跳 ----------
    def note_heartbeat(self):
        self._last_hb = time.monotonic()

    def hb_age_s(self):
        if self._last_hb <= 0:
            return float("inf")
        return time.monotonic() - self._last_hb

    def hb_ok(self):
        return self.hb_age_s() <= self.watchdog_s

    # ---------- 主闸 ----------
    def filter(self, q_target, q_measured, eef_pos=None, now=None):
        """返回 (q_safe, info)。

        q_target   期望关节角（6 × rad）
        q_measured 当前实测（6 × rad），作为第一次调用时的起点
        eef_pos    目标末端位置（**世界系**，这里会减完整 ARM_BASE 含 z）；
                   None 时跳过工作空间检查。当前 server 的关节链路传 None，
                   位姿链路走 filter_pose()，不要混用两条的 z 基准。
        """
        now = time.monotonic() if now is None else now
        q_t = np.asarray(q_target, dtype=float).reshape(6).copy()
        info = {"ok": True, "clamped": [], "reason": ""}

        # 闸 1：数值有效性
        if not np.all(np.isfinite(q_t)):
            self.stats["reject_bad"] += 1
            self._last_reject = "目标含 NaN/inf"
            return (self._q_prev if self._q_prev is not None
                    else np.asarray(q_measured, float)), {
                "ok": False, "reason": self._last_reject}

        # 闸 2：关节限位
        over = (q_t < self.lo - 1e-9) | (q_t > self.hi + 1e-9)
        if over.any():
            q_t = np.clip(q_t, self.lo, self.hi)
            self.stats["clamped_limit"] += 1
            info["clamped"].append("limit")

        # 闸 3：速度 / 加速度限幅
        if self._q_prev is None:
            self._q_prev = np.asarray(q_measured, float).copy()
            self._t_prev = now
        dt = max(now - self._t_prev, 1e-4)
        max_dq = self.max_speed * dt
        dq = q_t - self._q_prev
        if np.abs(dq).max() > max_dq:
            dq = np.clip(dq, -max_dq, max_dq)
            q_t = self._q_prev + dq
            self.stats["clamped_speed"] += 1
            info["clamped"].append("speed")
        # 加速度限幅（对速度的一阶差分做限制）
        max_dqd = self.max_acc * dt
        qd = dq / dt
        dqd = np.clip(qd - self._qd_prev, -max_dqd, max_dqd)
        qd = self._qd_prev + dqd
        q_t = self._q_prev + qd * dt
        q_t = np.clip(q_t, self.lo, self.hi)

        # 闸 4：工作空间盒
        if eef_pos is not None and self.ws:
            p = np.asarray(eef_pos, dtype=float) - np.asarray(ARM_BASE)
            for i, ax in enumerate("xyz"):
                rng = self.ws.get(ax)
                if rng and not (rng[0] <= p[i] <= rng[1]):
                    self.stats["reject_ws"] += 1
                    self._last_reject = "末端 %s=%.3f 越界 %s" % (
                        ax, p[i], rng)
                    info["ok"] = False
                    info["reason"] = self._last_reject
                    # 越界时保持上一帧目标，不执行任何「就近修正」
                    return self._q_prev.copy(), info

        self._q_prev = q_t.copy()
        self._qd_prev = qd.copy()
        self._t_prev = now
        return q_t, info

    # ---------- 位姿链路 ----------
    def filter_pose(self, pos, now=None, check_step=True):
        """笛卡尔闸：工作空间盒 + 单步最大位移。返回 (pos_safe, info)。

        === 为什么位姿链路不复用 filter() ===
        filter() 在关节空间做限位/限速，需要先有 IK 解。位姿链路里 IK
        属于后端（真机是厂商控制器），L3 拿不到关节目标，只能先挡笛卡尔
        这一层：
          1. 数值有效性；
          2. 工作空间盒（与 planner 的预检同一组范围，**xyz 均相对臂座**）；
          3. 单步位移上限 —— 模型出一块乱跳的 200 ms 动作时，截断比
             直接下发给伺服安全得多。
        关节限位交给厂商控制器（真机）/ solve_near（仿真）。这一条是
        「只控末端」这个约束的直接后果，不是遗漏。

        === check_step 只在「手动精确同步」时置 False ===
        单步位移上限是**按墙钟 dt 折算的速率整形**，用来吃掉异步循环的
        抖动。手动同步模式（LocalBackend.apply_step，每动作精确 16 物理
        步 = 32 ms）里没有抖动，而训练数据本身单步位移最大 26.6 mm
        （0.8 m/s），0.25 m/s 的闸会截掉 ~9% 的**分布内**动作 —— 那就不
        再是 rollout 的等价执行了（实测：截断让臂错过目标，抓到邻块）。
        工作空间盒与 NaN 闸仍然生效，它们不是速率整形。

        pos 的口径：**xyz 三个分量都相对臂座**（与 get_state()["eef_pos"]
        一致，见 hal/base.py），工作空间盒也是相对臂座，所以这里**不再**
        减 ARM_BASE。注意 filter() 那条关节链路入参是**世界系**（减完整
        ARM_BASE 得到相对量），与本函数的相对臂座入参不是同一口径 ——
        调用方必须自己先换算，不要混用。
        """
        now = time.monotonic() if now is None else now
        p = np.asarray(pos, dtype=float).reshape(3).copy()
        info = {"ok": True, "clamped": [], "reason": ""}

        if not np.all(np.isfinite(p)):
            self.stats["reject_bad"] += 1
            self._last_reject = "位姿含 NaN/inf"
            info.update(ok=False, reason=self._last_reject)
            return (self._p_prev.copy() if self._p_prev is not None
                    else np.zeros(3)), info

        if self.ws:
            # p 是**臂座系**，工作空间盒按**静态系**标定。先旋转到静态系再比：
            #   p_static = R_base_nom · p_base
            # 否则臂座 yaw=180° 时 x 反号，所有前伸目标都会被误判越界。
            p_ws = self._R_base_nom @ p
            for i, ax in enumerate("xyz"):
                rng = self.ws.get(ax)
                if rng and not (rng[0] <= p_ws[i] <= rng[1]):
                    self.stats["reject_ws"] += 1
                    self._last_reject = "末端 %s=%.3f 越界 %s" % (
                        ax, p_ws[i], rng)
                    info.update(ok=False, reason=self._last_reject)
                    return (self._p_prev.copy() if self._p_prev is not None
                            else p), info

        if check_step:
            if self._p_prev is None:
                self._p_prev = p.copy()
                self._t_prev_p = now
            dt = max(now - self._t_prev_p, 1e-4)
            max_d = self.max_eef_speed * dt
            d = p - self._p_prev
            dist = float(np.linalg.norm(d))
            if dist > max_d > 0:
                p = self._p_prev + d * (max_d / dist)
                self.stats["clamped_pose_step"] += 1
                info["clamped"].append("pose_step")
        else:
            self.stats["sync_no_step_clamp"] = (
                self.stats.get("sync_no_step_clamp", 0) + 1)

        self._p_prev = p.copy()
        self._t_prev_p = now
        return p, info

    # ---------- 底盘链路 ----------
    def filter_base_vel(self, vel, now=None):
        """底盘速度闸：数值 + 速度上限 + 加速度上限。返回 (v_safe, info)。

        === 为什么底盘的闸和关节/位姿都不同 ===
        底盘是**速度**指令（持续有效，每帧重发），不是位置目标：
          - 没有「工作空间盒」这个东西 —— 底盘走到哪由任务层决定；
          - 但速度必须限，模型输出一条 3 m/s 会直接冲出去；
          - 加速度也要限，否则速度从 0 跳到上限会让狗甩臂、甚至翻车。

        三轴**独立**限幅（不是按范数等比缩）：vx/vy/wz 单位不同，
        按欧氏范数混算没有物理意义。仿真执行器本身也是三轴独立夹紧。

        vel 的口径：**机体系** [vx, vy, wz]，m/s、m/s、rad/s，与
        hal/base.py 的 move_base 和 vla_spec.yaml 的 base_velocity 一致。
        """
        now = time.monotonic() if now is None else now
        v = np.asarray(vel, dtype=float).reshape(3).copy()
        info = {"ok": True, "clamped": [], "reason": ""}

        if not np.all(np.isfinite(v)):
            self.stats["reject_bad"] += 1
            self._last_reject = "底盘速度含 NaN/inf"
            info.update(ok=False, reason=self._last_reject)
            return self._v_prev.copy(), info

        # 1) 速度上限：三轴独立夹紧
        over = np.abs(v) > self.base_vel_max
        if np.any(over):
            v = np.clip(v, -self.base_vel_max, self.base_vel_max)
            self.stats["clamped_base_speed"] += 1
            info["clamped"].append("base_speed")

        # 2) 加速度上限：按 dt 折算每轴最大允许变化
        if self._t_prev_v <= 0:
            self._t_prev_v = now
            self._v_prev = v.copy()
            info["clamped"].append("base_first")
            return v, info
        dt = max(now - self._t_prev_v, 1e-4)
        lim = self.base_acc_max * dt
        dv = v - self._v_prev
        sat = np.abs(dv) > lim
        if np.any(sat):
            v = self._v_prev + np.clip(dv, -lim, lim)
            self.stats["clamped_base_acc"] += 1
            info["clamped"].append("base_acc")

        self._v_prev = v.copy()
        self._t_prev_v = now
        return v, info

    def reset_base_vel(self):
        """底盘链路归零（进入静态阶段/重规划时调用），避免跨段累积加速度约束。"""
        self._v_prev = np.zeros(3)
        self._t_prev_v = 0.0

    # ---------- 心跳闸（外部显式调用）----------
    def watchdog_hold(self, q_measured):
        if self.hb_ok():
            return None
        self.stats["held_hb"] += 1
        self._q_prev = np.asarray(q_measured, float).copy()
        self._qd_prev = np.zeros(6)
        self._t_prev = time.monotonic()
        return self._q_prev.copy()
