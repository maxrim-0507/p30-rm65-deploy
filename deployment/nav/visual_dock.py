# -*- coding: utf-8 -*-
"""VISUAL DOCK：从「目标在哪」算泊车位，再做机体系速度闭环。

=== 泊车位为什么这么算 ===
v36 的训练分布是「车停在桌前、臂座 x≈0.06、目标在 0.43 m 处」。所以
泊车位不是随便一个朝向距离，而是**目标点沿视线后退 standoff，再裁进
训练盒**（x∈[−0.34,−0.12]，|y|≤0.12，|yaw|≤0.15 rad）。盒子来自采集
数据的起点分布；出了盒子 VLA 仍能吸收一部分（实测 ±0.40 m），但没
必要拿成功率去赌。

=== 目标锁存，不逐帧重算 ===
如果在每个 tick 用「当前车→目标」的方向重算泊车位，车一动泊车位跟着
动，闭环会追自己的影子。这里在第一次看到目标时锁存泊车位；只有丢失
后重新捕获才重算。

=== 丢失就停，不凭记忆开 ===
观测丢失后沿用最后估计最多 stale_s（缺省 0.5 s），超时零速并报 LOST，
由上层重新搜索。允许「凭记忆继续开」是这类系统最常见的事故源。

=== 唯一的例外：已锁存车位、且只差最后一段 ===
目标在 SEARCH_Q 下 0.28 m（相机到方块）处会沉出画面。斜向进场时
「离车位还差 0.3 m」就已经盲了，0.5 s 的停等只够走 0.08 m，于是停在
降级交班盒外（2026-11-21 真 HSV seed 5 实测：残差 0.305 m / yaw 0.47，
超出 0.25 / 0.30）。所以：**车位已锁存、当前离车位 ≤0.45 m** 时，允许
再盲走 blind_goal_s（缺省 2 s），期间按车位朝向收敛；超时仍报 LOST，
由上层用降级盒判定。0.45 m 以外一律不盲开。
"""
import time

import numpy as np

# 训练分布盒 / 控制参数（与昨天演示脚本同一套数值）
DEFAULTS = {
    "standoff": 0.70,
    "stage_x": (-0.34, -0.12),
    "stage_y_abs": 0.12,
    "stage_yaw_abs": 0.15,
    "kp_pos": 1.2,
    "kp_yaw": 2.0,
    "lim_vx": 0.6,
    "lim_vy": 0.4,
    "lim_wz": 0.8,
    "tol_xy": 0.01,
    "tol_yaw": 0.01,
    "stale_s": 0.5,
    # 已锁存车位后的盲走预算（见模块文档的例外说明）
    "blind_goal_s": 2.0,
    "blind_goal_max_err": 0.45,
    "max_frames": 400,
}


def _wrap_pi(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def dock_pose(target_xy, base_xy, standoff=0.70, stage_x=(-0.34, -0.12),
              stage_y_abs=0.12, stage_yaw_abs=0.15):
    """目标世界 xy + 当前车位姿 → 泊车位 (x, y, yaw)，世界系。"""
    t = np.asarray(target_xy, float).reshape(2)
    b = np.asarray(base_xy, float).reshape(2)
    d = t - b
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        d, n = np.array([1.0, 0.0]), 1.0
    dock = t - d / n * float(standoff)
    dx = float(np.clip(dock[0], stage_x[0], stage_x[1]))
    dy = float(np.clip(dock[1], -stage_y_abs, stage_y_abs))
    dyaw = float(np.clip(np.arctan2(t[1] - dy, t[0] - dx),
                         -stage_yaw_abs, stage_yaw_abs))
    return np.array([dx, dy, dyaw])


class DockPhase(object):
    TRACK = "track"
    LOST = "lost"
    ARRIVED = "arrived"
    FAILED = "failed"

    TERMINAL = (ARRIVED, FAILED)


class DockStatus(object):
    __slots__ = ("phase", "vx", "vy", "wz", "err", "dyaw", "goal",
                 "stale_s", "frames", "reason")

    def __init__(self, phase, vx=0.0, vy=0.0, wz=0.0, err=0.0, dyaw=0.0,
                 goal=None, stale_s=0.0, frames=0, reason=""):
        self.phase = phase
        self.vx, self.vy, self.wz = float(vx), float(vy), float(wz)
        self.err = float(err)
        self.dyaw = float(dyaw)
        self.goal = goal
        self.stale_s = float(stale_s)
        self.frames = int(frames)
        self.reason = reason

    @property
    def done(self):
        return self.phase in DockPhase.TERMINAL


class VisualDock(object):

    def __init__(self, locate, get_base_pose, cfg=None, clock=None):
        """locate() -> {"ok":bool,"xy":(x,y)} | None
        get_base_pose() -> (x, y, yaw)，世界系
        """
        self.locate = locate
        self.get_base_pose = get_base_pose
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update(cfg)
        self._clock = clock or time.monotonic
        self.reset()

    def reset(self):
        self.goal = None
        self.phase = DockPhase.TRACK
        self.frames = 0
        self.reason = ""
        self._last_xy = None
        self._last_t = None

    # ---------- 主入口 ----------
    def tick(self, obs=None):
        if self.phase in DockPhase.TERMINAL:
            return DockStatus(self.phase, goal=self.goal, frames=self.frames,
                              reason=self.reason)

        now = self._clock()
        if obs is None:
            obs = self.locate()
        if obs is not None and obs.get("ok"):
            self._last_xy = np.asarray(obs["xy"], float).reshape(2)
            self._last_t = now
            if self.goal is None:
                self.goal = dock_pose(
                    self._last_xy, self.get_base_pose()[:2],
                    standoff=self.cfg["standoff"], stage_x=self.cfg["stage_x"],
                    stage_y_abs=self.cfg["stage_y_abs"],
                    stage_yaw_abs=self.cfg["stage_yaw_abs"])

        stale = float("inf") if self._last_t is None else max(0.0, now - self._last_t)
        self.frames += 1

        c = self.cfg
        bx, by, byaw = self.get_base_pose()
        # 盲走例外：车位已锁存且离车位足够近，才允许超出 stale_s 继续
        blind_ok = False
        if self.goal is not None:
            blind_ok = (float(np.hypot(self.goal[0] - bx,
                                       self.goal[1] - by))
                        <= float(c["blind_goal_max_err"]))
        if (self.goal is None
                or (stale > c["stale_s"]
                    and not (blind_ok
                             and stale <= c["stale_s"] + c["blind_goal_s"]))):
            self.phase = DockPhase.LOST
            self.reason = ("尚未看到目标" if self.goal is None
                           else "目标丢失 %.2f s" % stale)
            return DockStatus(self.phase, stale_s=stale, frames=self.frames,
                              goal=self.goal, reason=self.reason)
        ex, ey = self.goal[0] - bx, self.goal[1] - by
        # === 远距离先对准「实测目标」，不是泊车位朝向 ===
        # 泊车位朝向是「车位→目标」的角度；车还在侧面时它和「车→目标」
        # 能差几十度。一开始就转到车位朝向，目标会被甩出 SEARCH_Q 的
        # 窄视场（实测：起点偏 45°、走到离车位 0.68 m 处丢失）。车到
        # 位后两者近似相等，所以最后 0.10 m 再收敛到车位朝向。
        dist_goal = float(np.hypot(ex, ey))
        near_end = dist_goal <= 0.30
        yaw_ref = self.goal[2]
        if stale > c["stale_s"]:
            yaw_ref = self.goal[2]          # 盲走段只认锁存车位朝向
        elif self._last_xy is not None:
            bear = float(np.arctan2(self._last_xy[1] - by,
                                    self._last_xy[0] - bx))
            # === 参考角必须平滑过渡，不能硬切换 ===
            # 硬切换（距车位 0.30 m 时从「目标方位」直接换成「车位朝向」）
            # 在横向偏移时会产生 ~0.5 rad 跳变：车被迫原地转大角度，
            # 目标转出 SEARCH_Q 视场，随后盲开 0.5 s 只走 0.15 m，
            # 停在降级盒外（2026-11-21 真 HSV seed 5 实测：残差 0.295 m）。
            # 按 dist_goal 线性过渡：≥0.30 全用目标方位，≤0.10 全用车位
            # 朝向，中间按最短弧插值。
            w = float(np.clip((dist_goal - 0.10) / 0.20, 0.0, 1.0))
            yaw_ref = bear + (1.0 - w) * _wrap_pi(self.goal[2] - bear)
        eyaw = _wrap_pi(yaw_ref - byaw)

        eyaw_goal = _wrap_pi(self.goal[2] - byaw)
        if np.hypot(ex, ey) < c["tol_xy"] and abs(eyaw_goal) < c["tol_yaw"]:
            self.phase = DockPhase.ARRIVED
            return DockStatus(self.phase, goal=self.goal, stale_s=stale,
                              frames=self.frames, reason="已到泊车位")
        if self.frames > c["max_frames"]:
            self.phase = DockPhase.FAILED
            self.reason = "泊车超时（%d 帧）" % self.frames
            return DockStatus(self.phase, goal=self.goal, stale_s=stale,
                              frames=self.frames, reason=self.reason)

        # 世界误差 → 机体系
        ca, sa = np.cos(byaw), np.sin(byaw)
        ebx = ca * ex + sa * ey
        eby = -sa * ex + ca * ey
        vx = float(np.clip(c["kp_pos"] * ebx, -c["lim_vx"], c["lim_vx"]))
        vy = float(np.clip(c["kp_pos"] * eby, -c["lim_vy"], c["lim_vy"]))
        wz = float(np.clip(c["kp_yaw"] * eyaw, -c["lim_wz"], c["lim_wz"]))
        # === 先转后走 ===
        # 朝向误差大时同时平移会绕着目标画圈；末段（≤0.30 m）更要在进入
        # SEARCH_Q 近场盲区之前把 yaw 收敛（实测 seed 5：盲区前还差 0.43
        # rad，超出降级盒 0.30 → 判失败）。所以末段有新鲜观测时阈值
        # 收紧到 0.06 rad。
        # 观测已过期（快盲/已盲）时必须继续平移：此时没有相机可保，
        # 唯一要紧的是趁降级盒还接受残差把车开进去。
        gate = 0.06 if (near_end and stale <= 0.10) else 0.35
        if abs(eyaw) > gate:
            vx = vy = 0.0

        self.phase = DockPhase.TRACK
        return DockStatus(self.phase, vx=vx, vy=vy, wz=wz,
                          err=float(np.hypot(ex, ey)), dyaw=eyaw,
                          goal=self.goal, stale_s=stale, frames=self.frames)
