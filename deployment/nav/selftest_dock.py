# -*- coding: utf-8 -*-
"""VISUAL DOCK 自检：纯数学 + 假底盘，不需要 MuJoCo。

用法：python -m deployment.nav.selftest_dock
"""
import numpy as np

from .visual_dock import DockPhase, VisualDock, dock_pose


class FakeBase(object):
    """点机器人：机体系速度积分，效率按底盘实测 ~76%。"""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, dt=0.08, eff=0.76):
        self.x, self.y, self.yaw = x, y, yaw
        self.dt, self.eff = dt, eff

    def pose(self):
        return (self.x, self.y, self.yaw)

    def apply(self, st):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.x += (c * st.vx - s * st.vy) * self.dt * self.eff
        self.y += (s * st.vx + c * st.vy) * self.dt * self.eff
        self.yaw += st.wz * self.dt * self.eff


def obs_ok(xy):
    return {"ok": True, "xy": xy}


def t_dock_math():
    g = dock_pose((0.48, 0.16), (0.0, 0.0))
    assert abs(g[0] - (-0.19)) < 0.02, g
    assert abs(g[1] - (-0.064)) < 0.02, g
    assert abs(g[2] - 0.15) < 1e-9, g          # 偏航被裁到 +0.15
    return "泊车位 = 目标后退 0.70 m 并裁进训练盒：%s" % np.round(g, 3)


def t_clip():
    g = dock_pose((1.6, 0.9), (0.1, 0.0))
    assert -0.34 <= g[0] <= -0.12 and abs(g[1]) <= 0.12, g
    assert abs(g[2]) <= 0.15, g
    return "远处/侧向目标 → 泊车位仍在盒内 %s" % np.round(g, 3)


def t_converge():
    base = FakeBase()
    tgt = (0.48, 0.16)
    d = VisualDock(lambda: obs_ok(tgt), base.pose, cfg={"max_frames": 600})
    for i in range(600):
        st = d.tick()
        if st.done:
            break
        base.apply(st)
    assert d.phase == DockPhase.ARRIVED, (d.phase, d.frames, base.pose())
    err = np.hypot(base.x - d.goal[0], base.y - d.goal[1])
    assert err < 0.02, err
    return "从原点收敛：%d 帧到泊车位，残差 %.1f mm" % (d.frames, err * 1e3)


def t_goal_locked():
    base = FakeBase()
    d = VisualDock(lambda: obs_ok((0.48, 0.16)), base.pose)
    st = d.tick()
    g0 = d.goal.copy()
    base.apply(st)
    d.tick()
    assert np.allclose(d.goal, g0), (g0, d.goal)
    return "泊车位在首次捕获时锁存，不随车动"


def t_lost():
    base = FakeBase()
    d = VisualDock(lambda: None, base.pose)
    st = d.tick()
    assert st.phase == DockPhase.LOST and st.vx == st.vy == st.wz == 0.0
    return "从未看到目标 → LOST 且速度为零（不盲开）"


def t_stale_stop():
    base = FakeBase()
    clock = [0.0]
    tgt = (0.48, 0.10)
    seq = [obs_ok(tgt), obs_ok(tgt), None, None, None, None, None, None]
    # 关掉盲走收尾：本用例只验「stale_s 到点就停」，盲走另有 t_blind_finish
    d = VisualDock(lambda: seq.pop(0) if seq else None, base.pose,
                   cfg={"blind_goal_s": 0.0}, clock=lambda: clock[0])
    phases = []
    for i in range(5):
        clock[0] = i * 0.2
        st = d.tick()
        phases.append(st.phase)
        if st.phase != DockPhase.LOST:
            base.apply(st)
    assert phases[0] == DockPhase.TRACK and phases[1] == DockPhase.TRACK, phases
    assert DockPhase.LOST in phases[2:], phases
    return "丢失超过 stale_s(0.5s) 后转为 LOST：%s" % "/".join(phases)


def t_blind_finish():
    base = FakeBase()
    clock = [0.0]
    tgt = (0.48, 0.10)
    seq = [obs_ok(tgt), obs_ok(tgt)]
    # 关掉伺服（kp=0）避免提前 ARRIVED：本用例只验时间门限
    d = VisualDock(lambda: seq.pop(0) if seq else None, base.pose,
                   cfg={"kp_pos": 0.0, "kp_yaw": 0.0}, clock=lambda: clock[0])
    phases = []
    for i in range(20):
        clock[0] = i * 0.2
        phases.append(d.tick().phase)
    # 车位已锁存且离车位 0.19 m < blind_goal_max_err → 超过 stale_s 仍 TRACK
    assert phases[5] == DockPhase.TRACK, phases
    # 盲走预算 blind_goal_s=2.0 用完后必须 LOST，不能无限开
    assert phases[-1] == DockPhase.LOST, phases
    return "盲走收尾：stale 后仍跟踪，超过 blind_goal_s 转 LOST：%s" \
        % "/".join(phases)


def t_timeout():
    base = FakeBase()
    d = VisualDock(lambda: obs_ok((0.48, 0.16)), base.pose,
                   cfg={"max_frames": 3, "kp_pos": 0.0, "kp_yaw": 0.0})
    st = None
    for _ in range(4):
        st = d.tick()
    assert st.phase == DockPhase.FAILED, st.phase
    return "走不动又到不了 → 超时 FAILED（%d 帧）" % st.frames


def main():
    cases = [t_dock_math(), t_clip(), t_converge(), t_goal_locked(),
             t_lost(), t_stale_stop(), t_blind_finish(), t_timeout()]
    for i, msg in enumerate(cases, 1):
        print("  %d. %s" % (i, msg))
    print("dock selftest: %d/%d OK" % (len(cases), len(cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
