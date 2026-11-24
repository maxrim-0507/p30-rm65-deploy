# -*- coding: utf-8 -*-
"""导航↔VLA 协同层自检：五项，全部通过才谈得上「可以接导航」。

  1. mux 单写者   —— EXTERNAL 时一根手指不碰；NONE 时只写 0
  2. mux 混合     —— 切所有权不产生速度阶跃；意图过期平滑降为零
  3. 代理          —— 技能层的 move_base 变成「登记 VLA 意图」
  4. handoff 正路  —— 泊位 → 取消导航 → 停稳 → FWD2 → 可见 → VLA
  5. handoff 拒绝  —— 可见性不足 / 车位漂出必须快速失败且不抢底盘

用法（不需要 MuJoCo、不需要网络、约 1 秒跑完）：
    python3 -m deployment.nav.selftest_nav

=== 为什么用假时钟 ===
混合曲线（smoothstep 0.4 s）和静止窗口（0.6 s）都是时间判据。用真
时钟只能得到「大概对了」，而且每次跑的结果不一样。假时钟逐拍推进，
可以把每一拍的速度值精确断言，还能验证「同一次切换在不同拍点上
都不出现阶跃」。
"""
import sys

import numpy as np

from .handoff import HandoffSupervisor, Phase, wrap_fwd2
from .mux import BaseVelocityMux, Owner
from .proxy import MuxBackend


class FakeClock(object):
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeBackend(object):
    """只记录底盘指令。没有物理、没有线程、没有网络。"""

    supports_base = True
    supports_pose_command = True

    def __init__(self):
        self.calls = []
        self.closed = False

    def move_base(self, vx=0.0, vy=0.0, wz=0.0):
        self.calls.append((float(vx), float(vy), float(wz)))
        return {"ok": True, "err": None}

    def stop_base(self):
        self.calls.append((0.0, 0.0, 0.0))
        return {"ok": True, "err": None}

    def get_base_state(self):
        return {"pos": None, "yaw": None, "vel": (0.0, 0.0, 0.0)}

    def get_state(self):
        return {"joint": np.zeros(6), "gripper": 1.0,
                "eef_pos": np.zeros(3), "eef_rot": np.eye(3), "t": 0.0}

    def close(self):
        self.closed = True


# ---------------- 1 + 2. mux ----------------
def test_mux():
    clock = FakeClock()
    be = FakeBackend()
    mux = BaseVelocityMux(be, rate_hz=50, blend_s=0.4,
                          intent_timeout_s=0.3, clock=clock, verbose=False)

    # EXTERNAL：完全静默
    for _ in range(5):
        assert mux._tick(clock.t) is None
        clock.advance(0.02)
    assert be.calls == [], "EXTERNAL 时 mux 不应写底盘"

    # NONE：只写 0
    mux.set_owner(Owner.NONE)
    mux._tick(clock.t)
    assert be.calls and be.calls[-1] == (0.0, 0.0, 0.0)

    # VLA：从 0 平滑升到 0.4，不出现阶跃
    mux.set_vla(0.4, 0.0, 0.0)
    mux.set_owner(Owner.VLA, blend_s=0.4)
    t0 = clock.t
    xs = []
    for _ in range(9):                 # t = 0 … 0.40，逐拍刷新意图
        mux.set_vla(0.4, 0.0, 0.0)
        v = mux._tick(clock.t)
        xs.append(float(v[0]))
        clock.advance(0.05)
    steps = [abs(b - a) for a, b in zip(xs, xs[1:])]
    assert xs[0] == 0.0, xs
    assert abs(xs[-1] - 0.4) < 1e-9, xs
    assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:])), "混合不单调"
    assert max(steps) < 0.12, "单拍最大跳变 %.3f m/s 过大" % max(steps)
    assert abs((clock.t - t0) - 0.45) < 1e-9

    # 意图过期：不刷新 0.31 s 后开始降速，且不是一步归零
    import numpy as _np
    while clock.t - (t0 + 0.4) < 0.31:
        mux._tick(clock.t)
        clock.advance(0.02)
    v_at_stale = float(mux._tick(clock.t)[0])
    assert abs(v_at_stale - 0.4) < 1e-9, "过期那一拍应先保持上一值"
    clock.advance(0.02)
    v_next = float(mux._tick(clock.t)[0])
    assert v_next < 0.4 and v_next > 0.0, "过期应走混合，实测 %.3f" % v_next
    for _ in range(30):
        mux._tick(clock.t)
        clock.advance(0.02)
    assert abs(float(mux._tick(clock.t)[0])) < 1e-9, "过期后应收敛到 0"

    # 恢复刷新：从 0 再升上去
    mux.set_vla(0.2, 0.0, 0.0)
    clock.advance(0.05)
    mux._tick(clock.t)
    clock.advance(0.02)
    assert float(mux._tick(clock.t)[0]) > 0.0

    print("  1. mux 单写者 OK  EXTERNAL 零写入 / NONE 只写 0")
    print("  2. mux 混合   OK  0→0.4 单调、单拍 < 0.12 m/s、过期平滑归零")
    return mux, be, clock


# ---------------- 3. 代理 ----------------
def test_proxy(mux, be):
    n_before = len(be.calls)
    proxy = MuxBackend(be, mux)
    assert proxy.supports_base is True
    proxy.move_base(0.25, -0.1, 0.3)
    assert len(be.calls) == n_before, "代理不应绕过 mux 直写底盘"
    v, _t = mux._intent[Owner.VLA]
    assert np.allclose(v, [0.25, -0.1, 0.3]), v
    assert proxy.name.startswith("Mux(")
    assert proxy.get_state()["gripper"] == 1.0, "非底盘方法必须透传"
    print("  3. 代理       OK  move_base → 意图，其余透传")
    return proxy


# ---------------- 4 + 5. handoff ----------------
def _make_sup(mux, clock, vis_px=120.0, dock=(1.0, 0.0, 0.0), pose=None):
    state = {"pose": pose or (0.4, 0.0, 0.0), "cancel": 0, "preset": 0}

    def cancel_fn():
        state["cancel"] += 1

    def preset_fn():
        state["preset"] += 1
        return (True, "残差 0.8°")

    def vis_fn():
        return vis_px

    cfg = {"dock_pose": list(dock), "settle_s": 0.5,
           "still_window_s": 0.6, "still_ticks": 6,
           "still_xy_m": 0.010, "still_yaw_rad": 0.0175,
           "pose_tol": [0.15, 0.12, 0.20], "vis_min_px": 50.0,
           "nav_timeout_s": 2.0, "cancel_timeout_s": 2.0}
    sup = HandoffSupervisor(mux, pose_fn=lambda: state["pose"],
                            cancel_fn=cancel_fn, arm_preset_fn=preset_fn,
                            visibility_fn=vis_fn, cfg=cfg, clock=clock,
                            verbose=False)
    return sup, state


def test_handoff_ok():
    clock = FakeClock()
    be = FakeBackend()
    mux = BaseVelocityMux(be, clock=clock, verbose=False)
    mux.set_owner(Owner.NONE)          # 交班前由导航占着也行，这里先用 NONE
    sup, st = _make_sup(mux, clock)
    order = []

    # 车位外
    sup.tick()
    order.append(sup.phase)
    # 进车位
    st["pose"] = (1.02, 0.03, 0.05)
    r = sup.tick()
    order.append(sup.phase)
    assert r["action"] == "cancel_nav" and st["cancel"] == 1, r
    # 一直推到 VLA 或失败为止。
    # 注意静止窗口里还留着「车位外」的旧样本，必须等它滑出窗口
    # （still_window_s=0.6）才开始计静止 —— 这正是真机上的行为。
    n = 0
    for _ in range(200):
        clock.advance(0.05)
        st["pose"] = (1.02, 0.03, 0.05)
        r = sup.tick()
        order.append(sup.phase)
        n += 1
        if sup.phase in (Phase.VLA, Phase.ABORT):
            break

    assert st["preset"] == 1, "手臂预设应恰好执行一次"
    assert sup.phase == Phase.VLA, (sup.phase, sup.reason)
    assert r["action"] == "run_vla", r
    assert mux.owner == Owner.VLA
    assert r["metrics"]["vis_px"] == 120.0
    # 相位只前进不回退
    rank = {Phase.EXTERNAL_NAV: 0, Phase.CANCEL_NAV: 1, Phase.SETTLE: 2,
            Phase.ARM_PRESET: 3, Phase.VISIBILITY: 4, Phase.VLA: 5}
    ranks = [rank[p] for p in order]
    assert ranks == sorted(ranks), order
    print("  4. handoff 正路 OK  %s（交班耗时 %.2f s）"
          % (" → ".join(dict.fromkeys(order)), n * 0.05))
    return True


def test_handoff_reject():
    # (a) 可见性不足
    clock = FakeClock()
    be = FakeBackend()
    mux = BaseVelocityMux(be, clock=clock, verbose=False)
    mux.set_owner(Owner.NONE)
    sup, st = _make_sup(mux, clock, vis_px=10.0)
    st["pose"] = (1.0, 0.0, 0.0)
    sup.tick()
    for _ in range(80):
        clock.advance(0.05)
        r = sup.tick()
        if sup.phase == Phase.ABORT:
            break
    assert sup.phase == Phase.ABORT, sup.phase
    assert "px" in sup.reason, sup.reason
    assert mux.owner == Owner.NONE, "拒绝后必须写 0，不能留给 VLA"

    # (b) 停车过程中车位漂出
    clock2 = FakeClock()
    be2 = FakeBackend()
    mux2 = BaseVelocityMux(be2, clock=clock2, verbose=False)
    sup2, st2 = _make_sup(mux2, clock2)
    st2["pose"] = (1.0, 0.0, 0.0)
    sup2.tick()
    st2["pose"] = (1.6, 0.0, 0.0)      # 漂出 0.6 m
    clock2.advance(0.05)
    sup2.tick()
    assert sup2.phase == Phase.ABORT and "漂出" in sup2.reason, sup2.reason

    # (c) EXTERNAL 期间拒绝不得抢底盘
    clock3 = FakeClock()
    be3 = FakeBackend()
    mux3 = BaseVelocityMux(be3, clock=clock3, verbose=False)
    assert mux3.owner == Owner.EXTERNAL
    sup3, st3 = _make_sup(mux3, clock3, vis_px=0.0)
    st3["pose"] = (9.9, 9.9, 0.0)      # 永远进不了车位 → 超时
    for _ in range(10):
        clock3.advance(0.5)
        sup3.tick()
        if sup3.phase == Phase.ABORT:
            break
    assert sup3.phase == Phase.ABORT
    assert mux3.owner == Owner.EXTERNAL and be3.calls == [], \
        "还在导航手上时，拒绝不许碰底盘"
    print("  5. handoff 拒绝 OK  可见性不足 / 车位漂出 / 超时均快速失败")
    return True


def test_wrap():
    seed = np.radians([0.0, 51.0, 37.0, 0.0, 91.0, 265.0])
    q = wrap_fwd2([0.0, -13.6, 134.9, 0.0, -16.2, -90.0], seed)
    assert abs(np.degrees(q[5]) - 270.0) < 1e-6, np.degrees(q)
    assert abs(np.degrees(q[1]) - (-13.6)) < 1e-6
    print("  6. 预设绕圈   OK  joint6 选 270°（离 265° 只差 5°）")
    return True


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("nav 协同层自检")
    mux, be, _clock = test_mux()
    test_proxy(mux, be)
    test_handoff_ok()
    test_handoff_reject()
    test_wrap()
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
