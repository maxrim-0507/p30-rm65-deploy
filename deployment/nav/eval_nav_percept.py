# -*- coding: utf-8 -*-
"""S4 回归：N 局「真感知找目标 + 视觉泊车 + 收臂交班」，不读真值。

用法（Thor，无显示）：
    MUJOCO_GL=egl python3 -m deployment.nav.eval_nav_percept --trials 20

可选：
    --with-vla       泊车后继续跑 VLA 抓取（需要 GR00T 服务）
    --color red      固定颜色（缺省每局随机）
    --seed 7         复现同一组起点/摆位

=== 判据 ===
  1. 每局都要找到目标并泊进交班盒；
  2. 定位来源只能是 hsv，get_objects 调用必须为 0（真值防火墙）；
  3. 成功判定用环境真值（摆物体时的目标 xy），**只用于记账**，不参与
     控制闭环 —— 真机上换成「到位后 arm_cam 可见 + VLA verify」。

=== 为什么必须走 ControlServer（inline）===
L2 若直接拿 MujocoBackend，会和 L3 主循环并发写同一个 MjData：实测
表现为机械臂纹丝不动（看门狗每拍把 ctrl 写回实测值）。所以这里和
run_l2 一样只给 L2 一个 LocalBackend 代理。
"""
import argparse
import os
import threading
import time

from ..bootstrap import setup

setup()

import numpy as np  # noqa: E402

from ..hal import make_backend, wrap_local  # noqa: E402
from ..run_l2 import (COLORS, ZH, _far_start, _reset_arm_q,  # noqa: E402
                      _sample_layout, _target_xy_for)
from ..skills.manager import SkillManager  # noqa: E402
from .percept_pipeline import PerceptNav  # noqa: E402
from .visual_dock import dock_pose  # noqa: E402


def _build_inline(realtime=True, scene="p30_rm65_nav.xml", render=True):
    # 感知链路必须有图像 → 缺省开渲染（Thor 上走 EGL）。本机 Windows
    # 软渲染一次双相机要 0.14 s，仿真只有 ~0.07x 实时，跑视觉回归必须
    # 用 --no-render（此时只能验物理/控制，图像为空）。
    hw = make_backend("mujoco", realtime=realtime, scene=scene,
                      render=render)
    hw.connect()
    from ..control.server import ControlServer
    srv = ControlServer(hw, verbose=False)
    srv.connect()
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    backend = wrap_local(srv)
    backend.connect()
    time.sleep(0.3)
    return srv, backend


def _one(srv, backend, rng, color, with_vla, sm, nav_cfg=None):
    layout = _sample_layout(rng)
    text = "抓取%s的方块" % ZH[color]
    arm_q = _reset_arm_q(layout, text)
    tgt = _target_xy_for(layout, text)
    base_pose = _far_start(rng, tgt)

    rep = srv.handle({"op": "reset", "objects": layout, "arm_q": arm_q,
                      "base_pose": base_pose})
    if not rep.get("ok"):
        return {"ok": False, "reason": "reset 失败：%s" % rep.get("err"),
                "base_pose": base_pose}
    time.sleep(0.15)

    nav = PerceptNav(backend, color, cfg=nav_cfg, verbose=False)
    t0 = time.monotonic()
    res = nav.run()
    dt = time.monotonic() - t0

    true_goal = dock_pose(tgt, (base_pose[0], base_pose[1]))
    b = backend.get_base_state()
    rec = {"ok": res.ok, "color": color, "reason": res.reason,
           "phase": res.phase, "base_pose": base_pose, "true_tgt": tgt,
           "true_goal": true_goal, "target_xy": res.target_xy,
           "goal": res.goal, "vis_px": res.vis_px, "err_xy": res.err_xy,
           "frames": res.frames, "truth_calls": res.truth_calls,
           "sources": res.sources, "nav_s": dt}
    if res.target_xy is not None:
        rec["tgt_err"] = float(np.hypot(res.target_xy[0] - tgt[0],
                                        res.target_xy[1] - tgt[1]))
    if b.get("pos") is not None and b.get("yaw") is not None:
        rec["layout"] = layout
        rec["base_end"] = (float(b["pos"][0]), float(b["pos"][1]),
                           float(b["yaw"]))
        rec["dock_err_truth"] = float(np.hypot(b["pos"][0] - true_goal[0],
                                               b["pos"][1] - true_goal[1]))

    if res.ok and with_vla:
        if sm is None:
            raise RuntimeError("--with-vla 需要可用的策略（见 deploy.yaml）")
        task = {"instruction": text, "target_color": color,
                "target_xy": res.target_xy, "target_yaw": None}
        r = sm.run_task(task, max_attempts=2, skip_pregrasp=True)
        rec["grasp_ok"] = bool(r.ok)
        rec["grasp_msg"] = r.message
        rec["grasp_states"] = ((r.data or {}).get("states")
                              if isinstance(r.data, dict) else None)
        rec["ok"] = bool(r.ok)
        if os.environ.get("GR00T_DEBUG"):
            try:
                import mujoco as _mj
                mb = backend.srv.backend
                _mj.mj_kinematics(mb.m, mb.d)
                gs = np.asarray(mb.d.site_xpos[mb._grasp_site])
                print("[gap] eef_world=%s true_tgt=%s grip=%.3f"
                      % (np.round(gs, 3), np.round(tgt, 3),
                         float(backend.get_state()["gripper"])), flush=True)
                for o in mb.get_objects():
                    dd = float(np.linalg.norm(
                        np.asarray(o["pos"], float) - gs))
                    print("   cube %-6s pos=%s d=%.3f"
                          % (o["color"], np.round(o["pos"], 3), dd),
                          flush=True)
                # === 可达性探针：目标=真实方块位置，问原始 IK 能否解 ===
                cw=None; best=1e9
                for o in mb.get_objects():
                    q2=float(np.hypot(o["pos"][0]-tgt[0], o["pos"][1]-tgt[1]))
                    if q2<best: best=q2; cw=np.asarray(o["pos"],float)
                cb=mb._base_R.T @ (cw - mb._base)
                bp_nom,bR_nom=mb._nominal_base()
                tgt_static=bR_nom @ cb + bp_nom
                R_site=np.asarray(mb.d.site_xmat[mb._grasp_site],float).reshape(3,3)
                R_tgt=bR_nom @ (mb._base_R.T @ R_site)
                seed=(mb._q_ik_seed if mb._q_ik_seed is not None else mb._q_cmd)
                qq,ee=mb._ik_algo_solve(tgt_static,R_tgt,seed)
                print("[reach] cube_base=%s static=%s ik_ok=%s err=%.3g"
                      % (np.round(cb,3),np.round(tgt_static,3),qq is not None,ee),flush=True)
                for sh in (1.0,0.95,0.9,0.85,0.8,0.7):
                    ts=bp_nom+(tgt_static-bp_nom)*sh
                    qx,ex=mb._ik_algo_solve(ts,R_tgt,seed)
                    print("   [reach] shrink=%.2f ik_ok=%s"%(sh,qx is not None),flush=True)
            except Exception as _e:
                print("[gap] err", _e, flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="真感知导航回归（不读真值）")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--color", default=None, choices=COLORS)
    ap.add_argument("--with-vla", action="store_true")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--ckpt", default=None,
                    help="--policy local 的 checkpoint 目录"
                         "（含 pretrained_model）")
    ap.add_argument("--scene", default="p30_rm65_nav.xml",
                    help="导航场景（必须有 base_vx/vy/wz 与 front_cam/arm_cam）")
    ap.add_argument("--no-render", action="store_true",
                    help="关渲染（本机软渲染太慢时用；此时没有图像，感知必失败）")
    ap.add_argument("--approach-s", type=float, default=None,
                    help="泊车墙钟上限（秒）。本机软渲染 0.07x 实时，45 s 只"
                         "相当于 3 s 物理时间，诊断时要放大到 300")
    ap.add_argument("--search-s", type=float, default=None,
                    help="搜索墙钟上限（秒），同上")
    ap.add_argument("--manip-s", type=float, default=None,
                    help="VLA 操作墙钟上限（秒）；缺省读 deploy.yaml")
    a = ap.parse_args(argv)
    nav_cfg = {}
    if a.approach_s is not None:
        nav_cfg["max_approach_s"] = float(a.approach_s)
    if a.search_s is not None:
        nav_cfg["max_search_s"] = float(a.search_s)
    nav_cfg = nav_cfg or None

    rng = np.random.default_rng(a.seed)
    # 必须实时：L2 每帧 sleep frame_dt，物理若全速推进，底盘每拍会跑过头
    srv, backend = _build_inline(realtime=True, scene=a.scene,
                                 render=not a.no_render)
    sm = None
    if a.with_vla:
        from ..policy import make_policy
        kw = {"ckpt": a.ckpt} if a.ckpt else {}
        if a.policy == "local":
            # 流匹配动作头：不播种则同一观测动作不可复现（回归 0/18 的根因）。
            kw["seed"] = a.seed
        sm = SkillManager(backend, make_policy(a.policy, **kw),
                          rate_hz=30.0, manip_timeout_s=a.manip_s)

    recs = []
    try:
        for i in range(a.trials):
            color = a.color or COLORS[int(rng.integers(len(COLORS)))]
            try:
                rec = _one(srv, backend, rng, color, a.with_vla, sm,
                           nav_cfg=nav_cfg)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                rec = {"ok": False, "color": color, "reason": "异常：%s" % e}
            recs.append(rec)
            print("[%2d] %-5s %s  tgt_err=%s dock_err=%s vis=%s  %s"
                  % (i + 1, color, "OK  " if rec["ok"] else "FAIL",
                     _fmt(rec.get("tgt_err"), "%.3f"),
                     _fmt(rec.get("dock_err_truth"), "%.3f"),
                     _fmt(rec.get("vis_px"), "%.0f"),
                     rec.get("grasp_msg") or rec.get("reason", "")),
                  flush=True)
            if os.environ.get("EVAL_PRINT_LAYOUT"):
                print("[layout] randn=%s base_end=%s" % (repr(rec.get("layout")), rec.get("base_end")), flush=True)
    finally:
        srv.rep.stop()
        time.sleep(0.2)
        srv.close()

    n = len(recs)
    ok = sum(1 for r in recs if r["ok"])
    truth = sum(int(r.get("truth_calls", 0)) for r in recs)
    te = [r["tgt_err"] for r in recs if "tgt_err" in r]
    de = [r["dock_err_truth"] for r in recs if "dock_err_truth" in r]
    print("\n导航成功：%d/%d = %.1f%%" % (ok, n, 100.0 * ok / max(n, 1)))
    if te:
        print("反投影误差：均值 %.1f mm / 最大 %.1f mm"
              % (1e3 * float(np.mean(te)), 1e3 * float(np.max(te))))
    if de:
        print("泊车误差（对真值车位）：均值 %.1f mm / 最大 %.1f mm"
              % (1e3 * float(np.mean(de)), 1e3 * float(np.max(de))))
    print("真值调用：%d（必须为 0）" % truth)
    fails = [r for r in recs if not r["ok"]]
    for r in fails:
        print("  FAIL %-5s %s" % (r.get("color"),
                                      r.get("grasp_msg") or r.get("reason")))
        if r.get("grasp_states"):
            print("       states:", r["grasp_states"])
    if truth != 0 or ok != n:
        return 1
    return 0


def _fmt(v, f):
    return "-" if v is None else (f % v)


if __name__ == "__main__":
    raise SystemExit(main())
