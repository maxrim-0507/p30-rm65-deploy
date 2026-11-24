# -*- coding: utf-8 -*-
"""L1+L2 入口（Thor）：解析指令 → 技能执行。

用法：
  # 单进程（L3 也在本进程，仿真最省事）
  MUJOCO_GL=egl python3 -m deployment.run_l2 --inline --instruction "抓取红色的方块"

  # 跨进程（L3 已在别处跑）
  python3 -m deployment.run_l2 --backend remote --instruction "抓取蓝色的方块"

  # 批量 + 随机摆位（回归用）
  MUJOCO_GL=egl python3 -m deployment.run_l2 --inline --trials 20 --seed 7

=== --inline 的意义 ===
它把 L3 的 ControlServer 起在后台线程里，用**同一个 backend 对象**
给 L2。这样单进程调试时没有 msgpack/网络这一跳，能立刻区分
「技能逻辑错」和「总线错」。两种模式共用同一份技能与任务代码。
"""
import argparse
import os
import sys
import threading
import time

from .bootstrap import setup

setup()

import numpy as np  # noqa: E402

from .bus.config import POLICY, SKILLS  # noqa: E402
from .hal import make_backend  # noqa: E402
from .policy import make_policy  # noqa: E402
from .skills.manager import SkillManager  # noqa: E402

COLORS = ["red", "green", "blue", "yellow"]
ZH = {"red": "红色", "green": "绿色", "blue": "蓝色", "yellow": "黄色"}


def _sample_layout(rng, n=4):
    """随机摆位：从 reach_mask 的可用格点里取，保证间距与台面内。

    直接复用 collect_color_grasp.py 的选点逻辑思路（掩膜优先），
    但这里只需要一次摆位，不生成数据，所以写得轻一点。
    """
    import json
    from .spec import SIM_DIR
    mask_path = os.path.join(SIM_DIR, "reach_mask.json")
    cells = []
    if os.path.isfile(mask_path):
        with open(mask_path, encoding="utf-8") as f:
            mk = json.load(f)
        xs = np.asarray(mk["meta"]["xs"])
        ys = np.asarray(mk["meta"]["ys"])
        mask = np.asarray(mk["mask"], dtype=bool)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                if mask[j, i] and 0.440 <= x <= 0.480 and abs(y) <= 0.200:
                    cells.append((float(x), float(y)))
    if len(cells) < n:
        raise RuntimeError("reach_mask 可用格点不足 %d 个（实得 %d）"
                           % (n, len(cells)))

    for _ in range(2000):
        idx = rng.permutation(len(cells))[:n]
        pts = [cells[i] for i in idx]
        ok = True
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                if np.hypot(pts[a][0] - pts[b][0],
                            pts[a][1] - pts[b][1]) < 0.085:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            colors = COLORS[:n].copy()
            rng.shuffle(colors)
            return [(x, y, colors[k], float(rng.uniform(0, np.pi / 2)))
                    for k, (x, y) in enumerate(pts)]
    raise RuntimeError("摆位采样失败（间距约束过紧）")


def _reset_arm_q(layout, text):
    """按目标物体的 pre 点算一个 reset 用的关节角；算不出就返回 None。

    === 为什么用 pre 点而不是 keyframe ===
    采集数据每条 demo 都是从 pre 点开始的（collect_color_grasp 里
    `self.reset(poses, q_pre)`）。部署端如果从 keyframe 的 stand 姿态起，
    joint6 起点是 -331.6°，走到 pre 点要转 248°~505°。同一套策略在
    「臂已经在那附近」和「臂要先甩一大圈」两种状态下行为完全不同。
    用 pre 点让部署起点与训练起点一致。
    """
    from .spec import COLOR_ZH
    from .skills import kinematics

    color = None
    # COLOR_ZH 是 en→zh。这里曾把 key 当中文用，导致匹配永远失败、
    # arm_q 恒为 None（机械臂从 keyframe 收拢姿态起步，joint6 要甩圈）。
    for en, zh in COLOR_ZH.items():
        if zh in text:
            color = en
            break
    if color is None:
        return None
    for pose in layout:
        if len(pose) >= 3 and pose[2] == color:
            sol = kinematics.solve_approach((pose[0], pose[1]))
            if sol["ok"]:
                q = np.asarray(sol["q"]["pre"], float)
                # joint6 是多圈解，可能落在 −323° 这类分支。训练数据里
                # joint6 全在 (−180°, +180°]，不归一化有两个后果：
                #   1. VLA 收到分布外的关节角；
                #   2. FWD2/SEARCH 预设的等价角会选到 −450°（超出
                #      ±359.8° 限位），伺服爬不到 → 「残差 94.7°」。
                return (q + np.pi) % (2.0 * np.pi) - np.pi
    return None


def _target_xy_for(layout, text):
    """本局要抓颜色的世界 xy。这是**环境真值**（摆物体用），不是感知。"""
    from .task.instruction import parse
    color = parse(text).color
    for pose in layout:
        if len(pose) >= 3 and pose[2] == color:
            return (float(pose[0]), float(pose[1]))
    return None


def _far_start(rng, target_xy):
    """陌生房间里的远起点：目标周围 1.1–1.7 m 的环上，朝向目标 ±17°。

    环半径上限受房间与障碍箱限制：y 最多偏到 1.78 m（墙在 ±2.45），
    最近 x 到 −1.24 m（障碍箱在 −1.40）。朝向目标是为了让搜索头几帧
    就有用；真正的捕获靠原地扫描。
    """
    r = float(rng.uniform(1.1, 1.7))
    a = float(rng.uniform(-1.2, 1.2))
    x = float(target_xy[0] - r * np.cos(a))
    y = float(target_xy[1] - r * np.sin(a))
    yaw = a + float(rng.uniform(-0.3, 0.3))
    return (x, y, yaw)


def _run_search_then_grasp(backend, sm, planner, text):
    """--search：先真感知找目标 + 泊车，再把结论直接下给 L2 抓取。

    === 为什么不继续走 planner.run（L1）===
    泊好车时 front_cam 朝下、只看车前约 1 m，L1 的 grounding 看不到台面
    目标；而导航阶段已经在 arm_cam 里量出了目标 xy。这里把导航结论作为
    与 TaskSpec.to_task_dict() 同构的 dict 直接下给 SkillManager，不经
    L1 重定位 —— 不会拿「看不见」去赌一次随机 grounding。
    """
    from .bus.config import TASK
    from .nav.percept_pipeline import PerceptNav
    from .task.instruction import parse

    color = parse(text).color
    nav = PerceptNav(backend, color, verbose=True)
    nres = nav.run()
    rec = {"instruction": text,
           "nav": {"ok": nres.ok, "phase": nres.phase,
                   "reason": nres.reason,
                   "target_xy": nres.target_xy,
                   "goal": None if nres.goal is None else nres.goal.tolist(),
                   "vis_px": nres.vis_px, "err_xy": nres.err_xy,
                   "frames": nres.frames, "truth_calls": nres.truth_calls}}
    if not nres.ok:
        rec.update(ok=False, reason="导航失败：" + nres.reason)
        planner.history.append(rec)
        print("[nav] %-22s FAIL  %s" % (text, nres.reason), flush=True)
        return rec
    if nres.truth_calls != 0:
        rec.update(ok=False, reason="导航读了真值（必须为 0，疑似回退）")
        planner.history.append(rec)
        return rec

    task = {"instruction": text, "target_color": color,
            "target_xy": nres.target_xy, "target_yaw": None}
    t0 = time.monotonic()
    # 跳过经典 Pregrasp：TUCK 后已在 FWD2，泊车距离超出台上 IK 可达域，
    # 接近/底盘移动由 VLA 自己完成（与 v36 训练数据的首帧一致）。
    res = sm.run_task(task, max_attempts=int(
        TASK.get("replan_max_attempts", 2)), skip_pregrasp=True)
    rec.update(ok=bool(res.ok), state=res.state, message=res.message,
               detail=res.data, elapsed_s=time.monotonic() - t0,
               source="nav+hsv")
    planner.history.append(rec)
    print("[nav] %-22s %s  %s"
          % (text, "OK" if rec["ok"] else "FAIL",
             rec.get("message", rec.get("reason", ""))), flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="L1+L2 抓取闭环")
    ap.add_argument("--instruction", default=None,
                    help='例如 "抓取红色的方块"；不给则遍历四色')
    ap.add_argument("--backend", default=None,
                    help="mujoco / real / remote；--inline 时强制 mujoco")
    ap.add_argument("--inline", action="store_true",
                    help="L3 起在本进程后台线程（单进程调试）")
    ap.add_argument("--trials", type=int, default=1,
                    help="每条指令重复多少次（每次重新随机摆位）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default=None,
                    help="oracle / gr00t；缺省读 deploy.yaml 的 policy.kind")
    ap.add_argument("--grounding", default=None,
                    help="auto / vlm / hsv / truth；缺省读 deploy.yaml 的 grounding.prefer")
    ap.add_argument("--forbid-truth", action="store_true",
                    help="拦截 grounding 的仿真真值路径（A 方案评测必须打开）。"
                         "auto 会退化成 hsv；成功的判定/记账仍可用真值")
    ap.add_argument("--search", action="store_true",
                    help="B2 房间自主路线：真感知找目标 + 视觉泊车，再把"
                         "结论交给 VLA 抓取（需 nav 场景）")
    ap.add_argument("--start-far", action="store_true",
                    help="每局把底盘放到目标 1.1–1.7 m 外；--search 时默认开")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--realtime", action="store_true", default=True)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    # 不给 --backend 时默认单进程仿真：这是最省事、也最常用的调试方式。
    use_inline = a.inline or (a.backend is None) or (a.backend == "mujoco")

    # ---- 后端 ----
    if use_inline:
        # L3 与硬件后端在本进程；L2 只拿到一个代理。
        # 绝不能把 MujocoBackend 直接交给 L2 —— 两个线程会同时写 ctrl。
        kw = {"realtime": a.realtime}
        if a.no_render:
            kw["render"] = False
        hw = make_backend("mujoco", **kw)
    else:
        hw = None

    if use_inline:
        hw.connect()
        from .control.server import ControlServer
        srv = ControlServer(hw, verbose=True)
        srv.connect()
        th = threading.Thread(target=srv.run, daemon=True)
        th.start()
        from .hal import wrap_local
        backend = wrap_local(srv)
        backend.connect()
        time.sleep(0.3)
    else:
        srv = None
        backend = make_backend(a.backend)
        backend.connect()

    # ---- 策略与技能 ----
    policy = make_policy(a.policy)
    sm = SkillManager(backend, policy, rate_hz=30.0)

    # --forbid-truth：auto 的第一优先是仿真真值，不拦掉的话
    # 感知前端根本不会被调用，评测结果没有意义。这里只改 grounding
    # 的 prefer（truth -> hsv），verify/manager 的真值记账不受影响。
    prefer = a.grounding
    if a.forbid_truth and prefer in (None, "auto", "truth"):
        prefer = "hsv"
    from .task.planner import TaskPlanner
    planner = TaskPlanner(backend, sm, prefer_grounding=prefer,
                          forbid_truth=a.forbid_truth)

    # ---- 指令列表 ----
    if a.instruction:
        texts = [a.instruction]
    else:
        texts = ["抓取%s的方块" % ZH[c] for c in COLORS]

    # ---- 跑 ----
    results = []
    try:
        for trial in range(a.trials):
            layout = _sample_layout(rng)
            for text in texts:
                # 重置必须走 L3（唯一能碰硬件的一方）。
                # inline 与跨进程都统一走 handle/REQ —— reset 里面已经
                # 包含 settle，主进程绝不能再碰 backend（会与 L3 的 mj_step
                # 并发，实测直接段错误）。
                #
                # === 必须同时给 arm_q ===
                # 不给的话 reset 回到 keyframe 的 stand 姿态（joint6 = -331.6°），
                # 而采集数据里每条 demo 的起始位是 pre 点（joint6 在
                # [-173°, +180°]）。实测两者相差最大 505°，真机上关节要甩
                # 一圈半。这里用目标物体的 pre 点解作初始位，与采集一致。
                arm_q = _reset_arm_q(layout, texts[0])
                base_pose = None
                if a.search or a.start_far:
                    tgt = _target_xy_for(layout, text)
                    if tgt is None:
                        print("[warn] 本局找不到目标颜色，忽略远起点")
                    else:
                        base_pose = _far_start(rng, tgt)
                if srv is not None:
                    rep = srv.handle({"op": "reset", "objects": layout,
                                      "arm_q": arm_q,
                                      "base_pose": base_pose})
                else:
                    rep = backend.req.call({"op": "reset", "objects": layout,
                                            "arm_q": arm_q,
                                            "base_pose": base_pose})
                if not rep.get("ok"):
                    print("[warn] reset 失败：%s" % rep.get("err"))
                time.sleep(0.15)        # 等一帧新快照广播出去
                if a.search:
                    rec = _run_search_then_grasp(backend, sm, planner, text)
                else:
                    rec = planner.run(text)
                results.append(rec)
                print("[trial %d] %s -> %s  (%s)"
                      % (trial, text, "OK" if rec["ok"] else "FAIL",
                         rec.get("message", rec.get("reason", ""))),
                      flush=True)
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        print("\n" + planner.summary())
        n_ok = sum(1 for r in results if r["ok"])
        print("总成功率 %d/%d = %.1f%%"
              % (n_ok, len(results), 100.0 * n_ok / max(len(results), 1)))
        if srv is not None:
            srv.rep.stop()
            time.sleep(0.2)
            srv.close()
        else:
            backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
