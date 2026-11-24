# -*- coding: utf-8 -*-
"""L1 任务层：指令 → 目标 → 技能计划 → 结果汇总。

=== 为什么 L1 这么薄 ===
Phase 1 只有一个动作（抓）、四个目标（颜色）。任何更复杂的规划
都没有依据。这里刻意保持「一条直线」：解析 → 定位 → 执行 →
判定，出问题的地方一眼能看出来。

将来要扩的多步任务（导航到货架 → 抓 → 放到周转箱）在这里长成
steps 列表，L2 的技能管理器已经有状态机承接。现在写死单步不是
复杂度失控，是因为**还不需要**。

=== 可达性前置检查 ===
定位到目标后立刻判 `workspace`，不可达就不下发技能。理由：IK
失败在 L2 里表现为「预抓失败」，会被归进技能成功率，污染统计；
而不可达其实是场景布置问题，应该单独计数。
"""
import time

import numpy as np

from ..bus.config import CONTROL, GROUNDING, TASK
from ..spec import ARM_BASE
from .grounding import Grounding
from .instruction import describe, parse


class TaskPlanner:

    def __init__(self, backend, skill_manager, prefer_grounding=None,
                 verbose=True, forbid_truth=False):
        self.backend = backend
        self.sm = skill_manager
        cfg = GROUNDING or {}
        if prefer_grounding is None:
            prefer_grounding = cfg.get("prefer", "auto")
        self.ground = Grounding(backend, prefer=prefer_grounding,
                                camera=cfg.get("camera", "front_cam"),
                                vlm_cfg=cfg.get("vlm"),
                                forbid_truth=forbid_truth)
        self.verbose = verbose
        self.history = []

    # ---------- 单条任务 ----------
    def run(self, text):
        """指令 → TaskSpec → 可达性预检 → 下发 L2。

        L1 与 L2 之间只有 TaskSpec（contract.py），不要在这里临时拼
        dict：字段名一漂移，L2 拿到的是「看起来能用、实际偏了 3 cm」
        的数据，而且不会报错。
        """
        t0 = time.monotonic()
        intent = parse(text)                     # 仅用于日志和失败原因
        spec = self.ground.locate_task(text)
        rec = {"instruction": text, "intent": repr(intent),
               "spec": spec.to_dict()}

        errs = spec.validate()
        if errs:
            rec.update(ok=False, reason="任务不合法：" + "；".join(errs))
            self.history.append(rec)
            return rec

        ok_ws, why = self.reachable(spec.target_xy)
        if not ok_ws:
            rec.update(ok=False, reason="不可达：" + why)
            self.history.append(rec)
            return rec

        res = self.sm.run_task(spec.to_task_dict(), max_attempts=int(
            TASK.get("replan_max_attempts", 2)))

        rec.update(ok=bool(res.ok), state=res.state, message=res.message,
                   detail=res.data, elapsed_s=time.monotonic() - t0,
                   source=spec.source)
        self.history.append(rec)
        if self.verbose:
            self._log(rec)
        return rec

    # ---------- 可达性 ----------
    def reachable(self, xy):
        """先查工作空间盒，再做一次 IK 试解（比让技能失败更早、更省时）。"""
        ws = CONTROL.get("workspace") or {}
        p = np.asarray([xy[0], xy[1], 0.0], float) - np.asarray(ARM_BASE)
        for i, ax in enumerate("xy"):
            rng = ws.get(ax)
            if rng and not (rng[0] <= p[i] <= rng[1]):
                return False, "%s=%.3f 超出 [%.2f, %.2f]" % (
                    ax, p[i], rng[0], rng[1])
        try:
            from ..skills import kinematics
            sol = kinematics.solve_approach(xy)
        except Exception as e:
            return False, "IK 试解异常：%s" % e
        if not sol["ok"]:
            return False, sol["reason"]
        return True, ""

    # ---------- 批量 ----------
    def run_batch(self, instructions):
        out = []
        for t in instructions:
            out.append(self.run(t))
        return out

    def summary(self):
        n = len(self.history)
        ok = sum(1 for r in self.history if r.get("ok"))
        lines = ["L1 任务：%d/%d = %.1f%%" % (ok, n, 100.0 * ok / max(n, 1)),
                 self.sm.summary()]
        return "\n".join(lines)

    def _log(self, rec):
        print("[L1] %-22s %s  %s"
              % (rec["instruction"], "OK" if rec.get("ok") else "FAIL",
                 rec.get("message", rec.get("reason", ""))), flush=True)
