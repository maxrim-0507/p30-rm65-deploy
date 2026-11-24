# -*- coding: utf-8 -*-
"""L2 技能管理器：状态机 + 超时 + 重试 + DAgger 记录。

=== 为什么状态机放 L2 而不是 L1 ===
L1 是 1–5 Hz 的任务规划（选目标、定顺序），不该管 30 Hz 的推进。
把「预抓 → 操作 → 验证」的推进放在 L2，L1 只看每步的
SkillFeedback。这个分工直接决定了失败时的行为：L2 内部能重试的
（比如夹爪滑落）不必上抛给 L1 重规划。

=== 状态机 ===
    IDLE → PREGRASP → MANIP → VERIFY → DONE
                 ↘ FAULT ↗        ↘ RETRY
失败的重试策略按技能分：
  预抓失败（IK/碰撞）→ 不重试，直接 FAULT（换点位要靠 L1 重规划）
  操作失败（策略乱动）→ 回预抓位重试 1 次
  验证失败（夹空/夹错）→ 回预抓位重试 1 次
这些次数写在这里，不散落在各技能里。
"""
import time

import numpy as np

from ..bus.config import SKILLS
from .base import SkillResult
from .manipulate import ManipulateSkill
from .pregrasp import PregraspSkill
from .verify import VerifySkill


class SkillManager:

    def __init__(self, backend, policy, rate_hz=30.0,
                 manip_timeout_s=None):
        self.backend = backend
        self.policy = policy
        self.rate_hz = float(rate_hz)
        self.manip_timeout_s = manip_timeout_s
        self.stats = {"pregrasp": [0, 0], "manip": [0, 0], "verify": [0, 0]}
        self.log = []

    # ---------- 单次尝试 ----------
    def _skills(self):
        cfg = SKILLS
        return {
            "pregrasp": PregraspSkill(
                self.backend,
                pre_height=cfg["pregrasp"]["pre_height"],
                hover_height=cfg["pregrasp"]["hover_height"],
                rate_hz=self.rate_hz, timeout_s=20.0),
            "manip": ManipulateSkill(
                self.backend, self.policy,
                rate_hz=self.rate_hz,
                timeout_s=float(self.manip_timeout_s) if
                self.manip_timeout_s is not None else
                float(cfg["manipulate"]["timeout_s"])),
            "verify": VerifySkill(
                self.backend,
                settle_s=float(cfg["verify"]["settle_s"]),
                rate_hz=self.rate_hz, timeout_s=10.0),
        }

    def attempt(self, task, skip_pregrasp=False):
        """task = {"instruction", "target_color", "target_xy", "target_yaw"}"""
        sk = self._skills()
        t_start = time.monotonic()
        rec = {"instruction": task.get("instruction", ""),
               "target_color": task.get("target_color"),
               "states": []}

        # --- 1. 记录抓取前真值（用于 Verify 的抬升判据）---
        pre_z = {o["name"]: float(o["pos"][2])
                 for o in self.backend.get_objects()}
        ctx = dict(task)
        # 移动底盘：把当前停车位姿带进技能。任务里显式写 base_pose=None
        # 表示「不补偿」（模拟按标称位训练出来的 VLA），这里不再自动填。
        if ("base_pose" not in ctx
                and getattr(self.backend, "supports_base", False)):
            b = self.backend.get_base_state()
            if b.get("pos") is not None and b.get("yaw") is not None:
                ctx["base_pose"] = (float(b["pos"][0]), float(b["pos"][1]),
                                    float(b["yaw"]))
        ctx["pre_z_map"] = pre_z
        ctx["grasp_grip"] = float(self.backend.get_state()["gripper"])

        # --- 2. Pregrasp ---
        if skip_pregrasp:
            # 导航交班路径：TUCK 已把臂放到 FWD2（训练数据首帧），
            # 接近与底盘移动由 VLA 自己完成。不能再跑经典 IK 预抓：
            # 泊车位 0.70 m 时台上高度的目标 x≈0.71 m，超出 IK 可达
            # 域（实测 x≤0.49 m），必然「抓取点不可达」。
            rec["states"].append(("pregrasp", True, "跳过（FWD2 交班）"))
            ctx["q"] = None
        else:
            r_pre = sk["pregrasp"].run(ctx)
            self._count("pregrasp", r_pre.ok)
            rec["states"].append(("pregrasp", r_pre.ok, r_pre.message))
            if not r_pre.ok:
                rec["elapsed_s"] = time.monotonic() - t_start
                return SkillResult(False, "FAULT",
                                   "预抓失败：" + r_pre.message, rec)
            ctx["q"] = r_pre.data.get("q")

        # --- 3. Manipulate ---
        r_man = sk["manip"].run(ctx)
        self._count("manip", r_man.ok)
        rec["states"].append(("manip", r_man.ok, r_man.message))
        if not r_man.ok:
            rec["elapsed_s"] = time.monotonic() - t_start
            return SkillResult(False, "FAULT", "操作失败：" + r_man.message,
                               rec)

        # --- 4. Verify ---
        r_ver = sk["verify"].run(ctx)
        self._count("verify", r_ver.ok)
        rec["states"].append(("verify", r_ver.ok, r_ver.message))
        rec["verify"] = r_ver.data
        rec["elapsed_s"] = time.monotonic() - t_start
        self.log.append(rec)
        return SkillResult(r_ver.ok,
                           "DONE" if r_ver.ok else "FAULT",
                           r_ver.message, rec)

    # ---------- 带重试的完整流程 ----------
    def run_task(self, task, max_attempts=2, skip_pregrasp=False):
        last = None
        for i in range(max_attempts):
            last = self.attempt(task, skip_pregrasp=skip_pregrasp)
            if last.ok:
                return last
            # 预抓的 IK 失败重试没有意义：同一点位、同一种子会给同一结果
            if any(s[0] == "pregrasp" and not s[1] for s in
                   last.data.get("states", [])):
                break
            self._retreat()
        return last

    def _retreat(self):
        """失败后回到安全位（比当前高 15 cm 且后撤），再重试。"""
        st = self.backend.get_state()
        q_up = st["joint"].copy()
        # 使用抓取链里的抬升点语义：膝/肩各让一点，把末端抬高
        q_up[1] -= np.radians(8.0)
        q_up[2] += np.radians(5.0)
        sk = PregraspSkill(self.backend, rate_hz=self.rate_hz, timeout_s=8.0)
        sk.move_joints(st["joint"], q_up, st["gripper"], 1.0)
        sk.hold_for(0.3)

    def _count(self, key, ok):
        self.stats[key][1] += 1
        self.stats[key][0] += int(bool(ok))

    def summary(self):
        lines = []
        for k, (ok, n) in self.stats.items():
            pct = 100.0 * ok / n if n else 0.0
            lines.append("%-9s %d/%d = %.1f%%" % (k, ok, n, pct))
        return "\n".join(lines)
