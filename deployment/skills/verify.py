# -*- coding: utf-8 -*-
"""Verify 技能：判定「到底抓没抓到、抓的是不是目标色」。

=== 三层判据，从强到弱 ===
1. 真值（仿真）：物体 z 相对抓取前抬升 > lift_delta_min，且被夹的是
   目标色。这是 vla_spec.yaml 里 task.success 的原始定义。
2. 关节/夹爪：夹爪闭合量 vs 张开量的差值 —— 手里有东西时夹爪合不到底。
   真机上没有真值时用它做粗判。
3. 图像：腕部相机中心区域出现目标色的连通域。作为交叉验证。

=== 为什么真值优先而不是唯一 ===
真机上永远没有真值。判定逻辑必须能退化成 2/3 两条，否则 Phase 2
真机部署那天会突然没有成功信号。这里把三条都算出来，按可用性选。
"""
import time

import numpy as np

from ..spec import LIFT_DELTA_MIN
from .base import Skill, SkillResult


class VerifySkill(Skill):
    name = "verify"

    def __init__(self, backend, settle_s=1.0, **kw):
        super().__init__(backend, **kw)
        self.settle_s = float(settle_s)

    def run(self, ctx):
        """ctx 需要：
            target_color   "red"/"green"/...
            pre_z_map      {obj_name: z} 抓取前记录的物体高度（可选）
            grasp_grip     抓取前夹爪开度（可选）
        """
        self.hold_for(self.settle_s)
        st = self.backend.get_state()
        color = ctx.get("target_color")
        out = {"color_ok": None, "lift": None, "lift_ok": None,
               "grip_delta": None, "by": []}

        # --- 判据 1：真值 ---
        objs = self.backend.get_objects()
        if objs:
            pre = ctx.get("pre_z_map") or {}
            best = None
            for o in objs:
                z0 = pre.get(o["name"])
                if z0 is None:
                    continue
                dz = float(o["pos"][2]) - float(z0)
                if best is None or dz > best[1]:
                    best = (o, dz)
            if best is not None:
                o, dz = best
                out["lift"] = dz
                out["lift_ok"] = dz >= LIFT_DELTA_MIN
                out["color_ok"] = (o["color"] == color)
                out["picked"] = o["color"]
                out["by"].append("truth")
                ok = bool(out["lift_ok"] and out["color_ok"])
                return SkillResult(ok, "VERIFY_DONE" if ok else "FAULT",
                                   "真值：抬升 %+.3f m，抓到 %s"
                                   % (dz, o["color"]), out)

        # --- 判据 2：夹爪闭合量 ---
        g0 = ctx.get("grasp_grip")
        if g0 is not None:
            # HAL 的 gripper 已经是归一化开度（1=张开），不要再映射一次
            g_now = float(st["gripper"])
            out["grip_delta"] = float(g0) - g_now
            out["by"].append("gripper")
            # 空抓时夹爪会合到底（delta≈1）；夹住 6cm 方块时约 0.4~0.7
            ok = out["grip_delta"] > 0.15
            return SkillResult(ok, "VERIFY_DONE" if ok else "FAULT",
                               "夹爪闭合量 %.2f" % out["grip_delta"], out)

        # --- 判据 3：图像 ---
        imgs = self.backend.get_images()
        if imgs:
            from ..perception.color_grounding import detect_blobs
            from ..spec import camera_mjcf
            wrist = imgs.get(camera_mjcf("arm"))
            if wrist is not None:
                blobs = detect_blobs(wrist, color)
                h, w = wrist.shape[:2]
                # 腕部相机看夹持区：目标色块面积够大就算夹住了
                area_frac = (sum(b["area"] for b in blobs) /
                             float(h * w))
                out["area_frac"] = area_frac
                out["by"].append("image")
                ok = area_frac > 0.01
                return SkillResult(ok, "VERIFY_DONE" if ok else "FAULT",
                                   "腕部图像目标色占比 %.3f" % area_frac, out)

        return SkillResult(False, "FAULT", "没有任何可用判据", out)
