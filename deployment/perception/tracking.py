# -*- coding: utf-8 -*-
"""色块跟踪：单帧检测 → 「稳定可见」判定。

=== 为什么不能在单帧上判「发现」 ===
搜索时相机在转，一帧里的色块可能只是：
  - 背景/自车身上的高饱和色（ground_color 的 bounds 能过滤空间，
    但检测层还没反投影，管不到）；
  - 运动模糊边缘，面积忽大忽小；
  - 远处 6cm 方块只有 5~7 px 边长，单帧面积就是几十像素，噪声主导。

所以「发现」用时间证据：最近 n_window 帧里至少 min_hits 帧命中，
且像素质心抖动不超过 max_jitter_px。抖动是必要条件——真方块在
云台静止/匀速转时质心接近线性；随机高饱和斑块是跳变的。

=== 为什么单独一层 ===
SS2 搜索、SS3 泊车都要同一套判定，只是参数不同（搜索窗短、泊车
要求更稳）。放在这里，两边不各写一遍「连续 N 帧」。

=== 时间用注入的 clock ===
仿真里没有真实帧率抖动，测试要能推进虚拟时间；默认 time.monotonic。
"""
import time

import numpy as np


class TrackState(object):
    """一次 update 的结果。调用方只读，不要修改。"""

    __slots__ = ("found", "uv", "area", "hits", "window", "jitter_px",
                 "stale_s")

    def __init__(self, found, uv, area, hits, window, jitter_px, stale_s):
        self.found = bool(found)
        self.uv = uv
        self.area = area
        self.hits = int(hits)
        self.window = int(window)
        self.jitter_px = float(jitter_px)
        self.stale_s = float(stale_s)

    def __repr__(self):
        return ("TrackState(found=%s uv=%s area=%.1f %d/%d jit=%.1fpx "
                "stale=%.2fs)" % (self.found, self.uv, self.area, self.hits,
                                  self.window, self.jitter_px, self.stale_s))


def scale_min_area(min_area, width, height, base_w=320, base_h=240):
    """把在基准分辨率下标定的面积阈值换到当前分辨率。

    面积随分辨率平方变化：640×480 的阈值是 320×240 的 4 倍。
    远端 1.4 m 处方块在 320 宽下只有 ~5 px 边长，阈值定错就等于
    「永远发现不了」或「噪声都算目标」。
    """
    return float(min_area) * (float(width) * float(height)) / (
        float(base_w) * float(base_h))


class BlobTracker(object):

    def __init__(self, n_window=8, min_hits=5, max_jitter_px=2.5,
                 min_area=25.0, max_area=None, stale_s=0.6, clock=None):
        self.n_window = int(n_window)
        self.min_hits = int(min_hits)
        self.max_jitter_px = float(max_jitter_px)
        self.min_area = float(min_area)
        self.max_area = None if max_area is None else float(max_area)
        self.stale_s = float(stale_s)
        self._clock = clock or time.monotonic
        self.reset()

    def reset(self):
        self._samples = []          # [(t, uv|None, area)]，只留最近 n_window
        self._last_hit_t = None
        self._last_uv = None
        self._last_area = 0.0

    # ---------- 主入口 ----------
    def update(self, blobs, t=None):
        """blobs: detect_blobs 的输出（按面积降序）或空列表。"""
        now = self._clock() if t is None else float(t)
        best = self._pick(blobs)
        if best is None:
            self._samples.append((now, None, 0.0))
            self._trim()
        else:
            self._samples.append((now, best["uv"], float(best["area"])))
            self._trim()
            self._last_hit_t = now
            self._last_uv = (float(best["uv"][0]), float(best["uv"][1]))
            self._last_area = float(best["area"])

        hit = [(s[0], s[1]) for s in self._samples if s[1] is not None]
        hits = int(len(hit))
        jitter = 0.0
        if hits >= 2:
            ts = np.asarray([h[0] - hit[0][0] for h in hit], dtype=float)
            uv = np.asarray([h[1] for h in hit], dtype=float)
            if hits >= 3 and ts[-1] - ts[0] > 1e-6:
                # === 抖动要去掉匀速项，不能直接算 std ===
                # 原地扫描时相机以 ~0.6 rad/s 转，世界固定的方块在画面里
                # 每帧移 ~6 px。直接 std 会把它当抖动，永远判不出 found
                # （本机物理在环实测：48 帧可见，0 次 found）。真方块在
                # 匀速转时质心随时间近线性，去线性项后的残差才是抖动。
                a = np.stack([np.ones_like(ts), ts], axis=1)
                coef, _res, _rank, _sv = np.linalg.lstsq(a, uv, rcond=None)
                resid = uv - a @ coef
            else:
                resid = uv - uv.mean(axis=0)
            jitter = float(np.max(np.std(resid, axis=0)))

        if self._last_hit_t is None:
            stale = float("inf")
        else:
            stale = max(0.0, now - self._last_hit_t)

        found = (hits >= self.min_hits
                 and jitter <= self.max_jitter_px
                 and stale <= self.stale_s)
        return TrackState(found=found,
                          uv=self._last_uv if hits else None,
                          area=self._last_area if hits else 0.0,
                          hits=hits, window=len(self._samples),
                          jitter_px=jitter, stale_s=stale)

    # ---------- 内部 ----------
    def _pick(self, blobs):
        best = None
        for b in blobs or ():
            area = float(b.get("area", 0.0))
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
            if best is None or area > float(best["area"]):
                best = b
        return best

    def _trim(self):
        if len(self._samples) > self.n_window:
            del self._samples[:len(self._samples) - self.n_window]
