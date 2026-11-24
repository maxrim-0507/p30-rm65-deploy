# -*- coding: utf-8 -*-
"""SEARCH：原地扫描，直到目标在搜索相机里稳定可见。

=== 这个类只做决策，不碰硬件 ===
tick() 返回本帧该下的底盘速度，由调用方（L2 / 仿真 harness）执行。
机械臂预设、取图、检测、偏航读取都是注入的可调用对象，所以：
  - 仿真里可以用 MuJoCo 后端，真机换 RealBackend 不用改这里；
  - 单测可以喂合成检测结果，不需要 MuJoCo/cv2。

=== 为什么用实测偏航累积 ===
指令积分会在打滑/被碰时高估扫描角度：车其实没转够一圈，逻辑却认为
扫完了。这里每帧从 get_yaw() 读实测值，累加 |Δyaw|。

=== 失败策略 ===
max_spins 次（缺省 2）单圈扫描都没找到 → FAILED，调用方回安全姿态并
上报。绝不「凭记忆盲抓」。
"""
import time

import numpy as np

from ..perception.tracking import BlobTracker, scale_min_area
from .presets import SEARCH_MAX_SWEEP, SEARCH_WZ


class Phase(object):
    ARM = "arm"          # 抬臂到搜索姿态（预设函数阻塞完成）
    SPIN = "spin"        # 原地转，逐帧检测
    FOUND = "found"      # 已稳定可见，速度归零
    FAILED = "failed"

    TERMINAL = (FOUND, FAILED)


class SearchStatus(object):
    __slots__ = ("phase", "vx", "vy", "wz", "swept", "spins", "frames",
                 "track", "reason")

    def __init__(self, phase, vx=0.0, vy=0.0, wz=0.0, swept=0.0, spins=0,
                 frames=0, track=None, reason=""):
        self.phase = phase
        self.vx, self.vy, self.wz = float(vx), float(vy), float(wz)
        self.swept = float(swept)
        self.spins = int(spins)
        self.frames = int(frames)
        self.track = track
        self.reason = reason

    @property
    def done(self):
        return self.phase in Phase.TERMINAL


def _wrap_pi(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class SearchBehavior(object):

    def __init__(self, preset_fn, get_images, color, camera,
                 detect_fn, get_yaw, wz=SEARCH_WZ,
                 max_sweep=SEARCH_MAX_SWEEP, max_spins=2,
                 tracker=None, min_area=None, clock=None,
                 accept_fn=None):
        """
        preset_fn   () -> (ok, msg)，阻塞式抬臂
        get_images  () -> dict{相机名: HxWx3 uint8}
        detect_fn   (img, color, min_area) -> detect_blobs 结构
        get_yaw     () -> 实测偏航（rad）
        accept_fn   (blobs, img) -> list，整帧候选准入/剔除；None 表示全收。
                    用整帧而非单块，是为了能结合同帧其它颜色块剔除色溢伪点。

        === accept_fn 为什么必须存在 ===
        画面边缘的色块剪影被裁掉一角，质心偏移，反投影常落到物体带外。
        不过滤的话，hold 会每帧续期把车「钉」在原地，最后以「检测到
        目标但反投影越出台面」失败（2026-11-21 实测 seed 3 / yellow）。
        过滤后车继续转，等目标转到视场中间再接受。
        """
        self.preset_fn = preset_fn
        self.get_images = get_images
        self.color = color
        self.camera = camera
        self.detect_fn = detect_fn
        self.get_yaw = get_yaw
        self.wz = float(wz)
        self.max_sweep = float(max_sweep)
        self.max_spins = int(max_spins)
        self.tracker = tracker or BlobTracker()
        self.min_area = min_area
        self.accept_fn = accept_fn
        self._clock = clock or time.monotonic
        self.reset()

    def reset(self):
        self.phase = Phase.ARM
        self.swept = 0.0
        self.spins = 0
        self.frames = 0
        self.reason = ""
        self._yaw0 = None
        self._prev_yaw = None
        self._hold_until = None
        self.tracker.reset()

    # ---------- 主入口 ----------
    def tick(self):
        if self.phase in Phase.TERMINAL:
            return SearchStatus(self.phase, swept=self.swept, spins=self.spins,
                                frames=self.frames, reason=self.reason)

        if self.phase == Phase.ARM:
            ok, msg = self.preset_fn()
            if not ok:
                self.phase = Phase.FAILED
                self.reason = "抬臂失败：%s" % msg
                return SearchStatus(self.phase, reason=self.reason)
            self._yaw0 = self._prev_yaw = self.get_yaw()
            self.phase = Phase.SPIN

        # ---- SPIN ----
        self.frames += 1
        imgs = self.get_images() or {}
        img = imgs.get(self.camera)
        if img is None:
            self.phase = Phase.FAILED
            self.reason = "搜索相机 %s 无图像" % self.camera
            return SearchStatus(self.phase, swept=self.swept,
                                spins=self.spins, frames=self.frames,
                                reason=self.reason)

        if self.min_area is None:
            h, w = img.shape[:2]
            self.min_area = scale_min_area(25.0, w, h)
        blobs = self.detect_fn(img, self.color, self.min_area)
        # 先准入再喂 tracker：被拒的候选不能触发 hold，也不能算 found
        if self.accept_fn is not None:
            blobs = self.accept_fn(list(blobs), img)
        track = self.tracker.update(blobs, t=self._clock())

        # === 看到候选立刻停转，静止后再判 found ===
        # 转动中「最新像素」和「最新相机外参」不是同一时刻：L2 分开取图
        # 与取外参，实测 12.5 Hz 下每帧转 ~3°，台面边缘的目标反投影会
        # 越出台面（seed 11）。停下 1.5 s，等 tracker 在静止画面上攒够
        # hits；没有候选就继续扫。期间不累计 swept，不会误判扫完一圈。
        now = self._clock()
        if blobs:
            self._hold_until = now + 1.5
        hold = self._hold_until is not None and now < self._hold_until

        if track.found:
            self.phase = Phase.FOUND
            return SearchStatus(self.phase, swept=self.swept,
                                spins=self.spins, frames=self.frames,
                                track=track, reason="目标稳定可见")

        yaw = self.get_yaw()
        self.swept += abs(_wrap_pi(yaw - self._prev_yaw))
        self._prev_yaw = yaw
        if self.swept >= self.max_sweep:
            self.spins += 1
            if self.spins >= self.max_spins:
                self.phase = Phase.FAILED
                self.reason = "扫描 %d 圈未找到 %s" % (self.spins, self.color)
                return SearchStatus(self.phase, swept=self.swept,
                                    spins=self.spins, frames=self.frames,
                                    track=track, reason=self.reason)
            self.swept = 0.0
            self.tracker.reset()

        return SearchStatus(self.phase, wz=0.0 if hold else self.wz,
                            swept=self.swept, spins=self.spins,
                            frames=self.frames, track=track)
