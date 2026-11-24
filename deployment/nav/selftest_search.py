# -*- coding: utf-8 -*-
"""SEARCH 状态机自检：合成检测 + 假底盘，不需要 MuJoCo。

用法：python -m deployment.nav.selftest_search
"""
import inspect

import numpy as np

from . import handoff
from .presets import (FWD2_Q_DEG, JOINT_LIMITS_DEG, SEARCH_Q_DEG,
                     wrap_nearest)
from .search import Phase, SearchBehavior


class FakeRig(object):
    """假底盘：按指令偏航累积（76% 效率），可选在指定帧出现色块。"""

    def __init__(self, appear_at=None, dt=0.08, eff=0.76):
        self.yaw = 0.0
        self.appear_at = appear_at
        self.dt = dt
        self.eff = eff
        self.preset_calls = 0
        self.frames = 0
        self.imgs = {}

    def preset(self):
        self.preset_calls += 1
        return True, "ok"

    def get_images(self):
        return {"arm_cam": np.zeros((8, 8, 3), np.uint8)}

    def detect(self, img, color, min_area):
        self.frames += 1
        if self.appear_at is None or self.frames < self.appear_at:
            return []
        return [{"uv": (300.0, 200.0), "area": 200, "bbox": (299, 199, 301, 201)}]

    def apply(self, st):
        self.yaw += st.wz * self.dt * self.eff


def _make(rig, **kw):
    return SearchBehavior(rig.preset, rig.get_images, "red", "arm_cam",
                          rig.detect, lambda: rig.yaw, **kw)


def run(rig, max_ticks=2000, **kw):
    s = _make(rig, **kw)
    trace = []
    for _ in range(max_ticks):
        st = s.tick()
        trace.append(st)
        rig.apply(st)
        if st.done:
            break
    return s, trace


def t_found():
    rig = FakeRig(appear_at=40)
    s, tr = run(rig)
    assert s.phase == Phase.FOUND, s.phase
    assert rig.preset_calls == 1, rig.preset_calls
    assert 30 <= s.frames <= 45, s.frames
    assert 1.8 < s.swept < 2.4, s.swept
    assert tr[-1].wz == 0.0, tr[-1].wz
    return "第 40 帧出现 → FOUND（转 %.2f rad / %d 帧），停止指令归零" % (
        s.swept, s.frames)


def t_not_found():
    rig = FakeRig(appear_at=None)
    s, tr = run(rig)
    assert s.phase == Phase.FAILED, s.phase
    assert s.spins == 2, s.spins
    assert s.frames > 180, s.frames
    assert tr[-1].wz == 0.0
    return "始终没有目标 → 两圈后 FAILED（%d 帧）" % s.frames


def t_preset_fail():
    rig = FakeRig()
    rig.preset = lambda: (False, "残差 9°")
    s, _ = run(rig)
    assert s.phase == Phase.FAILED and "抬臂失败" in s.reason, s.reason
    assert s.frames == 0, s.frames
    return "抬臂失败立即终止，不转圈"


def t_no_image():
    rig = FakeRig()
    rig.get_images = lambda: {}
    s, _ = run(rig)
    assert s.phase == Phase.FAILED and "无图像" in s.reason, s.reason
    return "取不到搜索相机图像 → FAILED（不静默空转）"


def t_fwd2_consistency():
    default = inspect.signature(handoff.make_fwd2_preset)
    got = default.parameters["q_deg"].default
    assert tuple(float(x) for x in got) == FWD2_Q_DEG, (got, FWD2_Q_DEG)
    return "presets.FWD2 与 handoff.make_fwd2_preset 缺省值一致"


def t_wrap():
    # seed 必须物理可达（joint2 限位 ±129.9°，不能拿 265° 当种子）
    seed = np.radians([0, 52, 37, 0, -16, 265])
    q = wrap_nearest(SEARCH_Q_DEG, seed)
    # 1) 与预设同余
    k = (q - np.radians(SEARCH_Q_DEG)) / (2.0 * np.pi)
    assert np.allclose(k, np.round(k)), k
    # 2) 逐关节都落在限位内（整组平移会把 joint2 推出 ±129.9°）
    lim = np.radians(np.asarray(JOINT_LIMITS_DEG, float))
    assert np.all(q >= lim[:, 0] - 1e-9) and np.all(q <= lim[:, 1] + 1e-9), q
    # 3) −90° 在 265° 附近选 +270°（只走 5°），不是 −90°（倒转 355°）
    assert abs(q[5] - np.radians(270)) < 1e-9, q
    # 4) joint2 没有等价分支在限位内，保持预设角本身
    assert abs(q[1] - np.radians(53.9)) < 1e-9, q
    return "预设绕圈：joint6 −90°→+270°（少走 350°），joint2 受限不动"


def t_real_detect():
    """接真 detect_blobs：640×480 里画一个 12×12 红块，应被跟踪到。"""
    import cv2
    from ..perception.color_grounding import detect_blobs
    from ..perception.tracking import BlobTracker, scale_min_area

    img = np.zeros((480, 640, 3), np.uint8)
    cv2.rectangle(img, (300, 200), (312, 212), (217, 38, 31), -1)
    min_area = scale_min_area(25.0, 640, 480)
    tr = BlobTracker(n_window=6, min_hits=4)
    st = None
    for _ in range(6):
        st = tr.update(detect_blobs(img, "red", min_area=min_area))
    assert st.found and st.uv == (306.0, 206.0), st
    return "真实 HSV 检测 144 px 红块 → found（阈值 100 px @640×480）"


def main():
    cases = [t_found(), t_not_found(), t_preset_fail(), t_no_image(),
             t_fwd2_consistency(), t_wrap(), t_real_detect()]
    for i, msg in enumerate(cases, 1):
        print("  %d. %s" % (i, msg))
    print("search selftest: %d/%d OK" % (len(cases), len(cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
