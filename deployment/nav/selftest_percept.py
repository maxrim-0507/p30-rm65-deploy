# -*- coding: utf-8 -*-
"""PERCEPT NAV 自检：假底盘 + 假相机，不需要 MuJoCo / cv2。

用法：
    python -m deployment.nav.selftest_percept

覆盖：
  1. 正常路     远起点 → 搜索 → 泊车收敛 → TUCK，目标 xy 反投影正确
  2. 盒子前缘   接近途中视觉丢失，但已在交班盒内 → 仍成功
  3. 盒子外丢失 一远处丢目标 → 停车且 FAILED，不盲开
  4. 搜索失败   始终看不见 → FAILED
  5. 可见性闸   FWD2 下像素不足 → FAILED（不放给 VLA）
  6. 无位姿     底盘位姿不可用 → FAILED（不猜）
  7. 零真值     全程不碰 get_objects

相机用针孔模型真正做「投影 → 反投影」往返；可见与否由场景旋钮控制，
用真值 xy 作为反投影正确答案。这只在自检里成立 —— 生产代码没有真值通路。
"""
import numpy as np

from ..perception.color_grounding import CameraModel
from ..spec import CUBE_HALF, TABLE_TOP_Z
from .percept_pipeline import NavPhase, PerceptNav
from .visual_dock import dock_pose

TARGET = (0.46, 0.16)

# 顶视假相机：真正做投影/反投影往返，但不模拟遮挡
_CAM = CameraModel(320, 240, 60.0, pos=np.array([0.46, 0.0, 1.5]),
                   rot=np.eye(3))


class FakeClock(object):
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += float(dt)


class FakeBase(object):
    """点机器人：机体系速度积分，效率按底盘实测 ~76%。"""

    def __init__(self, x=-1.5, y=0.0, yaw=0.0, eff=0.76):
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.eff = float(eff)
        self.last = (0.0, 0.0, 0.0)

    def apply(self, dt):
        vx, vy, wz = self.last
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.x += (c * vx - s * vy) * dt * self.eff
        self.y += (s * vx + c * vy) * dt * self.eff
        self.yaw += wz * dt * self.eff


class FakeBackend(object):
    """假后端：底盘、相机、检测都按场景旋钮工作。"""

    supports_base = True
    supports_pose_command = False

    def __init__(self, base, visible=None, vis_px=120.0, pose_ok=True,
                 target=TARGET, target_color="red", bleed=None):
        self.base = base
        self.visible = visible or (lambda n, b: True)
        self.vis_px = vis_px
        self.pose_ok = pose_ok
        self.target = tuple(target)
        self.target_color = target_color
        # bleed: (other_color, (x,y), false_area, other_area) —— 目标色伪点
        # 落在另一颜色方块上，用于验证色溢剔除
        self.bleed = bleed
        self.n = 0
        self.calls = []
        self.stop_calls = 0
        self.preset_calls = []
        self.truth_calls = 0

    # ---- 底盘 ----
    def move_base(self, vx=0.0, vy=0.0, wz=0.0):
        self.last_cmd = (float(vx), float(vy), float(wz))
        self.base.last = self.last_cmd
        self.calls.append(self.last_cmd)
        return {"ok": True, "err": None}

    def stop_base(self):
        self.stop_calls += 1
        self.base.last = (0.0, 0.0, 0.0)
        self.last_cmd = (0.0, 0.0, 0.0)
        return {"ok": True, "err": None}

    def get_base_state(self):
        if not self.pose_ok:
            return {"pos": None, "yaw": None, "vel": (0.0, 0.0, 0.0)}
        return {"pos": (self.base.x, self.base.y), "yaw": self.base.yaw,
                "vel": (0.0, 0.0, 0.0)}

    # ---- 观测 ----
    def get_images(self):
        return {"arm_cam": np.zeros((240, 320, 3), np.uint8)}

    def camera_model(self, name):
        return _CAM

    def get_objects(self):
        self.truth_calls += 1
        raise AssertionError("PerceptNav 不许读真值")

    # ---- 检测 ----
    def _blob_at(self, xy, area):
        # 合成像素取方块中心高度 —— 与 PerceptNav._project 的求交平面一致，
        # 否则真值像素会带着系统偏差进闭环
        u, v, z = _CAM.project(np.array([xy[0], xy[1],
                                         TABLE_TOP_Z + CUBE_HALF]))
        return {"uv": (float(u), float(v)), "area": int(area),
                "bbox": (int(u) - 4, int(v) - 4, int(u) + 4, int(v) + 4)}

    def detect(self, img, color, min_area):
        # 只有目标色的调用推进帧计数；其它颜色仅服务于色溢剔除
        if color != self.target_color:
            if self.bleed is not None and color == self.bleed[0]:
                return [self._blob_at(self.bleed[1], self.bleed[3])]
            return []
        self.n += 1
        if not self.visible(self.n, self.base):
            return []
        out = [self._blob_at(self.target, 240)]
        if self.bleed is not None:
            # 伪点面积故意更大，复现「取最大块选错」
            out.append(self._blob_at(self.bleed[1], self.bleed[2]))
        return out

    def visibility(self):
        return self.vis_px


def _make(base, cfg_extra=None, color="red", **kw):
    clock = FakeClock()

    def sleep(dt):
        clock.advance(dt)
        base.apply(dt)

    cfg = {"frame_dt": 0.05, "realtime_sleep": True,
           "max_search_s": 30.0, "max_approach_s": 30.0}
    cfg.update(cfg_extra or {})
    preset = []
    be = FakeBackend(base, **kw)
    nav = PerceptNav(
        be, color, cfg=cfg,
        detect_fn=be.detect, clock=clock, sleep_fn=sleep,
        search_preset_fn=lambda: (preset.append("search"), (True, "ok"))[1],
        tuck_preset_fn=lambda: (preset.append("tuck"), (True, "0.4°"))[1],
        visibility_fn=be.visibility, verbose=False)
    return nav, be, preset


def t_happy():
    base = FakeBase(x=-1.5, y=0.0, yaw=0.0)
    nav, be, preset = _make(base)
    res = nav.run()
    assert res.ok and res.phase == NavPhase.DONE, res
    assert preset == ["search", "tuck"], preset      # TUCK 只在泊车后一次
    assert abs(res.target_xy[0] - TARGET[0]) < 1e-6
    assert abs(res.target_xy[1] - TARGET[1]) < 1e-6
    assert res.err_xy <= 0.12 and abs(res.err_yaw) <= 0.15, res
    assert be.last_cmd == (0.0, 0.0, 0.0), be.last_cmd
    return ("远起点 (-1.5, 0) → 目标 (%.2f, %.2f)，泊车残差 %.3f m，"
            "搜索 %d 帧 / 泊车 %d 帧"
            % (res.target_xy[0], res.target_xy[1], res.err_xy,
               res.frames["search"], res.frames["approach"]))


def t_lost_inside():
    # 目标偏侧 → 泊车位落到前缘（> −0.21）。在离车位 0.30 m 处故意失明
    # （模拟 SEARCH_Q 近处看不全台面）。这里把盲走预算关掉（blind_goal_s=0），
    # 专门验证「丢目标 → 降级交班盒」这条退路本身。
    base = FakeBase(x=-1.0, y=0.4, yaw=0.0)
    tgt = (0.52, 0.20)
    goal = dock_pose(tgt, (base.x, base.y))
    nav, be, preset = _make(
        base, cfg_extra={"blind_goal_s": 0.0}, target=tgt,
        visible=lambda n, b: float(np.hypot(b.x - goal[0],
                                            b.y - goal[1])) > 0.30)
    res = nav.run()
    assert res.ok, res
    assert res.goal[0] > -0.21, res.goal          # 泊车位确实在前缘
    assert 0.12 < res.err_xy <= 0.25, (res.err_xy, res.goal)
    assert be.last_cmd == (0.0, 0.0, 0.0)
    assert any("视觉近处丢失" in s for _p, s in res.phases), res.phases
    return ("前缘泊位 %.3f：离车位 0.30 m 处失明，残差 %.3f m 在降级盒内 → 成功"
            % (res.goal[0], res.err_xy))


def t_blind_finish():
    # 同一场景，但保留盲走预算：离车位 0.30 m 失明后按锁存车位继续开，
    # 应该进近交班盒（erro ≤ 0.12），而不是停在降级盒。
    base = FakeBase(x=-1.0, y=0.4, yaw=0.0)
    tgt = (0.52, 0.20)
    goal = dock_pose(tgt, (base.x, base.y))
    nav, be, preset = _make(
        base, target=tgt,
        visible=lambda n, b: float(np.hypot(b.x - goal[0],
                                            b.y - goal[1])) > 0.30)
    res = nav.run()
    assert res.ok, res
    assert res.err_xy <= 0.12, (res.err_xy, res.goal)
    assert be.last_cmd == (0.0, 0.0, 0.0), be.last_cmd
    return ("离车位 0.30 m 失明 → 盲走收敛，残差 %.3f m（进近盒）"
            % res.err_xy)


def t_lost_outside():
    base = FakeBase(x=-1.5, y=0.0)
    # 搜索够 6 帧就永远丢 —— 泊车第一帧后即失明，此时离盒子很远
    nav, be, preset = _make(base, visible=lambda n, b: n <= 6)
    res = nav.run()
    assert not res.ok and res.phase == NavPhase.FAILED, res
    assert "丢失" in res.reason, res.reason
    assert be.last_cmd == (0.0, 0.0, 0.0), be.last_cmd
    assert "tuck" not in preset
    return "远处失明 → 停车且 FAILED（%s）" % res.reason


def t_search_fail():
    base = FakeBase()
    nav, be, preset = _make(base, visible=lambda n, b: False)
    res = nav.run()
    assert not res.ok and "搜索" in res.reason, res.reason
    assert preset == ["search"], preset
    assert res.frames["search"] > 100, res.frames
    return "始终看不见 → FAILED（%s）" % res.reason


def t_vis_gate():
    base = FakeBase(x=-1.5, y=0.0)
    nav, be, preset = _make(base, vis_px=10.0)
    res = nav.run()
    assert not res.ok and "px" in res.reason, res.reason
    assert preset == ["search", "tuck"], preset
    return "FWD2 下仅 10 px → FAILED，不放给 VLA（%s）" % res.reason


def t_no_pose():
    base = FakeBase()
    nav, be, preset = _make(base, pose_ok=False)
    res = nav.run()
    assert not res.ok and "位姿" in res.reason, res.reason
    return "底盘位姿不可用 → FAILED（%s）" % res.reason


def t_truth_free():
    base = FakeBase(x=-1.5, y=0.0)
    nav, be, preset = _make(base)
    res = nav.run()
    assert res.ok
    assert be.truth_calls == 0, be.truth_calls
    assert res.truth_calls == 0 and set(res.sources) == {"hsv"}, res.sources
    return "定位来源只有 hsv，get_objects 调用 0 次"


def t_bleed_reject():
    # 真目标黄色在 (0.46,0.16)；红色方块在 (0.44,0.02)，其高光读成**更大**的
    # 黄色伪点；同处红色面积更大（主导度不足）。不剔除时会锁到伪点。
    # 验证按「与其它颜色块重合」剔除后仍锁到真目标。
    base = FakeBase(x=-1.5, y=0.0, yaw=0.0)
    tgt = (0.46, 0.16)
    bleed_xy = (0.44, 0.02)
    nav, be, preset = _make(
        base, color="yellow", target=tgt, target_color="yellow",
        bleed=("red", bleed_xy, 320, 340))
    res = nav.run()
    assert res.ok, res
    ex = float(np.hypot(res.target_xy[0] - tgt[0],
                        res.target_xy[1] - tgt[1]))
    ef = float(np.hypot(res.target_xy[0] - bleed_xy[0],
                        res.target_xy[1] - bleed_xy[1]))
    assert ex < 1e-6 and ef > 0.05, (res.target_xy, ex, ef)
    assert be.truth_calls == 0
    return ("黄色伪点压在红块上（更大）被剔除，仍锁真目标，偏 %.2f mm"
            % (1e3 * ex))


def main():
    cases = [t_happy(), t_lost_inside(), t_blind_finish(), t_lost_outside(),
             t_search_fail(), t_vis_gate(), t_no_pose(), t_truth_free(),
             t_bleed_reject()]
    for i, msg in enumerate(cases, 1):
        print("  %d. %s" % (i, msg))
    print("percept selftest: %d/%d OK" % (len(cases), len(cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
