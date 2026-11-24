# -*- coding: utf-8 -*-
"""交班监督器：导航 → 停稳 → 手臂预设 → 可见性闸 → VLA。

=== 为什么交班要分相位，而不是一个 if ===
「到点了就把底盘交给 VLA」在真实系统里会踩三个坑：

  1. 导航说「到了」不等于车停了。厂商 navstate 的 arrived 常常早于
     速度归零；此时接管 = 在两个命令源之间来回抢，底盘抖。
  2. 交班瞬间如果手臂还在收纳姿态（STOW），arm_cam 朝上朝后，VLA
     第一帧就是分布外观测 —— 模型再准也抓不到。
  3. 目标在 arm_cam 里根本不可见时，VLA 的行为是「横移到平均位置然后
     停」。这是 partial observability，不是模型不行（v32b 6/48 的根因）。
     这种局面**must fail fast**，而不是让模型跑 900 步再判失败。

所以相位是：
    EXTERNAL_NAV  厂商导航驱动底盘，mux 静默（一根手指都不碰）
    CANCEL_NAV    已请求导航取消；等位姿窗口证明车真的停住
    SETTLE        mux 写 0，确认车没被残留速度带走
    ARM_PRESET    手臂摆到 FWD2（arm_cam 朝正前方），全程零速
    VISIBILITY    arm_cam 里目标色块像素数达标才放行
    VLA           底盘所有权交给 VLA；调用方随后跑 ManipulateSkill

=== 闸门为什么是「松几何 + 严可见」 ===
VLA 实测的泊车位收敛在 2.2 cm 跨距内（`.tmpm/measure_base_stop.py`，
6/6），起点误差 ±0.40 m / ±0.15 m / ±0.30 rad 都能吸收。所以车位
闸门可以放得很松（±0.15 m / ±12 cm / ±0.20 rad）；真正决定成败的是
「目标在 arm_cam 里看得见」这一条，它是硬闸。

=== 调用约定 ===
本类**不拥有线程**，由 L2 主循环按 rate_hz 调 tick()。相位内需要
执行的阻塞动作（取消导航 / 手臂预设 / 可见性检测）由构造时注入的
可调用对象完成。手臂预设会阻塞约 2–4 s —— 这是可接受的：L2 心跳
由 RemoteBackend 自己的线程发，不受阻塞影响；mux 线程此时在写 0。
"""
import time

import numpy as np


class Phase(object):
    EXTERNAL_NAV = "external_nav"
    CANCEL_NAV = "cancel_nav"
    SETTLE = "settle"
    ARM_PRESET = "arm_preset"
    VISIBILITY = "visibility"
    VLA = "vla"
    DONE = "done"
    ABORT = "abort"

    TERMINAL = (DONE, ABORT)


def wrap_fwd2(q_deg, q_seed):
    """把一组**角度**预设换成离 q_seed 最近的等价角（弧度）。

    === 为什么必须绕圈选最近分支 ===
    joint1/4/6 在 RM65 上是多圈的（±360°）。预设写 −90°，而当前是 265°
    时，直接下发要倒转 355°；选等价角 +270° 只走 5°。真机上 355° 的
    空转不但慢，还会把 arm_cam 扫过整圈，交班观察全废。

    实现复用 presets.wrap_nearest：逐关节选分支并限制在关节限位内。
    早期版本整组 +2πk 且不看限位，joint6=−323.5° 时会选到 −450°
    （超出 ±359.8°），伺服爬不到，预设以「残差 94.7°」失败。
    """
    from .presets import wrap_nearest
    return wrap_nearest(q_deg, q_seed)


def _wrap_pi(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class _PoseWindow(object):
    """滑动窗口里的位姿漂移。用于判断「车真的停了」。

    === 为什么用位姿而不是速度 ===
    真机上 L3 的 get_base_state()["vel"] 来自底盘反馈，可能有也可能没有
    （RealBackend 允许给 None）。位姿来自导航栈，是必有的；用位姿的
    **实际位移**做静止判据，比信任一个可能恒为零的速度字段可靠。
    """

    def __init__(self, window_s=0.6):
        self.win = float(window_s)
        self.buf = []

    def push(self, t, pose):
        self.buf.append((float(t), float(pose[0]), float(pose[1]),
                         float(pose[2])))
        while len(self.buf) > 2 and t - self.buf[0][0] > self.win:
            self.buf.pop(0)

    def reset(self):
        self.buf = []

    def drift(self):
        """返回 (dxy 米, dyaw 弧度, span 秒)。样本不足返回 None。"""
        if len(self.buf) < 2:
            return None
        t0, x0, y0, yaw0 = self.buf[0]
        t1, x1, y1, yaw1 = self.buf[-1]
        return (float(np.hypot(x1 - x0, y1 - y0)),
                abs(_wrap_pi(yaw1 - yaw0)),
                float(t1 - t0))


DEFAULTS = {
    "dock_pose": None,           # 交班车位 [x, y, yaw]，导航地图系
    "pose_tol": [0.15, 0.12, 0.20],
    "nav_timeout_s": 90.0,       # 等导航进交班区的上限
    "pose_timeout_s": 1.0,       # 多久没有新位姿算失效
    "cancel_timeout_s": 4.0,     # 请求取消后等停车的上限
    "still_xy_m": 0.010,
    "still_yaw_rad": 0.0175,     # 1°
    "still_window_s": 0.6,
    "still_ticks": 6,
    "still_vel": [0.02, 0.02, 0.05],   # 有速度反馈时的附加判据
    "settle_s": 0.5,
    "vla_blend_s": 0.3,
    "vis_min_px": 50.0,
    "visibility_gate": True,
}


class HandoffSupervisor(object):
    """相位机。构造一次，每个 episode 调 reset() 复用。"""

    def __init__(self, mux, pose_fn, cancel_fn=None, arm_preset_fn=None,
                 visibility_fn=None, vel_fn=None, pose_age_fn=None,
                 cfg=None, clock=None, verbose=True):
        self.mux = mux
        self.pose_fn = pose_fn
        self.cancel_fn = cancel_fn
        self.arm_preset_fn = arm_preset_fn
        self.visibility_fn = visibility_fn
        self.vel_fn = vel_fn
        self.pose_age_fn = pose_age_fn
        self.clock = clock or time.monotonic
        self.verbose = bool(verbose)

        c = dict(DEFAULTS)
        c.update(cfg or {})
        self.cfg = c

        self.phase = Phase.EXTERNAL_NAV
        self.t_phase = self.clock()
        self.t_enter = self.t_phase
        self.reason = ""
        self._window = _PoseWindow(c["still_window_s"])
        self._still_n = 0
        self._last_pose_t = None
        self._pose_warned = False
        self.metrics = {"ticks": 0, "pose_none": 0, "still_n": 0,
                        "drift_at_handoff": None, "vis_px": None,
                        "phase_marks": {}}

    # ---------- 复位 ----------
    def reset(self):
        self.phase = Phase.EXTERNAL_NAV
        self.t_phase = self.t_enter = self.clock()
        self.reason = ""
        self._window.reset()
        self._still_n = 0
        self._last_pose_t = None
        self.metrics = {"ticks": 0, "pose_none": 0, "still_n": 0,
                        "drift_at_handoff": None, "vis_px": None,
                        "phase_marks": {}}

    # ---------- 主循环 ----------
    def tick(self):
        """推进一个控制周期。返回 report（见 _report）。"""
        now = self.clock()
        self.metrics["ticks"] += 1

        if self.phase in Phase.TERMINAL:
            return self._report("done")

        # 位姿更新（CANCEL_NAV 之后仍要更新窗口，SETTLE 靠它判静止）
        pose = self._read_pose(now)
        if pose is not None:
            self._window.push(now, pose)

        action = "wait"
        if self.phase == Phase.EXTERNAL_NAV:
            action = self._step_external_nav(now, pose)
        elif self.phase == Phase.CANCEL_NAV:
            action = self._step_cancel_nav(now, pose)
        elif self.phase == Phase.SETTLE:
            action = self._step_settle(now)
        elif self.phase == Phase.ARM_PRESET:
            self._step_arm_preset(now)
        elif self.phase == Phase.VISIBILITY:
            action = self._step_visibility(now)

        if self.phase == Phase.VLA:
            action = "run_vla"
        elif self.phase == Phase.ABORT:
            action = "abort"
        elif self.phase == Phase.DONE:
            action = "done"
        return self._report(action)

    # ---------- 各相位 ----------
    def _step_external_nav(self, now, pose):
        if pose is None:
            self.metrics["pose_none"] += 1
            if now - self.t_enter > self.cfg["pose_timeout_s"]:
                return self._abort("导航位姿不可用（%.1f s 无位姿）"
                                   % (now - self.t_enter))
            return "wait"

        if self._pose_ok(pose):
            self._enter(Phase.CANCEL_NAV, now,
                        "进入交班区 pose=(%.2f, %.2f, %.1f°)"
                        % (pose[0], pose[1], np.degrees(pose[2])))
            if self.cancel_fn is not None:
                try:
                    self.cancel_fn()
                except Exception as e:
                    self.reason += "｜取消导航失败：%s" % e
            return "cancel_nav"

        if now - self.t_enter > self.cfg["nav_timeout_s"]:
            return self._abort("导航超时未进入交班区（%.0f s）"
                               % self.cfg["nav_timeout_s"])
        return "wait"

    def _step_cancel_nav(self, now, pose):
        if pose is not None and not self._pose_ok(pose):
            return self._abort("停车过程中车位漂出交班区")
        if self._still(now):
            # 到这里才第一次碰底盘：写 0 接管
            self.mux.set_owner("none")
            self._enter(Phase.SETTLE, now, "导航已停稳，mux 接管并写 0")
            return "wait"
        if now - self.t_phase > self.cfg["cancel_timeout_s"]:
            return self._abort("导航未在 %.1f s 内停车"
                               % self.cfg["cancel_timeout_s"])
        return "wait"

    def _step_settle(self, now):
        if not self._still(now):
            return self._abort("接管后底盘仍在动（导航源可能还在发速度）")
        if now - self.t_phase < self.cfg["settle_s"]:
            return "wait"
        self._enter(Phase.ARM_PRESET, now, "零速保持 %.2f s 通过" % self.cfg["settle_s"])
        return "wait"

    def _step_arm_preset(self, now):
        if self.arm_preset_fn is None:
            self._enter(Phase.VISIBILITY, now, "未配置手臂预设，跳过")
            return
        try:
            res = self.arm_preset_fn()
        except Exception as e:
            self._abort("手臂预设异常：%s" % e)
            return
        ok, msg = (res if isinstance(res, tuple) else (bool(res), ""))
        if not ok:
            self._abort("手臂预设失败：%s" % msg)
            return
        self._enter(Phase.VISIBILITY, now, "FWD2 到位：%s" % msg)

    def _step_visibility(self, now):
        if not self.cfg["visibility_gate"]:
            self.metrics["vis_px"] = None
            self._hand_to_vla(now, "可见性闸已关闭（仅调试）")
            return "run_vla"

        if self.visibility_fn is None:
            return self._abort("开了可见性闸但没提供 visibility_fn")

        try:
            px = self.visibility_fn()
        except Exception as e:
            return self._abort("可见性检测异常：%s" % e)
        self.metrics["vis_px"] = px

        if px is None:
            return self._abort("arm_cam 无图像，无法确认目标可见")
        if px < float(self.cfg["vis_min_px"]):
            # partial observability：此时放行，VLA 只会横移到平均位置
            return self._abort("arm_cam 目标仅 %.0f px < %.0f px"
                               % (px, self.cfg["vis_min_px"]))
        self._hand_to_vla(now, "目标可见 %.0f px" % px)
        return "run_vla"

    # ---------- 闸门 ----------
    def _read_pose(self, now):
        try:
            pose = self.pose_fn()
        except Exception:
            pose = None
        if self.pose_age_fn is not None:
            try:
                age = self.pose_age_fn()
            except Exception:
                age = None
            if age is not None and age > self.cfg["pose_timeout_s"]:
                return None
        if pose is None:
            return None
        p = np.asarray(pose, dtype=float).reshape(3)
        if not np.all(np.isfinite(p)):
            return None
        self._last_pose_t = now
        return (float(p[0]), float(p[1]), float(p[2]))

    def _pose_ok(self, pose):
        dock = self.cfg["dock_pose"]
        if dock is None:
            if not self._pose_warned:
                self._pose_warned = True
                self._log("未配置 dock_pose：车位闸门放行（仅调试）")
            return True
        tol = self.cfg["pose_tol"]
        dx = pose[0] - float(dock[0])
        dy = pose[1] - float(dock[1])
        dyaw = _wrap_pi(pose[2] - float(dock[2]))
        return (abs(dx) <= float(tol[0]) and abs(dy) <= float(tol[1])
                and abs(dyaw) <= float(tol[2]))

    def _still(self, now):
        """静止判据：位姿窗口漂移 + （可选的）速度，连续 still_ticks 次。

        位姿窗口必须真的跨过 still_window_s 才有效 —— 否则窗口里只有
        两个相隔 5 ms 的样本，任何噪声都「静止」。
        """
        ok = True
        d = self._window.drift()
        need_span = 0.5 * float(self.cfg["still_window_s"])
        if d is None or d[2] < need_span:
            ok = False
        else:
            ok = (d[0] <= float(self.cfg["still_xy_m"]) and
                  d[1] <= float(self.cfg["still_yaw_rad"]))
            if ok and self.metrics["drift_at_handoff"] is None:
                self.metrics["drift_at_handoff"] = [round(d[0], 4),
                                                   round(d[1], 4)]
        if ok and self.vel_fn is not None:
            try:
                v = np.abs(np.asarray(self.vel_fn(), float).reshape(3))
            except Exception:
                v = None
            if v is not None and not np.all(np.isfinite(v)):
                v = None
            if v is not None:
                lim = np.asarray(self.cfg.get("still_vel", [0.02, 0.02, 0.05]),
                                 float)
                ok = bool(np.all(v <= lim))
        self._still_n = self._still_n + 1 if ok else 0
        self.metrics["still_n"] = self._still_n
        return self._still_n >= int(self.cfg["still_ticks"])

    # ---------- 状态迁移 ----------
    def _enter(self, phase, now, reason):
        self.phase = phase
        self.t_phase = now
        self.reason = reason
        self.metrics["phase_marks"][phase] = round(now, 3)
        self._log("%s ← %s" % (phase, reason))

    def _hand_to_vla(self, now, reason):
        self.mux.set_owner("vla", blend_s=float(self.cfg["vla_blend_s"]))
        self._enter(Phase.VLA, now, reason)

    def _abort(self, reason):
        self.phase = Phase.ABORT
        self.reason = reason
        # 只有已经接管过才停；仍在 EXTERNAL 时不许碰底盘（导航还在跑）
        if getattr(self.mux, "owner", None) not in (None, "external"):
            self.mux.set_owner("none")
        self._log("ABORT ← %s" % reason)
        return "abort"

    def mark_done(self, ok=True, reason=""):
        self.phase = Phase.DONE if ok else Phase.ABORT
        self.reason = reason or self.reason
        return self._report("done" if ok else "abort")

    # ---------- 输出 ----------
    def _report(self, action):
        now = self.clock()
        return {
            "phase": self.phase,
            "action": action,
            "reason": self.reason,
            "t_phase_s": now - self.t_phase,
            "metrics": dict(self.metrics),
        }

    def _log(self, s):
        if self.verbose:
            print("[handoff] " + s, flush=True)


# ---------------- 现成的可调用对象 ----------------

def make_armcam_visibility_fn(backend, color, camera_name="arm_cam",
                              min_area=8):
    """返回 () -> 目标色块总像素数（None = 没图像）。

    用 HSV 连通域而不是检测网络：4 个已知颜色 + 受控光照，检测器是
    多一层误差来源。宁可让可见性闸对光照敏感一点，也不要它悄悄失准。
    """
    def fn():
        try:
            imgs = backend.get_images()
        except Exception:
            return None
        img = imgs.get(camera_name) if imgs else None
        if img is None:
            return None
        from ..perception.color_grounding import detect_blobs
        blobs = detect_blobs(img, color, min_area=min_area)
        return float(sum(b["area"] for b in blobs))
    return fn


def make_fwd2_preset(backend, q_deg=(0.0, -13.6, 134.9, 0.0, -16.2, -90.0),
                     grip=1.0, rate_hz=30.0, speed_deg_s=60.0,
                     tol_deg=4.0):
    """返回 () -> (ok, msg)：把手臂摆到 FWD2（arm_cam 朝机器狗正前方）。

    === FWD2 是「看得见桌子」的起始姿态 ===
    实测 [0, −13.6, 134.9, 0, −16.2, −90]°：相机在 XY 上与 front_cam
    同位（0.393, 0）、抬到 z=1.089、俯角 57.5°，24 布局 × 5 距离
    4.0/4 全可见。它也是 v36 训练数据的首帧，VLA 必须从这里起步。
    """
    def fn():
        from ..skills.base import Skill      # 懒导入，nav 包保持轻量
        st = backend.get_state()
        q_now = np.asarray(st["joint"], dtype=float).reshape(6)
        q_tgt = wrap_fwd2(q_deg, q_now)
        sk = Skill(backend, rate_hz=rate_hz, speed_deg_s=speed_deg_s,
                   timeout_s=30.0)
        sk.move_joints(q_now, q_tgt, st["gripper"], grip)
        # 经 L3 时指令有滞后，最后一帧到位后实测还差几度；hold 用目标角
        sk.hold_for(0.3, grip=grip, q=q_tgt)
        # === 闭环收敛，不信墙钟排程 ===
        # move_joints 是墙钟插值；仿真跑不到实时（渲染/慢机器）时插值
        # 先跑完而关节还没到，实测残差可达 36°。这里继续下发目标角并
        # 用实测角判收敛：到位立即返回；误差不再下降（约 1 s 无改善）
        # 也提前退出，避免真卡住时干等；硬上限 30 s（慢机/非实时够用）。
        t_end = time.monotonic() + 30.0
        best = float("inf")
        stalled = 0
        while True:
            st2 = backend.get_state()
            err = float(np.degrees(np.abs(
                np.asarray(st2["joint"], float).reshape(6) - q_tgt)).max())
            if err <= tol_deg:
                break
            if time.monotonic() >= t_end:
                break
            if err < best - 0.05:
                best = err
                stalled = 0
            else:
                stalled += 1
                if stalled >= 20:          # ~1 s（20 x 50 ms）无改善
                    break
            backend.actuate(q_tgt, grip)
            time.sleep(0.05)
        if err > tol_deg:
            return (False, "残差 %.1f° > %.1f°" % (err, tol_deg))
        return (True, "残差 %.1f°" % err)
    return fn
