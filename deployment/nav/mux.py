# -*- coding: utf-8 -*-
"""底盘速度仲裁器（velocity mux）：同一时刻只有一个源写底盘。

=== 为什么需要它 ===
底盘有三个写入来源，语义却完全不同：

  1. 厂商导航（SlamNaviApp / ROS2 nav）—— 直接调 P30 motion 的
     SendVelocity，**不经过我们的进程**；
  2. 我们自己的导航（P 控制 / 局部重规划）—— 走 HAL 的 move_base；
  3. VLA —— chunk 的第 10:13 列是 [vx, vy, wz]，30 Hz 逐帧下发。

三个源抢同一个执行器，就会出现「导航刚停、VLA 又倒车」这类怪事。
P30 的 SendVelocity 是**持续速度**（每帧刷新，断流即停），所以
「谁最后写谁生效」不是可接受的语义，必须有显式的所有权。

=== 四种所有权 ===
  EXTERNAL  厂商导航在写 —— mux **完全不写**（写 0 等于把导航停掉）
  NONE      mux 写 0：安全停 / 交班窗口
  NAV       mux 写导航意图（我们自己的导航节点）
  VLA       mux 写 VLA 意图

**缺省是 EXTERNAL**：宁可什么都不做，也不要在厂商导航跑的时候抢底盘。
这条默认值是防事故的，不要改。

=== 切换为什么要混合 ===
切换瞬间新源意图往往和上一刻输出差很多（VLA 倒车 −0.3 m/s → 导航
前进 +0.4 m/s）。直接写新值底盘会「顿一下」，机械臂在 0.3 m 外能被
甩出十几厘米。所以 set_owner() 记下切换时刻的输出速度，在 blend_s
（0.3–0.5 s）内用 smoothstep 过渡到新意图。

加速限幅仍在 L3（SafetyLimiter.filter_base_vel），mux **不做第二套
限幅** —— 两层限幅会互相抵消，而且失败了没法归因是哪一层截的。

=== 意图过期 ===
每个源写入时打时间戳；owner 的意图超过 intent_timeout_s 没刷新，
mux 自动把目标降为 0（同样走 blend）。这不是多余的保险：VLA 推理
卡死时 L2 循环停住、而 L2 的心跳线程仍在发，只靠 L3 的 200 ms
看门狗会在「指令还在路上」时误判。这一层让「模型不出动作」和
「进程死了」都收敛到零速。
"""
import threading
import time

import numpy as np


class Owner(object):
    """底盘所有权。字符串常量而不是 enum，方便打日志/进 msgpack。"""

    EXTERNAL = "external"
    NONE = "none"
    NAV = "nav"
    VLA = "vla"

    ALL = (EXTERNAL, NONE, NAV, VLA)


def _smoothstep(s):
    s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else float(s))
    return s * s * (3.0 - 2.0 * s)


class BaseVelocityMux(object):
    """唯一写底盘的线程 + 所有权仲裁。

    用法：
        mux = BaseVelocityMux(backend).start()      # 缺省 EXTERNAL，不写
        mux.set_nav(vx, vy, wz)                     # 导航意图，10–50 Hz
        mux.set_vla(vx, vy, wz)                     # VLA 意图，30 Hz
        mux.set_owner(Owner.NONE)                   # 交班窗口：写 0
        mux.set_owner(Owner.VLA, blend_s=0.3)       # 交给 VLA
        mux.close()                                 # 退出前必须（内含停车）

    线程安全：set_* / set_owner 可在任意线程调用，只有内部线程碰后端。
    """

    def __init__(self, backend, rate_hz=50.0, blend_s=0.4,
                 intent_timeout_s=0.30, verbose=False, clock=None):
        self.backend = backend
        self.rate_hz = max(1.0, float(rate_hz))
        self.blend_s = max(0.0, float(blend_s))
        self.intent_timeout_s = max(0.05, float(intent_timeout_s))
        self.verbose = bool(verbose)
        # 可注入的时钟：自检里用假时钟逐拍验证混合曲线，不必 sleep
        self._clock = clock or time.monotonic

        self._lock = threading.RLock()
        self._owner = Owner.EXTERNAL
        self._intent = {Owner.NAV: (np.zeros(3), 0.0),
                        Owner.VLA: (np.zeros(3), 0.0)}
        self._v_out = np.zeros(3)
        self._blend_from = np.zeros(3)
        self._blend_t0 = 0.0
        self._blend_s = self.blend_s
        self._stale = False

        self._stop = threading.Event()
        self._thread = None
        self._last_err = ""

        self.stats = {"writes": 0, "switches": 0, "blend_ticks": 0,
                      "stale_ticks": 0, "external_ticks": 0,
                      "write_errors": 0}

    # ---------- 生命周期 ----------
    @property
    def supports_base(self):
        return bool(getattr(self.backend, "supports_base", False))

    def start(self):
        if not self.supports_base:
            self._log("后端不支持底盘，mux 只做空转")
            return self
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="base-mux",
                                        daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.supports_base:
            try:
                self.backend.stop_base()
            except Exception as e:          # 退出路径不许抛
                self._last_err = "close 停车失败：%s" % e
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # ---------- 输入 ----------
    def set_nav(self, vx, vy, wz):
        """导航意图。ownership=NAV 时生效。"""
        return self._set_intent(Owner.NAV, vx, vy, wz)

    def set_vla(self, vx, vy, wz):
        """VLA 意图。ownership=VLA 时生效。"""
        return self._set_intent(Owner.VLA, vx, vy, wz)

    def _set_intent(self, who, vx, vy, wz):
        v = np.asarray([vx, vy, wz], dtype=float).reshape(3)
        if not np.all(np.isfinite(v)):
            # 坏值必须可见：吞掉只会表现为「底盘不动」，查不到根因
            self._last_err = "%s 意图含 NaN/inf：%s" % (who, v)
            raise ValueError(self._last_err)
        with self._lock:
            self._intent[who] = (v, self._clock())
        return True

    @property
    def owner(self):
        with self._lock:
            return self._owner

    @property
    def last_error(self):
        return self._last_err

    def set_owner(self, owner, blend_s=None):
        """切换所有权。切换必定以当前输出为起点做混合，不会跳变。"""
        if owner not in Owner.ALL:
            raise ValueError("未知所有权 %r" % (owner,))
        now = self._clock()
        with self._lock:
            if owner == self._owner:
                return False
            self._blend_from = self._v_out.copy()
            self._blend_t0 = now
            self._blend_s = (self.blend_s if blend_s is None
                             else max(0.0, float(blend_s)))
            self._owner = owner
            self.stats["switches"] += 1
            if self.verbose:
                self._log("所有权 → %s（混合 %.2f s，起点 %s）"
                          % (owner, self._blend_s,
                             np.round(self._blend_from, 3).tolist()))
        return True

    # ---------- 紧急停车 ----------
    def stop_now(self):
        """立即写 0，不走混合。急停/退出用；正常情况下用 set_owner(NONE)。"""
        with self._lock:
            self._owner = Owner.NONE
            self._v_out = np.zeros(3)
            self._blend_from = np.zeros(3)
            self._blend_t0 = 0.0
            self._blend_s = 0.0
        if self.supports_base:
            try:
                self.backend.stop_base()
            except Exception as e:
                self._last_err = "stop_now 失败：%s" % e
                return False
        return True

    # ---------- 输出（内部线程） ----------
    def _loop(self):
        period = 1.0 / self.rate_hz
        t_next = self._clock()
        while not self._stop.is_set():
            now = self._clock()
            try:
                self._tick(now)
            except Exception as e:          # 线程不能因单次异常而死
                self._last_err = "mux tick 异常：%s" % e
                self.stats["write_errors"] += 1
            t_next += period
            sleep = t_next - self._clock()
            if sleep > 0:
                time.sleep(sleep)
            else:
                t_next = self._clock()

    def _tick(self, now):
        with self._lock:
            owner = self._owner
            if owner == Owner.EXTERNAL:
                # 厂商导航在写：一根手指都不碰。输出视为 0，接管时从 0 起混。
                self._v_out = np.zeros(3)
                self._blend_from = np.zeros(3)
                self._stale = False
                self.stats["external_ticks"] += 1
                return None

            was_stale = self._stale

            if owner == Owner.NONE:
                target = np.zeros(3)
            else:
                v, t = self._intent[owner]
                if now - t > self.intent_timeout_s:
                    if not self._stale:
                        self._stale = True
                        if self.verbose:
                            self._log("%s 意图过期（%.0f ms），降为零速"
                                      % (owner, (now - t) * 1000.0))
                    target = np.zeros(3)
                    self.stats["stale_ticks"] += 1
                else:
                    self._stale = False
                    target = v.copy()

            # 过期状态翻转（正常 ↔ 过期）时重新起混：异常停车不能是一步
            # 0.4 m/s 的阶跃，否则机械臂先甩出去。正常刷新时**不**重起，
            # 否则每帧意图变化都把输出拖后一个 blend，变成一个滞后滤波器。
            if self._stale != was_stale:
                self._blend_from = self._v_out.copy()
                self._blend_t0 = now
                self._blend_s = self.blend_s

            if self._blend_s > 0.0 and (now - self._blend_t0) < self._blend_s:
                s = _smoothstep((now - self._blend_t0) / self._blend_s)
                v_out = self._blend_from + (target - self._blend_from) * s
                self.stats["blend_ticks"] += 1
            else:
                v_out = target

            self._v_out = v_out
            self.stats["writes"] += 1

        # 后端调用放在锁外：真机 move_base 可能阻塞几毫秒
        res = self.backend.move_base(float(v_out[0]), float(v_out[1]),
                                     float(v_out[2]))
        if isinstance(res, dict) and not res.get("ok"):
            self.stats["write_errors"] += 1
            self._last_err = res.get("err") or "move_base 失败"
        return v_out

    # ---------- 诊断 ----------
    def state(self):
        now = self._clock()
        with self._lock:
            v, t = self._intent.get(self._owner,
                                    (np.zeros(3), 0.0))
            return {
                "owner": self._owner,
                "v_out": self._v_out.tolist(),
                "intent": v.tolist(),
                "intent_age_ms": (None if self._owner in (Owner.NONE,
                                                          Owner.EXTERNAL)
                                  else (now - t) * 1000.0),
                "blend_remaining_ms": max(
                    0.0, (self._blend_t0 + self._blend_s - now) * 1000.0),
                "stale": self._stale,
                "stats": dict(self.stats),
                "last_error": self._last_err,
            }

    def _log(self, s):
        print("[mux] " + s, flush=True)
