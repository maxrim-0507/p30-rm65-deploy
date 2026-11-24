# -*- coding: utf-8 -*-
"""L3 控制服务：唯一真正驱动硬件的进程。

=== 主循环结构 ===
  每 (1/rate_hz) 秒：
    1) 取最新指令缓冲（REQ 线程写入，这里只读）
    2) 心跳闸：超时就保持当前位置
    3) 安全层滤波（限位/限速/加速度/工作空间）
    4) 下发 + 推进物理/时间
    5) 按 state_rate / image_rate 分别广播

=== 为什么指令不阻塞在 REQ handler 里 ===
REQ handler 若直接执行动作，一次响应要等一个控制周期，30 Hz 上层会
被拖到 100 ms 抖动。这里 handler 只写缓冲并回 ok，真正的执行在
主循环里做 —— 于是 REP 的语义是「已入队」，与真机的伺服语义一致。

=== 为什么图像单独一路 PORT ===
240×320×3 两张图 ≈ 460 KB，msgpack 后同样量级。和关节状态挤在同一条
PUB 上会让 30 Hz 的状态帧排队等大包。分开之后，图像订阅端可以独立
丢帧，不影响控制。
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

from ..bus import Heartbeat, Publisher, Replier, Subscriber
from ..bus.config import CONTROL, ENDPOINTS, TRANSPORT
from ..hal import make_backend
from .safety import SafetyLimiter


class ControlServer:

    def __init__(self, backend=None, verbose=True):
        self.verbose = verbose
        self.backend = backend or make_backend()

        self._lock = threading.Lock()
        self._target = None          # (joint, gripper, enqueue_time)
        # 位姿目标：(pos, rot6d, gripper, enqueue_time)。与 _target 互斥，
        # 最后一条指令生效 —— 同一时刻只允许一种控制语义，免得关节与
        # 笛卡尔目标在主循环里打架。
        self._pose_target = None
        # 底盘：(vx, vy, wz), enqueue_time。None = 已请求停。
        # === 与关节目标分开存 ===
        # 关节是「到位即停」的位置伺服，底盘是「持续有效」的速度指令
        # （P30 SendVelocity 语义）。合并成一个目标会导致每次发臂指令
        # 都顺带把底盘速度清零/重发，上层没法独立控制。
        self._base_target = None
        self._pending_reset = None   # 待主循环处理的重置请求（含 objects）
        self._reset_seq = 0
        self._layout_seq = 0
        self._stop = False
        # === 手动步进模式（inline 精确同步）===
        # True 时主循环完全让出物理，由 LocalBackend.apply_step() 精确
        # 推进固定物理步数 —— 使部署与 rollout（每动作 16 步同步）等价。
        self._manual = False
        # 手动模式下本线程仍要负责渲染与广播（EGL renderer 绑定在创建它的
        # 线程上，L2 线程直接 get_images() 会段错误），所以用
        # _phys_lock 串行化「apply_step 推进物理」与「本线程取观测/渲染」。
        self._phys_lock = threading.Lock()
        self._manual_seen = 0        # 进入手动模式的应答计数（enter_sync 握手机）
        self._render_req = False     # L2 请求一帧新鲜图像（渲染必须在属主线程）
        self._render_seq = 0
        self._last_err = ""

        self.pub_state = Publisher(TRANSPORT["host"],
                                   TRANSPORT["state_port"])
        self.pub_image = Publisher(TRANSPORT["host"],
                                   TRANSPORT["image_port"])
        self.sub_hb = Subscriber(TRANSPORT["host"], TRANSPORT["hb_port"])
        self.rep = Replier(TRANSPORT["host"], TRANSPORT["cmd_port"])
        self.hb = Heartbeat(Publisher(TRANSPORT["host"],
                                      TRANSPORT["hb_port"] + 100))

        self.safety = None           # connect 之后才有关节限位

        # 最近一次广播的内容（供同进程的 inline 代理读取）。
        # === 为什么需要 ===
        # inline 模式里 L2 和 L3 在同一进程。若 L2 直接调 backend.get_state()
        # 读的是同一个 MjData，而 L3 线程正在 mj_step —— 数据竞争，且
        # L2 调 backend.actuate() 会覆盖 L3 写入的 ctrl，两边互相打断
        # （实测：L2 发的目标被 L3 的看门狗 hold 立刻抹掉，机械臂纹丝不动，
        # 表现为「工具轴偏 23.6°」）。
        # 让 L2 只读快照、只写队列，与跨进程模式语义完全一致。
        self._snap_lock = threading.Lock()
        self._snap_state = None
        self._snap_images = {}

    # ---------- 生命周期 ----------
    def connect(self):
        self.backend.connect()
        st = self.backend.get_state()
        lo, hi = self._joint_ranges()
        self.safety = SafetyLimiter(
            lo, hi,
            max_speed_deg_s=CONTROL["max_joint_speed_deg_s"],
            max_acc_deg_s2=CONTROL["max_joint_acc_deg_s2"],
            workspace=CONTROL.get("workspace"),
            watchdog_ms=CONTROL["watchdog_ms"],
            max_eef_speed_m_s=CONTROL.get("max_eef_speed_m_s", 0.25),
            base_vel_max=CONTROL.get("max_base_speed"),
            base_acc_max=CONTROL.get("max_base_acc"))
        self.safety._q_prev = st["joint"].copy()
        self.safety.note_heartbeat()      # 允许上层在 200ms 内接上
        return self

    def _joint_ranges(self):
        m = getattr(self.backend, "m", None)
        if m is not None:
            lo = np.array([m.jnt_range[m.joint("arm_joint%d" % i).id, 0]
                           for i in range(1, 7)])
            hi = np.array([m.jnt_range[m.joint("arm_joint%d" % i).id, 1]
                           for i in range(1, 7)])
            return lo, hi
        # 真机：RM65 官方行程（度 → 弧度）。与 RM_API2 手册一致。
        deg = np.array([[-360, 360], [-130, 130], [-135, 135],
                        [-360, 360], [-128, 128], [-360, 360]], dtype=float)
        return np.radians(deg[:, 0]), np.radians(deg[:, 1])

    def close(self):
        self._stop = True
        try:
            if getattr(self.backend, "supports_base", False):
                self.backend.stop_base()   # 退出前先给底盘一条 0 速度
            self.backend.close()
        finally:
            for s in (self.pub_state, self.pub_image, self.sub_hb, self.rep):
                try:
                    s.close()
                except Exception:
                    pass

    # ---------- 指令入口（REQ handler，另一个线程）----------
    def handle(self, msg):
        op = msg.get("op")
        if op == "ping":
            return {"ok": True, "backend": self.backend.name,
                    "supports_base": bool(
                        getattr(self.backend, "supports_base", False)),
                    "supports_pose": bool(
                        getattr(self.backend, "supports_pose_command", False)),
                    "t": time.monotonic()}
        if op == "actuate":
            q = np.asarray(msg["joint"], dtype=float).reshape(6)
            g = float(msg["gripper"])
            with self._lock:
                self._target = (q, g, time.monotonic())
                self._pose_target = None
            return {"ok": True}
        if op == "actuate_pose":
            # 只入队，由主循环下发 —— 与 actuate 同理，避免 handler 线程
            # 与 mj_step / 伺服调用并发。
            if not getattr(self.backend, "supports_pose_command", False):
                return {"ok": False,
                        "err": "%s 不支持位姿指令" % self.backend.name}
            p = np.asarray(msg["pos"], dtype=float).reshape(3)
            r6 = np.asarray(msg["rot6d"], dtype=float).reshape(6)
            g = float(msg["gripper"])
            if not (np.all(np.isfinite(p)) and np.all(np.isfinite(r6))):
                return {"ok": False, "err": "pose 含 NaN/inf"}
            with self._lock:
                self._pose_target = (p, r6, g, time.monotonic())
                self._target = None
            return {"ok": True}
        if op == "hold":
            with self._lock:
                st = self.backend.get_state()
                self._target = (st["joint"], st["gripper"], time.monotonic())
                self._pose_target = None
            return {"ok": True}
        if op == "move_base":
            # 只入队，由主循环下发 —— 与 actuate 同理，避免 handler 线程
            # 与 mj_step 并发写 ctrl。
            if not getattr(self.backend, "supports_base", False):
                return {"ok": False, "err": "backend 不支持底盘控制"}
            cmd = np.asarray([msg.get("vx", 0.0), msg.get("vy", 0.0),
                              msg.get("wz", 0.0)], dtype=float)
            if not np.all(np.isfinite(cmd)):
                return {"ok": False, "err": "base 指令含 NaN/inf"}
            with self._lock:
                self._base_target = (cmd, time.monotonic())
            return {"ok": True}
        if op == "stop_base":
            if not getattr(self.backend, "supports_base", False):
                return {"ok": False, "err": "backend 不支持底盘控制"}
            with self._lock:
                self._base_target = None
            return {"ok": True}
        if op == "status":
            st = self.backend.get_state()
            out = {"ok": True, "joint": st["joint"], "gripper": st["gripper"],
                   "safety": dict(self.safety.stats) if self.safety else {},
                   "hb_age_ms": (self.safety.hb_age_s() * 1000.0
                                 if self.safety else -1.0)}
            if getattr(self.backend, "supports_base", False):
                out["base"] = self.backend.get_base_state()
            return out
        if op == "reset":
            if not hasattr(self.backend, "reset"):
                return {"ok": False, "err": "backend 不支持 reset"}
            # === reset 必须由主循环执行，不能在 handler 线程里做 ===
            # mj_resetDataKeyframe / mj_forward 会写整块 MjData，若与主循环的
            # mj_step 并发，直接段错误（实测 faulthandler 抓到两线程同时在
            # mujoco 里），且无 Python 栈可看。这里只登记请求并等主循环处理完。
            with self._lock:
                # base_pose：远起点复位用（陌生环境里「车在房间另一头」）。
                # 与 objects/arm_q 一样只能在主循环里应用 —— mj_forward
                # 写整块 MjData，handler 线程执行会与 mj_step 并发段错误。
                self._pending_reset = {"objects": msg.get("objects"),
                                       "arm_q": msg.get("arm_q"),
                                       "base_pose": msg.get("base_pose")}
                self._reset_seq += 1
                want = self._reset_seq
            t0 = time.monotonic()
            while time.monotonic() - t0 < 5.0:
                with self._lock:
                    if self._layout_seq >= want:
                        break
                time.sleep(0.01)
            else:
                return {"ok": False, "err": "reset 超时"}
            return {"ok": True}
        return {"ok": False, "err": "未知 op=%r" % op}

    # ---------- 主循环 ----------
    def run(self):
        rate = float(CONTROL["rate_hz"])
        dt = 1.0 / rate
        t_state = 0.0
        t_image = 0.0
        period_state = 1.0 / float(CONTROL["state_rate_hz"])
        period_image = 1.0 / float(CONTROL["image_rate_hz"])

        th_rep = threading.Thread(target=self.rep.serve, args=(self.handle,),
                                  daemon=True)
        th_rep.start()

        step_period = float(getattr(self.backend, "m").opt.timestep) \
            if getattr(self.backend, "m", None) is not None else dt
        n_sub = max(1, int(round(dt / step_period)))

        self._log("L3 控制服务已启动：%s  %s  rate=%.0f Hz"
                  % (self.backend.name, ENDPOINTS["cmd"], rate))
        t_next = time.monotonic()
        try:
            while not self._stop:
                now = time.monotonic()
                if self._manual:
                    # === 手动步进模式：物理由 LocalBackend.apply_step 推进 ===
                    # 本线程只做三件事，且都必须留在**本线程**：
                    #   1) 应 L2 请求渲染一帧（renderer 线程亲和，不能跨线程）；
                    #   2) 按 state_rate 广播状态（L2 读快照，避免读半态的 MjData）；
                    #   3) 应答进入握手，让 enter_sync 确认循环已让出物理。
                    self._manual_seen += 1
                    if self._render_req:
                        self._render_req = False
                        with self._phys_lock:
                            self._publish_images()
                        self._render_seq += 1
                    if now - t_state >= period_state:
                        t_state = now
                        with self._phys_lock:
                            st_m = self.backend.get_state()
                            self._publish_state(st_m["joint"], st_m["gripper"])
                    t_next = time.monotonic()
                    time.sleep(0.002)
                    continue

                # 待处理的 reset（必须在这一线程里做，见 handle() 的注释）
                with self._lock:
                    pend = self._pending_reset
                    self._pending_reset = None
                if pend is not None:
                    objs = pend.get("objects")
                    self.backend.reset(objs, arm_q=pend.get("arm_q"))
                    # 必须在 reset 之后：mj_resetDataKeyframe 会把底盘放回
                    # 关键帧位置，先设位姿会被覆盖。
                    bp = pend.get("base_pose")
                    if bp is not None:
                        if not hasattr(self.backend, "set_base_pose"):
                            self._warn_once("后端不支持 set_base_pose，"
                                            "忽略 base_pose")
                        else:
                            r = self.backend.set_base_pose(*bp)
                            if isinstance(r, dict) and not r.get("ok"):
                                self._warn_once("set_base_pose 失败：%s"
                                                % r.get("err"))
                    if hasattr(self.backend, "settle"):
                        self.backend.settle(150)
                    self.safety._q_prev = None
                    self.safety._p_prev = None
                    with self._lock:
                        self._target = None
                        self._pose_target = None
                        self._base_target = None
                        self._layout_seq = self._reset_seq
                    self._log("场景已重置（%d 个物体，arm_q=%s，base_pose=%s）"
                              % (len(objs or []),
                                 "给定" if pend.get("arm_q") is not None
                                 else "默认",
                                 ("(%.2f, %.2f, %.0f°)"
                                  % (bp[0], bp[1], np.degrees(bp[2])))
                                 if pend.get("base_pose") is not None
                                 else "不动"))

                # 心跳
                hb_msg = self.sub_hb.latest(0)
                if hb_msg is not None and "hb" in hb_msg:
                    self.safety.note_heartbeat()

                # 底盘：速度指令有双重时效 —— 心跳丢失、或指令本身超过
                # watchdog_ms 没刷新，都立刻停车。
                # === 为什么要指令级时效而不仅是心跳 ===
                # 上层可能继续发心跳（还活着）但导航线程挂了，最后一条
                # 速度会一直生效。P30 的 SendVelocity 每周期不来也是停，
                # 这里保持同一语义。重复下发 0 是幂等的。
                if getattr(self.backend, "supports_base", False):
                    with self._lock:
                        btg = self._base_target
                    stale = (btg is None or not self.safety.hb_ok() or
                             (now - btg[1]) * 1000.0 >
                             float(CONTROL["watchdog_ms"]))
                    if stale:
                        self.backend.stop_base()
                        # 停车后重置加速度约束，否则下次起步要「还债」
                        self.safety.reset_base_vel()
                    else:
                        # 底盘速度闸：三轴独立限速 + 加速度限幅。
                        # VLA 直接出 base_velocity，模型跳变时靠这层截断
                        # （与位姿链路的 max_eef_speed 同一思路）。
                        v_safe, _info = self.safety.filter_base_vel(
                            btg[0], now=now)
                        self.backend.move_base(float(v_safe[0]),
                                               float(v_safe[1]),
                                               float(v_safe[2]))

                st = self.backend.get_state()
                q_meas = st["joint"]
                q_cmd, g_cmd = q_meas, st["gripper"]

                hold = self.safety.watchdog_hold(q_meas)
                if hold is not None:
                    q_cmd = hold
                    res = self.backend.actuate(q_cmd, g_cmd)
                else:
                    with self._lock:
                        tgt = self._target
                        ptgt = self._pose_target
                    if ptgt is not None:
                        # 位姿链路：笛卡尔闸 → 后端（仿真=ik_core，真机=厂商）
                        p_req, r6, g_cmd, _t_put = ptgt
                        p_safe, info = self.safety.filter_pose(p_req, now=now)
                        if not info["ok"]:
                            # 越界不截断、不「就近修正」：保持当前位姿
                            self._warn_once(info["reason"])
                            g_cmd = st["gripper"]
                            res = self.backend.actuate(q_meas, g_cmd)
                        else:
                            res = self.backend.actuate_pose(p_safe, r6, g_cmd)
                            if res.get("q") is not None:
                                q_cmd = np.asarray(res["q"], dtype=float)
                    elif tgt is not None:
                        q_req, g_cmd, _t_put = tgt
                        q_cmd, info = self.safety.filter(
                            q_req, q_meas, eef_pos=None)
                        if not info["ok"]:
                            self._warn_once(info["reason"])
                        res = self.backend.actuate(q_cmd, g_cmd)
                    else:
                        res = self.backend.actuate(q_cmd, g_cmd)

                if not res["ok"]:
                    self._warn_once(res["err"] or "actuate 失败")

                self.backend.step(n_sub)

                # 广播
                if now - t_state >= period_state:
                    t_state = now
                    self._publish_state(q_cmd, g_cmd)
                if now - t_image >= period_image:
                    t_image = now
                    self._publish_images()

                t_next += dt
                sleep = t_next - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    t_next = time.monotonic()
        finally:
            self.rep.stop()
            # 必须 join：REP 线程可能正卡在 recv/poll 里，解释器退出时
            # 带着 daemon 线程进 zmq 的 C 扩展会 core dump（实测）。
            th_rep.join(timeout=1.0)

    # ---------- inline 代理（同进程 L2 用）----------
    def snapshot(self):
        """最近一次广播的状态 / 图像（线程安全）。"""
        with self._snap_lock:
            return (dict(self._snap_state) if self._snap_state else None,
                    dict(self._snap_images))

    def publish(self, msg):
        """inline 模式下的状态写入（与 ZMQ 广播同一份数据）。"""
        with self._snap_lock:
            self._snap_state = msg
        self.pub_state.send(msg)

    def publish_images(self, imgs):
        with self._snap_lock:
            self._snap_images = imgs
        self.pub_image.send(imgs)

    def _publish_state(self, q_cmd, g_cmd):
        st = self.backend.get_state()
        msg = {
            "joint": st["joint"], "joint_cmd": np.asarray(q_cmd, float),
            "gripper": st["gripper"], "gripper_cmd": float(g_cmd),
            "eef_pos": st["eef_pos"], "eef_rot": st["eef_rot"],
            "t": st["t"], "wall": time.time(),
            "hb_ok": self.safety.hb_ok(),
        }
        if getattr(self.backend, "supports_base", False):
            msg["base"] = self.backend.get_base_state()
        objs = self.backend.get_objects()
        if objs:
            msg["objects"] = objs
        # 相机内外参随状态一起广播：L2 做视觉 grounding 需要它，
        # 而 L2 在真机上没有 MJCF 模型（相机外参只能由 L3 提供/标定）。
        # 频率低（每 state 帧一次，30 Hz），体积几十字节，可忽略。
        ci = getattr(self.backend, "camera_info", None)
        if ci is not None:
            try:
                msg["cameras"] = ci()
            except Exception:
                pass
        self.publish(msg)

    def _publish_images(self):
        imgs = self.backend.get_images()
        if imgs:
            self.publish_images({k: np.asarray(v) for k, v in imgs.items()})

    # ---------- 日志 ----------
    def _log(self, s):
        if self.verbose:
            print("[L3] " + s, flush=True)

    def _warn_once(self, s):
        if s and s != self._last_err:
            self._last_err = s
            print("[L3][warn] " + s, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="L3 控制服务")
    ap.add_argument("--backend", default=None, help="mujoco / real / remote")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="跑多少秒后退出（0=一直跑）")
    ap.add_argument("--no-render", action="store_true",
                    help="不渲染相机（只测总线时快很多）")
    ap.add_argument("--realtime", action="store_true", default=None,
                    help="按墙钟推进物理")
    a = ap.parse_args(argv)

    kw = {}
    if a.backend == "mujoco" or (a.backend is None):
        if a.no_render:
            kw["render"] = False
        if a.realtime is not None:
            kw["realtime"] = a.realtime
    backend = make_backend(a.backend, **kw)
    srv = ControlServer(backend)
    srv.connect()
    if a.duration > 0:
        timer = threading.Timer(a.duration, srv.rep.stop)
        timer.daemon = True
        timer.start()
    try:
        srv.run()
    except KeyboardInterrupt:
        print("\n[L3] 收到中断")
    finally:
        srv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
