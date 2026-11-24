# -*- coding: utf-8 -*-
"""远程后端：L2 侧通过 ZMQ 使用 L3 控制的机器人。

=== 为什么它也是一个 RobotBackend ===
这样 L2 的技能代码只有一份：`backend` 既可以传 MujocoBackend
（单进程调试），也可以传 RemoteBackend（L3 在 Orin Nano 上）。
切部署形态只改 runner 的一行构造，技能逻辑零改动。

=== 它不是真机 HAL ===
真正碰硬件的只有 L3 进程里的 MujocoBackend / RealBackend。
RemoteBackend 只是「把 HAL 的方法翻译成 ZMQ 请求」的代理，
所以 get_objects() 只能拿到 L3 愿意广播的真值（仿真时有，真机无）。

=== 控制语义 ===
`actuate` 是同步 REQ/REP：L3 回 ok 表示**已接受**（限幅后下发），
不代表已到位。到位判断靠 `get_state()` 读 PUB 的最新状态，与真机
的伺服语义一致。不要在 actuate 里等到位 —— 那会把 30 Hz 循环拖垮到
关节响应速度（几百 ms）。
"""
import threading
import time

import numpy as np

from ..bus import Publisher, Requester, Subscriber
from ..bus.config import CONTROL, ENDPOINTS, TRANSPORT
from .base import RobotBackend


class RemoteBackend(RobotBackend):

    def __init__(self, host=None, cmd_port=None, state_port=None,
                 image_port=None, hb_period_s=0.05):
        host = host or TRANSPORT["host"]
        self.req = Requester(host, cmd_port or TRANSPORT["cmd_port"])
        self.sub_state = Subscriber(host, state_port or TRANSPORT["state_port"])
        self.sub_image = Subscriber(host, image_port or TRANSPORT["image_port"])
        self.pub_hb = Publisher(host, TRANSPORT["hb_port"])
        self._hb_period = float(hb_period_s)
        self._hb_stop = threading.Event()
        self._hb_thread = None
        self._last_state = None
        self._last_images = {}
        self._t_state = 0.0
        self._t_image = 0.0
        # 底盘 / 位姿能力由对端 ping 回答一次并缓存（不能在控制循环里每帧问）
        self._supports_base = False
        self._supports_pose = False

    def connect(self):
        # 等到第一帧状态，确认对端在跑。L3 没起时宁可这里超时失败，
        # 也不要让上层拿到空观测继续往下走（会渲染出一串无意义的动作）。
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            if self._poll(timeout_ms=100) is not None:
                break
        else:
            raise TimeoutError("L3 控制服务无响应：%s" % ENDPOINTS["state"])

        try:
            rep = self.req.call({"op": "ping"}, timeout_ms=1000)
            self._supports_base = bool(rep.get("supports_base", False))
            self._supports_pose = bool(rep.get("supports_pose", False))
        except Exception:
            self._supports_base = False
            self._supports_pose = False

        # 心跳必须持续发。L3 的看门狗 200 ms 收不到就 hold，
        # 表现为「机械臂根本不动」，而 IK 全是对的。
        self._hb_stop.clear()
        self._hb_thread = threading.Thread(target=self._hb_loop, daemon=True)
        self._hb_thread.start()
        return self

    def _hb_loop(self):
        n = 0
        while not self._hb_stop.is_set():
            n += 1
            self.pub_hb.send({"hb": n, "t": time.monotonic()})
            self._hb_stop.wait(self._hb_period)

    def close(self):
        self._hb_stop.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=0.5)
            self._hb_thread = None
        self.req.close()
        self.sub_state.close()
        self.sub_image.close()
        self.pub_hb.close()

    # ---------- 内部 ----------
    def _poll(self, timeout_ms=0):
        st = self.sub_state.latest(timeout_ms)
        if st is not None and "joint" in st:
            self._last_state = st
            self._t_state = time.monotonic()
        img = self.sub_image.latest(0)
        if img is not None:
            self._last_images = img
            self._t_image = time.monotonic()
        return self._last_state

    # ---------- 执行 ----------
    def actuate(self, joint, gripper):
        rep = self.req.call({
            "op": "actuate",
            "joint": np.asarray(joint, dtype=np.float64).reshape(6),
            "gripper": float(gripper),
        }, timeout_ms=int(CONTROL["watchdog_ms"]))
        return {"ok": bool(rep.get("ok")), "err": rep.get("err")}

    @property
    def supports_pose_command(self):
        return bool(self._supports_pose)

    def actuate_pose(self, pos, rot6d, gripper):
        rep = self.req.call({
            "op": "actuate_pose",
            "pos": np.asarray(pos, dtype=np.float64).reshape(3),
            "rot6d": np.asarray(rot6d, dtype=np.float64).reshape(6),
            "gripper": float(gripper),
        }, timeout_ms=int(CONTROL["watchdog_ms"]))
        return {"ok": bool(rep.get("ok")), "err": rep.get("err")}

    def step(self, n=1):
        # 控制时钟在对端；这里只是给调用方一个节奏点
        time.sleep(0.001 * int(n))

    # ---------- 底盘 ----------
    @property
    def supports_base(self):
        return self._supports_base

    def move_base(self, vx=0.0, vy=0.0, wz=0.0):
        rep = self.req.call({"op": "move_base", "vx": float(vx),
                             "vy": float(vy), "wz": float(wz)},
                            timeout_ms=int(CONTROL["watchdog_ms"]))
        return {"ok": bool(rep.get("ok")), "err": rep.get("err")}

    def stop_base(self):
        rep = self.req.call({"op": "stop_base"},
                            timeout_ms=int(CONTROL["watchdog_ms"]))
        return {"ok": bool(rep.get("ok")), "err": rep.get("err")}

    def get_base_state(self):
        st = self._poll(0)
        if st is None or "base" not in st:
            return {"pos": None, "yaw": None, "vel": (0.0, 0.0, 0.0)}
        b = st["base"]
        return {"pos": (None if b.get("pos") is None
                        else tuple(float(v) for v in b["pos"])),
                "yaw": b.get("yaw"),
                "vel": tuple(float(v) for v in
                             b.get("vel", (0.0, 0.0, 0.0)))}

    # ---------- 观测 ----------
    def get_state(self):
        st = self._poll(0)
        if st is None:
            raise RuntimeError("尚未收到任何状态帧")
        return {
            "joint": np.asarray(st["joint"], dtype=np.float64),
            "joint_cmd": np.asarray(st.get("joint_cmd", st["joint"]),
                                    dtype=np.float64),
            "gripper": float(st["gripper"]),
            "gripper_cmd": float(st.get("gripper_cmd", st["gripper"])),
            "eef_pos": np.asarray(st.get("eef_pos", np.zeros(3)),
                                  dtype=np.float64),
            "eef_rot": np.asarray(st.get("eef_rot", np.eye(3)),
                                  dtype=np.float64),
            "t": float(st.get("t", 0.0)),
        }

    def get_images(self):
        self._poll(0)
        return {k: np.asarray(v) for k, v in self._last_images.items()}

    def get_objects(self):
        st = self._poll(0)
        if st is None or "objects" not in st:
            return []
        return list(st["objects"])

    def camera_model(self, name):
        """用 L3 广播的相机内外参构造 CameraModel（跨机部署时用这个）。"""
        from ..perception.color_grounding import CameraModel
        st = self._poll(0)
        if st is None or "cameras" not in st or name not in st["cameras"]:
            return None
        c = st["cameras"][name]
        return CameraModel(int(c["w"]), int(c["h"]), float(c["fovy"]),
                           pos=np.asarray(c["pos"], float),
                           rot=np.asarray(c["rot"], float).reshape(3, 3))

    # ---------- 诊断 ----------
    def state_age_ms(self):
        if self._t_state <= 0:
            return float("inf")
        return (time.monotonic() - self._t_state) * 1000.0

    def image_age_ms(self):
        if self._t_image <= 0:
            return float("inf")
        return (time.monotonic() - self._t_image) * 1000.0
