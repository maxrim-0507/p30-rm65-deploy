# -*- coding: utf-8 -*-
"""ZMQ 通道封装：REQ/REP 指令 + PUB/SUB 状态与图像 + 心跳。

=== 拓扑选择 ===
L3 控制服务是**唯一的执行入口**。上层（L2 技能）通过 REQ/REP 下发，
好处是同步语义清晰：L3 拒绝越界时 L2 立刻知道，不用等到看门狗超时。

状态与图像走 PUB/SUB：L2 只需要最新一帧，积压的旧帧只会把控制搞得
更差。SUB 端统一 `CONFLATE=1`（只保留最后一帧）。

=== 端口 ===
全部来自 configs/deploy.yaml。跨机部署只需把 host 改成对端 IP：
  Thor  192.168.43.113
  狗 Orin Nano 192.168.1.100（L3 在这边跑时，configure host=那台的 IP）
"""
import time

import zmq

from .codec import dumps, loads


def _ctx(ctx=None):
    return ctx or zmq.Context.instance()


class Publisher:
    """PUB 端。attach 后立刻发一帧空包，让订阅者不必等第一个真实帧。"""

    def __init__(self, host, port, ctx=None, bind=True):
        self.sock = _ctx(ctx).socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 4)
        self.sock.setsockopt(zmq.LINGER, 0)
        addr = "tcp://%s:%d" % (host, port)
        if bind:
            self.sock.bind(addr)
        else:
            self.sock.connect(addr)
        self.addr = addr

    def send(self, msg):
        try:
            self.sock.send(dumps(msg), zmq.NOBLOCK)
        except zmq.Again:
            # 没有订阅者或 HWM 满：直接丢。状态帧不值得排队。
            pass

    def close(self):
        self.sock.close(0)


class Subscriber:
    """SUB 端。`latest()` 非阻塞取最新帧（CONFLATE 保证不积压）。"""

    def __init__(self, host, port, topic=b"", ctx=None, bind=False):
        self.sock = _ctx(ctx).socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, topic)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.RCVHWM, 4)
        self.sock.setsockopt(zmq.LINGER, 0)
        addr = "tcp://%s:%d" % (host, port)
        if bind:
            self.sock.bind(addr)
        else:
            self.sock.connect(addr)
        self.addr = addr

    def latest(self, timeout_ms=0):
        if not self.sock.poll(timeout_ms):
            return None
        try:
            return loads(self.sock.recv(zmq.NOBLOCK))
        except (zmq.Again, Exception):
            return None

    def close(self):
        self.sock.close(0)


class Replier:
    """REP 端。`serve(handler)` 阻塞循环；handler(msg) -> dict。"""

    def __init__(self, host, port, ctx=None, bind=True):
        self.sock = _ctx(ctx).socket(zmq.REP)
        self.sock.setsockopt(zmq.RCVHWM, 8)
        self.sock.setsockopt(zmq.LINGER, 0)
        addr = "tcp://%s:%d" % (host, port)
        if bind:
            self.sock.bind(addr)
        else:
            self.sock.connect(addr)
        self.addr = addr
        self._stop = False

    def serve(self, handler, poll_ms=100):
        """阻塞服务循环。`handler` 返回 None 时回 {"ok": False, "err": "no-reply"}。"""
        while not self._stop:
            if not self.sock.poll(poll_ms):
                continue
            try:
                msg = loads(self.sock.recv())
            except Exception as e:              # 坏包不能让服务挂掉
                self.sock.send(dumps({"ok": False, "err": "bad-msg:%s" % e}))
                continue
            try:
                rep = handler(msg)
            except Exception as e:              # 业务异常同样要回包
                rep = {"ok": False, "err": "%s: %s" % (type(e).__name__, e)}
            self.sock.send(dumps(rep if rep is not None
                                 else {"ok": False, "err": "no-reply"}))

    def stop(self):
        self._stop = True

    def close(self):
        self.sock.close(0)


class Requester:
    """REQ 端。`call(msg, timeout_ms)` 带超时；超时后重建 socket。"""

    def __init__(self, host, port, ctx=None, bind=False):
        self._host, self._port = host, port
        self._ctx = _ctx(ctx)
        self.sock = None
        self._connect(bind)

    def _connect(self, bind):
        if self.sock is not None:
            self.sock.close(0)
        self.sock = self._ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVHWM, 4)
        self.sock.setsockopt(zmq.LINGER, 0)
        addr = "tcp://%s:%d" % (self._host, self._port)
        (self.sock.bind if bind else self.sock.connect)(addr)

    def call(self, msg, timeout_ms=2000):
        self.sock.send(dumps(msg))
        if not self.sock.poll(timeout_ms):
            # REQ 状态机在超时后不可再用，必须重建，否则下一次 send 直接 EFSM
            self._connect(False)
            return {"ok": False, "err": "timeout"}
        return loads(self.sock.recv())

    def close(self):
        if self.sock is not None:
            self.sock.close(0)


class Heartbeat:
    """单向心跳计数。L2 每 100 ms `tick()`，L3 检查 `age_ms()`。

    === 为什么不用消息时间戳 ===
    跨机时钟不同步时时间戳会给出负延迟或长期漂移。这里只传一个递增
    计数 + 发送端 monotonic 时间，接收端按**本地到达时刻**算年龄，
    与对端时钟无关。
    """

    def __init__(self, publisher):
        self.pub = publisher
        self.count = 0
        self._t_last = 0.0

    def tick(self):
        self.count += 1
        self._t_last = time.monotonic()
        self.pub.send({"hb": self.count, "t": self._t_last})

    def age_ms(self, last_seen):
        if last_seen <= 0:
            return float("inf")
        return (time.monotonic() - last_seen) * 1000.0
