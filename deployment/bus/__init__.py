# -*- coding: utf-8 -*-
"""ZMQ 消息总线（层间通信）。

=== 为什么全部懒加载 ===
`channels` 顶层 import zmq，`codec` 顶层 import msgpack。纯逻辑模块
（PolicyObs、eval_l1 自检、safety 单测）不该因为本机没装这两个运行期
依赖就 import 失败。按 PEP 562，只有真正取用时才加载。
"""
__all__ = ["Heartbeat", "Publisher", "Replier", "Requester", "Subscriber",
           "dumps", "loads"]

_CHANNELS = ("Heartbeat", "Publisher", "Replier", "Requester", "Subscriber")


def __getattr__(name):
    if name in _CHANNELS:
        from . import channels
        return getattr(channels, name)
    if name in ("dumps", "loads"):
        from . import codec
        return getattr(codec, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
