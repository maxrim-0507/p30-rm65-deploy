# -*- coding: utf-8 -*-
"""ZMQ 消息编解码：msgpack + numpy 数组。

=== 为什么不用 msgpack-numpy ===
Thor 外网 IPv4 出口不通，装不了新包。自己写 40 行够用，而且能把
ndarray 的 dtype/shape 一起塞进去，不解 pickle。实测 240×320×3 uint8
双相机编码 0.4 ms、解码 0.3 ms，30 Hz 下可忽略。

=== 约定 ===
消息是 dict（字符串键）。值可以是标量 / str / bool / None / list /
dict / np.ndarray。ndarray 统一编码成
    {"__nd__": "<dtype str>", "__sh__": [shape...], "d": <raw bytes>}

=== raw=False 而不是 raw=True ===
use_bin_type=True 时 str 走 str 家族、bytes 走 bin 家族；配 raw=False
解码后 str 还是 str、bytes 还是 bytes。写 raw=True 会让**所有键**变
bytes，于是调用方 `msg["skill"]` 报 KeyError —— 踩过一次。
"""
import numpy as np

try:
    import msgpack
except ImportError as e:  # pragma: no cover
    raise ImportError("需要 msgpack：pip install msgpack") from e

_ND = "__nd__"
_SH = "__sh__"


def _encode(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [_encode(x) for x in obj.tolist()]
        arr = np.ascontiguousarray(obj)
        return {_ND: arr.dtype.str, _SH: list(arr.shape), "d": arr.tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(x) for x in obj]
    return obj


def _decode(obj):
    if isinstance(obj, dict):
        if _ND in obj:
            arr = np.frombuffer(obj["d"], dtype=np.dtype(obj[_ND]))
            return arr.reshape(tuple(obj[_SH])).copy()
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(x) for x in obj]
    return obj


def dumps(msg):
    """dict → bytes。"""
    return msgpack.packb(_encode(msg), use_bin_type=True)


def loads(buf):
    """bytes → dict。"""
    return _decode(msgpack.unpackb(buf, raw=False, strict_map_key=False))
