# -*- coding: utf-8 -*-
"""τ0-VLA 客户端：L2 侧接口，与 Gr00tPolicy 输出的动作格式完全一致。

=== 协议 ===
REQ/REP（deployment.bus），msgpack：
  请求 {"op":"infer", "instruction":str, "state16":f32(16),
        "images":{"front_cam":u8(H,W,3), "arm_cam":u8(H,W,3)}}
  回包 {"ok":true, "chunk":f32(N,10), "info":{...,"latency_ms","calibrated"}}
chunk 已经是**绝对** EEF 位姿（服务端做完了 unrelativize），
执行端不能再次累加 —— 与 Gr00tPolicy 同一条规则。

=== 异步推理 ===
docs §5 的结论：Thor 上单次推理 352 ms，chunk 30 步 @30 Hz = 1000 ms。
同步调用会让机械臂每 15 步卡住 352 ms。这里用「提前一轮算」：
  · 第 1 次：提交并等待（必须拿到初始块）
  · 之后每次：直接返回**上一轮已算完**的块，同时把本次观测提交后台
因此每次返回的块基于「上一次重规划的观测」，与执行 15 步的窗口大致对齐。
代价是最多一个执行窗口（500 ms）的观测滞后 —— 这是 chunked policy 的
正常代价，比让机器人停住好得多。

=== 未标定 ===
回包 info["calibrated"] 为 False 时（预训练 ckpt 没有 norm_stats），
动作没有物理意义。ManipulateSkill 不检查这个标志；跑真实抓取前必须先
算 norm_stats（docs §6.2），否则看到的是「机械臂乱动、不报错」。
"""
import time

import numpy as np

from ..bus import Requester
from ..bus.config import POLICY
from .base import Policy

DEFAULT_PORT = 5566


class Tau0Policy(Policy):

    kind = "tau0"
    direct_joint = False

    def __init__(self, host="127.0.0.1", port=DEFAULT_PORT, chunk_size=30,
                 execute_steps=15, timeout_ms=None, async_infer=True,
                 instruction_prefix=""):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.execute_steps = int(execute_steps)
        self.async_infer = bool(async_infer)
        self.instruction_prefix = instruction_prefix
        # REQ 超时要比单次推理长：测到 352 ms，留 4 倍余量
        self.timeout_ms = int(timeout_ms or 4000)
        self._req = Requester(host, int(port))
        self._pending = None          # 上一轮提交的 future
        self._t_submit = 0.0
        if self.async_infer:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="tau0")

    def close(self):
        if self.async_infer:
            self._pool.shutdown(wait=False)
        self._req.close()

    def reset(self):
        self._pending = None

    # ---------- 打包 ----------
    def _payload(self, obs):
        imgs = {}
        for k in ("front_cam", "arm_cam"):
            img = obs.images.get(k)
            if img is None:
                return None, "缺少相机 %s" % k
            # 必须 copy：L3 的快照帧可能被下一帧复用
            imgs[k] = np.array(img, dtype=np.uint8, copy=True)
        text = (self.instruction_prefix + (obs.instruction or "")).strip()
        return {"op": "infer", "instruction": text,
                "state16": obs.to_state_vector().astype(np.float32),
                "images": imgs}, None

    def _call(self, payload):
        rep = self._req.call(payload, timeout_ms=self.timeout_ms)
        if not rep.get("ok"):
            raise RuntimeError(rep.get("err") or "推理服务返回失败")
        chunk = np.asarray(rep["chunk"], np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 10:
            raise RuntimeError("动作块形状异常：%s" % (chunk.shape,))
        if not np.all(np.isfinite(chunk)):
            raise RuntimeError("动作含 NaN/inf")
        return chunk[:self.chunk_size], rep.get("info", {})

    # ---------- 推理 ----------
    def infer(self, obs):
        payload, err = self._payload(obs)
        if payload is None:
            return None, {"reason": err}

        if not self.async_infer:
            try:
                return self._call(payload)
            except Exception as e:
                return None, {"reason": "%s: %s" % (type(e).__name__, e)}

        fut = self._pool.submit(self._call, payload)
        try:
            if self._pending is None:
                # 第一次必须同步拿到初始块，之后立刻预提交一份
                chunk, info = fut.result()
                self._pending = self._pool.submit(self._call, payload)
                self._t_submit = time.monotonic()
                return chunk, info
            chunk, info = self._pending.result()
            info = dict(info)
            info["obs_age_ms"] = (time.monotonic() - self._t_submit) * 1000.0
            self._pending = fut
            self._t_submit = time.monotonic()
            return chunk, info
        except Exception as e:
            self._pending = None
            return None, {"reason": "%s: %s" % (type(e).__name__, e)}


def make_tau0(**kw):
    cfg = POLICY.get("tau0", {})
    return Tau0Policy(
        host=kw.pop("host", cfg.get("host", "127.0.0.1")),
        port=kw.pop("port", cfg.get("port", DEFAULT_PORT)),
        chunk_size=kw.pop("chunk_size", cfg.get("chunk_size", 30)),
        execute_steps=kw.pop("execute_steps", cfg.get("execute_steps", 15)),
        async_infer=kw.pop("async", cfg.get("async", True)),
        timeout_ms=kw.pop("timeout_ms", cfg.get("timeout_ms")),
        **kw)
