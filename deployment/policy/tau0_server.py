# -*- coding: utf-8 -*-
"""τ0-VLA 推理服务：独立进程，ZMQ REP 收观测、回动作块。

=== 为什么必须是独立进程 ===
τ0 需要 transformers 5.5.4 / tokenizers 0.22.2，而部署栈用的是 5.19。
在同一进程里 import 会互相覆盖，而且 6 GB 权重 + 3 s 加载不该拖住 L3。
用 docs/tau0_Thor部署验证_2026-11-05.md §2 的隔离装法：

    PYTHONPATH=/home/user/tau0_pkgs \
    /home/user/mjl_collect_env/bin/python -m deployment.policy.tau0_server \
        --ckpt /home/user/tau0_ckpt --port 5566

（PYTHONPATH 只加隔离目录；deployment 由 cwd 找到，torch 仍是系统的
2.11+cu130，带 sm_110。）

=== 它不做的事 ===
不做视觉 grounding、不做 IK、不读写文件。输入是 16D 状态 + 两路图，
输出是 (N,10) 绝对 EEF 位姿 + 夹爪开度 —— 与 Gr00tPolicy 的接口一致。

=== 归一化 ===
预训练 ckpt 没有 norm_stats（docs §6.2）。没有 --norm-stats 时按恒等
归一化跑，回包带 calibrated=false。上层必须据此拒跑真实抓取，否则
看到的是「机械臂乱动但没有任何报错」。
"""
import argparse
import io
import sys
import time

import numpy as np

from ..bus import Replier, dumps, loads  # noqa: F401  (dumps/loads 供自检)
from ..bus.config import POLICY
from .unified40 import (NormStats, action_mask, pack_state16, unrelativize,
                        unpack_action40)


class Tau0Engine:

    def __init__(self, ckpt, norm_stats=None, vlm_model_type="qwen3vl",
                 device="cuda", dtype="bfloat16"):
        import torch
        from transformers import AutoProcessor

        from tau0_vla.models.vision_language_action_models.tau0_vla import (
            Tau0VLAModel,
        )

        self.torch = torch
        self.ckpt = ckpt
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.vlm_model_type = vlm_model_type
        # state 与 action 的分布完全不同，必须两套统计量（docs §6.2）。
        # 文件里优先读 state_*/action_*，没有才退回 mean/std。
        self.norm_state = NormStats.load(norm_stats, role="state")
        self.norm_action = NormStats.load(norm_stats, role="action")

        t0 = time.monotonic()
        self.model = Tau0VLAModel.from_pretrained(ckpt, torch_dtype=self.dtype)
        self.model.to(device).eval()
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.load_s = time.monotonic() - t0

        from tau0_vla.vlm.qwenvl_utils import build_vla_inference_data_dict
        self._build = build_vla_inference_data_dict

    # ---------- 推理 ----------
    def infer(self, images, state16, instruction):
        import numpy as np
        from PIL import Image

        t0 = time.monotonic()
        imgs = [Image.fromarray(np.asarray(images[k], np.uint8))
                for k in ("front_cam", "arm_cam")]

        d = self._build(self.processor, imgs, instruction,
                        vlm_model_type=self.vlm_model_type)
        d = {k: (v.to(self.device) if hasattr(v, "to") else v)
             for k, v in d.items()}

        state40 = pack_state16(state16)
        normed = self.norm_state.normalize(state40)
        d["state"] = self.torch.from_numpy(
            normed[None]).to(self.device, self.dtype)

        pre_ms = (time.monotonic() - t0) * 1000.0
        with self.torch.no_grad():
            out = self.model.sample_action(d)
        self.torch.cuda.synchronize()
        ms = (time.monotonic() - t0) * 1000.0

        rel = out[0].float().cpu().numpy()           # (N, 40) 相对动作
        ab = self.norm_action.denormalize(rel)
        # 必须用 action_mask：否则 unrelativize 会把 state 的关节角加到
        # 动作里本来为 0 的 24:30 槽位（当前 unpack 不读，但属于埋雷）。
        abs40 = unrelativize(ab, state40, mask=action_mask())
        chunk = unpack_action40(abs40)
        if not np.all(np.isfinite(chunk)):
            raise RuntimeError("τ0 输出含 NaN/inf")
        calibrated = bool(self.norm_state.calibrated
                          and self.norm_action.calibrated)
        return chunk, {"latency_ms": ms, "pre_ms": pre_ms,
                       "chunk": list(chunk.shape),
                       "calibrated": calibrated}


def main(argv=None):
    ap = argparse.ArgumentParser(description="τ0-VLA 推理服务")
    ap.add_argument("--ckpt", default="/home/user/tau0_ckpt")
    ap.add_argument("--norm-stats", default=None,
                    help="40D mean/std 的 .npz 或 .json；不给我们按恒等跑")
    ap.add_argument("--port", type=int,
                    default=int(POLICY.get("tau0", {}).get("port", 5566)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="跑多少秒后退出（0=一直）")
    a = ap.parse_args(argv)

    eng = Tau0Engine(a.ckpt, a.norm_stats)
    print("[tau0] 加载 %.1f s，norm_stats=%s（state=%s action=%s）  port=%d"
          % (eng.load_s, a.norm_stats or "无",
             eng.norm_state.calibrated, eng.norm_action.calibrated, a.port),
          flush=True)

    rep = Replier(a.host, a.port)
    n_ok = 0
    t_stop = time.monotonic() + a.duration if a.duration > 0 else None

    def handle(msg):
        nonlocal n_ok
        if msg.get("op") == "ping":
            return {"ok": True, "engine": "tau0",
                    "calibrated": bool(eng.norm.calibrated),
                    "load_s": eng.load_s}
        if msg.get("op") != "infer":
            return {"ok": False, "err": "未知 op=%r" % msg.get("op")}
        try:
            chunk, info = eng.infer(msg["images"],
                                    np.asarray(msg["state16"], float),
                                    msg.get("instruction", ""))
        except Exception as e:
            return {"ok": False, "err": "%s: %s" % (type(e).__name__, e)}
        n_ok += 1
        if n_ok <= 3 or n_ok % 10 == 0:
            print("[tau0] #%d %.0f ms chunk=%s calibrated=%s"
                  % (n_ok, info["latency_ms"], info["chunk"],
                     info["calibrated"]), flush=True)
        return {"ok": True, "chunk": chunk, "info": info}

    import threading
    if t_stop is not None:
        threading.Timer(a.duration, rep.stop).start()
    try:
        rep.serve(handle)
    finally:
        rep.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
