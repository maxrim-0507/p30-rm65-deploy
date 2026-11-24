# -*- coding: utf-8 -*-
"""在 Thor 本地进程里跑训练好的 GR00T N1.7 checkpoint（不走 ZMQ / 云端）。

和 rollout_color_grasp.py 里的 Gr00tPolicy 用同一个 act(obs, instruction, t)
接口，所以评估主流程不用改，只要 --policy 换成 local。

推理顺序完全照搬 lerobot 官方（common/control_utils.py:74-95 与
rollout/inference/sync.py:104-125）：
    prepare_observation_for_inference → preprocessor → policy.predict_action_chunk → postprocessor

为什么用 predict_action_chunk 而不是 select_action：
  select_action 内部也是「取整个 chunk 再逐帧吐」，但它把队列藏在 policy 里，
  我们在评估里每集都要 reset；官方注释也写明相对动作场景必须用
  predict_action_chunk + 自己对 chunk 做后处理。这里显式管理队列更清楚。
"""
import os

import numpy as np
import torch

N_ACTION_STEPS = 40


class LocalGr00tPolicy:
    name = "gr00t_local"

    def __init__(self, ckpt, device="cuda", n_action_steps=N_ACTION_STEPS,
                 seed=None, verbose=True):
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.groot.processor_groot import (
            make_groot_pre_post_processors_from_pretrained,
        )

        self.device = torch.device(device)
        ckpt = _resolve_pretrained_dir(ckpt)
        if verbose:
            print("[local_gr00t] 载入 %s" % ckpt, flush=True)

        self.policy = GrootPolicy.from_pretrained(ckpt)
        self.policy.to(self.device)
        self.policy.eval()
        self.pre, self.post = make_groot_pre_post_processors_from_pretrained(
            self.policy.config, ckpt)
        self.n_action_steps = int(n_action_steps)
        self.pending = []
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
        if verbose:
            cfg = self.policy.config
            print("[local_gr00t] chunk=%d n_action_steps=%d out_dim=%d"
                  % (cfg.chunk_size, self.n_action_steps,
                     cfg.output_features["action"].shape[0]), flush=True)

    def reset(self):
        self.pending = []
        self.policy.reset()

    def _build_batch(self, obs, instruction):
        """obs 是 rollout 的原始输出：uint8 HWC 图像 + float32 state16。"""
        from lerobot.policies.utils import prepare_observation_for_inference

        frame = {
            "observation.images.arm_cam": obs["arm"],
            "observation.images.front_cam": obs["front"],
            "observation.state": np.asarray(obs["state"], dtype=np.float32),
        }
        return prepare_observation_for_inference(
            frame, self.device, task=instruction, robot_type="rm65")

    def act(self, obs, instruction, t):
        if not self.pending:
            if self.seed is not None:
                torch.manual_seed(self.seed + t)
            with torch.inference_mode():
                batch = self.pre(self._build_batch(obs, instruction))
                chunk = self.policy.predict_action_chunk(batch)
                chunk = self.post(chunk)
            arr = chunk.detach().float().cpu().numpy()
            arr = arr.reshape(-1, arr.shape[-1])
            self.pending = list(arr[: self.n_action_steps])
            if os.environ.get("GR00T_DUMP"):
                print("[DUMP] st=%s c0=%s" % (np.round(np.asarray(obs["state"],float),4).tolist(), np.round(np.asarray(self.pending[0],float),4).tolist()), flush=True)
            _dd = os.environ.get("GR00T_DUMP_CHUNK")
            if _dd:
                import numpy as _np
                _k = int(os.environ.get("GR00T_CHUNK_K", "0"))
                _np.save("%s/roll_chunk_%03d.npy" % (_dd, _k),
                         _np.asarray(self.pending, float))
                _np.save("%s/roll_state_%03d.npy" % (_dd, _k),
                         _np.asarray(obs["state"], float))
                _np.save("%s/roll_front_%03d.npy" % (_dd, _k),
                         _np.asarray(obs["front"], _np.uint8))
                _np.save("%s/roll_wrist_%03d.npy" % (_dd, _k),
                         _np.asarray(obs["arm"], _np.uint8))
                os.environ["GR00T_CHUNK_K"] = str(_k + 1)
            if os.environ.get("GR00T_DEBUG"):
                st = np.asarray(obs["state"], dtype=np.float32)
                a0 = self.pending[0]
                aL = arr[min(39, len(arr) - 1)]
                print("[dbg t=%d] state eef=%s q=%s grip_cmd=%.3f"
                      % (t, np.round(st[:3], 4), np.round(st[10:16], 3), st[9]),
                      flush=True)
                print("[dbg] ch0=%s dpos=%s |d|=%.4f ch39=%s grip=%.3f base=%s"
                      % (np.round(a0[:3], 4), np.round(a0[:3] - st[:3], 4),
                         float(np.linalg.norm(a0[:3] - st[:3])),
                         np.round(aL[:3], 4), a0[9], np.round(a0[10:13], 3)),
                      flush=True)
                print("[dbg] base_xy=%s"
                      % (np.round(np.asarray(obs["base_xy"], dtype=np.float32), 4),),
                      flush=True)
            if not self.pending:
                raise RuntimeError("模型返回空 chunk")
        return self.pending.pop(0)


def _resolve_pretrained_dir(path):
    """接受 checkpoint 根目录或任意一层子目录，找到真正的 pretrained_model。"""
    cands = [
        path,
        os.path.join(path, "pretrained_model"),
        os.path.join(path, "checkpoints", "last", "pretrained_model"),
    ]
    if os.path.isdir(os.path.join(path, "checkpoints")):
        subs = sorted(d for d in os.listdir(os.path.join(path, "checkpoints"))
                      if d.isdigit())
        for d in reversed(subs):
            cands.append(os.path.join(path, "checkpoints", d, "pretrained_model"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    raise SystemExit("找不到 pretrained_model（找过：\n  %s）" % "\n  ".join(cands))
