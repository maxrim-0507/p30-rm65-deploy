# -*- coding: utf-8 -*-
"""本地直载 LeRobot 微调 checkpoint（不走 ZMQ 服务）。

=== 为什么需要它 ===
官方 `run_gr00t_server` 用 transformers 的 `AutoModel.from_pretrained`，
只认 HF 格式（config.json 里必须有 model_type）。我们用 lerobot_train
产出的 checkpoint 是 LeRobot 格式（config.json 的 type=groot，权重在
pretrained_model/ 下），服务端会直接报
「Unrecognized model ... Should have a model_type key」。

Thor 上 44/48 的评测正是用同一种本地加载（simulation/mujoco/
local_gr00t_policy.py）跑的。这里把它接进部署策略层，接口与
`Gr00tPolicy.infer(obs) -> (chunk Nx13, info)` 完全一致，技能层不用改。

=== 动作语义（与 gr00t_client 相同）===
[:, :9]  绝对 eef_9d（pos 相对臂座 + rot6d 前两行）
[:, 9]   夹爪绝对开度 [0,1]
[:, 10:13] 机体系底盘速度 [vx, vy, wz]
postprocessor 已把相对增量还原成绝对值，本类不再做任何坐标变换。
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

from ..spec import CAM_NAMES, camera_mjcf
from .base import Policy


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
            cands.append(os.path.join(path, "checkpoints", d,
                                      "pretrained_model"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    raise ValueError("找不到 pretrained_model（找过：\n  %s）"
                     % "\n  ".join(cands))


class LocalGr00tPolicy(Policy):
    """LeRobot GrootPolicy 的部署侧适配器。"""

    kind = "gr00t_local"
    direct_joint = False

    def __init__(self, ckpt, chunk_size=40, execute_steps=40,
                 device="cuda", n_action_steps=None, seed=0):
        super().__init__()
        import torch
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.groot.processor_groot import (
            make_groot_pre_post_processors_from_pretrained,
        )

        self.torch = torch
        self.device = torch.device(device)
        self.ckpt = _resolve_pretrained_dir(ckpt)
        self.chunk_size = int(chunk_size)
        # === 必须播种 ===
        # GR00T N1.7 的动作头是流匹配：每次 predict_action_chunk 内部
        # torch.randn 采初始噪声（groot_n1_7.py:657）。不播种时同一观测
        # 也会给出不同动作块，20 局回归实测 0/18；rollout 的 44/48 正是
        # 每块前 torch.manual_seed(seed + t)。这里用 chunk 计数复刻。
        self.seed = None if seed is None else int(seed)
        self._chunk_idx = 0
        if self.seed is not None:
            torch.manual_seed(self.seed)
        # 44/48 的评测是整块 40 步执行完再重规划；部署侧缺省保持一致。
        self.execute_steps = int(n_action_steps if n_action_steps is not None
                                 else execute_steps)
        if os.environ.get("GR00T_LOCAL_VERBOSE", "1") != "0":
            print("[local_gr00t] 载入 %s" % self.ckpt, flush=True)

        self.policy = GrootPolicy.from_pretrained(self.ckpt)
        self.policy.to(self.device)
        self.policy.eval()
        self.pre, self.post = make_groot_pre_post_processors_from_pretrained(
            self.policy.config, self.ckpt)
        self.out_dim = int(
            self.policy.config.output_features["action"].shape[0])
        if self.out_dim < 13:
            raise ValueError(
                "checkpoint 动作维度 %d < 13（需要 eef_9d+gripper+base）"
                % self.out_dim)

        # 与训练端同源的相机列名：front_cam / arm_cam
        self.cam_keys = {"head": "observation.images.%s"
                         % camera_mjcf("front"),
                         "wrist": "observation.images.%s"
                         % camera_mjcf("arm")}

    def reset(self):
        self._chunk_idx = 0
        try:
            self.policy.reset()
        except Exception:
            pass

    def infer(self, obs, frame_t=None):
        from lerobot.policies.utils import (
            prepare_observation_for_inference,
        )

        front = obs.images.get(CAM_NAMES[0])
        wrist = obs.images.get(CAM_NAMES[1])
        if front is None or wrist is None:
            missing = [n for n, v in ((CAM_NAMES[0], front),
                                      (CAM_NAMES[1], wrist)) if v is None]
            return None, {"reason": "缺少相机图像：%s" % missing}

        frame = {
            self.cam_keys["head"]: np.asarray(front, np.uint8),
            self.cam_keys["wrist"]: np.asarray(wrist, np.uint8),
            "observation.state": obs.to_state_vector(),
        }
        batch = prepare_observation_for_inference(
            frame, self.device, task=obs.instruction, robot_type="rm65")
        if self.seed is not None:
            # 每块固定播种：保证同一观测的动作可复现（流匹配采样）。
            _sd = self.seed + (int(frame_t) if frame_t is not None
                               else self._chunk_idx)
            self.torch.manual_seed(_sd)
            self._chunk_idx += 1
        with self.torch.inference_mode():
            batch = self.pre(batch)
            chunk = self.policy.predict_action_chunk(batch)
            chunk = self.post(chunk)

        arr = chunk.detach().float().cpu().numpy()
        arr = arr.reshape(-1, arr.shape[-1])[: self.chunk_size]
        if arr.shape[-1] < 13:
            return None, {"reason": "动作维度异常：%s" % (arr.shape,)}
        if not np.all(np.isfinite(arr)):
            return None, {"reason": "动作含 NaN/inf"}
        if os.environ.get("GR00T_LOCAL_DEBUG"):
            eef = np.asarray(obs.eef_pos, float)
            a0 = arr[0]
            print("[dbg] eef=%s -> a0=%s |d|=%.3f base=%s grip=%.3f"
                  % (np.round(eef, 3), np.round(a0[:3], 3),
                     float(np.linalg.norm(a0[:3] - eef)),
                     np.round(a0[10:13], 3), float(a0[9])), flush=True)
        if os.environ.get("GR00T_DUMP"):
            _sv = np.asarray(obs.to_state_vector(), float)
            print("[DUMP] st=%s c0=%s" % (np.round(_sv,4).tolist(), np.round(np.asarray(arr[0],float),4).tolist()), flush=True)
        _dd = os.environ.get("GR00T_DUMP_CHUNK")
        if _dd:
            _k = int(os.environ.get("GR00T_CHUNK_K", "0"))
            np.save("%s/dep_chunk_%03d.npy" % (_dd, _k), np.asarray(arr, float))
            np.save("%s/dep_state_%03d.npy" % (_dd, _k),
                    np.asarray(obs.to_state_vector(), float))
            np.save("%s/dep_front_%03d.npy" % (_dd, _k),
                    np.asarray(front, np.uint8))
            np.save("%s/dep_wrist_%03d.npy" % (_dd, _k),
                    np.asarray(wrist, np.uint8))
            os.environ["GR00T_CHUNK_K"] = str(_k + 1)
        return arr.astype(np.float32), {"source": "local",
                                        "chunk": arr.shape}
