# -*- coding: utf-8 -*-
"""GR00T 推理客户端：直接用官方 PolicyClient，不自己实现协议。

=== 为什么不再手写 msgpack ===
上一版自己拼 {"video","state","language","timestamp"}，与官方协议不符：
  · 缺 {"endpoint": "get_action", "data": {"observation": ..., "options": None}} 信封
  · language 必须是 dict[str, list[list[str]]]，不是裸字符串
  · 响应是 (action_dict, info_dict) 而非 {"actions": ...}
手写协议等于自造一套 ABI，服务端一升级就静默失配。现在直接复用
`gr00t.policy.server_client.PolicyClient`（官方 SO100 真机脚本同款）。

参考实现：
  Isaac-GR00T/gr00t/policy/server_client.py             （PolicyClient / MsgSerializer）
  Isaac-GR00T/gr00t/eval/real_robot/SO100/eval_so100.py （观测打包 + 动作解码）

=== 观测形状（官方 check_observation 强制）===
  video    : {key: uint8 (B, T, H, W, C)}     T = len(video.delta_indices) = 1
  state    : {key: float32 (B, T, D)}         T = len(state.delta_indices) = 1
  language : {key: list[list[str]]}           外层 B，内层 T=1，元素是 str

=== 动作（官方 check_action 强制）===
  {action_key: float32 (B, T, D)}   T = len(action.delta_indices) = 40
                                    （configs/gr00t/rm65_config.py 的
                                     ACTION_HORIZON，与 base ckpt 一致）
  eef_9d 到客户端手里就是**绝对位姿**：v2 配置（路线甲）声明 rep=ABSOLUTE，
  processor apply/unapply 两端直通（state_action_processor.py:369/:478）；
  客户端不做任何坐标变换，逐帧绝对值原样进 chunk，执行端逐帧用当帧臂座
  解释（actuate_pose 现取底盘）。客户端不能再做 abs = state + delta。
"""
import time

import numpy as np

from ..spec import CAM_NAMES, camera_mjcf
from .base import Policy

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555

# 与 examples/RM65/rm65_config.py 保持一致
LANGUAGE_KEY = "annotation.human.task_description"


def _add_batch_time(x):
    """(...,) → (1, 1, ...)，对齐官方 eval_so100.py 的 recursive_add_extra_dim。"""
    return np.asarray(x)[None, None]


class Gr00tPolicy(Policy):
    """VLA 策略：L2 技能层，出 40 步绝对 EEF 位姿块 + 夹爪开度。

    chunk_size=40 是 ckpt 的输出长度，不是随便取的：base N1.7 的
    action_horizon=40，我们的 modality 配置也声明 40。服务端多给就截断
    （n = min(..., chunk_size)），所以设小不报错、只是扔掉余量。
    execute_steps=15 对应 30 Hz 下 500 ms 执行窗口；官方 RTC 建议执行
    ≥32 步让块内更平滑，但代价是观测更陈旧 —— 等 G4 测出 Thor 的推理
    延迟后再定，别现在拍死。
    """

    kind = "gr00t"
    direct_joint = False        # 走 _run_policy 路径（执行端需要 IK）

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 chunk_size=40, execute_steps=15, timeout_ms=15000):
        super().__init__()
        self.host, self.port = host, int(port)
        self.chunk_size = int(chunk_size)
        self.execute_steps = int(execute_steps)
        self.timeout_ms = int(timeout_ms)

        # 延迟导入：官方包只在推理进程里需要，离线工具不该被迫装 torch
        from gr00t.policy.server_client import PolicyClient
        self._client = PolicyClient(host=self.host, port=self.port,
                                    timeout_ms=self.timeout_ms)
        self._modality = None

        # 相机 / 状态键从 vla_spec.yaml 取，与训练端同源。
        #
        # === 用 modality_keys 口径，不是 LeRobot 文件键 ===
        # vla_spec 里 head 有两个名字：
        #   mjcf_name   head_cam                    ← 场景里的相机名
        #   dataset_key observation.images.head_cam ← LeRobot 的列名
        # GR00T 的 modality_keys 指的是**前者**（rm65_config.py 里写的是
        # head_cam）。用 dataset_key 会得到
        #   "Video key observation.images.head_cam must be in observation"
        # 这种只报错不指路的断言。这里统一用 mjcf_name。
        # 逻辑角色 → 物理相机名统一走 spec.camera_mjcf()，不在各处下标。
        # v36 的口径是 front_cam / arm_cam（rm65_config.py 的 modality_keys）；
        # 旧代码里的 head/wrist 只是逻辑角色名，不参与键名。
        self.cam_keys = {"head": camera_mjcf("front"),
                         "wrist": camera_mjcf("arm")}
        self.state_keys = ["single_arm", "eef_9d", "gripper_position"]
        self.action_keys = ["eef_9d", "gripper_position", "base_velocity"]

    # ---------- 握手 ----------
    def ping(self):
        return self._client.ping()

    def close(self):
        self._client.close()

    def get_modality_config(self):
        if self._modality is None:
            self._modality = self._client.get_modality_config()
        return self._modality

    def check_alignment(self):
        """启动自检：客户端键名 vs 服务端声明。返回 (ok, 说明)。

        服务端 strict=True 时键名错了只报 'Video key X must be in
        observation'，不指出该填什么；这里主动比对一次，把差异一次报全。
        """
        try:
            mc = self.get_modality_config()
        except Exception as e:
            return False, "取 modality_config 失败：%s" % e

        want = {
            "video": sorted(self.cam_keys.values()),
            "state": sorted(self.state_keys),
            "action": sorted(self.action_keys),
        }
        got, problems = {}, []
        for mod, w in want.items():
            cfg = mc.get(mod) if isinstance(mc, dict) else None
            keys = getattr(cfg, "modality_keys", None)
            if keys is None:
                problems.append("服务端未返回 %s 配置" % mod)
                continue
            got[mod] = sorted(keys)
            if sorted(keys) != w:
                problems.append("%s 键不一致 服务端=%s 客户端=%s"
                                % (mod, sorted(keys), w))
        if problems:
            return False, "; ".join(problems)
        return True, "对齐 OK：%s" % got

    # ---------- 打包观测 ----------
    def _pack(self, obs):
        """PolicyObs → 官方 get_action 入参。返回 (observation, err)。"""
        video = {}
        for role, mjcf_name in (("head", CAM_NAMES[0]),
                                ("wrist", CAM_NAMES[1])):
            img = obs.images.get(mjcf_name)
            if img is None:
                return None, "缺少相机 %s" % mjcf_name
            video[self.cam_keys[role]] = _add_batch_time(
                np.asarray(img, np.uint8))          # (1,1,H,W,3)

        sv = obs.to_state_vector()                  # (16,) float32
        # 布局（2026-11-13）：eef_9d(0:9) + gripper(9:10) + single_arm(10:16)
        # state_keys 顺序是 [single_arm, eef_9d, gripper_position]，所以按名取。
        state = {
            self.state_keys[1]: _add_batch_time(sv[0:9]),    # eef_9d
            self.state_keys[2]: _add_batch_time(sv[9:10]),   # gripper_position
            self.state_keys[0]: _add_batch_time(sv[10:16]),  # single_arm
        }
        language = {LANGUAGE_KEY: [[obs.instruction]]}   # B=1, T=1
        return {"video": video, "state": state,
                "language": language}, None

    # ---------- 推理 ----------
    def infer(self, obs):
        """返回 (chunk (N,13) float32, info)。

        列顺序与 configs/vla_spec.yaml 的 A_eef_base_absolute 一致：
        [:9] = 绝对 eef_9d，[9] = gripper 开度，[10:13] = base_velocity
        = [vx, vy, wz]（机体系 m/s、rad/s）。
        """
        payload, err = self._pack(obs)
        if payload is None:
            return None, {"reason": err}

        t0 = time.monotonic()
        try:
            action, info = self._client.get_action(payload)
        except Exception as e:
            return None, {"reason": "get_action 失败：%s" % e}
        lat_ms = (time.monotonic() - t0) * 1000.0

        missing = [k for k in self.action_keys if k not in action]
        if missing:
            return None, {"reason": "响应缺动作键 %s，实际有 %s"
                          % (missing, sorted(action)),
                          "latency_ms": lat_ms}

        eef = np.asarray(action["eef_9d"], np.float32)
        grip = np.asarray(action["gripper_position"], np.float32)
        base = np.asarray(action["base_velocity"], np.float32)
        eef = eef[0] if eef.ndim == 3 else eef        # (T,9)
        grip = grip[0] if grip.ndim == 3 else grip    # (T,1)
        base = base[0] if base.ndim == 3 else base    # (T,3)
        if eef.shape[-1] != 9 or grip.shape[-1] != 1 or base.shape[-1] != 3:
            return None, {"reason": "动作维度异常 eef=%s grip=%s base=%s"
                          % (eef.shape, grip.shape, base.shape),
                          "latency_ms": lat_ms}
        n = min(len(eef), len(grip), len(base), self.chunk_size)
        chunk = np.concatenate([eef[:n], grip[:n], base[:n]],
                               axis=1).astype(np.float32)

        if not np.all(np.isfinite(chunk)):
            return None, {"reason": "动作含 NaN/inf", "latency_ms": lat_ms}
        return chunk, {"chunk": chunk.shape, "latency_ms": lat_ms,
                       "server_info": info}
