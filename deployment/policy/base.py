# -*- coding: utf-8 -*-
"""策略接口：观测 → 动作块。

=== 为什么要一个 base 而不是直接调 VLA ===
两条硬理由：
  1. VLA 还没训出来（云训练机凭据失效）。但全链路必须现在就能跑通，
     否则等模型训好那天才发现总线/时序/单位对不上，代价高得多。
  2. VLA 上线后，DAgger 需要「专家动作」与「模型动作」在同一接口下
     并列采样。base 把两者统一成 `infer(obs) -> (chunk, info)`。

=== 动作格式（与 GR00T 的 oxe_droid EEF 约定一致）===
chunk: (N, 10) float32
  [:, :9]  绝对 EEF 位姿 eef_9d = pos(3) + rot6d(前两行 6 个)
           pos 的 **xyz 均相对臂座**（2026-11-09 校准）；rot6d 是世界系
           R[:2,:]（口径见 hal/base.py）
  [:, 9]   夹爪目标开度 [0,1]（绝对，不是增量）

为什么不自己算相对增量：数据集存的是绝对位姿，modality 配置里声明
ActionRepresentation.RELATIVE + state_key="eef_9d"，服务端 processor 已经在
decode 时把相对增量还原成绝对位姿（unapply_action）。客户端/执行端再做一次
abs = state + delta 会**双重转换**，动作翻倍。
"""
import time

import numpy as np

from ..bus.config import POLICY

ACTION_DIM = 10


class PolicyObs:
    """策略输入。字段与 vla_spec.yaml 的 state/action 布局逐项对应。"""

    def __init__(self, images, state, instruction, eef_pos, prev_action=None,
                 t=None):
        self.images = images or {}
        self.state = state
        self.instruction = instruction or ""
        self.eef_pos = np.asarray(eef_pos, float)
        self.prev_action = prev_action
        self.t = time.monotonic() if t is None else float(t)

    def to_state_vector(self):
        """拼 16 维，顺序必须与训练数据一字不差（export_color_lerobot.py:365）：
            eef_9d(9) + gripper(1) + joint(6)
        eef_9d = pos(3) + rot6d 的**前两行** R[:2,:].flatten()

        === 为什么是 eef9 在前 ===
        2026-11-13 导出重排过：旧布局 joint(6)+eef_9d(9)+gripper(1) 已废弃，
        v36 训练数据用的是 eef9+grip+joint。顺序错了不会报错，模型会把关节角
        （~2.35 rad）当成 eef 的 z 坐标，输出直接飞到 2 m 外（实测 a0 z=2.44）。
        """
        joint = np.asarray(self.state["joint"], float).reshape(6)
        R = np.asarray(self.state["eef_rot"], float).reshape(3, 3)
        rot6 = R[:2, :].reshape(-1)          # 前两行 = 6 个数
        g = float(self.state["gripper"])
        eef9 = np.concatenate([self.eef_pos, rot6])
        return np.concatenate([eef9, [g], joint]).astype(np.float32)


class Policy:
    kind = "base"
    # 是否支持「直接给关节链」的快路径（只有 oracle 支持）。
    # ManipulateSkill 靠这个标志决定走哪条执行路径，不靠 kind 字符串 ——
    # 否则 oracle 策略的 infer() 会变成永远不会被调用的死代码。
    direct_joint = False

    def __init__(self):
        pass

    def infer(self, obs):
        """返回 (chunk Nx10 | None, info dict)。"""
        raise NotImplementedError

    def reset(self):
        pass


def make_policy(kind=None, **kw):
    kind = kind or POLICY["kind"]
    if kind == "oracle":
        from .oracle import OraclePolicy
        return OraclePolicy(**kw)
    if kind == "gr00t":
        from .gr00t_client import Gr00tPolicy
        cfg = POLICY.get("gr00t", {})
        return Gr00tPolicy(
            host=kw.pop("host", cfg.get("host", "127.0.0.1")),
            port=kw.pop("port", cfg.get("port", 5555)),
            chunk_size=kw.pop("chunk_size", cfg.get("chunk_size", 40)),
            execute_steps=kw.pop("execute_steps",
                                 cfg.get("execute_steps", 15)),
            **kw)
    if kind == "local":
        from .local_client import LocalGr00tPolicy
        cfg = POLICY.get("local", {})
        ckpt = kw.pop("ckpt", cfg.get("ckpt"))
        if not ckpt:
            raise ValueError("policy.kind=local 需要 ckpt（--ckpt 或 "
                             "deploy.yaml: policy.local.ckpt）")
        return LocalGr00tPolicy(
            ckpt=ckpt,
            chunk_size=kw.pop("chunk_size", cfg.get("chunk_size", 40)),
            execute_steps=kw.pop("execute_steps",
                                 cfg.get("execute_steps", 40)),
            device=kw.pop("device", cfg.get("device", "cuda")),
            **kw)
    raise ValueError("未知 policy.kind=%r（可选 oracle / gr00t / local）" % kind)
