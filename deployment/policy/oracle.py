# -*- coding: utf-8 -*-
"""Oracle 策略：用真值算出一条「接近 → 闭合」的 EEF 增量块。

=== 它不替代 IK，只是把 IK 的结论翻译成 VLA 的动作格式 ===
ManipulateSkill 的 oracle 路径直接走关节链（精度最高）。这个策略
存在的目的是**让 VLA 的接口在没模型时也能被完整调用一次**：
总线、msgpack、chunk 切分、相对增量累积、IK 重解、安全层 ——
全都跑一遍，这样接通真 GR00T 那天只剩换 IP 一件事。

=== 为什么输出相对增量而不是绝对位置 ===
vla_spec.yaml 的 action.choice 就是 A_relative_eef_delta。
服务端的 processor 负责把相对转绝对；客户端这里产出相对，
两端对「谁负责累积」必须只有一个答案 —— 见 docs §8.3 的警告
（重复累积会让动作翻倍）。
"""
import numpy as np

from .base import ACTION_DIM, Policy


class OraclePolicy(Policy):
    """kind = "oracle"（ManipulateSkill 见到这个 kind 会走关节链快路径）。"""

    direct_joint = True

    def __init__(self, chunk_size=16, execute_steps=8, step_m=0.004,
                 max_m_per_axis=0.01, **kw):
        super().__init__()
        self.kind = "oracle"
        self.chunk_size = int(chunk_size)
        self.execute_steps = int(execute_steps)
        self.step_m = float(step_m)          # 每步 4 mm，16 步 = 6.4 cm
        self.max_m_per_axis = float(max_m_per_axis)
        self._goal = None
        self._grip_goal = 0.0

    def reset(self):
        self._start = None

    def infer(self, obs):
        """朝目标点前进一步。

        obs 里没有目标点（PolicyObs 只有观测），所以 oracle 的用法是
        由调用方通过 `set_goal()` 预设目标 —— 这条路径只用于
        「验证 VLA 接口」的冒烟测试，真实训练数据仍走关节链。
        """
        if self._goal is None:
            return None, {"reason": "未设置目标（OraclePolicy 仅用于接口自检）"}

        cur = np.asarray(obs.eef_pos, float)
        delta = np.asarray(self._goal, float) - cur
        dist = float(np.linalg.norm(delta))
        if dist < 1e-4:
            return self._chunk(np.zeros(6), float(self._grip_goal)), {"reason": "已到位"}

        direction = delta / dist
        step = min(self.step_m * self.chunk_size, dist)
        per = direction * (step / self.chunk_size)
        per = np.clip(per, -self.max_m_per_axis, self.max_m_per_axis)

        # 到点后剩余的步数保持不动，夹爪按预设开度
        n_move = int(np.ceil(step / max(np.linalg.norm(per), 1e-9)))
        n_move = max(1, min(self.chunk_size, n_move))
        return self._chunk(per, float(self._grip_goal), n_move), {
            "dist": dist, "step": step, "n_move": n_move}

    def set_goal(self, pos, grip=0.0):
        self._goal = np.asarray(pos, float).reshape(3)
        self._grip_goal = float(grip)

    def _chunk(self, per_step, grip, n_move=None):
        n = self.chunk_size
        n_move = n if n_move is None else int(n_move)
        out = np.zeros((n, ACTION_DIM), np.float32)
        out[:n_move, :3] = per_step          # 只有位置增量，姿态不动
        out[:, 6] = grip                     # 夹爪绝对开度
        return out, {"chunk": out.shape}
