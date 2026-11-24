# -*- coding: utf-8 -*-
"""16D（vla_spec）↔ 40D（τ0 unified）的转换，纯 numpy、无 torch。

=== 为什么单独一个文件 ===
转换规则错一个转置，训练和推理都**不报错**，只是学到一个错的姿态表示
（见 docs/tau0_Thor部署验证_2026-11-05.md §6.4）。所以把它从模型代码里
拆出来，可以单独跑数值往返测试，也方便未来导出数据时复用同一份实现。

=== 两个致命差异（务必遵守）===
1. rot6d 的取法相反
     我们的 eef_9d 后 6 位 = R[:2, :] 展平        （前两**行**）
     τ0 的 40D           = [R[:,0], R[:,1]]       （前两**列**）
   行↔列不转换不会抛异常，只会让模型看到转置后的旋转。
2. 相对动作的坐标系
     τ0 的 EEF 相对增量在 **EEF 自身系**：
         delta_xyz = R_state.T @ (p_action - p_state)
         rel_rot   = R_state.T @ R_action
     关节块是简单差值，gripper/waist/chassis 保持绝对值（不做相对化）。

=== 坐标系的归属 ===
40D 只是一个**容器**。τ0 预训练权重里的 left_eef 是它自己机器人的
基座系；我们这里统一用**臂基座系**（与 get_state()/IK/工作空间一致）。
将来用我们的数据训练时，导出端也必须用同一个定义，否则训练/推理
静默错位。这一点在 docs 里单独写明了，没有跟随预训练权重的口径。

=== mask ===
τ0 的 relativize 需要 mask 指明哪些槽位是激活的。我们的 embodiment 只用
左臂：0:9（EEF）、18（夹爪）、24:30（6 个关节，左臂槽位宽 8，后 2 维补 0）。
"""
import numpy as np

UNIFIED_DIM = 40
LEFT_EEF = slice(0, 9)
LEFT_GRIP = 18
LEFT_ARM = slice(24, 32)
LEFT_ARM6 = slice(24, 30)

# 我们的 16D 布局
S_JOINT = slice(0, 6)
S_EEF_POS = slice(6, 9)
S_EEF_ROT6 = slice(9, 15)
S_GRIP = 15


# ---------------------------------------------------------------- rot6d
def matrix_to_rot6_cols(R):
    """3×3 → τ0 的 rot6d（前两**列**）。R 可以是 (...,3,3)。"""
    R = np.asarray(R, float)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6_cols_to_matrix(r6):
    """τ0 的 rot6d（两列）→ 3×3，Gram-Schmidt 与 τ0 实现一致。"""
    r6 = np.asarray(r6, float)
    a1, a2 = r6[..., :3], r6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def our_rot6_to_matrix(r6_rows):
    """我们的 eef_9d 后 6 位（前两**行**）→ 3×3。"""
    r6_rows = np.asarray(r6_rows, float).reshape(2, 3)
    r0, r1 = r6_rows[0], r6_rows[1]
    r2 = np.cross(r0, r1)
    return np.stack([r0, r1, r2])


def matrix_to_our_rot6(R):
    """3×3 → 我们的 eef_9d 后 6 位（前两**行**）。"""
    return np.asarray(R, float)[:2, :].reshape(-1)


def tau6_to_our6(r6_cols):
    return matrix_to_our_rot6(rot6_cols_to_matrix(r6_cols))


def our6_to_tau6(r6_rows):
    return matrix_to_rot6_cols(our_rot6_to_matrix(r6_rows))


# ---------------------------------------------------------------- mask
def active_mask():
    """**state** 的 40D 激活槽位。返回 (40,) float32。

    包含关节 24:30：state 里关节是有信息的输入（虽然是命令值），
    只是不用于执行。
    """
    m = np.zeros(UNIFIED_DIM, np.float32)
    m[LEFT_EEF] = 1.0
    m[LEFT_GRIP] = 1.0
    m[LEFT_ARM6] = 1.0
    return m


def action_mask():
    """**action** 的 40D 激活槽位：只有 EEF 0:9 + 夹爪 18。

    === 为什么不能复用 active_mask() ===
    relativize/unrelativize 按 mask 决定哪些槽位做增量。我们的动作里
    24:30 是 0（不输出关节），若用 state 的 mask，unrelativize 会把
    state 的关节角加到这些 0 上 —— 现在 unpack 不读这些槽所以看不出来，
    但一旦有人开始读，就会拿到「凭空出现」的关节目标。
    """
    m = np.zeros(UNIFIED_DIM, np.float32)
    m[LEFT_EEF] = 1.0
    m[LEFT_GRIP] = 1.0
    return m


# ---------------------------------------------------------------- 打包
def pack_state16(state16):
    """我们的 16D → τ0 40D（绝对量，未归一化）。

    state16 = joint(6) + eef_9d(9) + gripper(1)
    """
    s = np.asarray(state16, float).reshape(16)
    out = np.zeros(UNIFIED_DIM, np.float32)
    out[S_JOINT] = s[S_JOINT]
    out[0:3] = s[S_EEF_POS]
    out[3:9] = our6_to_tau6(s[S_EEF_ROT6].reshape(2, 3))
    out[LEFT_GRIP] = s[S_GRIP]
    out[LEFT_ARM6] = s[S_JOINT]
    return out


def pack_action10(action10):
    """我们的 10D 绝对动作 → τ0 40D（绝对量，未归一化）。

    action10 = pos(3) + rot6 前两行(6) + gripper(1)
    只有 0:9 和 18 被填；其余保持 0（配合 action_mask()）。

    === 为什么要经过 matrix 再取列 ===
    rot6d 行↔列不能直接重排：我们的前两行在数值上是 R 的行，
    τ0 要的是 R 的列。先重建 3×3（叉乘补第三行）再取前两列，
    与 unpack_action40 用同一对函数，保证互逆。
    """
    a = np.asarray(action10, float)
    squeeze = a.ndim == 1
    a = a.reshape(-1, 10)
    out = np.zeros((a.shape[0], UNIFIED_DIM), np.float32)
    out[:, 0:3] = a[:, 0:3]
    for i in range(a.shape[0]):
        out[i, 3:9] = our6_to_tau6(a[i, 3:9])
    out[:, LEFT_GRIP] = np.clip(a[:, 9], 0.0, 1.0)
    return out[0] if squeeze else out


def unpack_action40(a40):
    """τ0 40D（绝对量）→ (N,10)：pos(3) + rot6 前两行(6) + gripper(1)。"""
    a = np.asarray(a40, float)
    if a.ndim == 1:
        a = a[None]
    n = a.shape[0]
    out = np.zeros((n, 10), np.float32)
    out[:, 0:3] = a[:, 0:3]
    for i in range(n):
        out[i, 3:9] = tau6_to_our6(a[i, 3:9])
    out[:, 9] = np.clip(a[:, LEFT_GRIP], 0.0, 1.0)
    return out


# ---------------------------------------------------------------- 相对化
def relativize(action, state, mask=None):
    """绝对 40D 动作 + 状态参考 → 相对动作（与 τ0 训练目标同语义）。

    action 可带 batch 维；state/mask 必须 1D（τ0 的实现用 state[blk]）。
    """
    a = np.array(action, dtype=float, copy=True)
    s = np.asarray(state, float).reshape(UNIFIED_DIM)
    m = (active_mask() if mask is None
         else np.asarray(mask, float).reshape(UNIFIED_DIM))

    # EEF：body-frame 增量 + 相对旋转
    if m[0] and m[8]:
        ps, pa = s[0:3], a[..., 0:3]
        Rs = rot6_cols_to_matrix(s[3:9])
        Ra = rot6_cols_to_matrix(a[..., 3:9])
        a[..., 0:3] = np.einsum("ji,...j->...i", Rs, pa - ps)
        a[..., 3:9] = matrix_to_rot6_cols(np.einsum("ji,...jk->...ik", Rs, Ra))

    # 关节：简单差值（只对激活槽位）
    for blk in (slice(24, 32), slice(32, 40)):
        idx = np.arange(blk.start, blk.stop)
        on = m[idx] > 0.5
        if on.any():
            a[..., idx[on]] = a[..., idx[on]] - s[idx[on]]

    # gripper / waist / chassis 保持绝对，不动
    return a.astype(np.float32)


def unrelativize(rel, state, mask=None):
    """相对动作 → 绝对 40D。与 relativize 严格互逆。"""
    a = np.array(rel, dtype=float, copy=True)
    s = np.asarray(state, float).reshape(UNIFIED_DIM)
    m = (active_mask() if mask is None
         else np.asarray(mask, float).reshape(UNIFIED_DIM))

    if m[0] and m[8]:
        ps, dp = s[0:3], a[..., 0:3]
        Rs = rot6_cols_to_matrix(s[3:9])
        Rd = rot6_cols_to_matrix(a[..., 3:9])
        a[..., 0:3] = ps + np.einsum("ij,...j->...i", Rs, dp)
        a[..., 3:9] = matrix_to_rot6_cols(np.einsum("ij,...jk->...ik", Rs, Rd))

    for blk in (slice(24, 32), slice(32, 40)):
        idx = np.arange(blk.start, blk.stop)
        on = m[idx] > 0.5
        if on.any():
            a[..., idx[on]] = a[..., idx[on]] + s[idx[on]]

    return a.astype(np.float32)


# ---------------------------------------------------------------- 归一化
class NormStats:
    """40D 逐维 mean/std。没有统计量时退化为恒等，并把 calibrated 标出来。

    === 为什么必须显式标 uncalibrated ===
    τ0 预训练 ckpt 不含 norm_stats（docs §6.2）。恒等归一化下模型输出
    不是任何有意义的动作。上层若把它当成正常动作执行，会看到「机械臂
    乱动」却没有任何报错 —— 比直接失败更难查。这里让 infer 的 info
    带上 calibrated=False，技能层据此拒跑（或只在链路自检里跑）。
    """

    def __init__(self, mean=None, std=None, source="", role="state"):
        self.mean = None if mean is None else np.asarray(mean, float)
        self.std = None if std is None else np.asarray(std, float)
        if self.std is not None:
            # 恒定维度（例如我们从不输出的关节槽）std=0，直接除会出 inf。
            self.std = np.where(np.abs(self.std) < 1e-8, 1.0, self.std)
        self.source = source
        self.role = role

    @property
    def calibrated(self):
        return self.mean is not None and self.std is not None

    @classmethod
    def load(cls, path, role="state"):
        """读 40D 统计量。

        === 为什么按 role 分开 ===
        τ0 的 state 与 action 分布完全不同（state 是位姿+关节，action 是
        增量），共用一套 mean/std 会把动作尺度压错。文件里优先找
        `{role}_mean` / `{role}_std`，没有才退回旧的 `mean` / `std`
        （便于旧文件继续用，但两者相同只是在没算准之前的临时状态）。
        """
        if not path:
            return cls(role=role)
        if path.endswith(".npz"):
            z = np.load(path)
            return cls(cls._pick(z, role, has=lambda k: k in z.files),
                       cls._pick(z, role, "std",
                                 has=lambda k: k in z.files),
                       source=path, role=role)
        if path.endswith(".json"):
            import json
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return cls(cls._pick(d, role, has=lambda k: k in d),
                       cls._pick(d, role, "std", has=lambda k: k in d),
                       source=path, role=role)
        raise ValueError("不支持的 norm_stats 格式：%s" % path)

    @staticmethod
    def _pick(d, role, suffix="mean", has=None):
        key = "%s_%s" % (role, suffix)
        if has(key):
            return d[key]
        return d[suffix] if has(suffix) else None

    def normalize(self, x):
        if not self.calibrated:
            return np.asarray(x, np.float32)
        return ((np.asarray(x, float) - self.mean) / self.std).astype(np.float32)

    def denormalize(self, x):
        if not self.calibrated:
            return np.asarray(x, np.float32)
        return (np.asarray(x, float) * self.std + self.mean).astype(np.float32)


# ---------------------------------------------------------------- 自检
def _self_test():
    rng = np.random.default_rng(0)

    # 1. rot6d：行 → 列 → 重建，必须逐位回到原旋转
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    ours = matrix_to_our_rot6(R)
    tau = our6_to_tau6(ours)
    assert np.allclose(rot6_cols_to_matrix(tau), R, atol=1e-7), "rot6d 往返失败"
    assert np.allclose(tau6_to_our6(tau), ours, atol=1e-7)

    # 2. action10 → 40D → action10
    a10 = np.array([0.46, 0.10, 0.58, *ours, 0.8], float)
    a40 = pack_action10(a10)
    assert not a40[9:18].any() and not a40[19:].any(), "只激活 0:9 和 18"
    back = unpack_action40(a40)[0]
    assert np.allclose(back, a10, atol=1e-6), (back, a10)

    # 3. 相对化往返（必须用 action_mask，不能用 active_mask）
    state16 = np.concatenate([[0.0] * 6, [0.40, 0.0, 0.55], ours, [1.0]])
    s40 = pack_state16(state16)
    rel = relativize(a40, s40, mask=action_mask())
    abs40 = unrelativize(rel, s40, mask=action_mask())
    assert np.allclose(abs40, a40, atol=1e-6), np.abs(abs40 - a40).max()
    # 未激活槽位必须保持 0，不能被 state 的关节「补」进去
    assert not abs40[24:40].any(), "action_mask 没挡住关节槽"

    # 4. 归一化统计量
    assert not NormStats().calibrated
    assert not NormStats(mean=np.zeros(40)).calibrated
    ns = NormStats(mean=np.zeros(40), std=np.zeros(40))
    assert ns.calibrated and np.all(ns.std == 1.0), "std=0 必须换成 1"

    print("  unified40 self-test OK  rot6d 往返 / action10↔40D / "
          "相对化互逆 / mask / norm_stats")


if __name__ == "__main__":
    import sys
    if "--self-test" not in sys.argv:
        print("用法：python3 -m deployment.policy.unified40 --self-test")
        sys.exit(2)
    _self_test()
