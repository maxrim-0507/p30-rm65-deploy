# -*- coding: utf-8 -*-
"""机械臂预设位姿：同一个「附近」在不同阶段要用两套姿态。

=== 为什么需要两个预设 ===
FWD2 是 VLA 训练数据的首帧（arm_cam 朝正前方），交班时必须回去，
否则模型看到的是分布外画面。但 FWD2 在**远距离**只有 2–4° 的可见
扇区（实测），搜索阶段用它等于大部分朝向都看不见台面。

SEARCH_Q 把臂抬高、相机下俯（pitch −47.7°，cam_z 1.05），在 1.0–2.3 m
的远点上可见扇区 15–71°，才适合原地 360° 扫描。找到目标、泊车到位后
再 TUCK 回 FWD2 交班。

=== FWD2 必须与 handoff.make_fwd2_preset 的缺省值一致 ===
那边是执行端（含绕圈选最近分支），这里只是常量；selftest_search 会
用 inspect 比对两处，改一处忘了另一处会立刻失败。
"""
import numpy as np

# 搜索姿态（度）：肩抬到下视，腕保持与 FWD2 相同
SEARCH_Q_DEG = (0.0, 53.9, 60.0, 0.0, -16.2, -90.0)
# 交班姿态（度）：v36 训练数据首帧，arm_cam 朝前
FWD2_Q_DEG = (0.0, -13.6, 134.9, 0.0, -16.2, -90.0)

# 搜索角速度（rad/s）。底盘实测效率 ~76%：指令 0.8 → 实际 ~0.61。
# 不要按理论值算扫描时间。
SEARCH_WZ = 0.8
# 单圈判据：2π + 余量。用**实测 yaw 累积**，不要用指令积分。
SEARCH_MAX_SWEEP = 2.0 * np.pi + 0.4


# RM65 关节限位（度），与 p30_rm65_scene.xml 的 jnt_range 一致。
# joint1/4/6 是多圈（±359.8°），2/3/5 较窄。绕圈选等价角时必须逐关节
# 落在限位内：实测 joint6 从 −323.5° 起、预设 −90° 时，朴素的「最近
# 等价角」会选到 −450°，超出限位，位置伺服只能以 ~10°/s 向限位爬，
# 4.5 s 后还差 95° —— 表现为「抬臂失败：残差 94.7°」。
JOINT_LIMITS_DEG = ((-177.6, 177.6), (-129.9, 129.9), (-134.9, 134.9),
                    (-177.6, 177.6), (-127.9, 127.9), (-359.8, 359.8))


def wrap_nearest(q_deg, q_seed, limits_deg=None):
    """把角度预设换成离当前角最近的等价角（弧度）。

    RM65 的 joint1/4/6 是多圈的。预设 −90° 而当前 265° 时，直接下发
    要倒转 355°，真机上又多慢又会把 arm_cam 扫过整圈。

    === 逐关节选分支，且必须落在限位内 ===
    等价角是每个关节各自 +2πk 的关系，不能整组一起平移（那会把
    joint2/3/5 推到它们狭窄限位之外）。所以逐关节在限位内挑离 seed
    最近的候选；若该关节没有任何候选在限位内（例如 seed 本身已越界），
    退化为全候选里最近的，交给控制器截断。
    """
    q = np.radians(np.asarray(q_deg, dtype=float).reshape(6))
    seed = np.asarray(q_seed, dtype=float).reshape(6)
    lim = np.radians(np.asarray(limits_deg if limits_deg is not None
                                else JOINT_LIMITS_DEG, float)).reshape(6, 2)
    out = np.empty(6)
    for i in range(6):
        cand = q[i] + 2.0 * np.pi * np.arange(-2, 3)
        inside = (cand >= lim[i, 0] - 1e-9) & (cand <= lim[i, 1] + 1e-9)
        if inside.any():
            cand = cand[inside]
        out[i] = cand[np.argmin(np.abs(cand - seed[i]))]
    return out
