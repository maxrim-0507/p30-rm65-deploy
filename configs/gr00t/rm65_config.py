# -*- coding: utf-8 -*-
"""RM65（P30W 移动底盘 + RM65 + Dex1-1）的 GR00T N1.7 modality 配置（v2，路线甲）。

=== 这是什么 ===
GR00T 的「自定义机器人」入口：`EmbodimentTag.NEW_EMBODIMENT` 只有在
微调进程里被 register 过才会有 modality 配置，所以必须显式给：
    --modality-config-path configs/gr00t/rm65_config.py
    --embodiment-tag NEW_EMBODIMENT
注册在进程内是全局的，**一个进程只能注册一个 NEW_EMBODIMENT**。

实现参照官方 examples/SO100/so100_config.py 与 G1 的
`unitree_g1_full_body_with_waist_height_nav_cmd`（底盘键的写法）。
**取代** `simulation/collect/rm65_config.py`（旧版：cam_top / 关节动作 / 7D）。
v1（10D，底盘不进 VLA）的历史见 configs/vla_spec.yaml 的
`A_eef_9d_absolute` 段；本 v2 为路线甲（底盘进 VLA）的契约。

=== 数据契约（必须与数据集 meta/modality.json 逐字一致）===
state  float32[16] = single_arm(6) + eef_9d(9) + gripper_position(1)
action float32[13] = eef_9d(9) **绝对** + gripper_position(1) + base_velocity(3)
video  front_cam 320x240 + arm_cam 320x240，30 fps
language annotation.human.task_description

eef_9d 口径（**2026-11-09 校准**，见 configs/vla_spec.yaml）：
    pos   = 世界 xpos - ARM_BASE   # **xyz 三个分量都相对臂座**
    rot6d = 世界系 R[:2, :]
抓取帧实测 z=-0.118 = 0.592-0.710，坐实 z 也是相对量（早前本文写作
「世界高度」有误，采集代码从第一天减的就是全三维 ARM_BASE）。
底盘移动时臂座跟着动，所以「相对臂座」逐帧自洽 —— 执行端也必须逐帧
用当帧臂座解释（deployment/hal 的 actuate_pose 已是现取底盘语义）。

base_velocity 口径（照抄 G1 navigate_command，decoupled_wbc/data/utils.py:131）：
    [vx, vy, wz]，**机体系**，m/s、m/s、rad/s。速度指令不是目标位姿；
    缺键/超时由 L3 注入 [0,0,0]（G1 的 DEFAULT_NAV_CMD 同款）。

=== 为什么 eef_9d 声明 ABSOLUTE（v1 曾是 RELATIVE，底盘移动后必须改）===
v1 的 RELATIVE 依赖 processor 做 `T0^-1 @ Tt` 相对化、推理端再还原，
参考系取 `state[state_key][-1]`（观测末帧的臂座系，
state_action_processor.py:369-390）。底盘静止时臂座系恒定，往返自洽；
**底盘移动时 t0 帧与 tt 帧的臂座系之间隔着底盘位移，这个乘法物理上不成立**。
已读源码确认（2026-11-06，本地 /tmp/gr00t_src）：
  apply  端 :369  与 unapply 端 :478 都是
  `if rep == RELATIVE and use_relative_action:` 才转换 ——
  rep=ABSOLUTE 两端**完全直通**，模型出什么执行什么，无隐式换算。
注意：官方全部 ABSOLUTE 先例都是 NON_EEF（夹爪/腰/nav/G1 底盘键），
ABSOLUTE+EEF 无现成模板；处理器端只是透传所以机制安全，
学习效果待仿真闭环（阶段 2）实证。

=== 改了配置必须重算统计量 ===
`gr00t/data/stats.py:423` 只对 RELATIVE 键要求 relative_action 统计量；
ABSOLUTE 键只要普通统计。但 action 键集合/维度/rep 指纹变了，
`relative_stats.json` / 统计量都要按 v2 重新生成，不能复用 v1 的。
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ACTION_HORIZON = 40          # 与 base ckpt 的 action_horizon 一致

rm65_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front_cam", "arm_cam"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "eef_9d", "gripper_position"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["eef_9d", "gripper_position", "base_velocity"],
        action_configs=[
            # 末端位姿：**绝对直通**（理由见文件头「为什么 ABSOLUTE」）。
            # 执行端逐帧用当帧臂座解释；不做 processor 相对化。
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            # 夹爪：绝对目标开度。二值开合用绝对比增量稳（SO100 同款）。
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # 底盘速度：照抄 G1 navigate_command（ABSOLUTE / NON_EEF / DEFAULT）。
            # 机体系连续速度；L3 侧缺键注入 0 + 看门狗（G1 三层安全同款）。
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(rm65_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
