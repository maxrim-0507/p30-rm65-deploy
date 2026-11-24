# -*- coding: utf-8 -*-
"""P30W + RM65 可部署软件栈。

分层（用户口径，与 docs/cargobot/04 一致）：
  L1 任务层   task/     语言 → 目标 → 技能计划（1–5 Hz）
  L2 技能层   skills/   技能状态机 + IK + 策略推理（10–30 Hz）
  L3 控制层   control/  关节/夹爪指令 + 安全限幅（100–500 Hz）

横切：
  hal/    硬件抽象：仿真后端与真机后端同接口
  bus/    ZMQ 消息总线：跨进程、跨机（Thor ↔ Orin Nano）
  perception/  颜色 grounding（像素 → 台面 3D 点）
  policy/      策略实现（Oracle 真值 / τ0-VLA / GR00T 备选客户端）
"""
