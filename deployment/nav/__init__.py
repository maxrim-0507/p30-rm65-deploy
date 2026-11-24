# -*- coding: utf-8 -*-
"""导航↔VLA 协同层：底盘速度仲裁 + 交班监督。

分层位置在 L2（技能层）之上、HAL 之下：

    L1 task/     语言 → 目标 → 分工计划
    L2 nav/      ★ 谁在写底盘（本包）+ 交班时机
    L2 skills/   抓取技能
    L3 control/  唯一真正写硬件的一层

=== 为什么协同层不能放进 VLA 策略里 ===
「导航到桌前 → 抓取」天然是两个执行源。让 VLA 自己学会「先别动、
等导航停」，等于要求模型同时建模底盘运动学和厂商导航的行为，
训练数据里没有这部分，且换一台导航栈就失效。把它做成显式的所有权
仲裁，模型只出自己那段的速度，切换由代码保证。

=== 与 docs 的口径 ===
docs/cargobot/04 §4 定义的「Phase 2 底盘协同」；本包是它的实现。
"""
from .mux import BaseVelocityMux, Owner
from .proxy import MuxBackend
from .handoff import (HandoffSupervisor, Phase, make_armcam_visibility_fn,
                      make_fwd2_preset, wrap_fwd2)
from .percept_pipeline import NavPhase, NavResult, PerceptNav

__all__ = [
    "BaseVelocityMux", "Owner", "MuxBackend",
    "HandoffSupervisor", "Phase",
    "make_armcam_visibility_fn", "make_fwd2_preset", "wrap_fwd2",
    "PerceptNav", "NavPhase", "NavResult",
]
