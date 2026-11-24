# -*- coding: utf-8 -*-
"""L1 任务层：指令解析 → 目标 grounding → 技能计划。

`TaskPlanner` 用懒加载（PEP 562）：它依赖 bus → zmq，而契约/解析器/
评测脚本（contract.py、eval_l1.py）不需要运行时栈。之前的急切 import
会让「只想跑一个纯解析自检」也要求装 pyzmq，把离线工具和部署环境绑死。
`from deployment.task import TaskPlanner` 的用法保持不变。
"""
from .grounding import Grounding
from .instruction import Intent, describe, parse

__all__ = ["Grounding", "Intent", "describe", "parse", "TaskPlanner"]


def __getattr__(name):
    if name == "TaskPlanner":
        from .planner import TaskPlanner
        return TaskPlanner
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
