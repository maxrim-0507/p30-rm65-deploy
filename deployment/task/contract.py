# -*- coding: utf-8 -*-
"""L1 → L2 任务契约（冻结版 v1）。

=== 为什么要一个显式契约 ===
VLM 的输出、正则可解析的结果、仿真真值，三者字段和坐标系必须完全一致，
否则 L2 拿到的是"看起来能用、实际偏了 3 cm"的数据，而且不会报错。

=== 字段职责（不要混用）===
  instruction  真正喂给 VLA 的语言条件。必须与训练数据的模板同源。
  action       目前只有 "grasp"。
  color        目标颜色，给 Verify 判"夹的是不是指定颜色"用。
  bbox_2d      原图像素坐标 [x1,y1,x2,y2]。用于调试和可选的视觉提示。
  target_xy    世界系 (x, y)。只用于 L1 的可达性预检和下发 L2；
               **不要塞进 VLA 的 state**（模型没这样训练过）。
  target_yaw   方块 yaw，单目 bbox 无法可靠恢复时为 None。
  confidence   0~1，低置信度时 L1 应拒绝下发而不是猜。

=== 坐标系 ===
target_xy 是世界系（与 Grounding / planner 现有一致）。
L2 的 Pregrasp 会用 base_pose 转到臂基座系；VLA 的 eef_9d 则在臂基座系。
两个系不要混。
"""
import numpy as np

CONTRACT_VERSION = 1

COLORS = ("red", "green", "blue", "yellow")


class TaskSpec:
    """L1 输出的结构化任务。字段语义见模块 docstring。"""

    __slots__ = ("instruction", "action", "color", "bbox_2d", "target_xy",
                 "target_pos", "target_yaw", "confidence", "source", "reason")

    def __init__(self, instruction="", action=None, color=None, bbox_2d=None,
                 target_xy=None, target_pos=None, target_yaw=None,
                 confidence=0.0, source="", reason=""):
        self.instruction = str(instruction or "")
        self.action = action
        self.color = color
        self.bbox_2d = (None if bbox_2d is None
                        else [float(v) for v in bbox_2d])
        self.target_xy = (None if target_xy is None
                          else (float(target_xy[0]), float(target_xy[1])))
        self.target_pos = (None if target_pos is None
                           else np.asarray(target_pos, float).reshape(3))
        self.target_yaw = (None if target_yaw is None else float(target_yaw))
        self.confidence = float(confidence)
        self.source = str(source or "")
        self.reason = str(reason or "")

    @property
    def ok(self):
        return (self.action is not None and self.color is not None
                and self.target_xy is not None)

    def validate(self):
        """返回问题列表；空列表表示可下发。"""
        errs = []
        if self.action is None:
            errs.append("action 为空")
        elif self.action != "grasp":
            errs.append("不支持的动作 %r" % self.action)
        if self.color is None:
            errs.append("color 为空")
        elif self.color not in COLORS:
            errs.append("未知颜色 %r" % self.color)
        if self.target_xy is None:
            errs.append("target_xy 为空")
        if self.bbox_2d is not None and len(self.bbox_2d) != 4:
            errs.append("bbox_2d 维度不是 4")
        if not (0.0 <= self.confidence <= 1.0):
            errs.append("confidence 越界")
        return errs

    def to_dict(self):
        return {
            "contract_version": CONTRACT_VERSION,
            "instruction": self.instruction,
            "action": self.action,
            "color": self.color,
            "bbox_2d": self.bbox_2d,
            "target_xy": (None if self.target_xy is None
                          else list(self.target_xy)),
            "target_pos": (None if self.target_pos is None
                           else self.target_pos.tolist()),
            "target_yaw": self.target_yaw,
            "confidence": self.confidence,
            "source": self.source,
            "reason": self.reason,
        }

    def to_task_dict(self):
        """下发给 L2 SkillManager 的字段（只保留 L2 需要的）。"""
        return {
            "instruction": self.instruction,
            "target_color": self.color,
            "target_xy": self.target_xy,
            "target_yaw": self.target_yaw,
        }

    def __repr__(self):
        xy = "None" if self.target_xy is None else (
            "(%.3f, %.3f)" % self.target_xy)
        return ("TaskSpec(action=%s, color=%s, xy=%s, src=%s, conf=%.2f)"
                % (self.action, self.color, xy, self.source,
                   self.confidence))
