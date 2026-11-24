# -*- coding: utf-8 -*-
"""指令解析：自然语言 → {目标颜色}。**只做兜底，不做指代接地。**

=== 规则的定位（2026-11-05 调整）===
主路径是 VLM：`grounding.locate_task()` 让 Qwen2.5-VL 在同一次调用里
同时输出 action / color / bbox_2d，颜色由模型基于图像判断，零额外延迟。
这里的中英文词表只在两种情况兜底：
  1. 模型漏了 color 字段，或只回了 bbox；
  2. 仿真真值路径要按颜色选目标（truth 没有语言输入）。
「红色方块左边那个」「第二排的蓝块」这类指代必须交给模型，词表做不到，
也不该在这里假装能做到。接口保持 `parse(text) -> Intent`。
"""
from ..spec import COLOR_ZH


class Intent:
    def __init__(self, action, color=None, raw="", confidence=1.0):
        self.action = action          # "grasp"
        self.color = color            # "red" / ... / None
        self.raw = raw
        self.confidence = float(confidence)

    def __repr__(self):
        return "Intent(action=%s, color=%s)" % (self.action, self.color)


# 中英文两套词表：采集数据里有 variants（英文模板），所以推理端也要认
COLOR_ALIASES = {
    "red": ["红", "紅色", "red"],
    "green": ["绿", "綠", "green"],
    "blue": ["蓝", "藍", "blue"],
    "yellow": ["黄", "黃", "yellow"],
}

GRASP_WORDS = ["抓", "拿", "取", "pick", "grab", "grasp", "take"]


def parse(text):
    """返回 Intent。抽不出颜色时 color=None、confidence=0，由 L1 决定重问。"""
    s = (text or "").strip().lower()
    if not s:
        return Intent(None, None, text, 0.0)

    is_grasp = any(w in s for w in GRASP_WORDS)
    # 「把红色方块拿起来」这类句式里「拿」在尾部，上面的词表已覆盖

    hits = []
    for color, alts in COLOR_ALIASES.items():
        for a in alts:
            if a in s:
                hits.append((color, len(a)))
                break

    if len(hits) == 1:
        return Intent("grasp" if is_grasp else None, hits[0][0], text, 1.0)
    if len(hits) > 1:
        # 多个颜色同时出现：取最长的匹配（「黄色」优先于「黄」的误命中），
        # 长度相同时无法判断，返回低置信度让 L1 追问。
        hits.sort(key=lambda h: -h[1])
        if hits[0][1] > hits[1][1]:
            return Intent("grasp", hits[0][0], text, 0.8)
        return Intent("grasp", None, text, 0.2)

    # 中文里可能写成「抓取方块」而没指定颜色
    return Intent("grasp" if is_grasp else None, None, text, 0.0)


def describe(intent):
    if intent.color is None:
        return "未识别到目标颜色"
    return "抓取%s的方块" % COLOR_ZH[intent.color]
