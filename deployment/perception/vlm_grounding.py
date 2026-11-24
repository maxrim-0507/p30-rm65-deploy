# -*- coding: utf-8 -*-
"""VLM grounding：Qwen2.5-VL 出 2D bbox → 已知台面的 3D 点。

=== 为什么是 VLM 而不是颜色阈值 ===
vla_spec / 技术方案里 L1 的地面定位本来就用 VLM（Qwen2.5-VL 级，不微调，
prompt + 场景图）。颜色阈值在受控仿真里能用，但它把「指令理解」退化成了
「色相查表」：换光照、换相机、加干扰物就失效，也无法理解「红色方块左边那个」。
这里按方案改成 open-vocabulary 定位，语言指令直接进 prompt。

=== 用的是官方 grounding 协议，不自创 ===
Qwen2.5-VL 有两套 2D grounding 输出，取决于模型版本：

  微调版（官方 cookbook 2d_grounding 训练后）
    prompt : "Outline the position of the {object}."
    output : <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    坐标   : 0~1000 相对值

  Instruct 版（本项目用的，未微调）
    output : [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]
    坐标   : 输入图像的绝对像素（见官方 README 的 Grounding 段）

两条都支持。实测 Instruct 版必须显式要求 JSON，否则会写成散文
（"The red cube is located towards the right side of the image..."）。
参考 transformers 官方的 Qwen2.5-VL 用法（AutoProcessor +
apply_chat_template + process_vision_info + generate），不手写 chat
template。

=== 像素 → 3D 仍复用已标定的几何 ===
bbox 中心像素 → CameraModel.ray/hit_plane → 台面（或方块顶面）交点。
反投影误差已有实测（顶面 green 10.2 / blue 2.0 / yellow 6.6 mm），
VLM 只替换「是哪个像素」，不改变几何链路。
"""
import re

import numpy as np

# 官方 2d_grounding cookbook 的 prompt 是 "Outline the position of the {obj}."，
# 但那只对**微调过 grounding 的** Qwen2.5-VL 成立。Instruct 版实测：
#   · 纯 "Outline the position of ..." 常把结果写成一句散文（"The red cube is
#     located towards the right side of the image, ..."），解析不到框
#   · 明确要求 JSON 时 4/4 稳定，平均 6.9 mm，最大 13.4 mm
# 所以这里是**官方 grounding 的语义 + 官方 README Grounding 段的 JSON 格式**，
# 不是自创格式：Qwen2.5-VL README 的 Grounding 示例输出就是
#   [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]
# 实测（320×240 头部相机，4 色 × 3 次）：12/12 成功，均值 8.1 mm，中位 7.5 mm。

# 英文（color 路径）
_JSON_SPEC_EN = (' Output its bounding box as JSON, and nothing else: '
                 '[{{"bbox_2d": [x1, y1, x2, y2], "label": "{obj}"}}]')
PROMPT_TEMPLATE = "Locate the {obj} in this image." + _JSON_SPEC_EN

# 中文（指令原文路径）：只做格式约束，不解析语义。
# 结尾不再补句号 —— 调用侧已经保证指令以「。」结尾，这里直接接句子，
# 否则会出现「抓取红色方块。。只输出」这种双句号。
_JSON_SPEC_ZH = "只输出 JSON，不要其他文字，格式：[{{\"bbox_2d\": [x1, y1, x2, y2]}}]"

# === 结构化任务 prompt（L1→L2 契约 v1）===
# 一次调用同时拿到 action / color / bbox_2d，省掉第二次 VLM 往返。
# 键名沿用官方 README 的 bbox_2d，所以 parse_boxes 的像素/归一化判据
# 可以原样复用；只额外解析两个字符串字段。
# 颜色枚举写进 prompt 是为了收敛输出格式，不是让模型做「颜色分类」——
# 颜色仍然由模型基于图像判断，只是限制在四个合法值上。
_TASK_SPEC_EN = (
    ' Then output one JSON object and nothing else:'
    ' {"action": "grasp", "color": "<red|green|blue|yellow>",'
    ' "bbox_2d": [x1, y1, x2, y2]}')
_TASK_SPEC_ZH = (
    "然后只输出一个 JSON，不要其他文字，格式："
    "{\"action\": \"grasp\", \"color\": \"<red|green|blue|yellow>\", "
    "\"bbox_2d\": [x1, y1, x2, y2]}")

# 颜色 → 英文物体描述（Qwen 的 grounding 词汇以英文最稳）
COLOR_OBJ = {
    "red": "red cube",
    "green": "green cube",
    "blue": "blue cube",
    "yellow": "yellow cube",
}


def _has_cjk(s):
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)

# Qwen2.5-VL-Instruct 实际输出的是 JSON：{"bbox_2d": [x1,y1,x2,y2], "label": ...}
# 这是官方 README「Grounding」段给出的格式，坐标是**输入图像的绝对像素**。
_JSON_BOX_RE = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*'
    r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)
# 微调版 grounding 模型（2d_grounding cookbook 训练后）吐特殊 token，坐标 0~1000
_BOX_RE = re.compile(
    r"<\|box_start\|>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
    r"\s*,\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*<\|box_end\|>"
)
# 兜底：模型偶尔漏掉特殊 token，只剩 "(x1,y1),(x2,y2)"
_PLAIN_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
)


def parse_boxes(text):
    """从模型输出抽 bbox。返回 (boxes, kind)。

    kind：
      "px"    JSON bbox_2d —— 绝对像素（Instruct 版的实际输出）
      "norm"  <|box_start|> —— 0~1000 相对坐标（grounding 微调版）

    === 为什么必须区分，不能靠数值猜 ===
    320×240 的小图上，「0~1000 归一化」和「绝对像素」的数值区间重叠
    （都在 0~320 内），按 max>1000 判断会把像素值再除 1000，得到
    0.3 px 的 bbox —— 不报错，只是永远投影到台面同一个角。格式驱动
    的判据才可靠。
    """
    js = [[float(v) for v in m.groups()] for m in _JSON_BOX_RE.finditer(text)]
    if js:
        return js, "px"
    tok = [[float(v) for v in m.groups()] for m in _BOX_RE.finditer(text)]
    if tok:
        return tok, "norm"
    plain = [[float(v) for v in m.groups()] for m in _PLAIN_RE.finditer(text)]
    return plain, "norm"


_ACTION_RE = re.compile(r'"action"\s*:\s*"([^"]*)"')
_COLOR_RE = re.compile(r'"color"\s*:\s*"([^"]*)"')

# 动作词归一化：模型偶尔写成 "pick" / "grasp the red cube"，只留动词。
_ACTION_WORDS = ("grasp", "pick", "grab", "抓", "拿")


# prompt 里只列英文枚举，但中文指令下模型偶尔回「蓝色」。
# 语义没错、只是格式偏离，这里收下来并在 confidence 上体现，不当作失败。
_COLOR_WORDS = {
    "red": ("red", "红"),
    "green": ("green", "绿"),
    "blue": ("blue", "蓝"),
    "yellow": ("yellow", "黄"),
}


def norm_color(s):
    """把模型给的颜色字段收敛到四个合法值；认不出返回 None。"""
    if not s:
        return None
    t = str(s).strip().lower()
    for c, words in _COLOR_WORDS.items():
        if any(w in t for w in words):
            return c
    return None


def norm_action(s):
    """把动作字段收敛到契约里的动作名（目前只有 grasp）。"""
    if not s:
        return None
    t = str(s).strip().lower()
    for w in _ACTION_WORDS:
        if w in t:
            return "grasp"
    return None


def parse_task(text):
    """从一次结构化输出里抽 action / color / boxes。

    返回 {"action", "color", "boxes", "kind"}；缺失字段为 None，不抛异常。
    bbox 仍走 parse_boxes，像素/归一化判据与纯 grounding 路径完全一致。
    """
    boxes, kind = parse_boxes(text)
    m_a = _ACTION_RE.search(text)
    m_c = _COLOR_RE.search(text)
    return {
        "action": norm_action(m_a.group(1)) if m_a else None,
        "color": norm_color(m_c.group(1)) if m_c else None,
        "boxes": boxes,
        "kind": kind,
    }


def boxes_to_pixels(boxes, width, height, kind="auto"):
    """bbox → 像素坐标。kind: "px" / "norm" / "auto"（数值推断，仅兜底）。"""
    arr = np.asarray(boxes, float)
    if arr.size == 0:
        return arr.reshape(0, 4)
    if kind == "auto":
        kind = "px" if float(arr.max()) > 1000.0 else "norm"
    if kind == "px":
        px = arr.copy()
    else:
        px = arr * np.array([width, height, width, height], float) / 1000.0
    px[:, 0::2] = np.clip(px[:, 0::2], 0, width - 1)
    px[:, 1::2] = np.clip(px[:, 1::2], 0, height - 1)
    return px


class QwenVlGrounder:
    """Qwen2.5-VL 目标定位器（懒加载，进程内单例复用权重）。"""

    def __init__(self, model_path, device="cuda", dtype="bfloat16",
                 max_new_tokens=128):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens)
        self._model = None
        self._processor = None
        self.last_raw = ""

    # ---------- 加载 ----------
    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device,
        )
        self._model.eval()

    # ---------- 推理 ----------
    def bbox(self, img_rgb, prompt):
        """图像 + 文本 prompt → [[x1,y1,x2,y2], ...]（像素）。"""
        self._load()
        import torch
        from qwen_vl_utils import process_vision_info
        from PIL import Image

        rgb = np.asarray(img_rgb)
        h, w = rgb.shape[:2]
        pil = Image.fromarray(rgb[..., :3].astype(np.uint8))

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self._processor(text=[text], images=images, videos=videos,
                                 padding=True, return_tensors="pt").to(
                                     self.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs,
                                       max_new_tokens=self.max_new_tokens)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        self.last_raw = self._processor.batch_decode(
            trimmed, skip_special_tokens=False)[0]
        boxes, kind = parse_boxes(self.last_raw)

        # === 坐标空间 ===
        # process_vision_info 用 smart_resize 把图缩到 28 的倍数（320×240 →
        # 336×252），模型吐的 bbox_2d 是**缩放后**图像上的像素。直接当原图
        # 像素用会有 4~5% 系统性偏差，反投影到台面后横向偏 1~2 cm。
        # 这里按缩放后尺寸换算回原图尺寸。
        mw, mh = (images[0].size if images else (w, h))
        if kind == "px" and (mw, mh) != (w, h):
            arr = np.asarray(boxes, float)
            if arr.size:
                arr = arr * np.array([w / mw, h / mh, w / mw, h / mh], float)
                boxes = arr.tolist()
        return boxes_to_pixels(boxes, w, h, kind=kind)

    def locate_color(self, img_rgb, color):
        """按颜色找方块 bbox。返回 (pixel_box | None, 原始输出)。"""
        obj = COLOR_OBJ.get(color, "%s cube" % color)
        return self._boxes_for(img_rgb, PROMPT_TEMPLATE.format(obj=obj))

    def locate_instruction(self, img_rgb, instruction):
        """直接用自然语言指令做 grounding（主路径）。

        === 指令原文进 prompt，不做本地语义解析 ===
        不再用正则从指令里抽颜色：整条指令原样交给模型，由模型完成
        「红色方块 → 目标」的指代解析。这样「左边那个」「第二排的」这类
        指代也能处理，而正则永远做不到。
        只追加一句格式约束（中文指令配中文约束，英文配英文），因为
        Instruct 版不强制 JSON 时会写成散文，解析不到框。
        """
        text = instruction.strip()
        if not text.endswith(("。", ".")):
            text += "。" if _has_cjk(text) else "."
        spec = _JSON_SPEC_ZH if _has_cjk(text) else _JSON_SPEC_EN
        return self._boxes_for(img_rgb, text + spec.format(obj=text))

    def locate_task(self, img_rgb, instruction):
        """一次调用同时输出 action / color / bbox_2d（L1 契约 v1）。

        返回 {"action", "color", "box", "raw"}。box 是原图像素坐标或 None。
        与 locate_instruction 的区别只在 prompt 多要两个字段，几何链路
        （bbox 中心 → 台面反投影）完全复用，不做第二套。
        """
        text = instruction.strip()
        if not text.endswith(("。", ".")):
            text += "。" if _has_cjk(text) else "."
        spec = _TASK_SPEC_ZH if _has_cjk(text) else _TASK_SPEC_EN
        px = self.bbox(img_rgb, text + spec)
        parsed = parse_task(self.last_raw)
        return {
            "action": parsed["action"],
            "color": parsed["color"],
            "box": self._largest(px),
            "raw": self.last_raw,
        }

    def _boxes_for(self, img_rgb, prompt):
        """prompt → 最大的那个 bbox。返回 (pixel_box | None, 原始输出)。

        多个候选时取面积最大的：与 color_grounding.ground_color 的取舍一致，
        否则模型偶尔会同时框出「红色方块」和「红色的狗后背」。
        """
        return self._largest(self.bbox(img_rgb, prompt)), self.last_raw

    @staticmethod
    def _largest(px):
        if px.shape[0] == 0:
            return None
        areas = (px[:, 2] - px[:, 0]) * (px[:, 3] - px[:, 1])
        return px[int(np.argmax(areas))]

    def locate_object(self, img_rgb, obj_desc):
        """物体描述 → bbox。返回 (pixel_box | None, 原始输出)。"""
        return self._boxes_for(img_rgb, PROMPT_TEMPLATE.format(obj=obj_desc))


def box_center_uv(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def ground_box(cam, box, table_z, cube_half=0.0, bounds=None):
    """bbox → 台面上的 3D 点（方块中心高度）。

    === 为什么投顶面 ===
    与 color_grounding.ground_color 同一结论（实测误差最小）：剪影质心的
    射线落在方块顶面 z=table+2*cube_half 上，再把 z 折回中心高度。
    这里用 bbox 中心而不是连通域质心，但平面选择不变。
    """
    u, v = box_center_uv(box)
    plane_z = table_z + 2.0 * cube_half
    pts, ok = cam.hit_plane(np.array([u]), np.array([v]), plane_z)
    if not bool(ok[0]):
        return None
    p = np.asarray(pts[0], float).copy()
    p[2] = plane_z
    if bounds is not None:
        (x0, x1), (y0, y1) = bounds
        if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
            return None
    top_z = float(p[2])
    p[2] = table_z + cube_half
    return {"uv": (u, v), "pos": p, "pos_top_z": top_z}
