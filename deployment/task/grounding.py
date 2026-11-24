# -*- coding: utf-8 -*-
"""目标 grounding：从「指令里的颜色」找到「台面上的 3D 点」。

=== 三条路径 ===
  vlm    Qwen2.5-VL 出 bbox → 台面反投影（**默认**，方案要求的主路径）
  hsv    OpenCV 色相分割 → 台面反投影（仅作对照基线，不是默认）
  truth  仿真后端直接给物体真值（只用于链路自检）

三者输出结构一致，L1/L2 不关心走的哪条。

=== 为什么 VLM 是默认 ===
技术方案 §3 的 L3 就是「Qwen2.5-VL 级 VLM，不微调，prompt + 场景图」。
HSV 阈值把指令理解退化成色相查表：加干扰物、换光照就失效，也无法处理
「红色方块左边那个」这类指代。VLM 直接吃语言，才是 VLA 该有的形态。
"""
import numpy as np

from ..perception.color_grounding import ground_color
from ..spec import CUBE_HALF, GROUND_BOUNDS, TABLE_TOP_Z
from .contract import TaskSpec

_VLM_CACHE = {}


def _get_grounder(cfg):
    """进程内复用 VLM 权重（3B 模型加载一次数十秒，不能每次请求都加载）。"""
    if cfg is None:
        return None
    path = cfg.get("model_path")
    if not path:
        return None
    key = (path, cfg.get("device", "cuda"), cfg.get("dtype", "bfloat16"))
    if key not in _VLM_CACHE:
        from ..perception.vlm_grounding import QwenVlGrounder
        _VLM_CACHE[key] = QwenVlGrounder(
            path, device=cfg.get("device", "cuda"),
            dtype=cfg.get("dtype", "bfloat16"),
            max_new_tokens=cfg.get("max_new_tokens", 128))
    return _VLM_CACHE[key]


class Grounding:

    def __init__(self, backend, prefer="auto", camera="front_cam",
                 vlm_cfg=None, forbid_truth=False):
        """prefer: auto / vlm / hsv / truth

        auto 优先级：truth（仿真真值最准）→ vlm → hsv。
        真机上 truth 不存在，自然落到 vlm。

        === forbid_truth：评测 A 方案时必须打开 ===
        auto 的第一优先是仿真真值。用它跑评测，感知前端根本不会被调用，
        结果会「看起来完美」——这是自欺，不是验证。打开后 truth 分支直接
        跳过；prefer="truth" 则抛异常（配置自相矛盾，不该静默降级）。
        注意：只拦 grounding 的定位路径，不拦 L2 的成功判定/撤退
        （skills.verify / skills.manager 用真值做记账，真机上这些接口
        自然不存在，不参与控制闭环）。
        """
        if forbid_truth and prefer == "truth":
            raise ValueError("forbid_truth=True 时 prefer 不能是 truth")
        self.backend = backend
        self.prefer = prefer
        self.camera = camera
        self.vlm_cfg = vlm_cfg or {}
        self.forbid_truth = bool(forbid_truth)
        self.last_source = None
        self.last_debug = {}
        self._instruction = ""

    # ---------- L1 对外唯一出口 ----------
    def locate_task(self, instruction, color_hint=None):
        """指令 → TaskSpec（L1→L2 契约 v1）。

        === 与 locate() 的分工 ===
        locate() 只回答「目标在哪」，给仿真/诊断脚本用。
        locate_task() 回答「要做什么 + 什么颜色 + 在哪」，是 L1 下发给
        L2 的唯一结构。字段语义见 contract.py，不要在别处另拼 dict。

        === 为什么正则只当兜底 ===
        主路径让 VLM 在同一次 grounding 调用里同时输出 action/color/bbox，
        零额外延迟。只有模型漏了 color 字段时才用本地正则补 —— 正则不
        参与「指代理解」，只补一个枚举值。
        """
        inst = (instruction or "").strip()
        color = color_hint or _color_from_text(inst)

        # 1) 仿真真值优先（prefer=auto/truth）：xy/yaw 都是真的，用来
        #    交叉校验 VLM，不参与真机路径。
        #    forbid_truth 时整段跳过 —— 见 __init__ 的说明。
        if (not self.forbid_truth and self.prefer in ("auto", "truth")
                and color is not None):
            got = self._by_truth(color)
            if got is not None and got.get("ok"):
                self.last_source = "truth"
                return TaskSpec(
                    instruction=inst, action="grasp", color=color,
                    target_xy=got["xy"], target_pos=got.get("pos"),
                    target_yaw=got.get("yaw"), confidence=1.0,
                    source="truth")
            if self.prefer == "truth":
                return TaskSpec(instruction=inst, color=color, source="truth",
                                reason=got.get("reason", "真值不可用"))

        # 2) 结构化 VLM（prefer=auto/vlm）
        if self.prefer in ("auto", "vlm"):
            spec = self._task_by_vlm(inst, color)
            if spec is not None:
                if spec.ok:
                    return spec
                self.last_source = "vlm"

        # 3) 退回普通定位（vlm 的单字段路径 / hsv 基线）
        if color is None:
            return TaskSpec(instruction=inst, source="none",
                            reason="指令里没有可识别的颜色，VLM 也未给出")
        loc = self.locate(color, instruction=inst)
        if not loc.get("ok"):
            return TaskSpec(instruction=inst, color=color,
                            source=loc.get("source", ""),
                            reason=loc.get("reason", "定位失败"))
        return TaskSpec(
            instruction=inst, action="grasp", color=color,
            target_xy=loc["xy"], target_pos=loc.get("pos"),
            target_yaw=loc.get("yaw"),
            confidence={"truth": 1.0, "vlm": 0.6, "hsv": 0.4}.get(
                loc.get("source"), 0.5),
            source=loc.get("source", ""))

    # ---------- 主入口 ----------
    def locate(self, color, instruction=None):
        """返回 {"ok", "xy", "pos", "yaw", "source", "reason"}。

        instruction 非空时，VLM 路径直接用整条自然语言指令做 grounding，
        不再依赖本地正则抽出的 color；HSV/truth 路径仍按 color 走。
        """
        self._instruction = instruction or ""
        return self._dispatch(color)

    def _dispatch(self, color):
        order = {
            "auto": ["truth", "vlm", "hsv"],
            "truth": ["truth"],
            "vlm": ["vlm"],
            "hsv": ["hsv"],
        }.get(self.prefer, ["vlm"])
        if self.forbid_truth:
            order = [s for s in order if s != "truth"] or ["hsv"]
        last_err = None
        for src in order:
            got = getattr(self, "_by_" + src)(color)
            if got is None:            # 该方法不适用（如后端没有 get_objects）
                continue
            if got.get("ok"):
                self.last_source = src
                return got
            last_err = got
        self.last_source = order[-1] if order else None
        return last_err or {"ok": False, "source": self.last_source,
                            "reason": "无可用 grounding 路径"}

    # ---------- 真值 ----------
    def _by_truth(self, color):
        fn = getattr(self.backend, "get_objects", None)
        if fn is None:
            return None
        objs = fn()
        cand = [o for o in objs if o["color"] == color]
        if not cand:
            return {"ok": False, "source": "truth",
                    "reason": "真值里没有 %s" % color}
        base = _arm_base_xy(self.backend)
        cand.sort(key=lambda o: float(np.hypot(o["pos"][0] - base[0],
                                               o["pos"][1] - base[1])))
        o = cand[0]
        return {"ok": True, "xy": (float(o["pos"][0]), float(o["pos"][1])),
                "pos": np.asarray(o["pos"], float), "yaw": float(o["yaw"]),
                "source": "truth", "n_same_color": len(cand), "reason": ""}

    # ---------- VLM ----------
    def _by_vlm(self, color):
        grounder = _get_grounder(self.vlm_cfg)
        if grounder is None:
            return {"ok": False, "source": "vlm",
                    "reason": "未配置 vlm.model_path"}

        img = self._image()
        if img is None:
            return {"ok": False, "source": "vlm",
                    "reason": "没有相机 %s 的图像" % self.camera}
        cam = self._camera_model()
        if cam is None:
            return {"ok": False, "source": "vlm", "reason": "拿不到相机内外参"}

        # 有原始指令就用指令本身做 grounding（模型自己解析指代），
        # 否则退回颜色描述（例如诊断脚本只给 color）
        if self._instruction:
            box, raw = grounder.locate_instruction(img, self._instruction)
        else:
            box, raw = grounder.locate_color(img, color)
        self.last_debug = {"raw": raw, "prompt_src": self._instruction or color,
                           "box": None if box is None
                           else [float(v) for v in box]}
        if box is None:
            return {"ok": False, "source": "vlm",
                    "reason": "VLM 未返回 %s 的 bbox" % color,
                    "debug": self.last_debug}

        from ..perception.vlm_grounding import ground_box
        got = ground_box(cam, box, TABLE_TOP_Z, CUBE_HALF,
                         bounds=GROUND_BOUNDS)
        if got is None:
            return {"ok": False, "source": "vlm",
                    "reason": "bbox 反投影不在台面内（%s）" % color,
                    "debug": self.last_debug}
        p = np.asarray(got["pos"], float)
        return {"ok": True, "xy": (float(p[0]), float(p[1])), "pos": p,
                "yaw": None,               # 单目 bbox 无法可靠恢复方块 yaw
                "source": "vlm", "uv": got["uv"],
                "box": self.last_debug["box"], "reason": ""}

    # ---------- VLM 结构化任务 ----------
    def _task_by_vlm(self, instruction, color_hint):
        """一次调用拿 action/color/bbox。返回 TaskSpec | None。

        返回 None 表示「这条路不适用或没出框」，让上层退回单字段定位；
        不要在这里把失败硬转成 ok=False 的 TaskSpec，否则会把
        「结构化 prompt 没解析出来」伪装成「目标不存在」。
        """
        grounder = _get_grounder(self.vlm_cfg)
        if grounder is None:
            return None
        img = self._image()
        cam = self._camera_model()
        if img is None or cam is None:
            return None

        got = grounder.locate_task(img, instruction or "")
        box = got.get("box")
        self.last_debug = {"raw": got.get("raw", ""),
                           "prompt_src": instruction or "",
                           "box": None if box is None
                           else [float(v) for v in box]}
        if box is None:
            return None

        color = got.get("color") or color_hint
        if color is None:
            self.last_debug["reason"] = "模型给框但没给颜色"
            return None

        from ..perception.vlm_grounding import ground_box
        g = ground_box(cam, box, TABLE_TOP_Z, CUBE_HALF,
                       bounds=GROUND_BOUNDS)
        if g is None:
            self.last_debug["reason"] = "bbox 反投影不在台面内"
            return TaskSpec(instruction=instruction, color=color,
                            source="vlm", reason=self.last_debug["reason"])

        p = np.asarray(g["pos"], float)
        # 置信度是「契约完整度」，不是模型概率：三个字段都从模型来给 0.9，
        # action 或 color 靠默认/正则补的降一档。
        conf = 0.9
        if got.get("action") is None or got.get("color") is None:
            conf = 0.7
        return TaskSpec(
            instruction=instruction, action=got.get("action") or "grasp",
            color=color, bbox_2d=[float(v) for v in box],
            target_xy=(float(p[0]), float(p[1])), target_pos=p,
            target_yaw=None, confidence=conf, source="vlm")

    # ---------- HSV 基线 ----------
    def _by_hsv(self, color):
        img = self._image()
        if img is None:
            return {"ok": False, "source": "hsv",
                    "reason": "没有相机 %s 的图像" % self.camera}
        cam = self._camera_model()
        if cam is None:
            return {"ok": False, "source": "hsv",
                    "reason": "拿不到相机内外参"}

        cands = ground_color(img, cam, color, TABLE_TOP_Z, CUBE_HALF,
                             max_blobs=6, bounds=GROUND_BOUNDS)
        if not cands:
            return {"ok": False, "source": "hsv",
                    "reason": "台面上没有 %s 色块" % color,
                    "debug": {"n_blob": 0}}
        best = cands[0]
        p = np.asarray(best["pos"], float)
        return {"ok": True, "xy": (float(p[0]), float(p[1])), "pos": p,
                "yaw": None, "source": "hsv", "n_blob": len(cands),
                "area": int(best["area"]), "reason": ""}

    # ---------- 辅助 ----------
    def _image(self):
        return (self.backend.get_images() or {}).get(self.camera)

    def _camera_model(self):
        """取相机内外参。

        === 优先走后端的 camera_model() ===
        这套接口在三个后端上都成立：
          MujocoBackend  直接从 MjData 取（最准，仿真）
          LocalBackend   从 L3 广播的 cameras 字段取（同进程）
          RemoteBackend  同上，走 ZMQ（跨机）
        不要在这里直接 import mujoco 读 backend.m/d —— 真机和跨进程
        后端都没有这两个属性，那样写会让视觉 grounding 在真机上直接
        退化成「拿不到内参」，而仿真里又能跑，问题被藏起来。
        """
        fn = getattr(self.backend, "camera_model", None)
        if fn is None:
            return None
        try:
            return fn(self.camera)
        except Exception:
            return None


def _color_from_text(text):
    """从指令里抽颜色（只用于兜底和 truth 路径选目标，不参与接地）。"""
    if not text:
        return None
    try:
        from .instruction import parse
        return parse(text).color
    except Exception:
        return None


def _arm_base_xy(backend):
    base = getattr(backend, "arm_base", None)
    if base is None:
        from ..spec import ARM_BASE
        base = ARM_BASE
    return float(base[0]), float(base[1])
