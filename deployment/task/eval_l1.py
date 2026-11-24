# -*- coding: utf-8 -*-
"""L1 评测：VLM 是否需要后训练 + 契约/解析器自检。

=== 为什么要有这个脚本 ===
「VLM 要不要后训练」不能靠感觉。判据必须落到可测的数：格式成功率、
颜色正确率、定位误差。`decide()` 把这三项变成明确结论，失败样本连同
模型原始输出一起写进 --out JSON —— 那才是后训练 / 改 prompt 的依据，
而不是「今天感觉不太准」。

=== 三种模式 ===
  --self-test   纯 Python（不需要 GPU / 相机）：契约往返、指令解析、
                结构化输出解析、像素/归一化格式分流
  --sim         MuJoCo 仿真：4 色 × 中/英指令 × N 轮，VLM vs 真值
  --images DIR  离线图片：VLM vs GT（2D 中心误差 / IoU；给 --camera 时
                还能算 3D 误差）

=== 用法 ===
  python3 -m deployment.task.eval_l1 --self-test
  MUJOCO_GL=egl python3 -m deployment.task.eval_l1 --sim --repeat 3
  python3 -m deployment.task.eval_l1 --images samples/l1 \
      --gt samples/l1/gt.json --out l1_report.json

=== 真机基线的前置条件 ===
--images 模式不需要机械臂在线，只需要图片 + GT。真机图还没有时，先用
--sim 的数字判断 prompt/格式是否稳定；真机图到位后原样再跑一遍，两者
差得远说明是「域差异 → 需要后训练」，差得不远说明不用。
"""
import argparse
import json
import os
import sys
import time

from ..bootstrap import setup

setup()

import numpy as np  # noqa: E402

# 判定阈值（与 docs/VLA架构与代码审核_2026-11-05.md 的实测基线对齐：
# 英文模板 8.2 mm 均值 / 16.5 mm 最大，所以 10/20 mm 是「不比现在差」）
THRESH = {
    "format_rate": 0.95,   # 结构化字段齐全率
    "color_rate": 1.00,    # 颜色必须全对：错颜色 = 抓错物体
    "mean_mm": 10.0,
    "max_mm": 20.0,
}

BASE_LAYOUT = [(0.455, 0.10, "red"), (0.470, -0.10, "green"),
               (0.460, 0.00, "blue"), (0.475, 0.16, "yellow")]


# ============ 判定 ============
def decide(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0, "needs_post_train": True, "why": ["没有样本"]}
    fmt = sum(1 for r in rows if r.get("format_ok")) / float(n)
    col = sum(1 for r in rows if r.get("color_ok")) / float(n)
    errs = [r["err_mm"] for r in rows if r.get("err_mm") is not None]
    mean_mm = float(np.mean(errs)) if errs else None
    max_mm = float(np.max(errs)) if errs else None

    why = []
    if fmt < THRESH["format_rate"]:
        why.append("格式成功率 %.0f%% < %.0f%%"
                   % (100 * fmt, 100 * THRESH["format_rate"]))
    if col < THRESH["color_rate"]:
        why.append("颜色正确率 %.0f%% < %.0f%%"
                   % (100 * col, 100 * THRESH["color_rate"]))
    if mean_mm is not None and mean_mm > THRESH["mean_mm"]:
        why.append("定位均值 %.1f mm > %.0f mm"
                   % (mean_mm, THRESH["mean_mm"]))
    if max_mm is not None and max_mm > THRESH["max_mm"]:
        why.append("定位最大 %.1f mm > %.0f mm"
                   % (max_mm, THRESH["max_mm"]))
    return {
        "n": n,
        "format_rate": round(fmt, 4),
        "color_rate": round(col, 4),
        "mean_mm": None if mean_mm is None else round(mean_mm, 2),
        "max_mm": None if max_mm is None else round(max_mm, 2),
        "needs_post_train": bool(why),
        "why": why,
    }


def _fmt_verdict(v):
    if v["needs_post_train"]:
        return "需要后训练/改 prompt —— " + "；".join(v["why"])
    return ("无需后训练：格式 %.0f%%、颜色 %.0f%%、误差 mean=%s mm max=%s mm"
            % (100 * v["format_rate"], 100 * v["color_rate"],
               v["mean_mm"], v["max_mm"]))


# ============ 自检（无 GPU） ============
def _self_test():
    from ..perception.vlm_grounding import parse_task
    from .contract import CONTRACT_VERSION, TaskSpec
    from .grounding import _color_from_text
    from .instruction import parse

    # 1. 契约往返
    s = TaskSpec(instruction="pick the red cube", action="grasp",
                 color="red", bbox_2d=[10, 20, 40, 60],
                 target_xy=(0.46, 0.10), target_pos=[0.46, 0.10, 0.58],
                 target_yaw=0.3, confidence=0.9, source="vlm")
    assert s.ok, "TaskSpec.ok 为 False"
    assert s.validate() == [], s.validate()
    d = s.to_dict()
    assert d["contract_version"] == CONTRACT_VERSION
    assert d["target_pos"] == [0.46, 0.10, 0.58]
    assert set(s.to_task_dict()) == {
        "instruction", "target_color", "target_xy", "target_yaw"}
    bad = TaskSpec(action="grasp", color="purple", target_xy=(0.0, 0.0))
    assert len(bad.validate()) >= 1, "非法颜色没被 validate 拦住"

    # 2. 指令解析（兜底路径，真值选目标也用它）
    for text, want in [("抓取红色方块", "red"),
                       ("pick the blue cube", "blue"),
                       ("拿黄色的", "yellow")]:
        assert parse(text).color == want, text
        assert _color_from_text(text) == want, text

    # 3. 结构化输出解析：覆盖 Instruct 的纯 bbox、契约 JSON、中文颜色、
    #    微调版特殊 token。kind 必须是格式驱动，不是数值猜。
    cases = [
        ("instruct_bbox",
         '[{"bbox_2d": [198, 67, 308, 154], "label": "red cube"}]',
         None, None, "px"),
        ("task_en",
         '{"action": "grasp", "color": "red", "bbox_2d": [12, 20, 60, 70]}',
         "grasp", "red", "px"),
        ("task_pick_alias",
         '{"action": "pick", "color": "yellow cube", "bbox_2d": [1, 2, 3, 4]}',
         "grasp", "yellow", "px"),
        ("task_zh_color",
         '{"action": "抓取", "color": "蓝色", "bbox_2d": [5, 6, 7, 8]}',
         "grasp", "blue", "px"),
        ("grounding_tokens",
         "<|box_start|>(100,200),(300,400)<|box_end|>",
         None, None, "norm"),
    ]
    for name, raw, want_a, want_c, want_kind in cases:
        got = parse_task(raw)
        assert got["action"] == want_a, (name, got)
        assert got["color"] == want_c, (name, got)
        assert got["kind"] == want_kind, (name, got)
        assert len(got["boxes"]) == 1, (name, got)

    print("  self-test  OK  契约 / 指令解析 / 结构化解析 / 格式分流 全通过")
    return True


# ============ 仿真 ============
def _instruction(color, lang):
    if lang == "zh":
        from ..spec import COLOR_ZH
        return "抓取%s方块" % COLOR_ZH[color]
    return "pick the %s cube" % color


def _jitter_layout(rng):
    from ..spec import OBJ_X_RANGE, OBJ_Y_RANGE
    out = []
    for x, y, c in BASE_LAYOUT:
        # 抖动幅度受限于方块间距（0.06 m 方块），再大就会叠在一起，
        # 那时「定位错」其实是「物体被撞飞」，会污染误差统计。
        xx = float(np.clip(x + rng.uniform(-0.010, 0.010), *OBJ_X_RANGE))
        yy = float(np.clip(y + rng.uniform(-0.020, 0.020), *OBJ_Y_RANGE))
        out.append((xx, yy, c, float(rng.uniform(-np.pi, np.pi))))
    return out


def _row(instruction, color, lang, spec, truth_xy, debug, latency_ms):
    pred = spec.target_xy
    err = None
    if pred is not None and truth_xy is not None:
        err = 1000.0 * float(np.hypot(pred[0] - truth_xy[0],
                                      pred[1] - truth_xy[1]))
    return {
        "instruction": instruction,
        "color": color,
        "lang": lang,
        "source": spec.source,
        "pred_color": spec.color,
        "color_ok": spec.color == color,
        "action_ok": spec.action == "grasp",
        # format_ok 要求 bbox_2d 非空：说明走的是**结构化**路径，
        # 而不是退回单字段 locate() 才拿到坐标。
        "format_ok": (spec.source == "vlm" and spec.bbox_2d is not None
                      and spec.action == "grasp" and spec.color is not None),
        "pred_xy": None if pred is None else [round(v, 4) for v in pred],
        "truth_xy": None if truth_xy is None
        else [round(float(v), 4) for v in truth_xy],
        "err_mm": None if err is None else round(err, 2),
        "confidence": spec.confidence,
        "latency_ms": round(latency_ms, 1),
        "raw": debug.get("raw", ""),
        "reason": spec.reason,
    }


def _run_sim(args):
    from ..bus.config import GROUNDING
    from ..hal import make_backend
    from .contract import COLORS
    from .grounding import Grounding

    vlm_cfg = dict(GROUNDING.get("vlm") or {})
    if args.vlm_path:
        vlm_cfg["model_path"] = args.vlm_path
    if not vlm_cfg.get("model_path"):
        print("!! grounding.vlm.model_path 未配置，无法评测 VLM")
        return None

    be = make_backend("mujoco", realtime=False)
    be.connect()
    g = Grounding(be, prefer="vlm",
                  camera=GROUNDING.get("camera", "front_cam"),
                  vlm_cfg=vlm_cfg)
    rng = np.random.default_rng(args.seed)
    rows = []
    try:
        for rep in range(args.repeat):
            layout = _jitter_layout(rng)
            be.reset(layout)
            be.settle(200)
            be.get_state()                     # 刷新运动学，相机位姿才是最新
            truth = {o["color"]: np.asarray(o["pos"], float)
                     for o in be.get_objects()}
            print("[sim] 第 %d/%d 轮  %s" % (
                rep + 1, args.repeat,
                " ".join("%s(%.3f,%.3f)" % (c, t[0], t[1])
                         for c, t in truth.items())), flush=True)
            for color in COLORS:
                for lang in args.langs:
                    instr = _instruction(color, lang)
                    t0 = time.monotonic()
                    spec = g.locate_task(instr)
                    ms = (time.monotonic() - t0) * 1000.0
                    rows.append(_row(instr, color, lang, spec,
                                     truth.get(color), g.last_debug, ms))
    finally:
        be.close()
    return rows


# ============ 离线图片 ============
def _run_images(args):
    if not args.gt:
        print("!! --images 需要 --gt")
        return None
    from PIL import Image

    from ..bus.config import GROUNDING
    from ..perception.color_grounding import CameraModel
    from ..perception.vlm_grounding import QwenVlGrounder, ground_box
    from ..spec import CUBE_HALF, GROUND_BOUNDS, TABLE_TOP_Z

    path = args.vlm_path or (GROUNDING.get("vlm") or {}).get("model_path")
    if not path:
        print("!! 未配置 VLM 权重路径（--vlm-path 或 grounding.vlm.model_path）")
        return None
    grounder = QwenVlGrounder(path,
                              device=(GROUNDING.get("vlm") or {}).get(
                                  "device", "cuda"),
                              dtype=(GROUNDING.get("vlm") or {}).get(
                                  "dtype", "bfloat16"))
    with open(args.gt, encoding="utf-8") as f:
        gt = json.load(f)
    cam = None
    if args.camera:
        with open(args.camera, encoding="utf-8") as f:
            c = json.load(f)
        cam = CameraModel.from_fovy(c["width"], c["height"], c["fovy_deg"],
                                    pos=c.get("pos"), rot=c.get("rot"))

    rows = []
    for item in gt:
        fp = os.path.join(args.images, item["file"])
        img = np.asarray(Image.open(fp).convert("RGB"))
        color = item["color"]
        instr = item.get("instruction") or _instruction(color, "en")
        t0 = time.monotonic()
        got = grounder.locate_task(img, instr)
        ms = (time.monotonic() - t0) * 1000.0
        box = got["box"]
        r = {
            "file": item["file"], "instruction": instr, "color": color,
            "lang": "gt", "source": "vlm",
            "pred_color": got["color"], "color_ok": got["color"] == color,
            "action_ok": got["action"] == "grasp",
            "format_ok": bool(got["action"] and got["color"] and box),
            "pred_box": None if box is None
            else [round(float(v), 1) for v in box],
            "gt_box": item.get("bbox_2d"),
            "confidence": None, "latency_ms": round(ms, 1),
            "raw": got["raw"], "reason": "",
            "pred_xy": None, "truth_xy": item.get("xy"),
            "err_mm": None, "center_err_px": None, "iou": None,
        }
        if box is not None and item.get("bbox_2d"):
            gb = np.asarray(item["bbox_2d"], float)
            r["center_err_px"] = round(float(np.hypot(
                0.5 * (box[0] + box[2]) - 0.5 * (gb[0] + gb[2]),
                0.5 * (box[1] + box[3]) - 0.5 * (gb[1] + gb[3]))), 1)
            r["iou"] = round(_iou(box, gb), 3)
        if box is not None and cam is not None and item.get("xy"):
            g = ground_box(cam, box, TABLE_TOP_Z, CUBE_HALF,
                           bounds=GROUND_BOUNDS)
            if g is not None:
                p = np.asarray(g["pos"], float)
                r["pred_xy"] = [round(float(v), 4) for v in p[:2]]
                r["err_mm"] = round(1000.0 * float(np.hypot(
                    p[0] - item["xy"][0], p[1] - item["xy"][1])), 2)
        rows.append(r)
    return rows


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9)


# ============ 报告 ============
def _print_rows(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["color"], r["lang"]), []).append(r)
    print("\n  %-7s %-3s %4s %7s %7s %9s %9s"
          % ("color", "lang", "n", "fmt%", "color%", "mean_mm", "max_mm"))
    for (color, lang), rs in sorted(groups.items()):
        errs = [x["err_mm"] for x in rs if x.get("err_mm") is not None]
        print("  %-7s %-3s %4d %6.0f%% %6.0f%% %9s %9s"
              % (color, lang, len(rs),
                 100.0 * np.mean([x["format_ok"] for x in rs]),
                 100.0 * np.mean([x["color_ok"] for x in rs]),
                 "-" if not errs else "%.1f" % np.mean(errs),
                 "-" if not errs else "%.1f" % np.max(errs)))
    # 失败样本只打印前 5 条，完整内容在 --out 里
    fails = [r for r in rows if not (r["format_ok"] and r["color_ok"])
             or (r.get("err_mm") is not None and r["err_mm"] > THRESH["max_mm"])]
    if fails:
        print("\n  失败样本（前 5 条，完整见 --out）：")
        for r in fails[:5]:
            print("   - %s | %s | fmt=%s color=%s err=%s"
                  % (r.get("file", ""), r["instruction"], r["format_ok"],
                     r["color_ok"], r.get("err_mm")))
            print("     raw: %s" % str(r.get("raw", ""))[:200])


def _dump(rows, path):
    if not path:
        return
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    payload = {"rows": rows, "verdict": decide(rows), "thresh": THRESH}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\n[report] %s" % os.path.abspath(path))


# ============ 入口 ============
def main(argv=None):
    ap = argparse.ArgumentParser(description="L1 VLM 基线评测")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--images", default=None, help="图片目录")
    ap.add_argument("--gt", default=None, help="GT JSON")
    ap.add_argument("--camera", default=None,
                    help="相机内外参 JSON（可选，用于 3D 误差）")
    ap.add_argument("--langs", default="en,zh")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vlm-path", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    a.langs = [s.strip() for s in a.langs.split(",") if s.strip()]

    if a.self_test or not (a.sim or a.images):
        _self_test()
        if not (a.sim or a.images):
            return 0

    rows = _run_sim(a) if a.sim else _run_images(a)
    if not rows:
        return 1
    _print_rows(rows)
    v = decide(rows)
    print("\n[verdict] " + _fmt_verdict(v))
    _dump(rows, a.out)
    return 0 if not v["needs_post_train"] else 2


if __name__ == "__main__":
    sys.exit(main())
