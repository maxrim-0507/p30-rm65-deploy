# -*- coding: utf-8 -*-
"""全链路共用常量：路径解析 + mount.yaml / vla_spec.yaml 的唯一读取点。

=== 为什么集中在这里 ===
同一组物理量（臂基座原点、夹爪行程、台面高度、抓取 z）在 L2 技能层
和 L3 控制层都要用。分散计算过一次 1 cm 基准差，表现是「IK 说有解、
执行端每帧抓空」，排查了整整一轮。所以只允许这一个模块从 yaml 推数值。

=== 两种目录布局都要支持 ===
本地仓库：  <root>/deployment/spec.py   +  <root>/simulation/mujoco/ik_core.py
Thor 扁平：  /home/user/rm_test/sim/p30_rm65/deployment/spec.py
           + /home/user/rm_test/sim/p30_rm65/ik_core.py
所以 sim 目录与 configs 目录都是**按候选路径搜索**，不写死相对层级。
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _search_dir(marker, candidates):
    for d in candidates:
        if os.path.isfile(os.path.join(d, marker)):
            return d
    raise FileNotFoundError(
        "找不到 %s，已尝试：%s" % (marker, " | ".join(candidates)))


SIM_DIR = _search_dir("ik_core.py", [
    os.path.join(ROOT, "simulation", "mujoco"),
    ROOT,
])

CONFIG_DIR = _search_dir("mount.yaml", [
    os.path.join(ROOT, "configs"),
    ROOT,
    SIM_DIR,
])


def config_path(name):
    return os.path.join(CONFIG_DIR, name)


def _load(name):
    with open(config_path(name), encoding="utf-8") as f:
        return yaml.safe_load(f)


MOUNT = _load("mount.yaml")
SPEC = _load("vla_spec.yaml")

# ---- 机械臂基座原点（臂底座 mesh 底面中心）----
# 与 build_scene.py 用同一条 z 规则：显式 mount.z 优先，否则由躯干顶面 + 板厚推。
# 两边不一致会让 IK 的基准整体偏移。
ARM_BASE = [
    float(MOUNT["mount"]["x"]),
    float(MOUNT["mount"]["y"]),
    float(MOUNT["mount"]["z"]) if MOUNT["mount"].get("z") is not None
    else float(MOUNT["trunk_top_z"]) + 2 * float(MOUNT["mount"]["plate"]["half_t"]),
]
MOUNT_YAW_DEG = float(MOUNT["mount"].get("yaw_deg", 0.0))

# ---- 夹爪 qpos 行程（负=张开，正=闭合；实测反直觉）----
GRIP_OPEN = -0.020
GRIP_CLOSE = 0.0245
GRIP_RANGE = GRIP_CLOSE - GRIP_OPEN

# ---- 台面与物体 ----
TABLE_TOP_Z = float(MOUNT["table"]["top_z"])
CUBE_HALF = float(MOUNT["objects"]["cube_half"])
OBJ_COUNT = int(MOUNT["objects"]["count"])

# 台面矩形（世界系 xy 边界）。视觉 grounding 用它过滤「投到台面平面后
# 落在台面之外」的假色块。历史依据：旧 head_cam 能看到狗自己的背，那里
# 也有高饱和色块（实测红色 3948 px，比真方块还大），不过滤会选错目标。
# 现在定位相机是抬臂后的 arm_cam（只见台面，实测 4 色各只有真方块
# 一个色块），过滤作为兜底保留 —— front_cam 旁的夹爪/机身仍在视野里。
_T = MOUNT["table"]
TABLE_X = (float(_T.get("center_x", 0.0)) - float(_T["half_x"]),
           float(_T.get("center_x", 0.0)) + float(_T["half_x"]))
TABLE_Y = (float(_T.get("center_y", 0.0)) - float(_T["half_y"]),
           float(_T.get("center_y", 0.0)) + float(_T["half_y"]))

# 物体允许范围 = objects 声明的范围 ∩ 台面内缩一个方块半边长。
# 视觉 grounding 用它做空间过滤：只看「台面矩形」不够 —— 旧 head_cam
# 俯视时，狗身/背景上方的假色块投到台面远端（x≈0.60）仍在台面内，
# 实测会让红色定位偏 147 mm，且不报错。
_O = MOUNT["objects"]
from_table_x = (TABLE_X[0] + CUBE_HALF, TABLE_X[1] - CUBE_HALF)
from_table_y = (TABLE_Y[0] + CUBE_HALF, TABLE_Y[1] - CUBE_HALF)
OBJ_X_RANGE = (max(float(_O["x_range"][0]), from_table_x[0]),
               min(float(_O["x_range"][1]), from_table_x[1]))
OBJ_Y_RANGE = (max(float(_O["y_range"][0]), from_table_y[0]),
               min(float(_O["y_range"][1]), from_table_y[1]))
# grounding 过滤盒：物体范围再外扩一点（检测有毫米级噪声），
# 但不外扩到能容纳假色块的程度（假色块在 x≈0.60，离上界 0.48 有 12 cm）
GROUND_BOUNDS = ((OBJ_X_RANGE[0] - 0.02, OBJ_X_RANGE[1] + 0.04),
                 (OBJ_Y_RANGE[0] - 0.02, OBJ_Y_RANGE[1] + 0.02))

# ---- 相机（顺序即观测里的顺序，改这里改全局）----
# 逻辑名 -> vla_spec.yaml 里允许的键。历史上这套相机换过两次名字：
#   训练/数据集口径      front / arm      （v36 的 rm65_config.py：front_cam / arm_cam）
#   旧部署代码口径       head  / wrist
# vla_spec 只保留一种键名，代码按别名单边兼容 —— 直接用下标会在改名时
# 变成 import 期 KeyError（2026-11-22 实测：deployment/spec.py 第 102 行
# 'head' KeyError，整个 deployment 包都进不去）。
# 物理名（mjcf_name）必须和场景文件一致；决策层只准用 camera_mjcf()。
_CAM_ALIASES = {"front": ("front", "head"), "arm": ("arm", "wrist")}


def _cam_cfg(logical):
    cams = SPEC["cameras"]
    for key in _CAM_ALIASES[logical]:
        if key in cams:
            return cams[key]
    raise KeyError("vla_spec.cameras 里找不到 %s（尝试 %s）"
                   % (logical, "/".join(_CAM_ALIASES[logical])))


def camera_mjcf(logical):
    """逻辑名 → 当前场景里的相机名（front / arm）。"""
    return _cam_cfg(logical)["mjcf_name"]


CAM_NAMES = [camera_mjcf("front"), camera_mjcf("arm")]
CAM_W = int(_cam_cfg("front")["width"])
CAM_H = int(_cam_cfg("front")["height"])

# ---- 颜色：MJCF 里的名字 → 指令用词 → RGBA ----
COLOR_ZH = {"red": "红色", "green": "绿色", "blue": "蓝色", "yellow": "黄色"}
COLOR_RGBA = {
    "red": (0.85, 0.15, 0.12, 1.0),
    "green": (0.15, 0.60, 0.25, 1.0),
    "blue": (0.15, 0.30, 0.80, 1.0),
    "yellow": (0.90, 0.75, 0.10, 1.0),
}
ZH_TO_COLOR = {v: k for k, v in COLOR_ZH.items()}

# ---- 夹爪归一化 ----
def grip_to_qpos(g):
    """归一化开度 [0,1] → qpos。1=张开，0=闭合。"""
    g = min(1.0, max(0.0, float(g)))
    return GRIP_CLOSE - g * GRIP_RANGE


def grip_from_qpos(q):
    """qpos → 归一化开度 [0,1]。"""
    return float((GRIP_CLOSE - q) / GRIP_RANGE)


# ---- 成功判据 ----
SUCCESS = SPEC["task"]["success"]
LIFT_DELTA_MIN = float(SUCCESS["lift_delta_min"])
EPISODE_MAX_STEPS = int(SPEC["task"]["episode_max_steps"])
