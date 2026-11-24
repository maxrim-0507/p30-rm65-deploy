# -*- coding: utf-8 -*-
"""deploy.yaml 的读取点。所有进程共用一份，避免各写一个默认端口。"""
import os

import yaml

from ..spec import CONFIG_DIR

_PATH = os.path.join(CONFIG_DIR, "deploy.yaml")

with open(_PATH, encoding="utf-8") as _f:
    DEPLOY = yaml.safe_load(_f)

TRANSPORT = DEPLOY["transport"]
CONTROL = DEPLOY["control"]
BACKEND = DEPLOY["backend"]
POLICY = DEPLOY["policy"]
SKILLS = DEPLOY["skills"]
TASK = DEPLOY["task"]
# L1 视觉 grounding 的配置（prefer / camera / vlm 权重路径）。
# 用 get 而不是下标：老的 deploy.yaml 没有这一段时不该直接 KeyError。
GROUNDING = DEPLOY.get("grounding", {})


def addr(port):
    """按 transport.host 拼 tcp 地址（便于日志里打印）。"""
    return "tcp://%s:%d" % (TRANSPORT["host"], int(port))


ENDPOINTS = {
    "cmd": addr(TRANSPORT["cmd_port"]),
    "state": addr(TRANSPORT["state_port"]),
    "image": addr(TRANSPORT["image_port"]),
    "hb": addr(TRANSPORT["hb_port"]),
}
