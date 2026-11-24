# -*- coding: utf-8 -*-
"""L3 控制服务入口（建议跑在狗身 Orin Nano 上）。

用法（Thor 上仿真）：
  MUJOCO_GL=egl python3 -m deployment.run_l3 --duration 60

用法（真机）：
  python3 -m deployment.run_l3 --backend real

=== 为什么入口这么薄 ===
真正的服务在 control/server.py。入口只负责参数与环境变量，
这样 `python3 -m deployment.run_l3` 与把 ControlServer 嵌进别的
进程（单元测试、自检脚本）用的是同一份实现。
"""
import sys

from .bootstrap import setup

setup()

from .bus.config import CONTROL, ENDPOINTS  # noqa: E402
from .control.server import main  # noqa: E402

if __name__ == "__main__":
    print("[L3] endpoints: cmd=%s state=%s image=%s"
          % (ENDPOINTS["cmd"], ENDPOINTS["state"], ENDPOINTS["image"]))
    print("[L3] rate=%s Hz watchdog=%s ms"
          % (CONTROL["rate_hz"], CONTROL["watchdog_ms"]))
    sys.exit(main())
