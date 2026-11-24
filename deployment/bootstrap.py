# -*- coding: utf-8 -*-
"""import 路径引导：让 `import deployment.*` 与 `import ik_core` 同时可用。

=== 为什么要这一步 ===
deployment 是一个包（用相对 import），ik_core 是 simulation/mujoco 下
的**顶层模块**（采集脚本一直这么用）。两种布局：
  本地仓库   <root>/deployment + <root>/simulation/mujoco
  Thor 扁平  <p30_rm65>/deployment + <p30_rm65>
所以要把 parent(deployment) 和 SIM_DIR 都塞进 sys.path，缺一个就
ImportError。

所有入口脚本第一行 `from deployment.bootstrap import *`（或直接调用
`setup()`），不要在各处重复这段逻辑。
"""
import os
import sys


def setup():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(pkg_dir)
    added = []
    for p in (root,):
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    # SIM_DIR 由 spec 解析（它自己会搜索两种布局）
    from .spec import SIM_DIR, CONFIG_DIR
    for p in (SIM_DIR, CONFIG_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    # MuJoCo 后端默认走 EGL（无 DISPLAY 的服务器上唯一能用的一条）
    os.environ.setdefault("MUJOCO_GL", "egl")
    return {"root": root, "sim": SIM_DIR, "cfg": CONFIG_DIR, "added": added}


def describe():
    info = setup()
    print("[bootstrap] root=%s\n            sim=%s\n            cfg=%s"
          % (info["root"], info["sim"], info["cfg"]))
    return info


setup()
