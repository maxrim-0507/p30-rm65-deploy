# -*- coding: utf-8 -*-
"""HAL：一套接口 + 三套后端（仿真 / 真机 / 远程）。"""
from .base import RobotBackend


def make_backend(kind=None, **kw):
    """按 configs/deploy.yaml 的 backend.kind 懒加载对应后端。

    懒加载是为了不把 mujoco / Robotic_Arm 这些重依赖塞进每个进程：
    L1 任务层只需要总线，不需要物理引擎。
    """
    from ..bus.config import BACKEND

    kind = kind or BACKEND["kind"]

    if kind == "mujoco":
        from .mujoco_backend import MujocoBackend
        cfg = BACKEND.get("mujoco", {})
        return MujocoBackend(
            scene=kw.pop("scene", BACKEND.get("scene", None)),
            render=kw.pop("render", cfg.get("render", True)),
            realtime=kw.pop("realtime", cfg.get("realtime", True)),
            **kw)

    if kind == "real":
        from .real_backend import RealBackend
        cfg = BACKEND.get("real", {})
        return RealBackend(
            rm_ip=kw.pop("rm_ip", cfg.get("rm_ip", "192.168.1.18")),
            rm_port=kw.pop("rm_port", cfg.get("rm_port", 8080)),
            gripper=kw.pop("gripper", cfg.get("gripper", "rm65_io")),
            dog_sdk_path=kw.pop("dog_sdk_path", cfg.get("dog_sdk_path")),
            allow_base_motion=kw.pop("allow_base_motion",
                                     cfg.get("allow_base_motion", False)),
            **kw)

    if kind in ("remote", "l2"):
        # L2 侧默认假设 L3 在另一个进程（可能是另一台机器）。
        # 地址从 deploy.yaml 的 transport 段取，不在这里写死。
        from .remote_backend import RemoteBackend
        return RemoteBackend(**kw)

    raise ValueError("未知 backend.kind=%r（可选 mujoco / real / remote）"
                     % kind)


def wrap_local(server):
    """把同进程的 ControlServer 包成 L2 可用的 RobotBackend。

    这是 inline 模式的**唯一正确用法**：L2 不直接碰 MujocoBackend，
    否则与 L3 主循环并发写同一个 MjData（实测会让机械臂纹丝不动）。
    """
    from .local_backend import LocalBackend
    return LocalBackend(server)


__all__ = ["RobotBackend", "make_backend", "wrap_local"]
