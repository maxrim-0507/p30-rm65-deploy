# -*- coding: utf-8 -*-
"""真机后端：RM65（REALMAN SDK）+ Dex1-1 夹爪 + LatentOS 底盘。

=== 本文件的定位 ===
真机 SDK 的**具体调用签名**依赖现场拿到的那一版 RM_API2 与 LatentOS
运行时。所以这里做两件事：
  1. 把能确定的骨架写死（单位换算、速度限幅、看门狗、连接生命周期）；
  2. 把 SDK 调用封装成 `_driver` 对象，允许注入一个假驱动做离线测试，
     不 import 硬件库时也能跑通整条链路。

=== 单位边界 ===
仿真内部全弧度；REALMAN SDK 全**角度**。转换只在本文件第 `actuate`
一处发生，上层永远不碰角度。

=== 夹爪语义 ===
Dex1-1 在仿真里 qpos 越小越张（-0.020 张开 / +0.0245 闭合）。真机
通过 RM65 末端 IO 或手爪位置指令驱动，映射方式必须现场标定。
这里保留 `_driver.set_gripper(open_ratio)` 一个抽象口，不在上层暴露
qpos。

=== 为什么 base_motion 默认关 ===
Phase 1 是静态抓取。底盘一旦动，臂基座系到世界系的变换就变，
所有已标定的目标点全部失效。等 Phase 2 把导航接进来再打开。

=== 底盘（2026-11-09：VLA 输出 base_velocity）===
supports_base 由 allow_base_motion 决定 —— 这是**唯一的开关**，
deploy.yaml 的 backend.real.allow_base_motion 直通到这里。
关闭时 move_base 返回 ok=False 而不是静默成功：上层能立刻看出
「模型说要动但底盘被禁用」，而不是误以为动了。

P30 的 SDK 接口是 `SendVelocity(vx, vy, wz)`（机体系 m/s、m/s、rad/s），
与我们的 base_velocity 口径一致，不需要坐标变换。真机驱动还没有 ——
`_driver.set_velocity` 需要现场按拿到的 LatentOS Python SDK 补，
和臂的 `move_joint_deg` 一样放在 _driver 抽象层里。
"""
import time

import numpy as np

from .base import RobotBackend


class _NullDriver:
    """离线占位驱动：不连硬件，只回放「命令即状态」。"""

    def __init__(self):
        self.q_deg = np.zeros(6)
        self.grip = 1.0
        self.connected = False

    def connect(self):
        self.connected = True

    def close(self):
        self.connected = False

    def move_joint_deg(self, q_deg, speed_deg_s, block=False):
        self.q_deg = np.asarray(q_deg, dtype=float).copy()

    def get_joint_deg(self):
        return self.q_deg.copy()

    def set_gripper(self, open_ratio):
        self.grip = float(open_ratio)

    def get_gripper(self):
        return self.grip


class RealBackend(RobotBackend):

    def __init__(self, rm_ip="192.168.1.18", rm_port=8080, gripper="rm65_io",
                 dog_sdk_path=None, allow_base_motion=False, driver=None,
                 camera_calib=None, image_source=None):
        self.rm_ip = rm_ip
        self.rm_port = int(rm_port)
        self.gripper_kind = gripper
        self.dog_sdk_path = dog_sdk_path
        self.allow_base_motion = bool(allow_base_motion)
        self._driver = driver
        self._q_cmd_deg = np.zeros(6)
        self._grip_cmd = 1.0
        self._last_t = 0.0
        self._max_speed = 90.0        # 度/秒，与 deploy.yaml 的默认值一致
        self._base = np.asarray([0.0, 0.0, 0.0])
        self._base_R = np.eye(3)
        self._base_vel_cmd = np.zeros(3)
        self._cam_calib = None      # configs/real_cameras.yaml 缓存
        self._cam_calib_path = camera_calib
        self._image_source = image_source   # callable -> {name: HxWx3}

    # ---------- 生命周期 ----------
    def connect(self):
        if self._driver is None:
            self._driver = self._make_real_driver()
        self._driver.connect()
        self._q_cmd_deg = np.asarray(self._driver.get_joint_deg(),
                                    dtype=float).copy()
        self._grip_cmd = float(self._driver.get_gripper())
        self._last_t = time.monotonic()
        return self

    def _make_real_driver(self):
        """构造 RM65 驱动。按 RM_API2 的 Python 绑定。

        === SDK 用法（现场核对过）===
            from Robotic_Arm.rm_robot_interface import RoboticArm
            arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            h = arm.rm_create_robot_arm(ip, port, level)
            arm.rm_movej([q1..q6], v, r, connect, block)   # 角度制
            arm.rm_get_current_arm_state()                 # 返回 [code, state]
            arm.rm_set_gripper_position(pos, block, timeout)

        没有 SDK 时抛异常而不是静默退化 —— 静默退化成 NullDriver 会
        让「真机没接上」看起来像「抓取失败」。
        """
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        except ImportError as e:
            raise ImportError(
                "未找到 RM_API2 的 Python 绑定（Robotic_Arm）。"
                "真机部署前先按 docs/P30_RM65_移动操作技术方案.md §10 "
                "向厂商确认安装路径，或注入 driver=_NullDriver() 做离线联调。"
            ) from e

        outer = self

        class _RM65Driver:
            def __init__(self):
                self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
                self.handle = None
                self.grip = 1.0

            def connect(self):
                # level=3 拿到完整状态（关节角 + 力 + IO）
                self.handle = self.arm.rm_create_robot_arm(
                    outer.rm_ip, outer.rm_port, 3)
                if getattr(self.handle, "id", -1) < 0:
                    raise RuntimeError(
                        "RM65 连接失败 %s:%d（err=%s）"
                        % (outer.rm_ip, outer.rm_port,
                           getattr(self.handle, "id", "?")))

            def close(self):
                try:
                    self.arm.rm_delete_robot_arm()
                except Exception:
                    pass

            def move_joint_deg(self, q_deg, speed_deg_s, block=False):
                # v 是速度百分比(1-100)，r 是加速度百分比。这里把度/秒折算成
                # 百分比：RM65 关节满速 180 °/s。
                v = int(np.clip(round(abs(speed_deg_s) / 1.8), 1, 100))
                self.arm.rm_movej([float(x) for x in q_deg], v, 0, 1, 0)

            def get_joint_deg(self):
                code, state = self.arm.rm_get_current_arm_state()
                return np.asarray(state["joint"], dtype=float)

            def set_gripper(self, open_ratio):
                # RM65 末端 IO 驱动 Dex1-1：开度 0..1 映射到 0..1000
                self.arm.rm_set_gripper_position(
                    int(np.clip(open_ratio, 0.0, 1.0) * 1000), True, 0)
                self.grip = float(open_ratio)

            def get_gripper(self):
                return self.grip

        return _RM65Driver()

    def close(self):
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass

    # ---------- 执行 ----------
    def actuate(self, joint, gripper):
        q_deg = np.degrees(np.asarray(joint, dtype=float).reshape(6))
        # 速度限幅：按上次命令到现在的时间算允许的最大跳变。
        # 宁可在这一层慢一点，也不要让驱动器自己做突跳。
        now = time.monotonic()
        dt = max(now - self._last_t, 1e-3)
        lim = self._max_speed * dt
        q_deg = self._q_cmd_deg + np.clip(q_deg - self._q_cmd_deg, -lim, lim)
        try:
            self._driver.move_joint_deg(q_deg, self._max_speed, block=False)
            self._driver.set_gripper(float(gripper))
        except Exception as e:
            return {"ok": False, "err": "%s: %s" % (type(e).__name__, e)}
        self._q_cmd_deg = q_deg
        self._grip_cmd = float(gripper)
        self._last_t = now
        return {"ok": True, "err": None}

    # ---------- 位姿链路（当前未接入）----------
    @property
    def supports_pose_command(self):
        """真机必须走厂商笛卡尔接口，当前故意返回 False。

        不是「没实现」，而是三件事没确认前写进去就是瞎做：
          1. **用哪个 API**（2026-11-06 已从 SDK 源码确认，见下）：
             **没有 `rm_movep`**。RM_API2 只有
               `rm_movej_p(pose, v, r, connect, block)`  关节空间插值到目标位姿
               `rm_movel (pose, v, r, connect, block)`  笛卡尔空间直线
             （`third_party/RM_API2/.../rm_robot_interface.py:1660` 与 `:1498`；
             控制器协议命令名分别是 `movej_p` / `movel`）
             `rm_movej_p` 的中间路径是关节插值，**不保证直线**，贴台面横移
             可能擦台面；`rm_movel` 走直线但可能因奇异/超行程失败。抓取
             取放选 `rm_movel`，远距离换位选 `rm_movej_p`。
             **单位：位置 m、姿态 rad**（两者一样）。注意 `deploy.yaml` 里
             「SDK 用角度制」只对 `rm_movej` 的关节角成立，位姿不能用度。
             `block`：多线程模式 0=非阻塞、1=阻塞；单线程模式下非 0 是超时
             秒数。100 Hz 下发必须用非阻塞 + 自己插值，否则每帧阻塞在
             "到达目标"。
          2. **坐标系**：RM_API2 的 pose 在**机器人基座系**（`p_v, R_v`），
             而 eef_9d 的口径已由数据集定死（**2026-11-09 校准**）：
                 pos = 世界系 - ARM_BASE      （xyz 三个分量都减）
                 rot6d = 世界系 R[:2, :]
             抓取帧实测 z=-0.118 = 0.592-0.710，坐实 z 也相对臂座；
             2026-11-06 写的「z 是世界高度」是错的 —— 采集脚本从第一天
             减的就是全三维 ARM_BASE。两者不是同一个系，必须显式换算：
                 R_b = Rz(MOUNT_YAW_DEG)              # mount.yaml = 180°
                 p_w = arm_base_world + R_b @ p_v
                 p_eef = p_w - arm_base_world
                 R_eef = R_b @ R_v
             臂座 yaw=180° 时 xy 取反、z 减臂座高度 —— 这一步漏了
             就是「横移方向反 + 高度差 71 cm」。
          3. **T_tool**：数据集 site 是 grasp_site（夹持面中心），与厂商
             法兰相距 0.1113 m（仿真常量，真机必须实测）。
             **不要自己乘 T_tool**：SDK 支持把我们的 grasp_site 注册成工具
             坐标系（`rm_set_manual_tool_frame(rm_frame_t(name, pose, ...))`
             + `rm_change_tool_frame(name)`），之后 `rm_get_current_arm_state()`
             返回的 pose 就是「当前工具坐标系下」的位姿，`rm_movej_p/movel`
             的入参也按该 TCP 解释。这样 T_tool 只以**一个常量**的形式出现
             在被测值里，不在代码里做矩阵乘，少一处符号错的机会。
             仿真侧已按此实现（`deployment/hal/rm_algo_ik.py` 注册
             tool=(0,0,0.1113, π/2,0,0)），真机照抄同一组常量即可。

        还剩两个必须上台架实测的量（代码层面已无未知）：
          - grasp_site 相对法兰的真实偏移（0.1113 m 是仿真常量，真机要量）；
          - 欧拉角顺序。SDK 头文件只写「rad」，没写顺序
            （`rm_define.h:157-165`）；仿真侧实测为 `Rz·Ry·Rx` 外旋
            （六种组合里误差 5.96e-8），真机可用 `rm_algo_euler2matrix`
            做一次单轴 0.5 rad 自检复核。

        这两项量出来之后，在 `_RM65Driver` 上加 `move_pose_rad()` 并在这里
        转发即可，L2/L3 的接口不用再改。
        """
        return False

    def actuate_pose(self, pos, rot6d, gripper):
        return {"ok": False,
                "err": "真机笛卡尔接口未接入：需确认 rm_movep/rm_movej_p、"
                       "坐标系口径（见 real_backend.supports_pose_command "
                       "注释）与实测 T_tool"}

    def step(self, n=1):
        # 真机的时间由硬件流逝；这里只避免调用方空转打满 CPU
        time.sleep(0.002 * int(n))

    # ---------- 底盘（LatentOS SendVelocity）----------
    @property
    def supports_base(self):
        """只有显式允许时才对外宣称支持底盘。

        allow_base_motion=False 时返回 False，L2 的 manipulate 会跳过
        base_velocity（静态任务里那三维恒 0，跳过等价于发 0）。
        """
        return self.allow_base_motion

    def move_base(self, vx=0.0, vy=0.0, wz=0.0):
        """速度指令，语义同 P30 SendVelocity：持续有效，直到下一条。

        限速/限加速度已在 L3 的 SafetyLimiter.filter_base_vel 做过，
        这里不再重复夹紧 —— 重复限幅会让两层配置不一致时互相打架。
        只做数值有效性检查（NaN/inf 绝不能送到硬件）。
        """
        if not self.supports_base:
            return {"ok": False,
                    "err": "底盘被禁用（backend.real.allow_base_motion=false）"}
        cmd = np.array([vx, vy, wz], dtype=float)
        if not np.all(np.isfinite(cmd)):
            return {"ok": False, "err": "底盘速度含 NaN/inf"}
        fn = getattr(self._driver, "set_velocity", None)
        if fn is None:
            return {"ok": False,
                    "err": "驱动未实现 set_velocity（LatentOS SendVelocity 待接入）"}
        fn(float(cmd[0]), float(cmd[1]), float(cmd[2]))
        self._base_vel_cmd = cmd
        return {"ok": True, "err": None}

    def stop_base(self):
        if not self.supports_base:
            return {"ok": True, "err": None}
        return self.move_base(0.0, 0.0, 0.0)

    def get_base_state(self):
        """真机底盘位姿/速度依赖 LatentOS telemetry 模块，当前未接入。

        返回 None 而不是零位姿：上层不能把「没读到」当成「停在原点」，
        两者在导航逻辑里后果完全不同（见 hal/base.py 同名方法注释）。
        """
        if not self.supports_base:
            return {"pos": None, "yaw": None, "vel": (0.0, 0.0, 0.0)}
        fn = getattr(self._driver, "get_base_state", None)
        if fn is None:
            return {"pos": None, "yaw": None, "vel": (0.0, 0.0, 0.0)}
        return fn()

    # ---------- 观测 ----------
    def get_state(self):
        q = np.asarray(self._driver.get_joint_deg(), dtype=float)
        return {
            "joint": np.radians(q),
            "joint_cmd": np.radians(self._q_cmd_deg),
            "gripper": float(self._driver.get_gripper()),
            "gripper_cmd": float(self._grip_cmd),
            # 真机没有可靠的 EEF 正解时用 DH 正解；这里先返回基座系零位，
            # 由 L2 通过 IK 侧的正解补齐（见 skills/kinematics.py）。
            "eef_pos": np.zeros(3),
            "eef_rot": np.eye(3),
            "t": float(time.monotonic()),
        }

    def get_images(self):
        """真机图像：由注入的 image_source() 提供 {相机名: HxWx3 RGB}。

        为什么不在本文件里决定拉流方式：RTSP / RealSense / OAK-D 的 SDK
        和线程模型都不同，选型属于现场部署形态。这里只定义契约，
        现场写一个 10 行适配器传进构造函数即可，不用再改后端。

        三条防线（都返回空 dict，让上层显式报「无视觉」）：
          - 没注入 image_source；
          - 适配器抛异常；
          - 帧不是 3 通道。
        绝不返回全黑占位图：模型会拿着垃圾画面输出垃圾动作且不报错。
        """
        if self._image_source is None:
            return {}
        try:
            imgs = self._image_source() or {}
        except Exception:
            return {}
        out = {}
        for k, v in imgs.items():
            a = np.asarray(v)
            if a.ndim == 3 and a.shape[2] >= 3:
                out[k] = a[..., :3].astype(np.uint8, copy=False)
        return out

    # ---------- 相机内外参（L3 广播给 L2）----------
    def _camera_calib(self):
        if self._cam_calib is None:
            from ..perception.real_cameras import load_cameras
            self._cam_calib = load_cameras(self._cam_calib_path)
        return self._cam_calib

    def camera_info(self):
        """每帧相机内外参，格式同 MujocoBackend.camera_info。

        === 三种「返回 {}」的情形 ===
        1. 标定文件没填（calibrated=false / 内参 null）；
        2. 底盘位姿不可用（LatentOS telemetry 未接）；
        3. 正解失败。
        全部返回空 dict 而不是默认位姿：反投影拿错外参不会报错，只会
        把目标估偏几十厘米，比「明说没有视觉」危险得多。
        """
        try:
            cal = self._camera_calib()
        except Exception:
            return {}
        b = self.get_base_state() or {}
        pos, yaw = b.get("pos"), b.get("yaw")
        if pos is None or yaw is None:
            return {}
        from ..perception.real_cameras import camera_info
        q = np.radians(np.asarray(self._driver.get_joint_deg(), float))
        try:
            return camera_info(cal, (float(pos[0]), float(pos[1]),
                                     float(yaw)), q)
        except Exception:
            return {}

    def camera_model(self, name):
        """直接给 L2 用（L2 经 LocalBackend 时走广播的同一份数据）。"""
        from ..perception.color_grounding import CameraModel
        c = self.camera_info().get(name)
        if c is None:
            return None
        return CameraModel(int(c["w"]), int(c["h"]), c.get("fovy"),
                           pos=c["pos"], rot=c["rot"],
                           fx=c.get("fx"), fy=c.get("fy"),
                           cx=c.get("cx"), cy=c.get("cy"))
