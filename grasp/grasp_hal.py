# -*- coding: utf-8 -*-
"""抓取闭环 HAL：策略 / 闭环 / 可插拔后端 三层。

 策略(不换)  vla.LocalGr00tPolicy
 闭环(不换)  run_grasp_loop —— 读观测→VLA→执行16步，同步逐帧（勿用异步服务）
 后端(可换)  真机 = RealGraspBackend（rm_driver Topic + P30 LatentOS，见 docs/hardware/INTEGRATION_Realman.md）
             仿真 = ColorGraspEnv（examples/回归）

后端接口契约（见 GraspBackend）：
  reset(layout, q_init)  set_base_pose(x,y,yaw)  observe()  step(a13)  judge(color)  solve_pregrasp(layout,color)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from vla.local_gr00t_policy import LocalGr00tPolicy  # noqa: E402

CKPT_DEFAULT = os.path.join(ROOT, "checkpoints", "groot_rel_v36", "020000")
FWD2 = np.radians([0.0, -13.6, 134.9, 0.0, -16.2, -90.0])


class GraspBackend:
    def reset(self, layout, q_init):
        raise NotImplementedError

    def set_base_pose(self, x, y, yaw):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    def step(self, a13):
        raise NotImplementedError

    def judge(self, color):
        raise NotImplementedError

    def solve_pregrasp(self, layout, color):
        raise NotImplementedError


# ----------------------------------------------------------------------------
# 真机后端：ROS2 rm_driver(臂) + P30 LatentOS(底盘) + 真机相机
# 依赖真机环境（rclpy / rm_ros_interfaces / latentos_sdk），故全部 lazy import。
# ----------------------------------------------------------------------------
class RealGraspBackend(GraspBackend):
    def __init__(self, robot_ip="192.168.1.18", p30_client_config="",
                 calib_tec=None, speed=25, dof_base_xyz=(0.0, 0.0, 0.71)):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Pose, Point, Quaternion
        from rm_ros_interfaces.msg import Movej, Movejp, Movel, Gripperset
        from rm_ros_interfaces.msg import Armstate

        self._np = np
        self._Pose, self._Point, self._Quaternion = Pose, Point, Quaternion
        self._Movej, self._Movejp, self._Movel, self._Gripperset = (
            Movej, Movejp, Movel, Gripperset)
        self.speed = int(speed)
        self.base_xyz = np.asarray(dof_base_xyz, float)   # 臂座在底盘系（z≈0.71）
        self.T_EC = None if calib_tec is None else np.asarray(calib_tec, float)  # 手眼 4x4（相机外参）
        self._arm_state = None
        self._pose = None          # 末端 Pose（基座系）
        self._joint = np.zeros(6)
        self._grip_q = 0.0

        rclpy.init()
        self.node = Node("grasp_real_backend")
        self.pub_movejp = self.node.create_publisher(Movejp, "/rm_driver/movej_p_cmd", 10)
        self.pub_movel = self.node.create_publisher(Movel, "/rm_driver/movel_cmd", 10)
        self.pub_movej = self.node.create_publisher(Movej, "/rm_driver/movej_cmd", 10)
        self.pub_grip = self.node.create_publisher(Gripperset, "/rm_driver/set_gripper_position_cmd", 10)
        self.node.create_subscription(Armstate, "/rm_driver/arm_state", self._on_state, 10)
        self.node.create_subscription(Pose, "/rm_driver/udp_arm_position", self._on_pose, 10)

        # 底盘：P30 LatentOS
        from latentos_sdk import Session  # pip install Robotic... 否 -> latentos_sdk
        self.motion = None
        try:
            self.session = Session(client_config_path=p30_client_config)
            from latentos_sdk.motion import MotionClient  # 见 P30 文档模块名
            self.motion = MotionClient(self.session)
            self.motion.acquire_control()          # 申请底盘速度控制权
        except Exception as e:  # 底盘可选/延迟连接
            print("[real] P30 底盘未连：%r" % e, flush=True)

    # -- 状态回调 --
    def _on_state(self, m):
        self._arm_state = m

    def _on_pose(self, m):
        self._pose = m

    # -- 工具：四元数/旋转 --
    @staticmethod
    def _quat_from_R(R):
        t = np.trace(R)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            return (0.25 * s, (R[2, 1] - R[1, 2]) / s,
                    (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
        # 简化：一般姿态走上面分支；奇异姿态按需补

    @staticmethod
    def _R_from_rot6d(r6):
        a1 = np.asarray(r6[:3]); a2 = np.asarray(r6[3:6])
        b1 = a1 / (np.linalg.norm(a1) + 1e-9)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 /= (np.linalg.norm(b2) + 1e-9)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=1)

    def _base_R(self):
        # 底盘 yaw -> Rz；此处读取 P30/odom，占位用单位阵（真机填 yaw）
        yaw = 0.0
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    # -- 接口实现 --
    def reset(self, layout, q_init):
        from rm_ros_interfaces.msg import Movej
        msg = Movej()
        msg.joint = list(np.asarray(q_init, float).reshape(6))  # controller 用 radians
        msg.speed = self.speed
        msg.block = True
        self.pub_movej.publish(msg)
        # 推进 ROS 一圈等待到位
        for _ in range(50):
            import rclpy
            rclpy.spin_once(self.node, timeout_sec=0.05)
            break
        return {"ok": True}

    def set_base_pose(self, x, y, yaw):
        # 底盘定位：P30 由 VLA 自己开，这里通常只读当前位姿返回 ok
        return {"ok": True}

    def observe(self):
        cam = self._read_cameras()           # TODO: 接 camera_bridge / ROS2 图像
        front = cam.get("front"); arm = cam.get("arm")
        pose = self._pose
        if pose is None:
            eef_pos = np.zeros(3); rot6 = np.zeros(6)
        else:
            R = self._base_R().T
            p = np.array([pose.position.x, pose.position.y, pose.position.z])
            eef_pos = R @ (p - self.base_xyz)
            # orientation 四元数 -> R -> 臂座系 -> rot6d
            q = pose.orientation
            Rq = self._quat_to_R(q)
            Rb = R @ Rq
            rot6 = np.concatenate([Rb[:, 0], Rb[:, 1]])
        grip = float(self._grip_q)
        state = np.concatenate([eef_pos, rot6, [grip], self._joint]).astype(float)
        return {"front": front, "arm": arm, "state": state,
                "eef_pos": eef_pos, "base_xy": (0.0, 0.0), "base_yaw": 0.0}

    def _quat_to_R(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    def _read_cameras(self):
        # TODO 真机实现：从 camera_bridge 的 /tmp/camera_latest.jpg 读，或订阅 ROS2 图像
        # 需两路：front（前视）/ arm（腕部）。返回 uint8 HWC。
        return {"front": None, "arm": None}

    def step(self, a13):
        a = np.asarray(a13, float).reshape(-1)
        eef_pos = a[0:3]; rot6 = a[3:9]; grip = float(np.clip(a[9], 0, 1))
        vx, vy, wz = a[10], a[11], a[12]
        # 臂：臂座系 -> 基座系
        Rb = self._base_R()
        pos_base = Rb @ eef_pos + self.base_xyz
        R_tgt = Rb @ self._R_from_rot6d(rot6)
        qx, qy, qz, qw = self._quat_from_R(R_tgt)
        msg = self._Movejp()
        msg.pose = self._Pose(position=self._Point(x=float(pos_base[0]),
                            y=float(pos_base[1]), z=float(pos_base[2])),
                            orientation=self._Quaternion(x=qx, y=qy, z=qz, w=qw))
        msg.speed = self.speed
        msg.block = False            # 非阻塞，逐帧伺服
        msg.trajectory_connect = 0
        self.pub_movejp.publish(msg)
        # 夹爪
        gm = self._Gripperset();
        # TODO: 字段按 rm_ros_interfaces.msg.Gripperset；grip∈[0,1] -> 真机开度表
        self.pub_grip.publish(gm)
        # 底盘
        if self.motion is not None:
            self.motion.send_velocity(float(vx), float(vy), float(wz))
        # 等一个控制周期（~16*32ms，或真机控制频率）
        import time; time.sleep(0.512)
        return {"ik_err": 0.0, "tilt": 0.0, "rejected": False}

    def judge(self, color):
        # TODO: 相机/六维力判抬升
        return {"success": False, "lift": 0.0, "picked": None}

    def solve_pregrasp(self, layout, color):
        # TODO: 查表 / 真机 movej_p 内置 IK；返回 (q0,pre,hov,grp,p) 或 None
        return None


def make_sim_backend():
    try:
        from rollout_color_grasp import ColorGraspEnv, HERE
    except Exception as e:  # noqa
        raise RuntimeError("仿真后端不可用（需 simulation/mujoco + mesh）：%r" % e)
    return ColorGraspEnv(view=False, scene=os.path.join(HERE, "p30_rm65_nav.xml"))


def run_grasp_loop(backend, policy, layout, color, instruction,
                   base=None, q_init=FWD2, max_steps=600, receding=40):
    info = backend.solve_pregrasp(layout, color)
    if info is None:
        return {"success": False, "reason": "pregrasp IK 不可达"}
    q0, pre, hov, grp, p = info
    backend.reset(layout, q0 if q_init is None else q_init)
    if base is not None:
        r = backend.set_base_pose(*base)
        if not r.get("ok"):
            return {"success": False, "reason": r.get("err")}
    chunk = []
    done = 0
    while done < max_steps:
        if not chunk:
            obs = backend.observe()
            chunk = policy.infer(obs, instruction, frame_t=done)
            if not chunk:
                return {"success": False, "reason": "empty chunk"}
        for a in chunk[:max(1, receding)]:
            backend.step(np.asarray(a, float))
            done += 1
        chunk = []
    return backend.judge(color)
