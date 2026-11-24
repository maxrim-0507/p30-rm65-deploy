# -*- coding: utf-8 -*-
"""硬件抽象层（HAL）：一套接口，两套后端。

=== 为什么要抽象 ===
部署时最容易出的事故是「仿真跑通了、真机上换了套单位/坐标系/符号，
整套上层逻辑重写」。这里把差异全部压到后端里，上层只认这 6 个方法。

接口约定（全部为**物理量**，不是归一化名目）：
  joint      6 × rad，顺序 arm_joint1..6
  gripper    标量 [0,1]，1=张开
  eef_pos    3 × m，**xyz 均相对臂座**（eef_9d 口径，见下文）
  eef_rot    3×3 旋转矩阵，**世界系**
  images     {name: HxWx3 uint8 RGB}
  objects    [{color, pos(世界系), yaw}] —— 只有仿真/标定场景能提供

=== eef_9d 的口径（2026-11-06 由数据集定死）===
    pos   = 世界 xpos - ARM_BASE                 # xyz 三个分量都减
    rot6d = 世界系 R[:2, :]
**site = grasp_site（夹持面中心）**，不是法兰 eef_site（2026-11-09）。
两者沿工具轴差 0.1113 m；旧数据学的是法兰而执行端当夹持面解 IK，
末端系统性高 9.93 cm。现在采集/导出/仿真/部署四处同源，无偏移。
**z 也是相对臂座**（2026-11-09 校准）：抓取帧实测 -0.118 = 0.592-0.710。
这是 collect_color_grasp.eef_state() 的写法，训练数据就是按它生成的。
xyz 统一相对臂座是为了让「同一瞄准目标的动作」不随底盘平移而变；
rot 保持世界系是数据集的选择，改它必须重训。
任何一端擅自换成 base_R.T @ (...) 都会在臂座 yaw=180° 时把 xy 反号。

=== 单位边界在 SDK 调用处 ===
仿真内部全是弧度；REALMAN SDK 吃**角度**。转换只允许发生在
real_backend 里那一处，别在中间层来回转。
"""
from abc import ABC, abstractmethod


class RobotBackend(ABC):
    """仿真 / 真机 / 远程 三者的共同接口。"""

    # ---------- 生命周期 ----------
    @abstractmethod
    def connect(self):
        """建立连接、加载模型、进入可执行状态。"""

    @abstractmethod
    def close(self):
        """释放资源。必须可重复调用。"""

    # ---------- 执行 ----------
    @abstractmethod
    def actuate(self, joint, gripper):
        """下发目标。

        joint: 6 × rad（目标值，不是增量）
        gripper: [0,1]，1=张开

        返回 {"ok": bool, "err": str|None}。越界/限速由后端或 L3 的安全层
        处理后再下发；这里返回的是**是否被接受**，不代表已到位。
        """

    @property
    def supports_pose_command(self):
        """后端是否接受末端位姿指令（而不是关节角）。

        === 为什么要这个开关 ===
        真机上 IK 由机械臂控制器内部算，上层只能发笛卡尔目标。仿真里
        由 MujocoBackend 用 **REALMAN 官方 Algo**（`rm_algo_ik.RmAlgoIK`）
        充当厂商 IK —— 与真机 `rm_movej_p/movel` 是同一份算法库，
        构型分支一致。这不是可选的优化，而是两种后端的**能力差异**：
        上层必须先问能力，再决定走位姿链路还是关节链路。
        """
        return False

    def actuate_pose(self, pos, rot6d, gripper):
        """下发末端位姿（eef_9d 口径），IK 由后端/厂商负责。

        pos     3 × m，xyz 均相对臂座，与 get_state()["eef_pos"] 互逆
        rot6d   世界系 R[:2, :].flatten()，6 个数，与 eef_9d 的 rot6d 一致
        gripper [0, 1]，1 = 张开

        返回 {"ok", "err", "q"?}。q 是后端实际解出来的关节角（仿真有，
        真机没有），L3 用它填充状态帧的 joint_cmd。

        缺省实现**返回 not-ok**：真机上让仿真 IK 冒充厂商 IK 会静默
        偏离几十厘米，必须显式失败。
        """
        return {"ok": False,
                "err": "%s 不支持位姿指令" % type(self).__name__}

    @abstractmethod
    def step(self, n=1):
        """推进 n 个**物理步**。真机/远程后端是空转（等真实时间流逝）。"""

    def steps_per_control(self, hz):
        """一个控制周期（1/hz 秒）对应多少个物理步。

        === 默认 1 是给「有时钟在对端」的后端用的 ===
        LocalBackend / RemoteBackend / RealBackend 的时间由 L3 主循环或
        硬件推进，技能里的 step() 只是节奏点，填 1 即可。
        但直接接 MuJoCo 跑技能时（selftest、离线实验），没人替它推进
        物理时间：30 Hz 循环里每帧只 step 1 次 = 2 ms 物理时间，一个
        本该 0.5 s 走完的下压只走了 0.03 s，机械臂还差 16° 就合爪了。
        这类失败在日志里看起来像「IK 对、执行也返回成功、就是抓空」，
        极难归因，所以把换算显式放进接口。
        """
        return 1

    # ---------- 观测 ----------
    @abstractmethod
    def get_state(self):
        """返回
        {"joint": 6×rad, "gripper": float, "eef_pos": 3, "eef_rot": 3×3,
         "t": float}
        """

    @abstractmethod
    def get_images(self):
        """返回 {camera_name: HxWx3 uint8}。没有相机时返回空 dict。"""

    # ---------- 底盘（可选能力）----------
    @property
    def supports_base(self):
        """后端是否能驱动底盘。静态抓取场景/只有臂的真机返回 False。"""
        return False

    def move_base(self, vx=0.0, vy=0.0, wz=0.0):
        """速度指令：vx/vy 为 m/s（机体系，+x = 狗头方向），wz 为 rad/s。

        === 与 P30 SendVelocity 同语义 ===
        这是**持续速度**，不是位移增量：每帧都要重发；看门狗超时后
        上层必须显式 stop_base()，否则底盘会一直按最后一条指令走。
        """
        raise NotImplementedError("%s 不支持底盘控制" % self.name)

    def stop_base(self):
        if not self.supports_base:
            return {"ok": True, "err": None}
        return self.move_base(0.0, 0.0, 0.0)

    def get_base_state(self):
        """世界系底盘位姿与机体系速度。

        返回 {"pos": (x, y)|None, "yaw": rad|None, "vel": (vx, vy, wz)}。
        无法提供时 pos/yaw 为 None —— 上层不能假设它们一定有值。
        """
        return {"pos": None, "yaw": None, "vel": (0.0, 0.0, 0.0)}

    # ---------- 仅仿真/标定可用 ----------
    def get_objects(self):
        """真值物体列表。真机后端返回 []（要靠视觉 grounding）。"""
        return []

    # ---------- 便捷 ----------
    def hold(self):
        """保持当前位置（急停/看门狗超时用）。"""
        st = self.get_state()
        return self.actuate(st["joint"], st["gripper"])

    @property
    def name(self):
        return type(self).__name__
