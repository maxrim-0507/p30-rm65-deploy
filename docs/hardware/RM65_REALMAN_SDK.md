# RM65 开发文档 · REALMAN SDK 离线摘要

> 来源：<https://develop.realman-robotics.com/robot/summarize/> · 抓取日期：2026-11-25
> 文档：睿尔曼**第三代控制器**开发者文档（API2 Python/C++）。版权归 RealMan Robotics，本文件为**离线摘要**，以官网为准；建议内部/私有仓库使用。

---

## 1. RM65 本体参数

| 参数 | 数值 |
|---|---|
| 自由度 | **6** |
| 构型 | 仿人构型 |
| 最大工作半径(臂展) | 标准版 **610 mm**；6FB 627 mm；6F 638.5 mm |
| 有效负载 | 5 kg |
| 自重 | 7.2 kg（六维力版 7.3 kg） |
| 重复定位精度 | **±0.05 mm** |
| 末端 TCP 线速度 | ≤1.8 m/s |
| 安装方式 | 任意角度安装；底座 φ110 mm |

关节范围 / 最大角速度：

| 关节 | 范围 | 角速度 |
|---|---|---|
| J1 | -178°～+178° | 180°/s |
| J2 | -130°～+130° | 180°/s |
| J3 | -135°～+135° | 225°/s |
| J4 | -178°～+178° | 225°/s |
| J5 | -128°～+128° | 225°/s |
| J6 | -360°～+360° | 225°/s |

> 工作空间 = 半径 610 mm 球体，去掉基座正上方/正下方两个圆柱空间。

## 2. Python SDK 安装与连接

- Python ≥ 3.9；Windows 64/32、Linux x86/ARM。
```bash
pip install Robotic_Arm
# 或源码：git clone https://github.com/RealManRobot/RM_API2.git
```
```python
from Robotic_Arm.rm_robot_interface import *

robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = robot.rm_create_robot_arm("192.168.1.18", 8080)   # 实际 IP/端口按设备
print("机械臂ID：", handle.id)
# ... 使用 ...
robot.rm_delete_robot_arm()                                # 用完释放
```
- 所有 API 返回状态码，`0`=成功。

## 3. 轨迹控制（★ 末端运动）

```python
rm_movej(self, joint: list[float], v:int, r:int, connect:int, block:int) -> int   # 关节空间到目标关节角(°)
rm_movej_p(self, pose: list[float], v:int, r:int, connect:int, block:int) -> int  # 关节空间到目标位姿
rm_movel(self, pose: list[float], v:int, r:int, connect:int, block:int) -> int    # 笛卡尔直线运动
rm_moves(self, pose: list[float], v:int, r:int, connect:int, block:int) -> int    # 样条曲线
rm_movec(self, pose_via, pose_to, v, r, loop, connect, block) -> int              # 圆弧
```

| 参数 | 含义 |
|---|---|
| `joint` | 关节目标角，**度**（RM65 用前 6 个） |
| `pose` | `[x,y,z,rx,ry,rz]`，位置**米**、姿态**欧拉角(弧度)** |
| `v` | 速度比例 1～100（相对最大速度/加速度百分比）|
| `r` | 交融半径 0～100（`connect=1` 时生效）|
| `connect` | 0=立即规划执行；1=与下一条一起规划（样条需连续 ≥3 个 connect=1）|
| `block` | 0=非阻塞；单/多线程阻塞见文档（阻塞可设超时）|

7 元素位姿 = 四元数姿态（透传接口支持）。

### 透传 / 跟随（★ 视觉伺服闭环用这个）
```python
rm_movej_canfd(self, joint, follow, expand=0, trajectory_mode=0, radio=0) -> int   # 关节透传, 最高200Hz, 不经规划
rm_movep_canfd(self, pose, follow, trajectory_mode=0, radio=0) -> int             # 位姿透传, 先逆解, 适合周期位姿调整/视觉伺服
rm_movej_follow(self, joint) -> int
rm_movep_follow(self, pose) -> int
rm_set_movev_canfd_init(self, avoid_singularity_flag, frame_type, dt) -> int      # 速度透传初始化
rm_movev_canfd(self, cartesian_velocity, follow, trajectory_mode=0, radio=0) -> int  # 笛卡尔速度透传(≤0.25 m/s, ≤0.6 rad/s)
```
- `follow=True` 高跟随，透传周期建议 ≤10 ms。
- `trajectory_mode`：0 完全透传 / 1 曲线拟合 / 2 滤波；`radio` 对应平滑系数/滤波参数。
- `rm_movep_canfd` 逆解成功且关节差小 → 直接执行，**适合视觉伺服（我们每 16 步下发一次位姿）**。

## 4. 状态查询

```python
rm_get_joint_degree(self) -> (code, list[float])        # 关节角, 单位 °
rm_get_current_arm_state(self) -> (code, dict)          # rm_current_arm_state_t,含当前 pose
rm_get_arm_all_state(self) -> (code, dict)              # rm_arm_all_state_t(全部状态)
rm_get_init_pose / rm_set_init_pose(joint)
# 关节温度/电流/电压: rm_get_current_joint_temperature/current/voltage
```
> 末端位姿字段在结构体页 `apipython/struct/currentArmState/`、`apipython/struct/pose/`（本页未展开键名）。

## 5. 与「P30 + RM65 抓取部署」的对应

| 我们的动作 | RM65 SDK |
|---|---|
| **arm actuate**：step 的 `a13[0:9]`(eef9, 臂座系) → 转真机基座系笛卡尔 | `rm_movep_canfd(pose, follow=True)`（每 16 步/帧一次，视觉伺服）或 `rm_movel(pose,v,r,0,block)`（直线阻塞）；关节直发 `rm_movej_canfd([°..])` |
| 臂取 6 轴目标关节角 | `rm_movej(q_deg, v, r, 0, 1)` |
| **夹爪** `a13[9]`∈[0,1] | 末端工具/夹爪接口（`rmPlus` 末端工具协议 / `gripper` demo）|
| 状态反馈(state16 的 joint + eef9) | `rm_get_joint_degree()` + `rm_get_current_arm_state()['pose']`，再用真机 FK 转臂座系 eef9 |
| IK（eef9→关节） | 真机控制器自带（`rm_movep_canfd`/`movej_p` 内部逆解），或 `algo` 接口 |
| 连接/生命周期 | `rm_create_robot_arm(ip,8080)` … `rm_delete_robot_arm()` |

**关键**：arm 的动作最好走 `rm_movep_canfd`（位姿透传，周期下发，正好匹配我们同步逐帧 16 步闭环）；底盘走 P30 的 `motion.send_velocity`（见 `P30_LatentOS_SDK.md`）——**两套 SDK 分工**。

## 6. 在线文档（子页）

- 概述 `/robot/summarize/` · RM65 参数 `/robot/robotParameter/RM65OntologyParameters/`
- Python 快速开始 `/robot/apipython/getStarted/`
- 轨迹控制 `/robot/apipython/classes/movePlan/` · 状态 `/robot/apipython/classes/armState/`
- 连接控制 `/robot/apipython/classes/roboticArm/` · 运动状态 `/robot/apipython/classes/motionControl/`
- 工具/工作坐标系 `/robot/apipython/classes/toolCoordinateConfig/`、`/workCoordinateConfig/`
- 算法(IK) `/robot/apipython/classes/algo/` · 关节透传示例 `/robot/demo/python/movejCANFD/`
- 末端工具/夹爪 `/robot/demo/python/gripper/` · 错误码 `/robot/apierrorList/`
