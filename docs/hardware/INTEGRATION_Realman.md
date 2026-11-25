# 对接你的 Realman(ECO65) 真机系统

本文件说明如何用你现有的 **Realman/ECO65 ROS2 系统** 实现本仓库 `grasp/grasp_hal.py::RealGraspBackend`（真机后端），
从而让「P30 底盘 + ECO65/RM65 臂 + VLA 抓取」闭环在真机跑起来。策略/闭环逻辑不动，只做 HAL。

## 1. 复用的现成件（来自你的 Realman 仓库）

| 用途 | 来源 | 关键点 |
|---|---|---|
| 机械臂控制 | ROS2 `rm_driver` 自定义 Topic | 见下 Topic 表 |
| 控制器封装 | `eco65_dashboard/controller.py::ECO65Controller` | `move_to_pose/move_straight/move_joints` 等 |
| 相机 | `scripts/camera_bridge.py`（Orbbec Gemini 336L，眼在手上） | RGB(compressed)+Depth → 共享文件/可改订阅 |
| 坐标系 | `docs/HAND_EYE_CALIBRATION.md` | **T_EC** 相机↔末端 4×4（a13 臂座系↔相机 的关键）|
| 状态/力 | `rm_driver` `/arm_state`、`/udp_*` | 关节/位姿/六维力 |
| 底盘 | **P30 LatentOS SDK** `motion.send_velocity` | 见 `P30_LatentOS_SDK.md` |

## 2. rm_driver Topic 清单（真机）

**发布（控制）**
| Topic | 消息 | 字段 | 用途 |
|---|---|---|---|
| `/rm_driver/movej_cmd` | `Movej` | `joint[6](rad)`, `speed`, `block` | 关节空间 |
| `/rm_driver/movej_p_cmd` | `Movejp` | `pose(Pose)`, `speed`, `block`, `trajectory_connect` | 关节规划到**笛卡尔位姿**(控制器内置IK) ★抓取用 |
| `/rm_driver/movel_cmd` | `Movel` | `pose(Pose)`, `speed`, `block`, `trajectory_connect` | 笛卡尔直线 |
| `/rm_driver/set_joint_pos_cmd` | `Jointpos` | 关节 - | 无规划直接关节 |
| `/rm_driver/set_gripper_pick_cmd` | `Gripperpick` | - | 夹爪抓取(力/速) |
| `/rm_driver/set_gripper_position_cmd` | `Gripperset` | 位置 | 夹爪位置 ★ |
| `/rm_driver/move_stop_cmd` / `emergency_stop_cmd` | Empty/Stop | - | 停/急停 |

**订阅（状态）**
| Topic | 消息 | 内容 |
|---|---|---|
| `/rm_driver/arm_state` | `Armstate` | 机械臂状态 |
| `/rm_driver/udp_arm_position` | `Pose` | 末端位姿(基座系) ★ eef9 来源 |
| `/rm_driver/udp_joint_pose_euler` | `Jointposeeuler` | `position[3]`+`euler[3]` |
| `/rm_driver/udp_six_force` | `Sixforce` | 六维力（judge/力控）|

`Movejp/Movel` 的 `pose` = `geometry_msgs/Pose`：`position(x,y,z)` 米 + `orientation(四元数)`；
四元数由 roll/pitch/yaw 标准公式换算（见 controller.py）。

## 3. 字段级映射（← 我们的 obs / a13）

### observe()（每帧）
```
front, arm  = 真机相机读两路图（camera_bridge；20~30Hz）
pose        = /rm_driver/udp_arm_position           # 末端在基座系 Pose
joint6      = /rm_driver/arm_state 或 udp_joint_pose_euler # 6 关节 rad
# —— 转成训练口径 eef9(臂座系) ——
eef_pos(3)  = R_base^T @ (pose.position - base_xyz)      # base=Rz(yaw)·(机身后移)
rot6d(6)    = (R_base^T @ R_from_quat(pose.orientation))[:2,:].flatten()
grip_norm   = (0.0245 - gripper_qpos)/(0.0245+0.020)
state16     = [eef_pos, rot6d, grip_norm, joint6]
```
相机标定：用 **T_EC**（手眼）+ 若 arm_cam 固定在末端，图与训练一致需按安装位对齐（ECO65 vs RM65 安装可能不同，首测核对）。

### step(a13)（每帧）
```
eef9=a13[0:9]; grip=a13[9]; vx,vy,wz=a13[10:13]
# 臂：臂座系 → 基座系
pos_base = R_base @ eef_pos + base_xyz
R_tgt    = R_base @ R_from_rot6d(rot6d);  quat = R_to_quat(R_tgt)
pub Movejp(pose=Pose(pos_base, quat), speed=??, block=False,
           trajectory_connect=0)    # 非阻塞，按 16步/帧周期持续下发（视觉伺服）
# 夹爪
pub Gripperset(position = map(grip))   # grip∈[0,1] → 夹爪开度
# 底盘
P30 motion.send_velocity(vx, vy, wz)   # 机体系
# 等一个控制周期（~16×32ms 或真机控制频率）
```

## 4. 关键坑 / 注意

1. **坐标系**：a13 是「RM65 臂座系」，真机 rm_driver 用「基座系」——必须用 `R_base,yaw` 换算（同仿真 `MujocoBackend.get_state/actuate_pose` 的口径）。相机侧用 **T_EC 手眼**。这两处是唯一易错点。
2. **型号差异**：你实物是 **ECO65**（6DOF 5kg 0.65m），模型 v36 为 RM65 训——控制接口同，但**末端/相机安装位不同** → `observe()` 的图与 state 要先在 ECO65 上核对到训练口径（同尺寸/内参/相机相对末端位姿）。
3. **伺服频率**：抓取每帧发 `movej_p_cmd`（block=False 非阻塞）+ 等一个周期；**不要用 block=True 长阻塞**（会卡死闭环）。速度 `speed` 先在低速(如 20~30)验证。
4. **夹爪映射**：`grip_norm` 的训练口径是 `(0.0245-q)/(0.0445)`；真机夹爪位置/力需标一个对照表。
5. **安全**：真机首测 `move_stop_cmd`/`emergency_stop_cmd` 就位；先空跑 observe/step 单步，再整闭环。
6. **两套 SDK 分工**：臂 = rm_driver(ROS2)；底盘 = P30 LatentOS `motion`；各走各的。

## 5. 建议落地顺序

1. 在真机跑你的 `start_real_robot.sh` + `camera_bridge.py`，确认 `movej_p_cmd`、`udp_arm_position`、相机图都通。
2. 实现/核对 `RealGraspBackend.observe()`：单帧离线比对「真机图/state vs 训练首帧口径」。
3. 实现 `step()`：单步发一个固定笛卡尔位姿，看末端到位 + 夹爪动作 + 底盘动。
4. 再上完整 `run_grasp_loop`（慢速、E-stop 就位）。
