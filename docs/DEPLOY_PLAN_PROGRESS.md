# P30 + RM65 真机部署计划与进度

> 最后更新：2026-11-25 · 主控机：**THOR2**（`rsh.py 2`）
> 本文是项目级「安排 + 进度表」的唯一入口，每轮工作后更新。

## 0. 目标与边界

- **目标**：真机全自主 **SEARCH → DOCK → GRASP → VERIFY**（轮式狗 P30 + REALMAN RM65 + GR00T N1.7-3B）。
- **模型**：`outputs/groot_rel_v36/checkpoints/020000`（闭环上限 ~91.7%）。
- **边界**：sim2real 迁移训练**不在**本轮范围；本轮只做"仿真已验证策略 → 真机落地"。
- **算力分工**：THOR2 = 上层（VLA、感知、决策）；Orin Nano = 移动/相机中转（7GB 内存，跑不动 3B）；RK3588 = 运动控制 + SDK 服务端（**不登录、不改造，只用其 API**）。

## 1. 系统拓扑（2026-11-25 实测）

```
THOR2 enP2p1s0 (RTL8126)  ──1000M──  狗内部交换机
 192.168.20.26 (thor-link)              ├── Orin Nano  192.168.1.100   ssh nano/123
 192.168.1.50/24 (辅助，已持久化)        ├── RK3588    192.168.1.123   zenoh tcp/7447, http/8554
                                        ├── 雷达      192.168.1.122
                                        └── 相机      192.168.1.11
```

关键结论：
1. **不是点对点直连 Orin**：同段可见 4 个设备 MAC；到 `.100` TTL=64（无路由），Orin 是二层直达，**不依赖 RK3588 的软件**。
2. **只用 RK3588 的 SDK API**（`tcp/192.168.1.123:7447`），不登录/不改造它。
3. 另有一个"100M 口"（用户称 Orin 直连口）：实测只宣告 10/100M、60s 零帧、261 个探测 + WOL 全无回应 → **该口不可用**（对端无活网络栈）；Orin 只有一张网卡且已在内部交换机上，物理上不存在"单独直连 Orin"。

## 2. 阶段计划与状态

| 阶段 | 内容 | 状态 | 说明 |
|---|---|---|---|
| 0 | 前置确认：网络/拓扑/账号/算力分工 | ✅ 完成 | 见 §1 |
| 1 | SDK 就位（LatentOS / Realman / Orbbec） | ✅ 完成 | Orin 上 LatentOS SDK 可用；THOR2 上 Orbbec SDK 已装 |
| 2 | **P30 底盘 `send_velocity` + 急停** | ⏳ **明日重点** | 待做：状态查询→小速度→转向→急停 |
| 3 | RM65 控制 + 坐标转换（`rm_driver`） | ⬜ 未开始 | 待臂上电/确认网络与凭据 |
| 4 | 相机 + 手眼标定 `T_EC` + 观测口径 | 🟡 部分完成 | front_0(RTSP)+arm_cam(彩色/深度/内参)已通；**装臂 + 标定待做** |
| 5 | `RealGraspBackend` 6 接口真机化 | ⬜ 未开始 | 骨架已有（repo `grasp/grasp_hal.py`） |
| 6 | SEARCH（臂转）+ DOCK（3-phase yaw-first）真机化 | ⬜ 未开始 | 仿真 16/16 已验证策略 |
| 7 | VLA 抓取（真机回放/部署） | ⬜ 未开始 | 依赖阶段 4/5 |
| 8 | 端到端 + 安全 + 多 trial | ⬜ 未开始 | 急停/看门狗/超时必须有 |
| 9 | 收尾：文档、参数、复现 | ⬜ 未开始 | |

## 3. 已完成（2026-11-25）

- **front_cam（狗前视）**：`rtsp://192.168.1.123:8554/front_0`，1280×720@~33fps；THOR2 上 OpenCV(FFMPEG, `rtsp_transport;tcp`) 可直接拉流。
- **SEARCH 检测 demo**：HSV 7 色块检测 + 标注，1280×720 实时 ~30fps；最大色块在画面右侧（蓝色水瓶 ~3900px）。
- **arm_cam（臂上，未装臂）**：Orbbec Gemini 336L 已接 THOR2；`pyorbbecsdk 2.1.3` 安装完成；彩色 RGB 1280×800@30、深度 Y16 1280×800@30、`AlignFilter(COLOR_STREAM)` 对齐、内参 fx=fy=620.3 / cx=640 / cy=404。
- **网络**：`192.168.1.50/24` 写入 NM 配置 `thor-link`（重连/重启不再丢）；确认"Orin 直连口"不可用。
- **环境副作用**：装 SDK 时 pip 把 `av 15.1→12.3`、`opencv-python→5.1.0.95`；torch 2.11/lerobot 0.6.1 仍正常。

## 4. 明日计划：MOTION 专项（底盘控制与测试）

> 验收原则：先**只读**、再**小速度**、每步都有**超时/急停**；测试区域清空、狗放地面、有人站在急停位。

| # | 任务 | 验收标准 |
|---|---|---|
| M1 | 确定控制链路：THOR2 直连 SDK vs Orin 中转（推荐后者，SDK 已在 Orin） | 能稳定 `query_motion_control_state` |
| M2 | 只读状态：IMU / joint_state / power / motion state | 数值合理、无超时 |
| M3 | 获取控制权（`set_remote_velocity_control` / `switch_group` 等，按 SDK 语义） | 状态回执 OK |
| M4 | `send_velocity(vx=0.05, vy=0, yaw=0)` 直线 0.5m | 实际位移与指令一致（标定比例） |
| M5 | 原地转向 `yaw=±0.2 rad/s` 5s | 角度闭环误差记录 |
| M6 | 侧移 `vy` 测试 | 方向正确、无异常 |
| M7 | 急停/超时保护（速度归零看门狗） | 断开控制 0.5s 内停 |
| M8 | 速度标定表（vx/vy/yaw → 实际）并写入配置 | 复测误差 < 10% |

## 5. 已知问题 / 风险

1. **"Orin 直连口" 不可用**（100M、零帧）；用 RK3588 千兆口。
2. **front_cam 视角 OOD**：真机 `front_0` 的安装位姿/内参需与训练时的 front_cam 尽量一致。
3. **arm_cam 未装臂**：装臂后必须做手眼标定（`T_EC`）。
4. **av/opencv 版本被 SDK 改动**：若 lerobot 视频解码异常，优先怀疑 `av 12.3`。
5. **RK3588 SSH 凭据未知**：不依赖；只用 SDK API。
6. **安全**：所有运动测试必须有软件急停 + 物理急停位人员。

## 6. 关键参数速查

```
# 网络
THOR2 enP2p1s0: 192.168.20.26/24 + 192.168.1.50/24 (NM profile: thor-link)
Orin Nano:      192.168.1.100   ssh nano/123
SDK 服务端:     192.168.1.123    zenoh tcp/192.168.1.123:7447

# 相机
front_cam: rtsp://192.168.1.123:8554/front_0   (1280x720@~33fps)
arm_cam:   Orbbec Gemini 336L (USB 2bc5:0807)
           彩色 RGB 1280x800@30 / 深度 Y16 1280x800@30
           AlignFilter(OBStreamType.COLOR_STREAM)
           内参 fx=fy=620.3 cx=640 cy=404

# 环境
venv: /home/nio/mjl_collect_env   (py3.12, torch 2.11+cu130)
udev: /etc/udev/rules.d/99-obsensor-libusb.rules  (idVendor 2bc5, 0666)

# LatentOS SDK (Orin)
PYTHONPATH=/data/latentos/sdk_cpp/python/site-packages:/data/latentos/sdk_runtime/python/site-packages
client.yaml -> connect.endpoints: [tcp/192.168.1.123:7447]
相机: query_camera_capabilities / send_camera_stream_command(id, START/STOP, rtsp_ip)
底盘: send_velocity(x,y,yaw)  [机体系 m/s, rad/s]
```
