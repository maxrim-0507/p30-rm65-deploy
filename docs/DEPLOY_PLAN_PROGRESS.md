# P30 + RM65 真机部署计划与进度

> 最后更新：2026-11-27 · 主控机：**THOR2**（`rsh.py 2`）
> 本文是项目级「安排 + 进度表」的唯一入口，每轮工作后更新。

## 0. 目标与边界

- **目标**：真机全自主 **SEARCH → DOCK → GRASP → VERIFY**（轮式狗 P30 + REALMAN ECO65 + GR00T N1.7-3B）。
- **模型**：`outputs/groot_rel_v36/checkpoints/020000`（闭环上限 ~91.7%）。
- **边界**：sim2real 迁移训练**不在**本轮范围。
- **算力分工**：THOR2 = 上层（VLA、感知、决策）；Orin = 移动/相机中转（7GB，跑不动 3B）；RK3588 = 运动控制 + SDK 服务端（不登录、只用 API）。

## 1. 系统拓扑（2026-11-27 实测）

```
THOR2 enP2p1s0 (RTL8126) ── 狗主网口(RK3588 eth0) ── 狗内网 192.168.1.0/24
 192.168.20.26/24 + 192.168.1.50/24(NM: thor-link, 已持久化)
   ├── Orin        192.168.1.100   ssh nano/123（已配免密）
   ├── RK3588      192.168.1.123   zenoh tcp/7447, http/8554(RTSP), http/80(LatentOS Console, 需登录)
   ├── 雷达        192.168.1.122
   ├── 相机        192.168.1.11
   └── RM65 机械臂 192.168.1.18    API tcp/8080（ECO_65, 三代控制器）
```

关键结论：
1. THOR2→Orin 是二层直达（TTL=64），RK3588 只是同段邻居，不参与转发。
2. **狗外部 USB 口在 RK3588 上，不归 Orin**（逐口断电 + 鼠标 LED 验证）；Orin 的 8 个 USB-A 口是内部的（Realtek 4 口 Hub ×2，ppps）。
3. 狗 Type-C 是 **device 口**（role 切 host 也无效，无 host 控制器）。
4. 标 "ORIN" 的 100M 口实际对端是 **RK3588 的 eth1**（MAC `...f5:c0`，与 eth0 `...f5:bf` 连续），且两网卡同 IP → 非对称路由，不可用。
5. RM65 三代控制器 RS485 **只支持 Modbus RTU（≤460800）**，Dex1-1 是 4Mbps 自定义协议 → **夹爪不能走臂的 RS485**。

## 2. 阶段计划与状态

| 阶段 | 内容 | 状态 | 说明 |
|---|---|---|---|
| 0 | 前置确认：网络/拓扑/账号/算力分工 | ✅ 完成 | 见 §1 |
| 1 | SDK 就位（LatentOS / Realman / Orbbec） | ✅ 完成 | Orin: LatentOS；THOR2: `Robotic_Arm 1.1.7` + `pyorbbecsdk2 2.1.3` |
| 2 | P30 底盘 `send_velocity` + 急停 | 🟡 进行中 | 组/策略已通（sit_down/stand/locomotion+default），**速度指令不生效**，见 §4 |
| 3 | RM65 控制 + 坐标转换 | 🟡 部分完成 | 臂联通可读状态；夹爪（Dex1-1）待 USB 串口板 |
| 4 | 相机 + 手眼标定 `T_EC` + 观测口径 | 🟡 部分完成 | front_0 RTSP + Orbbec(彩色/深度/内参) 已通；**arm_cam 接 THOR2、标定待做** |
| 5 | `RealGraspBackend` 6 接口真机化 | ⬜ 未开始 | 骨架 repo `grasp/grasp_hal.py` |
| 6 | SEARCH + DOCK 真机化 | ⬜ 未开始 | 仿真 16/16 已验证 |
| 7 | VLA 抓取 | ⬜ 未开始 | 依赖 4/5 |
| 8 | 端到端 + 安全 + 多 trial | ⬜ 未开始 | |
| 9 | 收尾 | ⬜ 未开始 | |

## 3. 已完成（按日期）

### 2026-11-25
- front_0：`rtsp://192.168.1.123:8554/front_0`，1280×720@~33fps；SEARCH HSV 检测 demo ~30fps。
- arm_cam：Orbbec Gemini 336L + `pyorbbecsdk`（RGB 1280×800@30、Y16 深度、AlignFilter、内参 fx=fy=620.3/cx=640/cy=404）。
- 网络：`192.168.1.50/24` 写入 NM；确认 "Orin 直连口" 不可用。
- 环境副作用：pip 把 `av 15.1→12.3`、`opencv-python→5.1.0.95`（torch/lerobot 正常）。

### 2026-11-27
- **机械臂**：ECO_65 @ `192.168.1.18:8080`，可读关节角/位姿/型号；`Robotic_Arm` 装好。
- **夹爪**：Dex1-1 走 RM65 RS485 不可行（三代仅 Modbus RTU，4Mbps 不支持）；需 USB 串口板 → THOR2（4Mbps, M4010 协议）。
- **MOTION 排查**：见 §4，找到 `AllowsSdkVelocity` 关键线索。
- **USB/相机**：狗外部 USB 全在 RK3588；Orin USB 8 口（ppps，可用 `uhubctl` 逐口断电）；Type-C device 模式。
- **运维**：THOR2→Orin 免密 SSH；front_0 推流曾停，已用 SDK 重启；`uhubctl` 已装到 Orin。
- **抓取物体规格**（训练依据，见 §7）：6cm 立方体、50g、红绿蓝黄四色、台面 0.65m。

## 4. 下一步：MOTION（速度不生效的根因排查）

### 已确认
- 组状态机：`passive → stand → locomotion(default) → SendVelocity`；`locomotion` 必须带 policy（槽位名 `default`/`policy1`）。
- `stand`/`sit_down` 切换正常；`switch_group("locomotion","default")` accepted；`switch_current_policy("default")` = unchanged（已生效）。
- `set_remote_velocity_control` True/False、`set_velocity_smoothing` False、20Hz/100Hz、vx±/vy/wz 全部试过 → **狗不动**（前后帧特征匹配 scale≈1.000）。

### 关键线索
- SDK 的 `send_velocity` 发到 **候选话题** `latentos/mc/cmd_candidate/sdk`（"candidate" = 需仲裁）。
- 狗上导航程序 `SlamNaviAppCppNode`（Orin）日志有 `controller velocity cmd hz≈26.8` —— **它一直在发速度候选**，很可能压掉了 SDK 候选。
- 导航程序二进制里有开关 **`AllowsSdkVelocity`**（允许 SDK 速度）。

### 下次上电执行顺序
1. `switch_group("passive")` → `stand` → `locomotion("default")`（补齐文档流程）；
2. 发 `0.1 m/s × 1s`，看是否动；
3. 查导航进程日志（`pgrep -af SlamNavi` / journal）确认 `controller velocity cmd`；
4. `POST /api/v1/session/mode/query`，尝试 `mapmode/waypoint` enter/leave，找手动/遥控模式；
5. dump 导航二进制全部 `/api` 路由（上次截断），找控制权/参数接口；
6. 若确认导航占用：**临时停 `SlamNaviApp`** 再发速度（可恢复）；
7. 通了之后：零速→0.05→0.1 m/s 标定"指令速度→实际距离"（卷尺），再做 `move_distance(d)` 时间×标定（开环，~5-15% 误差）；后续可加前视视觉里程计闭环。

## 5. 已知问题 / 风险

1. **MOTION 速度不生效**（当前最大阻塞，见 §4）。
2. 夹爪 Dex1-1 需 USB 串口板（4Mbps）→ THOR2；RM65 RS485 不可行。
3. arm_cam 装臂后必须做手眼标定 `T_EC`；羽树相机视场/位姿与训练不一致需评估/改仿真。
4. front_cam 视角 OOD：真机 front_0 与训练 front_cam 需对齐。
5. 狗外部 USB 口不可用（在 RK3588）；相机接 THOR2。
6. `av`/`opencv` 版本被 SDK 改动（lerobot 解码异常优先怀疑 av 12.3）。
7. 安全：所有运动测试先只读→小速度→急停，现场有人守急停。

## 6. 关键参数速查

```
# 网络
THOR2 enP2p1s0: 192.168.20.26/24 + 192.168.1.50/24 (NM: thor-link)
Orin:    192.168.1.100  ssh nano/123（免密已配）
RK3588:  192.168.1.123  zenoh 7447 / RTSP 8554 / Web 80(需登录)
RM65:    192.168.1.18:8080 (ECO_65, controller v3)

# 相机
front_cam: rtsp://192.168.1.123:8554/front_0  (1280x720@~33fps, 卡时用 SDK RESTART)
arm_cam:   Orbbec Gemini 336L (THOR2 USB, 2bc5:0807)
           彩色 1280x800@30 / 深度 Y16 1280x800@30 / AlignFilter(COLOR_STREAM)
           内参 fx=fy=620.3 cx=640 cy=404

# 环境
venv: /home/nio/mjl_collect_env (py3.12, torch 2.11+cu130)
udev: /etc/udev/rules.d/99-obsensor-libusb.rules (2bc5, 0666)

# LatentOS SDK (Orin)
PYTHONPATH=/data/latentos/sdk_cpp/python/site-packages:/data/latentos/sdk_runtime/python/site-packages
client.yaml -> connect.endpoints: [tcp/192.168.1.123:7447]
运动: switch_group(group[, policy]) / set_remote_velocity_control / set_velocity_smoothing / send_velocity(x,y,yaw)
话题: latentos/mc/cmd_candidate/sdk
相机: query_camera_capabilities / send_camera_stream_command(id, START/RESTART/STOP, rtsp_ip)

# Realman SDK (THOR2 venv)
from Robotic_Arm import rm_robot_interface as I
arm=I.RoboticArm(I.rm_thread_mode_e.RM_TRIPLE_MODE_E); arm.rm_create_robot_arm("192.168.1.18",8080)
rm_get_robot_info / rm_get_joint_degree / rm_get_current_arm_state
```

## 7. 抓取物体规格（训练依据）

- 形状/尺寸：**立方体 6×6×6 cm**（MuJoCo 半边长 0.03）；质量 ≈50g；摩擦高、哑光。
- 颜色（RGBA）：red(0.85,0.15,0.12) / green(0.15,0.60,0.25) / blue(0.15,0.30,0.80) / yellow(0.90,0.75,0.10)。
- 数量：随机 6 个，间距 ≥7.5cm，摆在臂可达区（x 0.47~0.62m，y ±0.19m）。
- 台面：顶面 **z=0.65m**，约 0.58×0.6m，摩擦 0.9。
- 语言：`抓取{颜色}的方块` / `把{颜色}的方块拿起来` / `pick up the {color} cube`。
- 成功：抬升 >5cm + 颜色匹配 + 不碰落干扰物；时限 10s（300 帧@30fps）。
- 夹爪：Dex1-1 可夹约 4~13cm（6cm 正好）。
- 日常替代：5~6cm 小方盒/积木包哑光彩纸 + 配重到 50g（球/圆柱/透明/反光不可用）。

## 8. 采购清单（待买）

1. H3C S1G Pro 千兆交换机 ×1 + Cat6 网线 ×3（狗/THOR2/机械臂同网）
2. USB-C 公→USB-A 公 数据线 ×2（USB3.2，相机用）
3. 带 5V/2A 外接电源的 USB 3.0 Hub ×1（相机供电；狗/THOR2 总线供电不够）
4. Dex1-1 USB 串口板（4Mbps，若套件内没有；夹爪控制必需）
