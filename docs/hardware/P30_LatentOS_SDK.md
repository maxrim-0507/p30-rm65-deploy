# P30 开发文档 · LatentOS SDK 离线摘要

> 来源：<https://latent-action.com/docs/next/> · 抓取日期：2026-11-25
> 文档版本：**1.3.0（未发布，可能变动）**；最新已发布 **1.2.1**
> 版权归 Latent-Action Robotics。本文件为**离线摘要**，一切以官网为准；建议仅内部/私有仓库使用。

---

## 1. P30 产品规格

| 分类 | 参数 | 标准版 | 长续航版 |
|---|---|---|---|
| 机械 | 站立尺寸(长×宽×高) | 960×460×670 mm | 960×460×670 mm |
|  | 重量 | 45 kg | 55 kg |
|  | 自由度 | 12 DOF | 12 DOF |
|  | 最大关节扭矩 | 210 Nm | 210 Nm |
|  | 额定/极限负载 | 30 kg / 150 kg | 20 kg / 140 kg |
|  | 防护等级 | IP67 | IP67 |
| 电池 | 容量 | 20 Ah (936 Wh) | 2×20 Ah (1872 Wh) |
|  | 额定电压 | 46.8 V | 46.8 V |
|  | 充电功率 | 500 W(可选 1000 W 快充) | 同 |
| 续航 | 空载 / 额定负载 | 5 h/20 km；3 h/12.5 km | 10 h/40 km；3 h/12.5 km |
| 算力 | 控制板 / 算力板 | 8 核 CPU / 支持定制 | 同 |
| 运动 | 速度 | 0–4 m/s | 0–4 m/s |
|  | 楼梯/高台/斜坡 | 30 cm 工业楼梯 / 100 cm / 45° | 同 |
| 通信接口 | 无线 | WiFi·蓝牙·4G(可选5G)·GNSS·RTK(选配) | 同 |
|  | 对外供电 | BAT / 24V / 12V / 5V(USB) | 同 |
|  | 对外接口 | 千兆网口·USB3.0·USB2.0·RS485·CANFD | 同 |
| 传感 | 深度相机 / 激光雷达 / 全向避障 | 标配 / 选配 / 选配 | 同 |

## 2. LatentOS SDK 概述

三种使用方式（Python 是 C++ 的原生绑定；ROS2 驱动调 C++ SDK 暴露 Topic/Service）：
- **C++**（原生）· **Python**（脚本/应用）· **ROS 2**（ROS C++/Py/命令行）

能力模块：

| 模块 | 能力 |
|---|---|
| `core` | 连接会话、设备信息、整机健康/故障 |
| `motion` | 运动组/策略切换、策略快照、速度控制权、**速度发送** |
| `telemetry` | IMU / GNSS |
| `power` | 电池/PMU 状态、电源控制、电池切换、休眠 |
| `camera` | 相机能力/控件/推流/流参数 |
| `low_level` | 关节状态、**直接关节命令(MIT)** |
| `nav` | 点云流端点广播 |

## 3. Python API（关键签名）

`latentos_sdk` 导出 `Session`、`SdkOptions`；各 client 从子模块导入。

### core
```python
Session(options: SdkOptions | None = None, *, client_config_path: str = "")
DeviceClient(session, options=None).query_device_info(timeout_ms=1000)
```

### motion（底盘速度 / 控制权）★
```python
MotionClient(session)
  acquire_control(options=None) -> ControlAuthorityResult   # 申请速度控制权，SDK 自动续租
  release_control(options=None)
  send_velocity(x: float, y: float, yaw: float) -> None      # x/y m/s, yaw rad/s；须先取控制权并持续发送
  set_velocity_smoothing(enabled: bool)
  is_velocity_smoothing(timeout_ms=1000) -> bool             # False=关闭
  switch_group(target_group: str, policy: str = "")
  switch_current_policy(policy: str)
  set_remote_velocity_control(enabled: bool, remote=2)       # 1=SBUS 2=F710 3=两者
ControlAuthorityClient(session)  # 提供相同 查询/申请/释放/send_velocity
```

### low_level（关节）★
```python
LowLevelClient(session)
  subscribe_joint_state(callback=None)      # None=仅更新缓存
  get_latest_joint_state() -> JointState | None
  send_joint_command(command: JointCommand) # MIT 直接关节命令，不经运动仲裁，无执行回执
```

### camera
```python
CameraClient(session)
  query_capabilities(camera_id='', force_refresh=False) -> CameraCapabilities  # camera_id 空=全部
  set_control(camera_id, feature, value) -> seq
  set_stream_params(camera_id, width, height, fps, bitrate_kbps) -> seq        # 0=保留当前值
  send_stream_command(camera_id, operation, rtsp_ip='') -> seq                 # 启动/停止/重启推流
  subscribe_stream_status / get_latest_stream_status
```
（`set_control/send_stream_command/set_stream_params` 异步，返回 seq 匹配回执）

### status / telemetry / power / nav（摘要）
```python
StatusClient: query_status / subscribe_status / get_latest_status / drain_events
TelemetryClient: subscribe_imu/get_latest_imu, subscribe_gnss/get_latest_gnss
PowerClient: subscribe_*/get_latest_*, set_power_rail/light/fan_speed, switch_battery_pack, request_power_sleep(~10s)
NavClient: advertise_point_cloud_stream(ip, port, enabled=True)  # 建议 ~1Hz
```

## 4. 安装

先 **SDK Runtime** 再 **C++ SDK**，默认前缀 `/data/latentos`（写 /data 需 sudo）。
```bash
cd sdk_runtime &&  sudo ./activate.sh latentos-sdk_runtime-<ver>-<arch>.tar.gz
cd ../latentos_sdk_cpp && sudo ./activate.sh latentos-sdk_cpp-<ver>-<arch>.tar.gz
# 可选：--prefix DIR / --force / --skip-system-check（默认应先解决系统依赖）
```
安装后：头文件 `sdk_cpp/include/latentos/sdk`、库 `lib(64)`、Python 包 `sdk_cpp/python/site-packages`、wheel `share/latentos_sdk/wheels`、ROS `setup-ros.bash`。

**Python**：
```bash
export PYTHONPATH="/data/latentos/sdk_cpp/python/site-packages${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'from latentos_sdk import Session'
# 或 pip install /data/latentos/sdk_cpp/share/latentos_sdk/wheels/*.whl
```
**C++**：`export CMAKE_PREFIX_PATH="/data/latentos/sdk_cpp:/data/latentos/sdk_runtime:/data/latentos/sdk_runtime/third_party..."`
**ROS2**：`source /data/latentos/sdk_cpp/setup-ros.bash`

## 5. 第一个示例（Python，只读设备信息）
```bash
cd /path/to/latentos_sdk_example
CLIENT="$PWD/config/echo/mode/client.yaml"
python3 examples/python/core/query_device_info.py --client-config "$CLIENT"
# 成功输出含：设备ID / 型号 / 机器人ID / 平台 / 软件版本 / 硬件版本
```

## 6. 与「P30 + RM65 抓取部署」的对应

| 我们的动作 | P30 SDK 接口 |
|---|---|
| 底盘速度命令（step 的 `a13[10:13]=vx,vy,wz`） | `MotionClient.acquire_control()` + `send_velocity(x,y,yaw)` + 持续发送 |
| 底盘位姿/里程计 | 见 `core`/`status`（本页未列 pose，查 API 参考） |
| P30 自身关节 | `low_level.get_latest_joint_state()` / `send_joint_command()`（P30 12 DOF） |
| 相机图像 | `camera` 推流（RTSP）或外接相机 |
| **RM65 机械臂** | **独立 REALMAN SDK**（不在 LatentOS SDK 内）——臂的关节/笛卡尔指令走睿尔曼接口 |

> 关键：**底盘走 LatentOS `motion.send_velocity`；机械臂走 REALMAN SDK**，二者分属不同 SDK，部署时 HAL 的 `backend.step` 内部各调各的。

## 7. 在线文档子页

- 产品参数：`/docs/next/product/p30/`
- 遥控器：`/docs/next/remote-control/`（手柄 `/gamepad/`）
- SDK：`/docs/next/intro/`
- 快速开始：`/getting-started/{prerequisites,install,connect,first-example}/`
- 开发指南：`/integrate/` · 功能指南：`/examples/`
- API 参考：`/api/cpp/` · `/api/python/` · `/api/ros2/`
- 故障排查：`/faq/` · 版本迁移：`/migration/`
