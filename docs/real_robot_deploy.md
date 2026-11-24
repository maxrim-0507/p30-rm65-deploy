# 真机部署指引

目标：把仿真里验证的「找→泊车→VLA 抓取」搬到真机（P30 + RM65）。**策略/闭环逻辑不动，只实现 HAL 后端。**

## 一、要实现的后端（`grasp/grasp_hal.py::RealGraspBackend`）

| 方法 | 真机做什么 |
|---|---|
| `reset(layout, q_init)` | 机体摆到抓取起点：臂到 q_init（FWD2），底盘到泊车位 |
| `set_base_pose(x,y,yaw)` | 底盘定位（或直接读当前位姿返回 ok）|
| `observe()` | 真机相机读 front/arm 两路 + 真机 FK 算 eef9 + 关节 → 组装 obs |
| `step(a13)` | eef9(笛卡尔)+grip+base_vel 下发真机控制器，等 ~16×32ms |
| `judge(color)` | 视觉/力感知判抬升 |
| `solve_pregrasp(layout,color)` | 预备抓取位（查标定表或真机控制器 IK）|

## 二、obs 必须与训练同口径（关键）

```python
front = 真机front相机  # 320x240 RGB, 与训练 front_cam 位置/内参一致
arm   = 真机arm相机    # 320x240 RGB, 与训练 arm_cam 位置/内参一致
eef9  = 真机FK: grasp_site 在臂座系位姿 [pos3, rot6d6]
state = np.concatenate([eef9, [grip_norm], joint6])   # 16 维
# grip_norm = (0.0245 - qpos_grip) / (0.0245 + 0.020)
```
- 相机标定填 `configs/real_cameras.yaml`（模板 `calibrated:false`，未标定会报错，绝不编外参）。
- 相机尺寸/颜色序/内参要和训练一致，否则模型输出偏。

## 三、step（执行动作）

`a13 = [eef9(9 臂座系绝对位姿), grip(1∈[0,1]), base_vel(3 机体系 vx,vy,wz)]`
- 末端：eef9 → 真机控制器（`rm_movej_p` / `movel` 到笛卡尔目标）；
- 夹爪：grip → 真机夹爪开度（与 `(0.0245−q)/(0.0245+0.020)` 互逆）；
- 底盘：base_vel → P30 速度接口；
- 等待一个控制周期（约 16 × 32ms = 512ms，或按真机控制频率）。

## 四、闭环（务必同步）

`grasp_hal.run_grasp_loop`：`observe → VLA.infer → step(16步) → observe → ...`，**单线程同步**。
- **不要**把物理推进交给后台异步线程 + 读快照（实测 0%）。
- 已验证 80% 的是仿真后端的 `ColorGraspEnv.run_episode`；真机后端实现后，先在仿真后端对拍确认闭环逻辑一致，再上真机。

## 五、VLA 推理

见 `docs/vla_deploy.md`。同一份 `vla.LocalGr00tPolicy`，真机 GPU 加载 v36 checkpoint。

## 六、分阶段上线（建议）

1. 先单独验证 `observe()` 的图/state 与仿真同场景一致（离线比）。
2. 再验证 `step()` 单步动作（定一个目标位姿，看末端到位）。
3. 最后跑完整 `run_grasp_loop`，慢速、E-stop 就位、首次别让 VLA 全自主。
