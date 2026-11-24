# 架构说明

## 分层

```
┌─────────────────────────────────────────────────────────────┐
│ 入口: run_l3.py(控制服务) / run_l2.py(技能编排) / grasp_hal   │
├─────────────────────────────────────────────────────────────┤
│ task/        任务/指令 (planner, instruction, grounding)       │
│ skills/      技能 (manager 编排, pregrasp, manipulate, verify) │
│ policy/      策略 (LocalGr00tPolicy 适配, base)                │
│ vla/         VLA 推理 (local_gr00t_policy: GR00T, PyTorch)     │
├─────────────────────────────────────────────────────────────┤
│ nav/         导航 (search, visual_dock, handoff, mux, presets) │
│ perception/  感知 (color_grounding, real_cameras, tracking)    │
│ control/     控制 (server=L3 100Hz 循环, safety 限幅)           │
│ hal/         硬件抽象 (base 接口, real_backend, mujoco_backend, rm_algo_ik) │
│ grasp/       抓取闭环 HAL (同步 read→VLA→step16, 可插拔后端)    │
├─────────────────────────────────────────────────────────────┤
│ configs/     deploy.yaml, vla_spec.yaml, real_cameras.yaml ... │
│ assets/      p30_rm65_nav.xml (相机 mount / 几何)              │
│ spec.py      相机名/常量/规格                                   │
└─────────────────────────────────────────────────────────────┘
```

## 一次端到端（分层）

1. **SEARCH**：抬臂 + 原地转，相机找目标色（`nav/search.py` + `perception/color_grounding`）。
2. **DOCK**：视觉伺服把车开到目标正前方泊车（`nav/visual_dock.py`）。
3. **GRASP**：`grasp_hal.run_grasp_loop`（同步逐帧）：
   - `backend.observe()` → 双相机 + state16；
   - `vla.LocalGr00tPolicy.infer(obs)` → 40×13 动作 chunk（Thor GPU，~0.27s）；
   - `backend.step(a13)` → 精确推进 16 步（仿真 mj_step×16 / 真机发指令等 16 周期）；
   - 循环至抬升。
4. **VERIFY**：抬升 ≥ 4cm 且颜色对 = 成功。

## 关键设计决定

- **抓取用同步闭环，不用 L3 异步**：L3 后台 100Hz 推进物理 + 读快照会与观测时间轴错位，实测抓取系统偏到邻块（0%）。同步闭环（读当前→推理→精确16步）才 work。
- **VLA 与 HAL 解耦**：VLA 只 `obs→chunk`，不知道仿真/真机；obs 由后端提供、动作由后端执行。
- **策略/HAL/闭环三层，后端可插拔**：仿真后端 = ColorGraspEnv；真机后端 = RealGraspBackend（相机+指令+FK）。

## 观测口径（务必与训练一致）

- 相机：front_cam / arm_cam，320×240，RGB。
- state16 = `[eef9(9), grip(1), joint(6)]`：
  - eef9 = grasp_site 在**臂座系**的位姿：`pos(3) = R_base^T·(site−base)`，`rot6d(6) = (R_base^T·R_site)[:2,:].flatten()`；
  - grip = `(0.0245 − qpos_grip)/(0.0245+0.020)`；
  - joint = 6 个臂关节角（rad）。
- 指令："抓取{颜色}的方块"。

## 动作（13 维）

`a13 = [eef9(9, 臂座系绝对), grip(1, [0,1]), base_vel(3, 机体系 vx,vy,wz)]`。
