# p30-rm65-deploy

潜行未来 **P30 轮式机器人 + 睿尔曼 RM65 机械臂** 的「看到色块 → 开过去 → 抓起来」端到端部署工程。

策略：**GR00T N1.x (3B) VLA 微调模型**（v36, `use_relative_actions`）。
形态：**分层** —— 导航（找+泊车，规则视觉）+ 抓取（VLA，同步逐帧闭环）。

## 架构（三层）

```
 任务/指令  task/            (planner · instruction · grounding)
 技能层     skills/          (manager · pregrasp · manipulate · verify)
 策略层     policy/ + vla/    (LocalGr00tPolicy = VLA 推理, PyTorch)
 ───────────────────────────────────────────────────────────────
 感知/导航  nav/ + perception/ (search · visual_dock · handoff · color/real_cameras)
 控制层     control/          (server=L3 100Hz, safety)
 HAL        hal/              (base 接口 · real_backend 真机 · mujoco_backend 仿真)
 抓取闭环   grasp/grasp_hal.py (同步逐帧 read→VLA→step16，可插拔后端)
 ───────────────────────────────────────────────────────────────
 配置       configs/          (deploy · vla_spec · real_cameras · robot_spec · mount)
 场景/相机  assets/p30_rm65_nav.xml
```

数据流（一个抓取循环）：`backend.observe()`（相机+FK）→ `policy.infer`（VLA, Thor GPU）→ `backend.step(a13)`（发指令）→ 循环。

## 目录

- `deployment/` — 部署框架（HAL / 控制 / 导航 / 感知 / 技能 / 策略 / 任务）
- `vla/local_gr00t_policy.py` — VLA 推理（PyTorch safetensors，真机同一份）
- `grasp/grasp_hal.py` — 抓取闭环 HAL（同步；真机实现 RealGraspBackend）
- `configs/` — 运行配置（含真机相机标定模板 real_cameras.yaml）
- `assets/` — 场景/相机 mount/几何定义（真机相机标定参考）
- `docs/` — 架构、真机部署、VLA 部署说明

## 快速开始

```bash
pip install -r requirements.txt          # 见 requirements.txt（lerobot 需 GR00T 分支）
# 1) 模型：把 v36 checkpoint 放到 checkpoints/groot_rel_v36/020000（不入库，见下）
# 2) 仿真回归（可选，需 simulation/mujoco + mesh 资产）：
python examples/hybrid_eval.py --trials 3
# 3) 真机：见 docs/real_robot_deploy.md
```

## 重要说明（踩过的坑，务必遵守）

1. **抓取闭环必须同步逐帧**（读当前相机→推理→精确16步→再读）。**不要**把物理推进交给后台异步线程 + 读快照——实测会系统偏、抓空（0%）。
2. **观测必须与训练同口径**：双相机 320×240 + front_cam/arm_cam 位置、state16 = `[eef9(臂座系), grip归一化, joint6]`、指令 "抓取X色的方块"。口径不对 → 模型输出错。
3. **VLA 推理要固定 seed**（流匹配随机性），否则不可复现。
4. 已验证 80% 的闭环实现是 `ColorGraspEnv.run_episode`（仿真）。真机闭环应**移植它的逐帧逻辑**，不要自创手写循环（手写实测仅 ~25%）。

## 模型权重（不入库）

`checkpoints/groot_rel_v36/020000/pretrained_model/`（config.json + model.safetensors，~几 GB）。
用 Git LFS 或外部存储提供；`.gitignore` 已排除 `checkpoints/` 与 `*.safetensors`。

## 平台

Thor (Jetson, CUDA) + Ubuntu。VLA 推理 PyTorch/CUDA，~0.27s/chunk（可后续 TensorRT 加速）。
