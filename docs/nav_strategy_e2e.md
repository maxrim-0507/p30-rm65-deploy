# 导航/抓取端到端策略（e2e demo v2 · 2026-11-25 验证）

> 验证结果：仿真 **16/16 = 100%**（6 集 + 10 集，lift +0.050~0.052）。
> 文件：`simulation/mujoco/e2e_demo_v2.py`（THOR2 上固化）。

## 流程
`SEARCH（臂转找）→ APPROACH（3-phase yaw-first 泊车）→ GRASP（VLA v36）`

## 1. SEARCH：臂转一周找目标（底盘不动）
- **底盘保持不动**，臂 **J1 绕竖轴扫全程**（-175°~+175°，≈356° 近一圈）。
- 每个 J1 角：`ctrl[arm]=SEARCH_Q`(改 J1) → physics → 相机可见判定(视锥/遮挡/像素)，可见即停、记录方向。
- 为什么不用"底盘原地转"：不动底盘 → 里程不漂、不扰动泊位；J1 ±178° 已近整圈，能覆盖四周；且车头朝向对抓取不重要（VLA 自己会开）。
- 代码：`e2e_demo_v2.py::search()`

## 2. APPROACH：3-phase yaw-first 泊车（先转头、后前进）
- **phase1**：车头先转到「车→泊位」方向 `atan2(ey,ex)`（不平移）；
- **phase2**：对准后沿车头前进到泊位（侧向修正压到 0.1 倍，少横移）；
- **phase3**：到泊位后转到面向目标 `gyaw`。
- 之前错误做法：让车直接朝 `gyaw`（那是"停在 dock 时的朝向"），车在远处会走偏 → 残差 1~2m 抓空。
- 代码：`e2e_demo_v2.py::approach()`

## 3. 泊车位 clamp 训练分布（抓取率关键）
```
STAGE_X = (-0.34, -0.12)   # v36 训练抓取泊车带
STAGE_Y_ABS = 0.12
STAGE_YAW_ABS = 0.15
STANDOFF = 0.70            # 目标→泊车位 水平距离
```
交班时 base 位姿被 clamp 到训练盒内 → VLA 抓取率回到其上限（~91.7%）。

## 4. 实时 viewer 演示（THOR2 HDMI :1）
`ColorGraspEnv(view=True)` + **`import mujoco.viewer`**（THOR2 的 mujoco 3.15 需显式导入）+ `DISPLAY=:1` + `MUJOCO_GL=glfw`。

## 注意 / 上真机
- demo 的 SEARCH/APPROACH 的"看见"用**仿真真值替身**；真机须换成**真机相机检测**（HSV/分割）+ `P30 motion.send_velocity`（底盘）+ `RM65 rm_driver`（臂）。
- `e2e_demo_v2.py` 依赖 `simulation/mujoco` 资产；放进 examples 仅作**参考**，不能脱离仿真独立跑。
- 抓取上限是 VLA 固有（v36 ~91.7%），跑更大规模会偶发失败（模型层面，非流程 bug）。
