# -*- coding: utf-8 -*-
"""IK 核心：位置 + 工具轴竖直（yaw 自由），带碰撞检查。

=== 为什么不是 6 自由度 ===
最初用完整 6D 姿态（位置 3 + 旋转矩阵前两列 6 = 9 维残差）约束，
结果 y=±6cm 全判不可达，但同一目标点在「只约束工具轴竖直」时误差
恰好为 0.00000。原因是 6D 约束把绕垂直轴的 yaw 也钉死了，而
yaw 需要 joint6 独立补偿，在腕部接近奇异时会互相冲突 → 超定。

抓方块本来就不需要指定 yaw：
  - Dex1-1 两指对称，joint6 可任意转（±360°）
  - 方块有 90° 旋转对称
所以正确约束是 5 自由度：位置(3) + 工具轴朝下(2)，yaw 自由。

=== 姿态误差也不能用 0.5*sum(cross) ===
那是 sin(θ) 向量，θ=0° 和 θ=180° 都返回 0，会让 IK 收敛到
「夹爪完全倒转」的假解（实测 yaw=135° 时 total=0 但实际偏差 180°）。
这里直接用 轴向量之差，无退化。
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from scipy.optimize import least_squares

TOOL_AXIS_LOCAL = 1          # grasp_site 局部 +y 是抓取轴（实测 cos=1.000）
DOWN = np.array([0.0, 0.0, -1.0])

# grasp_site 与指垫中心几乎同高（2026-11-06 实测 MuJoCo：site 0.6195，
# 指垫中心 0.6192，差 0.3mm；指垫半高 0.014）。原值 0.036 会让指垫整体
# 高到方块顶面之上——闭合只咬住 ~8mm，一抬就滑（实测 cube 被压下 1.1cm
# 后原位不动）。
# 取 0.012：抓取位姿 site z = 0.55+0.03+0.012 = 0.592，指垫覆盖
# [0.578, 0.606]，稳稳在 6cm 方块(0.55~0.61) 的侧面中部；同时 mesh 指尖
# 仍高于台面。抓取高度一律用：方块中心 + GRASP_SITE_UP。
GRASP_SITE_UP = 0.012
CUBE_HALF = 0.030


def grasp_z_for(table_top_z):
    """给定台面高度，返回 grasp_site 应有的目标 z（抓方块中心）。"""
    return table_top_z + CUBE_HALF + GRASP_SITE_UP


RHO_SAFE = 0.24              # 预抓点距基座**轴线**的安全水平距离


def approach_points(pos_xy, grasp_z, base, pre=0.06, hover=0.03):
    """生成「预抓 → 悬停 → 抓取」三个点，预抓点按内死区自动外退。

    === 为什么不能竖直抬高 ===
    RM65 基座周围是球形内死区（最小可达半径约 0.165 m）。物体贴近基座时，
    抓取点本身还在死区外（r≈0.22），但**竖直上升会先穿过死区**：
    基座 z 平面上水平距离只有 0.144 m，抬到 +0.20 时 r 才 0.148，
    仍 < 0.165，IK 返回 1e-2 假解，下游误判成「跨 IK 分支」。
    竖直抬到够为止需要 +0.30 m，已接近臂的外包络，不是好解。

    === 默认值为什么是 0.06 / 0.03（2026-11-11 实测）===
    0.65 台面的抓取点 z=0.692 只比臂座 0.710 低 18 mm，
    pre 抬 0.20 会到 0.892，把腕部推到臂展边缘：整链可用格点
    24/84；改成 0.06/0.03 后 76/84，物理闭环 4/4 抓起。
    lift 取 0.07 而不是 0.05：指令 50mm 实测只到 43~44mm
    （50 步位置伺服未收敛），达不到 success.lift_delta_min=0.05。
    0.07 只少 2 格可达（74/84），实测抬升约 60mm。
    低台面（0.55）下两组参数都够用，所以这组默认值向下兼容。

    正确做法：水平距离不够时，把预抓点沿**基座轴线向外**推开到 RHO_SAFE，
    即「从前方水平靠近再下降」。这既绕开了死区，也比贴着躯干上升更安全。
    """
    base = np.asarray(base, dtype=float)
    p = np.array([pos_xy[0], pos_xy[1], grasp_z], dtype=float)
    dx, dy = p[0] - base[0], p[1] - base[1]
    rho = float(np.hypot(dx, dy))

    if rho >= RHO_SAFE:
        push = np.array([0.0, 0.0])
        dz_pre = pre
    else:
        if rho < 1e-6:                        # 正上方退化：默认朝前方退
            dx, dy, rho = 1.0, 0.0, 1.0
        push = np.array([dx, dy]) / rho * (RHO_SAFE - rho)
        dz_pre = min(pre, 0.14)               # 水平已退够，不必再抬那么高

    pre_pt = [p[0] + push[0], p[1] + push[1], p[2] + dz_pre]
    hov_pt = [p[0] + push[0] * 0.5, p[1] + push[1] * 0.5,
              p[2] + dz_pre * 0.5]
    return pre_pt, hov_pt, [p[0], p[1], p[2]]


def lift_point(pos_xy, grasp_z, base, dz=0.07):
    """抓取后的抬升点：直上直下会扎进内死区，改为「向外退 + 抬起」。

    物体贴近基座时，抓取点 r≈0.23 还在死区外，但竖直抬 0.07 m 后
    z 逼近基座高度、r 掉到 0.17，正好卡在最小可达半径上，IK 出 2e-2 假解。
    沿基座轴线外退同时抬升，既绕开死区，也更像真实的「抓起来往回收」。
    """
    base = np.asarray(base, dtype=float)
    dx, dy = pos_xy[0] - base[0], pos_xy[1] - base[1]
    rho = float(np.hypot(dx, dy))
    lift_z = grasp_z + dz
    # 保证 sqrt(rho'^2 + (lift_z - base_z)^2) >= MIN_REACH
    dzv = lift_z - base[2]
    need = MIN_REACH ** 2 - dzv ** 2
    rho_need = float(np.sqrt(need)) if need > 0 else 0.0
    rho_need = max(rho_need, rho)
    if rho < 1e-6:
        dx, dy, rho = 1.0, 0.0, 1.0
    k = rho_need / rho
    return [base[0] + dx * k, base[1] + dy * k, lift_z]

# 6 维残差（位置 3 + 轴向 3），判定阈值：1e-3 对应约 1mm / 0.06°
TOL = 1e-3

# RM65 的最小可达半径（SDK 实测约 0.165 m）。基座正上方 / 正下方是内死区：
# 目标点离基座中心比这个半径还近时，IK 只能收敛到 1e-2 量级的假解。
# diag_chain 40 组里失败的 4 组全是这一条：
#   (0.442, 0.023, 0.836) 到基座 (0.30, 0, 0.78) 只有 0.154 m ——「预抓点
#   固定抬 0.22 m」在 x 小的物体上正好捅进死区。可达性差不是分支跳变。
MIN_REACH = 0.20             # 留 3.5cm 余量，比实测 0.165 保守


class IK:
    def __init__(self, xml="p30_rm65_scene.xml", arm_joints=None,
                 base_body="arm_base_link", site="grasp_site"):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)

        self.names = arm_joints or ["arm_joint%d" % i for i in range(1, 7)]
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in self.names])
        self.lo = np.array([self.m.jnt_range[self.m.joint(n).id, 0] for n in self.names])
        self.hi = np.array([self.m.jnt_range[self.m.joint(n).id, 1] for n in self.names])
        self.site = self.m.site(site).id
        self.base = self.d.xpos[self.m.body(base_body).id].copy()

        self.arm_bodies = {self.m.body(n).id for n in
                           ["arm_base_link", "arm_Link1", "arm_Link2", "arm_Link3",
                            "arm_Link4", "arm_Link5", "arm_Link6", "gripper_base",
                            "finger_left", "finger_right"]}
        self.bad_bodies = {self.m.body("base_link").id, self.m.body("table").id}

    # ---------- 残差 ----------
    def resid(self, q, pos_tgt):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        R = self.d.site_xmat[self.site].reshape(3, 3)
        axis = R[:, TOOL_AXIS_LOCAL]
        return np.concatenate([self.d.site_xpos[self.site] - pos_tgt,
                               0.5 * (axis - DOWN)])

    def resid_reg(self, q, pos_tgt, q_seed, reg):
        """带「最小运动」正则的残差。

        === 为什么必须加 ===
        本 IK 只约束 5 维（位置 3 + 工具轴 2），绕工具轴的 yaw 是零空间：
        无穷多组关节角能精确到 0 误差，但它们的 joint6 可以从 -180° 到 +180°
        乱跳、joint3/4 也可能翻支。实测沿笛卡尔直线连续求解后，相邻点
        单关节仍跳 675°——因为零空间方向上 LM 没有任何偏好。
        追加 reg*||q - q_seed|| 相当于「在解集里挑离上一帧最近的那个」，
        reg 取 0.02：偏差 1 rad 才贡献 0.02 残差，对 1mm 位置精度影响 ~1e-4，
        但足以把解钉在当前分支上。
        """
        return np.concatenate([self.resid(q, pos_tgt),
                               reg * (q - q_seed)])

    # ---------- 求解 ----------
    def solve(self, pos_tgt, rng, n_restart=24, max_nfev=600):
        """多起点 LM。返回 (误差, q)。误差 < TOL 视为收敛。"""
        best = (9.9, None)
        for _ in range(n_restart):
            q0 = rng.uniform(self.lo, self.hi)
            r = least_squares(self.resid, q0, bounds=(self.lo, self.hi),
                              args=(pos_tgt,), xtol=1e-12, ftol=1e-12,
                              gtol=1e-12, max_nfev=max_nfev)
            e = float(np.linalg.norm(r.fun))
            if e < best[0]:
                best = (e, r.x.copy())
            if e < TOL:
                break
        return best

    def yaw_of(self, q):
        """末端绕世界 z 轴的朝向角（rad）。工具轴竖直时它等价于绕工具轴的转角。"""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        R = self.d.site_xmat[self.site].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def align_yaw(self, chain, target_yaw, symmetry=np.pi / 2):
        """用 joint6 把末端朝向对齐到 target_yaw 的 symmetry 整数倍。

        === 为什么这是零代价的 ===
        实测（40 组，对每个关节单独 +30°）：
            joint1  抓取点移动 113.9 mm   yaw +30°
            joint2  抓取点移动 141.0 mm   工具轴偏 30°
            joint3  抓取点移动 223.3 mm   工具轴偏 30°
            joint4  抓取点移动  92.1 mm   yaw -21.1°
            joint5  抓取点移动 132.2 mm   工具轴偏 30°
            joint6  抓取点移动   0.1 mm   工具轴 0.0°   yaw -30°
        只有 joint6 是精确的零空间：它绕竖直的工具轴转，抓取点和工具轴分毫不动。
        这是 5 自由度 IK 里唯一真正自由的量，不该让它被 LM 随机游走。

        === 不对齐的后果（实测 150 条采集） ===
        各 demo 的腕部 yaw 在 [-175°, +179°] 均匀分布，std 100°；
        夹爪与方块的对齐偏差平均 22.7°、最大 45°。
        同一张图像对应姿态完全不同的动作 → 训练信号多模态冲突，
        模型只能学到「位置」，学不到「该用哪个腕部姿态」。

        方块绕 z 有 90° 旋转对称，所以对齐到最近的 90° 整数倍即可。
        """
        if not chain:
            return chain
        q_ref = chain[-1][1]
        cur = self.yaw_of(q_ref)
        # joint6 增 Δ → yaw 减 Δ（实测 1:1 反向），故偏移量取 cur 到目标的差
        delta = cur - target_yaw
        delta = delta % symmetry
        if delta > symmetry / 2:                 # 取最近的那个对称等价角
            delta -= symmetry
        out = []
        for e, q in chain:
            q = np.array(q, dtype=float)
            if q.size < 6:
                out.append((e, q))
                continue
            q[5] += delta
            # joint6 行程 ±2π；折到最近的等价角，避免越界
            q[5] -= 2 * np.pi * np.round(q[5] / (2 * np.pi))
            q[5] = float(np.clip(q[5], self.lo[5], self.hi[5]))
            out.append((e, q))
        return out

    def wrap_to_seed(self, q, q_seed):
        """把解里等价于 2π 的关节角折到离 seed 最近的那个，且不越界。

        === 为什么需要 ===
        关节 6 的行程是 ±360°，同一个物理姿态可以写成 -72° 或 +288°。
        least_squares 偶尔给出后者，关节空间插值就从 -72° 一路转到 +288°，
        实测「悬停→抓取 关节最大变化 252°」——夹爪在下降途中原地自转大半圈。
        抓取结果虽然对（夹爪对称、方块 90° 对称），但动作完全不可用于训练数据。
        """
        out = np.array(q, dtype=float)
        for i in range(out.size):
            cand = out[i] + 2 * np.pi * np.round((q_seed[i] - out[i]) / (2 * np.pi))
            if self.lo[i] - 1e-9 <= cand <= self.hi[i] + 1e-9:
                out[i] = np.clip(cand, self.lo[i], self.hi[i])
        return out

    def solve_near(self, pos_tgt, q_seed, max_nfev=400, reg=0.02):
        """从给定种子出发做一次带正则的 LM（不重启）。

        === 为什么必须有这个 ===
        solve() 用随机多起点，同一个目标点每次可能收敛到不同 IK 分支
        （肘上/肘下）。如果对「悬停位」和「抓取位」分别独立调用 solve()，
        得到的两个解可能 joint4 相差 180°、joint3 相差 90°。对这样两个解
        做关节空间线性插值，机械臂会绕一大圈，永远到不了目标
        （实测：插值到 t=1.0 时 grasp_site 仍差 15cm）。

        === 为什么还要正则 ===
        即便从同一种子出发，零空间（yaw）仍被 LM 随机游走，实测相邻点
        单关节跳 675°。正则项把解钉在种子附近，返回误差只统计前 6 维
        （真实任务误差），正则项不污染可达性判断。
        """
        tgt = np.asarray(pos_tgt, dtype=float)
        seed = np.asarray(q_seed, dtype=float)
        r = least_squares(self.resid_reg, seed,
                          bounds=(self.lo, self.hi),
                          args=(tgt, seed, reg),
                          xtol=1e-12, ftol=1e-12, gtol=1e-12,
                          max_nfev=max_nfev)
        e = float(np.linalg.norm(self.resid(r.x, tgt)))
        return e, self.wrap_to_seed(r.x, seed)

    def continue_from(self, pos_tgt, pos_prev, q_seed, max_nfev=400, reg=0.02):
        """沿 prev → tgt 的直线分小步求解，每步用上一步的解作种子。

        返回 (误差, q)。pos_prev 为 None 时退化为单次 solve_near。
        """
        b = np.asarray(pos_tgt, dtype=float)
        s0 = np.asarray(q_seed, dtype=float).copy()
        if pos_prev is None:
            return self.solve_near(b, s0, max_nfev=max_nfev, reg=reg)
        a = np.asarray(pos_prev, dtype=float)
        best = (9.9, s0)
        for n in (4, 8, 16, 32):
            q_cur = s0.copy()
            e = 9.9
            for t in np.linspace(0.0, 1.0, n + 1)[1:]:
                e, q_cur = self.solve_near(a + t * (b - a), q_cur,
                                           max_nfev=max_nfev, reg=reg)
                if e >= TOL:
                    break
            if e < best[0]:
                best = (e, q_cur.copy())
            if e < TOL:
                break
        return best

    def solve_chain(self, pos_list, rng, n_restart=24, max_jump_deg=60.0,
                    seed=None, reg=0.02):
        """按顺序求解一串目标点，保证相邻两点落在同一 IK 分支。

        === 为什么不能「种子失败就退回多起点」 ===
        旧实现里 seed 解不到 TOL 就退回 solve()（随机多起点），那里可能
        收敛到肘上/肘下另一支，实测「悬停→抓取」单关节跳 139.5°。

        === 三层保障 ===
        1. 每步都用 solve_near（带最小运动正则），零空间不会乱漂；
        2. 关节跳变超阈值就用更强的正则再解一次（先把解钉住再谈精度）；
        3. 仍不行才沿笛卡尔直线做 continuation，每小步从上一解接力。

        seed：给链的起点一个明确姿态（如停放姿态 ARM_STOW_DEG）。
        有了它就不必用随机多起点猜第一点——随机起点往往一步就落到
        奇怪分支（实测 40 组里 35 组第一段就跳 400°+）。

        返回 [(err, q), ...]。用于生成连续轨迹：悬停 → 下降 → 闭合 → 提升。
        """
        out = []
        cur = None if seed is None else np.asarray(seed, dtype=float).copy()
        prev_p = None
        for p in pos_list:
            tgt = np.asarray(p, dtype=float)
            if cur is None:
                e, q = self.solve(tgt, rng, n_restart=n_restart)
            else:
                e, q = self.solve_near(tgt, cur, reg=reg)
                if float(np.degrees(np.abs(q - cur)).max()) > max_jump_deg:
                    # 先加大正则把解钉在当前分支上
                    e2, q2 = self.solve_near(tgt, cur, reg=reg * 10)
                    if e2 < TOL and \
                            float(np.degrees(np.abs(q2 - cur)).max()) <= max_jump_deg:
                        e, q = e2, q2
                if e >= TOL:
                    e2, q2 = self.continue_from(tgt, prev_p, cur, reg=reg)
                    if e2 < TOL:
                        e, q = e2, q2
            out.append((e, q))
            cur = q
            prev_p = tgt
        return out

    # ---------- 碰撞 ----------
    def bad_contacts(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, self.d)
        out = []
        for c in range(self.d.ncon):
            con = self.d.contact[c]
            b1, b2 = self.m.geom_bodyid[con.geom1], self.m.geom_bodyid[con.geom2]
            if (b1 in self.arm_bodies and b2 in self.bad_bodies) or \
               (b2 in self.arm_bodies and b1 in self.bad_bodies):
                n1 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or con.geom1
                n2 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or con.geom2
                out.append((str(n1), str(n2), float(con.dist)))
        return out

    def graspable(self, x, y, z, rng, n_restart=24):
        """返回 (是否可抓, 误差, q, 诊断)。"""
        e, q = self.solve(np.array([x, y, z]), rng, n_restart)
        if e >= TOL:
            return False, e, q, "运动学不可达"
        cs = self.bad_contacts(q)
        if cs:
            return False, e, q, "碰撞:%s" % cs[0][0]
        return True, e, q, "OK"

    def tool_tilt_deg(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        axis = self.d.site_xmat[self.site].reshape(3, 3)[:, TOOL_AXIS_LOCAL]
        return float(np.degrees(np.arccos(np.clip(axis @ DOWN, -1, 1))))
