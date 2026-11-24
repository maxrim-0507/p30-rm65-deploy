# -*- coding: utf-8 -*-
"""MuJoCo 后端：把已固化的 p30_rm65_scene.xml 包成 HAL。

=== 复用什么 ===
场景、IK、物理行为全部沿用 simulation/mujoco 下已提交的版本，不重写。
本文件只做四件事：
  1. 把 `ctrl` 与 `qpos` 绑定成「命令值」语义（与采集脚本一致）；
  2. 提供臂基座系的 eef 位姿（vla_spec.yaml 的口径）；
  3. 渲染 head/wrist 两路 RGB；
  4. 提供物体真值（供 oracle 策略与成功判定，真机没有这一步）。

=== 为什么不用 mj_step 的实时模式 ===
MuJoCo 默认尽快跑（200 步 10 ms）。闭环要按墙钟 30 Hz 观测，
所以 `realtime=True` 时在 step 里 sleep 到理论时刻。但 L3 内部
（100 Hz 控制插值）不需要 sleep —— 那是真机的语义。这里给
`realtime` 开关，测试总线时关掉能快 5~10 倍。

=== 相机朝向 ===
front_cam 的 xyaxes 与 arm_cam 的 zaxis 都在 build_scene.py 里按实测
改过（见 memory: mujoco_camera_convention）。这里直接按名字取，不碰
相机定义。
"""
import os
import time

import numpy as np

from ..policy.unified40 import our_rot6_to_matrix
from ..spec import ARM_BASE, CAM_H, CAM_NAMES, CAM_W, MOUNT_YAW_DEG, SIM_DIR
from .base import RobotBackend


class MujocoBackend(RobotBackend):

    def __init__(self, scene=None, render=True, realtime=True, seed=0,
                 ik="algo"):
        """ik: "algo" 用 REALMAN 官方 Algo（默认，与真机同一份 IK）；
        "core" 退回 ik_core（scipy 5 维），仅用于 A/B 诊断对照。
        """
        scene = scene or "p30_rm65_scene.xml"
        # 允许只给文件名：SIM_DIR 在本地与 Thor 两种布局下都能解析
        self.scene_path = (scene if os.path.isabs(scene)
                           else os.path.join(SIM_DIR, scene))
        self.render_enabled = bool(render)
        self.realtime = bool(realtime)
        self.m = None
        self.d = None
        self.ik = None          # 规划系：ik_core（可达性/碰撞/姿态约束）
        self.ik_exec = None     # 执行系：官方 Algo（与真机同源）
        self.ik_mode = str(ik)
        self.renderer = None
        self._q_cmd = None
        self._grip_cmd = None
        self._q_ik_seed = None       # 位姿链路 IK 的上一帧解（保持同一分支）
        self._pose_tgt_prev = None   # 位姿链路上一条目标（continue_from 续接用）
        self._t_next = 0.0
        # 底盘能力在 connect() 里探测；先给默认值，否则 connect 前访问
        # supports_base / get_base_state 会 AttributeError。
        self._base_acts = None
        self._base_cmd = np.zeros(3)
        self._base_jnts = None
        self.rng = np.random.default_rng(seed)

    # ---------- 生命周期 ----------
    def connect(self):
        import mujoco
        if not os.path.isfile(self.scene_path):
            raise FileNotFoundError(self.scene_path)
        self.m = mujoco.MjModel.from_xml_path(self.scene_path)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)

        self._arm_qadr = np.array([self.m.joint("arm_joint%d" % i).qposadr[0]
                                   for i in range(1, 7)])
        self._grip_qadr = [self.m.joint("gripper_left_joint").qposadr[0],
                           self.m.joint("gripper_right_joint").qposadr[0]]
        self._arm_act = [self.m.actuator("m%d" % i).id for i in range(1, 7)]
        self._gl_act = self.m.actuator("grip_l").id
        self._gr_act = self.m.actuator("grip_r").id
        self._eef_site = self.m.site("eef_site").id
        self._grasp_site = self.m.site("grasp_site").id

        # 底盘：只有 nav 场景（make_nav_scene.py 生成）才有这三个执行器。
        # 静态抓取场景取不到 → supports_base=False，上层据此跳过导航。
        self._base_acts = None
        self._base_cmd = np.zeros(3)
        self._base_jnts = None
        ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
               for nm in ("base_vx", "base_vy", "base_wz")]
        if all(i >= 0 for i in ids):
            self._base_acts = np.array(ids, dtype=int)
            self._chassis_body = self.m.body("base_planar").id
            self._base_jnts = [
                (self.m.joint(nm).id, self.m.joint(nm).dofadr[0])
                for nm in ("base_slide_x", "base_slide_y", "base_hinge_z")]

        # 臂基座原点：从场景里取真实 xpos，避免与 mount.yaml 推导值有毫米级分叉
        # === 必须每帧刷新，不能只取一次 ===
        # 静态抓取场景里 base_link 固定，缓存一次无所谓；但 nav 场景的底盘是
        # 三级运动学挂载（slide-x / slide-y / hinge-z），底盘一动臂基座就跟着动。
        # 若沿用 connect() 时缓存的值，get_state() 里的 eef xy 会随底盘平移量
        # 线性漂移 —— 表现为「底盘走了 1 m，末端相对臂座偏了 1 m」。
        self._base_body = self.m.body("arm_base_link").id
        self._refresh_base()

        self._obj_body_ids = []
        self._obj_geom_ids = []
        self._obj_names = []
        i = 0
        while mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                "obj_%d" % i) >= 0:
            self._obj_body_ids.append(
                mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                  "obj_%d" % i))
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM,
                                    "obj_%d_geom" % i)
            self._obj_geom_ids.append(gid)
            self._obj_names.append("obj_%d" % i)
            i += 1

        # 物体颜色在 MJCF 里只是占位；真值颜色由 build_scene 的 palette 写死，
        # 先按 geom_rgba 反查（reset 时可以覆盖）。
        self._obj_color = {}
        self._mujoco = mujoco

        # === 复刻 rollout ColorGraspEnv 的 foot joint 突变 ===
        # 训练/rollout 把脚轮 joint stiffness 设 0、damping 0.05（轮子自由）；
        # backend 默认保留 xml 的 200 会改变整机站姿 -> front_cam 视角偏 -> 图像/动作分叉。
        for _fnm in ("FL_foot_joint", "RL_foot_joint",
                     "FR_foot_joint", "RR_foot_joint"):
            try:
                _fj = self.m.joint(_fnm).id
            except Exception:
                continue
            self.m.jnt_stiffness[_fj] = 0.0
            self.m.dof_damping[self.m.jnt_dofadr[_fj]] = 0.05
        _foot_mutated = True

        self._q_cmd = self.d.qpos[self._arm_qadr].copy()
        self._grip_cmd = float(self.d.qpos[self._grip_qadr[0]])
        self._q_ik_seed = self._q_cmd.copy()
        self._t_next = time.monotonic()

        # === 执行系 IK：官方 Algo（默认）===
        # 规划系仍用 ik_core（可达性掩膜 / 碰撞检测 / 5 维姿态约束都是它算的，
        # 且 make_reach_mask.py 已按它固化了 reach_mask.json）。执行系换成
        # 官方 Algo 后，仿真下发的关节指令与真机控制器同源，构型分支一致。
        self.ik_exec = None
        if self.ik_mode == "algo":
            from .rm_algo_ik import RmAlgoIK
            self.ik_exec = RmAlgoIK(self._base)

        return self

    def close(self):
        # Renderer 的析构在 EGL 下会打 OpenGL 错误，显式关掉更干净
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None
        self.m = self.d = None

    # ---------- 场景控制（仿真专用，真机没有）----------
    def reset(self, obj_poses=None, arm_q=None):
        """重置场景。obj_poses=[(x,y,color,yaw)] 时一并布置物体。

        === arm_q 必须给，否则机械臂会从「收拢姿态」起转 ===
        keyframe 的 stand 姿态是狗站立用的收拢位，joint6 = -5.7875 rad
        （-331.6°）。而采集数据里每条 demo 的起始位是 solve_approach 的
        pre 点，joint6 落在 [-2.66, +2.77] rad。两者不是同一构型：
        实测从 keyframe 走到抓取点，joint6 要转 248°~505°，真机上就是
        关节狂甩一圈半，还会撞到限位。

        所以调用方必须传 arm_q（通常就是 Pregrasp 的解）。不传时退化为
        keyframe 并打印警告 —— 保持旧行为但不再静默。
        """
        mujoco = self._mujoco
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)
        if obj_poses:
            for nm, gid, pose in zip(self._obj_names, self._obj_geom_ids,
                                     obj_poses):
                x, y, color, yaw = pose[:4]
                adr = self.m.joint(nm + "_joint").qposadr[0]
                self.d.qpos[adr:adr + 3] = [x, y, self.table_top + self.cube_half]
                self.d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0,
                                                np.sin(yaw / 2)]
                self.m.geom_rgba[gid] = self._rgba(color)
                self._obj_color[nm] = color
        if arm_q is not None:
            self.d.qpos[self._arm_qadr] = np.asarray(arm_q, float).reshape(6)
        else:
            q6 = float(self.d.qpos[self._arm_qadr][5])
            if abs(q6) > np.pi:
                print("[warn] reset 未给 arm_q，沿用 keyframe 姿态，"
                      "joint6=%.1f° 超出 ±180°，Pregrasp 会大角度回转"
                      % np.degrees(q6), flush=True)
        mujoco.mj_forward(self.m, self.d)
        self._q_cmd = self.d.qpos[self._arm_qadr].copy()
        self._grip_cmd = float(self.d.qpos[self._grip_qadr[0]])
        self._q_ik_seed = self._q_cmd.copy()
        self._pose_tgt_prev = None
        self.d.ctrl[self._arm_act] = self._q_cmd
        self.d.ctrl[self._gl_act] = self._grip_cmd
        self.d.ctrl[self._gr_act] = self._grip_cmd
        if self._base_acts is not None:
            self._base_cmd = np.zeros(3)
            self.d.ctrl[self._base_acts] = self._base_cmd
        self._reset_qpos = self.d.qpos.copy()
        self._reset_ctrl = self.d.ctrl.copy()
        self._t_next = time.monotonic()

    @staticmethod
    def _rgba(color):
        from ..spec import COLOR_RGBA
        return COLOR_RGBA[color]

    def _refresh_base(self):
        """臂基座系必须每帧取，不能缓存。

        nav 场景的底盘是三级运动学挂载（slide-x / slide-y / hinge-z），
        底盘一动臂基座就跟着动。调用前必须保证 d.xpos/d.xmat 是当前值
        （要么刚 mj_kinematics 过，要么刚 mj_step 过）。

        位置与旋转都要取：eef_9d 是**臂座系**量（pos 与 rot6d 都相对
        R_base），只减平移会让 yaw!=0 时 xy 漂 22~66 mm、rot6d 偏 20°。
        （早前「不要旋转」的注释是错的：那时数据存的是世界系，
          现在 collect/export/rollout 全部改用臂座系，这里必须同源。）
        """
        self._base = self.d.xpos[self._base_body].copy()
        self._base_R = np.array(self.d.xmat[self._base_body],
                                dtype=float).reshape(3, 3).copy()

    def _nominal_base(self):
        """静态模型零位下的臂座位姿 (pos, R)，全程只算一次。

        IK 用的是静态模型（臂座恒在名义位姿），所以动作下发时必须用
        这个常量还原，不能用当帧实时臂座 —— 否则底盘位移会被算两遍。
        """
        if getattr(self, "_base_nom", None) is None:
            mj = self._mujoco
            m2 = mj.MjModel.from_xml_path(self.scene_path)
            d2 = mj.MjData(m2)
            d2.qpos[:] = 0.0
            mj.mj_forward(m2, d2)
            bid = m2.body("arm_base_link").id
            self._base_nom = (
                np.array(d2.xpos[bid], dtype=float),
                np.array(d2.xmat[bid], dtype=float).reshape(3, 3).copy())
        return self._base_nom

    # ---------- 执行 ----------
    def actuate(self, joint, gripper):
        from ..spec import GRIP_CLOSE, GRIP_RANGE
        q = np.asarray(joint, dtype=float).reshape(6)
        if not np.all(np.isfinite(q)):
            return {"ok": False, "err": "joint 含 NaN/inf"}
        g = float(gripper)
        # 归一化 → qpos。夹爪行程外的值会被位置驱动器夹住，但先夹好更直观。
        gq = GRIP_CLOSE - min(1.0, max(0.0, g)) * GRIP_RANGE
        self._q_cmd = q.copy()
        self._grip_cmd = gq
        self.d.ctrl[self._arm_act] = q
        self.d.ctrl[self._gl_act] = gq
        self.d.ctrl[self._gr_act] = gq
        return {"ok": True, "err": None}

    @property
    def supports_pose_command(self):
        """仿真里 ik_core 就是「厂商 IK 替身」，所以支持位姿指令。"""
        return True

    def actuate_pose(self, pos, rot6d, gripper):
        """末端位姿 → IK → 关节目标。

        === 求解器 ===
        默认走 `rm_algo_ik.RmAlgoIK`（REALMAN 官方 Algo），与真机控制器
        `rm_movej_p/movel` 是同一份 IK：输入完整 6D，构型分支一致。
        `ik="core"` 时退回 ik_core（5 维 scipy），仅用于 A/B 对照。

        === 口径（与 get_state() 严格互逆）===
        pos 与 rot6d 都相对**臂座系**（见 get_state）。IK 目标用名义臂座
        还原到静态世界系，**不再加任何偏移**：
        eef_9d 与 IK 目标都统一在 grasp_site（夹持面中心）上。
        旧口径里 eef_9d 是法兰、IK 目标是夹持面，靠减 GRASP_SITE_UP=0.012
        补偿，但两个 site 实际差 0.1113 m，只补 0.012 会让末端系统性高
        9.93 cm（实测）。

        === 三层回退（官方 IK 失败时没有残差数值，只能按返回码判断）===
        1. 种子单步解（保持分支连续）；
        2. 失败 → `solve_all` 拿 8 组全解，挑离种子最近的；
        3. 仍失败 → 用实测关节角再解一次。
        全失败才返回 not-ok，交由上层 FAULT。
        """
        p = np.asarray(pos, dtype=float).reshape(3)
        r6 = np.asarray(rot6d, dtype=float).reshape(6)
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(r6)):
            return {"ok": False, "err": "pose 含 NaN/inf"}
        # rot6d（前两**行**）→ 3×3。行优先是 vla_spec.yaml 定死的口径，
        # 顺手正交化，避免模型输出轻微非正交时姿态被放大。
        # === 臂座系 → 静态模型世界系（2026-11-10）===
        # 动作是臂座系量（见 get_state），IK 是静态模型（臂座恒在名义位姿），
        # 所以必须旋转+平移还原：
        #   p_world = R_base_nom . p + base_nom,  R = R_base_nom . R_6d
        # 用**名义**臂座旋转而非实时：实时会把底盘位移算两遍（末端的移动
        # 被算两次），且底盘 yaw=0 时 R_base=diag(-1,-1,1) 是个常量，
        # 漏掉它会让目标 x/y 整体反号。
        _bp_nom, _bR_nom = self._nominal_base()
        R_tgt = _bR_nom @ our_rot6_to_matrix(r6)
        target_world = _bR_nom @ p + _bp_nom
        seed = (self._q_cmd if self._q_ik_seed is None
                else self._q_ik_seed)

        if self.ik_exec is None:
            q, err = self._ik_core_track(target_world, R_tgt, seed)
        else:
            q, err = self._ik_algo_solve(target_world, R_tgt, seed)
        if q is None:
            # === IK 失败兜底：目标沿半径内收后重解（2026-11-22）===
            # 「失败即保持」会让臂冻结 -> observe 不变 -> 流匹配策略重复同一
            # 命令 -> 永久死锁（抓取永远抬升 +0.000）。rollout 侧早有这层。
            q, err = self._clamp_ik(target_world, R_tgt, seed)
        if q is None:
            return {"ok": False, "err": "IK err=%.2e" % err}

        self._pose_tgt_prev = target_world.copy()
        self._q_ik_seed = np.asarray(q, float).copy()
        res = self.actuate(q, gripper)
        res["q"] = np.asarray(q, float)
        return res

    def _clamp_ik(self, target_world, R_tgt, seed):
        """IK 失败兜底：目标朝名义臂座内收后重解，返回 (q|None, err)。

        可达域主要由到臂座的半径决定，沿半径内收是唯一既不改工具轴指向、
        又能把目标拉回域内的调整（1 cm 窄带在 0.98 档即可跨过）。
        与 simulation/mujoco/rollout_color_grasp._clamp_ik 同源。
        """
        base = np.asarray(ARM_BASE, dtype=float)
        for shrink in (0.98, 0.95, 0.90, 0.85, 0.80, 0.70):
            t_s = base + (np.asarray(target_world, float) - base) * shrink
            if self.ik_exec is None:
                q, err = self._ik_core_track(t_s, R_tgt, seed)
            else:
                q, err = self._ik_algo_solve(t_s, R_tgt, seed)
            if q is not None:
                if os.environ.get('GR00T_DEBUG'):
                    print('[clamp] shrink=%.2f tgt=%s ts=%s' % (shrink, np.round(np.asarray(target_world, float), 3), np.round(t_s, 3)), flush=True)
                return q, err
        return None, 1.0

    def _ik_algo_solve(self, target_world, R_tgt, seed):
        """官方 Algo：种子解 → 全解挑最近 → 实测角重解。返回 (q, err)。"""
        ik = self.ik_exec
        ok, q, info = ik.solve(target_world, R_tgt, np.asarray(seed, float))
        if ok:
            return q, 0.0
        # solve() 内部已在单步失败时试过全解；这里只剩「实测角」这一档
        q_meas = np.clip(self.d.qpos[self._arm_qadr], ik._lo, ik._hi)
        ok2, q2, _ = ik.solve(target_world, R_tgt, q_meas)
        if ok2:
            return q2, 0.0
        return None, 1.0

    def _ik_core_track(self, target_world, R_tgt, seed):
        """ik_core 路径（仅对照用）：5 维约束 + 按目标 yaw 对齐 joint6。"""
        from ..skills import kinematics
        ik = kinematics.get_ik()
        yaw = float(np.arctan2(R_tgt[1, 0], R_tgt[0, 0]))
        err, q = ik.solve_near(target_world, np.asarray(seed, float), reg=0.02)
        if err >= 1e-3 and self._pose_tgt_prev is not None:
            err2, q2 = ik.continue_from(target_world, self._pose_tgt_prev,
                                        np.asarray(seed, float), reg=0.02)
            if err2 < 1e-3:
                err, q = err2, q2
        if err >= 1e-3:
            q_meas = np.clip(self.d.qpos[self._arm_qadr], ik.lo, ik.hi)
            err3, q3 = ik.solve_near(target_world, np.asarray(q_meas, float),
                                     reg=0.02)
            if err3 < 1e-3:
                err, q = err3, q3
        if err >= 1e-3:
            return None, err
        q = ik.align_yaw([(err, q)], yaw)[0][1]
        return q, err

    def step(self, n=1):
        mujoco = self._mujoco
        for _ in range(int(n)):
            mujoco.mj_step(self.m, self.d)
        if self.realtime:
            self._t_next += int(n) * float(self.m.opt.timestep)
            dt = self._t_next - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                self._t_next = time.monotonic()

    def steps_per_control(self, hz):
        """一个控制周期要跑多少个物理步（默认 timestep=2 ms、30 Hz → 17）。

        直连 MuJoCo 跑技能时，技能循环按 1/hz 推进一次；若这里仍返回 1，
        物理时间只有真实控制周期的 1/17，位置伺服远远跟不上目标。
        """
        return max(1, int(round(1.0 / (float(hz) *
                                       float(self.m.opt.timestep)))))

    def settle(self, n=200):
        """无实时节流的物理推进（表示/收敛用）。"""
        mujoco = self._mujoco
        for _ in range(int(n)):
            mujoco.mj_step(self.m, self.d)
        self._t_next = time.monotonic()

    # ---------- 底盘（仅 nav 场景）----------
    @property
    def supports_base(self):
        return self._base_acts is not None

    def move_base(self, vx=0.0, vy=0.0, wz=0.0):
        """速度指令，语义同 P30 SendVelocity：持续有效，直到下一条指令。

        入参是**机体系**（+x = 狗头方向），与 vla_spec.yaml 的
        base_velocity.frame=body 一致。

        === 必须绕 z 反旋一次（2026-11-10 实测）===
        nav 场景的 base_slide_x/base_slide_y 是三条 joint 里的前两条，
        轴固定在**世界** x/y 上；velocity 执行器直接驱动这两条关节，
        所以 ctrl 是**世界系**速度。实测 yaw=1.2 rad、cmd vx=+0.4 时
        底盘只沿世界 +x 走了 12 cm，机体系应为 (0.046, 0.119)。
        不转过一次，狗一偏航「前进」就变成斜着走，与真机不一致。
        越界值按执行器 ctrlrange 夹住（世界系分量，±0.6/±0.4 足够
        包住机体系任何 yaw 下的旋转结果；旋转是保范数的，|v| <= 0.6）。
        """
        if self._base_acts is None:
            return {"ok": False,
                    "err": "当前场景没有底盘执行器（需要 p30_rm65_nav.xml）"}
        cmd = np.array([vx, vy, wz], dtype=float)
        if not np.all(np.isfinite(cmd)):
            return {"ok": False, "err": "底盘指令含 NaN/inf"}
        mujoco = self._mujoco
        mujoco.mj_kinematics(self.m, self.d)
        R = np.array(self.d.xmat[self._chassis_body],
                     dtype=float).reshape(3, 3)
        c, s = float(R[0, 0]), float(R[1, 0])
        world = np.array([c * cmd[0] - s * cmd[1],
                          s * cmd[0] + c * cmd[1], cmd[2]])
        lo = self.m.actuator_ctrlrange[self._base_acts, 0]
        hi = self.m.actuator_ctrlrange[self._base_acts, 1]
        self._base_cmd = np.clip(world, lo, hi)
        self.d.ctrl[self._base_acts] = self._base_cmd
        return {"ok": True, "err": None}

    def set_base_pose(self, x, y, yaw):
        """直接放置底盘位姿（仿真专用，用于停车误差实验）。

        与 move_base 的区别：这条**不经过速度积分**，一步到位。用来复现
        「狗停歪了」的初态，而不是验证行驶控制。会清零速度指令。
        """
        if self._base_acts is None:
            return {"ok": False, "err": "当前场景没有底盘"}
        adr = [self.m.joint(nm).qposadr[0] for nm in
               ("base_slide_x", "base_slide_y", "base_hinge_z")]
        self.d.qpos[adr[0]] = float(x)
        self.d.qpos[adr[1]] = float(y)
        self.d.qpos[adr[2]] = float(yaw)
        self.d.qvel[[self.m.joint(nm).dofadr[0] for nm in
                    ("base_slide_x", "base_slide_y", "base_hinge_z")]] = 0.0
        self._base_cmd = np.zeros(3)
        self.d.ctrl[self._base_acts] = self._base_cmd
        self._mujoco.mj_forward(self.m, self.d)
        self._refresh_base()
        return {"ok": True, "err": None}

    def get_base_state(self):
        """世界系 (x, y, yaw) + **机体系**速度 (vx, vy, wz)。

        三个 joint 的 qvel 是关节轴方向上的：slide 沿**世界** x/y，
        所以转成机体系要绕 z 反旋一次。真机 SDK 给的就是机体系速度，
        不转的话狗一偏航，日志里的 vx 就和指令对不上。
        """
        if self._base_acts is None:
            return {"pos": None, "yaw": None, "vel": (0.0, 0.0, 0.0)}
        mujoco = self._mujoco
        mujoco.mj_kinematics(self.m, self.d)
        R = self.d.xmat[self._chassis_body].reshape(3, 3)
        pos = self.d.xpos[self._chassis_body]
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        qx, qy, qwz = (float(self.d.qvel[adr]) for _, adr in self._base_jnts)
        c, s = np.cos(yaw), np.sin(yaw)
        vx = c * qx + s * qy
        vy = -s * qx + c * qy
        return {"pos": (float(pos[0]), float(pos[1])), "yaw": yaw,
                "vel": (vx, vy, qwz)}

    # ---------- 观测 ----------
    def get_state(self):
        mujoco = self._mujoco
        mujoco.mj_kinematics(self.m, self.d)   # 保证 site 位姿是最新的
        self._refresh_base()                   # 底盘可能刚动过，臂基座系要重取
        # === 基准是 grasp_site（夹持面中心），不是法兰 eef_site ===
        # 两者沿工具轴差 0.1113 m。数据集（collect/export）已统一到
        # grasp_site，这里必须同源，否则 state 与 action 差 9.93 cm。
        # === eef_9d 口径：臂座系（2026-11-10 定稿）===
        #   pos   = R_base^T . (site_xpos - base_pos)     base_pos 含 z
        #   rot6d = (R_base^T . R_site)[:2, :]
        # 与 collect_color_grasp._eef9_here()、rollout.observe()、
        # eef_frame.eef9_from_mjdata() 完全同源。
        #
        # 此前两版都不对，且都会静默出错：
        #   1. pos = pos_w - ARM_BASE（常量、不旋转）——
        #      底盘一平移就整体偏 304 mm（实测）。
        #   2. pos = pos_w - [live_x, live_y, ARM_BASE[2]] ——
        #      纯平移时恰好对，yaw!=0 时漂 22~66 mm；rot6d 更是完全
        #      没旋转到臂座系，yaw 0.35 时偏 20°。
        # 判据：同一组关节角，只改底盘位姿，eef_9d 必须完全不变。
        pos_eef = self._base_R.T @ (self.d.site_xpos[self._grasp_site]
                                    - self._base)
        rot_eef = self._base_R.T @ np.array(
            self.d.site_xmat[self._grasp_site],
            dtype=float).reshape(3, 3)
        return {
            "joint": self.d.qpos[self._arm_qadr].copy().astype(np.float64),
            "joint_cmd": self._q_cmd.copy(),
            "gripper": float((0.0245 - self.d.qpos[self._grip_qadr[0]]) /
                             (0.0245 - (-0.020))),
            "gripper_cmd": float((0.0245 - self._grip_cmd) /
                                 (0.0245 - (-0.020))),
            "eef_pos": pos_eef.astype(np.float64),
            "eef_rot": rot_eef.astype(np.float64),
            "t": float(self.d.time),
        }

    def get_images(self):
        """渲染双相机。

        === renderer 必须在本线程首次使用时才创建 ===
        EGL 的 GL context 绑定在创建它的**线程**上。若在 connect()（主线程）
        建好 Renderer、随后在 L3 的循环线程里调 render()，会直接段错误
        （实测：进程「core dumped」，无任何 Python 栈，极难定位）。
        所以这里惰性创建 —— 第一次调用的线程就是之后一直用的线程。
        """
        if not self.render_enabled:
            return {}
        if self.renderer is None:
            self.renderer = self._mujoco.Renderer(self.m, CAM_H, CAM_W)
        out = {}
        for name in CAM_NAMES:
            self.renderer.update_scene(self.d, camera=name)
            out[name] = self.renderer.render().copy()
        return out

    def get_objects(self):
        """真值物体：世界系位置 + 颜色（从 geom_rgba 反查最接近的调色板项）。

        yaw 从 freejoint 四元数取（绕 z）。用于 oracle 策略与成功判定；
        真机后端不实现这个能力，L2 必须能走视觉 grounding 的路径。
        """
        from ..spec import COLOR_RGBA
        mujoco = self._mujoco
        mujoco.mj_kinematics(self.m, self.d)
        out = []
        pal = list(COLOR_RGBA.items())
        for nm, bid, gid in zip(self._obj_names, self._obj_body_ids,
                                self._obj_geom_ids):
            rgba = self.m.geom_rgba[gid][:3]
            # 取最近调色板项（写入时是精确值，浮点比较用 argmin 更稳）
            color = min(pal, key=lambda kv: float(
                np.linalg.norm(np.asarray(kv[1][:3]) - rgba)))[0]
            qadr = self.m.joint(nm + "_joint").qposadr[0]
            quat = self.d.qpos[qadr + 3:qadr + 7]
            yaw = float(np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                                   1 - 2 * (quat[2] ** 2 + quat[3] ** 2)))
            out.append({"name": nm, "color": color,
                        "pos": self.d.xpos[bid].copy().astype(np.float64),
                        "yaw": yaw})
        return out

    # ---------- 便捷属性 ----------
    @property
    def table_top(self):
        from ..spec import TABLE_TOP_Z
        return TABLE_TOP_Z

    @property
    def cube_half(self):
        from ..spec import CUBE_HALF
        return CUBE_HALF

    def camera_model(self, name):
        """返回该相机的内外参（供视觉 grounding 用）。

        Must mj_kinematics first: cam_xpos/cam_xmat 只在 forward 后有效，
        而 L2 可能在两次 get_state 之间调用。data 内部缓冲区要 copy。
        """
        from ..perception.color_grounding import CameraModel
        mujoco = self._mujoco
        mujoco.mj_kinematics(self.m, self.d)
        return CameraModel.from_mujoco(self.m, self.d, name, CAM_W, CAM_H)

    def camera_info(self):
        """相机内外参的可序列化形式（经状态帧广播给 L2 用）。"""
        mujoco = self._mujoco
        mujoco.mj_kinematics(self.m, self.d)
        out = {}
        for name in CAM_NAMES:
            cid = self.m.camera(name).id
            out[name] = {
                "fovy": float(self.m.cam_fovy[cid]),
                "pos": np.array(self.d.cam_xpos[cid], dtype=float),
                "rot": np.array(self.d.cam_xmat[cid],
                                dtype=float).reshape(3, 3).copy(),
                "w": CAM_W, "h": CAM_H,
            }
        return out

    @property
    def arm_base(self):
        """臂基座世界坐标（从模型实取，不是 mount.yaml 的推算值）。"""
        if self.d is not None:
            self._mujoco.mj_kinematics(self.m, self.d)
            self._refresh_base()
        return self._base.copy()

    @property
    def arm_base_yaw_deg(self):
        return MOUNT_YAW_DEG
