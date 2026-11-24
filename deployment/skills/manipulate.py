# -*- coding: utf-8 -*-
"""Manipulate 技能：执行 VLA 输出的动作块，直到夹住并抬起。

=== 两条路径 ===
  policy.kind = gr00t   主路径：VLA 出 40 步绝对 EEF 位姿块，逐步下发位姿执行
  policy.kind = oracle  调试基线：IK 算好的关节链（仅用于链路自检/DAgger 专家）

=== 动作语义（关键，别改错）===
服务端返回的 eef_9d 已经是**绝对位姿**：数据集存绝对位姿，modality 声明
RELATIVE + state_key="eef_9d"，processor 在 decode_action 里调用
unapply_action 做了 delta→abs。所以执行端**不能**再累加，否则动作翻倍。

=== 位姿指令，不在 L2 做 IK ===
机械臂只接受末端位姿，IK 由机械臂控制器内部算。所以这里只发
`backend.actuate_pose(pos, rot6d, gripper)`：
  - 真机：由厂商笛卡尔接口执行（当前未接入，会明确 FAULT，不会静默降级）；
  - 仿真：MujocoBackend 用 ik_core 充当「厂商 IK 替身」。
L2 不再 import ik_core —— 否则真机上会误用仿真 IK，偏差 3.6 cm 起步
且不报错。IK 的种子/分支连续性由后端自己维护。

=== 底盘速度进 VLA（2026-11-09 接入，13D 契约）===
chunk 的第 10:13 列是 [vx, vy, wz]（机体系）。**每一帧都要把它下发给
底盘**，否则模型学会了底盘动作也执行不出来，表现为「模型学不会移动」。
三个执行细节：
  1. 速度是**持续指令**，不是位移增量：后端每帧收到才保持，断流即停。
     控制服务器（L3）有 watchdog，超过 watchdog_ms 没新指令就发 0。
  2. 静态场景（supports_base=False）静默跳过 —— 不是失败，因为这批
     数据的 base_velocity 恒为 0，跳过与发 0 等价。
  3. 限速/限加速度在 L3 的 SafetyLimiter.filter_base_vel 做，**不在这里**。
     L2 只负责「把模型的意图如实转达」，不擅自缩放模型输出。
"""
import os
import time

import numpy as np

from .base import Skill, SkillResult


class ManipulateSkill(Skill):
    name = "manipulate"

    def __init__(self, backend, policy, gripper_close=0.0, **kw):
        super().__init__(backend, **kw)
        self.policy = policy
        self.gripper_close = float(gripper_close)

    def run(self, ctx):
        """ctx 需要：
            q            Pregrasp 给出的关节链（oracle 路径用）
            instruction  语言指令（VLA 用）
        """
        deadline = self._deadline()

        if getattr(self.policy, "direct_joint", False):
            q_chain = ctx.get("q")
            if q_chain is None:
                return SkillResult(False, "FAULT",
                                   "缺少 Pregrasp 给出的关节链")
            return self._run_oracle(q_chain, deadline)
        return self._run_policy(ctx, deadline)

    # ---------- 调试基线：脚本专家 ----------
    def _run_oracle(self, q_chain, deadline):
        q_grp = np.asarray(q_chain["grasp"], float)
        q_up = np.asarray(q_chain["lift"], float)
        q_now = self.backend.get_state()["joint"]

        # 悬停 → 抓取：慢速下压，接近段速度减半以减少冲击
        self.speed_deg_s *= 0.5
        self.move_joints(q_now, q_grp, 1.0, 1.0, deadline=deadline)
        self.speed_deg_s *= 2.0

        # 先以**张开**状态在抓取位稳住，再合爪。
        # 经 L3 控服时指令有 30~100 ms 级滞后；直接合爪会在离方块
        # 1~2 cm 的高处闭合，随后闭着爪继续下压把方块推走。
        self.hold_for(0.35, grip=1.0, q=q_grp)

        # 闭合：给足时间建立接触力。经验值 0.8 s。
        # 必须传 q_grp：此刻实测角仍可能落后目标，用实测值 hold 会让
        # 夹爪停在高处、抬升滑脱。
        self.hold_for(0.8, grip=self.gripper_close, q=q_grp)

        self.move_joints(q_grp, q_up, self.gripper_close,
                         self.gripper_close, deadline=deadline)
        self.hold_for(0.3, grip=self.gripper_close, q=q_up)

        return SkillResult(True, "MANIP_DONE", "oracle 抓取完成",
                           {"q_up": q_up})

    # ---------- 主路径：VLA ----------
    def _run_policy(self, ctx, deadline):
        """执行 VLA 动作块。

        === 执行 execute_steps 步再重规划 ===
        官方 SO100 / DROID 的做法：chunk=16 只执行前 8 步。推理延迟
        （Thor 上数百毫秒）被隐藏在剩余 8 步的执行窗口里。
        """
        if not getattr(self.backend, "supports_pose_command", False):
            return SkillResult(
                False, "FAULT",
                "后端 %s 不支持位姿指令；VLA 路径必须走笛卡尔控制"
                % self.backend.name)

        steps_done = 0
        max_steps = int(self.policy.chunk_size) * 20     # 兜底上限（实测成功需 ~440，320 不够）

        # === 精确同步执行（inline 专用，2026-11-24）===
        # rollout 每动作精确推进 STEPS_PER_FRAME=16 个物理步（32 ms），且
        # 推理期间物理完全静止。默认的异步 L3 循环每动作会跑 ~25 步
        # （backend.step 睡 17 ms + 循环 sleep 33 ms），在混沌系统里逐步
        # 累积成 4~10 cm 的横向偏差（实测：同布局同起点，rollout 成功、
        # 部署抓到邻块）。这里让 L3 主循环让出物理，由 apply_step 精确推进。
        enter_sync = getattr(self.backend, "enter_sync", None)
        apply_step = getattr(self.backend, "apply_step", None)
        exit_sync = getattr(self.backend, "exit_sync", None)
        fresh_images = getattr(self.backend, "fresh_images", None)
        n_phys = 16           # 与 rollout 的 STEPS_PER_FRAME 一致
        if enter_sync is not None:
            enter_sync()
        try:
            return self._run_policy_loop(ctx, deadline, steps_done, max_steps,
                                         n_phys, apply_step, fresh_images)
        finally:
            if exit_sync is not None:
                exit_sync()

    def _run_policy_loop(self, ctx, deadline, steps_done, max_steps,
                         n_phys, apply_step, fresh_images):
        from ..policy.base import PolicyObs
        chunk = None
        while steps_done < max_steps and time.monotonic() < deadline:
            if chunk is None or len(chunk) == 0:
                st = self.backend.get_state()
                # 手动步进下快照不再每 33 ms 更新，必须显式要一帧新鲜图
                # （渲染留在 L3 属主线程，EGL context 有线程亲和性）。
                if fresh_images is not None:
                    imgs = fresh_images()
                else:
                    imgs = self.backend.get_images()
                obs = PolicyObs(
                    images=imgs,
                    state=st,
                    instruction=ctx.get("instruction", ""),
                    eef_pos=self._eef_base(st),
                )
                if os.environ.get("GR00T_SAVE_OBS") and steps_done == 0 and not getattr(self, "_obs_saved", False):
                    self._obs_saved = True
                    from PIL import Image as _Im
                    _keys = list(obs.images.keys())
                    for _k, _v in obs.images.items():
                        _Im.fromarray(np.asarray(_v, np.uint8)).save(
                            "/tmp/depobs_%s.png" % _k.replace(".", "_"))
                    _bs2 = self.backend.get_base_state()
                    print("[saveobs] keys=%s base_pos=%s yaw=%s eef=%s"
                          % (_keys, _bs2.get("pos"), _bs2.get("yaw"),
                             np.round(np.asarray(st["eef_pos"], float), 3)),
                          flush=True)
                    print("[saveobs] obs_means=%s" % ({_k: float(np.asarray(_v).mean()) for _k,_v in obs.images.items()}), flush=True)
                    try:
                        _fr = self.backend.get_images()
                        for _k2,_v2 in _fr.items():
                            _Im.fromarray(np.asarray(_v2,np.uint8)).save("/tmp/depfresh_%s.png"%str(_k2).replace(".","_"))
                        print("[saveobs] fresh_means=%s" % ({_k3: float(np.asarray(_v3).mean()) for _k3,_v3 in _fr.items()}), flush=True)
                    except Exception as _e:
                        print("[saveobs] fresh_err=%r"%_e, flush=True)
                if os.environ.get("GR00T_DEBUG"):
                    _bs = self.backend.get_base_state()
                    _bp = _bs.get("pos")
                    print("[vla] steps=%d base_xy=%s yaw=%s eef=%s grip=%.2f q=%s"
                          % (steps_done,
                             None if _bp is None else np.round(_bp, 3),
                             None if _bs.get("yaw") is None else round(float(_bs["yaw"]), 3),
                             np.round(obs.eef_pos, 3),
                             float(st["gripper"]),
                             np.round(np.asarray(st["joint"]), 3)), flush=True)
                # 推理期间底盘必须停：L3 是异步 100Hz 循环，若不停，
                # 旧速度会在 300ms 推理窗口里继续推进物理，底盘位置与
                # 训练/rollout 分叉 -> state/图像分叉 -> 动作分叉（实测）。
                if getattr(self.backend, "supports_base", False):
                    self.backend.move_base(0.0, 0.0, 0.0)
                chunk, info = self.policy.infer(obs, frame_t=steps_done)
                if chunk is None or len(chunk) == 0:
                    return SkillResult(False, "FAULT",
                                       "策略无输出：%s" % info)

            n_exec = min(int(self.policy.execute_steps), len(chunk))
            for a in chunk[:n_exec]:
                a = np.asarray(a, float)
                g = float(np.clip(a[9], 0.0, 1.0))    # 夹爪绝对开度
                res = self.backend.actuate_pose(a[:3], a[3:9], g)
                if not res.get("ok"):
                    return SkillResult(False, "FAULT",
                                       "位姿下发失败：%s" % res.get("err", ""))
                if os.environ.get("GR00T_DEBUG") and steps_done % 40 == 0:
                    print("[cmd] steps=%d tgt=%s actual_eef=%s" % (steps_done, np.round(np.asarray(a[:3],float),3), np.round(np.asarray(self._eef_base(self.backend.get_state()),float),3)), flush=True)
                self._push_base(a)
                if apply_step is not None:
                    apply_step(n_phys)      # 精确 16 步，物理时间 32 ms
                else:
                    self._advance()
                    time.sleep(1.0 / self.rate_hz)
                steps_done += 1
            chunk = None

        return SkillResult(True, "MANIP_DONE", "策略执行完毕",
                           {"steps": steps_done})

    # ---------- 辅助 ----------
    def _eef_base(self, st):
        return np.asarray(st["eef_pos"], float)

    def _push_base(self, a):
        """把 chunk 第 10:13 列的底盘速度下发给 L3。

        后端不支持底盘（静态场景/只有臂的真机）时静默跳过：这批数据的
        base_velocity 恒 0，跳过与发 0 等价，不该因此判失败。
        指令非法（NaN/inf）则**报错不吞** —— 模型吐出坏值必须可见，
        否则只会看到底盘不动，查不到根因。
        """
        if not getattr(self.backend, "supports_base", False):
            return
        if a.shape[0] < 13:
            return
        v = np.asarray(a[10:13], float)
        if not np.all(np.isfinite(v)):
            raise ValueError("底盘速度含 NaN/inf：%s" % v)
        self.backend.move_base(float(v[0]), float(v[1]), float(v[2]))
