# -*- coding: utf-8 -*-
"""端到端设计演示：房间里找目标色块 -> 靠近 -> 交班 v36 抓取。

这不是最终策略，而是按设计拆出来的**行为骨架 + 数据生成专家**：
  SEARCH   在原地 360° 扫描，用 arm_cam 视锥 + 遮挡 + 像素面积判断「看见目标」
  APPROACH 由看到的目标位姿算出泊车位（target - u*standoff，再收进训练分布），
           用底盘 P 控制开过去，臂保持 FWD2
  GRASP    交班 v36（真 VLA），由策略自己完成泊车微调 + 预抓 + 抓取 + 抬升

诚实标注（demo 里哪些是替身）：
  - 视锥/遮挡/像素面积判定用**仿真真值**实现，行为等价于部署端的分割前端；
  - 目标世界坐标取**仿真真值**；真实系统由单目+深度/多视重建得到。
  - 底盘里程计取真值。
  这三项都是「感知/定位」的替身；扫描、接近、泊车、抓取的**决策链**是真的。

用法（Thor）:
  cd /home/nio/p30_grasp/simulation/mujoco
  HF_HUB_OFFLINE=1 MUJOCO_GL=egl /home/nio/mjl_collect_env/bin/python \
      /home/nio/p30_grasp/.tmpm/demo_e2e_expert.py probe          # 只探可见性
  ...                                                    demo_e2e_expert.py [seed] [n]

输出: /home/nio/p30_grasp/outputs/e2e_videos/
"""
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np

MJ = "/home/nio/p30_grasp/simulation/mujoco"
sys.path.insert(0, MJ)

from rollout_color_grasp import (ColorGraspEnv, sample_layout, COLORS,
                                 FWD2_START, HERE, STEPS_PER_FRAME)
from local_gr00t_policy import LocalGr00tPolicy
from collect_color_grasp import SPEC, COLOR_NAMES, GRIP_OPEN

CKPT = "/home/nio/p30_grasp/outputs/groot_rel_v36/checkpoints/020000/pretrained_model"
OUTDIR = "/home/nio/p30_grasp/outputs/e2e_videos"
PROBE = (sys.argv[1].lower() if (len(sys.argv) > 1 and not sys.argv[1].isdigit())
         else "")
_args = sys.argv[2:] if PROBE else sys.argv[1:]
SEED = int(_args[0]) if len(_args) > 0 else 7000
N = int(_args[1]) if len(_args) > 1 else 3
START = int(_args[2]) if len(_args) > 2 else 0
W, H, FPS = 640, 480, 30

# ---------------- 空间常量 ----------------
TABLE_C = np.array([0.71, 0.0])      # 台面中心 (mount.yaml)
TABLE_LOOKAT = np.array([0.50, 0.0])
FAR_X = (-2.0, -1.2)
FAR_Y_MAG = (0.85, 1.35)             # 避开 obs_crate_a(-1.4,0) / b(1.78) / c(-1.78)
STAGE_X = (-0.34, -0.12)             # v36 训练分布内的泊车框
STAGE_Y_ABS = 0.12
STAGE_YAW_ABS = 0.15
STANDOFF = 0.70                      # 目标到臂座的水平距离（训练均值附近）
CUBE_SIDE = 0.06                     # 方块边长 (mount.yaml cube_half=0.03)

# ---------------- 行为常量 ----------------
SEARCH_WZ = 0.8                      # rad/s，底盘持续原地转
APPROACH_KP_POS = 1.2
APPROACH_KP_YAW = 2.0
APPROACH_MAX_FRAMES = 300
TOL_XY = 8e-3
TOL_YAW = 1e-2
MIN_PX2 = 25.0                       # 目标投影面积下限（等价于部署端「看得够清」）
# 搜索/接近用的抬臂看远姿态（空地实测可见扇区 15~71°，FWD2 只有 2~4°）
SEARCH_Q = np.radians([0.0, 53.9, 60.0, 0.0, -16.2, -90.0])
MAX_SWEEP = 2.0 * np.pi + 0.4
FRAME_DT = STEPS_PER_FRAME * 0.002   # 每个控制帧推进的仿真时间

PH = {"SEARCH": (255, 200, 0), "APPROACH": (0, 180, 255),
      "TUCK": (200, 120, 255), "GRASP": (0, 220, 0), "FAIL": (255, 0, 0),
      "IDLE": (140, 140, 140)}


# ------------------------------------------------------------------ 感知替身
def target_cam_metric(env, body_id, cam_name="arm_cam"):
    """返回 (可见, 投影面积px^2, 深度m)。视锥 + 遮挡 + 像素面积三关。"""
    mujoco.mj_kinematics(env.m, env.d)
    mujoco.mj_camlight(env.m, env.d)
    cid = env.m.camera(cam_name).id
    cb = int(env.m.cam_bodyid[cid])
    c = env.d.cam_xpos[cid].copy()
    cm = env.d.cam_xmat[cid].reshape(3, 3).copy()
    p = np.array(env.d.xpos[body_id], float)
    v = cm.T @ (p - c)
    dep = -v[2]
    if dep <= 0.02:
        return False, 0.0, dep
    th = np.tan(np.radians(float(env.m.cam_fovy[cid])) / 2)
    tw = th * 320 / 240
    u = 0.5 + 0.5 * (v[0] / dep) / tw
    vv = 0.5 - 0.5 * (v[1] / dep) / th
    if not (0 <= u <= 1 and 0 <= vv <= 1):
        return False, 0.0, dep
    side_px = CUBE_SIDE / (2 * dep * th) * 240
    area = float(max(side_px, 0.0) ** 2)
    vec = p - c
    dist = float(np.linalg.norm(vec))
    gid = np.zeros(1, np.int32)
    hit = mujoco.mj_ray(env.m, env.d, c, vec / dist, None, 1, cb, gid)
    if hit > 0 and int(env.m.geom_bodyid[gid[0]]) != int(body_id) and hit < dist - 0.01:
        return False, area, dep          # 被挡
    return (area >= MIN_PX2), area, dep


# ------------------------------------------------------------------ 底盘
def slide_state(env):
    """滑轨位姿（与 set_base_pose 同一坐标系）。"""
    adr = [env.m.joint(nm).qposadr[0] for nm in
           ("base_slide_x", "base_slide_y", "base_hinge_z")]
    q = env.d.qpos[adr]
    return float(q[0]), float(q[1]), float(q[2])


def hold_arm(env, q=FWD2_START, grip=GRIP_OPEN):
    env.d.ctrl[env.act_arm] = q
    env.d.ctrl[env.act_gl] = grip
    env.d.ctrl[env.act_gr] = grip


def drive_frame(env, vx=0.0, vy=0.0, wz=0.0, q=FWD2_START):
    env.move_base(vx, vy, wz)
    hold_arm(env, q)
    for _ in range(STEPS_PER_FRAME):
        env._physics()


def turn_to(env, yaw_goal, frames, render, color, max_frames=120, q=SEARCH_Q):
    """原地 P 控制转到 yaw_goal（扫描失败后的二次定位用）。"""
    for _ in range(max_frames):
        _, _, yaw = slide_state(env)
        e = float(np.arctan2(np.sin(yaw_goal - yaw), np.cos(yaw_goal - yaw)))
        if abs(e) < 0.05:
            break
        drive_frame(env, 0.0, 0.0, float(np.clip(1.5 * e, -0.8, 0.8)), q=q)
        frames.append(stamp(render(env), color))


def drive_forward(env, dist, frames, render, color, v=0.35, q=SEARCH_Q):
    n = max(1, int(round(abs(dist) / (v * FRAME_DT))))
    for _ in range(n):
        drive_frame(env, v * np.sign(dist), 0.0, 0.0, q=q)
        frames.append(stamp(render(env), color))


def search(env, body_id, frames, render):
    """Arm J1 (vertical axis) full sweep to find target; chassis stays still."""
    color = PH["SEARCH"]
    q0 = SEARCH_Q.copy()
    N = 72
    j1_seq = np.radians(np.linspace(-175.0, 175.0, N))
    for attempt in range(2):
        best = [0.0, 0.0, 0]
        for j in j1_seq:
            q = q0.copy(); q[0] = float(j)
            env.move_base(0.0, 0.0, 0.0)
            env.d.ctrl[env.act_arm] = q
            env.d.ctrl[env.act_gl] = GRIP_OPEN
            env.d.ctrl[env.act_gr] = GRIP_OPEN
            for _ in range(STEPS_PER_FRAME):
                env._physics()
            frames.append(stamp(render(env), color))
            ok, area, dep = target_cam_metric(env, body_id)
            if area > best[0]:
                best = [area, dep, 1 if ok else 0]
            if ok:
                print("  [search] J1=%.0fdeg area=%.1f dep=%.2f"
                      % (np.degrees(j), area, dep), flush=True)
                return True, attempt, float(np.degrees(j)), area, dep
        print("  [search] attempt=%d arm-sweep miss bestarea=%.1f"
              % (attempt, best[0]), flush=True)
        if attempt == 0:
            drive_forward(env, 0.7, frames, render, color)
    ok, area, dep = target_cam_metric(env, body_id)
    return ok, 2, 175.0, area, dep


def approach_goal(env, body_id):
    """由看到的目标算泊车位：站到目标前方 STANDOFF 处并收进训练分布。"""
    tx, ty = env.d.xpos[body_id][:2]
    rx, ry, _ = slide_state(env)
    u = np.array([tx - rx, ty - ry], float)
    u /= max(np.linalg.norm(u), 1e-6)
    dock = np.array([tx, ty]) - u * STANDOFF
    dx = float(np.clip(dock[0], *STAGE_X))
    dy = float(np.clip(dock[1], -STAGE_Y_ABS, STAGE_Y_ABS))
    dyaw = float(np.arctan2(ty - dy, tx - dx))
    dyaw = float(np.clip(dyaw, -STAGE_YAW_ABS, STAGE_YAW_ABS))
    return dx, dy, dyaw


def approach(env, goal, frames, render):
    """3-phase: turn head to DOCK direction first, drive there, then face target."""
    gx, gy, gyaw = goal
    color = PH["APPROACH"]
    n = 0
    for n in range(APPROACH_MAX_FRAMES):
        x, y, yaw = slide_state(env)
        ex, ey = gx - x, gy - y
        dist = float(np.hypot(ex, ey))
        eyaw = float(np.arctan2(np.sin(gyaw - yaw), np.cos(gyaw - yaw)))
        if dist < TOL_XY and abs(eyaw) < TOL_YAW:
            break
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        h = float(np.arctan2(ey, ex))      # world heading from car to dock
        herr = float(np.arctan2(np.sin(h - yaw), np.cos(h - yaw)))
        if dist > 0.12 and abs(herr) > 0.15:
            # phase 1: turn head toward dock direction, no translation
            drive_frame(env, 0.0, 0.0,
                        float(np.clip(APPROACH_KP_YAW * herr, -0.8, 0.8)),
                        q=SEARCH_Q)
        elif dist > 0.08:
            # phase 2: aligned -> drive forward to dock, small side correction
            vx = float(np.clip(APPROACH_KP_POS * dist, 0.0, 0.6))
            vy = float(np.clip(0.1 * APPROACH_KP_POS * (-s * ex + c * ey),
                               -0.3, 0.3))
            wz = float(np.clip(APPROACH_KP_YAW * herr, -0.8, 0.8))
            drive_frame(env, vx, vy, wz, q=SEARCH_Q)
        else:
            # phase 3: at dock -> rotate to face target
            drive_frame(env, 0.0, 0.0,
                        float(np.clip(APPROACH_KP_YAW * eyaw, -0.8, 0.8)),
                        q=SEARCH_Q)
        frames.append(stamp(render(env), color))
    return n + 1


def move_arm(env, q_from, q_to, frames, render, color, n=40):
    """平滑地把关节目标从 q_from 拉到 q_to（底盘速度清零，原地过渡）。"""
    env.move_base(0.0, 0.0, 0.0)
    for k in range(1, n + 1):
        u = k / float(n)
        u = u * u * (3.0 - 2.0 * u)
        env.d.ctrl[env.act_arm] = (1.0 - u) * q_from + u * q_to
        env.d.ctrl[env.act_gl] = GRIP_OPEN
        env.d.ctrl[env.act_gr] = GRIP_OPEN
        for _ in range(STEPS_PER_FRAME):
            env._physics()
        frames.append(stamp(render(env), color))


# ------------------------------------------------------------------ 出图
def stamp(frame, color):
    frame[:6, :, :] = color
    return frame


def encode(frames, path):
    if not frames:
        return False
    p = subprocess.Popen(
        ["/home/nio/mjl_collect_env/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "pipe:0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
        stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    return p.wait() == 0


def make_renderer(env):
    renderer = mujoco.Renderer(env.m, H, W)
    inset = mujoco.Renderer(env.m, 240, 320)      # 机载 arm_cam 画中画
    opt = mujoco.MjvOption()
    for g in range(6):
        opt.geomgroup[g] = 1 if g in (0, 1, 2) else 0
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 120
    # 正仰角：负仰角在远处（distance 大）会把相机压到地板以下，
    # 渲染出来整屏是地板的背面（一片灰紫）
    cam.elevation = 8

    def render(e):
        bx, by, _ = slide_state(e)
        sep = float(np.hypot(bx - TABLE_LOOKAT[0], by - TABLE_LOOKAT[1]))
        # 相机跟着车走：离台面远时拉远看全局，泊车后自动推近看抓取
        cam.lookat[:] = [0.55 * bx + 0.45 * TABLE_LOOKAT[0],
                         0.55 * by + 0.45 * TABLE_LOOKAT[1], 0.60]
        cam.distance = float(np.clip(1.6 + 0.55 * sep, 2.3, 3.6))
        # MuJoCo 自由相机: pos = lookat - dist*(cos az*cos el, sin az*cos el, sin el)。
        # az=120 时相机落在 -y 一侧；机器人在远端起点时 lookat_y 很负，
        # dist=2.9 会把相机推到南墙外(y<-2.45)，画面只剩墙背面一片纯色。
        # 按墙体反解距离上限，把相机压回房间内。
        _ce = np.cos(np.radians(cam.elevation))
        _off = (-np.cos(np.radians(cam.azimuth)) * _ce,
                -np.sin(np.radians(cam.azimuth)) * _ce)
        _lim = cam.distance
        for _l, _o, _lo, _hi in ((cam.lookat[0], _off[0], -2.95, 2.95),
                                 (cam.lookat[1], _off[1], -2.35, 2.35)):
            if abs(_o) > 1e-6:
                _lim = min(_lim, ((_hi - _l) if _o > 0 else (_lo - _l)) / _o)
        cam.distance = float(max(1.5, _lim))
        renderer.update_scene(e.d, camera=cam, scene_option=opt)
        frame = renderer.render().copy()
        # 左下角画中画：机器人自己看到的第一视角
        inset.update_scene(e.d, camera="arm_cam", scene_option=opt)
        small = inset.render()
        y0, x0 = H - 240 - 8, 8
        frame[y0 - 2:y0 + 240 + 2, x0 - 2:x0 + 320 + 2] = 0
        frame[y0:y0 + 240, x0:x0 + 320] = small
        return frame
    return render


# ------------------------------------------------------------------ 探针
def probe():
    env = ColorGraspEnv(view=True, scene=os.path.join(HERE, "p30_rm65_nav.xml"))
    rng = np.random.default_rng(SEED)
    print("cam fovy=%.1f" % float(env.m.cam_fovy[env.m.camera("arm_cam").id]))
    print("base ctrlrange=", env.m.actuator_ctrlrange[env._base_acts].tolist())
    for i in range(5):
        target = COLORS[i % len(COLORS)]
        layout = sample_layout(target, rng, n=env.n_objects)
        if layout is None:
            continue
        side = 1.0 if rng.random() < 0.5 else -1.0
        fx = float(rng.uniform(*FAR_X))
        fy = side * float(rng.uniform(*FAR_Y_MAG))
        env.reset(layout, FWD2_START)
        env.set_base_pose(fx, fy, 0.0)
        for _ in range(STEPS_PER_FRAME):
            env._physics()
        bid = env.m.body(env._target_name(target)).id
        tx, ty = env.d.xpos[bid][:2]
        yaw_face = float(np.arctan2(ty - fy, tx - fx))
        hits, best = [], (0.0, None)
        for k in range(36):
            yaw = -np.pi + 2 * np.pi * k / 36.0
            env.set_base_pose(fx, fy, yaw)
            mujoco.mj_kinematics(env.m, env.d)
            mujoco.mj_camlight(env.m, env.d)
            ok, area, dep = target_cam_metric(env, bid)
            if area > best[0]:
                best = (area, yaw)
            if ok:
                hits.append(yaw)
        print("probe %d %-6s far=(%.2f,%.2f) tgt=(%.2f,%.2f) 正对yaw=%+.2f "
              "命中%d/36 最大area=%.1f@yaw%+.2f" %
              (i, target, fx, fy, tx, ty, yaw_face, len(hits), best[0],
               best[1] if best[1] is not None else 0.0))
        print("        命中yaw:", " ".join("%+.2f" % h for h in hits))

    # ---- 近距离获取距离扫描 + 相机几何 ----
    print("\n--- 距离扫描（arm_cam 在多远处能看见目标）---")
    target = "red"
    layout = sample_layout(target, rng, n=env.n_objects)
    env.reset(layout, FWD2_START)
    bid = env.m.body(env._target_name(target)).id
    for k in range(12):
        x = -1.6 + 0.13 * k
        env.set_base_pose(x, 0.9, 0.0)
        for _ in range(STEPS_PER_FRAME):
            env._physics()
        tx, ty = env.d.xpos[bid][:2]
        yaw = float(np.arctan2(ty - 0.9, tx - x))
        best = None
        for j in range(-30, 31):
            env.set_base_pose(x, 0.9, yaw + 0.02 * j)
            mujoco.mj_kinematics(env.m, env.d)
            mujoco.mj_camlight(env.m, env.d)
            ok, area, dep = target_cam_metric(env, bid)
            if best is None or area > best[2]:
                best = (ok, yaw + 0.02 * j, area, dep)
        print("  x=%+.2f tgt=(%.2f,%.2f) 最近可见=%s yaw=%+.2f area=%.0f dep=%.2f"
              % (x, tx, ty, best[0], best[1], best[2], best[3]))

    env.set_base_pose(-2.2, 0.7, 0.0)
    for _ in range(STEPS_PER_FRAME):
        env._physics()
    mujoco.mj_kinematics(env.m, env.d)
    mujoco.mj_camlight(env.m, env.d)
    cid = env.m.camera("arm_cam").id
    print("cam pos=%s  xmat=" % np.round(env.d.cam_xpos[cid], 3).tolist())
    print(np.round(env.d.cam_xmat[cid].reshape(3, 3), 3))

    # 实测原地转向速率（直接给 ctrl，绕开 drive_frame 只测植物响应）
    print("timestep=%.4f" % env.m.opt.timestep)
    ah = env.m.joint("base_hinge_z").qposadr[0]
    dh = env.m.joint("base_hinge_z").dofadr[0]
    aw = env.m.actuator("base_wz").id
    env.set_base_pose(-2.2, 0.7, 0.0)
    for a, v in zip(env._base_acts, (0.0, 0.0, 0.8)):
        env.d.ctrl[a] = v
    for k in range(6):
        for _ in range(16):
            mujoco.mj_step(env.m, env.d)
        print("  k=%d qvel=%+.3f actfrc=%+.1f yaw=%+.3f"
              % (k, env.d.qvel[dh], env.d.actuator_force[aw], env.d.qpos[ah]))
    # 实测不同 wz 的稳定转向速率
    for w in (0.15, 0.25, 0.35, 0.5, 0.8):
        env.set_base_pose(-2.2, 0.7, 0.0)
        y0 = slide_state(env)[2]
        for _ in range(100):
            drive_frame(env, 0.0, 0.0, w)
        y1 = slide_state(env)[2]
        dy = abs(float(np.arctan2(np.sin(y1 - y0), np.cos(y1 - y0))))
        print("  wz=%.2f 100帧 实测转角=%.3f rad（期望 %.3f）"
              % (w, dy, w * 100 * FRAME_DT))
    # 关掉轮子视觉自转，看底盘是否恢复
    env._wheel = []
    for v in (0.2, 0.4, 0.6):
        env.set_base_pose(-2.2, 0.7, 0.0)
        x0 = slide_state(env)[0]
        for _ in range(60):
            drive_frame(env, v, 0.0, 0.0)
        x1 = slide_state(env)[0]
        print("  [无轮转] vx=%.2f 位移=%.3f m（期望 %.3f）"
              % (v, abs(x1 - x0), v * 60 * FRAME_DT))
    # reset 之后再测（与主流程同一条路径）
    layout2 = sample_layout("red", rng, n=env.n_objects) or layout
    for v in (0.2, 0.4, 0.6):
        env.reset(layout2, FWD2_START)
        env.set_base_pose(-2.2, 0.7, 0.0)
        for _ in range(20):
            env._physics()
        x0 = slide_state(env)[0]
        q0 = env.d.qpos[env.arm_adr].copy()
        for _ in range(60):
            drive_frame(env, v, 0.0, 0.0)
        x1 = slide_state(env)[0]
        arm_drift = float(np.max(np.abs(env.d.qpos[env.arm_adr] - q0)))
        print("  [reset后] vx=%.2f 位移=%.3f m（期望 %.3f） 臂漂移=%.3f rad"
              % (v, abs(x1 - x0), v * 60 * FRAME_DT, arm_drift))
    for v in (0.2, 0.4, 0.6):
        env.set_base_pose(-2.2, 0.7, 0.0)
        x0 = slide_state(env)[0]
        for _ in range(60):
            drive_frame(env, v, 0.0, 0.0)
        x1 = slide_state(env)[0]
        print("  vx=%.2f 60帧 实测位移=%.3f m（期望 %.3f）"
              % (v, abs(x1 - x0), v * 60 * FRAME_DT))
    env.close()


def probe_look():
    """哪些臂姿能在远处看见台面目标（在**空地**上测，避开障碍箱）。"""
    env = ColorGraspEnv(view=True, scene=os.path.join(HERE, "p30_rm65_nav.xml"))
    rng = np.random.default_rng(SEED)
    layout = sample_layout("red", rng, n=env.n_objects)
    env.reset(layout, FWD2_START)
    bid = env.m.body(env._target_name("red")).id
    cid = env.m.camera("arm_cam").id
    adr = [env.m.joint(n).qposadr[0] for n in
           ("base_slide_x", "base_slide_y", "base_hinge_z")]
    far_poses = [(-2.2, 0.7), (-2.0, 1.1), (-1.6, 1.1), (-1.6, -1.1),
                 (-1.3, 0.9), (-1.3, -0.9)]
    cands = [("FWD2", FWD2_START),
             ("LOOK_A q2=54 q3=21", np.radians([0, 53.9, 21.2, 0, -16.2, -90])),
             ("LOOK_B q2=80 q3=0", np.radians([0, 80.0, 0.0, 0, -16.2, -90])),
             ("LOOK_C q2=0 q3=60", np.radians([0, 0.0, 60.0, 0, -16.2, -90])),
             ("LOOK_D q2=54 q3=60", np.radians([0, 53.9, 60.0, 0, -16.2, -90]))]

    def sector(q, fx, fy):
        env.d.qpos[env.arm_adr] = q
        env.d.qpos[adr] = [fx, fy, 0.0]
        mujoco.mj_forward(env.m, env.d)
        tx, ty = env.d.xpos[bid][:2]
        yaw0 = float(np.arctan2(ty - fy, tx - fx))
        vis = []
        for j in range(360):
            env.d.qpos[adr] = [fx, fy, yaw0 + np.radians(j - 180)]
            mujoco.mj_forward(env.m, env.d)
            mujoco.mj_camlight(env.m, env.d)
            ok, area, dep = target_cam_metric(env, bid)
            vis.append(ok)
        vis = np.array(vis)
        vv = np.concatenate([vis, vis])
        run = mx = 0
        for b in vv:
            run = run + 1 if b else 0
            mx = max(mx, run)
        return min(mx, 360), int(vis.sum())

    print("=== 各臂姿在空地远端的可见扇区（1° 扫掠）===")
    for name, q in cands:
        env.d.qpos[env.arm_adr] = q
        env.d.qpos[adr] = [-2.2, 0.7, 0.0]
        mujoco.mj_forward(env.m, env.d)
        mujoco.mj_camlight(env.m, env.d)
        fwd = -np.array(env.d.cam_xmat[cid]).reshape(3, 3)[:, 2]
        pitch = float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1))))
        cz = float(env.d.cam_xpos[cid][2])
        ws = [sector(q, fx, fy)[0] for fx, fy in far_poses]
        print("  %-20s pitch=%+5.1f° cam_z=%.2f 扇区=%s"
              % (name, pitch, cz, ws), flush=True)
    env.close()


# ------------------------------------------------------------------ 主流程
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    env = ColorGraspEnv(view=True, scene=os.path.join(HERE, "p30_rm65_nav.xml"))
    vla = LocalGr00tPolicy(CKPT, seed=SEED, n_action_steps=40)
    rng = np.random.default_rng(SEED)
    render = make_renderer(env)

    for i in range(N):
        target = COLORS[i % len(COLORS)]
        layout = None
        for _ in range(8):
            layout = sample_layout(target, rng, n=env.n_objects)
            if layout is not None:
                break
        if layout is None:
            print("ep %d 摆位失败" % (i + 1), flush=True)
            continue

        side = 1.0 if rng.random() < 0.5 else -1.0
        far_x = float(rng.uniform(*FAR_X))
        far_y = side * float(rng.uniform(*FAR_Y_MAG))
        # 初始朝向：背对房间中心（这样必须先转身才能看到台面）
        yaw_away = float(np.arctan2(far_y - TABLE_C[1], far_x - TABLE_C[0]))
        if i < START:
            continue

        env.reset(layout, FWD2_START)
        env.set_base_pose(far_x, far_y, yaw_away)
        for _ in range(STEPS_PER_FRAME):
            env._physics()
        bid = env.m.body(env._target_name(target)).id
        tgt = env.d.xpos[bid][:2].copy()

        frames = [stamp(render(env), PH["SEARCH"])]
        move_arm(env, FWD2_START, SEARCH_Q, frames, render, PH["SEARCH"])
        found, attempt, swept, area, dep = search(env, bid, frames, render)
        if not found:
            out = os.path.join(OUTDIR, "e2e%02d_%s_FAILsearch.mp4" % (i + 1, target))
            enc = encode(frames, out)
            print("ep %2d %-6s 搜索失败（%d圈）%s" %
                  (i + 1, target, attempt + 1, "已写" + out if enc else ""), flush=True)
            continue
        n_search = len(frames)

        goal = approach_goal(env, bid)
        n_app = approach(env, goal, frames, render)
        x, y, yaw = slide_state(env)
        # 泊车到位后把臂收回训练分布里的 FWD2，再交班给 v36
        move_arm(env, SEARCH_Q, FWD2_START, frames, render, PH["TUCK"])

        # ---- 交班 v36 ----
        vla.reset()
        obs = env.observe()
        instr = SPEC["language"]["template"].format(color=COLOR_NAMES[target])
        ok, lift, reason, t = False, 0.0, "未判定", 0
        for t in range(900):
            a = vla.act(obs, instr, t)
            env.step(a)
            obs = env.observe()
            frames.append(stamp(render(env), PH["GRASP"]))
            j = env.judge(target)
            lift, reason = j["lift"], j["reason"]
            if j["success"] and t > 0:
                ok = True
                break
        if not ok:
            for k in range(3):
                frames.append(stamp(frames[-1], PH["FAIL"]))

        tag = "OK" if ok else "FAIL"
        out = os.path.join(OUTDIR, "e2e%02d_%s_%s.mp4" % (i + 1, target, tag))
        enc = encode(frames, out)
        print("ep %2d %-6s far=(%.2f,%.2f) tgt=(%.2f,%.2f) "
              "搜索=%d帧/%.1frad 视野px=%.0f 泊车=%d帧 残差=(%.3f,%.3f,%.3f) "
              "VLA=%d步 抬升=%+.3f %s 总帧=%d %s"
              % (i + 1, target, far_x, far_y, tgt[0], tgt[1],
                 n_search, swept, area, n_app,
                 goal[0] - x, goal[1] - y, goal[2] - yaw,
                 t + 1, lift, tag, len(frames),
                 "已写" + out if enc else "编码失败"), flush=True)
    env.close()


if __name__ == "__main__":
    if PROBE == "look":
        probe_look()
    elif PROBE:
        probe()
    else:
        main()
