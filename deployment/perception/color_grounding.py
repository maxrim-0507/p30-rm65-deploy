# -*- coding: utf-8 -*-
"""颜色 grounding：图像像素 → 台面平面上的 3D 点。

=== 为什么用色相而不是 RGB 距离 ===
场景里的方块是纯色 geom，但渲染带光照：同一块红方块在亮处是
(217,38,31)、在阴影里可能是 (120,20,18)。RGB 欧氏距离会被亮度
主导，换一个光就认错。色相在亮度/阴影下基本不变 —— 实测四个
颜色的色相中心分别是 0° / 49° / 133° / 226°，最小间隔 24°，
取 ±12° 的容忍带互不重叠。

=== 为什么投到台面平面而不是做深度 ===
Phase 1 是静态抓取，物体都躺在 z=table_top 的台面上，方块中心
比台面高 cube_half。知道台面高度后，像素 + 相机位姿就能解出唯一
的 3D 点，不需要深度相机。这是这项任务能用「单目 + 已知平面」
绕开深度传感器的关键前提，也是 docs §4.1 里把深度列为可选的原因。

=== 相机模型从哪来 ===
MuJoCo 的相机是无畸变针孔，fovy 是**垂直**视场。真机相机的内参
必须标定后填进来（见 `CameraModel.from_matrix`）。不要拿仿真的
fovy 直接套真机 —— 分辨率和视场都不一样。
"""
import numpy as np

# 色相中心（OpenCV H 通道 0..179 口径，即角度/2）
# 由 build_scene.py 的 COLOR_RGBA 换算（colorsys 标准公式），
#   红 (0.85,0.15,0.12) -> 2.47°  -> H 1.2
#   黄 (0.90,0.75,0.10) -> 48.75° -> H 24.4
#   绿 (0.15,0.60,0.25) -> 133.3° -> H 66.7
#   蓝 (0.15,0.30,0.80) -> 226.2° -> H 113.1
# 改调色板必须同步改这里，否则阈值整体错位且不报错（只是检测不到）。
HUE_CENTER = {"red": 1.2, "yellow": 24.4, "green": 66.7, "blue": 113.1}
# 最小间隔 = 红↔黄 23.2，取 ±10 留 3.2 的空档
HUE_TOL = 10.0
SAT_MIN = 80          # 台面/地面是低饱和灰白，被这条排除
VAL_MIN = 35          # 排除深阴影，但保留暗处的方块
MIN_AREA_PX = 40      # 240×320 下 6cm 方块约 300~900 像素，40 是宽松下限


class CameraModel:
    """针孔相机：焦距（像素）+ 外参（世界系）。

    统一用 fovy 作为参数化：换成别的分辨率时焦距要一起变，但视场不变。
    只存 fx/fy 会在「渲染 320×240、标注 640×480」时静默错位。
    """

    def __init__(self, width, height, fovy_deg=None, pos=None, rot=None,
                 fx=None, fy=None, cx=None, cy=None):
        """两种参数化二选一：

        - fovy_deg  ：仿真口径（无畸变、正方形像素、主点在画面中心）；
        - fx/fy/cx/cy：真机标定口径。fx != fy 或主点偏心时只能走这一种，
          否则按 fovy 折算会把畸变/偏心当成 2~5 px 的系统误差。

        ray() 的 +0.5 是 MuJoCo/OpenGL 的像素中心约定。OpenCV 标定的
        cx/cy 直接填进来即可：两者在 1 m 处差半像素 ≈ 1 mm，小于本项目
        5 mm 级的反投影误差，不为它分叉坐标约定。
        """
        self.width = int(width)
        self.height = int(height)
        if fx is None or fy is None:
            if fovy_deg is None:
                raise ValueError("CameraModel 需要 fovy_deg 或 fx/fy")
            f = (self.height / 2.0) / np.tan(
                np.radians(float(fovy_deg)) / 2.0)
            fx = fy = f
        self.fx = float(fx)
        self.fy = float(fy)
        self.fovy_deg = float(fovy_deg) if fovy_deg is not None else float(
            np.degrees(2.0 * np.arctan((self.height / 2.0) / self.fy)))
        self.cx = self.width / 2.0 if cx is None else float(cx)
        self.cy = self.height / 2.0 if cy is None else float(cy)
        self.pos = np.zeros(3) if pos is None else np.asarray(pos, float)
        self.rot = np.eye(3) if rot is None else np.asarray(rot, float)

    @classmethod
    def from_fovy(cls, width, height, fovy_deg, pos=None, rot=None):
        return cls(width, height, fovy_deg, pos=pos, rot=rot)

    @classmethod
    def from_mujoco(cls, model, data, cam_name, width, height):
        """从已 forward 过的 MjModel/MjData 取某台相机的内外参。

        注意 `data.cam_xmat` 取出来后必须立刻 copy：它指向 MjData 内部
        缓冲区，下一次 mj_kinematics 会覆盖（这个坑踩过，会得出
        「相机位姿永远不变」的假结论）。
        """
        cid = model.camera(cam_name).id
        fovy = float(model.cam_fovy[cid])
        pos = np.array(data.cam_xpos[cid], dtype=float).copy()
        rot = np.array(data.cam_xmat[cid], dtype=float).reshape(3, 3).copy()
        return cls(width, height, fovy, pos=pos, rot=rot)

    # ---------- 投影 ----------
    def ray(self, u, v):
        """像素 (u,v) → 世界系单位方向向量。u/v 可用浮点（亚像素）。"""
        # MuJoCo/OpenGL 相机自身看向 -z，right=+x_cam，up=+y_cam
        dx = (np.asarray(u, float) + 0.5 - self.cx) / self.fx
        dy = -(np.asarray(v, float) + 0.5 - self.cy) / self.fy
        d_cam = np.stack([dx, dy, -np.ones_like(dx)], axis=-1)
        d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
        # rot 的列是相机轴在世界系下的表示
        return d_cam @ self.rot.T

    def project(self, pts_world):
        """世界点 → (u, v, depth)。depth>0 表示在相机前方。"""
        p = np.asarray(pts_world, float) - self.pos
        c = p @ self.rot                       # 世界 → 相机
        z = -c[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * c[..., 0] / z + self.cx - 0.5
            v = -self.fy * c[..., 1] / z + self.cy - 0.5
        return u, v, z

    def hit_plane(self, u, v, plane_z):
        """像素射线与 z=plane_z 水平面的交点。返回 (xyz, 有效掩膜)。"""
        d = self.ray(u, v)
        dz = d[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (plane_z - self.pos[2]) / dz
        ok = np.isfinite(t) & (t > 0) & (dz < -1e-6)   # 只取朝下看的射线
        t = np.where(ok, t, 0.0)
        p = self.pos + d * t[..., None]
        return p, ok


def detect_blobs(img_rgb, color, hue_tol=HUE_TOL, sat_min=SAT_MIN,
                 val_min=VAL_MIN, min_area=MIN_AREA_PX):
    """在图像里找某一颜色的色块，返回按面积降序的
    [{"uv": (u,v), "area": int, "bbox": (x0,y0,x1,y1)}]。

    用 HSV 阈值 + 连通域。不引 YOLO：颜色是 4 个已知类别，
    训练一个检测器对这种受控场景是过度设计，而且多一层误差来源。
    """
    import cv2

    rgb = np.asarray(img_rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return []
    hsv = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.int16)            # 0..179
    center = HUE_CENTER[color]
    # 色相环首尾相接：红在 0 附近，要跨 ±180 绕回
    diff = np.abs(h - center)
    diff = np.minimum(diff, 180 - diff)
    mask = ((diff <= hue_tol) &
            (hsv[..., 1] >= sat_min) &
            (hsv[..., 2] >= val_min)).astype(np.uint8)
    if mask.sum() < min_area:
        return []
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, hgt = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        out.append({"uv": (float(cents[i][0]), float(cents[i][1])),
                    "area": area, "bbox": (x0, y0, x0 + w, y0 + hgt)})
    out.sort(key=lambda d: -d["area"])
    return out


def ground_color(img_rgb, cam, color, table_z, cube_half=0.0, max_blobs=4,
                 bounds=None):
    """图像 → 该颜色的候选 3D 位置（世界系，方块**中心**高度）。

    === 投到「方块中心高度」的平面，不是顶面 ===
    实体方块的剪影质心 ≈ 质心在该视角下的投影（凸体的性质），所以
    剪影质心的射线本来就该和**过方块中心的水平面**求交。

    旧代码投顶面（table+2·cube_half），那是被已删除的 head_cam 校准出来
    的：head_cam 俯角 72.9°，顶面在画面里占绝对主导，质心被拉向顶面。
    现在的臂载相机（arm_cam，SEARCH_Q 俯角 −47.7°）两种平面差一个量级。
    实测（2026-11-21，本机渲染，物体带 x 0.455~0.475，4 色）：
        底盘距目标   顶面平面误差     中心平面误差
         0.71 m      4~23 mm         1~11 mm
         1.01 m      22~30 mm        1~7 mm
         1.41 m      30~38 mm        1~7 mm
         1.61 m      55~63 mm        1~7 mm
    顶面误差随距离线性增大（浅俯角时剪影里背面占比变大），1.41 m 时
    红色会偏到 x=0.426，被 GROUND_BOUNDS 的下界 0.43 拒掉 —— 也就是
    「看得见但定位被过滤掉」。所以必须改中心平面。

    === bounds 为什么必须有 ===
    相机能看到狗自己的身体，那里也有高饱和色块（实测红色 3948 px，比
    真方块还大）。不按台面矩形过滤会稳定选错目标，而且**不报错**。
    bounds=((x0,x1),(y0,y1)) 时丢弃落在台面外的候选。

    返回 [{"uv", "area", "pos": xyz}]，按面积降序（面积大 = 完整、
    遮挡少）。调用方决定最终选哪个。
    """
    blobs = detect_blobs(img_rgb, color)[:max_blobs]
    if not blobs:
        return []
    uv = np.array([b["uv"] for b in blobs], dtype=float)
    plane_z = table_z + cube_half                 # 方块中心
    pts, ok = cam.hit_plane(uv[:, 0], uv[:, 1], plane_z)
    out = []
    for b, p, good in zip(blobs, pts, ok):
        if not good:
            continue
        q = p.copy()
        q[2] = plane_z
        if bounds is not None:
            (x0, x1), (y0, y1) = bounds
            if not (x0 <= q[0] <= x1 and y0 <= q[1] <= y1):
                continue
        out.append({"uv": b["uv"], "area": b["area"], "pos": q})
    out.sort(key=lambda d: -d["area"])
    for o in out:
        o["pos_plane_z"] = float(o["pos"][2])      # 实际求交平面（=中心）
    return out


def ground_scene(img_rgb, cam, table_z, cube_half=0.0, colors=None,
                 bounds=None):
    """整幅图 → 每个颜色取最大的一个色块。返回 {color: pos|None}。"""
    colors = colors or list(HUE_CENTER)
    out = {}
    for c in colors:
        cand = ground_color(img_rgb, cam, c, table_z, cube_half, max_blobs=1,
                            bounds=bounds)
        out[c] = cand[0] if cand else None
    return out
