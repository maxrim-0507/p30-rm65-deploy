# -*- coding: utf-8 -*-
"""部署栈自检：四项，全部通过才谈得上「可以跑闭环」。

  1. codec     ndarray / dict / str 往返无损（含 240×320×3 大图）
  2. bus       PUB/SUB 与 REQ/REP 在同一台机上连通
  3. hal       MuJoCo 后端能加载场景、读状态、渲染双相机
  4. perceive  颜色分割能在真实渲染图上找到四个方块，且反投影误差 < 3 cm

用法：
  MUJOCO_GL=egl python3 -m deployment.selftest
  MUJOCO_GL=egl python3 -m deployment.selftest --skip bus

=== 为什么感知要单独一项 ===
总线与 HAL 的错会直接抛异常；感知的错**不抛异常**，它只是找不到
色块或找到错位置。这类静默错误必须靠「拿真值对一遍」才能发现 ——
所以第 4 项在有真值时做定量对比，而不是只打印「检测到 4 个」。
"""
import argparse
import sys
import time

from .bootstrap import setup

setup()

import numpy as np  # noqa: E402
from .spec import CAM_NAMES, camera_mjcf  # noqa: E402


# ============ 1. codec ============
def test_codec():
    from .bus import dumps, loads
    rng = np.random.default_rng(0)
    cases = {
        "scalar": {"a": 1, "b": 2.5, "c": True, "d": None},
        "str_cn": {"instruction": "抓取红色的方块"},
        "arr_f32": {"x": rng.standard_normal((6,)).astype(np.float32)},
        "arr_u8_img": {"img": rng.integers(0, 256, (240, 320, 3),
                                           dtype=np.uint8)},
        "nested": {"objs": [{"name": "obj_0", "pos": [1.0, 2.0, 3.0],
                             "color": "red"}]},
        "empty": {},
    }
    for name, msg in cases.items():
        back = loads(dumps(msg))
        if name in ("str_cn",):
            assert back["instruction"] == msg["instruction"], name
        elif name in ("arr_f32", "arr_u8_img"):
            k = next(iter(msg))
            assert np.array_equal(back[k], msg[k]), name
            assert back[k].dtype == msg[k].dtype, name
        elif name == "nested":
            assert back["objs"][0]["color"] == "red", name
            assert back["objs"][0]["pos"] == [1.0, 2.0, 3.0], name
        else:
            assert back == msg, name

    # 大图吞吐
    img = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    msg = {CAM_NAMES[0]: img, CAM_NAMES[1]: img}
    t0 = time.monotonic()
    for _ in range(20):
        buf = dumps(msg)
        loads(buf)
    dt = (time.monotonic() - t0) / 20 * 1000
    print("  1. codec     OK  双相机往返 %.2f ms  帧 %.1f KB"
          % (dt, len(buf) / 1024.0))
    return True


# ============ 2. bus ============
def test_bus(port_base=15761):
    from .bus import Publisher, Replier, Requester, Subscriber
    pub = Publisher("127.0.0.1", port_base)
    sub = Subscriber("127.0.0.1", port_base)
    time.sleep(0.15)                       # 给 SUB 的 connect 一点时间
    payload = {"joint": np.arange(6, dtype=np.float64), "text": "你好"}
    for _ in range(5):
        pub.send(payload)
        time.sleep(0.02)
    got = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0:
        got = sub.latest(50)
        if got is not None:
            break
    assert got is not None, "PUB/SUB 没收到"
    assert np.allclose(got["joint"], payload["joint"]), "PUB/SUB 数据不一致"
    assert got["text"] == "你好", "PUB/SUB 中文串不一致"

    rep = Replier("127.0.0.1", port_base + 1)
    import threading
    threading.Thread(target=rep.serve,
                     args=(lambda m: {"ok": True, "echo": m.get("v")},),
                     daemon=True).start()
    time.sleep(0.1)
    req = Requester("127.0.0.1", port_base + 1)
    r = req.call({"v": np.ones(3, np.float32)}, timeout_ms=1500)
    assert r.get("ok") and np.allclose(r["echo"], 1.0), "REQ/REP 异常"
    # 先让 serve 线程退出再关 socket：直接 close 会让 poll() 报
    # "Socket operation on non-socket"（线程还活着但 fd 已经没了）
    rep.stop()
    time.sleep(0.25)
    for s in (pub, sub, rep, req):
        s.close()
    print("  2. bus       OK  PUB/SUB + REQ/REP 连通")
    return True


# ============ 3. hal ============
def test_hal(layout=None):
    from .hal import make_backend
    be = make_backend("mujoco", realtime=False)
    be.connect()
    if layout is not None:
        be.reset(layout)
        be.settle(150)
    st = be.get_state()
    assert st["joint"].shape == (6,), "关节维度不对"
    assert np.all(np.isfinite(st["joint"])), "关节含 NaN"
    tilt = None
    try:
        from .skills import kinematics
        tilt = kinematics.tool_tilt_deg(st["joint"])
    except Exception:
        pass
    imgs = be.get_images()
    assert set(imgs) == set(CAM_NAMES), "相机缺失：%s" % list(imgs)
    for k, v in imgs.items():
        assert v.shape == (240, 320, 3), "%s 尺寸 %s" % (k, v.shape)
        assert v.mean() > 5, "%s 画面几乎全黑（mean=%.1f）" % (k, v.mean())
    objs = be.get_objects()
    be.close()
    print("  3. hal       OK  关节范围 [%.0f°, %.0f°]  工具轴偏 %.1f°  "
          "双相机 mean=%.0f/%.0f  物体 %d"
          % (np.degrees(st["joint"].min()), np.degrees(st["joint"].max()),
             tilt or -1, imgs[CAM_NAMES[0]].mean(),
             imgs[CAM_NAMES[1]].mean(),
             len(objs)))
    return True


# ============ 4. perceive ============
def test_perceive(layout=None, tol_m=0.03):
    from .hal import make_backend
    from .nav.presets import SEARCH_Q_DEG
    from .perception.color_grounding import ground_color
    from .spec import CUBE_HALF, GROUND_BOUNDS, TABLE_TOP_Z, camera_mjcf

    # 用导航场景 + SEARCH_Q：这是感知前端真正的工作点（arm_cam，底盘在
    # 远处扫描）。front_cam 装在躯干前脸、下俯 45°：抓取带 x≈0.46 在它
    # 正下方，1.0 m 外的目标又出到像面上方 —— 它只给 VLA 当上下文，
    # 不参与定位（实测近处 100~300 mm）。
    be = make_backend("mujoco", realtime=False, scene="p30_rm65_nav.xml")
    be.connect()
    if layout is None:
        layout = [(0.455, 0.10, "red", 0.3),
                  (0.470, -0.10, "green", 1.1),
                  (0.460, 0.00, "blue", 0.7),
                  (0.475, 0.16, "yellow", 0.0)]
    be.reset(layout, arm_q=np.radians(SEARCH_Q_DEG))
    be.set_base_pose(-0.55, 0.0, 0.0)    # 目标 ~1.0 m 外（SEARCH/DOCK 距离）
    be.settle(200)
    be.get_state()                       # 触发 kinematics，相机位姿最新

    cam_name = camera_mjcf("arm")
    cam = be.camera_model(cam_name)
    img = be.get_images()[cam_name]

    errs, miss = [], 0
    for (x, y, color, _yaw) in layout:
        cand = ground_color(img, cam, color, TABLE_TOP_Z, CUBE_HALF,
                            max_blobs=4, bounds=GROUND_BOUNDS)
        if not cand:
            miss += 1
            continue
        truth = np.array([x, y, TABLE_TOP_Z + CUBE_HALF])
        d = min(np.linalg.norm(c["pos"] - truth) for c in cand)
        errs.append(d)
    be.close()

    assert miss == 0, "%d 个方块没检测到" % miss
    worst = max(errs)
    mean = float(np.mean(errs))
    assert worst < tol_m, "反投影误差 %.3f m > %.3f m" % (worst, tol_m)
    print("  4. perceive  OK  4/4 检测  反投影误差 mean=%.1f mm  max=%.1f mm"
          % (mean * 1000, worst * 1000))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["codec", "bus", "hal", "perceive"])
    a = ap.parse_args(argv)
    print("[selftest] P30W + RM65 部署栈自检")
    tests = [("codec", test_codec), ("bus", test_bus),
             ("hal", test_hal), ("perceive", test_perceive)]
    fails = []
    for name, fn in tests:
        if name in a.skip:
            print("  %-10s SKIP" % (name + "."))
            continue
        try:
            fn()
        except AssertionError as e:
            fails.append((name, str(e)))
            print("  %-10s FAIL  %s" % (name + ".", e))
        except Exception as e:
            fails.append((name, "%s: %s" % (type(e).__name__, e)))
            print("  %-10s ERROR %s: %s" % (name + ".", type(e).__name__, e))
    print("\n[selftest] %d 项通过 / %d 项失败"
          % (len(tests) - len(a.skip) - len(fails), len(fails)))
    for name, msg in fails:
        print("   - %s: %s" % (name, msg))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
