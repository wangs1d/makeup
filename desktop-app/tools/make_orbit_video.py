#!/usr/bin/env python3
"""make_orbit_video — 用渲染内核生成"环绕拍摄"式人脸视频（重建管线端到端测试用）。

模拟用户正对摄像头缓慢左右转头的采集方式：yaw 从 -55° 匀速扫到 +55°，
每个角度渲染一帧（含背景、光照、颗粒，与真实采集一致）。输出 MP4。
默认 4K UHD（3840×2160）+ H.264；小尺寸可传 --width/--height。
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "prc", ROOT / "makeup-skill" / "scripts" / "preview_render.py")
prc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prc)


def main(out_path: str, seconds: float = 24.0, fps: float = 25.0,
         width: int = 3840, height: int = 2160):
    r = prc.get_renderer(width, height)
    vw = prc.open_video_writer(out_path, fps, (width, height))
    n = int(seconds * fps)
    t0 = 1.5
    for i in range(n):
        # -55° → 0（回正停留）→ +55°，与采集引导的 bucket 语义一致
        u = i / (n - 1)
        yaw = -55.0 * np.cos(np.pi * u)
        img = r.render_still([], intensity=0.0, yaw_deg=float(yaw), smile=0.12)
        vw.write(img)
        if i % 60 == 0:
            print(f"[orbit] {i}/{n} 帧", flush=True)
    vw.release()
    print(f"OK {out_path}: {n} 帧 @ {fps}fps ({seconds}s), {width}x{height}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="out/orbit.mp4")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    a = ap.parse_args()
    main(a.out, a.seconds, a.fps, a.width, a.height)
