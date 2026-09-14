#!/usr/bin/env python3
"""make_orbit_video v2 — 几何一致的"环绕拍摄"合成测试视频。

背景：世界空间中的彩色点阵平面（带正确视差），人脸按 pose_explicit(-θ) 旋转。
等价于：静态人脸 + 相机环绕 → 对 COLMAP 是合法的刚性场景，特征充足。
人脸覆盖区域用三角形光栅化 mask 从渲染帧合成，边缘按分辨率比例羽化。
默认 4K UHD（3840×2160）+ H.264。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "prc", ROOT / "makeup-skill" / "scripts" / "preview_render.py")
prc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prc)


def face_mask(r, yaw_deg: float, w: int, h: int, k: int = 5) -> np.ndarray:
    """三角形光栅化的人脸覆盖 mask（0/1，float32）。k = 羽化核（随分辨率缩放）。"""
    V = r.model.pose_explicit(yaw_deg, -2.0, 0.0, 0.03, 0.12)
    depth = r.d - V[:, 2]
    px = w / 2 + V[:, 0] * r.f / depth
    py = h / 2 - V[:, 1] * r.f / depth
    mask = np.zeros((h, w), np.uint8)
    P2 = np.stack([px, py], axis=1)
    for tri in r.model.tris:
        pts = np.round(P2[tri]).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
    return cv2.GaussianBlur(mask, (k | 1, k | 1), 0).astype(np.float32) / 255.0


def main(out_path: str, seconds: float = 24.0, fps: float = 25.0,
         width: int = 3840, height: int = 2160,
         n_dots: int = 7000, seed: int = 3):
    rng = np.random.default_rng(seed)
    r = prc.get_renderer(width, height)
    h, w = height, width
    s = height / 538.0                            # 相对 640×538 基准的分辨率比例

    # 世界空间点阵背景（z=-2.5 平面，绕脸中心旋转时投影有正确视差）
    pts = np.stack([rng.uniform(-7, 7, n_dots), rng.uniform(-4.5, 4.5, n_dots),
                    np.full(n_dots, -2.5)], axis=1)
    dot_r = rng.uniform(1.5, 4.5, n_dots) * s
    hue = rng.uniform(0, 180, n_dots)
    sat = rng.uniform(120, 255, n_dots)
    val = rng.uniform(120, 255, n_dots)
    colors = np.stack([hue, sat, val], axis=1).astype(np.uint8)[:, None, :]
    colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[:, 0, :]

    vw = prc.open_video_writer(out_path, fps, (w, h))
    n = int(seconds * fps)
    for i in range(n):
        u = i / (n - 1)
        yaw = -55.0 * np.cos(np.pi * u)
        # 静态场景 + 相机环绕 ≡ 整个场景绕脸中心 Ry(-yaw) + 正面相机
        ang = np.deg2rad(yaw)
        cy, sy = np.cos(ang), np.sin(ang)
        R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        pts_t = pts @ R.T
        depth = r.d - pts_t[:, 2]
        px = (w / 2 + pts_t[:, 0] * r.f / depth).astype(int)
        py = (h / 2 - pts_t[:, 1] * r.f / depth).astype(int)
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:] = (18, 16, 15)
        for j in range(n_dots):
            if 0 <= px[j] < w and 0 <= py[j] < h and depth[j] > 0.2:
                cv2.circle(frame, (px[j], py[j]), int(dot_r[j]),
                           tuple(int(c) for c in colors[j]), -1, cv2.LINE_AA)
        face = r.render_still([], intensity=0.0, yaw_deg=float(yaw), smile=0.12)
        m = face_mask(r, float(yaw), w, h, k=int(round(5 * s)) | 1)[..., None]
        frame = (frame * (1 - m) + face * m).astype(np.uint8)
        vw.write(frame)
        if i % 60 == 0:
            print(f"[orbit2] {i}/{n} 帧", flush=True)
    vw.release()
    print(f"OK {out_path}: {n} 帧 @ {fps}fps ({seconds}s), {w}x{h}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="../out/orbit2.mp4")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--n-dots", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    main(a.out, a.seconds, a.fps, a.width, a.height, a.n_dots, a.seed)
