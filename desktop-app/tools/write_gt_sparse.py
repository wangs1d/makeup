#!/usr/bin/env python3
"""write_gt_sparse — 用合成视频的已知位姿直接写 COLMAP 3.11 二进制稀疏模型。

用途：本机 COLMAP SfM 不可用（几何验证/初始化异常）时，绕过 SfM 继续
验证 isolate/fit/brush 链路。位姿与 make_orbit_video2.py 严格一致：
    世界系 = 静态人脸（canonical 归一化坐标，yaw=0）
    相机 k 环绕：C' = Ry(-yaw_k) @ (0,0,2.3)，yaw_k = -55°·cos(π·u)
点云 = canonical 人脸表面采样 + 背景点阵（与视频同种子）。
"""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "prc", ROOT / "makeup-skill" / "scripts" / "preview_render.py")
prc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prc)

W, H = 640, 536
F = H * 1.85
D = 2.3


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """(w,x,y,z)，COLMAP 约定。"""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def main(project: Path, n_frames: int = 120, fps: float = 5.0, seconds: float = 24.0):
    r = prc.get_renderer(W, H)
    FLIP = np.diag([1.0, -1.0, -1.0])            # 内核坐标 → COLMAP 相机坐标

    out = project / "colmap" / "sparse_gt"
    out.mkdir(parents=True, exist_ok=True)

    # ---------- cameras.bin：SIMPLE_PINHOLE ----------
    with open(out / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iIQQQ", 1, 0, W, H, 3))
        f.write(struct.pack("<3d", F, W / 2, H / 2))

    n = int(seconds * fps)
    # ---------- 先算每帧可见点的 2D 观测（供 images.bin 与 points3D track 使用） ----------
    FLIP2 = np.diag([1.0, -1.0, -1.0])
    obs: list[dict[int, tuple[int, float, float]]] = []   # frame → {pt_idx: (idx2d, x, y)}
    poses = []
    for k in range(n):
        u = k / (n - 1)
        yaw = np.deg2rad(-55.0 * np.cos(np.pi * u))
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                       [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Ry_neg = Ry.T
        C = Ry_neg @ np.array([0.0, 0.0, D])
        Rw2c = FLIP @ Ry_neg.T
        t = -Rw2c @ C
        poses.append((rot_to_quat(Rw2c), t, C, Rw2c))
        obs.append({})

    face_cam_visible = []
    # ---------- 点云：人脸表面 + 背景点阵（与视频同种子） ----------
    V = r.model.base
    tris = r.model.tris
    area = 0.5 * np.linalg.norm(np.cross(V[tris[:, 1]] - V[tris[:, 0]],
                                         V[tris[:, 2]] - V[tris[:, 0]]), axis=1)
    rng_f = np.random.default_rng(7)
    rows = rng_f.choice(len(tris), size=6000, p=area / area.sum())
    brng = np.random.default_rng(11)
    b = brng.dirichlet([1, 1, 1], size=6000)
    face_pts = (V[tris[rows, 0]] * b[:, :1] + V[tris[rows, 1]] * b[:, 1:2]
                + V[tris[rows, 2]] * b[:, 2:])
    rng = np.random.default_rng(3)
    n_dots = 7000
    dots = np.stack([rng.uniform(-7, 7, n_dots), rng.uniform(-4.5, 4.5, n_dots),
                     np.full(n_dots, -2.5)], axis=1)
    import cv2
    hue = rng.uniform(0, 180, n_dots).astype(np.uint8)
    sat = rng.uniform(120, 255, n_dots).astype(np.uint8)
    v = rng.uniform(120, 255, n_dots).astype(np.uint8)
    hsv = np.stack([hue, sat, v], axis=1)[:, None, :]
    dot_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :][:, ::-1]  # BGR→RGB
    face_rgb = np.full((len(face_pts), 3), 190, np.uint8)
    xyz = np.concatenate([face_pts, dots]).astype(np.float64)
    rgb = np.concatenate([face_rgb, dot_rgb])

    # ---------- 先算每帧可见点的 2D 观测（供 images.bin 与 points3D track 使用） ----------
    obs: list[dict[int, tuple[int, float, float]]] = []   # frame → {pt_idx: (idx2d, x, y)}
    poses = []
    for k in range(n):
        u = k / (n - 1)
        yaw = np.deg2rad(-55.0 * np.cos(np.pi * u))
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                       [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Ry_neg = Ry.T
        C = Ry_neg @ np.array([0.0, 0.0, D])
        Rw2c = FLIP @ Ry_neg.T
        t = -Rw2c @ C
        poses.append((rot_to_quat(Rw2c), t, C, Rw2c))
        obs.append({})

    for i in range(len(xyz)):
        p = xyz[i]
        for k in range(n):
            q, t, _C, Rw2c = poses[k]
            pc = Rw2c @ p + t
            if pc[2] < 0.05:
                continue
            x = F * pc[0] / pc[2] + W / 2
            y = F * pc[1] / pc[2] + H / 2
            if 2 <= x < W - 2 and 2 <= y < H - 2:
                obs[k][i] = (len(obs[k]), x, y)

    # ---------- images.bin ----------
    with open(out / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", n))
        for k in range(n):
            q, t, _C, _R = poses[k]
            name = f"frame_{k + 1:05d}.jpg"
            f.write(struct.pack("<i", k + 1))
            f.write(struct.pack("<4d", *q))
            f.write(struct.pack("<3d", *t))
            f.write(struct.pack("<i", 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", len(obs[k])))
            for pt_idx, (_i2, x, y) in sorted(obs[k].items(), key=lambda kv: kv[1][0]):
                f.write(struct.pack("<2dq", x, y, pt_idx + 1))  # 第三字段 = 3D 点 id

    # ---------- points3D.bin ----------
    with open(out / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i in range(len(xyz)):
            track = [(k, obs[k][i][0]) for k in range(n) if i in obs[k]]
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<3d", *xyz[i]))
            f.write(bytes(rgb[i]))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", len(track)))
            for k, i2 in track:
                f.write(struct.pack("<Ii", k + 1, i2))
    print(f"OK {out}  images={n} points={len(xyz)} "
          f"obs_total={sum(len(o) for o in obs)}")
    rgb = np.concatenate([face_rgb, dot_rgb])

    with open(out / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i in range(len(xyz)):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<3d", *xyz[i]))
            f.write(bytes(rgb[i]))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", 0))              # 空 track
    print(f"OK {out}  images={n} points={len(xyz)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../out/face3dgs/proj"))
