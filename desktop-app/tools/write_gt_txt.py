#!/usr/bin/env python3
"""write_gt_txt — GT 位姿写成 COLMAP TXT 稀疏模型（再经 model_converter 转 BIN）。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "prc", ROOT / "makeup-skill" / "scripts" / "preview_render.py")
prc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prc)

W, H, F, D = 640, 536, 536 * 1.85, 2.3


def rot_to_quat(R):
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        q = np.array([.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, .25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, .25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, .25 * s])
    return q / np.linalg.norm(q)


def main(n: int = 120):
    r = prc.get_renderer(W, H)
    FLIP = np.diag([1.0, -1.0, -1.0])
    out = ROOT / "out" / "face3dgs" / "proj" / "colmap" / "sparse_txt"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cameras.txt").write_text(f"1 SIMPLE_PINHOLE {W} {H} {F:.6f} {W / 2:.1f} {H / 2:.1f}\n")

    V, tris = r.model.base, r.model.tris
    area = 0.5 * np.linalg.norm(np.cross(V[tris[:, 1]] - V[tris[:, 0]],
                                         V[tris[:, 2]] - V[tris[:, 0]]), axis=1)
    rows = np.random.default_rng(7).choice(len(tris), size=6000, p=area / area.sum())
    b = np.random.default_rng(11).dirichlet([1, 1, 1], size=6000)
    face_pts = V[tris[rows, 0]] * b[:, :1] + V[tris[rows, 1]] * b[:, 1:2] + V[tris[rows, 2]] * b[:, 2:]
    rng = np.random.default_rng(3)
    dots = np.stack([rng.uniform(-7, 7, 7000), rng.uniform(-4.5, 4.5, 7000),
                     np.full(7000, -2.5)], axis=1)
    hsv = np.stack([rng.uniform(0, 180, 7000).astype(np.uint8),
                    rng.uniform(120, 255, 7000).astype(np.uint8),
                    rng.uniform(120, 255, 7000).astype(np.uint8)], axis=1)[:, None, :]
    dot_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :][:, ::-1]
    face_rgb = np.full((6000, 3), 190, np.uint8)
    xyz = np.concatenate([face_pts, dots])
    rgb = np.concatenate([face_rgb, dot_rgb])

    poses, obs = [], []
    for k in range(n):
        u = k / (n - 1)
        yaw = np.deg2rad(-55.0 * np.cos(np.pi * u))
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        RyN = Ry.T
        C = RyN @ np.array([0, 0, D])
        R = FLIP @ RyN.T
        poses.append((rot_to_quat(R), -R @ C, R))
        obs.append({})
    for i in range(len(xyz)):
        for k in range(n):
            _q, t, R = poses[k]
            pc = R @ xyz[i] + t
            if pc[2] < 0.05:
                continue
            x = F * pc[0] / pc[2] + W / 2
            y = F * pc[1] / pc[2] + H / 2
            if 2 <= x < W - 2 and 2 <= y < H - 2:
                obs[k][i] = (len(obs[k]), x, y)

    with open(out / "images.txt", "w") as f:
        for k in range(n):
            q, t, _R = poses[k]
            pts = sorted(obs[k].items(), key=lambda kv: kv[1][0])
            f.write("%d %.10g %.10g %.10g %.10g %.10g %.10g %.10g 1 frame_%05d.jpg\n"
                    % (k + 1, *q, *t, k + 1))
            f.write(" ".join("%.4f %.4f %d" % (x, y, pi + 1) for pi, (_j, x, y) in pts) + "\n")
    with open(out / "points3D.txt", "w") as f:
        for i in range(len(xyz)):
            track = " ".join("%d %d" % (k + 1, obs[k][i][0]) for k in range(n) if i in obs[k])
            f.write("%d %.6g %.6g %.6g %d %d %d %d %s\n"
                    % (i + 1, xyz[i, 0], xyz[i, 1], xyz[i, 2],
                       rgb[i, 0], rgb[i, 1], rgb[i, 2], 0.0, track))
    print(f"OK {out} images={n} points={len(xyz)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
