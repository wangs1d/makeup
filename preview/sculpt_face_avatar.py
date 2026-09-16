#!/usr/bin/env python3
"""sculpt_face_avatar — 参数化"类真人"3DGS 头像生成器（演示/联调/预览基准用）。

在椭球头形上叠加解析特征位移（鼻梁/鼻尖/眼窝+眼球/眉弓/唇体/下巴/颧骨/下颌收窄），
正面撒 ~11 万微小各向异性高斯（σ≈3mm 切向贴面），肤色/眉/唇/虹膜/发色按特征场着色，
法线用有限差分求出后转四元数。同时导出与特征精确对齐的语义锚点 JSON（手动模式输入）。

用法：
    python sculpt_face_avatar.py --out out/demo/avatar-face
产出：
    avatar.ply / anchors.json（喂给 avatar_session preview --anchors）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "makeup-skill" / "scripts"))
from avatar_io import AvatarData, save_avatar  # noqa: E402

A, B, C = 0.345, 0.5, 0.43          # 头部椭球半轴（脸高=1 归一空间）


# ---------------- 特征位移场 D(x, y)（仅正面生效） ----------------

def _seg_dist(px, py, x0, y0, x1, y1):
    """点到线段距离（向量化）。"""
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy + 1e-12
    t = np.clip(((px - x0) * vx + (py - y0) * vy) / L2, 0, 1)
    return np.hypot(px - (x0 + t * vx), py - (y0 + t * vy))


def _bump(px, py, cx, cy, sx, sy=None):
    sy = sx if sy is None else sy
    return np.exp(-0.5 * ((px - cx) / sx) ** 2 - 0.5 * ((py - cy) / sy) ** 2)


def displacement(x, y):
    """正面特征位移（沿径向，单位=脸高）。"""
    d = np.zeros_like(x)
    d += 0.055 * _bump(*_seg_pts(x, y, 0, 0.16, 0, -0.04), 0.0, 0.0, sx=0.045)   # 鼻梁
    d += 0.035 * _bump(x, y, 0.0, -0.055, 0.05, 0.045)                        # 鼻尖
    for sx_ in (-1, 1):
        d += 0.02 * _bump(x, y, sx_ * 0.062, -0.10, 0.028)                    # 鼻翼
        d -= 0.038 * _bump(x, y, sx_ * 0.17, 0.155, 0.085, 0.055)             # 眼窝
        d += 0.02 * _bump(x, y, sx_ * 0.17, 0.155, 0.055, 0.038)              # 眼球/眼睑隆起
        d += 0.024 * _bump(x, y, sx_ * 0.17, 0.278, 0.11, 0.032)              # 眉弓
        d += 0.02 * _bump(x, y, sx_ * 0.245, 0.02, 0.1, 0.08)                 # 颧骨
    d += 0.022 * _bump(x, y, 0.0, -0.283, 0.14, 0.03)                         # 上唇
    d += 0.030 * _bump(x, y, 0.0, -0.337, 0.12, 0.034)                        # 下唇
    d -= 0.014 * _bump(x, y, 0.0, -0.309, 0.15, 0.012)                        # 口缝
    d += 0.028 * _bump(x, y, 0.0, -0.425, 0.075)                              # 下巴
    return d


def _seg_pts(x, y, x0, y0, x1, y1):
    """把点折算到线段上最近点（用于沿鼻梁的长条 bump）。"""
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy
    t = np.clip(((x - x0) * vx + (y - y0) * vy) / L2, 0, 1)
    return x - (x0 + t * vx), y - (y0 + t * vy)


def taper(y):
    """下颌收窄：y < -0.15 起 x 收 22%。"""
    t = np.clip((-y - 0.15) / 0.35, 0, 1)
    return 1 - 0.22 * t * t * (3 - 2 * t)


def surface_point(d, eps_fallback=1e-9):
    """方向 d (N,3) → 头面点。位移在有限差分里保持一致，法线自动带特征。"""
    r = 1.0 / np.sqrt((d[:, 0] / A) ** 2 + (d[:, 1] / B) ** 2 + (d[:, 2] / C) ** 2 + eps_fallback)
    p = d * r[:, None]
    xt = p[:, 0] * taper(p[:, 1])
    front = np.clip((p[:, 2] - 0.0) / 0.15, 0, 1)
    front = front * front * (3 - 2 * front)
    dv = displacement(xt, p[:, 1]) * front
    return p * (1 + dv / r)[:, None]


def normals(d):
    """有限差分法线（与位移场一致的表面真实法线）。"""
    t1 = np.cross(d, [0.0, 1.0, 0.0])
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-9
    t2 = np.cross(d, t1)
    t2 /= np.linalg.norm(t2, axis=1, keepdims=True) + 1e-9
    h = 2e-3
    p0 = surface_point(d)
    p1 = surface_point(_norm(d + t1 * h))
    p2 = surface_point(_norm(d + t2 * h))
    n = np.cross(p1 - p0, p2 - p0)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    flip = (n * d).sum(1) < 0
    n[flip] *= -1
    return n


def _norm(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------- 颜色场 ----------------

def colors_of(xt, y, z, n, rng):
    base = np.tile(np.array([0.925, 0.785, 0.695]), (len(xt), 1))
    base += 0.025 * np.stack([rng.normal(0, 1, len(xt)), ] * 3, 1) * np.array([1.0, 0.7, 0.55])
    base += 0.03 * _bump(xt, y, 0.245, -0.02, 0.12)[..., None] * np.array([0.10, 0.02, 0.01])
    base += 0.03 * _bump(xt, y, -0.245, -0.02, 0.12)[..., None] * np.array([0.10, 0.02, 0.01])
    base *= 1 - 0.10 * _bump(xt, y, 0.0, 0.36, 0.22)[..., None]            # 发际线过渡

    face_oval = ((xt / 0.30) ** 2 + ((y + 0.05) / 0.40) ** 2 < 1) & (z > 0.02)
    hair = ~face_oval
    col = base
    glabella = np.exp(-0.5 * (xt / 0.055) ** 2)                     # 眉心留白，防连眉
    brow_l = (_bump(xt, y, 0.17, 0.272, 0.082, 0.022)
              + _bump(xt, y, -0.17, 0.272, 0.082, 0.022)) * (1 - glabella)
    col = np.where((brow_l > 0.5)[..., None],
                   np.array([0.23, 0.16, 0.12]), col)
    lip = _bump(xt, y, 0.0, -0.283, 0.135, 0.030) + _bump(xt, y, 0.0, -0.337, 0.115, 0.033)
    lip = np.clip(lip, 0, 1)
    col = col * (1 - (lip * 0.85 * face_oval)[..., None]) + \
        (lip * 0.85 * face_oval)[..., None] * np.array([0.80, 0.52, 0.50])
    mouth = _bump(xt, y, 0.0, -0.309, 0.15, 0.011)
    col = np.where((mouth > 0.5)[..., None], np.array([0.45, 0.26, 0.26]), col)
    # 闭眼样式：睫 毛线 + 上睑晕影（妆前素颜感，眼影/眼线妆效留给 tint 叠加）
    lash = (_bump(xt, y, 0.17, 0.150, 0.052, 0.0075)
            + _bump(xt, y, -0.17, 0.150, 0.052, 0.0075)) > 0.5
    col = np.where(lash[..., None], np.array([0.24, 0.17, 0.14]), col)
    crease = (_bump(xt, y, 0.17, 0.185, 0.055, 0.016)
              + _bump(xt, y, -0.17, 0.185, 0.055, 0.016)) * 0.35
    col *= 1 - (crease * face_oval)[..., None] * 0.5
    col = np.where(hair[..., None],
                   np.array([0.15, 0.10, 0.08]) * (1 + 0.3 * rng.normal(0, 1, (len(xt), 1))),
                   col)
    return np.clip(col, 0, 1)


def quat_from_normal(n):
    """旋转 (0,0,1)→n 的四元数 (w,x,y,z)。"""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n)
    s = np.linalg.norm(v, axis=1)
    a = np.arctan2(s, n[:, 2])
    axis = v / (s[:, None] + 1e-12)
    half = a / 2
    return np.stack([np.cos(half), axis[:, 0] * np.sin(half),
                     axis[:, 1] * np.sin(half), axis[:, 2] * np.sin(half)], 1)


def surf_z(x, y):
    """近似表面 z（锚点用）：椭球 z + 位移。"""
    z = C * np.sqrt(np.clip(1 - (np.abs(x) / A) ** 2 - (y / B) ** 2, 1e-6, None))
    return z + displacement(np.asarray(x, float) * taper(np.asarray(y, float)), y)


def anchors_json():
    """与雕刻特征精确对齐的语义锚点（手动模式地面真值）。"""
    def pt(x, y, lift=0.006):
        return [round(float(x), 4), round(float(y), 4), round(float(surf_z(x, y)) + lift, 4)]

    out = {
        "lipstick": [pt(0, -0.283), pt(0, -0.337), pt(0.085, -0.29), pt(-0.085, -0.29),
                     pt(0.075, -0.328), pt(-0.075, -0.328), pt(0.04, -0.309), pt(-0.04, -0.309),
                     pt(0.115, -0.30), pt(-0.115, -0.30)],
        "eyeshadow": [pt(0.165, 0.20), pt(-0.165, 0.20), pt(0.125, 0.19), pt(-0.125, 0.19)],
        "eyeliner": [pt(0.17, 0.185), pt(-0.17, 0.185), pt(0.125, 0.175), pt(-0.125, 0.175)],
        "eyebrow": [pt(0.17, 0.278), pt(-0.17, 0.278), pt(0.10, 0.288), pt(-0.10, 0.288),
                    pt(0.245, 0.262), pt(-0.245, 0.262)],
        "blush": [pt(0.25, -0.06), pt(-0.25, -0.06), pt(0.19, -0.12), pt(-0.19, -0.12)],
        "foundation": [pt(0, 0.05), pt(0, 0.28), pt(0, -0.20), pt(0.20, 0.14), pt(-0.20, 0.14),
                       pt(0.24, -0.18), pt(-0.24, -0.18), pt(0.12, 0.33), pt(-0.12, 0.33),
                       pt(0.09, -0.40), pt(-0.09, -0.40)],
        "highlight": [pt(0, 0.06), pt(0, -0.02), pt(0, -0.26), pt(0.235, 0.02), pt(-0.235, 0.02)],
        "contour": [pt(0, 0.38), pt(0.28, -0.33), pt(-0.28, -0.33), pt(0.062, -0.10)],
        "concealer": [pt(0.17, 0.125), pt(-0.17, 0.125)],
    }
    return out


def build(n_points: int = 240000, seed: int = 3) -> tuple[AvatarData, dict]:
    rng = np.random.default_rng(seed)
    i = np.arange(n_points)
    golden = (1 + 5 ** 0.5) / 2
    theta = np.arccos(1 - 2 * (i + 0.5) / n_points)
    phi = 2 * np.pi * golden * i
    d = _norm(np.stack([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)], 1))

    pos = surface_point(d)
    nrm = normals(d)
    xt, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    col = colors_of(xt, y, z, nrm, rng)
    q = quat_from_normal(nrm)
    # σ 取点距的 ~1.6 倍保证相邻高斯重叠（点距 ≈ sqrt(4π·r̄²/N) ≈ 0.003），
    # 这是消摩尔纹的关键：真实 3DGS 的高斯彼此重叠，不是孤立圆点
    sig_t = rng.uniform(0.0038, 0.0060, (n_points, 1))
    scales = np.concatenate([sig_t, sig_t, np.full((n_points, 1), 0.0013)], 1).astype(np.float32)
    op = np.clip(0.92 + rng.normal(0, 0.03, n_points), 0.6, 1.0).astype(np.float32)

    av = AvatarData(pos.astype(np.float32), scales.astype(np.float32),
                    q.astype(np.float32), col.astype(np.float32), op, face_height=1.0)
    return av, anchors_json()


def main():
    ap = argparse.ArgumentParser(description="生成类真人 3DGS 头像（演示/联调）")
    ap.add_argument("--out", default="out/demo/avatar-face")
    ap.add_argument("--points", type=int, default=240000)
    args = ap.parse_args()
    av, anchors = build(args.points)
    out = Path(args.out)
    save_avatar(av, out / "avatar.ply")
    (out / "anchors.json").write_text(json.dumps(anchors, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[sculpt] 类真人头像已生成：{av.n} 高斯 → {out/'avatar.ply'}（anchors.json 同目录）")


if __name__ == "__main__":
    main()
