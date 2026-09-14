"""pose_ref — 头部姿态估计用的 canonical 参考点位。

canonical FaceMesh 拓扑关键点：鼻尖 1、下巴 152、左右眼外角 33/263、左右嘴角 61/291。
CANONICAL_XY 为这些点在 canonical 模型中的平面坐标（y 翻转到图像方向，脸高归一），
运行时从 landmark-regions.json 同目录的 canonical_face_model.obj 解析。
"""
from __future__ import annotations

import numpy as np

POSE_POINTS = [1, 152, 33, 263, 61, 291]

_OBJ = None


def _load_obj_verts(obj_path: str) -> np.ndarray:
    global _OBJ
    if _OBJ is None:
        verts = []
        for line in open(obj_path, encoding="utf-8"):
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append((float(x), float(y), float(z)))
        V = np.asarray(verts, np.float64)
        if V[10, 1] < V[152, 1]:
            V[:, 1] *= -1
        _OBJ = V
    return _OBJ


def _canonical_xy(obj_path: str) -> np.ndarray:
    V = _load_obj_verts(obj_path)
    lo, hi = V.min(0), V.max(0)
    Vn = (V - (lo + hi) / 2) / (hi[1] - lo[1])
    pts = Vn[POSE_POINTS, :2].copy()
    pts[:, 1] *= -1            # 模型 y 向上 → 图像 y 向下
    return pts


import os  # noqa: E402
_CANONICAL_OBJ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "..", "makeup-skill", "references", "canonical_face_model.obj")
CANONICAL_XY = _canonical_xy(_CANONICAL_OBJ)
