"""normals — 3DGS splat 法线估计（PBR 着色质量的关键输入）。

旧实现 `_quat_normals` 假设每个 splat 的薄轴都是旋转后的 Z——对 scale_z 不是
最小轴的 splat 法线直接错 90°，高光花掉。本模块两级修正：
    axis_normals    取 min(scale) 对应的旋转矩阵列（薄轴必然垂直于薄片），
                    一行级修复，向量化零成本；
    smooth_normals  SuGaR 式 kNN 邻域平均 + 自投影正交化——逐 splat 法线噪声
                    大（训练自由生长的薄片朝向互相打架），邻域平均后 PBR 高光
                    才连续。FLANN kNN 与 train_base 同款实现。
"""
from __future__ import annotations

import numpy as np


def _quats_to_rotmats(rot: np.ndarray) -> np.ndarray:
    """四元数 (n,4) xyzw → 旋转矩阵 (n,3,3)（列 = 局部基向量旋到世界）。"""
    q = np.asarray(rot, np.float64)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def axis_normals(rot: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """薄片法线 = min(scale) 轴的旋转列。(n,4)/(n,3) → (n,3) 世界系单位向量。"""
    R = _quats_to_rotmats(rot)
    axis = np.argmin(np.asarray(scale, np.float64), axis=1)
    # Rᵀ 的第 axis 行 = R 的第 axis 列
    n = np.take_along_axis(R.transpose(0, 2, 1), axis[:, None, None], axis=1)[:, 0, :]
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n


def smooth_normals(xyz: np.ndarray, normals: np.ndarray, k: int = 12,
                   iters: int = 2) -> np.ndarray:
    """kNN 邻域平均法线（SuGaR 式）：平均后投影回各自切平面再归一。

    孤立点（近邻均距离过大）保留原法线，避免飞点把噪声扩散出去。"""
    import cv2

    xyz = np.ascontiguousarray(xyz, np.float32)
    n = np.asarray(normals, np.float64).copy()
    fl = cv2.flann_Index(xyz, dict(algorithm=1, trees=4, checks=64))
    _idx, d2 = fl.knnSearch(xyz, k + 1, params=dict(checks=64))
    nn_mean = np.sqrt(np.maximum(d2[:, 1:].mean(1), 1e-12))
    iso = nn_mean > max(float(np.median(nn_mean)) * 6.0, 1e-6)   # 孤立飞点
    idx = _idx[:, 1:]                                            # 去自身
    for _ in range(iters):
        nb = n[idx]                                              # (n,k,3)
        avg = nb.mean(1)
        avg /= np.linalg.norm(avg, axis=1, keepdims=True) + 1e-12
        # 切向修正：邻域平均里垂直于自身法线的分量，加到自身上（保留原朝向，
        # 只吸收邻域一致性；直接把 avg 投影会翻转切向噪声符号）
        dot = (avg * n).sum(1, keepdims=True)
        n = n + (avg - dot * n)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        n[iso] = np.asarray(normals, np.float64)[iso]
    return n
