"""bench — 客观指标（杜绝"看起来还是差"的玄学迭代）。

    psnr            留出帧脸区光度误差（底模质量）
    lip_delta_e     唇区素颜→妆后的 Lab 色差（妆感强度，CIE76 距离）
    identity_shift  非妆区 Lab 色差（应≈0：底妆只匀肤不改肤色）

OpenCV 可算，无需 GPU/权重。主观 AIME 式打分（VLM）后续接入。
"""
from __future__ import annotations

import numpy as np


def lab_mean(img_rgb01: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """(H,W,3) 0..1 RGB → Lab 均值（可选 mask）。"""
    img = np.clip(img_rgb01, 0, 1)
    if mask is not None:
        img = img[mask > 0.5]
    if img.size == 0:
        return np.zeros(3, np.float64)
    import cv2
    lab = cv2.cvtColor((img[None] * 255).astype(np.uint8), cv2.COLOR_RGB2Lab)[0]
    return lab.reshape(-1, 3).mean(0).astype(np.float64)


def delta_e(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """CIE76 色差（0≈无差别，2≈可辨，5+≈明显）。"""
    return float(np.linalg.norm(np.asarray(lab_a) - np.asarray(lab_b)))


def region_delta_e(bare_rgb: np.ndarray, made_rgb: np.ndarray,
                   region_mask: np.ndarray) -> float:
    """同尺寸图像在给定区域的素颜/妆后 Lab 色差。"""
    lab_b = lab_mean(bare_rgb, region_mask)
    lab_m = lab_mean(made_rgb, region_mask)
    return delta_e(lab_b, lab_m)


def psnr_masked(gt_rgb01: np.ndarray, render_rgb01: np.ndarray,
                mask: np.ndarray) -> float:
    diff = (gt_rgb01 - render_rgb01) ** 2
    sel = mask > 0.5
    if not sel.any():
        return 0.0
    mse = float(diff[sel].mean())
    return float(-10 * np.log10(max(mse, 1e-12)))
