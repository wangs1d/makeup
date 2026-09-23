"""refshape — 参考妆照的**形状级**还原（零依赖 guidance 替代路径）。

颜色/浓度还原由 calibrate.calibrate_spec 完成（参考图逐区域 Lab 统计 → spec
色带/opacity），但形状（眼影晕染范围、眼线翼形、腮红位置）此前只能来自
canonical 模板——除非搭起 Stable-Makeup guidance 全家桶（独立 conda 环境 +
diffusion 权重）。本模块补上中间档：参考图与用户脸都有 468 地标，一个鲁棒
仿射（RANSAC）就能把参考图的图像空间区域蒙版搬到用户的参考视角帧上，再经
相机投影采样到每个 splat——"这个妆画成什么形状"第一次不依赖模板。

语义与 landmark_band_3d 一致：形状权重只改"涂在哪"，颜色仍取 UV 目标场。
消费者：pipeline.apply_makeup_to_asset 的 shape3d 参数（与 bands3d 观测锚定
的差别：bands3d 只做并集增强，shape3d 会把模板权重压到 *suppression*——
参考图明确没画的地方，模板不该替参考妆做主）。
"""
from __future__ import annotations

import numpy as np

# 可做形状迁移的区域（foundation 全脸无形状；lipstick 由用户自身唇拓扑 3D
# 锚定——把参考唇形 overline 到别人唇上风险大于收益）
SHAPE_REGIONS = ("eyeshadow", "eyeliner", "eyebrow", "blush")
MIN_MATCH = 30          # 仿射估计的最少有效对应点


def affine_between(ref_px: np.ndarray, user_px: np.ndarray) -> np.ndarray | None:
    """参考图地标 → 用户帧地标的鲁棒部分仿射（相似变换 + RANSAC）。

    两张脸的形状差本身就是"身份差"，相似变换吸收的只是尺度/旋转/平移——
    剩余的形变由 RANSAC 内点上的最小二乘兜住。对应点不足或拟合失败返回
    None（调用方回退模板形状）。"""
    import cv2
    ref = np.asarray(ref_px, np.float32)
    usr = np.asarray(user_px, np.float32)
    n = min(len(ref), len(usr), 468)
    ref, usr = ref[:n], usr[:n]
    ok = (np.linalg.norm(ref, axis=1) > 0) & (np.linalg.norm(usr, axis=1) > 0)
    if int(ok.sum()) < MIN_MATCH:
        return None
    M, inliers = cv2.estimateAffinePartial2D(
        ref[ok], usr[ok], method=cv2.RANSAC, ransacReprojThreshold=6.0,
        maxIters=5000, refineIters=20)
    if M is None or inliers is None or int(inliers.sum()) < MIN_MATCH // 2:
        return None
    return np.asarray(M, np.float64)


def warp_masks(masks: dict[str, np.ndarray], M: np.ndarray, out_hw: tuple[int, int]
               ) -> dict[str, np.ndarray]:
    """参考图区域蒙版经仿射 M 搬到用户帧坐标系。"""
    import cv2
    out: dict[str, np.ndarray] = {}
    for region, m in masks.items():
        w = cv2.warpAffine(np.asarray(m, np.float32), M,
                           (out_hw[1], out_hw[0]), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[region] = np.clip(w, 0, 1)
    return out


def project_splats(xyz: np.ndarray, w2c: np.ndarray,
                   K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """世界系 splat → 像素坐标 (n,2) 与帧内可见 (n,) bool。"""
    homo = np.concatenate([np.asarray(xyz, np.float64),
                           np.ones((len(xyz), 1))], axis=1)
    cam = (np.asarray(w2c, np.float64)[None, :3, :] @ homo[:, :, None])[..., 0]
    z = np.maximum(cam[:, 2], 1e-6)
    px = np.empty((len(xyz), 2), np.float64)
    px[:, 0] = K[0, 0] * cam[:, 0] / z + K[0, 2]
    px[:, 1] = K[1, 1] * cam[:, 1] / z + K[1, 2]
    ok = (cam[:, 2] > 0.05) & np.isfinite(px).all(1)
    return px, ok


def sample_mask(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """逐点双线性采样蒙版（越界=0）。"""
    h, w = mask.shape[:2]
    x = np.clip(np.asarray(xy[:, 0], np.float64), 0, w - 1.001)
    y = np.clip(np.asarray(xy[:, 1], np.float64), 0, h - 1.001)
    x0, y0 = x.astype(np.int32), y.astype(np.int32)
    fx, fy = (x - x0), (y - y0)
    m = np.asarray(mask, np.float32)
    return (m[y0, x0] * (1 - fx) * (1 - fy)
            + m[y0, x0 + 1] * fx * (1 - fy)
            + m[y0 + 1, x0] * (1 - fx) * fy
            + m[y0 + 1, x0 + 1] * fx * fy)


def reference_shape_bands(ref_masks: dict[str, np.ndarray], M: np.ndarray,
                          xyz: np.ndarray, w2c: np.ndarray, K: np.ndarray,
                          frame_hw: tuple[int, int],
                          regions: tuple[str, ...] = SHAPE_REGIONS,
                          ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """参考图区域蒙版 → 用户帧坐标系 → 逐 splat 形状权重 (w, cent)。

    cent 返回零（消费者对 bands/shape 权重的渐变坐标一律取 UV 场——参考
    形状只回答"涂不涂这里"，不回答"渐变方向"）。区域蒙版缺失或 splat 全部
    不可见时该区域不进返回值（调用方回退模板）。"""
    warped = warp_masks({r: m for r, m in ref_masks.items() if r in regions},
                        M, frame_hw)
    px, ok = project_splats(xyz, w2c, K)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for region, m in warped.items():
        w = np.zeros(len(xyz), np.float32)
        w[ok] = sample_mask(m, px[ok])
        if float((w > 0.05).sum()) < 20:     # 搬运后几乎空蒙版：形变失败，放弃
            continue
        out[region] = (w, np.zeros(len(xyz), np.float32))
    return out
