"""isolate — 从全场景重建结果中裁出"用户脸部"点云。

Brush 产出的是整个画面场景（脸 + 背景 + 肩颈）。裁剪依据：
视频各帧的人脸框（MediaPipe 检测，可注入）通过 COLMAP 相机投影到每个视角，
一个 splat 在 ≥ 多数含脸视角内落在人脸框里才保留（重投影投票，单一视角误检
不会误留/误删）。再剔除尺度异常的"漂浮物"大高斯。
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from . import colmap_io
from .reconstruct import ReconResult

# 人脸框外扩系数（覆盖发型/耳朵边缘；配合投票阈值防误留背景）
BBOX_MARGIN = 1.45
MIN_VOTES_FRACTION = 0.30   # splat 至少在 30% 含脸视角内（且 ≥2 视角）
SCALE_OUTLIER_K = 6.0       # σ 超过全云中位数 K 倍视为漂浮高斯

# 默认人脸框提供方：MediaPipe FaceDetection（无状态，适合任意帧子集）
BboxProvider = Callable[[np.ndarray], tuple[float, float, float, float] | None]
# 返回归一化 (x1, y1, x2, y2)；None = 该帧无脸


def mediapipe_bbox_provider(min_confidence: float = 0.5) -> BboxProvider:
    """MediaPipe Tasks FaceLandmarker（IMAGE 模式）→ 归一化人脸框。

    注：mediapipe>=1.0 只有 Tasks API，旧 solutions.face_detection 不可用。
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    import mediapipe as mp
    model_path = Path(__file__).resolve().parent.parent.parent / "models" / "face_landmarker.task"
    base = mp_python.BaseOptions(model_asset_path=str(model_path))
    opts = vision.FaceLandmarkerOptions(base_options=base,
                                        running_mode=vision.RunningMode.IMAGE,
                                        min_face_detection_confidence=min_confidence)
    landmarker = vision.FaceLandmarker.create_from_options(opts)

    def detect(frame_bgr: np.ndarray):
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = landmarker.detect(img)
        if not res.face_landmarks:
            return None
        xs = np.array([p.x for p in res.face_landmarks[0]])
        ys = np.array([p.y for p in res.face_landmarks[0]])
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    return detect


# 脸部区域蒙版提供方：返回 (H,W) float 蒙版（1=脸），None=该帧不可用
FaceMaskProvider = Callable[[np.ndarray], np.ndarray | None]


def parsing_mask_provider() -> FaceMaskProvider:
    """人脸解析（SegFormer）→ 像素级脸部蒙版。

    比 bbox 精确得多：头发/肩膀/衣服天然为 0，重投影投票不再被
    "落在人脸框里但其实在头发/背景上"的 splat 污染。解析不可用时返回 None
    （调用方回退 bbox）。"""
    from ..parser import FaceParser
    parser = FaceParser(device="cpu")

    def face_mask(frame_bgr: np.ndarray) -> np.ndarray | None:
        if not parser.available():
            return None
        masks = parser.parse(frame_bgr)
        if not masks:
            return None
        face = None
        for g in ("skin", "lips", "mouth", "eye_l", "eye_r", "brow_l", "brow_r"):
            if g in masks:
                face = masks[g] if face is None else np.maximum(face, masks[g])
        return face

    return face_mask


def isolate_face(result: ReconResult, out_ply: str | Path,
                 bbox_provider: BboxProvider | None = None,
                 mask_provider: FaceMaskProvider | None = None,
                 max_views: int = 60) -> dict:
    """重投影投票裁剪 → face.ply。返回统计信息。

    mask_provider（人脸解析蒙版）优先；为 None/不可用时回退 bbox 投票。"""
    if bbox_provider is None and mask_provider is None:
        try:
            mask_provider = parsing_mask_provider()
        except Exception:
            mask_provider = None
        if mask_provider is None:
            bbox_provider = mediapipe_bbox_provider()
    from .splat_io import read_ply, write_ply
    cloud = read_ply(result.ply_path)

    model = colmap_io.read_sparse(result.sparse_dir)
    cam = model.camera
    names = sorted(model.images)
    if max_views and len(names) > max_views:   # 均匀抽视角控制耗时
        idx = np.linspace(0, len(names) - 1, max_views).astype(int)
        names = [names[i] for i in idx]

    xyz = cloud["xyz"].astype(np.float64)
    votes = np.zeros(len(xyz), np.int32)
    views_with_face = 0
    for name in names:
        frame = result.images_dir / name
        if not frame.exists():
            continue
        img = cv2.imread(str(frame))
        if img is None:
            continue
        h, w = img.shape[:2]
        im = model.images[name]
        if im["cam_id"] != cam.cam_id:
            continue
        px = cam.project(xyz, im["qvec"], im["tvec"])
        z = (colmap_io.quat_to_rotmat(im["qvec"]) @ xyz.T).T[:, 2] + im["tvec"][2]
        in_front = z > 0.05
        if mask_provider is not None:
            m = mask_provider(img)
            if m is not None:
                views_with_face += 1
                # 蒙版采样：投影点落在解析出的"脸部像素"内才投票
                xi = np.clip(px[:, 0].round().astype(np.int64), 0, m.shape[1] - 1)
                yi = np.clip(px[:, 1].round().astype(np.int64), 0, m.shape[0] - 1)
                votes += (in_front & (m[yi, xi] > 0.3)).astype(np.int32)
                continue
        box = bbox_provider(img) if bbox_provider is not None else None
        if box is None:
            continue
        views_with_face += 1
        x1, y1, x2, y2 = box
        bw, bh = (x2 - x1) * BBOX_MARGIN, (y2 - y1) * BBOX_MARGIN
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1m, x2m = (cx - bw / 2) * w, (cx + bw / 2) * w
        y1m, y2m = (cy - bh / 2) * h, (cy + bh / 2) * h
        inside = in_front & (px[:, 0] >= x1m) & (px[:, 0] <= x2m) \
            & (px[:, 1] >= y1m) & (px[:, 1] <= y2m)
        votes += inside.astype(np.int32)

    if views_with_face == 0:
        raise RuntimeError("没有任何帧检测到人脸，无法隔离脸部点云")
    need = max(2, int(MIN_VOTES_FRACTION * views_with_face))
    keep = votes >= need
    # 漂浮高斯：尺度远超中位数的通常是背景/噪点
    s_med = float(np.median(np.prod(cloud["scale"], axis=1) ** (1 / 3)))
    keep &= (np.prod(cloud["scale"], axis=1) ** (1 / 3)) <= SCALE_OUTLIER_K * s_med

    face = {k: v[keep] for k, v in cloud.items()}
    write_ply(face, out_ply)
    return {"splats_total": int(len(xyz)), "splats_face": int(keep.sum()),
            "views_used": len(names), "views_with_face": views_with_face,
            "out": str(out_ply)}
