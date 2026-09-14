#!/usr/bin/env python3
"""tracker — 人脸追踪（MediaPipe Tasks FaceLandmarker）。

返回 478 点像素坐标（0-467 与 canonical FaceMesh / landmark-regions.json 同拓扑，
468-477 为虹膜）+ solvePnP 6DoF 头部姿态。关键点经 1€ 滤波平滑。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


class OneEuro:
    """1€ 滤波（低延迟自适应平滑），与 unity-app/OneEuroFilter.cs 同参数语义。"""

    def __init__(self, min_cutoff: float = 1.7, beta: float = 0.3, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = max(t - self.t_prev, 1e-3)
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = self.dx_prev + a_d * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(float(np.mean(cutoff)) if cutoff.ndim else float(cutoff), dt)
        x_hat = self.x_prev + a * (x - self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class FaceTracker:
    """视频流模式的关键点追踪器。detect() 输入 BGR 帧与单调递增的毫秒时间戳。"""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, num_faces: int = 1,
                 smooth: bool = True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base = mp_python.BaseOptions(model_asset_path=str(model_path))
        opts = vision.FaceLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.VIDEO, num_faces=num_faces)
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._mp = mp
        self.smooth = smooth
        self._filters: list[OneEuro] | None = None
        self._obj_points_cache: dict[int, np.ndarray] = {}

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: float) -> dict | None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect_for_video(mp_img, int(timestamp_ms))
        if not res.face_landmarks:
            self._filters = None
            return None
        lm = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], np.float32)
        px = np.stack([lm[:, 0] * w, lm[:, 1] * h], axis=1)
        if self.smooth:
            if self._filters is None or len(self._filters) != len(px):
                self._filters = [OneEuro() for _ in px]
            t_s = timestamp_ms / 1000.0
            px = np.stack([f(px[i], t_s) for i, f in enumerate(self._filters)])
        return {
            "px": px,                     # (478,2) 像素坐标
            "norm": lm,                   # (478,3) 归一化坐标
            "pose": self._pose(lm[:, :2], w, h),
        }

    def _pose(self, xy_norm: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
        """solvePnP 估计头部姿态（相对正脸的 yaw/pitch/roll，度）。"""
        from .pose_ref import POSE_POINTS, CANONICAL_XY

        idx = [i for i in POSE_POINTS if i < len(xy_norm)]
        pts = np.ascontiguousarray(
            np.stack([xy_norm[idx, 0] * w, xy_norm[idx, 1] * h], 1), dtype=np.float64)
        # OpenCV 5 要求 objectPoints 为 (N,3)；canonical 平面点补 z=0
        obj3 = np.ascontiguousarray(
            np.concatenate([np.asarray(CANONICAL_XY)[:len(idx)],
                            np.zeros((len(idx), 1))], axis=1)
            * np.array([w, h, 1.0], np.float64), dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(obj3, pts, np.array([[h * 1.85, 0, h / 2], [0, h * 1.85, 0], [0, 0, 1]], np.float64),
                                   None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return (0.0, 0.0, 0.0)
        rmat, _ = cv2.Rodrigues(rvec)
        pitch = np.degrees(np.arcsin(np.clip(-rmat[2, 1], -1, 1)))
        yaw = np.degrees(np.arctan2(rmat[2, 0], rmat[2, 2]))
        roll = np.degrees(np.arctan2(rmat[0, 1], rmat[1, 1]))
        return (float(yaw), float(pitch), float(roll))

    def close(self):
        self._landmarker.close()
