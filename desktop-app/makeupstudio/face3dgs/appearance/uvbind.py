"""uvbind — 把 canonical UV 与区域覆盖度绑到任意脸部点云（妆容 UV 化的前提）。

与 fit_makeup._splat_uv 同一归属算法（kNN 距离加权 + 跨 UV 岛保护 + 离群剔除），
但从"上色的临时查询"升级为"一等公民属性"：训练产物每个 splat 携带 uv 与
区域覆盖（region cov），妆容从此在 2048² UV 空间合成，再按 uv 采样回 splat——
锐度由贴图分辨率决定，不再受 splat 密度限制。

唇部例外：唇红带是 3D 拓扑概念（张嘴口腔面排除），由 makeup_uv 直接把
唇带三角光栅化进 UV，本模块只负责把"离群/表面"判定给足。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fit_makeup import FaceMakeupFitter, N_CANON_VERTS

# 妆容区域 → RegionMasks 组名（与 landmark-regions.json 一致）
REGION_GROUPS = ("foundation", "concealer", "contour", "blush", "highlight",
                 "eyeshadow", "eyebrow", "eyeliner", "lashes", "lipstick")


@dataclass
class UvBinding:
    uv: np.ndarray                    # (n,2) splat 的 canonical UV
    valid: np.ndarray                 # (n,) bool 在 canonical 表面上（离群=头发/背景永不涂妆）
    near: np.ndarray                  # (n,) 到表面距离（配准尺度的噪声度量）
    cov: dict[str, np.ndarray]        # region → (tex,tex) float32 覆盖 0..1（UV 空间）
    cent: dict[str, np.ndarray]       # region → (tex,tex) 渐变坐标 0..1
    tex: int

    def sample(self, region: str, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """splat UV → (coverage, centroid) 双线性采样。

        注意：_bilinear 内部按全局 TEX 钳制坐标，必须先切到绑定分辨率。"""
        from ..fit_makeup import _load_core
        core = _load_core()
        t = self.tex
        prev = core.TEX
        core.set_texture_size(t)
        try:
            tx = np.clip(uv[:, 0] * (t - 1), 0, t - 1.001)
            ty = np.clip((1 - uv[:, 1]) * (t - 1), 0, t - 1.001)
            c = core._bilinear(self.cov[region][..., None], tx, ty)[..., 0]
            e = core._bilinear(self.cent[region][..., None], tx, ty)[..., 0]
        finally:
            core.set_texture_size(prev)
        return c, e


def bind_uv(cloud: dict[str, np.ndarray], landmarks: np.ndarray,
            tex: int = 2048) -> UvBinding:
    """canonical UV + 区域覆盖绑定。landmarks 为 (468,3) 三角化地标
    （canonical 姿态点云可直接传 canonical_face_model 顶点）。"""
    fitter = FaceMakeupFitter(tracker_factory=lambda: (_ for _ in ()).throw(
        RuntimeError("bind_uv 不需要 tracker")))
    s, R, t, _ = fitter.register(landmarks)
    if "uv" in cloud:
        uv = np.asarray(cloud["uv"], np.float64)[:, :2]
        valid = np.isfinite(uv).all(1)
        near = np.zeros(len(uv), np.float32)
    else:
        uv, near, valid = fitter._splat_uv(cloud, s, R, t)

    prev = fitter.core.TEX
    fitter.core.set_texture_size(tex)
    try:
        cov: dict[str, np.ndarray] = {}
        cent: dict[str, np.ndarray] = {}
        for region in REGION_GROUPS:
            layer = {"region": region, "shape": {}, "color_stops": [
                {"at": 0.0, "hex": "#000000"}, {"at": 1.0, "hex": "#000000"}]}
            m = fitter.regions.bake(layer)
            cov[region] = m[..., 0].astype(np.float32)
            cent[region] = m[..., 1].astype(np.float32)
    finally:
        fitter.core.set_texture_size(prev)
    return UvBinding(uv=uv, valid=valid, near=near.astype(np.float32),
                     cov=cov, cent=cent, tex=tex)


def canonical_landmarks() -> np.ndarray:
    """canonical 姿态点云（无视频三角化时）用的 468 地标 = canonical 顶点。"""
    fitter = FaceMakeupFitter(tracker_factory=lambda: (_ for _ in ()).throw(
        RuntimeError("unused")))
    return fitter.model.base[:N_CANON_VERTS]
