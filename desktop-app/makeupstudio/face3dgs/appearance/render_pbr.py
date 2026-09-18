"""render_pbr — 带化妆品材质的前向泼溅预览（fit_makeup.render_cloud 的 PBR 升级版）。

区别：逐点颜色先经 pbr.shade_points 着色（rough/coat/sss/sheen 材质通道），
再走同款圆核后→前合成。旧 blinn_phong_spec（全局 gloss 标量）保留兼容但
不再被本链路使用。
"""
from __future__ import annotations

import numpy as np

from .. import colmap_io
from ..fit_makeup import _load_core
from .normals import axis_normals
from .pbr import Material, shade_points


def _mat_from_cloud(cloud: dict[str, np.ndarray]) -> Material:
    mat = cloud.get("material")
    if isinstance(mat, Material):
        return mat
    n = len(cloud["xyz"])
    gloss = np.asarray(cloud.get("gloss", np.zeros(n)), np.float64)
    shin = np.asarray(cloud.get("shin", np.full(n, 64.0)), np.float64)
    # 旧字段反推：gloss→coat，shin→rough（经验映射，仅兼容展示）
    rough = np.clip((150.0 / np.maximum(shin, 8.0)) ** 0.5, 0.05, 1.0)
    return Material(rough.astype(np.float32), gloss.astype(np.float32),
                    np.zeros(n, np.float32), np.zeros(n, np.float32))


def render_cloud_pbr(cloud: dict[str, np.ndarray], R: np.ndarray, t: np.ndarray,
                     cam: colmap_io.Camera, w: int = 512, h: int = 384,
                     env: str = "neutral", light_dir: np.ndarray | None = None,
                     spec_strength: float = 1.0,
                     ambient_floor: float = 0.30) -> np.ndarray:
    core = _load_core()
    light = core.ENVS.get(env, core.ENVS["neutral"])
    Lw = (np.asarray(light_dir, np.float64) if light_dir is not None
          else np.asarray(light["light"], np.float64))
    Lw = Lw / (np.linalg.norm(Lw) + 1e-12)
    Lc = R @ Lw
    tint = np.asarray(light["tint"], np.float64)

    xyz = (R @ np.asarray(cloud["xyz"], np.float64).T).T + t
    z = xyz[:, 2]
    scale = min(w / cam.width, h / cam.height)
    f = float(cam.params[0]) * scale
    cx = float(cam.params[1]) * scale
    cy = float(cam.params[2]) * scale
    px = cx + xyz[:, 0] * f / np.maximum(z, 1e-6)
    py = cy + xyz[:, 1] * f / np.maximum(z, 1e-6)
    rgba = np.asarray(cloud["rgba"], np.float64)
    scale3 = np.asarray(cloud["scale"], np.float64)

    mat = _mat_from_cloud(cloud)
    Vc = np.array([0.0, 0.0, -1.0])                  # 指向相机（COLMAP 相机看 +z）
    normals = axis_normals(np.asarray(cloud["rot"], np.float64),
                           np.asarray(cloud["scale"], np.float64))
    shade = shade_points(rgba, normals @ R.T,
                         Vc, Lc, tint, mat,
                         spec_strength=spec_strength, ambient_floor=ambient_floor)

    order = np.argsort(-z)
    canvas = np.full((h, w, 3), 0.09, np.float64)
    for i in order:
        a = rgba[i, 3]
        if a < 0.02 or z[i] <= 0.05:
            continue
        sig = float(scale3[i, 0]) * f / z[i]
        if sig > w * 0.4:
            continue
        sig = max(sig, 0.5)                      # 小于半像素的高斯按 1px 点渲染，不丢弃
        cxx, cyy = px[i], py[i]
        half = int(sig * 3) + 1
        x0, x1 = int(cxx) - half, int(cxx) + half + 1
        y0, y1 = int(cyy) - half, int(cyy) + half + 1
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            continue
        rows, cols = np.mgrid[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        g = np.exp(-0.5 * (((cols - cxx) / sig) ** 2 + ((rows - cyy) / sig) ** 2)) * a
        sl = canvas[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        sl[:] = sl * (1 - g[..., None]) + shade[i] * g[..., None]
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)[..., ::-1]
