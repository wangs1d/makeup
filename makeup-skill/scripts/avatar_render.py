#!/usr/bin/env python3
"""avatar_render — 3DGS 画像的软件光栅化预览（无 GPU/无 Unity 出效果图）。

EWA splatting 简化实现：3D 协方差 → 视空间 → 逐高斯焦距缩放 → 屏幕 2D 协方差 →
按视深远→近排序逐个 alpha 混合。背景渐变/环境色温与 preview_render 同预设，
保证参考图观感与既有管线一致。

规模策略：预览默认抽稀到 ≤60k 高斯（静帧 2~5 秒/帧可接受）；妆容预览 = tint
混色后的颜色直接参与光栅化，附加溅射（add_splats）最后按远→近叠 billboard 高斯
（与 preview_render._draw_splats 同画法）。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ENVS = {
    "neutral": (1.00, 1.00, 1.00),
    "warm":    (1.10, 0.98, 0.86),
    "cool":    (0.93, 0.98, 1.08),
    "dim":     (1.00, 0.92, 0.85),
}

PIXEL_BLUR = 0.3          # 屏幕 2D 协方差的像素抗锯齿项（标准 3DGS 同值）


def imwrite(path: str | Path, img_bgr: np.ndarray, quality: int = 90) -> bool:
    """Unicode 安全写图（cv2.imwrite 在 Windows 中文路径会静默失败）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
                           if ext.lower() in (".jpg", ".jpeg") else [])
    if not ok:
        return False
    buf.tofile(str(p))
    return True


def imread(path: str | Path) -> np.ndarray | None:
    """Unicode 安全读图（与 imwrite 对应；读不了返回 None）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(N,4) (w,x,y,z) → (N,3,3) 旋转矩阵。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), np.float32)
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


def _background(w: int, h: int) -> np.ndarray:
    yy = np.linspace(0, 1, h)[:, None]
    top = np.array([0.10, 0.085, 0.075])
    bot = np.array([0.16, 0.135, 0.115])
    bg = top * (1 - yy[..., None]) + bot * yy[..., None]
    xx = np.linspace(-1, 1, w)[None, :]
    return bg * (1 - 0.22 * (xx ** 2 + (yy * 2 - 1) ** 2 * 0.5))[..., None]


class AvatarRenderer:
    """一个画像一个实例；相机绕脸中心（归一化空间原点）轨道旋转，看 -Z 视向。"""

    def __init__(self, av, env: str = "neutral", max_gaussians: int = 130000):
        self.av = av.decimated(max_gaussians)
        av = self.av
        self.rot = _quat_to_rot(av.quats)
        S = av.scales
        M = self.rot * S[:, None, :]                   # R · diag(s)
        self.cov3 = (M @ np.transpose(M, (0, 2, 1))).astype(np.float32)   # R S Sᵀ Rᵀ
        self.env = env if env in ENVS else "neutral"

    def _camera(self, yaw_deg: float, pitch_deg: float, cam_dist: float):
        yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
        cy, sy, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        Rw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ \
             np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        campos = Rw @ np.array([0.0, 0.0, cam_dist])
        Rview = Rw.T                                    # world→view（相机看 -Z）
        return campos, Rview

    def render(self, yaw_deg: float = 0.0, pitch_deg: float = 0.0, size: int = 512,
               tint: np.ndarray | None = None, intensity: float = 1.0,
               add_splats: list[dict] | None = None) -> np.ndarray:
        """返回 (h,w,3) BGR uint8。tint: (N,4) rgba（a=覆盖权重）；intensity 运行时浓淡。"""
        av = self.av
        w = h = size
        f = h * 1.85
        cam_dist = 2.3
        tint_env = np.array(ENVS[self.env], np.float32)
        campos, Rview = self._camera(yaw_deg, pitch_deg, cam_dist)

        pos_c = (av.means - campos[None, :]) @ Rview.T
        depth = -pos_c[:, 2]                            # 相机看 -Z → 取负为正深度
        keep = depth > 0.05
        idx_k = np.nonzero(keep)[0]
        dep = depth[keep]
        px = w / 2 + pos_c[keep, 0] * f / dep
        py = h / 2 - pos_c[keep, 1] * f / dep

        # 颜色（tint 混妆：rgb = lerp(base, tint.rgb, a·intensity)）+ 环境色温
        col = av.colors * tint_env[None, :]
        if tint is not None:
            a = np.clip(tint[idx_k, 3:4] * intensity, 0, 1)
            col = col[keep] * (1 - a) + tint[idx_k, :3] * tint_env[None, :] * a
        else:
            col = col[keep]

        # 逐高斯屏幕 2D 协方差：J = diag(f/z, f/z)（对角）→
        #   cov2 = (f/z)² · [Sv00 Sv01; Sv01 Sv11] + 像素模糊
        Sv = Rview @ self.cov3[keep] @ Rview.T          # (Nk,3,3) 视空间协方差
        s2 = (f / dep) ** 2
        ca = s2 * Sv[:, 0, 0] + PIXEL_BLUR
        cb = s2 * Sv[:, 0, 1]
        cc = s2 * Sv[:, 1, 1] + PIXEL_BLUR
        det = np.maximum(ca * cc - cb * cb, 1e-6)
        mid = 0.5 * (ca + cc)
        disc = np.sqrt(np.maximum(mid * mid - det, 0.0))
        radius = np.clip(3.0 * np.sqrt(np.maximum(mid + disc, 1e-6)), 1.0, 48.0)
        # conic = cov2⁻¹（标量分量：i00, i01, i11）
        i00 = cc / det
        i01 = -cb / det
        i11 = ca / det

        order = np.argsort(-dep)                        # 远→近
        canvas = _background(w, h).astype(np.float32)

        for k in order:
            op = av.opacities[idx_k[k]]
            if op < 0.01:
                continue
            cx_, cy_ = px[k], py[k]
            r = int(radius[k])
            x0, y0 = int(cx_) - r, int(cy_) - r
            gx0, gy0 = max(0, x0), max(0, y0)
            gx1 = min(w, x0 + 2 * r + 1)
            gy1 = min(h, y0 + 2 * r + 1)
            if gx0 >= gx1 or gy0 >= gy1:
                continue
            rows, colsx = np.mgrid[gy0:gy1, gx0:gx1]
            dx = colsx + 0.5 - cx_
            dy = rows + 0.5 - cy_
            power = -0.5 * (i00[k] * dx * dx + 2 * i01[k] * dx * dy + i11[k] * dy * dy)
            g = np.clip(np.exp(power) * op, 0, 1)
            sl = canvas[gy0:gy1, gx0:gx1]
            sl[:] = sl * (1 - g[..., None]) + col[k] * g[..., None]

        if add_splats:
            self._draw_add_splats(canvas, Rview, campos, f, add_splats, intensity)
        return np.clip(canvas * 255, 0, 255).astype(np.uint8)[..., ::-1]

    def _draw_add_splats(self, canvas, Rview, campos, f, splats, intensity):
        """附加溅射层：billboard 高斯（归一化空间 σ 直接用），远→近。"""
        items = []
        for s in splats:
            p = np.asarray(s["pos"], np.float32)
            pc = (p - campos) @ Rview.T
            d = -pc[2]
            if d <= 0.05:
                continue
            items.append((-d, p, pc, s))
        items.sort(key=lambda t: t[0])
        for _d, p, pc, s in items:
            px_ = canvas.shape[1] / 2 + pc[0] * f / -_d
            py_ = canvas.shape[0] / 2 - pc[1] * f / -_d
            sigma = np.asarray(s["sigma"], np.float32)
            r_px = float(sigma.max() * 3 * f / -_d)
            if r_px < 1.0:
                continue
            col = np.array([int(s["color"][1:3], 16), int(s["color"][3:5], 16),
                            int(s["color"][5:7], 16)], np.float32) / 255.0
            peak = float(s.get("alpha", 0.7)) * intensity
            half = int(r_px) + 1
            x0, y0 = int(px_) - half, int(py_) - half
            gx0, gy0 = max(0, x0), max(0, y0)
            gx1, gy1 = min(canvas.shape[1], x0 + 2 * half + 1), min(canvas.shape[0], y0 + 2 * half + 1)
            if gx0 >= gx1 or gy0 >= gy1:
                continue
            rows, colsx = np.mgrid[gy0:gy1, gx0:gx1]
            sx = (colsx - px_) / (r_px / 3)
            sy = (rows - py_) / (r_px / 3)
            g = np.clip(np.exp(-0.5 * (sx * sx + sy * sy)) * peak, 0, 1)
            sl = canvas[gy0:gy1, gx0:gx1]
            sl[:] = sl * (1 - g[..., None]) + col * g[..., None]

    # ---------- 组合图 ----------

    def render_preview_pair(self, tint: np.ndarray | None, add_splats: list[dict] | None,
                            out_path: str | Path | None = None, size: int = 512,
                            intensity: float = 1.0, yaw_deg: float = 0.0) -> np.ndarray:
        """裸妆 | 目标妆 并排（confirm 前给用户看）。返回 BGR。"""
        bare = self.render(yaw_deg=yaw_deg, size=size)
        made = self.render(yaw_deg=yaw_deg, size=size, tint=tint,
                           add_splats=add_splats, intensity=intensity)
        gap = np.full((size, 6, 3), 24, np.uint8)
        pair = np.concatenate([bare, gap, made], axis=1)
        cv2.putText(pair, "bare", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pair, "makeup", (size + 16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if out_path is not None:
            imwrite(out_path, pair, 92)
        return pair

    def render_turntable(self, out_dir: str | Path, tint: np.ndarray | None = None,
                         add_splats: list[dict] | None = None, size: int = 512,
                         views: int = 5, intensity: float = 1.0) -> list[np.ndarray]:
        """±yaw 转台静帧（demo/确认用）。"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for k, yaw in enumerate(np.linspace(-24, 24, views)):
            img = self.render(yaw_deg=float(yaw), size=size, tint=tint,
                              add_splats=add_splats, intensity=intensity)
            imwrite(out_dir / f"turntable_{k:02d}.jpg", img, 90)
            frames.append(img)
        return frames


# ---------------- 面向 avatar_session 的一站式入口 ----------------

def load_add_splats(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("splats", [])


def front_frame_fn(av, size: int = 512):
    """给 avatar_semantics.detect_landmark_anchors 用的渲染闭包。

    返回 (BGR 图, 深度图, 焦距px, 相机距离)。深度图由最近点云 z-buffer 累积。
    """
    def render_fn():
        r = AvatarRenderer(av, max_gaussians=130000)
        avd = r.av
        f = size * 1.85
        cam_dist = 2.3
        depth = np.full((size, size), 1e9, np.float32)
        pos = avd.means - np.array([0.0, 0.0, cam_dist], np.float32)
        d = -pos[:, 2]
        ok = d > 0.05
        px = (size / 2 + pos[:, 0] * f / d).astype(int)
        py = (size / 2 - pos[:, 1] * f / d).astype(int)
        m = ok & (px >= 0) & (px < size) & (py >= 0) & (py < size)
        np.minimum.at(depth.reshape(-1), py[m] * size + px[m], d[m])
        depth[depth > 1e8] = 0.0
        return r.render(size=size), depth, f, cam_dist
    return render_fn
