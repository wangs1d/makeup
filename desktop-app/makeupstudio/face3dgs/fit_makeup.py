"""fit_makeup — 把妆容 spec 贴合到用户真实 3DGS 脸部点云上。

思路（无 GPU、无神经拟合，纯几何）：
    1. landmark 三角化：对采集视频的抽样帧跑 FaceTracker，得到 468×2 像素地标；
       配合 COLMAP 相机位姿做 DLT 线性三角化 → 用户脸部 468 个 3D 地标（splat 世界系）；
    2. 配准：canonical 脸模型（MediaPipe canonical_face_model.obj，顶点与地标一一对应）
       与用户地标做带尺度 Procrustes（Kabsch），把 canonical 网格（含 UV）变换进
       splat 世界系；
    3. 区域标签：splat 最近邻查询 canonical 表面采样点 → 得到 UV → 查各妆容层的
       蒙版（复用 bake_makeup 的 RegionMasks.bake，形状参数与 App 一致）；
    4. 上色：非破坏式混合 rgba = c·(1-w) + mc·shade·w，w = 蒙版覆盖率×opacity×
       intensity；光照 shade 用与内核一致的 wrap-diffuse（法线取自 splat 旋转）。
    导出 madeup.ply / madeup.splat + 前后对比预览图。
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import colmap_io
from .reconstruct import ReconResult
from .splat_io import export_splat, read_ply, write_ply

_REFS = Path(__file__).resolve().parent.parent.parent.parent / "makeup-skill" / "references"
N_CANON_VERTS = 468          # canonical 模型与 FaceTracker 468 地标一一对应
SAMPLE_SURFACE = 24000       # canonical 表面采样点数（KD 索引用）


def _load_core():
    import importlib.util
    if "preview_render_core" not in globals():
        spec = importlib.util.spec_from_file_location(
            "preview_render_core", _REFS.parent / "scripts" / "preview_render.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        globals()["preview_render_core"] = mod
    return globals()["preview_render_core"]


@dataclass
class FitResult:
    cloud: dict[str, np.ndarray]
    ply_path: Path
    splat_path: Path
    landmarks_3d: np.ndarray          # (468,3) 三角化出的用户地标（splat 世界系）
    n_views_used: int
    fit_rmse: float                   # 配准后 canonical↔用户地标 RMSE（canonical 单位）
    previews: list[Path]


# ---------------- 几何工具 ----------------

def projection_matrix(cam: colmap_io.Camera, qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R = colmap_io.quat_to_rotmat(qvec)
    K = np.array([[cam.params[0], 0, cam.params[1]],
                  [0, cam.params[0], cam.params[2]],
                  [0, 0, 1]])
    return K @ np.hstack([R, tvec.reshape(3, 1)])


def triangulate_dlt(Ps: list[np.ndarray], xys: list[np.ndarray]) -> np.ndarray:
    """多视角 DLT 三角化。Ps: 各视角 3×4 投影；xys: 该点各视角像素坐标。"""
    A = np.zeros((2 * len(Ps), 4))
    for i, (P, xy) in enumerate(zip(Ps, xys)):
        A[2 * i] = xy[0] * P[2] - P[0]
        A[2 * i + 1] = xy[1] * P[2] - P[1]
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    return X[:3] / max(X[3], 1e-9)


def kabsch_similarity(src: np.ndarray, dst: np.ndarray,
                      weights: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """带尺度相似变换 src→dst：返回 (scale, R, t)，src @ (sR).T + t ≈ dst。"""
    w = np.ones(len(src)) if weights is None else weights
    w = w / w.sum()
    cs, cd = (src * w[:, None]).sum(0), (dst * w[:, None]).sum(0)
    A, B = src - cs, dst - cd
    H = (A * w[:, None]).T @ B
    u, _, vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    R = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    s = float((w * (B * (A @ R.T)).sum(1)).sum() / max((w * (A * A).sum(1)).sum(), 1e-12))
    t = cd - s * (R @ cs)
    return s, R, t


# ---------------- 贴合器 ----------------

class FaceMakeupFitter:
    def __init__(self, tracker_factory: Callable[[], object] | None = None):
        self.core = _load_core()
        self.model = self.core.FaceModel(_REFS / "canonical_face_model.obj")
        self.model.prepare(_REFS / "landmark-regions.json", _REFS / "canonical_face_model.obj")
        self.regions = self.core.RegionMasks(_REFS / "landmark-regions.json",
                                             _REFS / "canonical_face_model.obj")
        self._tracker_factory = tracker_factory or self._default_tracker

    @staticmethod
    def _default_tracker():
        from ..tracker import FaceTracker
        return FaceTracker(smooth=False)

    # ---- 步骤 1：多帧地标三角化 ----

    def triangulate_landmarks(self, result: ReconResult, tracker=None,
                              max_views: int = 40,
                              progress: Callable[[float, str], None] | None = None
                              ) -> tuple[np.ndarray, int]:
        """抽样帧 → MediaPipe 2D 地标 + COLMAP 位姿 → DLT 三角化 (468,3)。"""
        tracker = tracker or self._tracker_factory()
        model = colmap_io.read_sparse(result.sparse_dir)
        cam = model.camera
        names = sorted(set(model.images) & {p.name for p in result.images_dir.glob("*.jpg")})
        names.sort()
        if not names:
            raise RuntimeError("COLMAP 位姿与抽帧文件名不匹配")
        if len(names) > max_views:
            names = [names[i] for i in
                     np.linspace(0, len(names) - 1, max_views).astype(int)]

        obs: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        used = 0
        t_ms = 0.0
        for name in names:
            img = cv2.imread(str(result.images_dir / name))
            if img is None:
                continue
            t_ms += 1000.0 / 25.0            # VIDEO 模式只要求时间戳单调递增
            det = tracker.detect(img, t_ms)
            if det is None:
                continue
            # 大偏航角下 MediaPipe 会对被遮挡侧地标给出幻觉位置，射线不交，
            # 三角化会爆出无穷远点 → 只采信接近正面的帧
            if abs(det["pose"][0]) > 40.0:
                continue
            im = model.images[name]
            if im["cam_id"] != cam.cam_id:
                continue
            used += 1
            P = projection_matrix(cam, im["qvec"], im["tvec"])
            px = det["px"][:N_CANON_VERTS]
            for li in range(N_CANON_VERTS):
                obs.setdefault(li, []).append((P, px[li]))
            if progress:
                progress(used / len(names), f"地标检测 {used}/{len(names)}")
        tracker.close() if hasattr(tracker, "close") else None
        if used < 3:
            raise RuntimeError(f"仅 {used} 帧检测到地标，无法三角化（检查视频质量）")

        L = np.zeros((N_CANON_VERTS, 3))
        ok = 0
        for li, pairs in obs.items():
            if len(pairs) < 2:
                continue
            L[li] = triangulate_dlt([p[0] for p in pairs], [p[1] for p in pairs])
            ok += 1
        # 鲁棒剔除：相对中位数偏离过远的飞点置为无效（register 会跳过）
        good = np.linalg.norm(L, axis=1) > 0
        if good.sum() > 10:
            med = np.median(L[good], axis=0)
            d = np.linalg.norm(L[good] - med, axis=1)
            thr = np.median(d) * 6 + 1e-6
            bad = good.copy()
            bad[good] = d > thr
            L[bad] = 0.0
            ok = int((good & ~bad).sum())
        if ok < 200:
            raise RuntimeError(f"仅 {ok}/468 个地标三角化成功，位姿/检测质量不足")
        return L, used

    # ---- 步骤 2：canonical → splat 世界系配准 ----

    def register(self, landmarks: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
        canon = self.model.base[:N_CANON_VERTS]
        valid = np.linalg.norm(landmarks, axis=1) > 0
        s, R, t = kabsch_similarity(canon[valid], landmarks[valid])
        err = canon[valid] @ (s * R).T + t - landmarks[valid]
        rmse = float(np.sqrt((np.linalg.norm(err, axis=1) ** 2).mean()))
        return s, R, t, rmse

    # ---- 步骤 3+4：区域标签与上色 ----

    def apply_makeup(self, cloud: dict[str, np.ndarray], layers: list[dict],
                     landmarks: np.ndarray, intensity: float = 0.8,
                     env: str = "neutral") -> dict[str, np.ndarray]:
        """返回新的 cloud（不修改入参）。layers 为 spec["layers"]（enabled 过滤后）。"""
        core = self.core
        s, R, t, _ = self.register(landmarks)

        # canonical 表面采样 → splat 世界系，建 KD 索引（uv 查询）
        Vc = self.model.base
        tris = self.model.tris
        p0, p1, p2 = Vc[tris[:, 0]], Vc[tris[:, 1]], Vc[tris[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        area = np.maximum(area, 1e-12)
        rows = np.random.default_rng(7).choice(len(tris), size=SAMPLE_SURFACE, p=area / area.sum())
        rng = np.random.default_rng(11)
        r1, r2 = rng.random(SAMPLE_SURFACE), rng.random(SAMPLE_SURFACE)
        su = np.sqrt(r1)
        b0, b1, b2 = 1 - su, su * (1 - r2), su * r2
        S_uv = (self.model.uvs[tris[rows, 0]] * b0[:, None]
                + self.model.uvs[tris[rows, 1]] * b1[:, None]
                + self.model.uvs[tris[rows, 2]] * b2[:, None])
        S_xyz = (p0[rows] * b0[:, None] + p1[rows] * b1[:, None] + p2[rows] * b2[:, None])
        S_xyz = (S_xyz @ (s * R).T) + t
        flann = cv2.flann_Index(S_xyz.astype(np.float32),
                                dict(algorithm=1, trees=4, checks=128))  # KDTree，避免引入 scipy
        nn, _ = flann.knnSearch(np.asarray(cloud["xyz"], np.float32), 1,
                                params=dict(checks=128))
        nn = nn.reshape(-1).astype(np.int64)
        uv = S_uv[nn]                                  # 每个 splat 的 canonical UV

        # splat 法线（旋转矩阵第三列）→ wrap-diffuse 光照
        qx, qy, qz, qw = cloud["rot"][:, 0], cloud["rot"][:, 1], cloud["rot"][:, 2], cloud["rot"][:, 3]
        nx = 2 * (qx * qz + qy * qw)
        ny = 2 * (qy * qz - qx * qw)
        nz = 1 - 2 * (qx * qx + qy * qy)
        Nn = np.stack([nx, ny, nz], axis=1)
        Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12
        light = core.ENVS.get(env, core.ENVS["neutral"])
        Ldir = np.asarray(light["light"], np.float64)
        Ldir /= np.linalg.norm(Ldir)
        tint = np.asarray(light["tint"], np.float64)
        ndl = np.clip(Nn @ Ldir, -1, 1)
        wrap = 0.25
        ndl_w = np.clip((ndl + wrap) / (1 + wrap), 0, 1)
        shade = (0.55 + 0.45 * ndl_w)[..., None] * tint

        out = {k: v.copy() for k, v in cloud.items()}
        TEX = core.TEX
        for layer in layers:
            if not layer.get("enabled", True):
                continue
            mask = self.regions.bake(layer)             # (TEX,TEX,2) cov+cent
            if mask[..., 0].max() < 1e-4:
                continue
            tx = np.clip(uv[:, 0] * (TEX - 1), 0, TEX - 1.001)
            ty = np.clip((1 - uv[:, 1]) * (TEX - 1), 0, TEX - 1.001)
            cov = core._bilinear(mask[..., 0][..., None], tx, ty)[..., 0]
            cent = core._bilinear(mask[..., 1][..., None], tx, ty)[..., 0]
            # 蒙版覆盖率峰值归一：区域中心的 splat 拿到完整 opacity，
            # 边缘按羽化衰减（覆盖率原样当权重会整体偏淡）
            peak = float(cov.max())
            if peak > 1e-4:
                cov = np.clip(cov / peak, 0, 1)
            cent = core._bilinear(mask[..., 1][..., None], tx, ty)[..., 0]
            col = core.sample_ramp(layer["color_stops"], np.clip(cent, 0, 1))
            opacity = float(layer.get("opacity", 0.7)) * float(intensity)
            finish = layer.get("finish", "satin")
            if finish == "gloss":
                opacity = min(1.0, opacity * 1.08)
            w = np.clip(cov * opacity, 0, 1)[..., None]
            mc = np.clip(col, 0, 1) * shade
            out["rgba"][:, :3] = out["rgba"][:, :3] * (1 - w) + mc * w
            out["rgba"][:, 3] = np.maximum(out["rgba"][:, 3], w[..., 0] * 0.6)
        return out

    # ---- 主入口 ----

    def fit(self, result: ReconResult, spec: dict, out_dir: str | Path,
            face_ply: str | Path | None = None, intensity: float = 0.8,
            env: str = "neutral") -> FitResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cloud = read_ply(face_ply or result.ply_path)
        landmarks, n_views = self.triangulate_landmarks(result)
        s, R, t, rmse = self.register(landmarks)
        layers = [dict(l) for l in spec.get("layers", []) if l.get("enabled", True)]
        made = self.apply_makeup(cloud, layers, landmarks, intensity=intensity, env=env)

        ply_path = out_dir / "madeup.ply"
        splat_path = out_dir / "madeup.splat"
        write_ply(made, ply_path)
        export_splat(made, splat_path)
        previews = self.render_previews(cloud, made, result, out_dir)
        (out_dir / "fit_report.json").write_text(json.dumps({
            "views_used": n_views, "fit_rmse_canonical": round(rmse, 4),
            "scale": s, "layers_applied": [l["id"] for l in layers],
            "splats": int(len(made["xyz"])),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return FitResult(cloud=made, ply_path=ply_path, splat_path=splat_path,
                         landmarks_3d=landmarks, n_views_used=n_views,
                         fit_rmse=rmse, previews=previews)

    # ---- 预览：用采集相机位姿做前向泼溅 ----

    def render_previews(self, before: dict, after: dict, result: ReconResult,
                        out_dir: Path, size: int = 512) -> list[Path]:
        model = colmap_io.read_sparse(result.sparse_dir)
        cam = model.camera
        names = sorted(model.images)
        picks = [names[len(names) // 4], names[len(names) // 2], names[3 * len(names) // 4]]
        outs = []
        for i, name in enumerate(picks):
            im = model.images[name]
            R = colmap_io.quat_to_rotmat(im["qvec"])
            t = im["tvec"]
            img = render_cloud(after, R, t, cam, w=size, h=int(size * 0.75))
            p = out_dir / f"preview_{i}.png"
            cv2.imwrite(str(p), img)
            outs.append(p)
        return outs


def render_cloud(cloud: dict[str, np.ndarray], R: np.ndarray, t: np.ndarray,
                 cam: colmap_io.Camera, w: int = 512, h: int = 384) -> np.ndarray:
    """通用前向泼溅（圆核、后→前合成），用于用户点云的快速预览。"""
    xyz = (R @ np.asarray(cloud["xyz"], np.float64).T).T + t
    z = xyz[:, 2]
    scale = min(w / cam.width, h / cam.height)
    f = float(cam.params[0]) * scale
    cx = float(cam.params[1]) * scale
    cy = float(cam.params[2]) * scale
    px = cx + xyz[:, 0] * f / np.maximum(z, 1e-6)
    py = cy + xyz[:, 1] * f / np.maximum(z, 1e-6)   # COLMAP 相机 y 轴向下（图像行向下）
    rgba = np.asarray(cloud["rgba"], np.float64)
    scale3 = np.asarray(cloud["scale"], np.float64)

    order = np.argsort(-z)
    canvas = np.full((h, w, 3), 0.09, np.float64)
    for i in order:
        a = rgba[i, 3]
        if a < 0.02 or z[i] <= 0.05:
            continue
        sig = float(scale3[i, 0]) * f / z[i]
        if sig < 0.35 or sig > w * 0.4:
            continue
        cxx, cyy = px[i], py[i]
        half = int(sig * 3) + 1
        x0, x1 = int(cxx) - half, int(cxx) + half + 1
        y0, y1 = int(cyy) - half, int(cyy) + half + 1
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            continue
        rows, cols = np.mgrid[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        g = np.exp(-0.5 * (((cols - cxx) / sig) ** 2 + ((rows - cyy) / sig) ** 2)) * a
        sl = canvas[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        sl[:] = sl * (1 - g[..., None]) + rgba[i, :3] * g[..., None]
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)[..., ::-1]
