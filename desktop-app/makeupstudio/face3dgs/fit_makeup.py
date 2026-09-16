"""fit_makeup — 把妆容 spec 贴合到用户真实 3DGS 脸部点云上。

思路（无 GPU、无神经拟合，纯几何）：
    1. landmark 三角化：对采集视频的抽样帧跑 FaceTracker，得到 468×2 像素地标；
       配合 COLMAP 相机位姿做 DLT 线性三角化 → 用户脸部 468 个 3D 地标（splat 世界系）；
    2. 配准：canonical 脸模型（MediaPipe canonical_face_model.obj，顶点与地标一一对应）
       与用户地标做带尺度 Procrustes（Kabsch），把 canonical 网格（含 UV）变换进
       splat 世界系；
    3. 区域标签：splat 对 canonical 表面采样点做 kNN 距离加权 UV 归属
       （远离 canonical 表面的 splat（头发/背景）判为离群，永不涂妆）→
       查各妆容层的蒙版（复用 bake_makeup 的 RegionMasks.bake，高分辨率烘焙）；
    4. 上色：非破坏式混合 rgba = c·(1-w) + mc·shade·w，w = 蒙版覆盖率×opacity×
       intensity；光照 shade 用与内核一致的 wrap-diffuse（法线取自 splat 旋转）；
       finish 烘为逐点 gloss/shininess，渲染端加 Blinn-Phong 镜面（gloss/dewy
       的"水光/珠光"第一次有了物理来源）；
    5. 可选密度增强（densify=True）：妆区 splat 克隆缩小 σ 的子高斯，
       恢复唇线/眼线的锐度（合成路径 splat3d 同款思路移植到真实点云）。
    另有两条不依赖视频三角化的入口：
       fit_canonical  — P2：直接给 FLAME 底座（FlashAvatar 导出的 canonical 点云）上妆；
       apply_guidance — P1：妆容外观改由 guidance 图（Stable-Makeup 等）多视角投影
                         采样，参数化蒙版降级为区域门控。
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
FIT_TEX = 1024               # 贴合路径的蒙版烘焙分辨率（内核默认 512，唇线/眼线更锐）
KNN_K = 4                    # UV 归属的近邻数（距离加权，抑制离散采样噪声）

# finish → 逐点镜面参数（与内核 FINISH 同基准；渲染端按 gloss×(n·h)^shin 加高光）
FINISH_GLOSS = {"matte": 0.0, "satin": 0.25, "dewy": 0.55, "gloss": 0.9}
FINISH_SHIN = {"matte": 0.0, "satin": 56.0, "dewy": 90.0, "gloss": 150.0}

# Lab 色彩迁移系数（与 compositor.py 同表同语义，真实照片纹理专用）：
# 明度只按 LUM_SHIFT 部分跟随（保留皮肤纹理/光影），色度按 CHROMA_GAIN 推向妆色。
# alpha-over 的象牙底妆叠在自带明暗的真实照片上会把暗部反差放大成"花斑"。
LUM_SHIFT = {"foundation": 0.25, "concealer": 0.25, "contour": 0.40, "blush": 0.55,
             "eyeshadow": 0.50, "eyebrow": 0.55, "lipstick": 0.95, "highlight": 0.25}
CHROMA_GAIN = {"blush": 1.6, "eyeshadow": 1.5, "lipstick": 1.25, "highlight": 0.3,
               "contour": 1.2, "eyebrow": 1.2, "foundation": 0.45, "concealer": 0.6}


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


def projection_matrix_rt(R: np.ndarray, t: np.ndarray, cam: colmap_io.Camera) -> np.ndarray:
    K = np.array([[cam.params[0], 0, cam.params[1]],
                  [0, cam.params[0], cam.params[2]],
                  [0, 0, 1]])
    return K @ np.hstack([R, np.asarray(t, np.float64).reshape(3, 1)])


def _first_cam(views: list[dict]) -> colmap_io.Camera:
    return views[0]["cam"]


def _quat_normals(rot: np.ndarray) -> np.ndarray:
    """四元数 (xyzw) 把 (0,0,1) 旋到哪儿 = splat 薄片法线。"""
    qx, qy, qz, qw = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    nx = 2 * (qx * qz + qy * qw)
    ny = 2 * (qy * qz - qx * qw)
    nz = 1 - 2 * (qx * qx + qy * qy)
    Nn = np.stack([nx, ny, nz], axis=1)
    Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12
    return Nn


def _rotmat_to_quat_xyzw(Rm: np.ndarray) -> np.ndarray:
    """批量旋转矩阵 (m,3,3)（列 = 局部 x/y/z 轴）→ 四元数 (m,4) xyzw。

    按最大对角元选分支（Shepperd 法），保证数值稳定。"""
    Rm = np.asarray(Rm, np.float64)
    m00, m11, m22 = Rm[:, 0, 0], Rm[:, 1, 1], Rm[:, 2, 2]
    tr = m00 + m11 + m22
    i = np.argmax(np.stack([m00, m11, m22, tr], axis=1), axis=1)
    qx, qy, qz, qw = (np.zeros(len(Rm)) for _ in range(4))
    for k in range(4):
        m = i == k
        if not m.any():
            continue
        if k == 3:
            S = np.sqrt(np.maximum(tr[m] + 1.0, 1e-12)) * 2.0
            qw[m] = 0.25 * S
            qx[m] = (Rm[m, 2, 1] - Rm[m, 1, 2]) / S
            qy[m] = (Rm[m, 0, 2] - Rm[m, 2, 0]) / S
            qz[m] = (Rm[m, 1, 0] - Rm[m, 0, 1]) / S
        elif k == 0:
            S = np.sqrt(np.maximum(1.0 + m00[m] - m11[m] - m22[m], 1e-12)) * 2.0
            qw[m] = (Rm[m, 2, 1] - Rm[m, 1, 2]) / S
            qx[m] = 0.25 * S
            qy[m] = (Rm[m, 0, 1] + Rm[m, 1, 0]) / S
            qz[m] = (Rm[m, 0, 2] + Rm[m, 2, 0]) / S
        elif k == 1:
            S = np.sqrt(np.maximum(1.0 + m11[m] - m00[m] - m22[m], 1e-12)) * 2.0
            qw[m] = (Rm[m, 0, 2] - Rm[m, 2, 0]) / S
            qx[m] = (Rm[m, 0, 1] + Rm[m, 1, 0]) / S
            qy[m] = 0.25 * S
            qz[m] = (Rm[m, 1, 2] + Rm[m, 2, 1]) / S
        else:
            S = np.sqrt(np.maximum(1.0 + m22[m] - m00[m] - m11[m], 1e-12)) * 2.0
            qw[m] = (Rm[m, 1, 0] - Rm[m, 0, 1]) / S
            qx[m] = (Rm[m, 0, 2] + Rm[m, 2, 0]) / S
            qy[m] = (Rm[m, 1, 2] + Rm[m, 2, 1]) / S
            qz[m] = 0.25 * S
    q = np.stack([qx, qy, qz, qw], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return q


def _frame_quats(n_axis: np.ndarray, tangent: np.ndarray | None = None) -> np.ndarray:
    """法线 (+ 可选切向) → 薄片高斯四元数 (xyzw)，局部 z = 法线。"""
    Nn = n_axis / (np.linalg.norm(n_axis, axis=1, keepdims=True) + 1e-12)
    if tangent is None:
        return _quats_from_normals(Nn)
    t1 = tangent - Nn * (tangent * Nn).sum(1, keepdims=True)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(Nn, t1)
    return _rotmat_to_quat_xyzw(np.stack([t1, t2, Nn], axis=2))


def _quats_from_normals(n: np.ndarray) -> np.ndarray:
    """(0,0,1)→n 的最短弧四元数 (xyzw)，向量化（与 _quat_normals 互逆）。"""
    q = np.empty((len(n), 4), np.float64)
    q[:, 0] = -n[:, 1]
    q[:, 1] = n[:, 0]
    q[:, 2] = 0.0
    q[:, 3] = 1.0 + n[:, 2]
    degenerate = np.linalg.norm(q, axis=1) < 1e-10   # n 精确 = (0,0,-1)
    q[degenerate] = [1.0, 0.0, 0.0, 0.0]             # 绕 x 翻 180°
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return q


def _sample_tris(V: np.ndarray, tris: np.ndarray, n: int, rng: np.random.Generator):
    """三角形面片面积加权采样：返回 (P (n,3), 重心坐标 (n,3), 采样到的行索引)。"""
    p0, p1, p2 = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    area = np.maximum(area, 1e-14)
    rows = rng.choice(len(tris), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    su = np.sqrt(r1)
    b0, b1, b2 = 1 - su, su * (1 - r2), su * r2
    P = p0[rows] * b0[:, None] + p1[rows] * b1[:, None] + p2[rows] * b2[:, None]
    return P, np.stack([b0, b1, b2], 1), rows


def _median_spacing(xyz: np.ndarray, cap: int = 3000, seed: int = 3) -> float:
    """点云中位邻间距（粗采样 kNN），用于把带宽度阈值锚定在 splat 噪声尺度上。"""
    n = len(xyz)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(cap, n), replace=False)
    sub = np.asarray(xyz[idx], np.float32)
    flann = cv2.flann_Index(sub, dict(algorithm=1, trees=4, checks=64))
    _nn, d2 = flann.knnSearch(sub, 2, params=dict(checks=64))
    return float(np.sqrt(np.maximum(d2[:, 1], 0)).mean() + 1e-9)


def blinn_phong_spec(cloud: dict[str, np.ndarray], R: np.ndarray,
                     env: str = "neutral", strength: float = 0.5,
                     light_dir: np.ndarray | None = None) -> np.ndarray:
    """逐点 Blinn-Phong 镜面（RGB，强度取自 apply_makeup 烘焙的 gloss/shin）。

    COLMAP 相机看 +z：可见表面法线朝 -z，指向眼睛的视线 V=(0,0,-1)。
    光源方向默认取环境预设（世界系），可用 light_dir 覆盖（如演示的顺光头灯）——
    唇釉/珠光的高光随视角流动，而不是静态烙在颜色里。
    """
    n = len(cloud["xyz"])
    if "gloss" not in cloud or float(np.max(np.asarray(cloud["gloss"]), initial=0.0)) < 1e-4:
        return np.zeros((n, 3), np.float32)
    core = _load_core()
    if light_dir is not None:
        Lc = np.asarray(light_dir, np.float64)
        Lc = R @ (Lc / (np.linalg.norm(Lc) + 1e-12))
        tint = np.ones(3)
    else:
        light = core.ENVS.get(env, core.ENVS["neutral"])
        Lw = np.asarray(light["light"], np.float64)
        Lw /= np.linalg.norm(Lw)
        Lc = R @ Lw
        tint = np.asarray(light["tint"], np.float64)
    V = np.array([0.0, 0.0, -1.0])               # 指向相机（相机在 -z 侧看 +z 处的脸）
    H = Lc + V
    H /= np.linalg.norm(H)
    Nc = _quat_normals(np.asarray(cloud["rot"], np.float64)) @ R.T
    ndv = np.clip(Nc @ V, 0.0, 1.0)
    ndh = np.clip(Nc @ H, 0.0, 1.0)
    gloss = np.asarray(cloud["gloss"], np.float64)
    shin = np.asarray(cloud.get("shin", np.full(n, 64.0)), np.float64)
    spec = gloss * (ndh ** np.maximum(shin, 1.0)) * strength
    spec += gloss * (1.0 - ndv) ** 3 * 0.35        # 掠射角 sheen（与内核 fres 项同源）
    return (spec[..., None] * tint).astype(np.float32)


def triangulate_dlt(Ps: list[np.ndarray], xys: list[np.ndarray]) -> np.ndarray:
    """多视角 DLT 三角化（带 Hartley 归一化）。Ps: 各视角 3×4 投影；xys: 该点各视角像素坐标。

    COLMAP 世界坐标可达 1e9 量级：像素坐标与投影矩阵量级悬殊会让齐次最小二乘
    数值爆炸、解跑到相机后方（cheirality 破坏）。对该点的全部观测做一次各向同性
    归一（质心移原点、均距缩到 √2），配套 P̂ᵢ = T·Pᵢ 后再解（世界坐标不变）。
    """
    xs = np.asarray(xys, np.float64)
    c = xs.mean(0)
    m = max(float(np.linalg.norm(xs - c, axis=1).mean()), 1e-9)
    s = float(np.sqrt(2.0)) / m
    T = np.array([[s, 0.0, -s * c[0]],
                  [0.0, s, -s * c[1]],
                  [0.0, 0.0, 1.0]])
    A = np.zeros((2 * len(Ps), 4))
    for i, (P, xy) in enumerate(zip(Ps, xys)):
        xh = T @ np.append(xy, 1.0)
        Ph = T @ P
        A[2 * i] = xh[0] * Ph[2] - Ph[0]
        A[2 * i + 1] = xh[1] * Ph[2] - Ph[1]
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    # SVD 解符号任意：w=X[3] 为负时必须整体翻转再反归一化，否则除以 max(w,1e-9)
    # 会把坐标爆到 1e9 量级并翻到相机后方（真实 COLMAP 大尺度场景下必现）
    if abs(X[3]) < 1e-12:
        X = X.copy()
        X[3] = 1e-12
    if X[3] < 0:
        X = -X
    return X[:3] / X[3]


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
            X = triangulate_dlt([p[0] for p in pairs], [p[1] for p in pairs])
            # cheirality：正深度观测数需过半，否则该地标解在相机后方，弃用
            depths = np.array([(P @ np.append(X, 1.0))[2] for P, _ in pairs])
            if (depths > 0).sum() < len(depths) * 0.5:
                continue
            L[li] = X
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

    # ---- 唇部拓扑与 3D 带状权重（L0：修"口红糊满嘴"） ----

    def lip_topology(self) -> dict:
        """canonical 网格的唇部分类（缓存，L0 改色与 L1 壳层共用）。

        inner_only — 顶点全在 lips_inner 环上的三角片：闭嘴时是唇线缝隙，张嘴时
                     被拉伸成口腔开口面（牙齿/暗部），永不承载口红；
        band      — 从 inner_only 出发做边邻接洪泛填充、被 lips_outer 环顶点阻挡
                    （唇缘行收入但不外扩）：恰好是唇红面，拓扑上不可能越唇缘一步；
        d_out/d_in — 顶点到两环的欧氏距离（渐变 t 与边缘羽化用）。
        """
        if getattr(self, "_lip_topo", None) is not None:
            return self._lip_topo
        V, tris = self.model.base, self.model.tris
        data = json.loads((_REFS / "landmark-regions.json").read_text(encoding="utf-8"))
        outer = np.array(data["regions"]["lips_outer"]["indices"])
        inner = np.array(data["regions"]["lips_inner"]["indices"])
        OL = set(outer.tolist())
        IL = set(inner.tolist())
        edge_tris: dict[tuple[int, int], list[int]] = {}
        for ti, tr in enumerate(tris):
            a, b, c = (int(x) for x in tr)
            for u, v in ((a, b), (b, c), (c, a)):
                edge_tris.setdefault((min(u, v), max(u, v)), []).append(ti)
        inner_only = [ti for ti, tr in enumerate(tris)
                      if set(int(x) for x in tr) <= IL]
        inner_only_set = set(inner_only)
        band, boundary = set(), set()
        grow = set(inner_only)
        while grow:
            ti = grow.pop()
            tr = tris[ti]
            for u, v in ((tr[0], tr[1]), (tr[1], tr[2]), (tr[2], tr[0])):
                key = (min(int(u), int(v)), max(int(u), int(v)))
                for nj in edge_tris.get(key, []):
                    if nj in band or nj in boundary or nj in inner_only_set:
                        continue
                    if set(int(x) for x in tris[nj]) & OL:
                        boundary.add(nj)      # 唇缘行（含外圈顶点）：收入但不外扩
                    else:
                        band.add(nj)
                        grow.add(nj)
        d_out = np.linalg.norm(V[:, None, :] - V[outer][None], axis=2).min(1)
        d_in = np.linalg.norm(V[:, None, :] - V[inner][None], axis=2).min(1)
        self._lip_topo = {"band": np.array(sorted(band | boundary), np.int64),
                          "mouth": np.array(inner_only, np.int64),
                          "d_out": d_out, "d_in": d_in,
                          "outer": outer, "inner": inner}
        return self._lip_topo

    def _world_lip_mesh(self, cloud: dict, landmarks: np.ndarray,
                        s: float, R: np.ndarray, t: np.ndarray):
        """世界系唇部网格。精确 UV 路径（landmark 稠密面片）用真实地标顶点——
        张嘴形态是真实的；通用路径（COLMAP 点云）用配准后的 canonical 底座。
        返回 (Vw, band_tris (m,3), mouth_tris (m,3))，行为顶点索引。"""
        topo = self.lip_topology()
        band = self.model.tris[topo["band"]]
        mouth = self.model.tris[topo["mouth"]]
        if "uv" in cloud:
            Vw = np.asarray(landmarks, np.float64)
            valid = np.linalg.norm(Vw, axis=1) > 0
            band = band[valid[band].all(1)]
            mouth = mouth[valid[mouth].all(1)]
        else:
            Vw = self.model.base @ (s * R).T + t
        return Vw, band, mouth

    def _lip_band_weight(self, cloud: dict, landmarks: np.ndarray,
                         s: float, R: np.ndarray, t: np.ndarray,
                         mouth: np.ndarray | None = None, return_aux: bool = False):
        """3D 唇红带权重 (n,) + 渐变坐标 (n,)。口红改色不再采 UV 实心填充
        （张嘴时口腔面共享同一片 UV，35% 内环顶点落在填充内 → 粉色糊满嘴），
        而是测 splat 到唇红面的距离 + 口腔开口面的"贴面/背面"侧别排除。
        return_aux=True 时额外返回 {"ib","band_n"}（最近带样点索引与其法线），
        供壳层在云自无法线时取法线兜底。"""
        xyz = np.asarray(cloud["xyz"], np.float64)
        n = len(xyz)
        Vw, band, mouth_tris = self._world_lip_mesh(cloud, landmarks, s, R, t)
        if len(band) == 0:
            zero2 = (np.zeros(n, np.float32), np.zeros(n, np.float32))
            return zero2 + ({},) if return_aux else zero2
        topo = self.lip_topology()
        Nw = self.model.vertex_normals(Vw)

        def _tris_normals(T):
            p0, p1, p2 = Vw[T[:, 0]], Vw[T[:, 1]], Vw[T[:, 2]]
            fn = np.cross(p1 - p0, p2 - p0)
            return fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)

        rng = np.random.default_rng(31)
        Pb, bb, rows_b = _sample_tris(Vw, band, 9000, rng)
        verts_b = band[rows_b]
        do_s = (topo["d_out"][verts_b] * bb).sum(1)     # 带内每样点到两环距离（渐变/羽化）
        di_s = (topo["d_in"][verts_b] * bb).sum(1)

        fl = cv2.flann_Index(Pb.astype(np.float32), dict(algorithm=1, trees=4, checks=128))
        _nn, d2b = fl.knnSearch(xyz.astype(np.float32), 1, params=dict(checks=128))
        db = np.sqrt(np.maximum(d2b[:, 0], 0.0))
        ib = _nn[:, 0].astype(np.int64)

        edges = np.stack([np.linalg.norm(Vw[band[:, 1]] - Vw[band[:, 0]], axis=1),
                          np.linalg.norm(Vw[band[:, 2]] - Vw[band[:, 0]], axis=1),
                          np.linalg.norm(Vw[band[:, 2]] - Vw[band[:, 1]], axis=1)], 1)
        loc_face = ((topo["d_out"][band] + topo["d_in"][band]).mean(1)) * 0.5
        # 核半径需盖过"地标↔纹理"系统性错位：说话/噘嘴时唇地标跨帧移动，
        # 三角化唇带偏离视频嘴唇可达 ~2% 脸高。越界与否不由 τ 决定，
        # 由 edge 的环形距离（最近带样点的 do/di）约束。
        tau = max(0.45 * float(np.median(edges)),
                  2.0 * float(np.median(loc_face)) * s,
                  2.2 * _median_spacing(xyz))
        w = np.clip(1.0 - db / tau, 0.0, 1.0)

        if len(mouth_tris):                              # 口腔开口面侧别排除
            Pm, _bm, rows_m = _sample_tris(Vw, mouth_tris, 4000, rng)
            fnm = _tris_normals(mouth_tris)[rows_m]
            flm = cv2.flann_Index(Pm.astype(np.float32), dict(algorithm=1, trees=4, checks=128))
            _nm, d2m = flm.knnSearch(xyz.astype(np.float32), 1, params=dict(checks=128))
            dm = np.sqrt(np.maximum(d2m[:, 0], 0.0))
            side = ((xyz - Pm[_nm[:, 0].astype(np.int64)]) * fnm[_nm[:, 0].astype(np.int64)]).sum(1)
            w[(dm < db * 1.5) & (np.abs(side) < 0.5 * tau)] = 0.0
        do_n, di_n = do_s[ib], di_s[ib]                  # 边缘羽化：外圈软、唇线侧饱满
        loc = 0.5 * (do_n + di_n) + 1e-6
        edge = np.clip(do_n / (0.28 * loc), 0, 1) * np.clip(di_n / (0.06 * loc), 0, 1)

        # 视频颜色门控：唇纹理比周围皮肤更红更暗。说话/噘嘴让三角化唇带与真实
        # 唇存在 ~2% 脸高的系统性错位，纯几何核会把唇色涂到唇外皮肤上；
        # 素颜纹理本身就是"嘴唇在哪"的最可靠证据，用它把权重收进真实唇内。
        gated = False
        if "rgba" in cloud:
            rgb = np.clip(np.asarray(cloud["rgba"], np.float64)[:, :3], 0, 1)
            rg = rgb[:, 0] - rgb[:, 1]
            lum = rgb.mean(1)
            near = w > 0.02
            if near.sum() > 50:
                rg0, lum0 = np.median(rg[near]), np.median(lum[near])
                score = (rg - rg0) + 0.8 * (lum0 - lum)
                thr = 0.5 * (np.percentile(score[near], 90)
                             - np.percentile(score[near], 50))
                if thr > 0.04:                           # 唇肤对比充分才启用
                    w_color = np.clip((score - 0.25 * thr) / (0.75 * thr), 0, 1)
                    w = np.clip(1.0 - db / tau, 0.0, 1.0) * edge * w_color
                    gated = True
        if not gated:                                    # 无颜色信号：核收回到带宽度
            w = np.clip(1.0 - db / max(0.45 * float(np.median(edges)),
                                       2.2 * _median_spacing(xyz)), 0.0, 1.0) * edge
        if mouth is not None:                            # 稠密路径：按采样三角形精确排除（最后施加，
            w[np.asarray(mouth, bool)] = 0.0             # 防止门控分支重算 w 时丢失）
        t_grad = di_n / (do_n + di_n + 1e-6)
        if return_aux:
            fnb = _tris_normals(band)[rows_b]            # 带样点法线（法线兜底用）
            return (w.astype(np.float32), t_grad.astype(np.float32),
                    {"ib": ib, "band_n": fnb})
        return w.astype(np.float32), t_grad.astype(np.float32)

    def _shade_for_normals(self, Nn: np.ndarray, env: str,
                           light_dir: np.ndarray | None = None,
                           floor: float = 0.0) -> np.ndarray:
        """任意法线的 wrap-diffuse 光照因子 (n,3)（与 _wrap_shade 同模型）。

        floor：亮度下限。口红/珠光自带高光层，漫反射压暗过多会发黑
        （floor=0.55 → 因子范围 [0.8,1]，妆容色相保持预设的明亮）。"""
        light = self.core.ENVS.get(env, self.core.ENVS["neutral"])
        Ldir = (np.asarray(light_dir, np.float64) if light_dir is not None
                else np.asarray(light["light"], np.float64))
        Ldir = Ldir / (np.linalg.norm(Ldir) + 1e-12)
        tint = np.asarray(light["tint"], np.float64)
        ndl = np.clip(Nn @ Ldir, -1, 1)
        wrap = 0.25
        fac = 0.55 + 0.45 * np.clip((ndl + wrap) / (1 + wrap), 0, 1)
        return (floor + (1.0 - floor) * fac)[..., None] * tint

    # ---- 步骤 3+4：区域标签与上色 ----

    def _surface_samples(self, s: float, R: np.ndarray, t: np.ndarray):
        """canonical 表面均匀采样（面积加权 + 重心坐标）→ splat 世界系 (xyz, uv)。"""
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
        return S_xyz.astype(np.float32), S_uv

    def _splat_uv(self, cloud: dict[str, np.ndarray], s: float, R: np.ndarray, t: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """splat → canonical UV。kNN 距离加权（单点 NN 在采样网格上会抖动，
        加权平均后妆容边界不再有采样格纹）。

        UV 平均带"跨岛保护"：canonical UV 是图集，3D 相邻的采样点可能分属不同
        UV 岛（纹理坐标相距很远），直接加权平均会落进蒙版空白区 → 妆容呈分形
        花斑。只平均与最近邻 UV 距离 < 0.15 的邻居，岛内平滑、永不跨岛。

        返回 (uv (n,2), near (n,) 最近邻距离, valid (n,) bool)。
        valid = 最近邻距离 ≤ 6×中位数：头发/眼镜/背景 splat 距 canonical 表面
        远超脸部噪声尺度，判为离群——它们永不被涂妆（发际线溢出的历史问题）。
        """
        S_xyz, S_uv = self._surface_samples(s, R, t)
        flann = cv2.flann_Index(S_xyz, dict(algorithm=1, trees=4, checks=128))  # KDTree
        nn, nd = flann.knnSearch(np.asarray(cloud["xyz"], np.float32), KNN_K,
                                 params=dict(checks=128))
        nn = nn.astype(np.int64)
        dist = np.sqrt(np.maximum(nd, 0.0))            # FLANN 返回平方距离
        w = 1.0 / (dist + 1e-6)
        uv0 = S_uv[nn[:, 0]]                           # 最近邻 UV（本岛锚点）
        d_uv = np.linalg.norm(S_uv[nn] - uv0[:, None, :], axis=2)
        in_island = d_uv < 0.15
        w = w * in_island                              # 跨岛邻居权重清零
        w[:, 0] = np.maximum(w[:, 0], 1e-6)            # 最近邻始终参与
        uv = (S_uv[nn] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        near = dist[:, 0]
        med = float(np.median(near))
        tau = max(6.0 * med, 1e-3)
        return uv, near, near <= tau

    def _wrap_shade(self, cloud: dict[str, np.ndarray], env: str,
                    light_dir: np.ndarray | None = None) -> np.ndarray:
        """wrap-diffuse 光照因子 (n,3)。法线取 splat 旋转第三列（与内核同模型）。

        light_dir：世界系光源方向覆盖（重建世界系朝向任意，ENVS 预设的 canonical
        帧光源未必对着脸；如点云被旋转为"鼻尖朝相机"时需相应旋转光源）。"""
        light = self.core.ENVS.get(env, self.core.ENVS["neutral"])
        if light_dir is not None:
            Ldir = np.asarray(light_dir, np.float64)
            Ldir = Ldir / (np.linalg.norm(Ldir) + 1e-12)
        else:
            Ldir = np.asarray(light["light"], np.float64)
            Ldir /= np.linalg.norm(Ldir)
        tint = np.asarray(light["tint"], np.float64)
        Nn = _quat_normals(np.asarray(cloud["rot"], np.float64))
        ndl = np.clip(Nn @ Ldir, -1, 1)
        wrap = 0.25
        ndl_w = np.clip((ndl + wrap) / (1 + wrap), 0, 1)
        return (0.55 + 0.45 * ndl_w)[..., None] * tint

    def apply_makeup(self, cloud: dict[str, np.ndarray], layers: list[dict],
                     landmarks: np.ndarray, intensity: float = 0.8,
                     env: str = "neutral", densify: bool = False,
                     light_dir: np.ndarray | None = None,
                     treatment: str = "alpha",
                     mouth: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """返回新的 cloud（不修改入参）。layers 为 spec["layers"]（enabled 过滤后）。

        densify=True 时在妆区克隆缩小 σ 的子高斯（数量 ≤ 原 35%），锐化唇线/眼线。
        结果附带逐点 gloss/shin 数组，供渲染端做 Blinn-Phong 镜面（物理 finish）。
        light_dir：世界系光源方向覆盖（语义见 _wrap_shade）。
        treatment："alpha"（默认，合成脸校准的 alpha-over）｜
                   "lab"（真实照片纹理：Lab 色彩迁移，明度保留、色度推向妆色，
                   与 compositor.py 的实时镜面同语义——底妆不再放大真实暗部）。
        mouth：稠密路径的逐 splat 口腔开口面标记（True=口腔，永不涂口红）；
               缺省时口红走 _lip_band_weight 的几何侧别排除。"""
        core = self.core
        s, R, t, _ = self.register(landmarks)

        prev_tex = core.TEX
        core.set_texture_size(FIT_TEX)            # 高分辨率蒙版：唇线/眼线采样更锐
        try:
            # cloud 自带精确 UV（canonical 拓扑网格/FLAME 导出）时直接用——
            # Kabsch 是全局相似变换，对非刚性真实脸会有 ~5% 的局部错位
            if "uv" in cloud:
                uv = np.asarray(cloud["uv"], np.float64)[:, :2]
                valid = np.isfinite(uv).all(1)
            else:
                uv, _near, valid = self._splat_uv(cloud, s, R, t)
            shade = self._wrap_shade(cloud, env, light_dir=light_dir)
            n = len(cloud["xyz"])
            paint_w = np.zeros(n, np.float32)
            gloss = np.zeros(n, np.float32)
            shin = np.zeros(n, np.float32)

            out = {k: v.copy() for k, v in cloud.items()}
            TEX = core.TEX
            for layer in layers:
                if not layer.get("enabled", True):
                    continue
                if layer["region"] == "lipstick":
                    # L0：口红改 3D 唇红带权重（张嘴口腔面被精确排除，
                    # UV 实心填充的"粉贴纸"问题不再存在）
                    cov, cent = self._lip_band_weight(cloud, landmarks, s, R, t,
                                                      mouth=mouth)
                    if float(cov.max()) < 1e-4:
                        continue
                else:
                    mask = self.regions.bake(layer)         # (TEX,TEX,2) cov+cent
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
                col = np.clip(core.sample_ramp(layer["color_stops"],
                                               np.clip(cent, 0, 1)), 0, 1)
                opacity = float(layer.get("opacity", 0.7)) * float(intensity)
                finish = layer.get("finish", "satin")
                if finish == "gloss":
                    opacity = min(1.0, opacity * 1.08)
                w = np.clip(cov * opacity, 0, 1)
                w[~valid] = 0.0                             # 离群 splat 永不涂妆
                paint_w = np.maximum(paint_w, w)
                region = layer["region"]
                wc = w[..., None]
                if treatment == "lab" and region in LUM_SHIFT:
                    # Lab 色彩迁移：明度按 LUM_SHIFT 部分跟随，色度按 CHROMA_GAIN
                    # 推向妆色——真实皮肤纹理/光影保留，只有颜色朝妆色走
                    cur = np.clip(out["rgba"][:, :3], 0, 1)
                    cur_lab = cv2.cvtColor(
                        (cur * 255).astype(np.uint8)[:, None, :], cv2.COLOR_RGB2Lab
                    ).astype(np.float32)[:, 0, :]
                    tgt_lab = cv2.cvtColor(
                        (col * 255).astype(np.uint8)[:, None, :], cv2.COLOR_RGB2Lab
                    ).astype(np.float32)[:, 0, :]
                    a_ch = np.clip(w * CHROMA_GAIN.get(region, 1.0), 0, 1)[..., None]
                    kL = (LUM_SHIFT[region] * w)[..., None]
                    new_lab = np.stack([
                        cur_lab[:, 0] + (tgt_lab[:, 0] - cur_lab[:, 0]) * kL[:, 0],
                        cur_lab[:, 1] + (tgt_lab[:, 1] - cur_lab[:, 1]) * a_ch[:, 0],
                        cur_lab[:, 2] + (tgt_lab[:, 2] - cur_lab[:, 2]) * a_ch[:, 0],
                    ], axis=1)
                    back = cv2.cvtColor(
                        np.clip(new_lab, 0, 255).astype(np.uint8)[:, None, :],
                        cv2.COLOR_Lab2RGB).astype(np.float32)[:, 0, :] / 255.0
                    out["rgba"][:, :3] = back
                else:
                    mc = col * shade
                    out["rgba"][:, :3] = out["rgba"][:, :3] * (1 - wc) + mc * wc
                if region == "lipstick":
                    # 口红区：底层 alpha 衰减，壳层成为主导贡献——避免底层
                    # 逐点色噪/深度排序抢像素把唇面打花
                    out["rgba"][:, 3] = out["rgba"][:, 3] * (1.0 - 0.7 * w)
                else:
                    out["rgba"][:, 3] = np.maximum(out["rgba"][:, 3], w * 0.6)
                g = FINISH_GLOSS.get(finish, 0.25) * w
                upd = g > gloss
                gloss[upd] = g[upd]
                shin[upd] = FINISH_SHIN.get(finish, 56.0)
            out["gloss"] = gloss
            out["shin"] = shin

            if densify:
                out = self._densify(out, paint_w)
        finally:
            core.set_texture_size(prev_tex)
        return out

    @staticmethod
    def _densify(cloud: dict[str, np.ndarray], paint_w: np.ndarray,
                 frac: float = 0.35) -> dict[str, np.ndarray]:
        """妆区密度增强：按 paint_w² 加权克隆子高斯（σ×0.55、位置微抖动），
        让高覆盖率区域（唇线/眼线/眉）的细节不被相邻大 σ splat 糊掉。"""
        n = len(cloud["xyz"])
        elig = np.nonzero(paint_w >= 0.2)[0]
        if len(elig) < 20:                       # 妆区几乎无有效 splat（贴合失败等）
            return cloud
        n_clone = min(int(n * frac), 40000)
        p = paint_w[elig] ** 2
        rows = np.random.default_rng(13).choice(elig, size=n_clone, p=p / p.sum())
        rng = np.random.default_rng(17)
        jit = rng.normal(0, 1, (n_clone, 3)) * (cloud["scale"][rows, :1] * 0.35)
        out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cloud.items()}
        out["xyz"] = np.concatenate([cloud["xyz"], cloud["xyz"][rows] + jit], axis=0)
        out["scale"] = np.concatenate([cloud["scale"], cloud["scale"][rows] * 0.55], axis=0)
        for k in ("rot", "rgba", "gloss", "shin"):
            if k in cloud:
                out[k] = np.concatenate([cloud[k], cloud[k][rows]], axis=0)
        return out

    # ---- L1：妆容壳层（独立薄高斯，叠加而非换色） ----

    SHELL_STROKE = {"eyeliner", "lashes", "eyebrow"}

    def _stroke_shell(self, pts: np.ndarray, Nw: np.ndarray, width: float,
                      rng: np.random.Generator, color: np.ndarray,
                      alpha: np.ndarray, gloss_v: float, shin_v: float,
                      out: dict[str, list]):
        """折线 → 沿线薄高斯：切向拉长、法向压扁（眼线/眉/唇线笔的锐度来源）。

        pts (k,3) 世界系折点；Nw (k,3) 折点法线；width 世界宽度（脸高=1 基准）。"""
        if len(pts) < 2:
            return
        seg = pts[1:] - pts[:-1]
        segl = np.linalg.norm(seg, axis=1)
        h = width * 0.55
        counts = np.maximum(np.ceil(segl / max(h, 1e-7)).astype(int), 1)
        P, Nn = [], []
        for c, (sv, ev), (nv, nw) in zip(counts, zip(pts[:-1], pts[1:]),
                                         zip(Nw[:-1], Nw[1:])):
            tt = (np.arange(c) + 0.5) / c
            P.append(sv + (ev - sv) * tt[:, None])
            Nn.append(nv + (nw - nv) * tt[:, None])
        P = np.concatenate(P)
        Nn = np.concatenate(Nn)
        Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12
        Tang = np.concatenate([np.broadcast_to(v / max(np.linalg.norm(v), 1e-9),
                                               (c, 3)) for c, v in zip(counts, seg)])
        m = len(P)
        sig_along = h * 0.8
        sig_across = width * 0.62
        sig_n = width * 0.12
        out["xyz"].append(P + Nn * (width * 0.35))      # 沿法线外偏避免深度混排
        out["scale"].append(np.broadcast_to([sig_along, sig_across, sig_n], (m, 3)))
        out["rot"].append(_frame_quats(Nn, Tang))
        out["rgba"].append(np.column_stack([
            np.clip(color * np.ones((m, 1)), 0, 1),
            np.clip(alpha * np.ones(m), 0, 1)]).astype(np.float64))
        out["gloss"].append(np.full(m, gloss_v * float(np.mean(alpha)), np.float32))
        out["shin"].append(np.full(m, shin_v, np.float32))

    def build_makeup_shell(self, cloud: dict[str, np.ndarray], layers: list[dict],
                           landmarks: np.ndarray, intensity: float = 0.8,
                           env: str = "neutral", light_dir: np.ndarray | None = None,
                           mouth: np.ndarray | None = None, seed: int = 23,
                           include_strokes: bool = True
                           ) -> tuple[dict[str, np.ndarray], list[dict]]:
        """L1 妆容壳层：锐利/镜面层（口红/眼线/睫毛/眉）不改写原 splat 颜色，
        而是在拟合几何上直接生成独立薄高斯叠加进点云——口红只长在唇红拓扑面上
        （张嘴的口腔/牙齿天然不在壳层里），换妆只需重生成壳层、脸云保持不动。

        软性大面积层（底妆/腮红/眼影等）仍走 apply_makeup 换色路径。
        include_strokes=False 时只生成口红壳层：脸云自带真实视频纹理（landmark
        稠密路径）时，眉/眼线已在纹理里，参数化描边受地标噪声影响会悬空穿帮。
        返回 (shell_cloud, ranges)；ranges 供 Unity/查看器按层开关。"""
        rng = np.random.default_rng(seed)
        s, R, t, _ = self.register(landmarks)
        Vw, band, _mouth_tris = self._world_lip_mesh(cloud, landmarks, s, R, t)
        topo = self.lip_topology()
        Nw = self.model.vertex_normals(Vw)
        out: dict[str, list] = {k: [] for k in
                                ("xyz", "scale", "rot", "rgba", "gloss", "shin")}
        ranges: list[dict] = []

        def push(layer_id, region, chunk0):
            start_splat = sum(len(c) for c in out["xyz"][:chunk0])
            ranges.append({"id": str(layer_id), "region": region,
                           "start": int(start_splat),
                           "count": int(sum(len(c) for c in out["xyz"][chunk0:]))})

        for layer in layers:
            if not layer.get("enabled", True):
                continue
            region = layer["region"]
            shape = layer.get("shape") or {}
            opacity = float(layer.get("opacity", 0.7)) * float(intensity)
            finish = layer.get("finish", "satin")
            gl, sh = FINISH_GLOSS.get(finish, 0.25), FINISH_SHIN.get(finish, 56.0)
            col_stops = layer.get("color_stops") or \
                [{"at": 0.0, "hex": "#B03040"}, {"at": 1.0, "hex": "#D04858"}]
            chunk0 = len(out["xyz"])

            if region == "lipstick":
                if len(band) == 0 or opacity <= 0:
                    continue
                edges = np.stack([np.linalg.norm(Vw[band[:, 1]] - Vw[band[:, 0]], axis=1),
                                  np.linalg.norm(Vw[band[:, 2]] - Vw[band[:, 0]], axis=1),
                                  np.linalg.norm(Vw[band[:, 2]] - Vw[band[:, 1]], axis=1)], 1)
                loc_face = ((topo["d_out"][band] + topo["d_in"][band]).mean(1)) * 0.5
                h = min(0.5 * float(np.median(edges)),
                        0.40 * float(np.median(loc_face)) * s)
                area = 0.5 * np.linalg.norm(
                    np.cross(Vw[band[:, 1]] - Vw[band[:, 0]],
                             Vw[band[:, 2]] - Vw[band[:, 0]]), axis=1).sum()
                if "uv" in cloud:
                    # 观测唇域锚定（landmark 稠密路径专属）：说话/噘嘴使三角化
                    # 唇带相对视频真实唇有 ~2% 脸高的系统性偏移，"贴"不住——壳层
                    # 直接长在颜色门控筛出的可见唇 splat 上，像素级贴合真实唇形。
                    # canonical/配准底座路径唇带即真实唇，带面采样更均匀锐利。
                    w_field, cent_f, aux = self._lip_band_weight(
                        cloud, landmarks, s, R, t, mouth=mouth, return_aux=True)
                    cand = np.nonzero(w_field > 0.15)[0]
                    if len(cand) < 24:                   # 防空壳层；真实数据约 1500+
                        continue
                    ww = w_field[cand].astype(np.float64)
                    n = int(min(max(len(cand) * 3, 1800), 9000))
                    pick = rng.choice(len(cand), size=n, p=ww / ww.sum())
                    idx = cand[pick]
                    xyz_c = np.asarray(cloud["xyz"], np.float64)
                    # σ 取观测唇点的局部邻间距（kNN），保证无缝又不出界
                    csub = xyz_c[cand].astype(np.float32)
                    flc = cv2.flann_Index(csub, dict(algorithm=1, trees=4, checks=64))
                    _nc, d2c = flc.knnSearch(csub, 4, params=dict(checks=64))
                    sig_c = np.clip(np.sqrt(np.maximum(d2c[:, 1:].mean(1), 0.0)) * 1.05,
                                    1e-5, None)
                    sig_t = sig_c[pick]
                    # 法线：云自带（landmark 网格路径的顶点法线）→ 兜底最近带样点法线
                    rot_c = np.asarray(cloud.get("rot"), np.float64)
                    if rot_c.shape == (len(xyz_c), 4) and \
                            float(np.abs(rot_c[:, 3] - 1.0).max()) > 1e-3:
                        Nn = _quat_normals(rot_c[idx])
                    else:
                        Nn = aux["band_n"][aux["ib"]][idx]
                    Nn = Nn / (np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12)
                    col = np.clip(self.core.sample_ramp(
                        col_stops, np.clip(cent_f[idx], 0, 1)), 0, 1)
                    shade = self._shade_for_normals(Nn, env, light_dir=light_dir,
                                                    floor=0.55)
                    a = np.clip(opacity * (0.7 + 0.3 * w_field[idx] ** 0.8) * 1.2, 0, 1)
                    out["xyz"].append(xyz_c[idx] + Nn * (0.9 * sig_t[:, None]))
                    out["scale"].append(np.stack([sig_t, sig_t, sig_t * 0.16], 1))
                    out["rot"].append(_quats_from_normals(Nn))
                    out["rgba"].append(np.column_stack(
                        [np.clip(col * shade, 0, 1), a]).astype(np.float64))
                    out["gloss"].append((gl * a).astype(np.float32))
                    out["shin"].append(np.full(n, sh, np.float32))
                    push(layer.get("id", region), region, chunk0)
                    continue
                # 带面采样（canonical/配准底座路径）
                n = int(np.clip(area / (h * h), 2600, 9000))
                P, bb, rows = _sample_tris(Vw, band, n, rng)
                verts = band[rows]
                do = (topo["d_out"][verts] * bb).sum(1)
                di = (topo["d_in"][verts] * bb).sum(1)
                loc = 0.5 * (do + di) + 1e-6
                edge = (np.clip(do / (0.28 * loc), 0, 1)
                        * np.clip(di / (0.06 * loc), 0, 1))
                Nn = (Nw[verts] * bb[..., None]).sum(1)
                Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12
                sig_t = h * 0.75
                col = np.clip(self.core.sample_ramp(
                    col_stops, np.clip(di / (do + di + 1e-6), 0, 1)), 0, 1)
                shade = self._shade_for_normals(Nn, env, light_dir=light_dir,
                                                floor=0.55)
                a = np.clip(opacity * edge ** 0.75 * 1.15, 0, 1)   # 盖住底层换色的逐点色噪
                out["xyz"].append(P + Nn * (0.45 * sig_t))
                out["scale"].append(np.broadcast_to([sig_t, sig_t, sig_t * 0.14], (n, 3)))
                out["rot"].append(_quats_from_normals(Nn))
                out["rgba"].append(np.column_stack(
                    [np.clip(col * shade, 0, 1), a]).astype(np.float64))
                out["gloss"].append((gl * a).astype(np.float32))
                out["shin"].append(np.full(n, sh, np.float32))
                # overline：沿外圈环补一条细唇线（稍外扩的唇峰）
                ov = min(max(float(shape.get("overline", 0.0) or 0.0), 0.0), 1.0)
                if ov > 0.01:
                    o_pts = Vw[topo["outer"]]
                    o_Nn = Nw[topo["outer"]]
                    self._stroke_shell(o_pts, o_Nn, (0.004 + ov * 0.012) * s, rng,
                                       col.mean(0), a.mean() * 0.25, gl, sh, out)
                push(layer.get("id", region), region, chunk0)

            elif region in self.SHELL_STROKE:
                if not include_strokes:
                    continue
                data = json.loads((_REFS / "landmark-regions.json")
                                  .read_text(encoding="utf-8"))["regions"]
                side = layer.get("side", "both")
                sides = ("left", "right") if side == "both" else (side,)
                # width 是"脸高=1"的 canonical 单位 → 世界系要乘配准尺度 s
                width = (0.006 if region == "lashes" else 0.0)
                width += float(shape.get("thickness", 0.25 if region != "eyebrow" else 0.4)) * \
                    (0.008 if region != "eyebrow" else 0.006)
                width = max(width, 0.0035) * s
                wing = float(shape.get("wing", 0.3) or 0.0)
                for one in sides:
                    key = f"{region if region != 'lashes' else 'eyeliner'}_{one}"
                    if key not in data:
                        continue
                    idx = np.array(data[key]["indices"], int)
                    self._stroke_shell(Vw[idx], Nw[idx], width, rng,
                                       self.core._hex(col_stops[-1]["hex"]),
                                       min(opacity, 0.95), gl, sh, out)
                    if region == "eyeliner" and wing > 0.01 and len(idx) >= 2:
                        # 外端 = 离面部中轴更远的端点；沿"外上"方向拉出眼尾
                        end_a, end_b = Vw[idx[0]], Vw[idx[-1]]
                        tail_i, prev_i = ((idx[0], idx[1]) if abs(end_a[0]) > abs(end_b[0])
                                          else (idx[-1], idx[-2]))
                        tail, prev = Vw[tail_i], Vw[prev_i]
                        d = tail - prev
                        d /= np.linalg.norm(d) + 1e-9
                        nrm = Nw[tail_i] / (np.linalg.norm(Nw[tail_i]) + 1e-9)
                        outv = d - nrm * float(d @ nrm)
                        outv /= np.linalg.norm(outv) + 1e-9
                        upv = np.cross(nrm, outv)
                        if upv[1] < 0:
                            upv = -upv
                        wdir = outv * 0.65 + upv * 0.35
                        wdir /= np.linalg.norm(wdir) + 1e-9
                        ext = (0.008 + 0.018 * wing) * s
                        self._stroke_shell(np.stack([tail, tail + wdir * ext]),
                                           np.stack([nrm, nrm]), width * 0.8, rng,
                                           self.core._hex(col_stops[-1]["hex"]),
                                           min(opacity, 0.95), gl, sh, out)
                push(layer.get("id", region), region, chunk0)

        if not out["xyz"]:
            empty = {k: np.zeros((0, 4 if k == "rgba" else 3),
                                 np.float32 if k != "rgba" else np.float64)
                     for k in ("xyz", "scale", "rot", "rgba", "gloss", "shin")}
            empty["gloss"] = np.zeros((0,), np.float32)
            empty["shin"] = np.zeros((0,), np.float32)
            return empty, ranges
        shell = {k: (np.concatenate(v, axis=0).astype(np.float64)
                     if k == "rgba" else np.concatenate(v, axis=0))
                 for k, v in out.items()}
        shell["rgba"] = np.asarray(shell["rgba"], np.float64)
        return shell, ranges

    @staticmethod
    def merge_shell(made: dict[str, np.ndarray], shell: dict[str, np.ndarray]
                    ) -> dict[str, np.ndarray]:
        """妆容壳层并入主点云（标准 6 键拼接；主云缺 gloss/shin 时补零）。"""
        n = len(made["xyz"])
        m = len(shell["xyz"])
        if m == 0:
            return {k: v.copy() for k, v in made.items()}
        out = {}
        for k in ("xyz", "scale", "rot", "rgba"):
            out[k] = np.concatenate([made[k], shell[k]], axis=0)
        for k in ("gloss", "shin"):
            a = made.get(k, np.zeros(n, np.float32))
            b = shell.get(k, np.zeros(m, np.float32))
            out[k] = np.concatenate([np.broadcast_to(a, (n,)).astype(np.float32)
                                     if np.ndim(a) == 0 else a, b])
        return out

    # ---- 主入口 ----

    def _export(self, made: dict, out_dir: Path, layers: list[dict], *,
                mode: str, n_views: int, rmse: float, scale: float,
                densified: int, shell: dict | None = None,
                ranges: list[dict] | None = None) -> tuple[Path, Path]:
        ply_path = out_dir / "madeup.ply"
        splat_path = out_dir / "madeup.splat"
        write_ply(made, ply_path)
        export_splat(made, splat_path)
        if shell is not None and len(shell["xyz"]):
            write_ply(shell, out_dir / "makeup_shell.ply")
            (out_dir / "shell_manifest.json").write_text(json.dumps({
                "face_splats": int(len(made["xyz"])) - int(len(shell["xyz"])),
                "shell_splats": int(len(shell["xyz"])),
                "ranges": ranges or [],
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        (out_dir / "fit_report.json").write_text(json.dumps({
            "mode": mode, "views_used": n_views, "fit_rmse_canonical": round(rmse, 4),
            "scale": round(float(scale), 4), "layers_applied": [l["id"] for l in layers],
            "splats": int(len(made["xyz"])), "densified": densified,
            "shell_splats": int(len(shell["xyz"])) if shell is not None else 0,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return ply_path, splat_path

    def fit(self, result: ReconResult, spec: dict, out_dir: str | Path,
            face_ply: str | Path | None = None, intensity: float = 0.8,
            env: str = "neutral", densify: bool = False) -> FitResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cloud = read_ply(face_ply or result.ply_path)
        landmarks, n_views = self.triangulate_landmarks(result)
        s, R, t, rmse = self.register(landmarks)
        layers = [dict(l) for l in spec.get("layers", []) if l.get("enabled", True)]
        made = self.apply_makeup(cloud, layers, landmarks, intensity=intensity, env=env,
                                 densify=densify)
        shell, ranges = self.build_makeup_shell(cloud, layers, landmarks,
                                                intensity=intensity, env=env)
        made_full = self.merge_shell(made, shell) if len(shell["xyz"]) else made

        ply_path, splat_path = self._export(made_full, out_dir, layers, mode="video",
                                            n_views=n_views, rmse=rmse, scale=s,
                                            densified=len(made["xyz"]) - len(cloud["xyz"]),
                                            shell=shell if len(shell["xyz"]) else None,
                                            ranges=ranges)
        previews = self.render_previews(cloud, made_full, result, out_dir)
        return FitResult(cloud=made_full, ply_path=ply_path, splat_path=splat_path,
                         landmarks_3d=landmarks, n_views_used=n_views,
                         fit_rmse=rmse, previews=previews)

    def fit_canonical(self, cloud: dict[str, np.ndarray], spec: dict, out_dir: str | Path,
                      intensity: float = 0.8, env: str = "neutral",
                      densify: bool = False) -> dict[str, np.ndarray]:
        """P2 入口：给已是 canonical 姿态的点云上妆（FLAME 底座适配器导出的
        canonical 点云，如 FlashAvatar），跳过视频三角化与配准。"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        layers = [dict(l) for l in spec.get("layers", []) if l.get("enabled", True)]
        made = self.apply_makeup(cloud, layers, self.model.base[:N_CANON_VERTS],
                                 intensity=intensity, env=env, densify=densify)
        shell, ranges = self.build_makeup_shell(cloud, layers,
                                                self.model.base[:N_CANON_VERTS],
                                                intensity=intensity, env=env)
        made_full = self.merge_shell(made, shell) if len(shell["xyz"]) else made
        self._export(made_full, out_dir, layers, mode="canonical", n_views=0, rmse=0.0,
                     scale=1.0, densified=len(made["xyz"]) - len(cloud["xyz"]),
                     shell=shell if len(shell["xyz"]) else None, ranges=ranges)
        return made_full

    def apply_guidance(self, cloud: dict[str, np.ndarray], layers: list[dict],
                       views: list[dict], landmarks: np.ndarray,
                       intensity: float = 0.8, env: str = "neutral",
                       shade: bool = True) -> dict[str, np.ndarray]:
        """P1 入口：妆容外观由 guidance 图多视角投影采样（AvatarMakeup 思路的简化版）。

        views: [{"R"(3x3), "t"(3,), "cam"(Camera), "img"(BGR uint8)}...]，稳定妆容
        迁移模型（如 Stable-Makeup）对素颜渲染图的输出。几何归属仍走 canonical 蒙版
        （参数化 spec 降级为区域门控 + 强度滑杆），颜色取各视角投影采样的逐通道中位数
        （对 guidance 的局部瑕疵/视角闪烁鲁棒），采样失败的 splat 回退参数化路径。
        shade=False 时不乘 wrap-diffuse（guidance 已是带光照的真实纹理时用）。"""
        core = self.core
        s, R, t, _ = self.register(landmarks)
        prev_tex = core.TEX
        core.set_texture_size(FIT_TEX)
        try:
            if "uv" in cloud:
                uv = np.asarray(cloud["uv"], np.float64)[:, :2]
                valid = np.isfinite(uv).all(1)
            else:
                uv, _near, valid = self._splat_uv(cloud, s, R, t)
            shade_f = (self._wrap_shade(cloud, env) if shade
                       else np.ones((len(cloud["xyz"]), 1), np.float64))
            xyz = np.asarray(cloud["xyz"], np.float64)
            n = len(xyz)
            TEX = core.TEX
            gate_map = np.zeros((TEX, TEX), np.float32)
            for layer in layers:
                if not layer.get("enabled", True):
                    continue
                mask = self.regions.bake(layer)
                gate_map = np.maximum(gate_map, mask[..., 0])
            tx = np.clip(uv[:, 0] * (TEX - 1), 0, TEX - 1.001)
            ty = np.clip((1 - uv[:, 1]) * (TEX - 1), 0, TEX - 1.001)
            gate = core._bilinear(gate_map[..., None], tx, ty)[..., 0]
            peak = float(gate.max())
            if peak > 1e-4:
                gate = np.clip(gate / peak, 0, 1)
            gate[~valid] = 0.0

            # 多视角投影采样（splat 世界坐标直接投影到各 guidance 相机）→ 逐通道中位数
            samples = []
            for v in views:
                cam = v.get("cam") or _first_cam(views)
                Rv = np.asarray(v["R"], np.float64)
                tv = np.asarray(v["t"], np.float64)
                cam_xyz = (xyz @ Rv.T) + tv
                z = cam_xyz[:, 2]
                f = float(cam.params[0])
                px = cam.params[1] + cam_xyz[:, 0] * f / np.maximum(z, 1e-6)
                py = cam.params[2] + cam_xyz[:, 1] * f / np.maximum(z, 1e-6)
                img = np.asarray(v["img"], np.float32) / 255.0
                h, w_ = img.shape[:2]
                ok = (z > 0.05) & (px >= 0) & (px < w_ - 1) & (py >= 0) & (py < h - 1)
                x0 = np.clip(px, 0, w_ - 1.001).astype(np.int32)
                y0 = np.clip(py, 0, h - 1.001).astype(np.int32)
                fx = (px - x0)[..., None]
                fy = (py - y0)[..., None]
                rgb = (img[y0, x0, ::-1] * (1 - fx) * (1 - fy)
                       + img[y0, x0 + 1, ::-1] * fx * (1 - fy)
                       + img[y0 + 1, x0, ::-1] * (1 - fx) * fy
                       + img[y0 + 1, x0 + 1, ::-1] * fx * fy)
                samples.append(np.where(ok[:, None], rgb, np.nan))
            stack = np.stack(samples, axis=0) if samples else np.full((1, n, 3), np.nan, np.float32)
            with np.errstate(invalid="ignore"):
                guids = np.nanmedian(stack, axis=0)
            have = np.isfinite(guids).all(1) & (gate > 0.02)
            guids = np.where(np.isfinite(guids), guids, 0.5)

            out = {k: v.copy() for k, v in cloud.items()}
            w = np.clip(gate * float(intensity), 0, 1)
            wc = (w * have)[..., None]
            out["rgba"][:, :3] = out["rgba"][:, :3] * (1 - wc) + np.clip(guids, 0, 1) * shade_f * wc
            out["rgba"][:, 3] = np.maximum(out["rgba"][:, 3], (w * have) * 0.6)
            finish_top = max((FINISH_GLOSS.get(l.get("finish", "satin"), 0.25)
                              for l in layers if l.get("enabled", True)), default=0.0)
            out["gloss"] = finish_top * gate * have
            out["shin"] = np.full(n, 120.0, np.float32)
        finally:
            core.set_texture_size(prev_tex)
        return out

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
                 cam: colmap_io.Camera, w: int = 512, h: int = 384,
                 env: str = "neutral", light_dir: np.ndarray | None = None,
                 spec_strength: float = 0.5) -> np.ndarray:
    """通用前向泼溅（圆核、后→前合成），用于用户点云的快速预览。

    cloud 带 gloss/shin（apply_makeup 产物）时叠加逐点 Blinn-Phong 镜面。"""
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
    spec = blinn_phong_spec(cloud, R, env=env, light_dir=light_dir,
                            strength=spec_strength)

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
        sl[:] = sl * (1 - g[..., None]) + (rgba[i, :3] + spec[i]) * g[..., None]
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)[..., ::-1]
