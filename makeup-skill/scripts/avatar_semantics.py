#!/usr/bin/env python3
"""avatar_semantics — 给 3DGS 画像的每个 Gaussian 标注妆容 region-ID。

画像是任意来源的高斯点云，本身没有"嘴唇/眼睑"语义。本模块用现有 landmark
语义表（landmark-regions.json + canonical_face_model.obj 的 468 点序）给画像
打上 region 标签，供妆容编译器按部位着色：

    0 none | 1 foundation | 2 concealer | 3 contour | 4 eyebrow | 5 eyeshadow
    6 eyeliner | 7 blush | 8 highlight | 9 lipstick

锚点来源三选一：
    mode="landmarks"  正脸渲染一帧 → MediaPipe FaceMesh 468 点 → 沿视线用
                      深度图反投影回画像 3D 锚点（自动，需 mediapipe；模型
                      首次运行自动下载）
    mode="manual"     用户/测试直接给 {region: [[x,y,z],...]} 锚点（归一化空间）
    mode="cache"      复用已保存的 semantics.bin

区域归属 = 高斯点到各 region 锚点的"相对距离"（d / 半径剖面）最小者，
超过 1.0 归为 none。半径剖面 REGION_PROFILES 以脸高为单位，与 preview_render
的 UV 蒙版扩散量同源标定（eyeshadow 向上外扩、blush 大半径、eyeliner 紧贴）。

产出：region_ids (N,) uint8 + confidence (N,) float32；save_semantics 存 MKSEM1。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REFS = Path(__file__).resolve().parent.parent.parent / "makeup-skill" / "references"

REGION_IDS = {"none": 0, "foundation": 1, "concealer": 2, "contour": 3, "eyebrow": 4,
              "eyeshadow": 5, "eyeliner": 6, "blush": 7, "highlight": 8, "lipstick": 9}
ID_REGIONS = {v: k for k, v in REGION_IDS.items()}

# region → (基准组, 半径剖面)。半径单位=脸高；offset 是各锚点的法向附加外扩。
# z_min：认领高斯的 z 下限（归一化空间脸朝 +Z）——防止大半径 region（粉底/腮红）
# 越过脸侧轮廓染到头发/耳侧高斯（渲染上表现为轮廓亮圈）。
REGION_PROFILES: dict[str, dict] = {
    "foundation": {"groups": ["foundation_face_oval", "cheekbone_left", "cheekbone_right"],
                   "radius": 0.30, "offset": 0.0, "z_min": 0.10},
    "concealer":  {"groups": ["lower_lid_left", "lower_lid_right"],
                   "radius": 0.075, "offset": -0.008, "z_min": 0.02},
    "contour":    {"groups": ["contour_forehead", "contour_jaw_left", "contour_jaw_right",
                              "contour_nose"],
                   "radius": 0.085, "offset": 0.0, "z_min": 0.04},
    "eyebrow":    {"groups": ["eyebrow_left", "eyebrow_right"],
                   "radius": 0.045, "offset": 0.0, "z_min": 0.02},
    "eyeshadow":  {"groups": ["eyelid_left", "eyelid_right"],
                   "radius": 0.075, "offset": 0.006, "z_min": 0.02},
    "eyeliner":   {"groups": ["eyeliner_left", "eyeliner_right"],
                   "radius": 0.022, "offset": 0.002, "z_min": 0.02},
    "blush":      {"groups": ["blush_left", "blush_right", "cheekbone_left", "cheekbone_right"],
                   "radius": 0.16, "offset": 0.004, "z_min": 0.10},
    "highlight":  {"groups": ["highlight_cheek_left", "highlight_cheek_right",
                              "highlight_nose", "highlight_cupid"],
                   "radius": 0.045, "offset": 0.003, "z_min": 0.05},
    "lipstick":   {"groups": ["lips_outer", "lips_inner"],
                   "radius": 0.045, "offset": 0.002, "z_min": 0.02},
}


# ---------------- 锚点获取 ----------------

def unproject_landmarks(image: np.ndarray, depth: np.ndarray,
                        landmarks_px: np.ndarray, focal_px: float,
                        cam_dist: float) -> np.ndarray:
    """渲染相机模型下，2D 关键点沿视线与深度图求交 → (K,3) 归一化空间点。

    深度图/图像尺寸同源；landmarks_px 为 (K,2) 像素坐标（图像 y 向下）。
    """
    h, w = depth.shape[:2]
    pts = []
    for x, y in landmarks_px:
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        # 3×3 中值深度抗光栅毛刺
        patch = depth[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2]
        zt = float(np.median(patch[patch > 0])) if (patch > 0).any() else cam_dist
        dir3 = np.array([(xi - w / 2) / focal_px, -(yi - h / 2) / focal_px, 1.0])
        dir3 /= np.linalg.norm(dir3)
        pts.append(dir3 * zt)
    return np.asarray(pts, np.float32)


def detect_landmark_anchors(av, render_fn) -> dict[str, np.ndarray]:
    """landmarks 模式：render_fn(image_bgr, depth) → (468×2 像素点)。

    render_fn 由 avatar_render 提供（正脸一帧 + 深度图）；本函数负责 MediaPipe
    检测与反投影，region 模板形状对齐（procrustes 平移缩放）不必要——反投影直接
    得到画像空间点。mediapipe 缺失/未检出时抛 RuntimeError，由上层回退 manual。
    """
    import mediapipe as mp                                   # 延迟导入（测试环境不装）
    image, depth, focal_px, cam_dist = render_fn()
    rgb = np.ascontiguousarray(image[..., ::-1])
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         refine_landmarks=False) as fm:
        res = fm.process(rgb)
    if not res.multi_face_landmarks:
        raise RuntimeError("画像正脸帧未检出人脸关键点（检查画像朝向/光照）")
    lms = res.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]
    px = np.array([[p.x * w, p.y * h] for p in lms], np.float32)
    P = unproject_landmarks(image, depth, px, focal_px, cam_dist)
    return _anchors_from_canonical(P)


def _anchors_from_canonical(P468: np.ndarray) -> dict[str, np.ndarray]:
    """反投影出的 468 点 → 各 region 锚点（用 region 组的 canonical 索引取子集）。"""
    regions_json = REFS / "landmark-regions.json"
    data = json.loads(regions_json.read_text(encoding="utf-8"))["regions"]

    def group_pts(group: str) -> np.ndarray:
        info = data.get(group) or {}
        idx = info.get("indices") or ([info["center_anchor"]] if info.get("center_anchor") else [])
        return P468[np.asarray(idx, np.int64)]

    out = {}
    for region, prof in REGION_PROFILES.items():
        pts = [group_pts(g) for g in prof["groups"]]
        out[region] = np.vstack([p for p in pts if len(p)])
    return out


# ---------------- 法线估计（局部 PCA，锚点贴表面用） ----------------

class SurfaceProbe:
    """体素网格近邻查询 + 局部平面拟合。画像规模 ≤30 万点、锚点 ≤500 时亚秒。"""

    def __init__(self, means: np.ndarray, cell: float = 0.03):
        self.means = means
        self.cell = float(cell)
        keys = np.floor(means / self.cell).astype(np.int64)
        self.grid: dict[tuple, np.ndarray] = {}
        uk, inv = np.unique(keys, axis=0, return_inverse=True)
        order = np.argsort(inv)
        counts = np.bincount(inv)
        starts = np.concatenate([[0], np.cumsum(counts)])
        for i, k in enumerate(map(tuple, uk)):
            self.grid[k] = order[starts[i]:starts[i + 1]]

    def neighbors(self, p: np.ndarray, radius: float) -> np.ndarray:
        c = np.floor(p / self.cell).astype(np.int64)
        r = int(np.ceil(radius / self.cell))
        idxs = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    idxs.append(self.grid.get((c[0] + dx, c[1] + dy, c[2] + dz)))
        idxs = [i for i in idxs if i is not None and len(i)]
        if not idxs:
            return np.empty((0,), np.int64)
        sel = np.concatenate(idxs)
        pts = self.means[sel]
        return sel[np.linalg.norm(pts - p, axis=1) <= radius]

    def plane_at(self, p: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """返回 (中心, 单位法线)。近邻 < 6 个时退化为 +Z。"""
        idx = self.neighbors(p, radius)
        if len(idx) < 6:
            return p, np.array([0.0, 0.0, 1.0])
        q = self.means[idx] - self.means[idx].mean(0)
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        n = vt[-1]
        if n[2] < 0:
            n = -n                                   # 朝向相机一侧（+Z 为正脸方向）
        return self.means[idx].mean(0), n / (np.linalg.norm(n) + 1e-9)


# ---------------- 主入口 ----------------

def assign_regions(av, anchors: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """按相对距离给每个 Gaussian 分配 region。返回 (ids (N,) u8, conf (N,) f32)。

    关键：小半径 region（唇/眼线等，语义更具体）优先认领并冻结——否则大半径
    region（粉底 0.30）会按相对距离抢走小 region 的边缘高斯，导致唇色只上在
    中心一小块。认领阈值 1.0，冻结阈值 0.95。
    """
    n = av.n
    ids = np.zeros(n, np.uint8)
    conf = np.zeros(n, np.float32)
    best_ratio = np.full(n, np.inf, np.float32)

    means = av.means
    face_pts = np.concatenate(list(anchors.values()))
    lo = face_pts.min(0) - 0.35
    hi = face_pts.max(0) + 0.35
    cand = np.nonzero(np.all((means >= lo) & (means <= hi), axis=1))[0]
    if len(cand) == 0:
        return ids, conf

    def radius_of(region: str) -> float:
        prof = REGION_PROFILES[region]
        return float(prof["radius"]) + float(prof.get("offset", 0.0))

    free = np.ones(len(cand), bool)
    order = sorted(anchors, key=radius_of)                       # 小半径优先

    # 脸型椭圆约束：foundation 锚点环绕脸部一周 → 拟合外接椭圆（+6% 余量），
    # 大半径 region（粉底/腮红/轮廓）只认领椭圆内的高斯——否则会把发际线外、
    # 鬓角/头顶的头发染上底妆（渲染为轮廓亮圈）。小 region 锚点本身贴特征，无需约束。
    oval = None
    if "foundation" in anchors:
        fp = anchors["foundation"]
        cx, cy = (fp[:, 0].min() + fp[:, 0].max()) / 2, (fp[:, 1].min() + fp[:, 1].max()) / 2
        rx = max((fp[:, 0].max() - fp[:, 0].min()) / 2, 0.12) * 1.06
        ry = max((fp[:, 1].max() - fp[:, 1].min()) / 2, 0.12) * 1.06
        oval = (cx, cy, rx, ry)

    for region in order:
        prof = REGION_PROFILES[region]
        pts = anchors[region]
        radius = radius_of(region)
        z_min = float(prof.get("z_min", 0.0))
        eligible = free & (means[cand, 2] >= z_min)
        if oval is not None and radius >= 0.12:
            cx, cy, rx, ry = oval
            px, py = means[cand, 0], means[cand, 1]
            eligible &= ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2 <= 1.0
        free_idx = cand[eligible]
        if len(free_idx) == 0:
            continue
        sub = means[free_idx]
        ratio = np.full(len(sub), np.inf, np.float32)
        rid = np.full(len(sub), 0, np.uint8)
        cf = np.zeros(len(sub), np.float32)
        for p in pts:
            d = np.linalg.norm(sub - p[None, :], axis=1)
            r = d / radius
            upd = r < ratio
            ratio[upd] = r[upd]
            cf[upd] = np.clip(1.0 - r[upd], 0.0, 1.0)
        hit = ratio <= 1.0
        ids[free_idx[hit]] = REGION_IDS[region]
        conf[free_idx[hit]] = cf[hit]
        best_ratio[free_idx[hit]] = ratio[hit]
        frozen = free_idx[ratio < 0.95]                          # 具体区域认领后不再让渡
        fm = np.isin(cand, frozen)
        free &= ~fm
    return ids, conf


def annotate(av, mode: str = "landmarks", manual_anchors: dict[str, list] | None = None,
             render_fn=None, cache_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """统一入口。manual_anchors: {region: [[x,y,z],...]}（归一化空间）。"""
    if mode == "cache" and cache_path and Path(cache_path).exists():
        return load_cached(cache_path)
    if mode == "manual":
        if not manual_anchors:
            raise ValueError("manual 模式需要 manual_anchors")
        anchors = {r: np.asarray(v, np.float32) for r, v in manual_anchors.items()}
    elif mode == "landmarks":
        anchors = detect_landmark_anchors(av, render_fn)
    else:
        raise ValueError(f"未知语义模式：{mode}")
    ids, conf = assign_regions(av, anchors)
    if cache_path:
        save_semantics_cache(cache_path, ids, conf)
    return ids, conf


def save_semantics_cache(path: str | Path, ids: np.ndarray, conf: np.ndarray) -> None:
    from avatar_io import save_semantics
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_semantics(path, ids, conf)


def load_cached(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    from avatar_io import load_semantics
    ids, conf = load_semantics(path)
    return ids, (conf if conf is not None else np.zeros(len(ids), np.float32))


def region_anchors_3d(av, ids: np.ndarray, region: str,
                      probe: SurfaceProbe | None = None) -> tuple[np.ndarray, np.ndarray]:
    """某 region 的高斯位置与法线（编译器放溅射/算向心度用）。法线用局部 PCA。"""
    probe = probe or SurfaceProbe(av.means)
    idx = np.nonzero(ids == REGION_IDS[region])[0]
    if len(idx) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    centers = np.zeros((len(idx), 3), np.float32)
    normals = np.zeros((len(idx), 3), np.float32)
    normals[:] = np.array([0.0, 0.0, 1.0])
    for k, i in enumerate(idx):
        c, n = probe.plane_at(av.means[i], radius=float(av.scales[i].max() * 6) + 0.02)
        centers[k] = av.means[i]
        normals[k] = n
    return centers, normals
