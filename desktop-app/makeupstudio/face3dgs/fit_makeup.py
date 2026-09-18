"""fit_makeup — 脸部语义基础设施（写实链路 appearance/ 的几何/语义工具层）。

产品链路（上传/扫描 → SfM → gsplat 训练 → UV 妆容）中，本模块只承担与
"语义"相关、与渲染无关的部分：
    · 三角化/配准：DLT（Hartley 归一 + cheirality）与带尺度 Kabsch——
      真实地标 → canonical/splat 世界系；
    · 唇拓扑：inner/outer 环洪泛出唇红带与口腔开口面（拓扑上不可能越唇缘）；
    · 3D 唇带权重：观测唇域锚定（真实地标顶点）+ 颜色门控——唇妆的
      空间真相来源（appearance.makeup_uv.lip_band_3d 消费）；
    · canonical UV 归属：kNN 距离加权 + 跨 UV 岛保护 + 离群剔除
      （appearance.uvbind 消费）；
    · RegionMasks 区域蒙版 / canonical 模板 / 渲染内核加载。
旧链路（模板染色 apply_makeup、程序化壳层 build_makeup_shell、fit/fit_canonical/
apply_guidance 旧入口、render_cloud 旧渲染器）已随 R 写实化重构退役：
底模训练见 appearance/train_base.py，妆容合成见 appearance/makeup_uv.py，
可微外观优化见 appearance/optimize.py。
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from . import colmap_io

_REFS = Path(__file__).resolve().parent.parent.parent.parent / "makeup-skill" / "references"
N_CANON_VERTS = 467 + 1       # canonical 模型与 FaceTracker 468 地标一一对应
KNN_K = 4                     # UV 归属的近邻数（距离加权，抑制离散采样噪声）


def _load_core():
    import importlib.util
    if "preview_render_core" not in globals():
        spec = importlib.util.spec_from_file_location(
            "preview_render_core", _REFS.parent / "scripts" / "preview_render.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        globals()["preview_render_core"] = mod
    return globals()["preview_render_core"]


# ---------------- 几何工具 ----------------

def projection_matrix(cam: colmap_io.Camera, qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R = colmap_io.quat_to_rotmat(qvec)
    K = np.array([[cam.params[0], 0, cam.params[1]],
                  [0, cam.params[0], cam.params[2]],
                  [0, 0, 1]])
    return K @ np.hstack([R, tvec.reshape(3, 1)])


def _quat_normals(rot: np.ndarray) -> np.ndarray:
    """四元数 (xyzw) 把 (0,0,1) 旋到哪儿 = splat 薄片法线。"""
    qx, qy, qz, qw = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    nx = 2 * (qx * qz + qy * qw)
    ny = 2 * (qy * qz - qx * qw)
    nz = 1 - 2 * (qx * qx + qy * qy)
    Nn = np.stack([nx, ny, nz], axis=1)
    Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12
    return Nn


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


# ---------------- 语义贴合器（三角化/配准/唇拓扑/UV 归属） ----------------

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

    # ---- 多帧地标三角化 ----

    def triangulate_landmarks(self, result, tracker=None,
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

    # ---- canonical → splat 世界系配准 ----

    def register(self, landmarks: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
        canon = self.model.base[:N_CANON_VERTS]
        valid = np.linalg.norm(landmarks, axis=1) > 0
        s, R, t = kabsch_similarity(canon[valid], landmarks[valid])
        err = canon[valid] @ (s * R).T + t - landmarks[valid]
        rmse = float(np.sqrt((np.linalg.norm(err, axis=1) ** 2).mean()))
        return s, R, t, rmse

    # ---- 唇部拓扑与 3D 带状权重 ----

    def lip_topology(self) -> dict:
        """canonical 网格的唇部分类（缓存，唇妆 3D 锚定与 UV 光栅共用）。

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
        return_aux=True 时额外返回 {"ib","band_n"}（最近带样点索引与其法线）。
        """
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

    # ---- canonical UV 归属 ----

    def _surface_samples(self, s: float, R: np.ndarray, t: np.ndarray):
        """canonical 表面均匀采样（面积加权 + 重心坐标）→ splat 世界系 (xyz, uv)。"""
        Vc = self.model.base
        tris = self.model.tris
        p0, p1, p2 = Vc[tris[:, 0]], Vc[tris[:, 1]], Vc[tris[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        area = np.maximum(area, 1e-12)
        rows = np.random.default_rng(7).choice(len(tris), size=24000, p=area / area.sum())
        rng = np.random.default_rng(11)
        r1, r2 = rng.random(24000), rng.random(24000)
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
