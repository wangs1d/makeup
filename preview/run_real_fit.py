#!/usr/bin/env python3
"""run_real_fit — 真实视频 → COLMAP SfM → 脸部隔离 → 地标三角化 → 上妆 → 对比图。

d6.mp4（真人、有头部转动）抽帧后已由 pycolmap 完成增量 SfM
（out/real/sfm/sparse/<k>，取最大子模型）。本脚本把 COLMAP 模型接进
fit_makeup 管线（与 App 内 COLMAP+Brush 产物同一契约）：
    points3D → final.ply → isolate_face → FaceMakeupFitter.fit(densify=True)
    → 素颜/妆容多视角对比图 out/compare/compare_real_fit.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "desktop-app"))
sys.path.insert(0, str(ROOT / "makeup-skill" / "scripts"))

import pycolmap  # noqa: E402

from makeupstudio.face3dgs import colmap_io  # noqa: E402
from makeupstudio.face3dgs.fit_makeup import FaceMakeupFitter, render_cloud  # noqa: E402
from makeupstudio.face3dgs.isolate import isolate_face  # noqa: E402
from makeupstudio.face3dgs.reconstruct import ReconResult  # noqa: E402
from makeupstudio.face3dgs.splat_io import read_ply, write_ply  # noqa: E402

ROOT_P = ROOT
SFMS = sorted((ROOT_P / "out/real/sfm/sparse").glob("[0-9]*"))
PROJ = ROOT_P / "out/real/project"
OUT_CMP = ROOT_P / "out/compare"
PRESET = json.loads((ROOT_P / "makeup-skill/presets/date-rose.json").read_text(encoding="utf-8"))


def pick_model() -> Path:
    best, best_n = None, -1
    for d in SFMS:
        rec = pycolmap.Reconstruction(d)
        n = rec.num_points3D()
        if rec.num_reg_images() >= 5 and n > best_n:
            best, best_n = d, n
    if best is None:
        raise RuntimeError("SfM 无可用子模型")
    print(f"SfM 子模型: {best.name}（{best_n} 点）")
    return best


def knn_mean_dist(xyz: np.ndarray, k: int = 4) -> np.ndarray:
    import cv2
    flann = cv2.flann_Index(xyz.astype(np.float32),
                            dict(algorithm=1, trees=4, checks=128))
    _nn, dist = flann.knnSearch(xyz, k + 1, params=dict(checks=128))
    dist = np.sqrt(np.maximum(dist, 0.0))[:, 1:]        # 去掉自身
    return dist.mean(1)


def _quats_from_normals(n: np.ndarray) -> np.ndarray:
    """(0,0,1)→n 的四元数 (w,x,y,z)，向量化。"""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(np.tile(z, (len(n), 1)), n)
    s = np.linalg.norm(v, axis=1)
    a = np.arctan2(s, n[:, 2])
    axis = v / (s[:, None] + 1e-12)
    half = a / 2
    return np.stack([np.cos(half), axis[:, 0] * np.sin(half),
                     axis[:, 1] * np.sin(half), axis[:, 2] * np.sin(half)], 1).astype(np.float32)


def dense_cloud_from_landmarks(fitter: FaceMakeupFitter, L: np.ndarray,
                               n_samples: int = 48000) -> tuple[dict, np.ndarray]:
    """真实地标 = canonical 网格顶点（468 一一对应）→ 真实脸形面片 → 面积加权
    稠密采样（位置/法线/UV 由重心坐标插值），σ 取 kNN 自适应。
    三角化失败的无效地标为 (0,0,0)，必须连同其三角形一并剔除（否则巨型三角形
    会吞掉全部采样点）。
    返回 (dense, mouth_mask)：mouth_mask 标记落在口腔开口面（lips_inner 环内
    三角片，张嘴时被拉伸）上的采样点——口红/壳层永不涂这些点。"""
    core = fitter.core
    valid = np.linalg.norm(L, axis=1) > 0
    V = L.astype(np.float64)
    tris = fitter.model.tris
    keep_orig = valid[tris].all(1)
    tris = tris[keep_orig]
    topo = fitter.lip_topology()
    is_mouth = np.zeros(len(fitter.model.tris), bool)
    is_mouth[topo["mouth"]] = True
    mouth_tri = is_mouth[keep_orig]              # 过滤后数组 → 原始三角形是否口腔
    print(f"    有效地标 {int(valid.sum())}/468，参与构网三角形 {len(tris)}/{len(fitter.model.tris)}")
    p0, p1, p2 = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    area = np.maximum(area, 1e-18)
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    # 顶点法线（累加面法线）
    N = np.zeros_like(V)
    for kk in range(3):
        np.add.at(N, tris[:, kk], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12

    rng = np.random.default_rng(7)
    rows = rng.choice(len(tris), size=n_samples, p=area / area.sum())
    b = rng.dirichlet([1, 1, 1], size=n_samples)
    P = p0[rows] * b[:, :1] + p1[rows] * b[:, 1:2] + p2[rows] * b[:, 2:]
    Nn = N[tris[rows, 0]] * b[:, :1] + N[tris[rows, 1]] * b[:, 1:2] + N[tris[rows, 2]] * b[:, 2:]
    Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12
    uv = (fitter.model.uvs[tris[rows, 0]] * b[:, :1]
          + fitter.model.uvs[tris[rows, 1]] * b[:, 1:2]
          + fitter.model.uvs[tris[rows, 2]] * b[:, 2:])
    sig = np.clip(knn_mean_dist(P.astype(np.float32)) * 1.15, 1e-6, None).astype(np.float32)
    dense = {"xyz": P.astype(np.float32), "scale": np.stack([sig] * 3, 1),
             "rot": _quats_from_normals(Nn),
             "rgba": np.full((n_samples, 4), 0.7, np.float32),
             "uv": uv.astype(np.float32)}
    return dense, mouth_tri[rows]


def collect_views(model_dir: Path, images_dir: Path, max_views: int = 40) -> list[dict]:
    model = colmap_io.read_sparse(model_dir)
    cam = model.camera
    names = sorted(set(model.images) & {p.name for p in Path(images_dir).glob("*.jpg")})
    step = max(1, len(names) // max_views)
    views = []
    for name in names[::step]:
        im = model.images[name]
        img = cv2.imread(str(Path(images_dir) / name))
        if img is None:
            continue
        views.append({"R": colmap_io.quat_to_rotmat(im["qvec"]), "t": im["tvec"],
                      "cam": cam, "img": img})
    return views


def paint_bare_texture(fitter: FaceMakeupFitter, dense: dict, views: list[dict],
                       L: np.ndarray) -> None:
    """多视角投影采样真实视频肤色（逐通道中位数），门控 = 全脸特征区并集
    （底妆椭圆+唇+眉+眼影+遮瑕+高光+修容），采样即"素颜纹理"。"""
    gate_layers = [{"id": f"gate-{r}", "region": r, "enabled": True,
                    "opacity": 1.0, "finish": "matte",
                    "color_stops": [{"at": 0.0, "hex": "#888888"},
                                    {"at": 1.0, "hex": "#888888"}]}
                   for r in ("foundation", "lipstick", "eyebrow", "eyeshadow",
                             "concealer", "highlight", "contour")]
    tex = fitter.apply_guidance(dense, gate_layers, views, L,
                                intensity=1.0, shade=False)
    dense["rgba"] = tex["rgba"]


def main() -> None:
    model_dir = pick_model()
    rec = pycolmap.Reconstruction(model_dir)
    rec.write(model_dir)                       # 落盘 cameras.bin/images.bin（colmap_io 契约）

    # ---- 点云 ----
    xyz = np.array([p.xyz for p in rec.points3D.values()], np.float32)
    rgb = np.array([p.color for p in rec.points3D.values()], np.float32) / 255.0
    n = len(xyz)
    sig = np.clip(knn_mean_dist(xyz) * 1.4, 1e-4, None).astype(np.float32)
    cloud = {"xyz": xyz, "scale": np.stack([sig] * 3, 1),
             "rot": np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32),
             "rgba": np.column_stack([np.clip(rgb, 0, 1), np.full(n, 0.95, np.float32)])}
    PROJ.mkdir(parents=True, exist_ok=True)
    write_ply(cloud, PROJ / "final.ply")

    result = ReconResult(project_dir=PROJ, ply_path=PROJ / "final.ply",
                         sparse_dir=model_dir,
                         images_dir=ROOT_P / "out/real/capture/frames", seconds=0.0)
    stats = isolate_face(result, PROJ / "face.ply",
                         bbox_provider=lambda frame: (0.05, 0.05, 0.95, 0.95),
                         max_views=30)
    face = read_ply(PROJ / "face.ply")
    print(f"隔离: {stats} | 脸部点 {len(face['xyz'])}/{n}")

    # ---- 上妆（真实地标三角化 + densify + 镜面烘焙）----
    fitter = FaceMakeupFitter()
    fit = fitter.fit(result, PRESET, ROOT_P / "out/real/fitted",
                     face_ply=PROJ / "face.ply", intensity=0.85, densify=True)
    report = json.loads((ROOT_P / "out/real/fitted/fit_report.json").read_text(encoding="utf-8"))
    print("fit_report:", report)

    # ---- 对比图：按 tracker 偏航挑 3 个视角（近正脸 / 左 15° / 右 15°）----
    import preview_render  # noqa: F401
    from makeupstudio.tracker import FaceTracker
    model = colmap_io.read_sparse(model_dir, dialect="standard")
    cam = model.camera
    tracker = FaceTracker(smooth=False)
    yaw_of = {}
    t_ms = 0.0
    for name in sorted(model.images):
        p = result.images_dir / name
        img = cv2.imread(str(p))
        if img is None:
            continue
        det = tracker.detect(img, t_ms)
        t_ms += 1000.0 / 25.0
        if det is not None:
            yaw_of[name] = det["pose"][0]
    tracker.close()

    made_sparse = fit.cloud
    bare_sparse = read_ply(PROJ / "face.ply")
    L = fit.landmarks_3d                     # (468,3) 真实地标（SfM 世界系）

    # ---- 稠密重建：canonical 拓扑 × 真实地标顶点 ----
    # MediaPipe canonical 网格的顶点即 468 地标（拓扑一一对应）。把三角化的真实
    # 地标直接作为网格顶点 → 这个人真实脸形的稠密面片；再按面积采样成稠密 splat，
    # 肤色由多视角投影取逐通道中位数（真实视频纹理），最后走完整妆容管线。
    dense, mouth = dense_cloud_from_landmarks(fitter, L)
    views = collect_views(model_dir, result.images_dir)
    paint_bare_texture(fitter, dense, views, L)
    write_ply(dense, PROJ / "face_dense.ply")
    np.save(PROJ / "landmarks.npy", L)
    # 顺光（surface→light 约定）：光源取采集相机质心方向——SfM 世界朝向任意，
    # canonical ENVS 预设光不一定照着脸
    centroid = np.asarray(L).mean(0)
    cam_centers = np.array([-np.asarray(v["R"]).T @ np.asarray(v["t"]) for v in views])
    light_dir = cam_centers.mean(0) - centroid
    light_dir = (light_dir / (np.linalg.norm(light_dir) + 1e-9)).astype(np.float64)
    made = fitter.apply_makeup(dense, PRESET["layers"], L,
                               intensity=0.8, densify=True, treatment="lab",
                               mouth=mouth)
    shell, ranges = fitter.build_makeup_shell(dense, PRESET["layers"], L,
                                              intensity=0.8, light_dir=light_dir,
                                              mouth=mouth, include_strokes=False)
    print("壳层分层:", [(r["region"], r["count"]) for r in ranges])
    if len(shell["xyz"]):
        print("壳层色均值:", shell["rgba"][:, :3].mean(0).round(3),
              "alpha均值:", round(float(shell["rgba"][:, 3].mean()), 3),
              "σ中位:", np.median(shell["scale"], axis=0).round(4))
    made = fitter.merge_shell(made, shell) if len(shell["xyz"]) else made
    print(f"稠密重建: {len(dense['xyz'])} splats -> 妆容 {len(made['xyz'])}"
          f"（壳层 {len(shell['xyz'])}，分层 {len(ranges)}）")

    # ---- 对比图：按 COLMAP 相机偏航挑 3 个视角（近正脸 / 左 / 右）----
    import preview_render  # noqa: F401
    model = colmap_io.read_sparse(model_dir)
    cam = model.camera
    centroid = np.asarray(L).mean(0)
    yaw_of = {}
    for name, im in model.images.items():
        R = colmap_io.quat_to_rotmat(im["qvec"])
        C = -R.T @ im["tvec"]
        d = centroid - C
        d = d / (np.linalg.norm(d) + 1e-9)
        yaw_of[name] = float(np.degrees(np.arctan2(d[0], d[2])))

    wanted = [("front", 0.0), ("left", 20.0), ("right", -20.0)]
    picks = []
    used = set()
    for tag, target in wanted:
        cand = sorted(((abs(y - target), nm) for nm, y in yaw_of.items()
                       if nm in model.images and nm not in used))
        if not cand:
            continue
        _, nm = cand[0]
        used.add(nm)
        picks.append((tag, nm))
        print(f"    视角 {tag}: {nm} (yaw={yaw_of[nm]:.1f}°)")
    sys.path.insert(0, str(ROOT_P / "preview"))
    from compare_bare_vs_makeup import hstack_border, label  # noqa: E402

    rows = []
    front = {}
    for tag, nm in picks:
        im = model.images[nm]
        R = colmap_io.quat_to_rotmat(im["qvec"])
        t = im["tvec"]
        bare_img = label(render_cloud(dense, R, t, cam, w=560, h=560),
                         f"BARE  ({tag}, {nm})")
        made_img = label(render_cloud(made, R, t, cam, w=560, h=560),
                         f"MAKEUP  date-rose  ({tag})")
        rows.append(hstack_border([bare_img, made_img]))
        if tag == "front" and "front" not in front:
            front = {"bare": bare_img[28:], "made": made_img[28:]}   # 去掉标签条

    def lip_zoom(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        crop = img[int(h * 0.58):int(h * 0.90), int(w * 0.30):int(w * 0.70)]
        return cv2.resize(crop, (0, 0), fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)

    if front:
        rows.append(hstack_border([label(lip_zoom(front["bare"]), "LIP ZOOM  bare"),
                                   label(lip_zoom(front["made"]), "LIP ZOOM  makeup")]))
    OUT_CMP.mkdir(parents=True, exist_ok=True)
    wmax = max(r.shape[1] for r in rows)     # 各行宽度对齐（右侧补黑边）
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(str(OUT_CMP / "compare_real_fit.png"), np.vstack(rows))
    print(f"-> {OUT_CMP / 'compare_real_fit.png'}")


if __name__ == "__main__":
    main()
