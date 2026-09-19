"""pipeline — 写真试妆流水线编排：选帧 → 光度训练 → UV 绑定 → 妆容合成 → 导出。

这是 R 升级后的主链路（fit_makeup 的模板面具路径降级为无视频时的兜底）：
    1. frames.select_frames    表情一致簇（3DGS 不再被表情漂移撕碎）
    2. train_base.train_base   gsplat 光度训练（densify 全开，20 万+ 高斯）
    3. uvbind.bind_uv          canonical UV + 区域覆盖绑定为点云一等属性
    4. makeup_uv.bake/apply    2048² UV 妆容目标场 → Lab 迁移烘焙 + 材质通道
    5. optimize（可选）        Stable-Makeup guidance 存在时图像空间联合求解
    6. 导出 ply/splat + 材质 sidecar + 素颜|妆后对比图 + report
"""
from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import colmap_io
from ..splat_io import read_ply
from ...tracker import FaceTracker
from ..fit_makeup import FaceMakeupFitter
from . import train_base as tb
from .frames import FrameSelection, select_frames
from .makeup_uv import UvMakeupBaker, UvMakeupMaps, merge_makeup_layer
from .render_pbr import render_cloud_pbr
from .train_base import TrainConfig, build_views
from .uvbind import bind_uv

ProgressCB = Callable[[str, float, str], None]


@dataclass
class PhotorealResult:
    project_dir: Path
    base_cloud: dict[str, np.ndarray]
    made_cloud: dict[str, np.ndarray]
    maps: UvMakeupMaps | None
    selection: FrameSelection
    train_report: dict
    previews: list[Path] = field(default_factory=list)


def _triangulate_landmarks(model: colmap_io.SparseModel, sel: FrameSelection,
                           px_scale: float = 1.0) -> np.ndarray:
    """复用选帧阶段的地标观测做 DLT 三角化（468,3），免去二次 MediaPipe 检测。

    px_scale：sel.px 像素空间 -> 相机原生像素空间（SR/超分帧是放大图，观测
    坐标比 COLMAP 内参大 N 倍，不缩放三角化结果会整体飞出 N 倍远）。
    与 fit_makeup.triangulate_landmarks 同款鲁棒性：cheirality 过半校验 +
    中位数距离飞点剔除。"""
    from ..fit_makeup import N_CANON_VERTS, projection_matrix, triangulate_dlt
    cam = model.camera
    obs: dict[int, list] = {}
    for name in sel.names:
        if name not in model.images:
            continue
        im = model.images[name]
        if im["cam_id"] != cam.cam_id:
            continue
        P = projection_matrix(cam, im["qvec"], im["tvec"])
        px = sel.px[name][:N_CANON_VERTS] * px_scale
        for li in range(N_CANON_VERTS):
            obs.setdefault(li, []).append((P, px[li]))

    L = np.zeros((N_CANON_VERTS, 3))
    for li, pairs in obs.items():
        if len(pairs) < 2:
            continue
        X = triangulate_dlt([p[0] for p in pairs], [p[1] for p in pairs])
        depths = np.array([(P @ np.append(X, 1.0))[2] for P, _ in pairs])
        if (depths > 0).sum() < len(depths) * 0.5:
            continue
        L[li] = X
    good = np.linalg.norm(L, axis=1) > 0
    if good.sum() > 10:
        med = np.median(L[good], axis=0)
        d = np.linalg.norm(L[good] - med, axis=1)
        thr = np.median(d) * 6 + 1e-6
        bad = good.copy()
        bad[good] = d > thr
        L[bad] = 0.0
    if (np.linalg.norm(L, axis=1) > 0).sum() < 200:
        raise RuntimeError("地标三角化失败（<200 个有效），检查位姿/检测质量")
    return L


def export_material(made: dict[str, np.ndarray], out_dir: Path,
                    smooth: bool = True) -> None:
    """材质 sidecar：material.bin（MKMA v1，Unity 端 GaussianAvatarParser.LoadMaterial
    消费）+ material.json（f16 base64，WebGL/调试用）。

    MKMAT1 布局：magic(4)+"ver u16|flags u16"+N u32+rsv u32，每 splat
    [nx,ny,nz,rough] + [coat,sss,sheen,rsv] f32×8。法线 = min(scale) 薄轴
    （normals.axis_normals；旧 _quat_normals 假设薄轴恒为 Z，对各向异性 splat
    错 90°）+ kNN 邻域平滑（flags bit0=1 标记已平滑），高光连续性取决于此。"""
    from ..fit_makeup import FaceMakeupFitter  # noqa: F401  (语义基础设施同源)
    from .normals import axis_normals, smooth_normals
    mat = made.get("material")
    n = len(made["xyz"])
    rot = np.asarray(made["rot"], np.float64)
    normals = axis_normals(rot, np.asarray(made["scale"], np.float64))
    flags = 0
    if smooth and n > 64:
        normals = smooth_normals(np.asarray(made["xyz"], np.float32), normals,
                                 k=12, iters=2)
        flags = 1
    normals = normals.astype(np.float32)
    body = np.zeros((n, 8), np.float32)
    body[:, 0:3] = normals
    body[:, 3] = 0.52 if mat is None else np.asarray(mat.rough, np.float32)
    if mat is not None:
        body[:, 4] = np.asarray(mat.coat, np.float32)
        body[:, 5] = np.asarray(mat.sss, np.float32)
        body[:, 6] = np.asarray(mat.sheen, np.float32)
    (out_dir / "material.bin").write_bytes(
        b"MKMA" + (1).to_bytes(2, "little") + (flags).to_bytes(2, "little")
        + n.to_bytes(4, "little") + (0).to_bytes(4, "little")
        + body.astype("<f4").tobytes())

    payload = {"count": int(n), "normals_smoothed": bool(flags)}
    if mat is not None:
        for key in ("rough", "coat", "sss", "sheen"):
            arr = np.asarray(getattr(mat, key), np.float16)
            payload[key] = base64.b64encode(arr.tobytes()).decode("ascii")
    (out_dir / "material.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_asset(project_dir: str | Path, sfm_dir: str | Path,
                init_ply: str | Path, out_dir: str | Path,
                train_cfg: TrainConfig | None = None, reuse_base: bool = True,
                progress: ProgressCB | None = None):
    """扫描 → 3DGS 资产：表情选帧 → gsplat 光度训练 → 地标三角化。

    产品链路第 1-2 步（数字资产化），与妆容解耦——换妆不重训。
    返回 (cloud, sel, landmarks, model, train_report)。"""
    cb = progress or (lambda *a: None)
    project_dir, sfm_dir, out_dir = (Path(p) for p in (project_dir, sfm_dir, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = project_dir / "capture" / "frames"
    if not images_dir.exists():
        images_dir = project_dir / "images"

    model = colmap_io.read_sparse(sfm_dir)
    # ---- 1. 表情一致选帧 ----
    cb("frames", 0.0, "MediaPipe 表情检测…")
    tracker = FaceTracker(smooth=False)
    try:
        sel = select_frames(images_dir, list(model.images), tracker.detect,
                            min_frames=12, progress=lambda f, m: cb("frames", f, m))
    finally:
        tracker.close()
    # 观测像素空间 -> 相机原生空间（SR/放大帧的 sel.px 比内参大 N 倍）
    first_img = next(images_dir / n for n in sel.names if (images_dir / n).exists())
    frame_w = cv2.imread(str(first_img)).shape[1]
    px_scale = model.camera.width / frame_w
    cb("frames", 1.0, f"表情簇 {len(sel)} 帧 / 参考帧 {sel.ref} "
                      f"(落选 {len(sel.rejected)})")

    # ---- 2. 光度训练（或复用） ----
    base_ply = out_dir / "base.ply"
    if reuse_base and base_ply.exists():
        cloud = read_ply(base_ply)
        train_report = json.loads((out_dir / "train_report.json").read_text(encoding="utf-8"))
        cb("train", 1.0, f"复用已有底模 splats={len(cloud['xyz'])}")
    else:
        views = build_views(model, sel, images_dir, train_cfg or TrainConfig())
        cb("train", 0.0, f"gsplat 光度训练（{len(views)} 视图）…")
        cloud, train_report = tb.train_base(
            views, init_ply, out_dir, train_cfg,
            on_progress=lambda f, m: cb("train", f, m))

    # ---- 3. 地标三角化 ----
    cb("bind", 0.2, "三角化 468 地标…")
    landmarks = _triangulate_landmarks(model, sel, px_scale=px_scale)
    np.save(out_dir / "landmarks.npy", landmarks)
    return cloud, sel, landmarks, model, train_report


def apply_makeup_to_asset(cloud: dict, landmarks: np.ndarray, spec: dict,
                          out_dir: str | Path, tex: int = 2048,
                          intensity: float = 0.8,
                          guidance: list[dict] | None = None,
                          as_layer: bool = True,
                          progress: ProgressCB | None = None) -> dict:
    """在 3DGS 资产上渲染妆容：UV 绑定 → 目标场合成 → 3D 锚定 → 烘焙导出。

    产品链路第 3 步（换妆只重跑本步，秒级）。返回 (madeup cloud, 妆区覆盖)。
    as_layer=True（默认）：妆容是**真实高斯壳层**——每个妆区底模 splat 生成
    一个重叠新高斯（对应位置表面上的薄层：沿外法线偏移 0.15×min_scale、
    妆色×底模真实纹理、finish 材质、opacity=w×底模、sh_rest 置零），底模
    本身保持素颜；合并点云导出 madeup.ply/.splat。妆在几何上存在，不依赖
    底模 splat 的颜色被改写。as_layer=False 退回原位 Lab 重染色（诊断用）。
    3D 锚定区域：唇（lip_band_3d）+ 眼线/睫毛/眉（landmark_band_3d）——
    canonical 模板在这些高频区域有 ~2% 系统错位，观测地标带优先、UV 蒙版兜底。
    guidance：_render_guidance_views 产物存在时，先聚合进 UV albedo
    （bake_guidance）重建壳层，再走 optimize_appearance 图像空间外观精修
    （identity 锁以素颜底模为参考）。"""
    cb = progress or (lambda *a: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cb("bind", 0.6, "UV/区域绑定…")
    binding = bind_uv(cloud, landmarks, tex=tex)
    cb("bind", 1.0, f"UV 绑定 valid={float(binding.valid.mean()):.2f}")

    cb("makeup", 0.2, "UV 妆容目标场合成…")
    baker = UvMakeupBaker(tex=tex)
    maps = baker.bake(spec.get("layers", []), intensity=intensity)

    cb("makeup", 0.5, "3D 地标锚定（唇/眼线/睫毛/眉）…")
    layers = [l for l in spec.get("layers", []) if l.get("enabled", True)]
    lip3d = None
    bands3d: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if any(l.get("region") == "lipstick" for l in layers):
        try:
            lip3d = baker.lip_band_3d(cloud, landmarks)
            cb("makeup", 0.55, f"3D 唇带 splats={int((lip3d[0] > 0.02).sum())}")
        except RuntimeError as e:
            cb("makeup", 0.55, f"3D 唇带失败（回退 UV 兜底）：{e}")
    for l in layers:
        region = l.get("region")
        if region not in ("eyeliner", "lashes", "eyebrow"):
            continue
        try:
            bw, bc = baker.landmark_band_3d(
                cloud, landmarks, region, l.get("shape") or {},
                strength=float(l.get("opacity", 0.7)) * intensity * 1.3)
            if float(np.max(bw)) > 0.02:
                bands3d[region] = (bw, bc)
                cb("makeup", 0.6, f"3D {region} splats={int((bw > 0.02).sum())}")
        except (RuntimeError, ValueError) as e:
            cb("makeup", 0.6, f"3D {region} 失败（回退 UV 模板）：{e}")

    def _bake_to_made(m: UvMakeupMaps) -> dict:
        if as_layer:
            layer, src_idx = baker.build_makeup_layer(
                cloud, m, binding.uv, binding.valid,
                lip3d=lip3d, near=binding.near, bands3d=bands3d or None)
            made_m = merge_makeup_layer(cloud, layer, src_idx)
            if len(src_idx):
                # 薄层自检：壳层 splat 必须落在对应底模 splat 的足迹内
                # （偏移 ≤ 0.3×min_scale；0 = 纯重叠，默认含薄层厚度）
                base_xyz = np.asarray(cloud["xyz"], np.float32)
                off = np.linalg.norm(made_m["xyz"][len(cloud["xyz"]):]
                                     - base_xyz[src_idx], axis=1)
                thin = np.asarray(cloud["scale"], np.float64)[src_idx].min(axis=1)
                ratio = off / np.maximum(thin, 1e-12)
                flag = "✓" if float(ratio.max()) <= 0.3 + 1e-6 else "✗ 超限"
                cb("makeup", 0.7, f"妆容壳层 splats={len(src_idx)} "
                                  f"薄层偏移 mean={float(ratio.mean()):.2f}"
                                  f"/max={float(ratio.max()):.2f}×min_scale {flag}")
            else:
                cb("makeup", 0.7, "妆容壳层为空（妆权重低于门限）")
            return made_m
        return baker.apply_to_cloud(cloud, m, binding.uv, binding.valid,
                                    lip3d=lip3d, intensity=intensity,
                                    near=binding.near, bands3d=bands3d or None)

    cb("makeup", 0.65, "生成妆容壳层（near 软门控 + pigment-safe Lab 迁移）…")
    made = _bake_to_made(maps)

    # ---- guidance 聚合 + 外观精修（可选：spec.guidance 存在且已生成） ----
    if guidance:
        cb("makeup", 0.7, "guidance 聚合进 UV albedo…")
        try:
            s, R, t, _ = baker.fitter.register(landmarks)
            S_xyz, S_uv = baker.fitter._surface_samples(s, R, t)
            views_uv = [{"R": v["R"], "t": v["t"], "cam": v["cam"],
                         "img": v["img_bgr"]} for v in guidance]
            maps = baker.bake_guidance(maps, views_uv, (S_xyz, S_uv))
            made = _bake_to_made(maps)
        except Exception as e:                   # guidance 聚合失败不阻断主链路
            cb("makeup", 0.75, f"guidance 聚合失败（跳过）：{e}")
        try:
            from .optimize import OptConfig, optimize_appearance
            cb("makeup", 0.8, "外观精修（几何冻结，颜色残差求解）…")
            opt_views = [{"w2c": v["w2c"], "K": v["K"], "img": v["img"],
                          "size": v["size"]} for v in guidance]
            made = optimize_appearance(
                made, made["makeup_w"], opt_views, bare_cloud=cloud,
                cfg=OptConfig(iters=int((spec.get("guidance") or {})
                                        .get("iters", 1500))))
        except (ImportError, RuntimeError) as e:  # CUDA/torch 缺失时不阻断
            cb("makeup", 0.9, f"外观精修跳过（{e}）")

    _save_uv_debug(maps, out_dir / "makeup_uv_debug.png")
    cov_mask = maps.w > 0.05
    if maps.lip_w is not None:                     # 唇妆独立通道计入覆盖
        cov_mask = cov_mask | (maps.lip_w > 0.05)
    coverage = float(cov_mask.mean())
    cb("makeup", 1.0, f"妆区覆盖 {coverage:.3f}")
    return made, coverage


def _render_guidance_views(cloud: dict, model: colmap_io.SparseModel,
                           sel: FrameSelection, out_dir: Path, spec: dict,
                           cb: ProgressCB) -> list[dict] | None:
    """素颜多视角渲染 × Stable-Makeup → 多视角 guidance 图。

    spec.guidance：{"reference": 妆效参考图路径, "views": 视角数(默认5),
    "size": 渲染分辨率(默认512), "iters": optimize 迭代}。环境不完整时返回
    None 并给出可执行提示（主链路不阻断）。"""
    from .. import guidance as gd
    from .offline_render import load_prepared, render_pose

    gcfg = spec.get("guidance") or {}
    ref_path = Path(gcfg.get("reference", ""))
    if not ref_path.is_file():
        cb("guidance", 1.0, "guidance.reference 缺失，跳过 guidance 链路")
        return None
    ok, hint = (gd.status().ok, gd.status().missing_hint)
    if not ok:
        cb("guidance", 1.0, f"Stable-Makeup 未就绪，跳过：{hint}")
        return None
    import torch
    if not torch.cuda.is_available():
        cb("guidance", 1.0, "无 CUDA，跳过 guidance 链路")
        return None

    names = [n for n in sel.names if n in model.images]
    k = max(1, int(gcfg.get("views", 5)))
    picks = [names[int(i)] for i in np.linspace(0, len(names) - 1, min(k, len(names)))]
    gdir = out_dir / "guidance"
    gdir.mkdir(parents=True, exist_ok=True)
    size = int(gcfg.get("size", 512))
    prepared = load_prepared(cloud, use_sh=True)
    cam = model.camera
    K = np.array([[cam.params[0], 0, cam.params[1]],
                  [0, cam.params[0], cam.params[2]],
                  [0, 0, 1]], np.float64)
    views: list[dict] = []
    for i, name in enumerate(picks):
        im = model.images[name]
        R = colmap_io.quat_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], np.float64)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        bare = render_pose(prepared, w2c, K, size=size, ssaa=1, denoise=False)
        bare_png = gdir / f"bare_{i}.png"
        cv2.imwrite(str(bare_png), cv2.cvtColor(bare, cv2.COLOR_RGB2BGR))
        out_png = gdir / f"guidance_{i}.png"
        gd.generate(bare_png, ref_path, out_png)
        g_bgr = cv2.imread(str(out_png))
        if g_bgr is None:
            cb("guidance", (i + 1) / len(picks), f"guidance {i} 读取失败，跳过")
            continue
        views.append({"w2c": w2c, "K": K, "R": R, "t": t, "cam": cam,
                      "img_bgr": g_bgr,
                      "img": cv2.cvtColor(g_bgr, cv2.COLOR_BGR2RGB
                                          ).astype(np.float32) / 255.0,
                      "size": (size, size)})
        cb("guidance", (i + 1) / len(picks), f"guidance {i + 1}/{len(picks)}")
    return views or None


def run_photoreal(project_dir: str | Path, sfm_dir: str | Path,
                  init_ply: str | Path, spec: dict, out_dir: str | Path,
                  train_cfg: TrainConfig | None = None,
                  tex: int = 2048, intensity: float = 0.8,
                  reuse_base: bool = True,
                  progress: ProgressCB | None = None) -> PhotorealResult:
    """一键：build_asset（选帧/训练/地标）+ apply_makeup_to_asset（上妆/导出）。"""
    cb = progress or (lambda *a: None)
    t0 = time.time()
    out_dir = Path(out_dir)
    project_dir = Path(project_dir)
    cloud, sel, landmarks, model, train_report = build_asset(
        project_dir, sfm_dir, init_ply, out_dir, train_cfg=train_cfg,
        reuse_base=reuse_base, progress=progress)
    cb("bind", 1.0, "资产就绪")

    # ---- guidance（可选）：spec.guidance.reference 存在且 Stable-Makeup 就绪 ----
    guidance = None
    if spec.get("guidance"):
        cb("guidance", 0.0, "素颜多视角渲染 × Stable-Makeup…")
        guidance = _render_guidance_views(cloud, model, sel, out_dir, spec, cb)

    made, coverage = apply_makeup_to_asset(
        cloud, landmarks, spec, out_dir, tex=tex,
        intensity=intensity, guidance=guidance, progress=progress)

    # ---- 预览（PBR + 妆感材质） ----
    cb("export", 0.2, "素颜|妆后对比渲染…")
    images_dir = (project_dir / "capture" / "frames"
                  if (project_dir / "capture" / "frames").exists()
                  else project_dir / "images")
    if not (out_dir / "light.bin").exists():
        # 复用旧底模（无 light.bin）：补估主光方向，保证 Unity 高光与烘焙光照同向
        cb("export", 0.1, "估计主光方向…")
        d_w, s_w, t_w = tb.estimate_light_dir(build_views(model, sel, images_dir,
                                                          train_cfg or TrainConfig()))
        tb.write_light_bin(out_dir / "light.bin", d_w, s_w, t_w)
    previews, madeup_renders = _render_previews(model, sel, cloud, made,
                                                images_dir, out_dir)

    # ---- 还原度度量：渲染帧妆区 vs spec 目标色的逐区域 ΔE00 ----
    delta_e = _delta_e_report(spec, sel, madeup_renders, images_dir)
    cb("export", 0.5, "还原度 ΔE00 " + (str(delta_e.get("_mean", "n/a"))
                                        if delta_e else "（无妆区可评）"))

    # ---- 导出 ----
    from ..splat_io import export_splat, write_ply
    write_ply(made, out_dir / "madeup.ply")
    export_splat(made, out_dir / "madeup.splat")
    export_material(made, out_dir)
    report = {
        "frames_selected": len(sel.names), "frames_rejected": len(sel.rejected),
        "ref_frame": sel.ref, "train": train_report,
        "splats": int(len(made["xyz"])),
        "makeup_layer_splats": int((np.asarray(made["makeup_w"]) > 0.02).sum()),
        "uv_tex": tex, "makeup_coverage": round(coverage, 4),
        "layers": [l.get("id", l.get("region")) for l in spec.get("layers", [])
                   if l.get("enabled", True)],
        "guidance": bool(guidance),
        "makeup_delta_e": delta_e,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "report.json").write_text(json.dumps(
        report, ensure_ascii=False, indent=1), encoding="utf-8")
    cb("export", 1.0, f"完成 → {out_dir}")
    return PhotorealResult(project_dir=project_dir, base_cloud=cloud,
                           made_cloud=made, maps=None, selection=sel,
                           train_report=train_report, previews=previews)


def _delta_e_report(spec: dict, sel: FrameSelection,
                    madeup_renders: dict[str, np.ndarray],
                    images_dir: Path) -> dict:
    """逐帧妆区 ΔE00（渲染帧 vs spec 目标色）→ 逐区域均值 + 总均值。

    蒙版由该帧 MediaPipe 观测地标（sel.px）栅格化，与渲染帧同分辨率——
    与烘焙用同一套区域语义，度量的是"交付图里的妆色离 spec 目标多远"。"""
    from .calibrate import image_region_masks, region_delta_e, spec_targets

    targets = spec_targets(spec)
    if not targets or not madeup_renders:
        return {}
    acc: dict[str, list[float]] = {}
    for name, img_m in madeup_renders.items():
        px = getattr(sel, "px", {}).get(name)
        if px is None:
            continue
        ref = cv2.imread(str(images_dir / name))
        if ref is None:
            continue
        h, w = ref.shape[:2]
        size = img_m.shape[0]
        px_s = px.copy()
        px_s[:, 0] *= size / w
        px_s[:, 1] *= size / h
        masks = image_region_masks(px_s, size, size,
                                   regions=tuple(targets.keys()))
        per = region_delta_e(img_m[..., ::-1], masks, targets)
        for rgn, v in per.items():
            acc.setdefault(rgn, []).append(v)
    if not acc:
        return {}
    out = {r: round(float(np.mean(v)), 2) for r, v in acc.items()}
    out["_mean"] = round(float(np.mean(list(out.values()))), 2)
    return out


def _save_uv_debug(maps: UvMakeupMaps, path: Path) -> None:
    """UV 调试图：albedo×w 叠加（透明底）。"""
    t = maps.tex
    bg = np.full((t, t, 3), 245, np.uint8)
    w3 = (maps.w[..., None] > 0.02)
    fore = (maps.albedo * 255).astype(np.uint8)
    vis = np.where(w3, fore, bg)
    cv2.imwrite(str(path), vis[..., ::-1])


def _render_previews(model: colmap_io.SparseModel, sel: FrameSelection,
                     bare: dict, made: dict, images_dir: Path,
                     out_dir: Path) -> tuple[list[Path], dict[str, np.ndarray]]:
    """素颜|妆后对比图：真实帧 | gsplat 渲染（训练同款光栅化器 + 妆感材质
    合成），无 CUDA 时回退 numpy 圆核预览（仅诊断用）。

    返回（对比图路径列表, {帧名: 妆后渲染 BGR}）——后者供 ΔE00 还原度评测。"""
    from .offline_render import build_shade
    from .render_pbr import render_cloud_pbr
    names = [n for n in sel.names if n in model.images]
    picks = [names[0], names[len(names) // 2], names[-1]]
    size = 760                       # 460 看不见唇纹/粉感，验收环节提高分辨率

    shade_made = build_shade(made, mat_dir=out_dir)
    shade_bare = build_shade(bare, mat_dir=out_dir)

    def gsplat_at(cloud: dict, name: str,
                  shade: "object | None" = None) -> np.ndarray | None:
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            from .offline_render import load_prepared, render_pose
            im = model.images[name]
            R = colmap_io.quat_to_rotmat(im["qvec"])
            t = np.asarray(im["tvec"], np.float64)
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t
            cam = model.camera
            K = np.array([[cam.params[0], 0, cam.params[1]],
                          [0, cam.params[0], cam.params[2]],
                          [0, 0, 1]], np.float64)
            img = render_pose(load_prepared(cloud), w2c, K, size=size, ssaa=1,
                              shade=shade)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    outs, renders = [], {}
    for i, name in enumerate(picks):
        img_m = gsplat_at(made, name, shade_made)
        if img_m is None:
            im = model.images[name]
            img_m = render_cloud_pbr(
                made, colmap_io.quat_to_rotmat(im["qvec"]),
                np.asarray(im["tvec"], np.float64), model.camera, w=size, h=size)
        img_b = gsplat_at(bare, name, shade_bare)
        if img_b is None:
            im = model.images[name]
            img_b = render_cloud_pbr(
                bare, colmap_io.quat_to_rotmat(im["qvec"]),
                np.asarray(im["tvec"], np.float64), model.camera, w=size, h=size)
        renders[name] = img_m
        ref = cv2.imread(str(images_dir / name))
        if ref is not None:
            ref = cv2.resize(ref, (size, size))
            row = np.concatenate([ref, img_b, img_m], axis=1)
        else:
            row = np.concatenate([img_b, img_m], axis=1)
        p = out_dir / f"compare_{i}.png"
        cv2.imwrite(str(p), row)
        outs.append(p)
    return outs, renders
