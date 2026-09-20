"""train_base — gsplat 光度训练：把"模板面具"升级成训练出来的 3DGS 头像。

根因：历史产物（final.ply=1918 高斯）不是一次成功的光度优化——单目视频 +
表情漂移让梯度互相抵消，densify 从未起飞。本模块在**表情一致帧子集**上
（frames.select_frames）跑标准 3DGS 训练（densify/split/opacity-reset 全开），
并用 MediaPipe 地标凸包蒙版把损失聚焦在脸区。

与 Brush 的分工：Brush 是通用场景训练器（全背景、无表情选择、无蒙版）；
本模块面向"头像试妆"领域：选帧 + 脸区蒙版 + 小图快迭代，8GB 显存即可。
"""
from __future__ import annotations

import json
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import colmap_io
from ..splat_io import read_ply
from .frames import FrameSelection

ProgressCB = Callable[[float, str], None]

SH_C0 = 0.28209479112561376


@dataclass
class TrainConfig:
    iters: int = 20000
    max_gs: int = 400_000
    refine_stop_frac: float = 0.75
    reset_every: int = 3000
    refine_every: int = 100
    ssim_weight: float = 0.2
    sh_degree: int = 2               # SH 高阶：视角相关外观（油光/高光），0=仅 DC
    antialiased: bool = True         # Mip-Splatting 式 2D 滤波：拉近拉远不呼吸
    mip3d_gamma: float = 0.3         # 3D 平滑滤波 γ：尺度下限=γ×近邻距（0=关）；
                                     # 训练分辨率下亚像素 splat 的点采样花斑主修复
    hull_margin: float = 1.12        # 蒙版外扩（含发际边缘，不含背景墙）
    mask_shape: str = "hull"         # "hull"=468 点凸包（凸，覆盖发型）；"oval"=
                                     # FACE_OVAL 轮廓多边形（跟随脸型含凹陷）——
                                     # 光头/贴脸背景（凸包凹陷区是背景）必须用 oval，
                                     # 否则背景被"合法"训进资产（实测满屏彩点）
    feather_px: int = 13             # 蒙版羽化（软权重）
    eval_holdout: int = 6            # 留出验证帧数
    seed: int = 0


@dataclass
class View:
    name: str
    w2c: np.ndarray                  # (4,4) world-to-cam（COLMAP R,t 直接构成）
    K: np.ndarray                    # (3,3)
    img: np.ndarray                  # (H,W,3) float32 0..1 RGB
    mask: np.ndarray                 # (H,W) float32 0..1 脸区软权重


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def build_views(model: colmap_io.SparseModel, sel: FrameSelection,
                images_dir: Path, cfg: TrainConfig) -> list[View]:
    """COLMAP 位姿 + 表情簇帧 → 训练视图（含脸区软蒙版）。"""
    cam = model.camera
    K = np.array([[cam.params[0], 0, cam.params[1]],
                  [0, cam.params[0], cam.params[2]],
                  [0, 0, 1]], np.float64)
    views: list[View] = []
    for name in sel.names:
        img = cv2.imread(str(images_dir / name))
        if img is None:
            continue
        h, w = img.shape[:2]
        sx, sy = w / cam.width, h / cam.height
        Kf = K.copy()
        Kf[0] *= sx
        Kf[1] *= sy
        im = model.images[name]
        R = colmap_io.quat_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], np.float64)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        # 脸区软蒙版：地标轮廓外扩 + 羽化（hull=凸包 / oval=脸型轮廓多边形）
        pts = sel.px[name][:468].astype(np.int32)
        cen = pts.mean(0)
        m = np.zeros((h, w), np.uint8)
        if cfg.mask_shape == "oval":
            from .calibrate import FACE_OVAL
            poly = (pts[list(FACE_OVAL)] - cen) * cfg.hull_margin + cen
            cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        else:
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))
            hull_out = ((hull[:, 0, :] - cen) * cfg.hull_margin + cen).astype(np.int32)
            cv2.fillConvexPoly(m, hull_out, 255)
        m = cv2.GaussianBlur(m, (0, 0), cfg.feather_px / 2.5)
        views.append(View(name=name, w2c=w2c, K=Kf,
                          img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
                          mask=m.astype(np.float32) / 255.0))
    return views


def _init_params(init_cloud: dict[str, np.ndarray], sh_degree: int = 0):
    """PLY 初始点 → torch 参数。

    colors 一律参数化为 SH 系数 (n, K, 3)，K=(sh_degree+1)²：sh0=(rgb−0.5)/C0
    （与 3DGS 标准初始化一致，渲染时直接线性求值，不再 sigmoid），高阶全零起步。
    """
    import torch

    xyz = np.ascontiguousarray(init_cloud["xyz"], np.float32)
    n = len(xyz)
    fl = cv2.flann_Index(xyz, dict(algorithm=1, trees=4, checks=64))
    _nn, d2 = fl.knnSearch(xyz, 4, params=dict(checks=64))
    nn3 = np.sqrt(np.maximum(d2[:, 1:4].mean(1), 1e-12))     # 3 近邻均距
    scale = np.log(np.clip(nn3, 1e-8, None)[:, None] * np.ones((1, 3)))
    scale = scale.astype(np.float32)
    quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (n, 1))  # gsplat wxyz 单位四元数
    opacity = np.full((n, 1), -2.0, np.float32)              # sigmoid(−2)≈0.12
    rgb = np.clip(init_cloud["rgba"][:, :3], 0.01, 0.99).astype(np.float64)
    sh0 = np.clip((rgb - 0.5) / SH_C0, -2.0, 2.0)
    K = (sh_degree + 1) ** 2
    sh = np.zeros((n, K, 3), np.float32)
    sh[:, 0, :] = sh0
    return {k: torch.tensor(v, device="cuda") for k, v in
            {"means": xyz, "quats": quats, "scales": scale,
             "opacities": opacity, "colors": sh}.items()}


def _ssim(img: "object", other: "object"):
    """紧凑 torch SSIM（11×11 高斯窗，输入 (C,H,W)）。"""
    import torch
    import torch.nn.functional as F

    ch = img.shape[0]
    coords = torch.arange(11, device=img.device, dtype=torch.float32) - 5
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    win = (g[:, None] @ g[None, :]).expand(ch, 1, 11, 11) / (g.sum() ** 2)
    mu1 = F.conv2d(img[None], win, padding=4, groups=ch)[0]
    mu2 = F.conv2d(other[None], win, padding=4, groups=ch)[0]
    s11 = F.conv2d((img * img)[None], win, padding=4, groups=ch)[0] - mu1 * mu1
    s22 = F.conv2d((other * other)[None], win, padding=4, groups=ch)[0] - mu2 * mu2
    s12 = F.conv2d((img * other)[None], win, padding=4, groups=ch)[0] - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu1 * mu2 + c1) * (2 * s12 + c2))
            / ((mu1 * mu1 + mu2 * mu2 + c1) * (s11 + s22 + c2))).mean()


def train_base(views: list[View], init_ply: str | Path,
               out_dir: str | Path, cfg: TrainConfig | None = None,
               on_progress: ProgressCB | None = None) -> dict:
    """主训练循环。返回报告（含留出帧 PSNR 与最终 splat 数）。"""
    import torch
    from gsplat import DefaultStrategy, rasterization

    cfg = cfg or TrainConfig()
    cb = on_progress or (lambda *a: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    hold_idx = set(np.linspace(0, len(views) - 1, cfg.eval_holdout).astype(int))
    train = [v for i, v in enumerate(views) if i not in hold_idx]
    eval_ = [v for i, v in enumerate(views) if i in hold_idx]

    params = _init_params(read_ply(init_ply), sh_degree=cfg.sh_degree)
    params = {k: torch.nn.Parameter(v) for k, v in params.items()}
    extent = float(torch.linalg.vector_norm(
        params["means"].detach() - params["means"].detach().mean(0), dim=1).mean())
    optimizers = {
        "means": torch.optim.Adam([{"params": [params["means"]], "lr": 1.6e-4 * extent}]),
        "quats": torch.optim.Adam([{"params": [params["quats"]], "lr": 1e-3}]),
        "scales": torch.optim.Adam([{"params": [params["scales"]], "lr": 5e-3}]),
        "opacities": torch.optim.Adam([{"params": [params["opacities"]], "lr": 5e-2}]),
        "colors": torch.optim.Adam([{"params": [params["colors"]], "lr": 2.5e-3}]),
    }
    strategy = DefaultStrategy(
        verbose=False, absgrad=True,
        refine_start_iter=500, refine_stop_iter=int(cfg.iters * cfg.refine_stop_frac),
        reset_every=cfg.reset_every, refine_every=cfg.refine_every,
        pause_refine_after_reset=256)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=extent * 1.1)

    # 训练帧存 CPU（pinned）按步上传：1080p × 百帧量级时全量进显存会 OOM
    # （114 帧 × 1920×1080×3×4B ≈ 2.8GB），单帧 H2D ≈ 数 ms 可忽略
    imgs = torch.stack([torch.tensor(v.img).pin_memory() for v in train])
    masks = torch.stack([torch.tensor(v.mask).pin_memory() for v in train])
    w2cs = torch.tensor(np.stack([v.w2c for v in train]), dtype=torch.float32, device="cuda")
    Ks = torch.tensor(np.stack([v.K for v in train]), dtype=torch.float32, device="cuda")
    H, W = train[0].img.shape[:2]

    t0 = time.time()
    for step in range(1, cfg.iters + 1):
        ci = int(rng.integers(0, len(train)))
        renders, alphas, info = rasterization(
            params["means"], params["quats"] / params["quats"].norm(dim=1, keepdim=True),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"][..., 0]),
            params["colors"], w2cs[ci:ci + 1], Ks[ci:ci + 1],
            W, H, sh_degree=cfg.sh_degree,
            rasterize_mode="antialiased" if cfg.antialiased else "classic",
            packed=False, absgrad=True, backgrounds=torch.zeros(1, 3, device="cuda"))
        info["means2d"].retain_grad()
        gt = imgs[ci].to("cuda", non_blocking=True).permute(2, 0, 1)
        mw = masks[ci].to("cuda", non_blocking=True)[None]
        loss = ((renders[0].permute(2, 0, 1) - gt).abs().mean(0) * mw).sum() / mw.sum()
        loss = loss + cfg.ssim_weight * (1 - _ssim(renders[0].permute(2, 0, 1) * mw, gt * mw))
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        if params["means"].shape[0] < cfg.max_gs:      # 高斯数上限（1.5.3 无内建 cap）
            strategy.step_post_backward(params, optimizers, state, step, info,
                                        packed=False)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step % 500 == 0 or step == cfg.iters:
            n_gs = int(params["means"].shape[0])
            cb(step / cfg.iters, f"iter {step}/{cfg.iters}  loss={float(loss):.4f}  splats={n_gs}")
    train_s = time.time() - t0

    # ---- 留出帧 PSNR ----
    with torch.no_grad():
        psnrs = []
        for v in eval_:
            r, _a, _i = rasterization(
                params["means"], params["quats"] / params["quats"].norm(dim=1, keepdim=True),
                torch.exp(params["scales"]), torch.sigmoid(params["opacities"][..., 0]),
                params["colors"],
                torch.tensor(v.w2c, dtype=torch.float32, device="cuda")[None],
                torch.tensor(v.K, dtype=torch.float32, device="cuda")[None],
                v.img.shape[1], v.img.shape[0],
                sh_degree=cfg.sh_degree,
                rasterize_mode="antialiased" if cfg.antialiased else "classic",
                packed=False, backgrounds=torch.zeros(1, 3, device="cuda"))
            mw = torch.tensor(v.mask, device="cuda")
            gt = torch.tensor(v.img, device="cuda")
            mse = ((r[0] - gt) ** 2).mean(2)[mw > 0.5].mean()
            psnrs.append(float(-10 * torch.log10(mse)))

    means = params["means"].detach().cpu().numpy()
    quats = params["quats"].detach().cpu().numpy()
    quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12)
    quats = quats[:, [3, 0, 1, 2]]          # gsplat wxyz → 内部约定 xyzw
    scales_np = np.exp(params["scales"].detach().cpu().numpy()).astype(np.float32)
    colors_np = params["colors"].detach().cpu().numpy()          # (n, K, 3) SH
    dc_rgb = np.clip(0.5 + SH_C0 * colors_np[:, 0, :], 0.0, 1.0)
    rgba_np = np.concatenate(
        [dc_rgb, torch.sigmoid(params["opacities"]).detach().cpu().numpy()], axis=1)

    # ---- 主光方向估计（世界系，指向光源）：逐视图取脸区最亮 1/4 像素质心，
    # 沿相机光线方向反投影成"脸中心→亮区"方向，按视图脸区亮度加权平均。
    # 新增 PBR 高光必须与烘焙在颜色里的真实光照同向，否则高光出现在"不该亮"处。
    light_dir, light_strength, light_tint = estimate_light_dir(views=train)

    # ---- 剪枝：蒙版外漂浮物 / 巨块 / 低不透明度 ----
    # 蒙版外无监督 → 高斯自由生长成白色漂浮团。把每个高斯投影到全部训练视图，
    # 只保留"至少在 1/4 视图的脸区内"且尺度和 opacity 合理的。
    n = len(means)
    inside_votes = np.zeros(n, np.int32)
    w2cs = np.stack([v.w2c for v in train])
    Ks_np = np.stack([v.K for v in train])
    masks_np = np.stack([v.mask for v in train])
    homo = np.concatenate([means, np.ones((n, 1), np.float32)], axis=1)  # (n,4)
    for vi in range(len(train)):
        cam_pts = (w2cs[vi][None, :3, :] @ homo[:, :, None])[..., 0]
        z = np.maximum(cam_pts[:, 2], 1e-6)
        px = Ks_np[vi, 0, 0] * cam_pts[:, 0] / z + Ks_np[vi, 0, 2]
        py = Ks_np[vi, 1, 1] * cam_pts[:, 1] / z + Ks_np[vi, 1, 2]
        h_i, w_i = masks_np[vi].shape
        ok = (z > 0.05) & (px >= 0) & (px < w_i) & (py >= 0) & (py < h_i)
        inside_votes[ok] += (masks_np[vi][py[ok].astype(int), px[ok].astype(int)] > 0.5)
    max_scale = scales_np.max(1)
    med_scale = float(np.median(max_scale)) if n else 0.0
    keep = ((inside_votes >= max(len(train) // 4, 1))
            & (rgba_np[:, 3] > 0.05)
            & (max_scale < med_scale * 20.0))     # 巨块白团 = 蒙版外自由生长的产物
    pruned = int(n - int(keep.sum()))
    cloud = {
        "xyz": means[keep],
        "scale": scales_np[keep],
        "rot": quats[keep].astype(np.float32),
        "rgba": rgba_np[keep],
    }
    if cfg.sh_degree > 0:
        cloud["sh_rest"] = colors_np[keep, 1:, :].astype(np.float32)
    print(f"prune: {n} -> {int(keep.sum())} (removed {pruned})")
    mip3d_filter(cloud, gamma=cfg.mip3d_gamma)
    from ..splat_io import export_splat, write_ply
    write_ply(cloud, out_dir / "base.ply")
    export_splat(cloud, out_dir / "base.splat")
    write_light_bin(out_dir / "light.bin", light_dir, light_strength, light_tint)
    report = {"iters": cfg.iters, "splats": int(len(cloud["xyz"])),
              "train_views": len(train), "eval_views": len(eval_),
              "sh_degree": cfg.sh_degree, "antialiased": cfg.antialiased,
              "mip3d_gamma": cfg.mip3d_gamma,
              "light_dir_world": [round(float(x), 4) for x in light_dir],
              "light_strength": round(float(light_strength), 4),
              "light_tint": [round(float(x), 4) for x in light_tint],
              "psnr_masked": [round(p, 2) for p in psnrs],
              "psnr_mean": round(float(np.mean(psnrs)), 2),
              "seconds": round(train_s, 1), "extent": round(extent, 4)}
    (out_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    cb(1.0, f"训练完成 splats={report['splats']} PSNR={report['psnr_mean']}dB")
    return cloud, report


def mip3d_filter(cloud: dict[str, np.ndarray], gamma: float) -> None:
    """Mip-Splatting §3.1 的 3D 平滑滤波（就地修改）：尺度下限 = γ × 3 近邻均距。

    训练分辨率下（480² 帧 ~34 万 splats，脸部 ~10 splats/px）亚像素 splat 渲染成
    点采样花斑/横向拉丝；尺度下限让每个 splat 在屏幕上至少覆盖有限面积，
    颜色被邻域平均，花斑消失。γ 实测 0.3 去斑保特征，0.5 起糊嘴。"""
    if gamma <= 0 or len(cloud["xyz"]) == 0:
        return
    xyz = np.ascontiguousarray(cloud["xyz"], np.float32)
    fl = cv2.flann_Index(xyz, dict(algorithm=1, trees=4, checks=64))
    _nn, d2 = fl.knnSearch(xyz, 4, params=dict(checks=64))
    nn3 = np.sqrt(np.maximum(d2[:, 1:4].mean(1), 1e-12)).astype(np.float32)
    cloud["scale"] = np.maximum(cloud["scale"], (gamma * nn3)[:, None])


def estimate_light_dir(views: list[View]) -> tuple[np.ndarray, float, np.ndarray]:
    """主光方向估计（世界系，指向光源）+ 强度 + 偏色。

    算法：逐视图在脸区蒙版内取亮度前 1/4 像素的质心，用相机内参反投影到
    "脸中心平面"得到 2D→3D 方向（相机系），经 w2c 旋转转世界系；以视图脸区
    平均亮度加权平均。 Strength = 脸区平均亮度（0..1），tint = 脸区均色。
    纯启发式但足够稳：高光方向是唯一需要的信息，不需要完整光度标定。"""
    lum_w = np.array([0.2126, 0.7152, 0.0722])
    dirs: list[np.ndarray] = []
    weights: list[float] = []
    tints: list[np.ndarray] = []
    for v in views:
        lum = v.img @ lum_w
        m = v.mask > 0.5
        if int(m.sum()) < 100:
            continue
        vals = lum[m]
        thr = float(np.quantile(vals, 0.75))
        ys, xs = np.nonzero(m & (lum >= thr))
        if len(xs) < 16:
            continue
        fx, fy = float(v.K[0, 0]), float(v.K[1, 1])
        cx0, cy0 = float(v.K[0, 2]), float(v.K[1, 2])
        d_cam = np.array([(xs.mean() - cx0) / fx, (ys.mean() - cy0) / fy, 1.0])
        d_cam /= np.linalg.norm(d_cam) + 1e-12
        d_world = v.w2c[:3, :3].T @ d_cam          # cam → world
        dirs.append(d_world)
        weights.append(float(vals.mean()))
        tints.append(v.img[m].mean(0))
    if not dirs:
        return np.array([0.30, 0.55, 0.80]), 0.5, np.array([1.0, 1.0, 1.0])
    D = np.stack(dirs)
    w = np.asarray(weights)
    w = w / (w.sum() + 1e-12)
    d = (D * w[:, None]).sum(0)
    nrm = np.linalg.norm(d)
    if nrm < 1e-6:                                 # 方向对冲：回退相机平均朝向
        d = D.mean(0)
        nrm = np.linalg.norm(d) + 1e-12
    strength = float(np.clip(np.mean(weights), 0.0, 1.0))
    tint = np.clip(np.stack(tints).mean(0), 0.0, 1.0)
    return d / nrm, strength, tint


def write_light_bin(path: Path, light_dir: np.ndarray, strength: float,
                    tint: np.ndarray) -> None:
    """light.bin（MKLT1）：magic(4)+ver u16+flags u16+n u32+rsv u32
    + dir f32×3（世界系指向光源）+ strength f32 + tint f32×3（线性 0..1）。"""
    body = struct.pack("<3f f 3f",
                       *[float(x) for x in light_dir],
                       float(strength),
                       *[float(x) for x in tint])
    (Path(path)).write_bytes(
        b"MKLT" + (1).to_bytes(2, "little") + (0).to_bytes(2, "little")
        + (1).to_bytes(4, "little") + (0).to_bytes(4, "little") + body)
