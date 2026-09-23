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
    max_gs: int = 900_000          # 高斯数上限。400k 时代 30k iter 在 15k 就触顶，
                                   # densify 后半程被饿死——唇纹/眼睑的高频细节
                                   # 靠密度换。8GB 显存 @480² 实测 900k 可容
    refine_stop_frac: float = 0.75
    reset_every: int = 3000
    refine_every: int = 100
    grow_grad2d: float = 0.00013   # 稠密化梯度阈（gsplat 默认 2e-4）：调低让
                                   # 唇线/眼睑等高频区更早分裂出新高斯
    ssim_weight: float = 0.2
    sh_degree: int = 1               # SH 阶数：≤15 视角的采集默认 1。A/B 实测
                                     # （2026-09-22，同数据 30k iters）：deg2 的
                                     # 21 个高阶通道过拟合视角噪声——残差渲染成
                                     # 彩斑，稍偏训练视线即灾难（deliver 实测）；
                                     # deg1 留出 PSNR 更高（23.96 vs 23.69）、
                                     # orbit 渲染质量分最优、SH 可安全参与渲染。
                                     # ≥1080p 多视角重录后可回 2（视角相关油光）
    sh_weight: float = 2e-4          # SH 高阶 L2 正则：≤15 个训练视角时 21 个高阶
                                     # 通道必然过拟合 → 合成视角外推成彩色碎斑；
                                     # 眉眼碎斑主要是 SH 残差，密度上来后逐 splat
                                     # 约束变弱，2e-4 再压一档（0=旧行为）
    antialiased: bool = True         # Mip-Splatting 式 2D 滤波：拉近拉远不呼吸
    mip3d_gamma: float = 0.25        # 3D 平滑滤波 γ：尺度下限=γ×近邻距（0=关）。
                                     # 0.3 压花斑但明显糊唇；曝光补偿 + 密度上来
                                     # 之后花斑的根因减弱，降到 0.25 换回锐度
    hull_margin: float = 1.12        # 蒙版外扩（含发际边缘，不含背景墙）
    mask_shape: str = "hull"         # "hull"=468 点凸包（凸，覆盖发型）；"oval"=
                                     # FACE_OVAL 轮廓多边形（跟随脸型含凹陷）——
                                     # 光头/贴脸背景（凸包凹陷区是背景）必须用 oval，
                                     # 否则背景被"合法"训进资产（实测满屏彩点）
    feather_px: int = 13             # 蒙版羽化（软权重）
    eval_holdout: int = 4            # 留出验证帧数（6→4：7 个训练视角养不活
                                     #  densify，验证密度让位于视图数）
    exposure_comp: bool = True       # 逐视图 log-gain 曝光补偿：自拍视频的自动
                                     # 曝光漂移让同一表面点在帧间亮度不一致 →
                                     # 3DGS 折中成"降饱和+漂浮物"。联合优化
                                     # 每视图 3 维 log-gain，颜色一致性问题
                                     # 从几何层挪到光照层
    # ---- 器官一致性屏蔽（跨帧外观漂移的器官从全权重损失里降权）----
    # 嘴内（牙齿/口腔）：闭嘴 canonical（frames.closed_lips）下内唇环退化，
    # 本屏蔽自动失效（闭嘴时根本没有牙齿问题）；它只作为张嘴兜底带的安全冗余。
    # 眼球：**不再挖洞**——上一版 0.35 降权把眼睛的细节也一起饿死了（用户
    # 可见回退）。视线一致性交给 gaze 特征聚类（gaze_weight=2），眼球保持
    # 全监督；1.0 = 无屏蔽。
    mouth_weight: float = 0.12
    eye_weight: float = 1.0
    mouth_shrink: float = 0.85       # 内唇环多边形向质心收缩（保护唇线像素）
    eye_shrink: float = 0.55         # 眼开多边形收缩（仅 eye_weight<1 时有意义）
    lpips_eval: bool = True          # 留出帧 LPIPS（感知指标；缺包自动跳过）
    seed: int = 0


@dataclass
class View:
    name: str
    w2c: np.ndarray                  # (4,4) world-to-cam（COLMAP R,t 直接构成）
    K: np.ndarray                    # (3,3)
    img: np.ndarray                  # (H,W,3) float32 0..1 RGB
    mask: np.ndarray                 # (H,W) float32 0..1 脸区软权重（含器官孔）


# 眼开多边形（FaceMesh 468 拓扑，顺/逆时针均可——fillPoly 不要求有序方向）。
# 收缩后只盖眼球（虹膜/巩膜），眼睑缘保持全监督（眼线/睫毛区域的真相来源）。
EYEBALL_RING = {
    "left": (33, 246, 161, 160, 159, 158, 157, 173, 133,
             155, 154, 153, 145, 144, 163, 7),
    "right": (263, 466, 388, 387, 386, 385, 384, 398, 362,
              382, 381, 380, 374, 373, 390, 249),
}


def _suppression_mask(h: int, w: int, poly_px: np.ndarray, shrink: float,
                      weight: float, feather_px: float) -> np.ndarray:
    """器官屏蔽图 (H,W)：多边形内向 weight 软衰减，边缘 feather_px 羽化。

    返回乘性权重（1=不衰减）；多边形无效（地标缺失/退化）返回全 1。"""
    import cv2
    poly = np.asarray(poly_px, np.float64)
    if len(poly) < 3 or not np.isfinite(poly).all():
        return np.ones((h, w), np.float32)
    if (np.linalg.norm(poly, axis=1) <= 0).any():
        return np.ones((h, w), np.float32)
    if shrink != 1.0:
        cen = poly.mean(0)
        poly = (poly - cen) * float(shrink) + cen
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
    if float(m.sum()) / 255.0 < 4.0:            # 闭嘴等退化：孔不存在
        return np.ones((h, w), np.float32)
    soft = cv2.GaussianBlur(m.astype(np.float32) / 255.0,
                            (0, 0), max(feather_px, 1.0) / 2.5)
    return (1.0 - (1.0 - float(weight)) * np.clip(soft, 0, 1)).astype(np.float32)


def _mouth_interior_ring() -> np.ndarray:
    """内唇环地标索引（landmark-regions.json 的 lips_inner，与唇拓扑同源）。"""
    import json
    from ..fit_makeup import _REFS
    data = json.loads((_REFS / "landmark-regions.json").read_text(encoding="utf-8"))
    return np.asarray(data["regions"]["lips_inner"]["indices"], np.int64)


def build_views(model: colmap_io.SparseModel, sel: FrameSelection,
                images_dir: Path, cfg: TrainConfig) -> list[View]:
    """COLMAP 位姿 + 表情簇帧 → 训练视图（脸区软蒙版 + 嘴内/眼球屏蔽孔）。"""
    cam = model.camera
    K = np.array([[cam.params[0], 0, cam.params[1]],
                  [0, cam.params[0], cam.params[2]],
                  [0, 0, 1]], np.float64)
    mouth_ring = _mouth_interior_ring()
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
        px = sel.px[name]
        pts = px[:468].astype(np.int32)
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
        m = cv2.GaussianBlur(m, (0, 0), cfg.feather_px / 2.5).astype(np.float32) / 255.0
        # 器官屏蔽：跨帧外观漂移的器官降权（牙齿/口腔、虹膜/巩膜）。
        # 地标不足 478 时眼球环仍可用（<468 的观测帧已在选帧阶段剔除）
        if len(px) >= 468:
            m = m * _suppression_mask(h, w, px[mouth_ring], cfg.mouth_shrink,
                                      cfg.mouth_weight, cfg.feather_px)
            for side in ("left", "right"):
                m = m * _suppression_mask(h, w, px[list(EYEBALL_RING[side])],
                                          cfg.eye_shrink, cfg.eye_weight,
                                          cfg.feather_px)
        views.append(View(name=name, w2c=w2c, K=Kf,
                          img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
                          mask=m.astype(np.float32)))
    return views
def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], np.float64)


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

    # 留出帧数自适应：视图池小时验证最多吃 20%（14 帧 → 留 2 验 12）
    hold_n = int(min(cfg.eval_holdout, max(1, len(views) // 5)))
    hold_idx = set(np.linspace(0, len(views) - 1, hold_n).astype(int))
    train = [v for i, v in enumerate(views) if i not in hold_idx]
    eval_, eval_orig = [], []
    for i, v in enumerate(views):
        if i in hold_idx:
            eval_.append(v)
            eval_orig.append(i)
    train_orig = [i for i in range(len(views)) if i not in hold_idx]

    params = _init_params(read_ply(init_ply), sh_degree=cfg.sh_degree)
    params = {k: torch.nn.Parameter(v) for k, v in params.items()}
    extent = float(torch.linalg.vector_norm(
        params["means"].detach() - params["means"].detach().mean(0), dim=1).mean())
    # 逐视图曝光补偿：log-gain (V,3)，初始 0（=增益 1）。渲染后乘增益再对 GT。
    # 自拍视频自动曝光/白平衡漂移是颜色不一致的主要来源——不补偿时 3DGS 把
    # 帧间亮度差折中成降饱和 + 漂浮物。
    gains = torch.nn.Parameter(torch.zeros(len(train), 3, device="cuda"))
    optimizers = {
        "means": torch.optim.Adam([{"params": [params["means"]], "lr": 1.6e-4 * extent}]),
        "quats": torch.optim.Adam([{"params": [params["quats"]], "lr": 1e-3}]),
        "scales": torch.optim.Adam([{"params": [params["scales"]], "lr": 5e-3}]),
        "opacities": torch.optim.Adam([{"params": [params["opacities"]], "lr": 5e-2}]),
        "colors": torch.optim.Adam([{"params": [params["colors"]], "lr": 2.5e-3}]),
    }
    if cfg.exposure_comp:
        optimizers["gains"] = torch.optim.Adam([{"params": [gains], "lr": 5e-3}])
    # gains 是纯外观校正参数，不属于 splat 几何/外观参数组——gsplat 的
    # strategy 只认 (means/quats/scales/opacities/colors)，多余的键会断言
    splat_optimizers = {k: v for k, v in optimizers.items() if k != "gains"}
    strategy = DefaultStrategy(
        verbose=False, absgrad=True,
        grow_grad2d=cfg.grow_grad2d,
        refine_start_iter=500, refine_stop_iter=int(cfg.iters * cfg.refine_stop_frac),
        reset_every=cfg.reset_every, refine_every=cfg.refine_every,
        pause_refine_after_reset=256)
    strategy.check_sanity(params, splat_optimizers)
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
        pred = renders[0].permute(2, 0, 1)
        if cfg.exposure_comp:
            pred = pred * torch.exp(gains[ci])[:, None, None]
        loss = ((pred - gt).abs().mean(0) * mw).sum() / mw.sum()
        loss = loss + cfg.ssim_weight * (1 - _ssim(pred * mw, gt * mw))
        if cfg.sh_weight > 0 and params["colors"].shape[1] > 1:
            # SH 高阶 L2：训练视角少时高阶系数在无监督视线方向自由生长，
            # 合成视角一外推就碎成彩色斑；L2 把残差压向 DC 主色
            loss = loss + cfg.sh_weight * (params["colors"][:, 1:, :] ** 2).mean()
        strategy.step_pre_backward(params, splat_optimizers, state, step, info)
        loss.backward()
        if params["means"].shape[0] < cfg.max_gs:      # 高斯数上限（1.5.3 无内建 cap）
            strategy.step_post_backward(params, splat_optimizers, state, step, info,
                                        packed=False)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step % 500 == 0 or step == cfg.iters:
            n_gs = int(params["means"].shape[0])
            cb(step / cfg.iters, f"iter {step}/{cfg.iters}  loss={float(loss):.4f}  splats={n_gs}")
    train_s = time.time() - t0

    # ---- 留出帧 PSNR（+ 可选 LPIPS：感知质量，蒙版 bbox 裁剪去背景差异）----
    lpips_fn = None
    if cfg.lpips_eval:
        try:
            import lpips as _lpips
            lpips_fn = _lpips.LPIPS(net="vgg").to("cuda")
            for p in lpips_fn.parameters():
                p.requires_grad = False
        except Exception:
            lpips_fn = None

    with torch.no_grad():
        psnrs, lpips_vals = [], []
        for e_i, v in enumerate(eval_):
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
            if cfg.exposure_comp:
                # 留出视图没有自己的增益：借用时间最近的训练视图的
                # （视频帧序相邻 → 曝光状态最接近）
                j = int(np.argmin(np.abs(np.asarray(train_orig) - eval_orig[e_i])))
                r = r * torch.exp(gains[j])[None, None, None, :]
            mw = torch.tensor(v.mask, device="cuda")
            gt = torch.tensor(v.img, device="cuda")
            mse = ((r[0] - gt) ** 2).mean(2)[mw > 0.5].mean()
            psnrs.append(float(-10 * torch.log10(mse)))
            if lpips_fn is not None:
                ys, xs = torch.nonzero(mw > 0.5, as_tuple=True)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                crop_r = r[0, y0:y1, x0:x1].permute(2, 0, 1)[None]
                crop_g = gt[y0:y1, x0:x1].permute(2, 0, 1)[None]
                lpips_vals.append(float(lpips_fn(2 * crop_r - 1, 2 * crop_g - 1)))

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
              "sh_weight": cfg.sh_weight, "mip3d_gamma": cfg.mip3d_gamma,
              "grow_grad2d": cfg.grow_grad2d,
              "max_gs": cfg.max_gs,
              "exposure_comp": cfg.exposure_comp,
              "gain_abs_max_db": (round(float(gains.detach().abs().max()) * 8.686, 2)
                                  if cfg.exposure_comp else None),
              "mouth_weight": cfg.mouth_weight, "eye_weight": cfg.eye_weight,
              "light_dir_world": [round(float(x), 4) for x in light_dir],
              "light_strength": round(float(light_strength), 4),
              "light_tint": [round(float(x), 4) for x in light_tint],
              "psnr_masked": [round(p, 2) for p in psnrs],
              "psnr_mean": round(float(np.mean(psnrs)), 2),
              "lpips_masked": ([round(v, 4) for v in lpips_vals]
                               if lpips_vals else None),
              "lpips_mean": (round(float(np.mean(lpips_vals)), 4)
                             if lpips_vals else None),
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
