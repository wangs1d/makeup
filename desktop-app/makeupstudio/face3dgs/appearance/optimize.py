"""optimize — 可微外观优化回路（AvatarMakeup 核心思想的本地化）。

参数化 UV 烘焙（makeup_uv.apply_to_cloud）本身已是"最优解"：目标场直接写进
splat 颜色。本模块解决的是烘焙解决不了的问题——**有外部 2D 目标**（Stable-Makeup
等妆容迁移模型的多视角 guidance 图）时，逐 splat 颜色必须在图像空间联合求解：
    L = 妆区 L1(+LPIPS)(render_made, guidance)
      + λ·非妆区 L1(render_made, render_bare)   ← identity 锁（保牙齿/眼白/肤色）
几何冻结（AvatarMakeup 同款约束：位置/旋转/尺度不动），只学颜色与 opacity 残差。

妆区蒙版不引外部解析模型：把 makeup_w 作为"颜色"再光栅化一次即得每视角妆区。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ProgressCB = Callable[[float, str], None]


@dataclass
class OptConfig:
    iters: int = 1500
    lr_rgb: float = 2.5e-3
    lr_opacity: float = 2e-2
    identity_weight: float = 2.0
    lpips_weight: float = 0.15     # 0 = 纯 L1（lpips 缺失时自动 0）
    seed: int = 0


def optimize_appearance(base_cloud: dict[str, np.ndarray],
                        makeup_w: np.ndarray,
                        guidance_views: list[dict],
                        cfg: OptConfig | None = None,
                        bare_cloud: dict[str, np.ndarray] | None = None,
                        on_progress: ProgressCB | None = None) -> dict:
    """guidance 监督的外观优化。

    base_cloud     优化起点（xyz/scale/rot/rgba）——壳层架构下传合并点云
                   （素颜底模 + 妆容壳层），颜色残差直接修在壳层 splat 上
    makeup_w       (n,) 每 splat 妆区权重（与 base_cloud 等长）
    bare_cloud     identity 锁的素颜参考（缺省用 base_cloud 本身）。壳层
                   架构必须传素颜底模，否则"素颜目标"就是带妆渲染自身，
                   锁失效
    guidance_views [{"w2c"(4,4), "K"(3,3), "img"(H,W,3) RGB 0..1,
                      "size"(H,W)}] —— Stable-Makeup 对各视角素颜渲染的输出
    返回新 cloud（rgba 更新；几何原样）。
    """
    import torch
    from gsplat import rasterization

    cfg = cfg or OptConfig()
    bare_ref = base_cloud if bare_cloud is None else bare_cloud
    cb = on_progress or (lambda *a: None)
    rng = np.random.default_rng(cfg.seed)

    def T(a, dtype=np.float32):
        return torch.tensor(np.ascontiguousarray(a), dtype=dtype, device="cuda")

    means = T(base_cloud["xyz"])
    # 内部约定 xyzw → gsplat wxyz（顺序错 = 协方差朝向全错，优化在错误渲染上求解）
    quats = T(base_cloud["rot"][:, [3, 0, 1, 2]])
    scales = T(np.log(np.maximum(base_cloud["scale"], 1e-10)))
    opac0 = np.clip(base_cloud["rgba"][:, 3], 1e-4, 1 - 1e-4)
    rgb0_logit = np.log(np.clip(base_cloud["rgba"][:, :3], 1e-3, 1 - 1e-3)
                        / (1 - np.clip(base_cloud["rgba"][:, :3], 1e-3, 1 - 1e-3)))
    d_rgb = torch.nn.Parameter(T(rgb0_logit))
    d_op = torch.nn.Parameter(T(np.log(opac0 / (1 - opac0))[..., None]))
    opt = torch.optim.Adam([{"params": [d_rgb], "lr": cfg.lr_rgb},
                            {"params": [d_op], "lr": cfg.lr_opacity}])

    n = len(means)
    w_col = torch.zeros(1, n, 3, device="cuda")
    w_col[0, :, 0] = T(makeup_w)

    # 每视角素颜渲染（identity 锁目标，冻结）：用素颜参考云自身的几何+颜色
    bare_means = T(bare_ref["xyz"])
    bare_quats = T(bare_ref["rot"][:, [3, 0, 1, 2]])
    bare_scales = T(np.log(np.maximum(bare_ref["scale"], 1e-10)))
    bare_op = np.clip(bare_ref["rgba"][:, 3], 1e-4, 1 - 1e-4)
    bare_rgb = np.clip(bare_ref["rgba"][:, :3], 1e-3, 1 - 1e-3)
    bare_colors = torch.sigmoid(T(np.log(bare_rgb / (1 - bare_rgb))))
    bare_targets, guidance_imgs = [], []
    for gv in guidance_views:
        H, W = gv["size"]
        with torch.no_grad():
            r, _a, _i = rasterization(
                bare_means, bare_quats / bare_quats.norm(dim=1, keepdim=True),
                torch.exp(bare_scales), torch.sigmoid(T(bare_op)),
                bare_colors[None, ...],
                torch.tensor(gv["w2c"], dtype=torch.float32, device="cuda")[None],
                torch.tensor(gv["K"], dtype=torch.float32, device="cuda")[None],
                W, H, packed=False, backgrounds=torch.zeros(1, 3, device="cuda"))
        bare_targets.append(r[0].permute(2, 0, 1))
        guidance_imgs.append(torch.tensor(gv["img"], device="cuda").permute(2, 0, 1))

    lpips_fn = None
    if cfg.lpips_weight > 0:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg").to("cuda")
            for p in lpips_fn.parameters():
                p.requires_grad = False
        except Exception:
            lpips_fn = None

    def render_now(view: dict):
        H, W = view["size"]
        return rasterization(
            means, quats / quats.norm(dim=1, keepdim=True), torch.exp(scales),
            torch.sigmoid(d_op[..., 0]), torch.sigmoid(d_rgb)[None, ...],
            torch.tensor(view["w2c"], dtype=torch.float32, device="cuda")[None],
            torch.tensor(view["K"], dtype=torch.float32, device="cuda")[None],
            W, H, packed=False, backgrounds=torch.zeros(1, 3, device="cuda"))

    for step in range(1, cfg.iters + 1):
        gi = int(rng.integers(0, len(guidance_views)))
        renders, _a, _i = render_now(guidance_views[gi])
        info = _i
        info["means2d"].retain_grad()
        img = renders[0].permute(2, 0, 1)
        with torch.no_grad():
            wmap, _aw, _iw = rasterization(
                means, quats / quats.norm(dim=1, keepdim=True), torch.exp(scales),
                torch.ones_like(torch.sigmoid(d_op[..., 0])), w_col,
                torch.tensor(guidance_views[gi]["w2c"], dtype=torch.float32,
                             device="cuda")[None],
                torch.tensor(guidance_views[gi]["K"], dtype=torch.float32,
                             device="cuda")[None],
                guidance_views[gi]["size"][1], guidance_views[gi]["size"][0],
                packed=False, backgrounds=torch.zeros(1, 3, device="cuda"))
        mk = torch.clamp(wmap[0, :, :, 0], 0, 1)
        makeup_px = mk > 0.02
        if makeup_px.any():
            l_make = (img - guidance_imgs[gi]).abs().mean(0)[makeup_px].mean()
        else:
            l_make = torch.zeros((), device="cuda")
        rest_px = ~makeup_px
        if rest_px.any():
            l_res = (img - bare_targets[gi]).abs().mean(0)[rest_px].mean()
        else:
            l_res = torch.zeros((), device="cuda")
        loss = l_make + cfg.identity_weight * l_res
        if lpips_fn is not None and makeup_px.sum() > 64:
            loss = loss + cfg.lpips_weight * lpips_fn(
                (2 * img - 1)[None], (2 * guidance_imgs[gi] - 1)[None]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            cb(step / cfg.iters, f"opt {step}/{cfg.iters} make={float(l_make):.4f} "
                                 f"res={float(l_res):.4f}")

    out = {k: v.copy() for k, v in base_cloud.items()}
    out["rgba"][:, :3] = torch.sigmoid(d_rgb).detach().cpu().numpy()
    out["rgba"][:, 3] = torch.sigmoid(d_op[..., 0]).detach().cpu().numpy()
    return out
