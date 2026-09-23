"""compare_pixel_makeup — 逐像素妆容路径 vs 壳层路径的前后对比渲染。

对比列：
    bare           素颜底模（两条路径共同的参考）
    shell(old)     旧壳层路径渲染（splat 常数色 + 线性 alpha，未分裂加密）
    shell+subdiv   新导出壳层（②2×2 分裂加密后的 madeup.ply）
    pixel beer     逐像素 UV 合成 + Beer-Lambert（①默认交付）
    pixel linear   逐像素 UV 合成 + 线性 alpha（A/B 对照：隔离合成模型差异）

输出（--out 目录）：
    compare_full.png   正面全帧 5 列
    compare_lips.png   唇部特写（old | subdiv | beer | linear，×放大）
    compare_eyes.png   眼部特写（同上）
    still_{left,right}_old.png / still_{left,right}_pixel.png   侧视角对照
用法：
    py310 preview/compare_pixel_makeup.py --asset out/photoreal/hd \
        --sfm out/scan_hd/colmap/sparse_gt --size 1024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "desktop-app"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _project(lm: np.ndarray, w2c: np.ndarray, K: np.ndarray) -> np.ndarray:
    """3D 地标 → 像素坐标 (n,2)。"""
    hom = np.concatenate([lm, np.ones((len(lm), 1))], 1)
    cam = (hom @ w2c.T)[:, :3]
    ok = cam[:, 2] > 1e-6
    px = K[0, 0] * cam[:, 0] / np.maximum(cam[:, 2], 1e-9) + K[0, 2]
    py = K[1, 1] * cam[:, 1] / np.maximum(cam[:, 2], 1e-9) + K[1, 2]
    return np.stack([px, py], 1) * ok[:, None]


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (8 + 11 * len(text), 30), (255, 255, 255), -1)
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2)
    return out


def _crop_zoom(img: np.ndarray, cx: float, cy: float, half: float,
               out_side: int) -> np.ndarray:
    """大图缓冲内取 (cx,cy) 邻域 2×half 方块并缩放到 out_side（LANCZOS）。"""
    big = img.shape[0]
    half = int(half)
    x0 = int(max(0, min(big - 2 * half, cx - half)))
    y0 = int(max(0, min(big - 2 * half, cy - half)))
    crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    out = cv2.resize(crop, (out_side, out_side), interpolation=cv2.INTER_LANCZOS4)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8) \
        if out.dtype != np.uint8 else out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", required=True, help="资产目录（base.ply/madeup.ply/landmarks.npy）")
    ap.add_argument("--shell-ply", default=None,
                    help="旧壳层 ply（未分裂加密；缺省 <asset>/madeup_shell.ply）")
    ap.add_argument("--sfm", required=True, help="COLMAP sparse 目录（真实位姿环绕）")
    ap.add_argument("--out", default=None, help="输出目录（缺省 <asset>/pixel_compare）")
    ap.add_argument("--size", type=int, default=1024, help="全帧对比边长")
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--zoom", type=float, default=3.0, help="特写放大倍数")
    args = ap.parse_args()

    from makeupstudio.face3dgs import colmap_io
    from makeupstudio.face3dgs.appearance.offline_render import (
        _pick_views, _resolve_background, _shade_aovs, build_shade, infer_axes,
        load_prepared, orbit_from_poses, preset_light_world, render_pose_pixel,
        face_height)
    from makeupstudio.face3dgs.appearance.makeup_pack import MakeupPack
    from makeupstudio.face3dgs.splat_io import read_ply

    asset = Path(args.asset)
    out_dir = Path(args.out) if args.out else asset / "pixel_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    made = read_ply(asset / "madeup.ply")                  # 新导出（含②分裂）
    shell_old_path = Path(args.shell_ply) if args.shell_ply else asset / "madeup_shell.ply"
    shell_old = read_ply(shell_old_path) if shell_old_path.exists() else made
    base = read_ply(asset / "base.ply")
    lm = np.load(asset / "landmarks.npy")
    pack = MakeupPack.load(asset / "makeup_maps.npz")

    # ---- 相机：与 deliver 完全同款的真实位姿环绕 ----
    center, up, front = infer_axes(made["xyz"], lm)
    model = colmap_io.read_sparse(args.sfm)
    w2cs, Ks = [], []
    for name in sorted(model.images):
        im = model.images[name]
        R = colmap_io.quat_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], np.float64)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        w2cs.append(w2c)
        Ks.append(np.array([[model.camera.params[0], 0, model.camera.params[1]],
                            [0, model.camera.params[0], model.camera.params[2]],
                            [0, 0, 1]], np.float64))
    q = made["xyz"] - center
    front_extent = float(np.percentile(q @ front, 99.5))
    fh = face_height(made["xyz"], up, center)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for w in w2cs:
        d = w[2, :3]
        o = -w[:3, :3].T @ w[:3, 3]
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    center = np.linalg.solve(A + 1e-9 * np.eye(3), b)
    cams = orbit_from_poses(w2cs, Ks, center, n=36, size=args.size, face_h=fh,
                            frame_fill=0.45, radius_min=front_extent + 0.35 * fh,
                            cam_size=(model.camera.width, model.camera.height))
    use_sh = True
    light = preset_light_world("capture", up, front)

    shade_base = build_shade(base, mat_dir=asset, light_override=light)
    shade_old = build_shade(shell_old, mat_dir=asset, light_override=light)
    shade_sub = build_shade(made, mat_dir=asset, light_override=light)

    from makeupstudio.face3dgs.appearance.offline_render import make_pixel_ctx
    ctx_beer = make_pixel_ctx(base, pack, use_sh=use_sh, shade=shade_base)
    ctx_beer.mode = "beer"
    ctx_lin = make_pixel_ctx(base, pack, use_sh=use_sh, shade=shade_base)
    ctx_lin.mode = "linear"

    prepared_base = load_prepared(base, use_sh=use_sh)
    prepared_old = load_prepared(shell_old, use_sh=use_sh)
    prepared_sub = load_prepared(made, use_sh=use_sh)

    def _render_big(prepared, w2c, K, big, shade):
        """SSAA 大图缓冲渲染（供特写裁剪；与 render_pose 同流程）。"""
        import torch
        from gsplat import rasterization
        base_col, bg_arr = _resolve_background("white", big)
        r, a, _i = rasterization(
            prepared["xyz"], prepared["rot"], prepared["scale"], prepared["opacity"],
            prepared["colors"], torch.tensor(w2c, dtype=torch.float32,
                                             device="cuda")[None],
            torch.tensor(K, dtype=torch.float32, device="cuda")[None], big, big,
            sh_degree=prepared["deg"], rasterize_mode="antialiased", packed=False,
            backgrounds=torch.tensor(base_col, dtype=torch.float32,
                                     device="cuda")[None])
        alpha = a[0][..., 0].cpu().numpy()
        img = r[0].cpu().numpy().astype(np.float64)
        if bg_arr is not None:
            img = img + bg_arr.astype(np.float64) * (1.0 - alpha)[..., None]
        img = np.clip(_shade_aovs(shade, prepared, w2c, K, big, img, alpha), 0, 1)
        return img, alpha

    # ---- 1. 正面全帧 5 列 ----
    name, idx = "front", _pick_views(cams)["front"]
    w2c, K, _yaw = cams[idx]
    big = args.size * args.ssaa
    Kb = K.copy()
    Kb[:2] *= big / args.size
    cols = []

    import torch
    from gsplat import rasterization

    def _render_pixel_big(ctx):
        from makeupstudio.face3dgs.appearance.offline_render import _pixel_render_big
        _bc, bg_arr = _resolve_background("white", big)
        _black, img_f, _a = _pixel_render_big(ctx, w2c, Kb, big, bg_arr)
        return img_f

    r, a_, _i = rasterization(
        prepared_base["xyz"], prepared_base["rot"], prepared_base["scale"],
        prepared_base["opacity"], prepared_base["colors"],
        torch.tensor(w2c, dtype=torch.float32, device="cuda")[None],
        torch.tensor(Kb, dtype=torch.float32, device="cuda")[None], big, big,
        sh_degree=prepared_base["deg"], rasterize_mode="antialiased", packed=False,
        backgrounds=torch.ones(1, 3, device="cuda"))
    img_b = r[0].cpu().numpy().astype(np.float64)
    alpha_b = a_[0][..., 0].cpu().numpy()
    img_b = np.clip(_shade_aovs(shade_base, prepared_base, w2c, Kb, big,
                                img_b, alpha_b), 0, 1)
    cols.append(_label((img_b * 255).astype(np.uint8), "bare"))

    img_old, _ = _render_big(prepared_old, w2c, Kb, big, shade_old)
    cols.append(_label((img_old * 255).astype(np.uint8), "shell(old)"))

    img_sub, _ = _render_big(prepared_sub, w2c, Kb, big, shade_sub)
    cols.append(_label((img_sub * 255).astype(np.uint8), "shell+subdiv"))

    img_beer = _render_pixel_big(ctx_beer)
    cols.append(_label((img_beer * 255).astype(np.uint8), "pixel beer"))

    img_lin = _render_pixel_big(ctx_lin)
    cols.append(_label((img_lin * 255).astype(np.uint8), "pixel linear"))

    row = np.concatenate(cols, axis=1)
    p_full = out_dir / "compare_full.png"
    cv2.imwrite(str(p_full), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    print(f"[out] {p_full}")

    # ---- 2. 唇/眼特写（专用 4096² 缓冲渲染：特写接近原生分辨率，
    #         不受全帧缓冲的放大糊化；old | subdiv | beer | linear） ----
    crop_big = max(4096, big)
    Kc = K.copy()
    Kc[:2] *= crop_big / args.size
    w2c_c = w2c

    def _render_variant_big(variant: str) -> np.ndarray:
        if variant == "shell(old)":
            img, _ = _render_big(prepared_old, w2c_c, Kc, crop_big, shade_old)
            return img
        if variant == "shell+subdiv":
            img, _ = _render_big(prepared_sub, w2c_c, Kc, crop_big, shade_sub)
            return img
        ctx = ctx_beer if variant == "pixel beer" else ctx_lin
        from makeupstudio.face3dgs.appearance.offline_render import (
            _pixel_render_big, _resolve_background)
        _bc, bg_arr = _resolve_background("white", crop_big)
        _black, img_f, _a = _pixel_render_big(ctx, w2c_c, Kc, crop_big, bg_arr)
        return img_f

    big_imgs = {k: _render_variant_big(k) for k in
                ("shell(old)", "shell+subdiv", "pixel beer", "pixel linear")}
    px = _project(lm, w2c_c, Kc)
    face_w = float(np.percentile(px[np.linalg.norm(lm, axis=1) > 0, 0], 97)
                   - np.percentile(px[np.linalg.norm(lm, axis=1) > 0, 0], 3))
    zoom_side = int(face_w * 0.22 * args.zoom)

    regions = {
        "lips": (0.5 * (px[13] + px[14]), 0.20),
        "eyes": (0.5 * (px[33] + px[263]), 0.26),
    }
    for rname, (center_px, half_frac) in regions.items():
        if np.linalg.norm(center_px) < 1e-3:
            print(f"[warn] {rname} 地标投影失败，跳过特写")
            continue
        half = face_w * half_frac
        cells = [_label(_crop_zoom(v, center_px[0], center_px[1], half, zoom_side),
                        k) for k, v in big_imgs.items()]
        grid = np.concatenate(cells, axis=1)
        p = out_dir / f"compare_{rname}.png"
        cv2.imwrite(str(p), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"[out] {p}")

    # ---- 3. 侧视角对照（old vs beer） ----
    picks = _pick_views(cams)
    for sname in ("left", "right"):
        w2c_s, K_s, _y = cams[picks[sname]]
        big_s = args.size * args.ssaa
        Kb_s = K_s.copy()
        Kb_s[:2] *= big_s / args.size
        img_o, _ = _render_big(prepared_old, w2c_s, Kb_s, big_s, shade_old)
        img_p = _render_pixel_big_at(ctx_beer, w2c_s, Kb_s, big_s)
        row_s = np.concatenate([
            _label((img_o * 255).astype(np.uint8), "shell(old)"),
            _label((img_p * 255).astype(np.uint8), "pixel beer")], axis=1)
        p = out_dir / f"still_{sname}_cmp.png"
        cv2.imwrite(str(p), cv2.cvtColor(row_s, cv2.COLOR_RGB2BGR))
        print(f"[out] {p}")

    print("[done] 像素 vs 壳层对比渲染完成")
    return 0


def _render_pixel_big_at(ctx, w2c, Kb, big):
    from makeupstudio.face3dgs.appearance.offline_render import _pixel_render_big
    from makeupstudio.face3dgs.appearance.offline_render import _resolve_background
    _bc, bg_arr = _resolve_background("white", big)
    _black, img_f, _a = _pixel_render_big(ctx, w2c, Kb, big, bg_arr)
    return img_f


if __name__ == "__main__":
    raise SystemExit(main())
