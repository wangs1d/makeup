"""run_sr1080 — 低清源的超分重训交付：Real-ESRGAN ×4 → 复用 SfM 位姿 → gsplat 重训 → 1080 渲染。

源视频只有 480² 时（无重录条件）的画质路径：把 COLMAP 已配准帧超分到 1920²，
位姿/内参几何不变（build_views 按图像尺寸自动缩放 K），重训让 3DGS 从超分细节
学习高频结构，再以 1080 出图。SR 有轻微绘画感，属"超分辅助增强"，非真光学 1080。

    python preview/run_sr1080.py --frames-src out/real/capture/frames \
        --sfm out/real/sfm/sparse/3 --init out/real/project/face_dense.ply \
        --weights out/sr/RealESRGAN_x4plus.pth \
        --spec makeup-skill/presets/date-rose.json --out out/photoreal/d6-1080
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "desktop-app"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import cv2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-src", required=True, help="原始低清帧目录（480²）")
    ap.add_argument("--project", default="out/real-sr", help="超分帧工程目录")
    ap.add_argument("--sfm", required=True, help="已有 COLMAP sparse（位姿复用）")
    ap.add_argument("--init", required=True, help="初始点云 ply（世界系不变，直接复用）")
    ap.add_argument("--weights", default="out/sr/RealESRGAN_x4plus_nateraw.pth")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", default="out/photoreal/d6-1080")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--render-size", type=int, default=1080)
    ap.add_argument("--skip-sr", action="store_true", help="超分帧已就位时跳过")
    args = ap.parse_args()

    from makeupstudio.face3dgs import colmap_io
    from upscale_frames import load_model, upscale_image

    t0 = time.time()
    images = Path(args.project) / "images"
    images.mkdir(parents=True, exist_ok=True)

    # ---- 1. 超分 COLMAP 已配准帧（文件名 = 位姿键名，保持一致） ----
    model = colmap_io.read_sparse(args.sfm)
    names = sorted(model.images)
    todo = [n for n in names if not (images / n).exists()]
    if args.skip_sr or not todo:
        print(f"[sr] 超分帧已就位 {len(names) - len(todo)}/{len(names)}")
    else:
        print(f"[sr] 超分 {len(todo)}/{len(names)} 帧（×4）…", flush=True)
        sr_model = load_model(args.weights)
        for i, name in enumerate(todo):
            img = cv2.imread(str(Path(args.frames_src) / name))
            if img is None:
                print(f"[sr] 缺源帧 {name}，跳过")
                continue
            out = upscale_image(sr_model, img)
            cv2.imwrite(str(images / name), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                print(f"[sr] {i + 1}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)
        del sr_model
        import torch
        torch.cuda.empty_cache()

    def cb(stage, frac, msg):
        print(f"[{stage:>7}] {frac * 100:5.1f}%  {msg}", flush=True)

    # ---- 2. 重训（位姿复用；K 按新尺寸自动缩放；羽化随 4× 分辨率放大） ----
    from makeupstudio.face3dgs.appearance import offline_render
    from makeupstudio.face3dgs.appearance.pipeline import (
        apply_makeup_to_asset, build_asset, export_material)
    from makeupstudio.face3dgs.appearance.train_base import TrainConfig
    from makeupstudio.face3dgs.splat_io import read_ply, write_ply

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    cloud, sel, landmarks, model, report = build_asset(
        args.project, args.sfm, args.init, out_dir,
        train_cfg=TrainConfig(iters=args.iters, feather_px=52),
        reuse_base=False, progress=cb)
    print(f"[train] {report['splats']} splats · PSNR {report['psnr_mean']}dB · "
          f"{report['seconds']}s")

    made, coverage = apply_makeup_to_asset(cloud, landmarks, spec, out_dir)
    write_ply(made, out_dir / "madeup.ply")
    export_material(made, out_dir)

    cb("render", 0.0, "1080 离线渲染交付…")
    render_cb = lambda f, m: cb("render", f, m)   # deliver 回调是 (frac, msg)
    outs = offline_render.deliver(
        out_dir / "madeup.ply", out_dir / "renders",
        base_ply=out_dir / "base.ply",
        landmarks_path=out_dir / "landmarks.npy",
        sfm_dir=args.sfm, n_frames=36, size=args.render_size, progress=render_cb)
    print(json.dumps({"report": report, "coverage": round(coverage, 4),
                      "renders": outs}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
