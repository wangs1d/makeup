"""face3dgs CLI — 用户脸部 3DGS 化与妆容贴合的命令行入口。

    py -m makeupstudio.face3dgs status
    py -m makeupstudio.face3dgs capture  -o out/face3dgs/me.mp4 [--cam 0]
    py -m makeupstudio.face3dgs rebuild  -v me.mp4 -p out/face3dgs/proj --quality standard
    py -m makeupstudio.face3dgs isolate  -p out/face3dgs/proj -o out/face3dgs/face.ply
    py -m makeupstudio.face3dgs fit      -p out/face3dgs/proj --spec makeup-skill/presets/date-rose.json \
                                         -f out/face3dgs/face.ply -o out/face3dgs/fitted
    py -m makeupstudio.face3dgs render   -p out/photoreal/me [--spec 妆容.json] [--ply 已有资产.ply]
                                         → 定妆照 + turntable.mp4 + 素颜对比（后台离线交付，无 Unity）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .capture import capture_from_camera


def _cmd_capture(args) -> int:
    def on_g(g, _frame):
        bar = "#" * int(g.progress * 30)
        msgs = " | ".join(g.messages) or "保持缓慢转头"
        print(f"\r[{bar:<30}] {g.progress*5:.0f}s/{msgs[:40]:<40}", end="", flush=True)
        return True
    try:
        res = capture_from_camera(args.out, cam_id=args.cam, on_guidance=on_g)
    except RuntimeError as e:
        print(f"\n采集失败: {e}")
        return 2
    print(f"\n已保存 {res.video_path}（{res.duration:.1f}s / {res.frames} 帧）")
    print("下一步: py -m makeupstudio.face3dgs rebuild -v "
          f"{res.video_path} -p {res.video_path.with_suffix('').with_name('proj')}")
    return 0


def _cmd_status(_args) -> int:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
    try:
        import gsplat  # noqa: F401
        print("gsplat OK")
    except Exception as e:
        print("gsplat 缺失/未编译：", e)
    try:
        import pycolmap  # noqa: F401
        print("pycolmap OK")
    except ImportError:
        print("pycolmap 缺失：pip install pycolmap")
    return 0


def _cmd_asset(args) -> int:
    from .appearance import sfm
    from .appearance.pipeline import build_asset
    from .appearance.train_base import TrainConfig
    proj = Path(args.project)
    images = proj / "images"
    images.mkdir(parents=True, exist_ok=True)
    sfm.extract_frames(args.video, images, fps=args.fps)
    result = sfm.run_sfm(images, proj)
    def cb(stage, frac, msg):
        print(f"[{stage:>7}] {frac*100:5.1f}%  {msg}", flush=True)
    cloud, _sel, _lm, _model, report = build_asset(
        proj, result.sparse_dir, images, proj / "asset",
        train_cfg=TrainConfig(iters=args.iters), reuse_base=not args.no_reuse,
        progress=cb)
    print(f"资产完成：{proj / 'asset' / 'base.ply'}")
    print(f"  {report['splats']} splats · PSNR {report['psnr_mean']}dB · {report['seconds']}s")
    return 0


def _cmd_makeup(args) -> int:
    from .appearance.pipeline import apply_makeup_to_asset
    from .splat_io import read_ply, write_ply
    from .appearance.pipeline import export_material
    import numpy as np
    proj = Path(args.project)
    asset = proj / "asset"
    cloud = read_ply(asset / "base.ply")
    landmarks = np.load(asset / "landmarks.npy")
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    made, _cov = apply_makeup_to_asset(cloud, landmarks, spec, asset,
                                       intensity=args.intensity)
    write_ply(made, asset / "madeup.ply")
    export_material(made, asset)
    print(f"妆容渲染完成：{asset / 'madeup.ply'}")
    return 0


def _cmd_render(args) -> int:
    """已有资产 → 后台高保真渲染交付物（定妆照 + 环绕视频 + 素颜对比，无 Unity）。"""
    from pathlib import Path as _Path
    from .appearance.offline_render import deliver
    from .splat_io import read_ply, write_ply

    proj = _Path(args.project) if args.project else None
    if proj is not None:
        # 兼容两种布局：<proj>/asset（工程目录）或 <proj> 本身就是资产目录
        asset = (proj / "asset") if (proj / "asset").exists() else proj
    else:
        asset = None

    out = _Path(args.out) if args.out else (asset / "renders" if asset else _Path("renders"))
    out.mkdir(parents=True, exist_ok=True)

    ply = _Path(args.ply) if args.ply else None
    base_ply = _Path(args.base_ply) if args.base_ply else (
        asset / "base.ply" if asset else None)
    landmarks = (asset / "landmarks.npy" if asset
                 else ply.parent / "landmarks.npy" if ply else None)

    if args.spec:
        # 渲染前重新上妆（换妆秒级）：需要工程目录里的 base.ply + landmarks
        import json
        import numpy as np
        from .appearance.pipeline import apply_makeup_to_asset, export_material
        if not (asset and (asset / "base.ply").exists()
                and landmarks is not None and landmarks.exists()):
            print("重新上妆需要工程目录（-p）含 asset/base.ply 与 landmarks.npy")
            return 2
        cloud = read_ply(asset / "base.ply")
        made, _cov = apply_makeup_to_asset(
            cloud, np.load(landmarks),
            json.loads(_Path(args.spec).read_text(encoding="utf-8")),
            asset, intensity=args.intensity)
        ply = out / f"madeup_{_Path(args.spec).stem}.ply"
        write_ply(made, ply)
        export_material(made, out)
        base_ply = asset / "base.ply"
    elif ply is None:
        for name in ("madeup.ply", "base.ply"):
            if asset is not None and (asset / name).exists():
                ply = asset / name
                break
    if ply is None or not ply.exists():
        print("未找到资产 ply（用 --ply 指定，或 -p 工程目录含 asset/madeup.ply|base.ply）")
        return 2

    def cb(f, m):
        print(f"\r[{f*100:5.1f}%] {m:<46}", end="", flush=True)
    print(f"离线渲染：{ply} → {out}{'（真实位姿环绕+SH）' if args.sfm else '（DC-only 窄幅环绕）'}")
    outs = deliver(ply, out, base_ply=base_ply, landmarks_path=landmarks,
                   sfm_dir=_Path(args.sfm) if args.sfm else None,
                   n_frames=args.frames, size=args.size, progress=cb)
    print()
    for k, v in outs.items():
        print(f"  {k}: {v if isinstance(v, str) else (', '.join(v) if isinstance(v, list) else v)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="face3dgs", description="扫描/上传 → 3DGS 资产 → 妆容渲染")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="检查建模环境（CUDA/gsplat/pycolmap）").set_defaults(func=_cmd_status)

    p = sub.add_parser("capture", help="摄像头环绕采集引导")
    p.add_argument("-o", "--out", required=True, help="输出 MP4 路径")
    p.add_argument("--cam", type=int, default=0)
    p.set_defaults(func=_cmd_capture)

    p = sub.add_parser("asset", help="视频 → SfM → gsplat 训练 → base.ply（3DGS 资产）")
    p.add_argument("-v", "--video", required=True)
    p.add_argument("-p", "--project", required=True)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--no-reuse", action="store_true")
    p.set_defaults(func=_cmd_asset)

    p = sub.add_parser("makeup", help="在资产上渲染妆容（秒级换妆）")
    p.add_argument("-p", "--project", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--intensity", type=float, default=0.8)
    p.set_defaults(func=_cmd_makeup)

    p = sub.add_parser("render", help="已有资产 → 离线高保真渲染（turntable/定妆照/对比，无 Unity）")
    p.add_argument("-p", "--project", help="工程目录（含 asset/base.ply、landmarks.npy）")
    p.add_argument("--ply", help="直接指定资产 ply（缺省用工程的 madeup.ply，再回退 base.ply）")
    p.add_argument("--base-ply", help="素颜 ply（对比图用，缺省 asset/base.ply）")
    p.add_argument("--spec", help="可选：渲染前重新上妆的妆容 spec json（秒级换妆）")
    p.add_argument("--intensity", type=float, default=0.8)
    p.add_argument("--sfm", help="COLMAP sparse 目录（给定时走真实位姿环绕+SH，推荐）")
    p.add_argument("-o", "--out", help="输出目录（缺省 project/asset/renders）")
    p.add_argument("--frames", type=int, default=36, help="环绕帧数（默认 36）")
    p.add_argument("--size", type=int, default=1024, help="输出边长（默认 1024）")
    p.set_defaults(func=_cmd_render)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
