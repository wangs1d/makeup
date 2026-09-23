"""face3dgs CLI — 用户脸部 3DGS 化与妆容贴合的命令行入口。

    py -m makeupstudio.face3dgs status
    py -m makeupstudio.face3dgs capture  -o out/face3dgs/me.mp4 [--cam 0]
    py -m makeupstudio.face3dgs rebuild  -v me.mp4 -p out/face3dgs/proj --quality standard
    py -m makeupstudio.face3dgs isolate  -p out/face3dgs/proj -o out/face3dgs/face.ply
    py -m makeupstudio.face3dgs fit      -p out/face3dgs/proj --spec makeup-skill/presets/date-rose.json \
                                         -f out/face3dgs/face.ply -o out/face3dgs/fitted
    py -m makeupstudio.face3dgs single-image -i photo.jpg -p out/lam/me [--spec 妆容.json] \
                                         → LAM 单图入口（零门槛：一张照片 → 3DGS 资产）
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
    """环境自检。首行是**当前解释器**——3DGS 链路要求 gsplat/pycolmap 就装在这个
    解释器里，跑之前先确认它不是"默认但没装的那一个"。"""
    import sys
    import torch
    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
    try:
        import gsplat
        print("gsplat", getattr(gsplat, "__version__", "OK"))
    except Exception as e:
        print(f"gsplat 缺失/未编译：{e}\n"
              "  asset/render 依赖它 → 换用已装 gsplat 的解释器（见 docs/photoreal-pipeline.md 四）"
              "，或 pip install gsplat")
    try:
        import pycolmap
        print("pycolmap", getattr(pycolmap, "__version__", "OK"))
    except ImportError:
        print("pycolmap 缺失：pip install pycolmap（--video 自动链路的 SfM 需要）")
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


def _cmd_single_image(args) -> int:
    """单张照片 → LAM canonical 3DGS 资产（可选直接上妆）——零门槛入口（P4）。"""
    from .appearance import lam_adapter as lam
    from .appearance.pipeline import build_asset_from_image

    ply = Path(args.ply) if args.ply else None
    if ply is None and not lam.status().ok:
        print(f"LAM 环境不完整：{lam.status().missing_hint}\n"
              "  视频链路不受影响：face3dgs asset -v 视频 -p 工程\n"
              "  已跑过推理的话用 --ply 指定高斯 ply 直接复跑上妆")
        return 2
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8")) if args.spec else None

    def cb(stage, frac, msg):
        print(f"[{stage:>7}] {frac*100:5.1f}%  {msg}", flush=True)

    out = Path(args.project) / "asset"
    cloud, _landmarks, report = build_asset_from_image(
        args.image, out, spec=spec, ply=ply, template=args.template, tex=args.tex,
        intensity=args.intensity, reference=args.reference, progress=cb)
    print(f"单图资产完成（base_source={report['base_source']}，"
          f"{report['splats']} splats）→ {out / 'base.ply'}")
    if report.get("makeup_zones"):
        print("  语义妆区（single_seg）：", report["makeup_zones"])
    if spec:
        print(f"  妆容渲染 → {out / 'madeup.ply'}")
    print(f"  后续：face3dgs render -p {args.project}"
          f"{' --spec ' + args.spec if args.spec else ' --spec 妆容.json'}")
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


def _cmd_pack_bake(args) -> int:
    """妆容 spec → canonical UV 图集 pack（跨用户可移植，④）。"""
    import json
    from .appearance.makeup_pack import bake_preset_pack
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    pack = bake_preset_pack(spec, tex=args.tex, intensity=args.intensity)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pack.save(out)
    print(f"preset pack 已烘焙：{out}（tex={pack.tex}，无用户绑定）\n"
          f"跨用户应用：bind_uv(用户云) → pack.with_binding(uv, valid) → "
          f"apply_pack_to_cloud / 像素渲染")
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
    denoise = {"auto": None, "on": True, "off": False}[args.denoise]
    print(f"离线渲染：{ply} → {out}{'（真实位姿环绕+SH）' if args.sfm else '（DC-only 窄幅环绕）'}")
    outs = deliver(ply, out, base_ply=base_ply, landmarks_path=landmarks,
                   sfm_dir=_Path(args.sfm) if args.sfm else None,
                   n_frames=args.frames, size=args.size, ssaa=args.ssaa,
                   denoise=denoise, background=args.background, light=args.light,
                   force=args.force,
                   pixel_makeup={"auto": None, "on": True,
                                 "off": False}[args.pixel_makeup],
                   composite_mode=args.composite,
                   sigma=args.sigma, skip_video=args.no_video, progress=cb)
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

    p = sub.add_parser("single-image",
                       help="单张照片 → LAM canonical 3DGS 资产（可选直接上妆）")
    p.add_argument("-i", "--image", required=True, help="正面照（jpg/png）")
    p.add_argument("-p", "--project", required=True, help="工程目录（产物落 <p>/asset/）")
    p.add_argument("--ply", help="已有的 LAM 高斯 ply（跳过推理，直接配准+上妆）")
    p.add_argument("--spec", help="妆容 spec json（给定时直接上妆出 madeup.ply）")
    p.add_argument("--reference", help="妆效参考图（P2 参考驱动颜色标定）")
    p.add_argument("--template", help="LAM 帧下的 canonical 468 地标模板 .npy（缺省自动）")
    p.add_argument("--tex", type=int, default=2048, help="UV 图集边长（默认 2048）")
    p.add_argument("--intensity", type=float, default=0.8)
    p.set_defaults(func=_cmd_single_image)

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
    p.add_argument("--size", type=int, default=1024,
                   help="输出边长（默认 1024；1080 交付档用 --size 1080 --ssaa 2）")
    p.add_argument("--ssaa", type=int, default=3, help="超采样倍数（默认 3）")
    p.add_argument("--background", default="white",
                   help="背景：white/studio/warm/cold/transparent（默认 white）")
    p.add_argument("--light", default="capture",
                   help="环境光：capture（采集光）/studio/warm/cold/beauty")
    p.add_argument("--denoise", default="auto", choices=["auto", "on", "off"],
                   help="泼溅颗粒滤波：auto=按质量分级（A 关其余开）")
    p.add_argument("--force", action="store_true",
                   help="越过质量门禁（C 级资产仅诊断用途）")
    p.add_argument("--pixel-makeup", default="auto", choices=["auto", "on", "off"],
                   help="逐像素妆容合成（2048² pack 采样 + Beer-Lambert）："
                        "auto=存在 makeup_maps.npz 即启用")
    p.add_argument("--composite", default="beer", choices=["beer", "linear"],
                   help="妆层合成模型：beer=颜料吸收（默认）/ linear=旧壳层线性 alpha")
    p.add_argument("--sigma", type=float, default=None,
                   help="Beer-Lambert 吸收系数（默认用 pack 内置 2.0）")
    p.add_argument("--no-video", action="store_true",
                   help="跳过环绕视频（只出定妆照/对比图）")
    p.set_defaults(func=_cmd_render)

    p = sub.add_parser("pack-bake", help="妆容 spec → canonical UV 图集 pack"
                                         "（跨用户可移植）")
    p.add_argument("--spec", required=True, help="妆容 spec json")
    p.add_argument("-o", "--out", required=True, help="输出 .pack.npz 路径")
    p.add_argument("--tex", type=int, default=2048)
    p.add_argument("--intensity", type=float, default=1.0)
    p.set_defaults(func=_cmd_pack_bake)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
