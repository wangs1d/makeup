"""face3dgs CLI — 用户脸部 3DGS 化与妆容贴合的命令行入口。

    py -m makeupstudio.face3dgs status
    py -m makeupstudio.face3dgs capture  -o out/face3dgs/me.mp4 [--cam 0]
    py -m makeupstudio.face3dgs rebuild  -v me.mp4 -p out/face3dgs/proj --quality standard
    py -m makeupstudio.face3dgs isolate  -p out/face3dgs/proj -o out/face3dgs/face.ply
    py -m makeupstudio.face3dgs fit      -p out/face3dgs/proj --spec makeup-skill/presets/date-rose.json \
                                         -f out/face3dgs/face.ply -o out/face3dgs/fitted
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import engines
from .capture import capture_from_camera
from .isolate import isolate_face
from .reconstruct import ReconstructionError, run_reconstruction


def _cmd_status(_args) -> int:
    st = engines.status_all()
    for name, s in st.items():
        mark = "OK " if s.ok else "缺失"
        print(f"[{mark}] {name:8s} {s.path or ''} {s.error or s.version or ''}")
    return 0 if all(s.ok for s in st.values()) else 1


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


def _cmd_rebuild(args) -> int:
    def on_p(stage, frac, msg):
        print(f"\r[{stage:8s}] {frac*100:5.1f}% {msg[:90]:<90}", end="", flush=True)
    try:
        res = run_reconstruction(args.video, args.project, quality=args.quality,
                                 on_progress=on_p)
    except (ReconstructionError, RuntimeError) as e:
        print(f"\n重建失败: {e}")
        return 2
    print(f"\nfinal.ply: {res.ply_path}")
    return 0


def _cmd_isolate(args) -> int:
    from .reconstruct import ReconResult
    proj = Path(args.project)
    result = ReconResult(project_dir=proj, ply_path=proj / "final.ply",
                         sparse_dir=proj / "colmap" / "sparse" / "0",
                         images_dir=proj / "images", seconds=0.0)
    stats = isolate_face(result, args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 0


def _cmd_fit(args) -> int:
    from .fit_makeup import FaceMakeupFitter
    from .reconstruct import ReconResult
    proj = Path(args.project)
    result = ReconResult(project_dir=proj, ply_path=proj / "final.ply",
                         sparse_dir=proj / "colmap" / "sparse" / "0",
                         images_dir=proj / "images", seconds=0.0)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    fitter = FaceMakeupFitter()
    fit = fitter.fit(result, spec, args.out, face_ply=args.face, intensity=args.intensity)
    print(f"妆容贴合完成: {fit.ply_path}\n"
          f"  视角 {fit.n_views_used} | 配准 RMSE(canonical) {fit.fit_rmse:.4f} | "
          f"splats {len(fit.cloud['xyz'])}")
    for p in fit.previews:
        print(f"  预览: {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="face3dgs", description="用户脸部 3DGS 化 + 妆容贴合")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="检查 FFmpeg/COLMAP/Brush 引擎").set_defaults(func=_cmd_status)

    p = sub.add_parser("capture", help="摄像头环绕采集引导")
    p.add_argument("-o", "--out", required=True, help="输出 MP4 路径")
    p.add_argument("--cam", type=int, default=0)
    p.set_defaults(func=_cmd_capture)

    p = sub.add_parser("rebuild", help="视频 → 3DGS（FFmpeg+COLMAP+Brush）")
    p.add_argument("-v", "--video", required=True)
    p.add_argument("-p", "--project", required=True)
    p.add_argument("--quality", default="standard", choices=("draft", "standard", "high"))
    p.set_defaults(func=_cmd_rebuild)

    p = sub.add_parser("isolate", help="裁剪出脸部点云 face.ply")
    p.add_argument("-p", "--project", required=True)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(func=_cmd_isolate)

    p = sub.add_parser("fit", help="把妆容 spec 贴合到脸部点云")
    p.add_argument("-p", "--project", required=True)
    p.add_argument("-f", "--face", default=None, help="face.ply（默认用 final.ply）")
    p.add_argument("--spec", required=True, help="妆容 spec JSON（presets/*.json）")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--intensity", type=float, default=0.8)
    p.set_defaults(func=_cmd_fit)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
