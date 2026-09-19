"""run_photoreal — 写真试妆主链路 CLI：上传/扫描 → 3DGS 资产 → 妆容渲染。

方式一（上传视频，全自动）：
    python preview/run_photoreal.py --video 我的视频.mp4         --project out/scan --spec makeup-skill/presets/date-rose.json         --out out/photoreal/me

方式二（已有 SfM 工程，分步复用）：
    python preview/run_photoreal.py --project out/real --sfm out/real/sfm/sparse/3         --init out/real/project/face_dense.ply --spec makeup-skill/presets/date-rose.json         --out out/photoreal/d6

换妆只重跑妆容段（资产已存在时自动复用底模，秒级）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "desktop-app"))

from makeupstudio.face3dgs.appearance.pipeline import run_photoreal  # noqa: E402
from makeupstudio.face3dgs.appearance.train_base import TrainConfig  # noqa: E402


def _make_init_cloud(sfm_dir: str, out_ply: str, project: Path) -> None:
    """COLMAP 稀疏点 → 训练初始点云（kNN 尺度 + 点色），供 --video 全自动模式。"""
    import numpy as np
    import pycolmap
    from makeupstudio.face3dgs.splat_io import write_ply
    rec = pycolmap.Reconstruction(sfm_dir)
    pts = np.array([p.xyz for p in rec.points3D.values()], np.float32)
    cols = np.array([p.color for p in rec.points3D.values()], np.float32) / 255.0
    if len(pts) < 500:
        raise RuntimeError(f"SfM 稀疏点过少（{len(pts)}），无法初始化训练")
    import cv2
    fl = cv2.flann_Index(pts, dict(algorithm=1, trees=4, checks=64))
    _nn, d2 = fl.knnSearch(pts, 4, params=dict(checks=64))
    nn3 = np.sqrt(np.maximum(d2[:, 1:4].mean(1), 1e-12))
    cloud = {
        "xyz": pts,
        "scale": (np.log(np.clip(nn3, 1e-8, None))[:, None] * np.ones((1, 3))).astype(np.float32),
        "rot": np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (len(pts), 1)),
        "rgba": np.concatenate([np.clip(cols, 0.01, 0.99), np.full((len(pts), 1), 0.12, np.float32)], 1),
    }
    write_ply(cloud, out_ply)
    print(f"[entry] 初始点云 {len(pts)} 点 → {out_ply}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=None, help="上传视频路径（全自动：抽帧+SfM+训练+妆容）")
    ap.add_argument("--project", default=None, help="采集工程目录（含 capture/frames 或 images）")
    ap.add_argument("--sfm", default=None, help="COLMAP sparse 目录（--video 模式自动生成）")
    ap.add_argument("--init", default=None, help="初始化点云 ply（缺省用 COLMAP 点稠密化）")
    ap.add_argument("--spec", required=True, help="妆容 spec json")
    ap.add_argument("--reference", default=None,
                    help="妆效参考图（有则先逐区域标定 spec 颜色/浓度再上妆，"
                         "并在 report 输出逐区域 ΔE00 还原度）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--max-gs", type=int, default=400_000)
    ap.add_argument("--tex", type=int, default=2048)
    ap.add_argument("--intensity", type=float, default=0.8)
    ap.add_argument("--no-reuse-base", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    if not args.video and not (args.project and args.sfm):
        print("需要 --video（全自动）或 --project+--sfm（已有工程）", file=sys.stderr)
        return 2
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if args.reference:
        import cv2
        from makeupstudio.face3dgs.appearance.calibrate import calibrate_spec
        from makeupstudio.transfer import imread_unicode
        spec = calibrate_spec(spec, imread_unicode(args.reference))
        hit = spec.get("calibration", {}).get("regions", [])
        print(f"[calibrate] 参考图标定完成：{hit or '（未匹配到妆区）'}")
    cfg = TrainConfig(iters=args.iters, max_gs=args.max_gs)

    if args.video:
        from makeupstudio.face3dgs.appearance import sfm
        project = Path(args.project or "out/scan")
        images = project / "images"
        sfm.extract_frames(args.video, images, fps=10.0)
        result = sfm.run_sfm(images, project)
        args.project, args.sfm = str(project), str(result.sparse_dir)
        if not args.init:
            args.init = str(project / "init_dense.ply")
            _make_init_cloud(args.sfm, args.init, project)
    if not (args.project and args.sfm and args.init):
        print("--project/--sfm/--init 缺失（--video 模式会自动生成）", file=sys.stderr)
        return 2

    def cb(stage: str, frac: float, msg: str) -> None:
        print(f"[{stage:>7}] {frac * 100:5.1f}%  {msg}", flush=True)

    print(f"[entry] project={args.project} sfm={args.sfm} init={args.init}")

    run_photoreal(args.project, args.sfm, args.init, spec, args.out,
                  train_cfg=cfg, tex=args.tex, intensity=args.intensity,
                  reuse_base=not args.no_reuse_base, progress=cb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
