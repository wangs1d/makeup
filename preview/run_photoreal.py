"""run_photoreal — 写真试妆主链路 CLI：上传/扫描 → 3DGS 资产 → 妆容渲染。

方式一（上传视频，全自动）：
    python preview/run_photoreal.py --video 我的视频.mp4         --project out/scan --spec makeup-skill/presets/date-rose.json         --out out/photoreal/me

方式二（已有 SfM 工程，分步复用）：
    python preview/run_photoreal.py --project out/real --sfm out/real/sfm/sparse/3         --init out/real/project/face_dense.ply --spec makeup-skill/presets/date-rose.json         --out out/photoreal/d6

方式三（单张正面照，零门槛）：
    python preview/run_photoreal.py --image 我的照片.jpg         --spec makeup-skill/presets/date-rose.json --out out/lam/me
    → LAM（aigc3d，SIGGRAPH 2025）单图回归 canonical 高斯；底模逼真度低于视频链路
      （report["honesty"] 如实标注），语义妆区只有一帧观测（single_seg 级）。

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


def _run_single_image(args) -> int:
    """方式三：单张正面照 → LAM canonical 3DGS 资产 → 妆容（P4 零门槛入口）。

    与视频链路的差别只在底模来源；UV 绑定/目标场/壳层/材质/离线渲染完全复用。
    环境缺失时给可执行提示并返回 2，视频主链路不受影响。"""
    from makeupstudio.face3dgs.appearance import lam_adapter as lam
    from makeupstudio.face3dgs.appearance.pipeline import build_asset_from_image

    st = lam.status()
    if not st.ok:
        print(f"LAM 环境不完整：{st.missing_hint}\n"
              "  视频链路不受影响：python preview/run_photoreal.py "
              "--video 视频.mp4 --spec 妆容.json --out out/photoreal/me", file=sys.stderr)
        return 2
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out = Path(args.out)
    asset = out / "asset"

    def cb(stage: str, frac: float, msg: str) -> None:
        print(f"[{stage:>7}] {frac * 100:5.1f}%  {msg}", flush=True)

    print(f"[entry] image={args.image} out={out}", flush=True)
    _cloud, _lm, report = build_asset_from_image(
        args.image, asset, spec=spec, tex=args.tex, intensity=args.intensity,
        reference=args.reference, progress=cb)
    # 落一份 report.json 到 --out，与视频链路同位置（下游工具按 --out/report.json 找）
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[quality] base_source={report['base_source']} / "
          f"{report['splats']} splats / 地标配准 rmse="
          f"{report['lam']['landmarks_rmse']:.4f}")
    if report.get("makeup_zones"):
        print(f"[zones] {report['makeup_zones']}")
    print(f"[honesty] {report['honesty']}")
    print(f"完成 → {asset / 'madeup.ply'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=None, help="上传视频路径（全自动：抽帧+SfM+训练+妆容）")
    ap.add_argument("--image", default=None,
                    help="单张正面照（LAM 单图入口 → canonical 3DGS 资产 → 妆容）")
    ap.add_argument("--project", default=None, help="采集工程目录（含 capture/frames 或 images）")
    ap.add_argument("--sfm", default=None, help="COLMAP sparse 目录（--video 模式自动生成）")
    ap.add_argument("--init", default=None, help="初始化点云 ply（缺省用 COLMAP 点稠密化）")
    ap.add_argument("--spec", required=True, help="妆容 spec json")
    ap.add_argument("--reference", default=None,
                    help="妆效参考图（有则先逐区域标定 spec 颜色/浓度再上妆，"
                         "并在 report 输出逐区域 ΔE00 还原度）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--max-gs", type=int, default=900_000)
    ap.add_argument("--tex", type=int, default=2048)
    ap.add_argument("--intensity", type=float, default=0.8)
    ap.add_argument("--no-reuse-base", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--no-auto-cal", action="store_true",
                    help="关闭 ΔE 超阈值的自动重标定闭环（默认开）")
    ap.add_argument("--look2d", default=None,
                    help="2D 人台渲染图路径：同一 spec 的 2D 预览逐区域 ΔE00 "
                         "进 report（2D/3D 一致性交叉检查）")
    ap.add_argument("--preview-2d", action="store_true",
                    help="EleGANt 就绪时产出参考帧的 2D 迁移预览 preview_2d.png")
    args = ap.parse_args()

    if args.image:                      # 方式三：单图入口（不预标定 spec，交给管线做）
        return _run_single_image(args)
    if not args.video and not (args.project and args.sfm):
        print("需要 --video（全自动）或 --project+--sfm（已有工程）或 --image（单图）",
              file=sys.stderr)
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
                  reuse_base=not args.no_reuse_base,
                  auto_calibrate=not args.no_auto_cal,
                  preview_2d=args.preview_2d,
                  look2d=args.look2d, reference=args.reference, progress=cb)
    report_path = Path(args.out) / "report.json"
    if report_path.exists():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        q, gate = rep.get("quality") or {}, rep.get("fidelity_gate") or {}
        print(f"[quality] 分级 {q.get('grade')}（短边 {q.get('min_side')}px / "
              f"PSNR {q.get('psnr')}dB / {q.get('splats')} splats）"
              + ("；" + "；".join(q.get("reasons", [])) if q.get("reasons") else ""))
        print(f"[fidelity] ΔE00 mean={rep.get('makeup_delta_e', {}).get('_mean', 'n/a')} "
              f"gate={gate.get('status')}"
              + (f" 超预算区域={gate.get('over')}" if gate.get("over") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
