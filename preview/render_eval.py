"""render_eval — 用 gsplat 真实渲染器评测底模/妆后资产（R+ 验证工具）。

render_cloud_pbr（numpy 圆核）只做链路诊断，保真度不足以评判渲染质量；
本工具走与训练/Unity 相同的 EWA+SH 光栅化：
    python preview/render_eval.py --project out/real --sfm out/real/sfm/sparse/3 \
        --ply out/photoreal/d6/base.ply out/photoreal/d6-v2/base.ply \
        --labels baseline deg2 --out out/eval
输出：eval_views 渲染网格（真实帧 | 各版本）+ 掩码 PSNR 表（脸区）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "desktop-app"))

import cv2
import numpy as np


def load_cloud_for_gsplat(ply: str):
    from makeupstudio.face3dgs.splat_io import read_ply
    import torch
    c = read_ply(ply)
    sh = c.get("sh_rest")
    if sh is not None:
        deg = int(round(np.sqrt(sh.shape[1] + 1))) - 1        # pc=8→deg2, pc=3→deg1
        colors = np.zeros((len(c["xyz"]), (deg + 1) ** 2, 3), np.float32)
        colors[:, 0, :] = (c["rgba"][:, :3] - 0.5) / 0.28209479112561376
        colors[:, 1:, :] = sh
    else:
        deg = None
        colors = c["rgba"][:, :3].copy()                      # DC-only：直接 RGB
    # 内部约定 xyzw → gsplat wxyz（顺序错 = 各向异性 splat 全部朝向错误）
    rot_wxyz = np.ascontiguousarray(c["rot"][:, [3, 0, 1, 2]])
    return {k: torch.tensor(c[k], device="cuda") for k in ("xyz", "scale")} | \
        {"rot": torch.tensor(rot_wxyz, device="cuda"),
         "colors": torch.tensor(colors, device="cuda"), "opacity": torch.tensor(c["rgba"][:, 3], device="cuda"),
         "deg": deg}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--sfm", required=True)
    ap.add_argument("--ply", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=6)
    args = ap.parse_args()

    import torch
    from gsplat import rasterization
    from makeupstudio.face3dgs import colmap_io

    model = colmap_io.read_sparse(args.sfm)
    names = [n for n in model.images if (Path(args.project) / "capture" / "frames" / n).exists()
             or (Path(args.project) / "images" / n).exists()]
    names = sorted(names)
    pick = [names[int(round(i))] for i in np.linspace(0, len(names) - 1, args.views)]
    clouds = [load_cloud_for_gsplat(p) for p in args.ply]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, psnrs = [], {lb: [] for lb in args.labels}
    for name in pick:
        im = model.images[name]
        R = colmap_io.quat_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], np.float64)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        w2c_t = torch.tensor(w2c, dtype=torch.float32, device="cuda")[None]
        cam = model.camera
        Kt = torch.tensor([[cam.params[0], 0, cam.params[1]],
                           [0, cam.params[0], cam.params[2]],
                           [0, 0, 1]], dtype=torch.float32)[None].to("cuda")
        cells = []
        img_path = next(p for p in (Path(args.project) / "capture" / "frames",
                                    Path(args.project) / "images") if (p / name).exists()) / name
        ref = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        cells.append(cv2.resize(ref, (480, 480)))
        for lb, c in zip(args.labels, clouds):
            r, _a, _i = rasterization(
                c["xyz"], c["rot"], c["scale"], c["opacity"], c["colors"],
                w2c_t, Kt, cam.width, cam.height,
                sh_degree=c["deg"], rasterize_mode="antialiased", packed=False,
                backgrounds=torch.ones(1, 3, device="cuda"))
            img = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            img = cv2.resize(img, (480, 480))
            # 脸区掩码 PSNR：三角化地标投影 bbox 外扩 15%
            lm_path = Path(args.ply[0]).parent / "landmarks.npy"
            m = np.zeros(img.shape[:2], np.uint8)
            if lm_path.exists():
                lm = np.load(lm_path)
                hom = np.concatenate([lm, np.ones((len(lm), 1))], 1) @ w2c.T
                z = np.maximum(hom[:, 2], 1e-6)
                px = cam.params[0] * hom[:, 0] / z + cam.params[1]
                py = cam.params[0] * hom[:, 1] / z + cam.params[2]
                ok = (z > 0.05) & (px >= 0) & (px < cam.width) & (py >= 0) & (py < cam.height)
                if ok.sum() > 50:
                    x0, x1 = np.percentile(px[ok], [1, 99])
                    y0, y1 = np.percentile(py[ok], [1, 99])
                    w_, h_ = x1 - x0, y1 - y0
                    x0, x1 = int(max(0, x0 - 0.15 * w_)), int(min(cam.width, x1 + 0.15 * w_))
                    y0, y1 = int(max(0, y0 - 0.15 * h_)), int(min(cam.height, y1 + 0.15 * h_))
                    m[y0:y1, x0:x1] = 1
            if m.any():
                gt = cv2.cvtColor(cv2.resize(ref, (cam.width, cam.height)), cv2.COLOR_BGR2RGB)
                mse = ((r[0].clamp(0, 1).cpu().numpy() - gt / 255.0) ** 2)[m > 0].mean()
                psnrs[lb].append(float(-10 * np.log10(mse)))
            cells.append(img)
        rows.append(np.concatenate(cells, axis=1))
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_dir / "eval_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    report = {lb: {"psnr_mean": round(float(np.mean(v)), 2),
                   "psnr_per_view": [round(float(x), 2) for x in v]}
              for lb, v in psnrs.items() if v}
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"→ {out_dir / 'eval_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
