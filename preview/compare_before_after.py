"""compare_before_after — 优化前后资产的并排对比图（同相机同取景）。

对两个 3DGS 工程目录（如 out/photoreal/d6-cap vs out/photoreal/d7）在**同一
SfM 相机**下分别渲染素颜与妆后资产，拼成带标签的对比图：

    行1  修复前·素颜 | 修复后·素颜      ← 底模质量（牙齿/碎斑/轮廓）
    行2  修复前·妆后 | 修复后·妆后      ← 妆效（粉感/唇妆/边缘）
    行3  唇部特写·修复前 | 修复后       ← 高频区域放大

两个工程的资产必须在同一世界系（同一 SfM 稀疏重建 + 初始化训练），否则
相机不通用。取景框由"修复后"工程的 landmarks.npy 投影决定（裁剪同一矩形，
保证逐像素可比）。

用法：
    python preview/compare_before_after.py \
        --before out/photoreal/d6-cap --after out/photoreal/d7 \
        --sfm out/real/sfm/sparse/3 --out out/compare/before_after_q.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "desktop-app"))

from makeupstudio.face3dgs import colmap_io  # noqa: E402
from makeupstudio.face3dgs.appearance.offline_render import (  # noqa: E402
    build_shade,
    load_prepared,
    render_pose,
)
from makeupstudio.face3dgs.splat_io import read_ply  # noqa: E402
from makeupstudio.transfer import imread_unicode, imwrite_unicode  # noqa: E402

_REFS = Path(__file__).resolve().parent.parent / "makeup-skill" / "references"
_LIPS_OUTER = None


def _lips_outer() -> np.ndarray:
    global _LIPS_OUTER
    if _LIPS_OUTER is None:
        data = json.loads((_REFS / "landmark-regions.json").read_text(encoding="utf-8"))
        _LIPS_OUTER = np.asarray(data["regions"]["lips_outer"]["indices"], np.int64)
    return _LIPS_OUTER


def _project(landmarks: np.ndarray, w2c: np.ndarray, K: np.ndarray) -> np.ndarray:
    """世界系地标 → 画布像素（gsplat 预览路径 = 原帧像素坐标的恒等映射）。"""
    hom = np.concatenate([np.asarray(landmarks, np.float64),
                          np.ones((len(landmarks), 1))], axis=1)
    cam = (np.asarray(w2c, np.float64)[None, :3, :] @ hom[:, :, None])[..., 0]
    ok = cam[:, 2] > 1e-6
    px = np.full((len(landmarks), 2), np.nan)
    px[ok, 0] = K[0, 0] * cam[ok, 0] / cam[ok, 2] + K[0, 2]
    px[ok, 1] = K[1, 1] * cam[ok, 1] / cam[ok, 2] + K[1, 2]
    return px


def _rect_from(px: np.ndarray, size: int, margin: float = 0.16) -> tuple[int, int, int, int]:
    v = px[np.isfinite(px).all(1)]
    x0, y0 = np.percentile(v[:, 0], 1), np.percentile(v[:, 1], 1)
    x1, y1 = np.percentile(v[:, 0], 99), np.percentile(v[:, 1], 99)
    w, h = x1 - x0, y1 - y0
    x0, y0 = x0 - w * margin, y0 - h * margin
    x1, y1 = x1 + w * margin, y1 + h * margin
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), size), min(int(y1), size)
    return x0, y0, x1, y1


def _label_strip(text: str, width: int, height: int = 44) -> np.ndarray:
    strip = np.full((height, width, 3), 255, np.uint8)
    from makeupstudio.transfer import _label_font
    font = _label_font(24)
    if font is not None:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(strip)
        draw = ImageDraw.Draw(pil)
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - (box[2] - box[0])) / 2 - box[0],
                   (height - (box[3] - box[1])) / 2 - box[1]),
                  text, font=font, fill=(29, 29, 31))
        return np.array(pil)
    cv2.putText(strip, text, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (29, 29, 31), 2, cv2.LINE_AA)
    return strip


def _panel(img: np.ndarray, rect: tuple[int, int, int, int], width: int,
           label: str) -> np.ndarray:
    x0, y0, x1, y1 = rect
    crop = img[y0:y1, x0:x1]
    h = max(1, round(crop.shape[0] * width / crop.shape[1]))
    crop = cv2.resize(crop, (width, h), interpolation=cv2.INTER_AREA)
    return np.vstack([_label_strip(label, width), crop])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", required=True, help="修复前工程目录（含 base/madeup.ply）")
    ap.add_argument("--after", required=True, help="修复后工程目录")
    ap.add_argument("--sfm", required=True, help="COLMAP sparse 目录（两工程共用的世界系）")
    ap.add_argument("--out", default="out/compare/before_after.png")
    ap.add_argument("--size", type=int, default=760, help="渲染边长")
    ap.add_argument("--width", type=int, default=620, help="每栏显示宽度")
    args = ap.parse_args()

    model = colmap_io.read_sparse(args.sfm)
    rep = json.loads((Path(args.after) / "report.json").read_text(encoding="utf-8"))
    frame = rep.get("ref_frame") or next(iter(model.images))
    im = model.images[frame]
    R = colmap_io.quat_to_rotmat(im["qvec"])
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = np.asarray(im["tvec"], np.float64)
    cam = model.camera
    K = np.array([[cam.params[0], 0, cam.params[1]],
                  [0, cam.params[0], cam.params[2]],
                  [0, 0, 1]], np.float64)
    print(f"[cmp] 相机 = {frame}（两工程同世界系）")

    lm_after = np.load(Path(args.after) / "landmarks.npy")
    px = _project(lm_after, w2c, K)
    face_rect = _rect_from(px, args.size)
    lip_rect = _rect_from(px[_lips_outer()], args.size, margin=0.7)

    panels: dict[tuple[str, str], np.ndarray] = {}
    for tag, pdir in (("before", args.before), ("after", args.after)):
        d = Path(pdir)
        for kind, ply in (("bare", "base.ply"), ("made", "madeup.ply")):
            cloud = read_ply(d / ply)
            shade = build_shade(cloud, mat_dir=d)
            img = render_pose(load_prepared(cloud, use_sh=True), w2c, K,
                              size=args.size, ssaa=1, shade=shade)
            panels[(tag, kind)] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print(f"[cmp] 渲染 {tag}/{kind}: {len(cloud['xyz'])} splats")

    w = args.width
    rows = [
        np.hstack([_panel(panels[("before", "bare")], face_rect, w, "修复前 · 素颜"),
                   _panel(panels[("after", "bare")], face_rect, w, "修复后 · 素颜")]),
        np.hstack([_panel(panels[("before", "made")], face_rect, w, "修复前 · 妆后"),
                   _panel(panels[("after", "made")], face_rect, w, "修复后 · 妆后")]),
        np.hstack([_panel(panels[("before", "made")], lip_rect, w, "唇部特写 · 修复前"),
                   _panel(panels[("after", "made")], lip_rect, w, "唇部特写 · 修复后")]),
    ]
    gap = 6
    width = max(r.shape[1] for r in rows)
    height = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1)
    canvas = np.full((height, width, 3), 255, np.uint8)
    y = 0
    for r in rows:
        canvas[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + gap
    imwrite_unicode(args.out, canvas)
    print(f"[cmp] 对比图 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
