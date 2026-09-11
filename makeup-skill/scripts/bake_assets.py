#!/usr/bin/env python3
"""由 makeup_spec.json 烘焙试妆 App 可用的渲染资产。

产出（--out 目录）：
    ramp_<region>_<side>.png   渐变色带（材质采样用，256x8 RGBA）
    grain_<region>_<side>.png  可平铺粉感噪点（256x256 灰度）
    splat_layers.json          3DGS 溅射层配置（唇釉/高光等体积层的锚点与高斯参数）
    baked_spec.json            附带 baked 引用与关键点绑定的最终 spec
    manifest.json              文件清单（apply_spec.py --assets 用它打包传输）

用法：
    python bake_assets.py --spec presets/daily-natural.json --out out/baked
    python bake_assets.py --spec out/parsed/makeup_spec.json --out out/baked --pack
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from preview_render import SPLAT_GROUP_HINTS, build_splats  # noqa: E402

SKILL_DIR = Path(__file__).resolve().parent.parent
LANDMARKS_PATH = SKILL_DIR / "references" / "landmark-regions.json"

RAMP_W, RAMP_H = 256, 8
GRAIN_SIZE = 256

# splat 渲染支持的 region（规则在 preview_render.build_splats，与 App/预览渲染器共用）
SPLAT_REGIONS = set(SPLAT_GROUP_HINTS)


def hex_to_rgb(hexv: str) -> tuple[int, int, int]:
    hexv = hexv.lstrip("#")
    return int(hexv[0:2], 16), int(hexv[2:4], 16), int(hexv[4:6], 16)


def sample_stops(stops: list[dict], t: float) -> tuple[int, int, int]:
    """在 color_stops 渐变上取 t 处的颜色。"""
    pts = [(s["at"], hex_to_rgb(s["hex"])) for s in sorted(stops, key=lambda s: s["at"])]
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for (a0, c0), (a1, c1) in zip(pts, pts[1:]):
        if a0 <= t <= a1:
            f = 0.0 if a1 == a0 else (t - a0) / (a1 - a0)
            return tuple(int(round(c0[i] + (c1[i] - c0[i]) * f)) for i in range(3))  # type: ignore
    return pts[-1][1]


def make_ramp(stops: list[dict]) -> Image.Image:
    img = np.zeros((RAMP_H, RAMP_W, 4), dtype=np.uint8)
    for x in range(RAMP_W):
        r, g, b = sample_stops(stops, x / (RAMP_W - 1))
        img[:, x, :3] = (r, g, b)
        img[:, x, 3] = 255
    return Image.fromarray(img, "RGBA")


def make_grain(strength: float, seed: int) -> Image.Image:
    """可平铺粉感噪点：白噪声 + 轻微模糊 + 2x2 镜像平铺消缝。"""
    rng = np.random.default_rng(seed)
    n = rng.normal(128, 22 + 30 * strength, (GRAIN_SIZE // 2, GRAIN_SIZE // 2))
    img = Image.fromarray(np.clip(n, 0, 255).astype(np.uint8), "L")
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    big = Image.new("L", (GRAIN_SIZE, GRAIN_SIZE))
    h = GRAIN_SIZE // 2
    for fx, flipped_x in ((0, False), (h, True)):
        for fy, flipped_y in ((0, False), (h, True)):
            tile = img
            if flipped_x:
                tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
            if flipped_y:
                tile = tile.transpose(Image.FLIP_TOP_BOTTOM)
            big.paste(tile, (fx, fy))
    return big


def splat_config(layer: dict, lm: dict) -> dict | None:
    """为 splat 渲染层生成高斯溅射配置（规则见 preview_render.build_splats）。

    位置运行时由 App 从人脸关键点取（随脸动），本文件只带拓扑与外观参数。
    不支持的 region 显式提示（旧版会静默丢弃）。
    """
    if layer["region"] not in SPLAT_REGIONS:
        print(f"[bake] 提示：region={layer['region']} 不支持 splat 渲染，退回 mesh 层"
              f"（支持：{sorted(SPLAT_REGIONS)}）", file=sys.stderr)
        return None
    out = build_splats([layer], lm.get("regions", {}))
    return out[0] if out else None


def bake(spec: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not LANDMARKS_PATH.exists():
        fail(f"关键点映射表缺失：{LANDMARKS_PATH}")
    lm = json.loads(LANDMARKS_PATH.read_text(encoding="utf-8"))

    baked_layers, splat_layers, files = [], [], []
    for i, layer in enumerate(spec.get("layers", [])):
        if not layer.get("enabled", True):
            continue
        stem = f"{layer['region']}_{layer.get('side', 'both')}"
        ramp = make_ramp(layer["color_stops"])
        ramp_name = f"ramp_{stem}.png"
        ramp.save(out_dir / ramp_name)

        grain_name = None
        if float(layer.get("texture_strength", 0)) > 0.01:
            grain_name = f"grain_{stem}.png"
            make_grain(float(layer["texture_strength"]), seed=i + 7).save(out_dir / grain_name)
            files.append(grain_name)
        files.append(ramp_name)

        bl = dict(layer, baked={"ramp": ramp_name, **({"grain": grain_name} if grain_name else {})})
        baked_layers.append(bl)

        if layer.get("render", {}).get("type") == "splat":
            sc = splat_config(layer, lm)
            if sc:
                splat_layers.append(sc)

    out_spec = dict(spec, layers=baked_layers, splat_layers=splat_layers)
    spec_path = out_dir / "baked_spec.json"
    spec_path.write_text(json.dumps(out_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "name": spec.get("name", "look"),
        "baked_spec": spec_path.name,
        "intensity": spec.get("intensity", 0.8),
        "files": files,
        "splat_layers": len(splat_layers),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def pack(out_dir: Path, name: str) -> Path:
    zpath = out_dir / f"{name}.mkasset"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.iterdir()):
            if p.suffix in {".png", ".json"} and p.name != zpath.name:
                z.write(p, p.name)
    return zpath


def fail(msg: str) -> None:
    print(f"[bake] {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="烘焙妆容渲染资产")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack", action="store_true", help="额外打包为 <name>.mkasset")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        fail(f"spec 不存在：{spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("spec_version") != "1.0":
        fail("需要 spec_version 1.0")

    out_dir = bake(spec, Path(args.out))
    digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()[:12]
    print(f"[bake] 完成 ✓（{len(list(out_dir.glob('*.png')))} 张纹理，"
          f"splat 层 {len(json.loads((out_dir / 'baked_spec.json').read_text(encoding='utf-8')).get('splat_layers', []))} 个）"
          f"\n  资产目录：{out_dir}（spec 指纹 {digest}）")
    if args.pack:
        z = pack(out_dir, spec.get("name", "look"))
        print(f"  打包：{z}")


if __name__ == "__main__":
    main()
