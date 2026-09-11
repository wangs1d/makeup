#!/usr/bin/env python3
"""能力二：解析用户上传的妆容照片/视频 → 结构化 makeup_spec.json。

流程：抽帧选帧 → 逐帧 VLM 妆容解析 → 多帧汇总 → 规范化为 spec（含默认化妆步骤）。

用法：
    python parse_look.py --input 用户素材.jpg --out out/parsed
    python parse_look.py --input 用户素材.mp4 --out out/parsed --max-frames 6
    python parse_look.py --input look.png --out out/parsed --focus eyeshadow,lipstick
    python parse_look.py --input look.png --out out/parsed --test   # 离线 canned 流程验证

产出（--out 目录下）：
    makeup_spec.json   可直接 apply_spec.py 试妆
    analysis.md        人话版妆面报告（用于向用户复述确认）
    frames/            参与分析的帧
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vlm  # noqa: E402

SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = SKILL_DIR / "references" / "prompt-templates" / "parse_look.md"

CANONICAL_ORDER = ["foundation", "concealer", "contour", "eyebrow", "eyeshadow",
                   "eyeliner", "lashes", "blush", "highlight", "lipstick"]

VALID_REGIONS = set(CANONICAL_ORDER)
VALID_FINISH = {"matte", "satin", "dewy", "gloss"}
HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


# ---------------- 提示词模板 ----------------

def _extract_block(md: str, heading_prefix: str) -> str | None:
    """从模板 md 中取出某小节的第一个 fenced code block。"""
    for section in md.split("\n## "):
        if section.strip().startswith(heading_prefix):
            m = re.search(r"```\n(.*?)\n```", section, re.DOTALL)
            if m:
                return m.group(1).strip()
    return None


def load_prompts(focus: list[str] | None) -> tuple[str, str]:
    try:
        md = TEMPLATE_PATH.read_text(encoding="utf-8")
        system = _extract_block(md, "单帧解析")
        merge = _extract_block(md, "多帧汇总")
    except OSError:
        system = merge = None
    if not system or not merge:
        fail(f"提示词模板缺失或不完整：{TEMPLATE_PATH}")
    focus_line = (f"用户只想关注这些部位：{'、'.join(focus)}。只输出这些 region，其余一律不要。"
                  if focus else "")
    return system.replace("{FOCUS}", focus_line).strip(), merge.strip()


# ---------------- 抽帧与选帧 ----------------

def is_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def video_frames(path: Path, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        fail(f"无法打开视频：{path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    picks: list[np.ndarray] = []
    if total > 0:
        step = max(1, total // (max_frames * 4))
        candidates = list(range(0, total, step))[: max_frames * 4]
    else:  # 时长未知则顺序读
        candidates = list(range(max_frames * 4))
    for idx in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        picks.append(frame)
        if len(picks) >= max_frames * 4:
            break
    cap.release()
    return picks


def score_frame(frame: np.ndarray) -> float:
    """清晰度（Laplacian 方差）为主，亮度适中加成。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    bright = gray.mean()
    light_ok = 1.0 - min(abs(bright - 125.0) / 125.0, 1.0)
    return sharp * (0.5 + 0.5 * light_ok)


def pick_frames(path: Path, out_dir: Path, max_frames: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_video(path):
        frames = video_frames(path, max_frames)
        if not frames:
            fail("视频里没有读到可用的帧")
        ranked = sorted(frames, key=score_frame, reverse=True)
        chosen = ranked[:max_frames]
        source = f"video:{path.name}"
    else:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            fail(f"无法读取图片：{path}")
        chosen = [img]
        source = f"image:{path.name}"
    saved = []
    for i, frame in enumerate(chosen):
        p = out_dir / "frames" / f"{path.stem}_frame{i:02d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(p))
        saved.append(p)
    print(f"[parse] 已从 {source} 选出 {len(saved)} 帧参与分析")
    return saved


# ---------------- VLM 解析 ----------------

def analyze_frame(image_path: Path, system_prompt: str) -> dict:
    b64 = vlm.encode_image_b64(image_path)
    raw = vlm.chat_with_image(system_prompt, b64)
    return vlm.extract_json(raw)


def merge_frames(per_frame: list[dict], merge_prompt: str) -> dict:
    payload = json.dumps(per_frame, ensure_ascii=False, indent=1)
    raw = vlm.chat([{"role": "user", "content": f"{merge_prompt}\n\n{payload}"}],
                   temperature=0.1)
    return vlm.extract_json(raw)


# ---------------- 规范化 ----------------

def _norm_hex(c: dict) -> dict | None:
    if not isinstance(c, dict):
        return None
    hexv = c.get("hex")
    if not isinstance(hexv, str) or not HEX_RE.match(hexv.strip()):
        return None
    at = c.get("at", 0.0)
    try:
        at = min(max(float(at), 0.0), 1.0)
    except (TypeError, ValueError):
        at = 0.0
    return {"at": at, "hex": ("#" + hexv.strip().lstrip("#")).upper()}


def normalize_layer(layer: dict, seen: set) -> dict | None:
    region = layer.get("region")
    if region not in VALID_REGIONS:
        return None
    side = layer.get("side", "both")
    if side not in {"both", "left", "right"}:
        side = "both"
    key = (region, side)
    if key in seen:
        return None
    seen.add(key)
    stops = [s for s in (_norm_hex(c) for c in (layer.get("color_stops") or [])) if s]
    if not stops:
        stops = [{"at": 0.0, "hex": "#C88D7A"}, {"at": 1.0, "hex": "#B0705F"}]
    stops.sort(key=lambda s: s["at"])
    if len(stops) == 1:
        stops = [stops[0], dict(stops[0], at=1.0)]
    try:
        opacity = min(max(float(layer.get("opacity", 0.7)), 0.0), 1.0)
    except (TypeError, ValueError):
        opacity = 0.7
    finish = layer.get("finish") if layer.get("finish") in VALID_FINISH else "satin"
    render = layer.get("render") if isinstance(layer.get("render"), dict) else {}
    if render.get("type") not in {"mesh", "splat"}:
        render = {"type": "splat"} if region == "lipstick" and finish in {"gloss", "dewy"} \
            else {"type": "mesh"}
    out = {
        "id": f"{region}-{side}",
        "region": region,
        "side": side,
        "enabled": True,
        "opacity": round(opacity, 2),
        "finish": finish,
        "color_stops": stops,
        "texture_strength": min(max(float(layer.get("texture_strength", 0.3)), 0.0), 1.0),
        "render": render,
    }
    if isinstance(layer.get("shape"), dict) and layer["shape"]:
        out["shape"] = layer["shape"]
    if isinstance(layer.get("notes"), str) and layer["notes"].strip():
        out["notes"] = layer["notes"].strip()
    return out


def default_steps(layers: list[dict]) -> list[dict]:
    """解析结果默认化妆步骤：按部位顺序，引用层 notes 作为指导要点。"""
    group = {"foundation": ("base", 1), "concealer": ("base", 1), "contour": ("base", 1),
             "eyebrow": ("eyebrow", 2), "eyeshadow": ("eye", 3), "eyeliner": ("eye", 3),
             "lashes": ("eye", 3), "blush": ("blush", 4), "highlight": ("blush", 4),
             "lipstick": ("lip", 5)}
    by_area: dict[str, dict] = {}
    for layer in layers:
        area, order = group[layer["region"]]
        slot = by_area.setdefault(area, {"order": order, "area": area, "regions": [], "hints": []})
        if layer["region"] not in slot["regions"]:
            slot["regions"].append(layer["region"])
        if layer.get("notes"):
            slot["hints"].append(layer["notes"])
    steps = []
    for i, slot in enumerate(sorted(by_area.values(), key=lambda s: s["order"]), start=1):
        inst = "；".join(slot["hints"]) if slot["hints"] else f"完成{slot['area']}部位"
        steps.append({"order": i, "area": slot["area"], "regions": slot["regions"],
                      "instruction": inst})
    return steps


def build_spec(merged: dict, frames_used: int, src_name: str, focus: list[str] | None) -> dict:
    style = str(merged.get("style", "未命名妆容")).strip() or "解析妆容"
    try:
        intensity = min(max(float(merged.get("overall_intensity", 0.75)), 0.0), 1.0)
    except (TypeError, ValueError):
        intensity = 0.75
    try:
        confidence = min(max(float(merged.get("confidence", 0.6)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.6

    seen: set = set()
    layers = []
    for layer in (merged.get("layers") or []):
        if not isinstance(layer, dict):
            continue
        norm = normalize_layer(layer, seen)
        if norm:
            layers.append(norm)
    if not layers:
        fail("VLM 未能从素材中识别出任何妆容部位（layers 为空）。可换更清晰的素材重试。")
    layers.sort(key=lambda l: CANONICAL_ORDER.index(l["region"]))
    if focus:
        wanted = set(focus)
        layers = [l for l in layers if l["region"] in wanted]
        if not layers:
            fail(f"--focus 限定后没有剩余层（识别到：{[l['region'] for l in layers]}）")
    intensity = round(intensity * (len(layers) / max(len(seen), 1)) ** 0.3, 2) if focus else intensity

    slug = unicodedata.normalize("NFKD", src_name).encode("ascii", "ignore").decode().strip() or "look"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", slug.lower()).strip("-") or "look"
    return {
        "spec_version": "1.0",
        "name": f"parsed-{slug}",
        "description": f"{style}（由 {src_name} 解析）",
        "intensity": intensity,
        "layers": layers,
        "steps": default_steps(layers),
        "source": {
            "type": "parsed",
            "origin": src_name,
            "created_at": __import__("datetime").date.today().isoformat(),
            "style": style,
            "confidence": round(confidence, 2),
            "frame_count": frames_used,
        },
    }


def write_analysis(spec: dict, out_path: Path, preview_path: Path | None = None) -> None:
    src = spec["source"]
    lines = [
        f"# 妆面解析报告：{src.get('style', spec['name'])}",
        "",
        f"- 素材来源：{src.get('origin')}（{src.get('frame_count')} 帧参与分析）",
        f"- 整体妆感浓度：{spec['intensity']:.0%}，解析置信度：{src.get('confidence', '-'):.0%}",
        f"- 一句话：{spec['description']}",
    ]
    if preview_path:
        lines.append(f"- 效果预览：`{preview_path.name}`（左无妆 / 右同款妆，渲染示意）")
    lines += [
        "",
        "| 部位 | 侧 | 质感 | 显色 | 颜色 | 画法 |",
        "|---|---|---|---|---|---|",
    ]
    cn = {"foundation": "粉底", "concealer": "遮瑕", "contour": "修容", "eyebrow": "眉",
          "eyeshadow": "眼影", "eyeliner": "眼线", "lashes": "睫毛", "blush": "腮红",
          "highlight": "高光", "lipstick": "唇"}
    for l in spec["layers"]:
        colors = " → ".join(c["hex"] for c in l["color_stops"])
        lines.append(f"| {cn.get(l['region'], l['region'])} | {l['side']} | {l['finish']} "
                     f"| {l['opacity']:.0%} | {colors} | {l.get('notes', '-')} |")
    lines += ["", "## 建议化妆顺序", ""]
    for s in spec["steps"]:
        lines.append(f"{s['order']}. **{s['area']}**：{s['instruction']}")
    lines += ["", f"试妆：`python apply_spec.py --spec {spec['name']}.json`", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def fail(msg: str) -> None:
    print(f"[parse] {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------- main ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="解析妆容照片/视频为 makeup_spec.json")
    ap.add_argument("--input", required=True, help="素材路径（图片或视频）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--max-frames", type=int, default=4, help="最多分析几帧（默认 4）")
    ap.add_argument("--focus", help="只关注部分部位，逗号分隔 region 名")
    ap.add_argument("--test", action="store_true", help="离线 canned 模式，不调 VLM（流程验证用）")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        fail(f"素材不存在：{src}")
    out_dir = Path(args.out)
    focus = [r.strip() for r in args.focus.split(",") if r.strip()] if args.focus else None
    if focus and (bad := set(focus) - VALID_REGIONS):
        fail(f"--focus 含未知 region：{sorted(bad)}；可用：{sorted(VALID_REGIONS)}")

    frames = pick_frames(src, out_dir, max(1, args.max_frames))
    system_prompt, merge_prompt = load_prompts(focus)

    if args.test:
        canned = {
            "style": "清透日常",
            "overall_intensity": 0.6, "confidence": 0.88,
            "layers": [
                {"region": "eyeshadow", "side": "both", "opacity": 0.5, "finish": "matte",
                 "color_stops": [{"at": 0.0, "hex": "#E3B79C"}, {"at": 1.0, "hex": "#B98263"}],
                 "shape": {"spread": 0.5, "height": 0.35, "angle_deg": 5},
                 "notes": "浅棕眼影沿双眼皮褶向上晕染，眼尾略深"},
                {"region": "lipstick", "side": "both", "opacity": 0.8, "finish": "satin",
                 "color_stops": [{"at": 0.0, "hex": "#C9776F"}, {"at": 1.0, "hex": "#B85F62"}],
                 "render": {"type": "splat", "splat": {"thickness": 0.0012, "density": 0.7}},
                 "notes": "豆沙色渐变咬唇"},
            ]}
        merged, used = canned, 1
        print("[parse] --test 模式：跳过 VLM，使用 canned 解析结果")
    else:
        if not vlm.is_configured():
            fail("VLM 未配置：请设置 MAKEUP_VLM_API_KEY（见 SKILL.md「VLM 配置」）")
        print(f"[parse] 逐帧解析中（{len(frames)} 帧，模型 {vlm.VLM_MODEL}）…")
        per_frame = []
        for f in frames:
            try:
                per_frame.append(analyze_frame(f, system_prompt))
            except (ValueError, RuntimeError) as e:
                print(f"[parse] 跳过 {f.name}：{e}")
        if not per_frame:
            fail("所有帧解析失败，请检查素材清晰度或 VLM 配置")
        print(f"[parse] {len(per_frame)} 帧成功，汇总中…")
        merged = per_frame[0] if len(per_frame) == 1 else merge_frames(per_frame, merge_prompt)
        used = len(per_frame)

    spec = build_spec(merged if isinstance(merged, dict) else {}, used, src.stem, focus)
    spec_path = out_dir / "makeup_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    preview_path = render_preview(spec, out_dir / "preview.jpg")
    write_analysis(spec, out_dir / "analysis.md", preview_path)
    print(f"[parse] 完成 ✓\n  spec：{spec_path}\n  报告：{out_dir / 'analysis.md'}"
          + (f"\n  预览：{preview_path}（无妆 | 同款妆 并排，先给用户看再试妆）" if preview_path else "")
          + f"\n  下一步：python apply_spec.py --spec {spec_path} --bake")


def render_preview(spec: dict, out_path: Path) -> Path | None:
    """解析结果 → 无妆|有妆 并排效果图（与 App 同一套渲染规则），失败不影响主流程。"""
    try:
        from preview_render import render_reference
        import numpy as np
        bare = render_reference(dict(spec, layers=[]), None, size=512, intensity=0.0)
        look = render_reference(spec, None, size=512)
        cv2.imencode(".jpg", np.hstack([bare, look]), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(out_path))
        return out_path
    except Exception as e:  # noqa: BLE001
        print(f"[parse] 预览图生成失败（不影响 spec）：{e}")
        return None


if __name__ == "__main__":
    main()
