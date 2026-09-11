#!/usr/bin/env python3
"""render_look — 把 makeup_spec.json 渲染成静态"目标妆效果图"（免 Unity，纯 Python）。

用途：
    · live_coach.py 启动时生成目标妆参考图，与摄像头帧一起给 VLM 做视觉对比；
    · parse_look.py 解析后生成预览图，让用户先确认"同款"再试妆；
    · 任何机器上验证 spec 的妆效（与 Unity App 同一套蒙版/溅射/光照规则）。

用法：
    python render_look.py --spec presets/date-rose.json --out out/date-rose.jpg
    python render_look.py --spec look.json --out a.jpg --env warm --yaw 15 --intensity 0.7
    python render_look.py --spec look.json --out grid.jpg --compare      # 无妆 | 有妆 并排
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preview_render import ENVS, render_reference  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="渲染妆容效果图")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True, help="输出图片路径（jpg/png）")
    ap.add_argument("--size", type=int, default=640, help="宽度像素（默认 640）")
    ap.add_argument("--env", default="neutral", choices=sorted(ENVS), help="环境光预设")
    ap.add_argument("--yaw", type=float, default=0.0, help="偏航角（度）")
    ap.add_argument("--intensity", type=float, default=None, help="覆盖 spec 浓度")
    ap.add_argument("--compare", action="store_true", help="输出 无妆|有妆 并排对比图")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"[render] spec 不存在：{spec_path}", file=sys.stderr)
        sys.exit(1)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    img = render_reference(spec, None, size=args.size, env=args.env,
                           intensity=args.intensity, yaw_deg=args.yaw)
    if args.compare:
        bare = render_reference(dict(spec, layers=[]), None, size=args.size, env=args.env,
                                intensity=0.0, yaw_deg=args.yaw)
        img = np.hstack([bare, img])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        print(f"[render] 写图失败：{out}", file=sys.stderr)
        sys.exit(1)
    print(f"[render] 完成 ✓ {out}（{img.shape[1]}x{img.shape[0]}，env={args.env}）")


if __name__ == "__main__":
    main()
