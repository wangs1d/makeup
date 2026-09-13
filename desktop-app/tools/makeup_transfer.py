#!/usr/bin/env python3
"""妆效迁移 CLI：素颜照 + 真实妆效参考图 → 迁移对比图。

用法：
    py tools/makeup_transfer.py --source 素颜.jpg --reference 妆效参考.jpg
    py tools/makeup_transfer.py --source bare.png --reference look.png --out out/transfer

输出（默认 desktop-app/out/transfer/）：
    transfer_<时间戳>_compare.png   三联对比图：素颜 | 参考妆效 | 迁移后
    transfer_<时间戳>_result.png    仅妆后结果

引擎为 EleGANt（见 makeupstudio/transfer.py 与 research/INTEGRATION.md），
权重缺失时会打印就绪检查的缺失清单。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from makeupstudio.transfer import (imread_unicode, missing_assets,  # noqa: E402
                                   transfer_and_save)

DEFAULT_OUT = APP / "out" / "transfer"
IMAGE_FILTER = "*.jpg *.jpeg *.png *.bmp *.webp"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="素颜照 + 妆效参考图 → 迁移对比图（EleGANt）")
    ap.add_argument("--source", required=True, help="素颜照路径（清晰正面人脸）")
    ap.add_argument("--reference", required=True, help="妆效参考图路径（清晰正面人脸）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"输出目录（默认 {DEFAULT_OUT}）")
    ap.add_argument("--height", type=int, default=560, help="对比图单栏高度（默认 560）")
    ap.add_argument("--stem", default=None, help="输出文件名主体（默认时间戳）")
    args = ap.parse_args(argv)

    missing = missing_assets()                     # 文件级秒查，先给缺失清单再谈推理
    if missing:
        print("EleGANt 未就绪，缺失项：")
        for m in missing:
            print(f"  {m}")
        return 2

    source = imread_unicode(args.source)
    reference = imread_unicode(args.reference)
    print(f"素颜照 {source.shape[1]}x{source.shape[0]} · 参考图 {reference.shape[1]}x{reference.shape[0]}")

    cmp_path, res_path = transfer_and_save(
        source, reference, args.out, height=args.height, stem=args.stem)
    print(f"迁移对比图：{cmp_path}")
    print(f"妆后结果图：{res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
