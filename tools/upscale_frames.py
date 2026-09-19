#!/usr/bin/env python3
"""upscale_frames — Real-ESRGAN x4 帧超分（低清源 → 高清训练素材）。

自实现 RRDBNet 推理（state dict 与 BasicSR/Real-ESRGAN 官方权重键名一致），
不引入 basicsr 依赖。输入帧 480² 全图单次前向（fp16），单卡显存占用 <2GB。

    python tools/upscale_frames.py --in out/real/capture/frames --out out/real-sr/images \
        --names frame_00064.jpg frame_00065.jpg ...     # 缺省全目录
    python tools/upscale_frames.py --weights out/sr/RealESRGAN_x4plus.pth ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def build_rrdbnet(num_block: int = 23, num_feat: int = 64, num_grow_ch: int = 32):
    """RRDBNet x4（键名与 RealESRGAN_x4plus.pth 完全对齐）。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1 = ResidualDenseBlock()
            self.rdb2 = ResidualDenseBlock()
            self.rdb3 = ResidualDenseBlock()

        def forward(self, x):
            out = self.rdb3(self.rdb2(self.rdb1(x)))
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB() for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet()


def load_model(weights: str, device: str = "cuda"):
    import torch
    model = build_rrdbnet()
    w = Path(weights)
    if w.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(w), device="cpu")
    else:
        sd = torch.load(w, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "params_ema" in sd:
            sd = sd["params_ema"]          # 官方 release 的包装键
    # Comfy repackaged 前缀兼容（如 "model." / "module." 前缀）
    model_keys = set(model.state_dict().keys())
    if not model_keys.issubset(set(sd.keys())):
        for prefix in ("model.", "module."):
            stripped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if model_keys.issubset(set(stripped.keys())):
                sd = stripped
                break
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    if device == "cuda":
        model.half()
    return model


@torch.no_grad()
def upscale_image(model, img_bgr: np.ndarray) -> np.ndarray:
    import torch
    x = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).to("cuda")
    x = x.permute(2, 0, 1)[None].half() / 255.0
    out = model(x)[0].clamp(0, 1).float().cpu().numpy()
    return cv2.cvtColor((out.transpose(1, 2, 0) * 255).round().astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--weights", default="out/sr/RealESRGAN_x4plus.pth")
    ap.add_argument("--names", nargs="*", help="只处理这些文件名（缺省全部）")
    ap.add_argument("--quality", type=int, default=95, help="JPG 保存质量")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    names = args.names or sorted(p.name for p in src.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    model = load_model(args.weights if Path(args.weights).exists() else str(
        Path(__file__).resolve().parent.parent / args.weights))
    t0 = time.time()
    for i, name in enumerate(names):
        img = cv2.imread(str(src / name))
        if img is None:
            print(f"skip {name}")
            continue
        out = upscale_image(model, img)
        cv2.imwrite(str(dst / name), out, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        print(f"[{i + 1}/{len(names)}] {name}: {img.shape[1]}x{img.shape[0]} -> "
              f"{out.shape[1]}x{out.shape[0]}  ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
