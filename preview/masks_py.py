#!/usr/bin/env python3
"""masks_py — 兼容薄封装：RegionMasks 已迁入 makeup-skill/scripts/preview_render.py。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "makeup-skill" / "scripts"))
from preview_render import TEX as MASK, RegionMasks, load_uvs, sample_ramp  # noqa: E402,F401
