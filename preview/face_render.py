#!/usr/bin/env python3
"""face_render — 兼容薄封装：渲染内核已迁入 makeup-skill/scripts/preview_render.py（skill 自包含）。

演示视频渲染器（render_demo_video.py）仍从这里导入 FaceRenderer / build_splats。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "makeup-skill" / "scripts"))
from preview_render import (  # noqa: E402,F401
    ENVS, FACE_HEIGHT_M, TEX, FaceModel, FaceRenderer, bake_makeup, bake_skin, build_splats,
    open_video_writer, render_reference,
)
