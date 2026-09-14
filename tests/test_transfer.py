#!/usr/bin/env python3
"""妆效迁移（EleGANt 门控）测试：对比图合成、中文路径读写、资产门控与端到端。"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "desktop-app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from makeupstudio.transfer import (compare_image, imwrite_unicode,  # noqa: E402
                                   imread_unicode, missing_assets)

TOOL = APP / "tools" / "makeup_transfer.py"


def _solid(bgr: tuple[int, int, int], w=320, h=240) -> np.ndarray:
    return np.full((h, w, 3), bgr, np.uint8)


# ---------------- 三联对比图 ----------------

def test_compare_image_layout():
    src, ref, out = _solid((60, 60, 60)), _solid((60, 180, 60)), _solid((180, 60, 60))
    canvas = compare_image(src, ref, out, height=120)
    assert canvas.shape[0] == 120 + 44                # 单栏高 + 标签条
    # 每栏中心色与输入一致（等高缩放不改变纯色）：320x240→160 宽，栏心 x=80/244/408
    h, w = canvas.shape[:2]
    assert tuple(canvas[44 + h // 2, 80]) == (60, 60, 60)
    assert tuple(canvas[44 + h // 2, 244]) == (60, 180, 60)
    assert tuple(canvas[44 + h // 2, 408]) == (180, 60, 60)
    # 标签条有文字（存在深色像素，中文/ASCII 回退均成立）
    assert (canvas[:44].mean(axis=2) < 128).sum() > 20


def test_compare_image_mixed_sizes_same_height():
    src = _solid((10, 10, 10), w=640, h=480)
    ref = _solid((10, 10, 10), w=300, h=300)
    out = _solid((10, 10, 10), w=200, h=800)
    canvas = compare_image(src, ref, out, height=100)
    assert canvas.shape[0] == 144
    widths = [round(640 * 100 / 480), round(300 * 100 / 300), round(200 * 100 / 800)]
    assert canvas.shape[1] == sum(widths) + 8          # 两条 4px 栏间分隔


# ---------------- 中文/非 ASCII 路径读写 ----------------

def test_unicode_path_roundtrip(tmp_path: Path):
    p = tmp_path / "中文目录" / "素颜照.png"
    img = _solid((1, 2, 3), w=8, h=6)
    imwrite_unicode(p, img)
    back = imread_unicode(p)
    assert np.array_equal(back, img)


# ---------------- 资产门控 ----------------

def test_missing_assets_is_list_of_strings():
    missing = missing_assets()
    assert isinstance(missing, list)
    assert all(isinstance(m, str) and m for m in missing)


def test_cli_help_runs():
    r = subprocess.run([sys.executable, str(TOOL), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    assert "--source" in r.stdout


# ---------------- 端到端（权重就绪时才跑） ----------------

@pytest.mark.skipif(missing_assets(), reason="EleGANt 权重未部署")
def test_end_to_end_synthetic_transfer(tmp_path: Path):
    """合成素颜/参考图（与 research/test_transfer.py 同源）→ CLI 全链路出对比图。"""
    spec = importlib.util.spec_from_file_location(
        "prc", ROOT / "makeup-skill" / "scripts" / "preview_render.py")
    prc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prc)
    preset = (ROOT / "makeup-skill" / "presets" / "date-rose.json").read_text(encoding="utf-8")
    import json
    layers = json.loads(preset)
    bare = prc.render_reference({"layers": []}, intensity=0.0, size=640)
    ref = prc.render_reference(layers, intensity=1.0, size=640)
    src_p, ref_p = tmp_path / "素颜.png", tmp_path / "参考.png"
    imwrite_unicode(src_p, bare)
    imwrite_unicode(ref_p, ref)

    r = subprocess.run(
        [sys.executable, str(TOOL), "--source", str(src_p), "--reference", str(ref_p),
         "--out", str(tmp_path / "出图"), "--stem", "demo"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    cmp_img = imread_unicode(tmp_path / "出图" / "transfer_demo_compare.png")
    assert cmp_img.shape[0] == 560 + 44
    # 迁移栏与素颜栏应有可见差异（妆已上脸）：三栏各 560 宽，中间隔 4px
    bare_panel = cmp_img[44:, 0:560].astype(int)
    made_panel = cmp_img[44:, 1128:1688].astype(int)
    diff = np.abs(made_panel - bare_panel).mean()
    assert diff > 2.0, f"迁移结果与素颜无可见差异（mean diff={diff:.2f}）"
