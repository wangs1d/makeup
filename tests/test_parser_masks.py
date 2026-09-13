#!/usr/bin/env python3
"""parser/compositor 集成测试：解析蒙版分组、帧间对齐、合成器像素级蒙版路径。

不依赖 torch——模型可用性走 FakeParser；真实模型冒烟测试仅在权重已缓存时执行。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "desktop-app"
sys.path.insert(0, str(APP))

from makeupstudio.compositor import WebcamCompositor  # noqa: E402
from makeupstudio.parser import (FaceParser, _group_labels,  # noqa: E402
                                 estimate_face_affine, warp_masks)

_skill_scripts = ROOT / "makeup-skill" / "scripts"
_spec = importlib.util.spec_from_file_location("preview_render_core",
                                               _skill_scripts / "preview_render.py")
prc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prc)

PRESET = json.loads((ROOT / "makeup-skill" / "presets" / "date-rose.json")
                    .read_text(encoding="utf-8"))
_REGIONS = json.loads((ROOT / "makeup-skill" / "references" / "landmark-regions.json")
                      .read_text(encoding="utf-8"))["regions"]


def _synthetic_landmarks(w=640, h=537, dx=0.0, dy=0.0):
    r = prc.get_renderer(w, h)
    V = r.model.pose_explicit(0, -2, 0, 0.03, 0.12)
    depth = 2.3 - V[:, 2]
    return np.stack([w / 2 + V[:, 0] * r.f / depth + dx,
                     h / 2 - V[:, 1] * r.f / depth + dy], 1)


# ---------------- 分组 ----------------

def test_group_labels_by_name():
    id2label = {0: "Background", 1: "skin", 2: "nose", 4: "l_eye", 5: "r_eye",
                6: "l_brow", 11: "u_lip", 12: "l_lip", 13: "hair", 10: "mouth"}
    g = _group_labels(id2label)
    assert set(g["lips"]) == {11, 12}
    assert set(g["skin"]) == {1, 2}          # 鼻梁算皮肤
    assert set(g["eye_l"]) == {4}
    assert 13 in g["hair"]
    assert 0 not in g.get("skin", [])        # 背景不属于任何组


# ---------------- 帧间对齐 ----------------

def test_warp_masks_follows_landmark_shift():
    h, w = 200, 260
    src_lm = _synthetic_landmarks(260, 200)
    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, (130, 100), 30, 1.0, -1)
    dx, dy = 24.0, -12.0
    dst_lm = src_lm + (dx, dy)
    affine = estimate_face_affine(src_lm, dst_lm)
    warped = warp_masks({"lips": mask}, affine, (h, w))["lips"]
    ys, xs = np.nonzero(warped > 0.5)
    cy0, cx0 = [v.mean() for v in np.nonzero(mask > 0.5)]
    assert abs(xs.mean() - (cx0 + dx)) < 3
    assert abs(ys.mean() - (cy0 + dy)) < 3


def test_warp_identity_when_no_motion():
    src_lm = _synthetic_landmarks()
    affine = estimate_face_affine(src_lm, src_lm)
    assert np.allclose(affine, np.eye(2, 3), atol=1e-3)


# ---------------- 合成器像素级蒙版路径 ----------------

def _fake_parse(lm, w=640, h=537):
    """用 landmark 几何构造假解析蒙版：唇=lips_outer 多边形，皮肤=脸椭圆挖洞。"""
    lips = np.zeros((h, w), np.float32)
    lp = lm[_REGIONS["lips_outer"]["indices"]]
    cv2.fillPoly(lips, [np.round(lp).astype(np.int32)], 1.0)
    skin = np.zeros((h, w), np.float32)
    ov = lm[_REGIONS["foundation_face_oval"]["indices"]]
    cv2.fillPoly(skin, [np.round(ov).astype(np.int32)], 1.0)
    # 像素级分割的卖点：皮肤蒙版天然没有眉/眼/唇
    for g in ("eyebrow_left", "eyebrow_right"):
        cv2.polylines(skin, [np.round(lm[_REGIONS[g]["indices"]]).astype(np.int32)],
                      False, 0.0, 9)
    cv2.fillPoly(skin, [np.round(lp).astype(np.int32)], 0.0)
    return {"lips": lips, "skin": skin}


def test_compositor_uses_parse_lips_mask():
    lm = _synthetic_landmarks()
    comp = WebcamCompositor()
    comp.set_look(PRESET["layers"])
    comp.set_parse(_fake_parse(lm), lm)
    frame = np.full((537, 640, 3), (110, 105, 100), np.uint8)
    out = comp.process(frame.copy(), lm, 1.0)
    lp = lm[_REGIONS["lips_outer"]["indices"]]
    cx, cy = int(lp[:, 0].mean()), int(lp[:, 1].mean())
    patch = out[cy - 3:cy + 4, cx - 5:cx + 6].reshape(-1, 3).astype(int)
    red = patch[(patch[:, 2] > patch[:, 0] + 30) & (patch[:, 2] > patch[:, 1] + 30)]
    assert len(red) > 20, "解析唇蒙版路径未上口红"


def test_compositor_parse_skin_keeps_brows():
    """解析皮肤蒙版盖不到眉毛 → 只有粉底时眉毛区域不被平涂。"""
    lm = _synthetic_landmarks()
    comp = WebcamCompositor()
    comp.set_look([l for l in PRESET["layers"] if l["region"] == "foundation"])
    comp.set_parse(_fake_parse(lm), lm)
    frame = np.full((537, 640, 3), (110, 105, 100), np.uint8)
    out = comp.process(frame.copy(), lm, 1.0)
    brow = lm[_REGIONS["eyebrow_left"]["indices"]]
    cx, cy = int(brow[:, 0].mean()), int(brow[:, 1].mean())
    diff = np.abs(out[cy - 2:cy + 3, cx - 4:cx + 5].astype(int)
                  - frame[cy - 2:cy + 3, cx - 4:cx + 5].astype(int)).sum(2)
    assert (diff > 40).mean() < 0.3, "解析皮肤蒙版下眉毛仍被粉底平涂"


def test_compositor_without_parse_still_works():
    lm = _synthetic_landmarks()
    comp = WebcamCompositor()
    comp.set_look(PRESET["layers"])
    frame = np.full((537, 640, 3), (110, 105, 100), np.uint8)
    out = comp.process(frame.copy(), lm, 0.8)
    diff = np.abs(out.astype(int) - frame.astype(int)).sum(2)
    assert (diff > 30).sum() > 5000, "无解析时 landmark 回退路径失效"


def test_compositor_parse_stale_landmarks_aligned():
    """解析结果比当前帧旧（脸移了 20px）→ 仿射对齐后唇色仍在唇上。"""
    lm = _synthetic_landmarks()
    stale_lm = lm - (20.0, 8.0)                      # 解析时的脸在左上
    comp = WebcamCompositor()
    comp.set_look(PRESET["layers"])
    comp.set_parse(_fake_parse(stale_lm), stale_lm)
    frame = np.full((537, 640, 3), (110, 105, 100), np.uint8)
    out = comp.process(frame.copy(), lm, 1.0)
    lp = lm[_REGIONS["lips_outer"]["indices"]]
    cx, cy = int(lp[:, 0].mean()), int(lp[:, 1].mean())
    patch = out[cy - 3:cy + 4, cx - 5:cx + 6].reshape(-1, 3).astype(int)
    red = patch[(patch[:, 2] > patch[:, 0] + 30) & (patch[:, 2] > patch[:, 1] + 30)]
    assert len(red) > 20, "蒙版未随 landmark 对齐到当前帧"


def test_compositor_lip_gloss_survives_mask_bbox_mismatch():
    """回归：解析唇蒙版缺失左侧嘴角（侧脸/遮挡常见）时，解析 bbox 左缘
    越过下唇 landmark bbox 左缘（mx0 < bx0）→ gloss 高光条切片起点为负，
    numpy 负索引回绕使 sub_win 坍缩为空，旧代码在 band * (sub_win > 0.05)
    处广播失败、相机线程崩溃。"""
    lm = _synthetic_landmarks()
    lips_only = [dict(l, finish="gloss") for l in PRESET["layers"]
                 if l["region"] == "lipstick"]
    assert lips_only, "预设缺少 lipstick 层"
    comp = WebcamCompositor()
    comp.set_look(lips_only)
    masks = _fake_parse(lm)
    cx = int(lm[_REGIONS["lips_outer"]["indices"]][:, 0].mean())
    masks["lips"][:, :cx + 10] = 0.0             # 左侧嘴角缺失
    comp.set_parse(masks, lm)
    frame = np.full((537, 640, 3), (110, 105, 100), np.uint8)
    out = comp.process(frame.copy(), lm, 1.0)        # 旧代码此处 ValueError
    assert out.shape == frame.shape, "输出形状异常"


# ---------------- 真实模型冒烟（仅在权重已缓存时执行） ----------------

def test_face_parser_real_model_smoke():
    pytest.importorskip("transformers")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("jonathandinu/face-parsing", local_files_only=True)
    except Exception:
        pytest.skip("face-parsing 权重未缓存（首次下载较慢，不入常规测试）")
    p = FaceParser(device="cpu")
    assert p.available(), f"模型加载失败：{p.load_error}"
    img = np.full((256, 256, 3), 120, np.uint8)
    masks = p.parse(img)
    assert masks is None or isinstance(masks, dict)
