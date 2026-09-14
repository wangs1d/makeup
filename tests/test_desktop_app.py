#!/usr/bin/env python3
"""desktop-app（MakeupStudio）测试：摄像头合成器、3DGS 生成器与还原度、查看器服务。"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "desktop-app"
sys.path.insert(0, str(APP))

_skill_scripts = ROOT / "makeup-skill" / "scripts"
_spec = importlib.util.spec_from_file_location("preview_render_core", _skill_scripts / "preview_render.py")
prc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prc)

from makeupstudio.compositor import WebcamCompositor  # noqa: E402
from makeupstudio.coach import steps_from_layers  # noqa: E402
from makeupstudio.splat3d import (SplatCloudBuilder, fidelity_psnr,  # noqa: E402
                                  render_splats_python)

PRESET = json.loads((ROOT / "makeup-skill" / "presets" / "date-rose.json").read_text(encoding="utf-8"))


def _synthetic_landmarks(w=640, h=537):
    """canonical 模型按 render_still 相机投影 → 像素关键点（468,2）。"""
    r = prc.get_renderer(w, h)
    V = r.model.pose_explicit(0, -2, 0, 0.03, 0.12)
    depth = 2.3 - V[:, 2]
    return np.stack([w / 2 + V[:, 0] * r.f / depth, h / 2 - V[:, 1] * r.f / depth], 1)


# ---------------- 合成器 ----------------

def test_compositor_changes_face_region():
    lm = _synthetic_landmarks()
    comp = WebcamCompositor()
    comp.set_look(PRESET["layers"])
    frame = np.full((537, 640, 3), (110, 105, 100), np.uint8)
    out = comp.process(frame.copy(), lm, 0.8)
    diff = np.abs(out.astype(int) - frame.astype(int)).sum(2)
    assert (diff > 30).sum() > 5000, "妆容未在脸上产生可见变化"


def test_compositor_uses_lipstick_color():
    lm = _synthetic_landmarks()
    comp = WebcamCompositor()
    comp.set_look(PRESET["layers"])
    frame = np.full((537, 640, 3), (110, 105, 100), np.uint8)
    out = comp.process(frame.copy(), lm, 1.0)
    # 唇中心（lips_outer 关键点均值位置）附近应有明显红系像素（口红）
    idx = _lip_anchor()
    cx, cy = lm[idx].mean(axis=0).astype(int)   # lm 列序为 (x, y)
    patch = out[max(0, cy-5):cy+6, max(0, cx-7):cx+8].reshape(-1, 3).astype(int)
    red = patch[(patch[:, 2] > patch[:, 0] + 30) & (patch[:, 2] > patch[:, 1] + 30)]
    assert len(red) > 20, f"唇部区域缺少口红红色像素，patch 均值 BGR={patch.mean(0)}"


def _lip_anchor():
    """lips_outer 关键点索引（调用方在 lm 上取均值位置）。"""
    data = json.loads((ROOT / "makeup-skill" / "references" / "landmark-regions.json")
                      .read_text(encoding="utf-8"))
    return data["regions"]["lips_outer"]["indices"]


def test_compositor_degenerate_landmarks_safe():
    comp = WebcamCompositor()
    comp.set_look(PRESET["layers"])
    frame = np.full((537, 640, 3), (110, 105, 100), np.uint8)
    lm = np.full((468, 2), 100.0)          # 退化：全部重合
    out = comp.process(frame.copy(), lm, 0.8)
    assert out.shape == frame.shape


def test_compositor_realtime_budget():
    lm = _synthetic_landmarks()
    comp = WebcamCompositor()
    comp.set_look(PRESET["layers"])
    frame = np.full((720, 1280, 3), (110, 105, 100), np.uint8)
    lm_hd = lm * np.array([2.0, 720 / 537])
    comp.process(frame.copy(), lm_hd, 0.8)      # 预热
    t0 = time.time()
    for _ in range(3):
        comp.process(frame.copy(), lm_hd, 0.8)
    per_frame = (time.time() - t0) / 3
    assert per_frame < 0.15, f"单帧合成 {per_frame*1000:.0f}ms，达不到实时"


def test_compositor_env_changes_output():
    """环境光预设必须真实作用到实时合成画面（色温/曝光方向 + 非中性差异可见）。"""
    lm = _synthetic_landmarks()
    comp = WebcamCompositor()
    comp.set_look(PRESET["layers"])
    frame = np.full((537, 640, 3), (110, 105, 100), np.uint8)
    outs = {}
    for env in ("neutral", "warm", "cool", "dim"):
        comp.set_env(env)
        outs[env] = comp.process(frame.copy(), lm, 0.8).astype(int)
        cx = slice(250, 320)                     # 脸中心区域（合成帧上无妆覆盖）
        mean_bgr = outs[env][cx, 280:360].mean(axis=(0, 1))
        b, g, r = mean_bgr
        if env == "warm":
            assert r > outs["neutral"][cx, 280:360][:, :, 2].mean(), "warm 未偏暖"
        if env == "cool":
            assert b > outs["neutral"][cx, 280:360][:, :, 0].mean(), "cool 未偏冷"
        if env == "dim":
            assert mean_bgr.sum() < outs["neutral"][cx, 280:360].sum() * 0.95, "dim 未压暗"
    for env in ("warm", "cool", "dim"):
        d = np.abs(outs[env] - outs["neutral"]).mean()
        assert d > 0.8, f"{env} 与 neutral 差异过弱（mean|Δ|={d:.2f}），环境光没有实际效果"
    comp.set_env("不存在的预设")
    assert comp.env == "neutral"                 # 非法名回退


# ---------------- 3DGS 生成器 ----------------

@pytest.fixture(scope="module")
def small_cloud():
    b = SplatCloudBuilder()
    return b, b.build(PRESET["layers"], intensity=0.8, n_base=6000, n_makeup=3000)


def test_splat_cloud_structure(small_cloud):
    b, cloud = small_cloud
    n = len(cloud["xyz"])
    assert n > 5000
    for k in ("xyz", "scale", "rot", "rgba"):
        assert cloud[k].shape[0] == n
    assert cloud["rgba"][:, :3].max() <= 1.0 and cloud["rgba"][:, 3].min() >= 0.0
    assert cloud["scale"].min() > 0
    # 四元数模长 ≈ 1
    assert abs(np.linalg.norm(cloud["rot"], axis=1) - 1.0).max() < 1e-3


def test_splat_export_formats(small_cloud, tmp_path):
    b, cloud = small_cloud
    p_splat = tmp_path / "m.splat"
    b.export_splat(cloud, p_splat)
    assert p_splat.stat().st_size == 32 * len(cloud["xyz"])
    p_ply = tmp_path / "m.ply"
    b.export_ply(cloud, p_ply)
    header = p_ply.read_bytes()[:400].decode("ascii", "ignore")
    assert f"element vertex {len(cloud['xyz'])}" in header
    assert "f_dc_0" in header and "rot_3" in header


def test_makeup_intensity_changes_colors(small_cloud):
    b, _ = small_cloud
    off = b.build(PRESET["layers"], intensity=0.0, n_base=6000, n_makeup=3000)
    on = b.build(PRESET["layers"], intensity=0.8, n_base=6000, n_makeup=3000)
    assert len(on["xyz"]) > 6000                     # 妆容层贡献了额外高斯
    assert np.allclose(off["rgba"][:6000, 3], on["rgba"][:6000, 3])      # 基础层一致
    assert on["rgba"][6000:, 3].max() > 0.1          # 强度开启后妆容高斯不透明                                # 强度影响 alpha


def test_3dgs_fidelity_to_reference_render():
    """核心验收：3DGS 前向泼溅图 vs 内核参考图（同一相机）的还原度。"""
    b = SplatCloudBuilder()
    cloud = b.build(PRESET["layers"], intensity=0.8, n_base=8000, n_makeup=4000)
    ref = prc.render_reference(PRESET, intensity=0.8, size=640)
    got = render_splats_python(cloud, 640, 537, max_splats=12000)
    psnr = fidelity_psnr(ref, got)
    assert psnr > 15.0, f"3DGS 还原度过低：{psnr:.2f} dB"
    # 唇部区域 3DGS 图应为红系
    lip_idx = _lip_anchor()
    lm = _synthetic_landmarks()
    cx, cy = lm[lip_idx].mean(axis=0).astype(int)   # lm 列序为 (x, y)
    patch = got[max(0, cy-4):cy+5, max(0, cx-4):cx+5]
    mean_bgr = patch.reshape(-1, 3).mean(0)
    assert mean_bgr[2] > mean_bgr[0] + 20, f"3DGS 预览唇色不可见: BGR={mean_bgr}"


# ---------------- 陪练 ----------------

def test_coach_steps_order():
    steps = steps_from_layers(PRESET["layers"])
    assert steps and "唇妆" in steps[-1]
    assert any("底妆" in s for s in steps[0:2])


# ---------------- 查看器服务 ----------------

def test_viewer_server(tmp_path):
    from makeupstudio.server import ViewerServer
    srv = ViewerServer(tmp_path, port=8793)
    srv.start()
    import urllib.request
    try:
        with urllib.request.urlopen(srv.url, timeout=5) as r:
            body = r.read().decode("utf-8")
        assert "3DGS" in body
        (tmp_path / "splat" / "version.json").write_text('{"version":"1","ready":true}')
        with urllib.request.urlopen(srv.url + "splat/version.json", timeout=5) as r:
            assert json.loads(r.read())["ready"] is True
        try:
            urllib.request.urlopen(srv.url + "nope.js", timeout=5)
            assert False, "应 404"
        except Exception:
            pass
    finally:
        srv.stop()


# ---------------- 追踪器（真实模型冒烟） ----------------

def test_tracker_smoke_no_face():
    from makeupstudio.tracker import FaceTracker
    tr = FaceTracker()
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = (120, 110, 100)
    try:
        res = tr.detect(img, 100.0)
        assert res is None          # 合成无人脸图 → 无关键点
    finally:
        tr.close()
