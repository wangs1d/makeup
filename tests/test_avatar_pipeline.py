"""画像管线离线单测：synthetic 3DGS 画像 + manual 锚点，全程不依赖 mediapipe/网络。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from avatar_io import (AvatarData, load_avatar, load_semantics, load_tint,
                       normalize, save_avatar, save_semantics, save_tint)
from avatar_render import AvatarRenderer, load_add_splats
from avatar_semantics import REGION_IDS, SurfaceProbe, annotate, assign_regions
from makeup_compiler import compile_look, compile_tint


# ---------------- synthetic 画像 ----------------

def make_avatar(n: int = 20000, seed: int = 7) -> AvatarData:
    """椭球"脸"：x 半轴 0.35、y 0.5、z 0.25，正面（z>0）高斯暖肤色，已近归一化。"""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = rng.uniform(0.92, 1.0, (n, 1))
    means = dirs * radii * np.array([0.35, 0.5, 0.25])
    front = means[:, 2] > 0
    means = means[front]
    m = len(means)
    colors = np.tile(np.array([0.92, 0.78, 0.68]), (m, 1)).astype(np.float32)
    colors += rng.normal(0, 0.02, (m, 3)).astype(np.float32)
    quats = np.tile(np.array([1.0, 0, 0, 0]), (m, 1)).astype(np.float32)
    scales = np.full((m, 3), 0.008, np.float32) * rng.uniform(0.7, 1.4, (m, 1))
    return AvatarData(means.astype(np.float32), scales, quats,
                      np.clip(colors, 0, 1), np.full(m, 0.95, np.float32),
                      face_height=1.0)


def ellipsoid_z(x: float, y: float) -> float:
    r = 1.0 - (x / 0.35) ** 2 - (y / 0.5) ** 2
    return 0.25 * float(np.sqrt(max(r, 1e-6)))


def manual_anchors() -> dict[str, list]:
    """归一化空间手工锚点（椭球正面）：唇/眼/眉/腮红等中心。"""
    def p(x, y, jitter=0.0):
        z = ellipsoid_z(x, y)
        return [[x, y, z + 0.004], [x * 0.9 + jitter, y + 0.01, ellipsoid_z(x * 0.9, y + 0.01) + 0.004]]

    return {
        "lipstick":  p(0.0, -0.30) + p(0.06, -0.31) + p(-0.06, -0.31),
        "eyeshadow": p(0.16, 0.16) + p(-0.16, 0.16),
        "eyeliner":  p(0.16, 0.13) + p(-0.16, 0.13),
        "eyebrow":   p(0.16, 0.24) + p(-0.16, 0.24),
        "blush":     p(0.24, -0.06) + p(-0.24, -0.06),
        "foundation": p(0.0, 0.0) + p(0.2, 0.3) + p(-0.2, 0.3) + p(0.28, -0.2) + p(-0.28, -0.2),
        "highlight": [[0.0, 0.05, ellipsoid_z(0, 0.05) + 0.004],
                      [0.0, -0.22, ellipsoid_z(0, -0.22) + 0.004]],
        "contour":   [[0.0, 0.42, ellipsoid_z(0, 0.42) + 0.004],
                      [0.3, -0.34, ellipsoid_z(0.3, -0.34) + 0.004],
                      [-0.3, -0.34, ellipsoid_z(-0.3, -0.34) + 0.004]],
    }


@pytest.fixture(scope="module")
def avatar() -> AvatarData:
    return normalize(make_avatar())   # 先归一化（load_avatar 会再做一次幂等归一）


@pytest.fixture(scope="module")
def ids_conf(avatar):
    ids, conf = assign_regions(avatar, {k: np.asarray(v, np.float32)
                                        for k, v in manual_anchors().items()})
    return ids, conf


# ---------------- avatar_io ----------------

def test_ply_roundtrip(tmp_path, avatar):
    p = tmp_path / "av.ply"
    save_avatar(avatar, p)
    av2 = load_avatar(p)
    assert av2.n == avatar.n
    assert np.allclose(av2.means, avatar.means, atol=1e-4)
    assert np.allclose(av2.colors, avatar.colors, atol=0.02)


def test_tint_bin_roundtrip(tmp_path, avatar):
    tint = np.zeros((avatar.n, 4), np.float32)
    tint[:, :3] = 0.6
    tint[:100, 3] = 0.8
    p = tmp_path / "tint.bin"
    save_tint(p, tint)
    back = load_tint(p)
    assert back.shape == tint.shape
    assert np.allclose(back, tint, atol=1e-5)


def test_semantics_bin_roundtrip(tmp_path, ids_conf):
    ids, conf = ids_conf
    p = tmp_path / "semantics.bin"
    save_semantics(p, ids, conf)
    ids2, conf2 = load_semantics(p)
    assert np.array_equal(ids2, ids)
    assert conf2 is not None and np.allclose(conf2, conf, atol=0.01)


# ---------------- avatar_semantics ----------------

def test_assign_regions_hits_lips_and_avoids_backhead(avatar, ids_conf):
    ids, conf = ids_conf
    assert ids.shape[0] == avatar.n
    lips = REGION_IDS["lipstick"]
    lip_mask = ids == lips
    assert lip_mask.sum() > 50, "唇锚点附近应有高斯被标为 lipstick"
    lip_pts = avatar.means[lip_mask]
    assert lip_pts[:, 2].min() > 0, "唇部高斯应在正面"
    assert np.abs(lip_pts[:, 1] + 0.30).max() < 0.08, "唇部高斯应贴近唇锚点 y"
    # 背面（z<0）不应有区域标签
    back = avatar.means[:, 2] < -0.1
    assert (ids[back] == 0).all(), "背面高斯应为 none"


def test_manual_annotate_cache(tmp_path, avatar):
    ids, conf = annotate(avatar, mode="manual", manual_anchors=manual_anchors(),
                         cache_path=tmp_path / "sem.bin")
    ids2, conf2 = annotate(avatar, mode="cache", cache_path=tmp_path / "sem.bin")
    assert np.array_equal(ids, ids2)
    assert conf2.shape == conf.shape


def test_surface_probe_normals(avatar):
    probe = SurfaceProbe(avatar.means)
    p = np.array([0.0, 0.0, 0.25], np.float32)
    _c, n = probe.plane_at(p, radius=0.05)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-3
    assert n[2] > 0.9, "正面中心法线应朝 +Z"


# ---------------- makeup_compiler ----------------

def test_compile_tint_colors_lipstick(presets, avatar, ids_conf):
    ids, _ = ids_conf
    spec = presets["daily-natural"]
    layers = [l for l in spec["layers"] if l["region"] in ("lipstick",)]
    tint = compile_tint(avatar, ids, layers)
    sel = ids == REGION_IDS["lipstick"]
    assert (tint[sel, 3] > 0.3).any(), "唇部应有覆盖权重"
    assert (tint[sel, 0] > tint[sel, 1]).mean() > 0.9, "豆沙唇色应为红>绿"
    other = (ids == REGION_IDS["eyeshadow"])
    assert (tint[other, 3] < 1e-4).all(), "未编译的 region 不应被着色"


def test_compile_look_outputs(tmp_path, presets, avatar, ids_conf):
    ids, _ = ids_conf
    spec = presets["daily-natural"]
    out = compile_look(avatar, ids, spec, tmp_path / "look", avatar_fingerprint="abc")
    assert (out / "tint.bin").exists()
    splats = load_add_splats(out / "add_splats.json")
    assert len(splats) > 0, "唇釉 splat 层应产出附加高斯"
    s = splats[0]
    for key in ("pos", "normal", "axis", "sigma", "color", "alpha"):
        assert key in s
    assert abs(np.linalg.norm(s["normal"]) - 1.0) < 1e-2
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["add_splats"] == len(splats)
    assert "lipstick" in manifest["regions"]


def test_compile_look_only_filters(tmp_path, presets, avatar, ids_conf):
    ids, _ = ids_conf
    spec = presets["daily-natural"]
    out = compile_look(avatar, ids, spec, tmp_path / "look_only", only={"lipstick"})
    tint = load_tint(out / "tint.bin")
    non_lip = (ids != REGION_IDS["lipstick"]) & (ids != 0)
    assert (tint[non_lip, 3] < 1e-6).all(), "--only lipstick 时其他 region 必须为零"


# ---------------- avatar_render ----------------

def test_render_bare_vs_makeup_differ(tmp_path, presets, avatar, ids_conf):
    ids, _ = ids_conf
    spec = presets["daily-natural"]
    tint = compile_tint(avatar, ids, [l for l in spec["layers"]
                                      if l["region"] in ("lipstick", "blush", "eyeshadow")])
    splats = load_add_splats(compile_look(avatar, ids, spec, tmp_path / "look2") / "add_splats.json")
    r = AvatarRenderer(avatar)
    bare = r.render(size=192)
    made = r.render(size=192, tint=tint, add_splats=splats)
    assert bare.shape == (192, 192, 3) and bare.std() > 5, "渲染不应是纯背景"
    assert not np.array_equal(bare, made), "妆容前后画面应有差异"
    diff = np.abs(made.astype(int) - bare.astype(int))
    assert diff.max() > 8, "妆容区域像素应有可见变化"
