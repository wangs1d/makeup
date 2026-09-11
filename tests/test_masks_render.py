"""蒙版烘焙 / 渲染内核 / 溅射规则 的单元测试（preview_render，与 Unity RegionMaskBaker 同规则）。"""
from __future__ import annotations

import json

import numpy as np
import pytest

import preview_render as pr

REFS = pr.REFS


@pytest.fixture(scope="module")
def masks():
    return pr.RegionMasks(REFS / "landmark-regions.json", REFS / "canonical_face_model.obj")


@pytest.fixture(scope="module")
def regions():
    return json.loads((REFS / "landmark-regions.json").read_text(encoding="utf-8"))["regions"]


def _layer(region, side="both", **shape):
    return {"id": f"{region}-{side}", "region": region, "side": side, "enabled": True,
            "opacity": 0.8, "finish": "satin", "texture_strength": 0.2,
            "color_stops": [{"at": 0, "hex": "#C88D7A"}, {"at": 1, "hex": "#B0705F"}],
            "shape": shape}


def _halves(cov):
    """按 UV 左右半区统计覆盖量（canonical UV 左右对称）。"""
    mid = cov.shape[1] // 2
    return cov[:, :mid].sum(), cov[:, mid:].sum()


def test_eyebrow_both_sides_present_and_symmetric(masks):
    """P0 回归：side=both 必须画两条眉（旧版 Unity 只画左眉）。"""
    cov = masks.bake(_layer("eyebrow"))[..., 0]
    left, right = _halves(cov)
    assert left > 50 and right > 50
    assert abs(left - right) / max(left, right) < 0.25


def test_eyebrow_single_side_only(masks):
    cov_l = masks.bake(_layer("eyebrow", side="left"))[..., 0]
    cov_r = masks.bake(_layer("eyebrow", side="right"))[..., 0]
    l1, r1 = _halves(cov_l)
    l2, r2 = _halves(cov_r)
    # 各自只出现在一侧（左右哪一半取决于 UV 布局，二者必须互补）
    assert (l1 > 20) != (r1 > 20)
    assert (l2 > 20) != (r2 > 20)
    assert (l1 > 20) != (l2 > 20)


def test_eyeliner_wing_symmetric(masks):
    cov = masks.bake(_layer("eyeliner", thickness=0.4, wing=0.9))[..., 0]
    left, right = _halves(cov)
    assert abs(left - right) / max(left, right) < 0.2


def test_concealer_shape_flags(masks):
    on = masks.bake(_layer("concealer"))[..., 0].sum()
    off = masks.bake(_layer("concealer", under_eye=False))[..., 0].sum()
    big = masks.bake(_layer("concealer", size=1.6))[..., 0].sum()
    assert on > 0 and off == 0 and big > on * 1.5


def test_lipstick_overline_and_blur(masks):
    base = masks.bake(_layer("lipstick"))[..., 0]
    over = masks.bake(_layer("lipstick", overline=1.0))[..., 0]
    assert over.sum() > base.sum() * 1.05
    soft = masks.bake(_layer("lipstick", blur=0.8))[..., 0]
    # blur 越大，边缘半透明像素占比越高
    def partial_ratio(c):
        return ((c > 0.05) & (c < 0.95)).sum() / max((c > 0.05).sum(), 1)
    assert partial_ratio(soft) > partial_ratio(base)


def test_blush_soft_feather_gradient(masks):
    """P2：腮红按到边缘距离软衰减——半透明像素应占相当比例（旧版几乎是硬边）。"""
    cov = masks.bake(_layer("blush"))[..., 0]
    inside = cov > 0.05
    partial = ((cov > 0.05) & (cov < 0.9)).sum() / max(inside.sum(), 1)
    assert partial > 0.35
    # 中心仍然饱满
    assert cov.max() > 0.95


def test_thin_lines_not_eroded(masks):
    """细线条（眉/眼线）不做距离衰减，否则会被侵蚀成断线。"""
    cov = masks.bake(_layer("eyeliner", thickness=0.25))[..., 0]
    assert cov.max() > 0.25                     # 1px 线经 AA/模糊后仍有可见峰值
    assert (cov > 0.1).sum() > 100              # 线没有消失


def test_contour_flags(masks):
    full = masks.bake(_layer("contour"))[..., 0].sum()
    no_jaw = masks.bake(_layer("contour", jaw=False))[..., 0].sum()
    no_nose = masks.bake(_layer("contour", nose=False))[..., 0].sum()
    assert no_jaw < full and no_nose < full


def test_unknown_region_raises(masks):
    with pytest.raises(ValueError):
        masks.bake(_layer("unicorn"))


def test_centrality_in_range(masks):
    m = masks.bake(_layer("blush"))
    cent = m[..., 1]
    assert cent.min() >= 0 and cent.max() <= 1
    assert cent.max() > 0.9      # 中心接近 1


# ---------------- 溅射规则 ----------------

def test_build_splats_lips_inset_and_axis(regions):
    layer = dict(_layer("lipstick"), render={"type": "splat", "splat": {"thickness": 0.0015, "density": 0.8}})
    out = pr.build_splats([layer], regions)
    assert len(out) == 1
    anchors = out[0]["anchors"]
    outer = [a for a in anchors if a["group"] == "lips_outer"]
    inner = [a for a in anchors if a["group"] == "lips_inner"]
    assert len(outer) == 20 and len(inner) == 20
    assert all("toward" in a and 0 < a["inset"] < 1 for a in outer)
    assert all("toward" not in a for a in inner)
    # 外唇锚点尺寸/强度略减
    assert outer[0]["sigma"][0] < inner[0]["sigma"][0]
    assert outer[0]["alpha"] < inner[0]["alpha"]
    # 组内 t 单调，供 App 端确定长轴方向
    ts = [a["t"] for a in outer]
    assert ts == sorted(ts) and ts[0] == 0 and ts[-1] == 1


def test_build_splats_new_regions_supported(regions):
    for region in ("eyeshadow", "blush", "eyebrow", "lashes", "highlight", "contour"):
        layer = dict(_layer(region), render={"type": "splat"})
        out = pr.build_splats([layer], regions)
        assert out and out[0]["anchors"], region


def test_build_splats_unsupported_region_skipped(regions):
    layer = dict(_layer("foundation"), render={"type": "splat"})
    assert pr.build_splats([layer], regions) == []


def test_build_splats_mesh_layers_ignored(regions):
    assert pr.build_splats([_layer("lipstick")], regions) == []


# ---------------- 渲染 ----------------

@pytest.fixture(scope="module")
def renderer():
    return pr.FaceRenderer(256, 216)


def test_render_still_makeup_changes_pixels(renderer, presets):
    spec = presets["date-rose"]
    layers = [l for l in spec["layers"] if l.get("enabled", True)]
    bare = renderer.render_still([], intensity=0.0)
    look = renderer.render_still(layers, intensity=1.0)
    assert bare.shape == look.shape == (216, 256, 3)
    diff = np.abs(bare.astype(int) - look.astype(int)).mean()
    assert diff > 2.0, f"上妆前后差异过小 {diff}"


def test_render_intensity_monotonic(renderer, presets):
    layers = [l for l in presets["date-rose"]["layers"] if l.get("enabled", True)]
    bare = renderer.render_still([], intensity=0.0).astype(int)
    d_half = np.abs(renderer.render_still(layers, intensity=0.5).astype(int) - bare).mean()
    d_full = np.abs(renderer.render_still(layers, intensity=1.0).astype(int) - bare).mean()
    assert d_full > d_half > 0


def test_env_tint_changes_color_balance(renderer, presets):
    layers = [l for l in presets["daily-natural"]["layers"] if l.get("enabled", True)]
    renderer.set_env("warm")
    warm = renderer.render_still(layers).astype(float)
    renderer.set_env("cool")
    cool = renderer.render_still(layers).astype(float)
    renderer.set_env("neutral")
    face = warm.sum(2) > 60      # 排除深色背景
    # BGR：warm 的 R/B 比值应高于 cool
    ratio_w = warm[..., 2][face].mean() / max(warm[..., 0][face].mean(), 1)
    ratio_c = cool[..., 2][face].mean() / max(cool[..., 0][face].mean(), 1)
    assert ratio_w > ratio_c * 1.05


def test_render_reference_writes_file(tmp_path, presets):
    out = tmp_path / "ref.jpg"
    img = pr.render_reference(presets["office-polish"], out, size=256, env="neutral")
    assert out.exists() and out.stat().st_size > 2000
    assert img.shape[1] == 256


def test_render_yaw_changes_silhouette(renderer, presets):
    layers = [l for l in presets["daily-natural"]["layers"] if l.get("enabled", True)]
    a = renderer.render_still(layers, yaw_deg=0).astype(int)
    b = renderer.render_still(layers, yaw_deg=30).astype(int)
    assert np.abs(a - b).mean() > 3


def test_face_edge_feather_map(renderer):
    edge = renderer.edge
    assert edge.shape == (pr.TEX, pr.TEX)
    assert edge.max() == 1.0 and edge.min() == 0.0
    # 有一圈过渡带
    assert ((edge > 0.05) & (edge < 0.95)).sum() > 2000
