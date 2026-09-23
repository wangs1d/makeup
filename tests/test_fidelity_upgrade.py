"""还原度升级回归：float Lab / pigment-safe 迁移 / near 软门控 / 3D 眼线眉锚定 /
珠光闪点 / 妆感材质合成 / light+material sidecar 往返 / ΔE00 标定与度量。
全部 CPU 合成数据；CUDA 路径（AOV 光栅化/guidance）不在离线测试范围。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "desktop-app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from makeupstudio.face3dgs.appearance.calibrate import (  # noqa: E402
    LIPS_OUTER,
    calibrate_spec,
    ciede2000,
    extract_makeup_colors,
    image_region_masks,
    region_delta_e,
    spec_targets,
)
from makeupstudio.face3dgs.appearance.makeup_uv import (  # noqa: E402
    FINISH_TARGET,
    UvMakeupBaker,
    hex_to_rgb01,
    lab2rgb,
    merge_makeup_layer,
    rgb2lab,
)
from makeupstudio.face3dgs.appearance.offline_render import (  # noqa: E402
    build_shade,
    composite_shade,
    read_light_bin,
    read_material_bin,
)
from makeupstudio.face3dgs.appearance.pipeline import (  # noqa: E402
    apply_makeup_to_asset,
    export_material,
)
from makeupstudio.face3dgs.appearance.train_base import write_light_bin  # noqa: E402
from makeupstudio.face3dgs.appearance.uvbind import bind_uv  # noqa: E402


@pytest.fixture(scope="module")
def canonical_cloud():
    from makeupstudio.face3dgs.fit_makeup import FaceMakeupFitter, _sample_tris
    f = FaceMakeupFitter(tracker_factory=lambda: (_ for _ in ()).throw(RuntimeError))
    rng = np.random.default_rng(11)
    P, _b, _rows = _sample_tris(f.model.base, f.model.tris, 12000, rng)
    cloud = {
        "xyz": P.astype(np.float32),
        "scale": np.full((len(P), 3), 0.002, np.float32),
        "rot": np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (len(P), 1)),
        "rgba": np.concatenate([np.full((len(P), 3), 0.6), np.ones((len(P), 1))], 1),
        "sh_rest": np.zeros((len(P), 3, 3), np.float32),
    }
    return cloud, f.model.base[:468]


def _lip_layers() -> list[dict]:
    return [{"id": "lip", "region": "lipstick", "enabled": True, "opacity": 0.9,
             "finish": "gloss",
             "color_stops": [{"at": 0.0, "hex": "#C21858"},
                             {"at": 1.0, "hex": "#E91E63"}]}]


# ---------------- 颜色空间 / CIEDE2000 ----------------

def test_rgb_lab_roundtrip_and_srgb_red():
    lab = rgb2lab(np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    assert lab[0, 1] > 70                      # sRGB 红 a* 显著为正
    assert abs(lab[1, 0] - 100) < 1e-3         # 白 L=100
    assert abs(lab[2, 0]) < 1e-6               # 黑 L=0
    back = lab2rgb(lab)
    assert np.abs(back - np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]])).max() < 1e-4


def test_ciede2000_sharma_reference_pairs():
    """Sharma et al. 2005 测试向量（实现正确性的外部基准）。"""
    l1 = np.array([[50, 2.6772, -79.7751],
                   [50, -1.3802, -84.2814],
                   [50, 2.5, 0.0]])
    l2 = np.array([[50, 0.0, -82.7485],
                   [50, 0.0, -82.7485],
                   [50, 3.2592, 0.335]])
    d = ciede2000(l1, l2)
    assert d[0] == pytest.approx(2.0425, abs=2e-3)
    assert d[1] == pytest.approx(1.0000, abs=2e-3)
    assert d[2] == pytest.approx(1.0000, abs=2e-3)
    assert ciede2000(l1, l1).sum() == pytest.approx(0.0, abs=1e-9)   # 对称/自反


# ---------------- pigment-safe 跨肤色迁移 ----------------

def _apply_with_skin(cloud, landmarks, base_rgb, tex=256):
    cloud = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in cloud.items()}
    cloud["rgba"][:, :3] = base_rgb
    binding = bind_uv(cloud, landmarks, tex=tex)
    baker = UvMakeupBaker(tex=tex)
    maps = baker.bake(_lip_layers(), intensity=1.0)
    made = baker.apply_to_cloud(cloud, maps, binding.uv, binding.valid)
    return made["makeup_w"], made["rgba"][:, :3]


def test_pigment_safe_keeps_chroma_on_deep_skin(canonical_cloud):
    """深肤 + 亮唇 hex：色度保底不褪色（还原度不随肤色塌成灰调）。"""
    cloud, landmarks = canonical_cloud
    deep = np.tile(np.array([0.22, 0.13, 0.11], np.float32), (len(cloud["xyz"]), 1))
    w, out = _apply_with_skin(cloud, landmarks, deep)
    zone = w > 0.5
    assert zone.any()
    base_lab = rgb2lab(deep[zone])
    out_lab = rgb2lab(np.clip(out[zone], 0, 1))
    assert (np.hypot(out_lab[:, 1], out_lab[:, 2])
            >= np.hypot(base_lab[:, 1], base_lab[:, 2]) - 1e-3).mean() > 0.95


def test_lab_adapt_formula():
    """明度跟随系数：差小→1（不削弱迁移），差大→收缩，单调，下限 0.4。"""
    from makeupstudio.face3dgs.appearance.makeup_uv import lab_adapt
    assert lab_adapt(40.0, 41.0) == pytest.approx(1.0 - 0.55 * 0.01, abs=1e-9)
    assert lab_adapt(76.0, 44.0) < 0.85        # 亮肤 vs 中亮唇色：明显收缩
    assert lab_adapt(16.0, 44.0) > 0.8         # 深肤 vs 唇色：几乎不收缩
    d = np.linspace(0.0, 120.0, 121)
    v = lab_adapt(np.full_like(d, 50.0), 50.0 + d)
    assert (np.diff(v) <= 1e-12).all()                         # |ΔL| 单调不增
    assert v.min() == pytest.approx(0.40)                      # 触及下限


def test_pigment_safe_lightness_direction(canonical_cloud):
    """积分方向性：亮肤被唇色拉暗、深肤被拉亮（迁移方向正确）。"""
    cloud, landmarks = canonical_cloud
    deep = np.tile(np.array([0.22, 0.13, 0.11], np.float32), (len(cloud["xyz"]), 1))
    light = np.tile(np.array([0.87, 0.72, 0.65], np.float32), (len(cloud["xyz"]), 1))
    w_d, out_d = _apply_with_skin(cloud, landmarks, deep)
    w_l, out_l = _apply_with_skin(cloud, landmarks, light)
    zone_d, zone_l = w_d > 0.6, w_l > 0.6
    assert zone_d.any() and zone_l.any()
    L_d = rgb2lab(np.clip(out_d[zone_d], 0, 1))[:, 0].mean()
    L_l = rgb2lab(np.clip(out_l[zone_l], 0, 1))[:, 0].mean()
    assert L_d > rgb2lab(deep[zone_d])[:, 0].mean()            # 深肤拉向唇色明度
    assert L_l < rgb2lab(light[zone_l])[:, 0].mean()           # 亮肤拉向唇色明度


# ---------------- near 软门控 ----------------

def test_near_soft_gate_suppresses_uv_outliers(canonical_cloud):
    """远离 canonical 表面的 splat（UV 归属误差）妆权重被衰减，内部不动。"""
    cloud, landmarks = canonical_cloud
    binding = bind_uv(cloud, landmarks, tex=256)
    baker = UvMakeupBaker(tex=256)
    maps = baker.bake(_lip_layers(), intensity=1.0)
    near = np.full(len(cloud["xyz"]), 1e-4, np.float32)
    rng = np.random.default_rng(3)
    outlier = rng.random(len(cloud["xyz"])) < 0.3
    near[outlier] = 1.0                          # 远超中位数
    made = baker.apply_to_cloud(cloud, maps, binding.uv, binding.valid, near=near)
    w = made["makeup_w"]
    assert w[outlier].max() < 0.05               # 离群被门控归零
    assert (w[~outlier] > 0.5).any()             # 内部妆区保留


# ---------------- 3D 眼线/眉锚定 ----------------

def test_landmark_band_3d_eyeliner_and_brow(canonical_cloud):
    cloud, landmarks = canonical_cloud
    baker = UvMakeupBaker(tex=256)
    bw, bc = baker.landmark_band_3d(cloud, landmarks, "eyeliner",
                                    {"thickness": 0.5, "wing": 0.2})
    assert float(bw.max()) > 0.3                              # 带上有权重
    assert 0 <= float(bc.min()) and float(bc.max()) <= 1.0    # 弧长渐变坐标
    # 权重集中在带附近：top-100 加权 splat 的平均带距 < 全体均距
    from makeupstudio.face3dgs.appearance.makeup_uv import EYE_UPPER, _catmull_rom
    lm = np.asarray(landmarks, np.float64)
    poly = np.vstack([_catmull_rom(lm[list(EYE_UPPER["left"])], 24),
                      _catmull_rom(lm[list(EYE_UPPER["right"])], 24)])
    d = np.linalg.norm(cloud["xyz"][:, None, :] - poly[None, :, :], axis=2).min(1)
    on_band = bw > 0.3
    assert on_band.sum() >= 8                       # 带上有采样点被赋权
    assert d[on_band].max() < 0.02                  # 带权点物理贴在睑缘
    assert d[on_band].mean() < 0.1 * d.mean()       # 显著集中于睑缘
    mw, mc = baker.landmark_band_3d(cloud, landmarks, "eyebrow", {"thickness": 0.6})
    assert float(mw.max()) > 0.3 and mc.max() <= 1.0


def test_landmark_band_3d_fallback_on_invalid_landmarks(canonical_cloud):
    cloud, _ = canonical_cloud
    baker = UvMakeupBaker(tex=256)
    bad = np.zeros((468, 3))
    bad[:50] = np.random.default_rng(1).random((50, 3))
    bw, bc = baker.landmark_band_3d(cloud, bad, "eyeliner", {})
    assert float(bw.max()) == 0.0 and float(bc.max()) == 0.0


# ---------------- 珠光闪点 ----------------

def test_glitter_finish_sparkle_sheen_variation(canonical_cloud):
    cloud, landmarks = canonical_cloud
    binding = bind_uv(cloud, landmarks, tex=256)
    baker = UvMakeupBaker(tex=256)
    layers = [{"id": "esh", "region": "eyeshadow", "enabled": True,
               "opacity": 0.9, "finish": "glitter",
               "color_stops": [{"at": 0.0, "hex": "#8E6BA8"}]}]
    maps = baker.bake(layers, intensity=1.0)
    assert "glitter" in FINISH_TARGET
    zone = (maps.w > 0.4) & (maps.channels["sheen"] > 0)
    assert zone.any()
    # 闪点 = 空间变化（稀疏亮片把局部 sheen 推向饱和）+ 哑光严格无变化
    assert float(maps.channels["sheen"][zone].std()) > 0.005
    assert float(maps.channels["sheen"][zone].max()) > 0.9
    matte = baker.bake([{**layers[0], "finish": "matte"}], intensity=1.0)
    z2 = matte.w > 0.4
    assert z2.any()
    assert float(matte.channels["sheen"][z2].std()) < 1e-6     # 哑光无闪点


# ---------------- sidecar 往返 + 妆感合成 ----------------

def test_light_and_material_bin_roundtrip(tmp_path):
    from makeupstudio.face3dgs.appearance.pbr import Material
    n = 200
    rng = np.random.default_rng(5)
    cloud = {
        "xyz": rng.random((n, 3)).astype(np.float32),
        "scale": rng.random((n, 3)).astype(np.float32) * 0.01 + 0.001,
        "rot": np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (n, 1)),
        "rgba": np.concatenate([np.full((n, 3), 0.6), np.ones((n, 1))], 1),
        "material": Material(rng.random(n).astype(np.float32) * 0.5 + 0.1,
                             np.full(n, 0.7, np.float32),
                             np.full(n, 0.8, np.float32),
                             np.full(n, 0.4, np.float32)),
    }
    export_material(cloud, tmp_path)
    mb = read_material_bin(tmp_path / "material.bin")
    assert len(mb["rough"]) == n
    assert mb["coat"].mean() == pytest.approx(0.7, abs=1e-5)
    assert mb["sss"].mean() == pytest.approx(0.8, abs=1e-5)
    assert np.abs(np.linalg.norm(mb["normal"], axis=1) - 1).max() < 1e-5

    d, s, tint = np.array([0.1, 0.2, 0.97]), 0.8, np.array([1.0, 0.95, 0.9])
    write_light_bin(tmp_path / "light.bin", d, s, tint)
    d2, s2, t2 = read_light_bin(tmp_path / "light.bin")
    assert np.allclose(d2, d, atol=1e-5) and s2 == pytest.approx(s)
    assert np.allclose(t2, tint, atol=1e-5)


def test_build_shade_prefers_in_memory_material(tmp_path):
    from makeupstudio.face3dgs.appearance.pbr import Material
    n = 150
    cloud = {
        "xyz": np.random.default_rng(7).random((n, 3)).astype(np.float32),
        "scale": np.full((n, 3), 0.01, np.float32),
        "rot": np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (n, 1)),
        "rgba": np.ones((n, 4), np.float32),
        "material": Material(np.full(n, 0.3, np.float32), np.full(n, 0.9, np.float32),
                             np.zeros(n, np.float32), np.full(n, 0.5, np.float32)),
    }
    ctx = build_shade(cloud, mat_dir=tmp_path)
    assert ctx is not None
    assert ctx.coat.mean() == pytest.approx(0.9, abs=1e-6)
    assert abs(np.linalg.norm(ctx.light_dir) - 1) < 1e-6        # 回退光已归一


def test_composite_shade_spec_sss_and_alpha_gate():
    """清漆高光只加不减、sss 背光红移偏红、alpha=0 背景不动。"""
    h = w = 8
    img = np.full((h, w, 3), 0.6)
    alpha = np.ones((h, w))
    normal = np.zeros((h, w, 3)); normal[..., 2] = 1.0
    rough = np.full((h, w), 0.2)
    coat = np.zeros((h, w)); coat[:, :4] = 1.0
    sss = np.zeros((h, w))
    sheen = np.zeros((h, w))
    zmap = np.full((h, w), 3.0)
    # 相机在 world (0,0,3)、z 轴指向原点（OpenCV 约定：R=diag(1,-1,-1)）
    w2c = np.diag([1.0, -1.0, -1.0, 1.0])
    w2c[:3, 3] = [0.0, 0.0, 3.0]
    K = np.array([[8.0, 0, 4], [0, 8.0, 4], [0, 0, 1]], np.float64)
    L = np.array([0.0, 0.0, 1.0])                # 光与视线同向 → 正对高光
    out = composite_shade(img, alpha, normal, rough, coat, sss, sheen,
                          zmap, w2c, K, L, np.ones(3))
    assert (out[:, :4] >= out[:, 4:] - 1e-9).all()             # 清漆区更亮
    assert (out[:, :4] > img[:, :4] + 1e-6).any()

    sss2 = np.ones((h, w))
    back = composite_shade(img, alpha, normal, rough, coat * 0, sss2, sheen,
                           zmap, w2c, K, -L, np.ones(3))       # 光从背面 → 透光
    lift = back - img
    assert lift[..., 0].mean() > lift[..., 2].mean()           # 红移

    alpha0 = alpha.copy(); alpha0[0, :] = 0.0
    out0 = composite_shade(img, alpha0, normal, rough, coat, sss, sheen,
                           zmap, w2c, K, L, np.ones(3))
    assert np.abs(out0[0] - img[0]).max() < 1e-9               # 背景不被污染


# ---------------- 图像空间蒙版 / 参考图标定 / ΔE 度量 ----------------

@pytest.fixture(scope="module")
def face_canvas():
    """canonical UV 展开成 512² 合成正脸图：肤色底 + 红唇。"""
    import cv2
    from makeupstudio.face3dgs.fit_makeup import FaceMakeupFitter
    f = FaceMakeupFitter(tracker_factory=lambda: (_ for _ in ()).throw(RuntimeError))
    size = 512
    px = np.zeros((478, 2), np.float32)
    uv = np.clip(f.model.uvs[:468], 0, 1)
    px[:468, 0] = uv[:, 0] * size
    px[:468, 1] = (1.0 - uv[:, 1]) * size
    img = np.full((size, size, 3), 190, np.uint8)              # BGR 肤色灰底
    img[..., 0] = 170; img[..., 1] = 185; img[..., 2] = 205
    lip = px[list(LIPS_OUTER)].astype(np.int32)
    cv2.fillPoly(img, [lip], (60, 60, 200), lineType=cv2.LINE_AA)   # 红唇
    return img, px


def test_image_region_masks_geometry(face_canvas):
    img, px = face_canvas
    masks = image_region_masks(px, img.shape[1], img.shape[0])
    for r in ("foundation", "lipstick", "eyeshadow", "eyebrow", "eyeliner", "blush"):
        assert r in masks and float(masks[r].max()) > 0.5
    # 唇在脸内；眼影主体在睑缘上方（上睑中点 y 的上方）
    lip_core = masks["lipstick"] > 0.75
    fd_core = masks["foundation"] > 0.75
    assert (lip_core & ~fd_core).sum() < 0.2 * lip_core.sum()
    from makeupstudio.face3dgs.appearance.calibrate import EYE_UPPER
    lid_y = float(np.mean([px[list(EYE_UPPER[s])][:, 1].mean()
                           for s in ("left", "right")]))
    ys, _xs = np.nonzero(masks["eyeshadow"] > 0.6)
    assert ys.mean() < lid_y + 2                               # 眼影主体在睑缘上方


def test_extract_and_calibrate_spec_roundtrip(face_canvas):
    img, px = face_canvas
    ext = extract_makeup_colors(img, px=px)
    assert "lipstick" in ext
    lip_lab = ext["lipstick"]["lab"]
    tgt = rgb2lab(hex_to_rgb01("#C83C3C")[None, :])[0]         # 画的 (200,60,60) BGR
    assert ciede2000(np.array([lip_lab]), tgt[None, :])[0] < 12
    assert ext["lipstick"]["opacity"] > 0.4                    # 红唇浓度显著

    template = {"layers": [
        {"region": "lipstick", "enabled": True, "opacity": 0.5, "finish": "gloss",
         "color_stops": [{"at": 0.0, "hex": "#111111"}, {"at": 1.0, "hex": "#222222"}]},
        {"region": "blush", "enabled": True, "opacity": 0.5, "finish": "matte",
         "color_stops": [{"at": 0.0, "hex": "#333333"}]}]}
    spec = calibrate_spec(template, img, px=px)
    lip_layer = spec["layers"][0]
    assert lip_layer["color_stops"][0]["hex"] != "#111111"     # 已被参考图替换
    assert lip_layer["opacity"] == ext["lipstick"]["opacity"]
    assert "lipstick" in spec["calibration"]["regions"]


def test_region_delta_e_zero_when_exact_and_grows_with_error(face_canvas):
    img, px = face_canvas
    masks = image_region_masks(px, img.shape[1], img.shape[0])
    targets = {"lipstick": rgb2lab(hex_to_rgb01("#C83C3C")[None, :])[0]}
    d0 = region_delta_e(img[..., ::-1] / 255.0, masks, targets)
    assert d0["lipstick"] < 8                                  # 画的就是目标色
    off = img.copy().astype(np.float32)
    off[..., 2] = np.clip(off[..., 2].astype(np.float32) * 0.4, 0, 255)  # 唇色拉暗
    d1 = region_delta_e(np.clip(off, 0, 255)[..., ::-1] / 255.0, masks, targets)
    assert d1["lipstick"] > d0["lipstick"]


def test_spec_targets_averages_stops():
    spec = {"layers": [
        {"region": "lipstick", "enabled": True,
         "color_stops": [{"at": 0.0, "hex": "#000000"}, {"at": 1.0, "hex": "#FEFEFE"}]},
        {"region": "blush", "enabled": False,
         "color_stops": [{"at": 0.0, "hex": "#123456"}]}]}
    t = spec_targets(spec)
    assert "blush" not in t                                    # 关闭层不参与
    expect = rgb2lab(np.array([[0.5, 0.5, 0.5]]))[0]           # 黑白均色 = 中灰
    assert np.abs(t["lipstick"] - expect).max() < 1.0


# ---------------- 妆容壳层 = 真实高斯球 ----------------

def _build_layer(cloud, landmarks, tex=256, **kw):
    binding = bind_uv(cloud, landmarks, tex=tex)
    baker = UvMakeupBaker(tex=tex)
    maps = baker.bake(_lip_layers(), intensity=1.0)
    return baker, maps, binding, *baker.build_makeup_layer(
        cloud, maps, binding.uv, binding.valid, near=binding.near, **kw)


def test_makeup_layer_splats_overlap_base_positions(canonical_cloud):
    """壳层 = 底模对应 splat 表面上的薄层：同 rot，沿外法线偏移
    ≤ normal_offset×min(scale)（几何上存在、在足迹内不外凸、方向朝外）。
    scale 薄轴与底模一致；面内轴按边缘补偿放大（w<edge_ref 的低权重
    splat，≤1+SHELL_EDGE_BOOST）——妆缘摊匀不斑驳，层厚不因边缘而变。"""
    from makeupstudio.face3dgs.appearance.makeup_uv import (
        SHELL_EDGE_BOOST, SHELL_EDGE_REF)
    cloud, landmarks = canonical_cloud
    _baker, _maps, _binding, layer, idx = _build_layer(cloud, landmarks)
    assert len(idx) > 100                                      # 唇区有足量壳层
    assert np.abs(layer["rot"] - cloud["rot"][idx]).max() == 0.0
    base_scale = np.asarray(cloud["scale"], np.float32)[idx]
    thin = base_scale.argmin(axis=1)
    rows = np.arange(len(idx))
    assert np.allclose(layer["scale"][rows, thin], base_scale[rows, thin],
                       rtol=1e-5)                              # 薄轴不动
    other = np.ones_like(layer["scale"], dtype=bool)
    other[rows, thin] = False
    ratio = layer["scale"][other] / base_scale[other]
    assert ratio.min() >= 1.0 - 1e-5                           # 只放不缩
    assert ratio.max() <= 1.0 + SHELL_EDGE_BOOST + 1e-4        # 补偿上界
    # 放大只发生在低权重（边缘）splat：w≥edge_ref 的核心完全一致
    core = layer["makeup_w"] >= SHELL_EDGE_REF
    if core.any():
        assert np.allclose(layer["scale"][core], base_scale[core], rtol=1e-5)
    off = layer["xyz"] - cloud["xyz"][idx]
    d = np.linalg.norm(off, axis=1)
    thin_w = np.asarray(cloud["scale"], np.float64)[idx].min(axis=1)
    assert (d <= 0.15 * thin_w * 1.01 + 1e-9).all()            # 薄层厚度上界
    assert d.mean() > 0                                        # 确有物理厚度
    radial = cloud["xyz"][idx] - np.median(np.asarray(cloud["xyz"]), axis=0)
    assert ((off * radial).sum(1) >= -1e-9).mean() > 0.95      # 偏移朝外


def test_makeup_layer_opacity_and_color_semantics(canonical_cloud):
    """opacity = w×底模；颜色 = 完整迁移（比原位折叠迁移更贴近妆色目标）。"""
    cloud, landmarks = canonical_cloud
    baker, maps, binding, layer, idx = _build_layer(cloud, landmarks)
    made = baker.apply_to_cloud(cloud, maps, binding.uv, binding.valid,
                                near=binding.near)
    w_sel = made["makeup_w"][idx]
    base_op = np.clip(cloud["rgba"][idx, 3], 0, 0.98)
    expect_op = np.clip(w_sel * base_op, 1e-4, 0.98)
    assert np.abs(layer["rgba"][:, 3] - expect_op).max() < 1e-6
    # 完整迁移（壳层色）的迁移位移 ≥ 折叠迁移（原位色）：L 位移更大、
    # 色度更饱和（chroma>1 的"强色度"外推语义在壳层上保持）
    zone = w_sel > 0.6
    cur_lab = rgb2lab(cloud["rgba"][idx][zone, :3])
    full_lab = rgb2lab(layer["rgba"][zone, :3])
    fold_lab = rgb2lab(made["rgba"][idx][zone, :3])
    dL_full = np.abs(full_lab[:, 0] - cur_lab[:, 0]).mean()
    dL_fold = np.abs(fold_lab[:, 0] - cur_lab[:, 0]).mean()
    assert dL_full > dL_fold                                   # 明度位移更大
    c_full = np.hypot(full_lab[:, 1], full_lab[:, 2]).mean()
    c_fold = np.hypot(fold_lab[:, 1], fold_lab[:, 2]).mean()
    assert c_full > c_fold                                     # 色度更饱和
    # sh_rest 置零：妆色视角无关，不被底模素颜 SH 残差调制
    assert float(np.abs(layer["sh_rest"]).max()) == 0.0


def test_makeup_layer_texture_passthrough(canonical_cloud):
    """底模真实纹理透出：唇区亮度有空间变化的底模，壳层颜色的纹理方差
    随 hp_gain 增大（妆下纹理是训练出来的，不是程序噪声平涂）。"""
    cloud, landmarks = canonical_cloud
    cloud = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in cloud.items()}
    binding = bind_uv(cloud, landmarks, tex=256)
    lum = 0.6 + 0.15 * np.sin(binding.uv[:, 0] * 40.0)     # UV 高频亮度条纹
    cloud["rgba"][:, :3] = np.clip(lum[:, None] * np.array([[1.0, 0.9, 0.9]]), 0, 1)
    baker = UvMakeupBaker(tex=256)
    maps = baker.bake(_lip_layers(), intensity=1.0)
    layer0, _ = baker.build_makeup_layer(cloud, maps, binding.uv, binding.valid,
                                         near=binding.near, hp_gain=0.0)
    layer1, _ = baker.build_makeup_layer(cloud, maps, binding.uv, binding.valid,
                                         near=binding.near, hp_gain=0.25)
    zone = layer1["makeup_w"] > 0.6
    assert zone.sum() > 30
    std0 = float(layer0["rgba"][zone, 0].std())
    std1 = float(layer1["rgba"][zone, 0].std())
    assert std1 > std0 * 1.5                                   # 纹理透出增强


def test_makeup_layer_material_full_strength(canonical_cloud):
    """唇区壳层 splat 携带 gloss finish 全强度（coat≈0.82/sss=1），底模保持皮肤。"""
    cloud, landmarks = canonical_cloud
    _baker, _maps, _binding, layer, idx = _build_layer(cloud, landmarks)
    zone = layer["makeup_w"] > 0.6
    assert zone.any()
    coat_t = FINISH_TARGET["gloss"][1]
    assert layer["material"].coat[zone].mean() == pytest.approx(coat_t, abs=0.05)
    assert layer["material"].sss[zone].mean() == pytest.approx(1.0, abs=0.05)
    assert layer["material"].rough[zone].mean() == pytest.approx(
        FINISH_TARGET["gloss"][0], abs=0.05)


def test_merge_makeup_layer_structure(canonical_cloud):
    """合并点云：底模保持素颜、长度 = 底模+壳层、逐 splat 字段自洽。"""
    cloud, landmarks = canonical_cloud
    _baker, _maps, _binding, layer, idx = _build_layer(cloud, landmarks)
    n_b = len(cloud["xyz"])
    merged = merge_makeup_layer(cloud, layer, idx)
    assert len(merged["xyz"]) == n_b + len(idx)
    assert np.abs(merged["rgba"][:n_b] - cloud["rgba"]).max() < 1e-6   # 底模未被染
    assert merged["rgba"][n_b:, 3].min() > 0.0                          # 壳层带透明度
    assert np.abs(merged["sh_rest"][n_b:]).max() == 0.0                 # 壳层 DC-only
    assert merged["sh_rest"][:n_b].shape == cloud["sh_rest"].shape
    assert merged["makeup_w"][:n_b].max() == 0.0                        # identity 锁
    assert merged["makeup_w"][n_b:].min() > 0.05
    mat = merged["material"]
    assert len(mat.coat) == n_b + len(idx)
    # uv 绑定字段延伸（换妆重绑自洽）
    cloud_uv = {**cloud, "uv": np.random.default_rng(2).random((n_b, 2)).astype(np.float32)}
    _b2, _m2, _bd2, layer2, idx2 = _build_layer(cloud_uv, landmarks)
    merged2 = merge_makeup_layer(cloud_uv, layer2, idx2)
    assert np.abs(merged2["uv"][n_b:] - cloud_uv["uv"][idx2]).max() < 1e-6


def test_apply_makeup_to_asset_layer_mode_default(canonical_cloud, tmp_path):
    """产品入口默认壳层架构：madeup 比底模多出壳层 splat，底模 rgba 原样。"""
    cloud, landmarks = canonical_cloud
    spec = {"layers": _lip_layers()}
    made, cov = apply_makeup_to_asset(cloud, landmarks, spec, tmp_path, tex=256)
    n_b = len(cloud["xyz"])
    assert cov > 0.0
    assert len(made["xyz"]) > n_b                              # 壳层真实存在
    assert np.abs(made["rgba"][:n_b] - cloud["rgba"]).max() < 1e-6
    assert (made["makeup_w"][n_b:] > 0.02).any()
    # 壳层 splat 落在对应底模 splat 足迹内（薄层偏移 ≤ 0.16×min_scale）
    layer_xyz = made["xyz"][n_b:]
    src_w = made["makeup_w"][n_b:]
    # 每个壳层 splat 的最近底模 splat 就是其对应者（薄层偏移量级）
    import cv2
    fl = cv2.flann_Index(cloud["xyz"].astype(np.float32),
                         dict(algorithm=1, trees=4, checks=64))
    _nn, d2 = fl.knnSearch(layer_xyz.astype(np.float32), 1, params=dict(checks=64))
    d = np.sqrt(np.maximum(d2, 0))
    thin_med = float(np.median(np.asarray(cloud["scale"], np.float64).min(axis=1)))
    assert d.max() <= 0.16 * thin_med * 1.01
    assert src_w.max() > 0.5


def test_apply_makeup_to_asset_recolor_fallback(canonical_cloud, tmp_path):
    """as_layer=False 退回原位重染色：长度不变、颜色被迁移。"""
    cloud, landmarks = canonical_cloud
    spec = {"layers": _lip_layers()}
    made, _cov = apply_makeup_to_asset(cloud, landmarks, spec, tmp_path,
                                       tex=256, as_layer=False)
    assert len(made["xyz"]) == len(cloud["xyz"])
    moved = np.abs(made["rgba"][:, :3] - cloud["rgba"][:, :3]).sum(1)
    assert (moved > 0.02).any()
