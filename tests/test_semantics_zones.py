"""P1 妆区语义观测 + P3 边界羽化/SH 分区 的离线单测（全 CPU 合成数据）。

覆盖：投影/蒙版采样、多视角投票（含背面剔除）、语义蒙版装配（唇/眼环/眼影）、
唇线 edge-snap、四级回退留痕（multiview_seg > single_seg > landmark_band >
uv_template）、观测妆区与几何兜底的权重融合（replace/multiply）、逐 splat IoU、
P3 边界羽化场（kL/chroma 场）与 SH 按色偏分区。

face-parsing 权重不在离线测试范围：prepare_views 的门控分支由 test_parser_masks
覆盖，本文件的观测视角用合成蒙版直接构造。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "desktop-app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from makeupstudio.face3dgs.appearance.makeup_uv import (  # noqa: E402
    EDGE_KL_FLOOR,
    PHOTOREAL_LAB,
    SH_SAT_THRESHOLD,
    UvMakeupBaker,
    ZoneFields,
    ZoneSpec,
    _edge_ramp,
    rgb2lab,
)
from makeupstudio.face3dgs.appearance.semantics import (  # noqa: E402
    MULTIVIEW_MIN,
    camera_matrix,
    build_zones,
    edge_snap,
    front_facing,
    group_mask,
    iou,
    project_points,
    region_mask,
    sample_mask,
    save_zone_closeups,
    snap_to_edge,
    vote_regions,
)
from makeupstudio.face3dgs.appearance.uvbind import bind_uv  # noqa: E402

SIZE = 512


# ---------------- 合成数据 ----------------

def _cam(fx: float = 500.0, size: int = SIZE):
    """相机在 z=-1.5 看向 +Z（COLMAP：x 右 / y 下 / z 前），脸平面放在 z=0。"""
    K = np.array([[fx, 0.0, size / 2], [0.0, fx, size / 2], [0.0, 0.0, 1.0]])
    return np.eye(3), np.array([0.0, 0.0, 1.5]), K


def _view(mask: np.ndarray, R=None, t=None, K=None, key: str = "lipstick") -> dict:
    R2, t2, K2 = _cam()
    return {"R": R if R is not None else R2, "t": t if t is not None else t2,
            "K": K if K is not None else K2, "masks": {key: mask},
            "size": (SIZE, SIZE)}


def _left_half_mask() -> np.ndarray:
    m = np.zeros((SIZE, SIZE), np.float32)
    m[:, :SIZE // 2] = 1.0
    return m


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


def _lip_layer() -> dict:
    return {"id": "lip", "region": "lipstick", "enabled": True, "opacity": 0.9,
            "finish": "gloss",
            "color_stops": [{"at": 0.0, "hex": "#C21858"},
                            {"at": 1.0, "hex": "#E91E63"}]}


# ---------------- 投影 / 采样 ----------------

def test_project_points_axes_and_cheirality():
    R, t, K = _cam()
    xyz = np.array([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.0, 0.06, 0.0],
                    [0.0, 0.0, -2.0]])
    xy, ok = project_points(xyz, R, t, K)
    assert ok[:3].all() and not ok[3]                # z<0 在相机后方
    assert xy[0] == pytest.approx((SIZE / 2, SIZE / 2))
    assert xy[1, 0] > SIZE / 2                       # +x → 图像右侧
    assert xy[2, 1] > SIZE / 2                       # +y → 图像下方


def test_sample_mask_bilinear_and_out_of_bounds():
    m = np.zeros((8, 8), np.float32)
    m[3:5, 3:5] = 1.0
    assert sample_mask(m, np.array([[3.5, 3.5]]))[0] == pytest.approx(1.0)
    assert sample_mask(m, np.array([[-5.0, 3.5]]))[0] == 0.0     # 越界 → 0
    assert sample_mask(m, np.array([[20.0, 3.5]]))[0] == 0.0
    assert 0.0 < sample_mask(m, np.array([[2.5, 3.5]]))[0] < 1.0  # 双线性过渡


def test_camera_matrix_scales_native_intrinsics():
    class _Cam:
        width = 1024
        params = (900.0, 512.0, 384.0, 0.0)
    K = camera_matrix(_Cam(), 2048)                  # 观测帧放大 2×
    assert K[0, 0] == pytest.approx(1800.0)
    assert K[0, 2] == pytest.approx(1024.0)
    assert K[1, 2] == pytest.approx(768.0)


# ---------------- 多视角投票 ----------------

def test_vote_regions_multiview_average():
    xyz = np.array([[-0.03, 0.0, 0.0], [0.03, 0.0, 0.0]])
    m = _left_half_mask()
    p, c = vote_regions(xyz, [_view(m)], ("lipstick",))["lipstick"]
    assert p[0] == pytest.approx(1.0) and p[1] == pytest.approx(0.0)
    assert (c == 1).all()
    # 第二视角：相机转到背对侧（R 翻转 x/y），左右互换 → 均值 0.5
    R2 = np.diag([-1.0, -1.0, 1.0])
    p2, c2 = vote_regions(xyz, [_view(m), _view(m, R=R2)],
                          ("lipstick",))["lipstick"]
    assert p2[0] == pytest.approx(0.5) and (c2 == 2).all()


def test_vote_regions_rejects_back_facing():
    xyz = np.array([[-0.03, 0.0, 0.0], [0.03, 0.0, 0.0]])
    v = [_view(_left_half_mask())]
    R, t, _K = _cam()
    toward = np.tile(np.array([0.0, 0.0, -1.0]), (2, 1))    # 法线朝相机
    away = np.tile(np.array([0.0, 0.0, 1.0]), (2, 1))
    assert front_facing(xyz, R, t, toward).all()
    assert not front_facing(xyz, R, t, away).any()
    _p, c_front = vote_regions(xyz, v, ("lipstick",),
                               normals=toward)["lipstick"]
    _p2, c_back = vote_regions(xyz, v, ("lipstick",),
                               normals=away)["lipstick"]
    assert (c_front == 1).all() and (c_back == 0).all()


def test_group_mask_and_iou():
    a = np.zeros((4, 4), np.float32)
    a[0, 0] = 1.0
    b = np.zeros((4, 4), np.float32)
    b[0, 1] = 1.0
    g = group_mask({"lips": a, "mouth": b}, ("lips", "mouth"))
    assert g is not None and g.sum() == 2.0
    assert group_mask({}, ("lips",)) is None
    assert iou(np.array([1, 1, 0, 0], bool), np.array([1, 0, 0, 0], bool)) \
        == pytest.approx(0.5)
    assert iou(np.zeros(3, bool), np.zeros(3, bool)) == 0.0


# ---------------- 语义蒙版装配 ----------------

def test_region_mask_shapes():
    eye = np.zeros((SIZE, SIZE), np.float32)
    eye[200:260, 150:250] = 1.0
    lips = np.zeros((SIZE, SIZE), np.float32)
    lips[300:340, 200:300] = 1.0
    skin = np.ones((SIZE, SIZE), np.float32)
    parsed = {"eye_l": eye, "eye_r": np.zeros_like(eye), "lips": lips,
              "skin": skin, "brow_l": np.zeros_like(eye)}

    lip = region_mask("lipstick", parsed, SIZE)
    assert lip is not None and lip[320, 250] == 1.0
    assert lip[10, 10] == 0.0
    # 眼线 = 眼球内外环带：眼球中心为空（不能在眼球上画眼线），环上有值
    ring = region_mask("eyeliner", parsed, SIZE)
    assert ring is not None
    assert ring[230, 200] < 0.5 and ring.sum() > 0
    # 眼影 = 眼结构外扩：眼球中心在内、外扩区也有值
    sh = region_mask("eyeshadow", parsed, SIZE)
    assert sh is not None and sh[230, 200] > 0.5 and sh.sum() > eye.sum()
    # 无对应语义类别 → None（保持几何兜底）
    assert region_mask("blush", parsed, SIZE) is None
    assert region_mask("contour", parsed, SIZE) is None


def test_snap_to_edge_pulls_contour_to_gradient():
    gray = np.zeros((SIZE, SIZE), np.float32)
    gray[:, :200] = 0.2
    gray[:, 200:] = 0.8
    gray = np.repeat(gray, 1, 0)
    mask = np.zeros((SIZE, SIZE), np.float32)
    mask[:, 202:262] = 1.0                 # 左边界离真实边缘 2px
    snapped = snap_to_edge(mask, gray, search=3)
    assert snapped[:, 200].mean() > 0.9    # 吸附到梯度极值处
    assert snapped[:, 197].mean() < 0.1    # 没有越过去
    assert snapped[:, 260].mean() > 0.9    # 无梯度的边界保持不动


def test_edge_snap_softens_boundary():
    gray = np.zeros((SIZE, SIZE), np.float32)
    gray[:, :256] = 0.2
    gray[:, 256:] = 0.8
    mask = np.zeros((SIZE, SIZE), np.float32)
    mask[:, 256:] = 1.0
    hard = edge_snap(mask, gray, feather_px=0.0)
    soft = edge_snap(mask, gray, feather_px=8.0)
    assert set(np.unique(hard)) <= {0.0, 1.0}
    mid = soft[(soft > 0.02) & (soft < 0.98)]
    assert mid.size > 0                    # 边界带内出现渐变值
    assert soft.min() >= 0.0 and soft.max() <= 1.0


# ---------------- 四级回退 / 报告留痕 ----------------

def _plane_xyz(n: int = 300):
    rng = np.random.default_rng(5)
    x = rng.uniform(-0.04, 0.04, n)
    y = rng.uniform(-0.04, 0.04, n)
    return np.stack([x, y, np.zeros(n)], 1)


def test_build_zones_multiview_and_levels():
    xyz = _plane_xyz()
    _R, _t, K = _cam()
    # 两个视角都看到同一半为唇（视差位移 → 边界处概率被投票平均）
    t2 = np.array([0.02, 0.0, 1.5])
    views = [_view(_left_half_mask()), _view(_left_half_mask(), t=t2)]
    z = build_zones(xyz, views, ("lipstick", "blush", "eyeliner"),
                    region_scale={"lipstick": 0.72})
    assert not z.fields.empty()
    spec = z.fields.seg["lipstick"]
    assert spec.mode == "replace"                       # 唇：观测即边界
    assert spec.scale == pytest.approx(0.72)            # spec opacity × intensity
    assert spec.trust > 0.8 and spec.fallback_ratio < 0.2
    assert z.report["lipstick"]["level"] == "multiview_seg"
    assert z.report["lipstick"]["splats"] > 40
    assert z.report["lipstick"]["obs_views_median"] >= MULTIVIEW_MIN
    # 无语义类别 / 观测过稀 → 几何兜底级，且不进场
    assert z.report["blush"]["level"] == "uv_template"
    assert "blush" not in z.fields.seg
    assert z.report["eyeliner"]["level"] == "uv_template"
    assert "eyeliner" not in z.fields.seg


def test_build_zones_single_view_and_no_observation():
    xyz = _plane_xyz()
    z1 = build_zones(xyz, [_view(_left_half_mask())], ("lipstick",))
    assert z1.report["lipstick"]["level"] == "single_seg"
    assert z1.fields.seg["lipstick"].mode == "replace"
    assert z1.fields.seg["lipstick"].fallback_ratio > 0.2   # 单图更依赖几何兜底

    z0 = build_zones(xyz, [], ("lipstick", "blush"))
    assert z0.fields.empty()
    assert z0.report["lipstick"]["level"] == "uv_template"
    assert z0.report["blush"]["level"] == "uv_template"
    # 几何级可由调用方指定（landmark_band 优先于 uv_template）
    z2 = build_zones(xyz, [], ("lipstick",),
                     geometric_level={"lipstick": "landmark_band"})
    assert z2.report["lipstick"]["level"] == "landmark_band"


def test_build_zones_iou_vs_geometry():
    xyz = _plane_xyz()
    views = [_view(_left_half_mask())]
    geom = np.zeros(len(xyz), np.float32)
    _, ok = project_points(xyz, *_cam())
    geom[ok] = 1.0                        # 几何兜底 = 全脸
    z = build_zones(xyz, views, ("lipstick",), geom_w=geom)
    # 观测（左半）与几何（全脸）的 IoU = 0.5 左右
    assert 0.3 < z.report["lipstick"]["iou_vs_geometry"] < 0.7


# ---------------- 观测 → 权重融合（replace / multiply） ----------------

def _assignment(cloud, landmarks, zones=None, layers=None, tex=256):
    binding = bind_uv(cloud, landmarks, tex=tex)
    baker = UvMakeupBaker(tex=tex)
    maps = baker.bake(layers or [_lip_layer()], intensity=0.8)
    a = baker.assignment(cloud, maps, binding.uv, binding.valid, zones=zones)
    return baker, maps, a


def test_replace_mode_observed_zone_overrides_geometry(canonical_cloud):
    """观测即边界：语义唇区权重由观测决定，模板带外/带内错位处都被纠正。"""
    cloud, landmarks = canonical_cloud
    _b, _m, geom = _assignment(cloud, landmarks)
    g_w = np.asarray(geom["w"])
    xyz = np.asarray(cloud["xyz"], np.float64)
    hit = np.argsort(-g_w)[:400]
    center = xyz[hit].mean(0)
    r = 1.3 * float(np.linalg.norm(xyz[hit] - center, axis=1).mean())
    # 观测唇区 = 与几何唇带同心但整体位移（模拟"模板与真实唇错位"）
    p = (np.linalg.norm(xyz - (center + np.array([0.012, -0.010, 0.0])),
                        axis=1) < r).astype(np.float32)
    zones = ZoneFields(seg={"lipstick": ZoneSpec(p=p, mode="replace",
                                                 scale=0.72, trust=0.92,
                                                 fallback_ratio=0.12)})
    _b2, _m2, a2 = _assignment(cloud, landmarks, zones=zones)
    w = np.asarray(a2["w"])
    missing = (g_w > 0.5) & (p < 0.02)      # 几何点了但观测说不是唇
    extra = (p > 0.5) & (g_w < 0.02)        # 观测说是唇但几何漏了（P1 的核心收益）
    assert missing.any() and extra.any()
    assert float(w[missing].mean()) < 0.35 * float(g_w[missing].mean())
    assert float(w[extra].mean()) > 0.4


def test_multiply_mode_only_tightens_boundary(canonical_cloud):
    """观测只收边：multiply 不会把几何权重抬到超过自身（不收编形状）。"""
    cloud, landmarks = canonical_cloud
    _b, _m, geom = _assignment(cloud, landmarks)
    g_w = np.asarray(geom["w"])
    p = (g_w > 0.5).astype(np.float32)      # 观测 = 几何（理想一致）
    zones = ZoneFields(seg={"lipstick": ZoneSpec(p=p, mode="multiply",
                                                 trust=0.92, fallback_ratio=0.12)})
    _b2, _m2, a2 = _assignment(cloud, landmarks, zones=zones)
    w = np.asarray(a2["w"])
    assert float(w.max()) <= float(g_w.max()) + 1e-6
    assert float(w[p > 0.5].mean()) > float(g_w[p > 0.5].mean()) * 0.9


def test_zone_color_field_overrides_target(canonical_cloud):
    """P2 钩子：观测参考色（color 场）覆盖 UV 目标色。"""
    cloud, landmarks = canonical_cloud
    _b, _m, geom = _assignment(cloud, landmarks)
    p = (np.asarray(geom["w"]) > 0.5).astype(np.float32)
    ref = np.tile(np.array([0.05, 0.35, 0.10], np.float32), (len(p), 1))
    zones = ZoneFields(seg={"lipstick": ZoneSpec(p=p, mode="replace", scale=0.8)},
                       color={"lipstick": ref})
    _b2, _m2, a2 = _assignment(cloud, landmarks, zones=zones)
    tgt = np.asarray(a2["tgt"])
    upd = (p > 0.5) & (np.asarray(a2["w"]) > 0.02)
    assert upd.any()
    assert np.abs(tgt[upd] - ref[upd]).max() < 1e-6


# ---------------- P3 边界羽化场 / SH 分区 ----------------

def test_edge_ramp_is_distance_shaped():
    cov = np.zeros((64, 64), np.float32)
    cov[16:48, 16:48] = 1.0
    ramp = _edge_ramp(cov, feather_px=8.0)
    assert ramp[32, 32] == pytest.approx(1.0, abs=1e-6)     # 内部全量
    assert ramp[16, 32] < 0.2                               # 边界带内被压低
    assert ramp.min() >= 0.0 and ramp.max() <= 1.0
    flat = _edge_ramp(cov, feather_px=0.0)                  # 线条区（feather=0）
    assert np.allclose(flat, 1.0)


def test_bake_kL_field_gradates_at_boundary(canonical_cloud):
    """P3：底妆 kL 场在边界带内从 EDGE_KL_FLOOR 渐入，内部回到全量。"""
    cloud, landmarks = canonical_cloud
    baker = UvMakeupBaker(tex=256)
    maps = baker.bake([{"region": "foundation", "opacity": 0.8,
                        "finish": "satin",
                        "color_stops": [{"at": 0.0, "hex": "#E8C4A8"},
                                        {"at": 1.0, "hex": "#DFBBA0"}]}],
                      intensity=1.0)
    kL_t = PHOTOREAL_LAB["foundation"][0]
    on = maps.kL > 0.0
    assert on.any()
    assert float(maps.kL[on].max()) == pytest.approx(kL_t, rel=0.02)
    assert float(maps.kL[on].min()) >= kL_t * EDGE_KL_FLOOR - 1e-6
    assert float(maps.kL[on].min()) < kL_t * 0.98


def test_build_makeup_layer_sh_partition(canonical_cloud):
    """P3 SH 分区：强色层（唇）sh_rest 归零，低饱和层（底妆）继承底模 SH。"""
    cloud, landmarks = canonical_cloud
    rng = np.random.default_rng(3)
    base_sh = rng.normal(0, 0.05, (len(cloud["xyz"]), 3, 3)).astype(np.float32)
    cloud = {**cloud, "sh_rest": base_sh}
    cloud["rgba"][:, :3] = np.array([0.62, 0.55, 0.50], np.float32)   # 素颜肤色
    binding = bind_uv(cloud, landmarks, tex=256)
    baker = UvMakeupBaker(tex=256)

    def _layer(layers, **kw):
        maps = baker.bake(layers, intensity=1.0)
        lay, idx = baker.build_makeup_layer(cloud, maps, binding.uv,
                                            binding.valid, **kw)
        return lay, idx

    strong, idx_s = _layer([_lip_layer()])
    assert len(idx_s) and float(np.abs(strong["sh_rest"]).max()) == 0.0

    soft_spec = [{"region": "foundation", "opacity": 0.7, "finish": "satin",
                  "color_stops": [{"at": 0.0, "hex": "#9E9186"},
                                  {"at": 1.0, "hex": "#9A8D82"}]}]
    soft, idx_f = _layer(soft_spec)
    assert len(idx_f)
    dev = np.linalg.norm(rgb2lab(np.asarray(soft["rgba"])[:, :3])
                         - rgb2lab(np.asarray(cloud["rgba"])[idx_f, :3]), axis=1)
    assert float(dev.max()) <= SH_SAT_THRESHOLD          # 确属低饱和层
    assert float(np.abs(soft["sh_rest"]).max()) > 0.0    # 继承了底模 SH 残差
    assert np.allclose(soft["sh_rest"], base_sh[idx_f], atol=1e-6)

    off, _idx = _layer(soft_spec, sh_partition=False)
    assert float(np.abs(off["sh_rest"]).max()) == 0.0    # 关掉分区 → 旧行为


# ---------------- 验收产物 ----------------

def test_save_zone_closeups(tmp_path):
    px = np.zeros((478, 2), np.float64)
    px[61] = (180, 300)
    px[291] = (300, 300)
    px[0] = (240, 290)
    px[17] = (240, 320)
    px[33] = (160, 200)
    px[133] = (210, 200)
    px[159] = (185, 190)
    px[145] = (185, 215)
    px[362] = (280, 200)
    px[263] = (330, 200)
    px[386] = (305, 190)
    px[374] = (305, 215)
    ref = np.full((512, 512, 3), 200, np.uint8)
    made = np.full((512, 512, 3), 120, np.uint8)
    outs = save_zone_closeups(px, ref, None, made, tmp_path, scale=4)
    assert len(outs) == 3
    for p in outs:
        assert p.exists() and p.stat().st_size > 0
    assert (tmp_path / "closeup_mouth.png").exists()