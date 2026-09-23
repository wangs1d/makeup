"""底模质量 + 还原度闭环 + 形状迁移回归（Q 升级）：嘴内/眼球损失屏蔽、
SH 正则配置、powder 压油光（SH 衰减）、壳层边缘补偿、参考妆照形状反投影、
ΔE 自动重标定方向逻辑。全部 CPU 合成数据；CUDA 训练循环不在离线测试范围。"""
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
    POWDER_SH_GAIN,
    SHELL_EDGE_BOOST,
    SHELL_EDGE_REF,
    UvMakeupBaker,
    merge_makeup_layer,
)
from makeupstudio.face3dgs.appearance.pipeline import (  # noqa: E402
    _adjust_spec_from_delta,
)
from makeupstudio.face3dgs.appearance.refshape import (  # noqa: E402
    affine_between,
    reference_shape_bands,
    sample_mask,
)
from makeupstudio.face3dgs.appearance.train_base import (  # noqa: E402
    EYEBALL_RING,
    TrainConfig,
    _mouth_interior_ring,
    _suppression_mask,
)
from makeupstudio.face3dgs.appearance.uvbind import bind_uv  # noqa: E402


@pytest.fixture(scope="module")
def canonical_cloud():
    from makeupstudio.face3dgs.fit_makeup import FaceMakeupFitter, _sample_tris
    f = FaceMakeupFitter(tracker_factory=lambda: (_ for _ in ()).throw(RuntimeError))
    rng = np.random.default_rng(23)
    P, _b, _rows = _sample_tris(f.model.base, f.model.tris, 12000, rng)
    cloud = {
        "xyz": P.astype(np.float32),
        "scale": np.full((len(P), 3), 0.002, np.float32),
        "rot": np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (len(P), 1)),
        "rgba": np.concatenate([np.full((len(P), 3), 0.6), np.ones((len(P), 1))], 1),
        "sh_rest": np.full((len(P), 3, 3), 0.05, np.float32),
    }
    return cloud, f.model.base[:468]


def _foundation_layers() -> list[dict]:
    return [{"id": "fd", "region": "foundation", "enabled": True, "opacity": 0.8,
             "finish": "matte",
             "color_stops": [{"at": 0.0, "hex": "#F0C8B0"},
                             {"at": 1.0, "hex": "#F4D0BC"}]}]


# ---------------- 训练屏蔽（P0-1/P0-2） ----------------

def test_mouth_interior_ring_loads():
    ring = _mouth_interior_ring()
    assert len(ring) >= 10
    assert ring.min() >= 0 and ring.max() < 468


def test_suppression_mask_hole_and_fallback():
    h = w = 128
    poly = np.array([[32.0, 32.0], [96.0, 32.0], [96.0, 96.0], [32.0, 96.0]])
    m = _suppression_mask(h, w, poly, shrink=0.85, weight=0.12, feather_px=8)
    assert m.shape == (h, w)
    assert m[64, 64] == pytest.approx(0.12, abs=0.02)          # 洞内 = weight
    assert m[4, 4] == pytest.approx(1.0, abs=1e-6)             # 洞外不衰减
    # 退化多边形（闭嘴零面积/无效地标）→ 全 1 不屏蔽
    deg = np.zeros((4, 2), np.float64)
    assert _suppression_mask(h, w, deg, 0.85, 0.12, 8).min() == 1.0
    assert _suppression_mask(h, w, poly, 1.0, 0.12, 8)[64, 64] < 0.2


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.eval_holdout == 4                # 训练视图优先
    assert cfg.sh_weight > 0                    # SH 正则默认开
    assert 0 < cfg.mouth_weight < 0.3
    assert cfg.eye_weight == 1.0                # 眼球不挖洞（细节优先）
    assert cfg.max_gs >= 900_000                # 密度上限不再饿死 densify
    assert cfg.grow_grad2d < 2e-4               # 高频区更早分裂
    assert cfg.exposure_comp                    # 逐视图曝光补偿默认开
    assert len(EYEBALL_RING["left"]) >= 12 and len(EYEBALL_RING["right"]) >= 12


# ---------------- powder 压油光 + 边缘补偿（P3-1/P3-2） ----------------

def test_powder_attenuates_base_sh_in_place(canonical_cloud):
    cloud, landmarks = canonical_cloud
    binding = bind_uv(cloud, landmarks, tex=256)
    baker = UvMakeupBaker(tex=256)
    maps = baker.bake(_foundation_layers(), intensity=1.0)
    assert maps.powder_w is not None and float(maps.powder_w.max()) > 0.2
    made = baker.apply_to_cloud(cloud, maps, binding.uv, binding.valid)
    w = np.clip(made["makeup_w"], 0, 1)
    base = cloud["sh_rest"]
    hit = w > 0.6
    assert hit.any()
    # 粉覆盖处 SH 残差按 1 - gain*min(w,1) 衰减（逐 splat 权重）
    expect = base * (1 - POWDER_SH_GAIN * w)[:, None, None]
    assert np.allclose(made["sh_rest"], expect, atol=1e-6)


def test_shell_layer_sh_att_and_edge_boost(canonical_cloud):
    cloud, landmarks = canonical_cloud
    binding = bind_uv(cloud, landmarks, tex=256)
    baker = UvMakeupBaker(tex=256)
    maps = baker.bake(_foundation_layers(), intensity=1.0)
    layer, idx = baker.build_makeup_layer(cloud, maps, binding.uv, binding.valid)
    assert len(idx) > 100
    att = layer["_sh_att"]
    assert ((att > 0) & (att < 1)).all()          # 有粉 → 衰减系数 ∈ (0,1)
    # 边缘补偿：低 w splat 面内轴 > 底模 scale，薄轴不动
    w_layer = layer["makeup_w"]
    low = w_layer < SHELL_EDGE_REF * 0.5
    thin = np.asarray(cloud["scale"], np.float32)[idx].argmin(axis=1)
    if low.any():
        i = int(np.nonzero(low)[0][0])
        scale_l = layer["scale"][i]
        scale_b = np.asarray(cloud["scale"], np.float32)[idx[i]]
        assert scale_l[thin[i]] == pytest.approx(scale_b[thin[i]], rel=1e-5)
        assert scale_l.sum() > scale_b.sum()      # 其余两轴被放大
    made = merge_makeup_layer(cloud, layer, idx)
    assert "_sh_att" not in made                  # 私有键被消费后剥离
    n_b = len(cloud["xyz"])
    base_att = 1 - POWDER_SH_GAIN * np.minimum(w_layer, 1.0)
    expect = cloud["sh_rest"][idx] * base_att[:, None, None]
    assert np.allclose(made["sh_rest"][:n_b][idx], expect, atol=1e-6)


# ---------------- 参考形状反投影（P2） ----------------

def _synthetic_layout(n: int = 468, seed: int = 3) -> np.ndarray:
    """确定性"地标布局"：伪随机但可复现的 2D 点集（归一到 [64,192]）。"""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2)) * 128 + 64
    pts[0] = (128, 64)                            # 固定几个锚点防退化
    pts[1] = (64, 192)
    pts[2] = (192, 192)
    return pts.astype(np.float32)


def test_affine_between_recovers_similarity():
    ref = _synthetic_layout()
    theta = 0.08
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    user = (ref @ R.T * 1.12 + np.array([12.0, -7.0])).astype(np.float32)
    M = affine_between(ref, user)
    assert M is not None
    one = np.array([128.0, 100.0, 1.0])
    got = M @ one
    want = user[0] if False else (R @ np.array([128.0, 100.0]) * 1.12
                                  + np.array([12.0, -7.0]))
    assert np.allclose(got, want, atol=2.0)


def test_affine_between_insufficient_points():
    assert affine_between(np.zeros((5, 2), np.float32),
                          np.zeros((5, 2), np.float32)) is None


def test_reference_shape_bands_project_and_sample():
    rng = np.random.default_rng(9)
    ref = _synthetic_layout()
    user = ref + 3.0                              # 平移即可
    M = affine_between(ref, user)
    assert M is not None
    # 256² 帧，蒙版 = 左上角方块（参考图空间）
    mask = np.zeros((256, 256), np.float32)
    mask[40:120, 40:120] = 1.0
    # splat 世界系摆在 z=5 平面，K 使投影铺满帧
    K = np.array([[200.0, 0, 128], [0, 200.0, 128], [0, 0, 1]])
    xyz = np.column_stack([rng.uniform(-1.2, 1.2, 4000),
                           rng.uniform(-1.2, 1.2, 4000),
                           np.full(4000, 5.0)])
    w2c = np.eye(4)
    bands = reference_shape_bands({"eyeshadow": mask}, M, xyz, w2c, K,
                                  (256, 256), regions=("eyeshadow",))
    assert "eyeshadow" in bands
    w, cent = bands["eyeshadow"]
    px = (K[0, 0] * xyz[:, 0] / 5.0 + 128, K[1, 1] * xyz[:, 1] / 5.0 + 128)
    inside = (px[0] >= 40) & (px[0] < 120) & (px[1] >= 40) & (px[1] < 120)
    assert inside.any()
    assert (w[inside] > 0.9).mean() > 0.9         # 投影进方块内的 splat 全权重
    assert (w[~inside] < 0.1).mean() > 0.9
    assert np.allclose(cent, 0.0)


def test_sample_mask_bilinear_bounds():
    m = np.zeros((16, 16), np.float32)
    m[8, 8] = 1.0
    v = sample_mask(m, np.array([[7.5, 7.5], [0.0, 0.0], [99.0, 99.0]]))
    assert v[0] == pytest.approx(0.25, abs=1e-6)   # 四格各 1/4
    assert v[1] == 0.0 and v[2] == 0.0             # 越界钳边=0


# ---------------- 自动重标定方向逻辑（P1-2） ----------------

def test_adjust_spec_scales_opacity_toward_target():
    spec = {"layers": [
        {"id": "lip", "region": "lipstick", "opacity": 0.6,
         "color_stops": [{"at": 0.0, "hex": "#C21858"}]},
        {"id": "fd", "region": "foundation", "opacity": 0.5,
         "color_stops": [{"at": 0.0, "hex": "#F0C8B0"}]},
    ]}
    # 唇：渲染 chroma 只有一半（缺妆）→ 提浓
    delta = {"lipstick": {"de": 20.0, "lab": [50, 20, 10],
                          "tgt": [50, 40, 20]},
             "foundation": {"de": 4.0, "lab": [60, 5, 8],
                            "tgt": [65, 5, 8]}}
    spec2 = _adjust_spec_from_delta(spec, delta)
    assert spec2 is not None
    lip = next(l for l in spec2["layers"] if l["region"] == "lipstick")
    assert lip["opacity"] > 0.6                   # 缺妆 → 提浓
    fd = next(l for l in spec2["layers"] if l["region"] == "foundation")
    assert fd["opacity"] == 0.5                   # 未超阈值 → 不动
    # 唇：渲染 chroma 是目标两倍（过妆）→ 压低
    delta_over = {"lipstick": {"de": 18.0, "lab": [50, 80, 40],
                               "tgt": [50, 40, 20]}}
    spec3 = _adjust_spec_from_delta(spec, delta_over)
    lip3 = next(l for l in spec3["layers"] if l["region"] == "lipstick")
    assert lip3["opacity"] < 0.6
    # 全部区域低于阈值 → None（无需迭代）
    assert _adjust_spec_from_delta(spec, {"lipstick": {"de": 5.0,
                                                       "lab": [50, 40, 20],
                                                       "tgt": [50, 40, 20]}}) is None


def test_vlm_score_gated_without_key(monkeypatch):
    from makeupstudio.face3dgs.appearance import vlm_score
    monkeypatch.delenv("MAKEUP_VLM_API_KEY", raising=False)
    img = np.zeros((8, 8, 3), np.float32)
    assert vlm_score.score_makeup([img], [img]) is None
