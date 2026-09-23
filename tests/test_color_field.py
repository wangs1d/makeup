"""test_color_field — P2 颜色忠实：参考驱动色彩场 + 迁移系数自求解 + ΔE00 闭环。

覆盖：向心度场（inness_field）、参考图 Lab 剖面（profile_from_mask /
extract_profiles：唇内深外浅的层次必须落在剖面里）、剖面 → 多档 color_stops、
spec_from_profiles / calibrate_spec 的参考优先与两档兜底、逐 splat 参考色配对
（pair_region_samples：同一区域里唇线 splat 拿到唇线像素色）、最小二乘自求解
（solve_region_coeffs / compare_coeffs：解出的 ΔE00 必须优于手工表）、闭环修正
（loop_delta 方向/封顶 + shift_profile 保梯度形状）、渲染帧实测 Lab 与 ΔE00、
以及"系数不依赖语义分割也生效"的端到端接线（apply_makeup_to_asset +
color_report）。全部 CPU/numpy，不依赖 torch/CUDA。"""
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
    extract_makeup_colors,
)
from makeupstudio.face3dgs.appearance.colorfield import (  # noqa: E402
    COEFF_BOUNDS,
    LOOP_STEP_MAX,
    compare_coeffs,
    extract_profiles,
    inness_field,
    loop_delta,
    manual_coeff,
    measure_region_lab,
    open_loop_delta_e,
    pair_region_samples,
    profile_from_mask,
    profile_to_stops,
    reference_core_lab,
    region_cent_uv,
    region_lab_profile,
    sample_profile_lab,
    save_coeff_table,
    shift_profile,
    solve_region_coeffs,
    spec_from_profiles,
)
from makeupstudio.face3dgs.appearance.makeup_uv import (  # noqa: E402
    UvMakeupBaker,
    ZoneFields,
    lab2rgb,
    rgb2lab,
)
from makeupstudio.face3dgs.appearance.pipeline import (  # noqa: E402
    _nudge_profiles,
    _solve_color_coeffs,
    apply_makeup_to_asset,
)
from makeupstudio.face3dgs.appearance.uvbind import bind_uv  # noqa: E402

LIP_IN = np.array([150.0, 40.0, 40.0])      # 唇心深红（RGB 0..255）
LIP_OUT = np.array([215.0, 90.0, 90.0])     # 唇缘浅红
SKIN = np.array([205.0, 185.0, 170.0])      # 素颜肤色（RGB）


# ---------------- 合成数据 ----------------

@pytest.fixture(scope="module")
def canonical_cloud():
    """canonical 点云 + 地标（与 test_fidelity_upgrade 同款合成资产）。"""
    from makeupstudio.face3dgs.fit_makeup import FaceMakeupFitter, _sample_tris
    f = FaceMakeupFitter(tracker_factory=lambda: (_ for _ in ()).throw(RuntimeError))
    rng = np.random.default_rng(11)
    P, _b, _rows = _sample_tris(f.model.base, f.model.tris, 12000, rng)
    cloud = {
        "xyz": P.astype(np.float32),
        "scale": np.full((len(P), 3), 0.002, np.float32),
        "rot": np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (len(P), 1)),
        "rgba": np.concatenate([np.tile(SKIN / 255.0, (len(P), 1)).astype(np.float32),
                                np.ones((len(P), 1), np.float32)], 1),
        "sh_rest": np.zeros((len(P), 3, 3), np.float32),
    }
    return cloud, f.model.base[:468]


def _lip_layers() -> list[dict]:
    return [{"id": "lip", "region": "lipstick", "enabled": True, "opacity": 0.9,
             "finish": "gloss",
             "color_stops": [{"at": 0.0, "hex": "#C21858"},
                             {"at": 1.0, "hex": "#E91E63"}]}]


@pytest.fixture(scope="module")
def face_canvas():
    """canonical UV 展开成 512² 合成妆照：肤色底 + 红唇（唇心比唇缘暗）。"""
    import cv2
    from makeupstudio.face3dgs.fit_makeup import FaceMakeupFitter
    f = FaceMakeupFitter(tracker_factory=lambda: (_ for _ in ()).throw(RuntimeError))
    size = 512
    px = np.zeros((478, 2), np.float32)
    uv = np.clip(f.model.uvs[:468], 0, 1)
    px[:468, 0] = uv[:, 0] * size
    px[:468, 1] = (1.0 - uv[:, 1]) * size
    img = np.full((size, size, 3), 190, np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = SKIN[2], SKIN[1], SKIN[0]
    lip = px[list(LIPS_OUTER)].astype(np.float32)
    c = lip.mean(0)
    cv2.fillPoly(img, [lip.astype(np.int32)],
                 tuple(int(x) for x in LIP_OUT[::-1]), lineType=cv2.LINE_AA)
    inner = (c + (lip - c) * 0.45).astype(np.int32)
    cv2.fillPoly(img, [inner],
                 tuple(int(x) for x in LIP_IN[::-1]), lineType=cv2.LINE_AA)
    return img, px


def _ref_profile() -> tuple[np.ndarray, np.ndarray]:
    """程序化参考剖面：t=0 唇缘浅红 → t=1 唇心深红（3 档）。"""
    return (np.array([0.0, 0.5, 1.0]),
            np.stack([rgb2lab((LIP_OUT / 255.0)[None, :])[0],
                      rgb2lab((np.array([182.5, 65.0, 65.0]) / 255.0)[None, :])[0],
                      rgb2lab((LIP_IN / 255.0)[None, :])[0]]))


# ---------------- 向心度场 / 剖面提取 ----------------

def test_inness_field_zero_at_edge_one_at_core():
    m = np.zeros((64, 64), np.float32)
    m[8:56, 8:56] = 1.0
    t = inness_field(m)
    assert t[8, 8] == pytest.approx(0.0, abs=0.35)      # 边界
    assert t[32, 32] == pytest.approx(1.0, abs=1e-6)    # 最深处
    assert t[32, 20] < t[32, 32]                        # 单调向内
    assert inness_field(np.zeros((8, 8), np.float32)).max() == 0.0


def test_profile_from_mask_recovers_inner_darker_gradient():
    """剖面必须保住"唇内深、唇缘浅"的空间层次（两档均值会把它抹平）。"""
    rgb = np.tile((LIP_OUT / 255.0).astype(np.float64), (96, 96, 1))
    m = np.zeros((96, 96), np.float32)
    m[8:88, 8:88] = 1.0
    rgb[36:60, 36:60] = LIP_IN / 255.0                  # 核心深红
    prof = profile_from_mask(rgb, m)
    assert prof is not None
    ts, lab = prof
    assert len(ts) >= 3 and np.all(np.diff(ts) > 0)     # t 单调增
    assert lab[-1, 0] < lab[0, 0] - 8                   # 核心明显更暗
    core = lab2rgb(lab[-1][None, :])[0] * 255.0
    assert np.abs(core - LIP_IN).max() < 25             # 核心色≈画的深红
    assert float(lab[:, 0].std()) > 1.0                 # 层次真的被保留


def test_profile_from_mask_rejects_tiny_mask():
    rgb = np.zeros((32, 32, 3), np.float64)
    m = np.zeros((32, 32), np.float32)
    m[10:14, 10:14] = 1.0                               # < MIN_PROFILE_PX
    assert profile_from_mask(rgb, m) is None


def test_extract_profiles_from_reference_image(face_canvas):
    img, px = face_canvas
    profs = extract_profiles(img, ("lipstick", "foundation", "eyeshadow"), px=px)
    assert "lipstick" in profs and "foundation" in profs
    ts, lab = profs["lipstick"]
    assert lab[-1, 0] < lab[0, 0]                       # 唇心比唇缘暗
    # 与两档均值标定对照：剖面档数 > 2（层次没被压平）
    assert len(ts) > 2
    stops = profile_to_stops(profs["lipstick"])
    assert 2 <= len(stops) <= 5
    ats = [s["at"] for s in stops]
    assert ats == sorted(ats) and all(0.0 <= a <= 1.0 for a in ats)
    assert all(s["hex"].startswith("#") and len(s["hex"]) == 7 for s in stops)


def test_region_lab_profile_and_sampling_are_consistent(face_canvas):
    img, px = face_canvas
    prof = region_lab_profile(img, "lipstick", px=px)
    assert prof is not None
    ts, lab = prof
    core = reference_core_lab(prof)
    d = open_loop_delta_e(core, rgb2lab((LIP_IN / 255.0)[None, :])[0])
    assert d < 12                                       # 核心色≈参考深红
    # 端点外取端点值；端点内插值介于两端之间
    edge = sample_profile_lab(prof, np.array([-1.0, 2.0]))
    assert np.abs(edge[0] - lab[0]).max() < 1e-9
    assert np.abs(edge[1] - lab[-1]).max() < 1e-9
    mid = sample_profile_lab(prof, np.array([0.5]))[0]
    assert min(lab[0, 0], lab[-1, 0]) - 1e-6 <= mid[0] <= max(lab[0, 0], lab[-1, 0])


def test_measure_region_lab_matches_drawn_color(face_canvas):
    img, px = face_canvas
    m = measure_region_lab(img[..., ::-1] / 255.0, px, ("lipstick", "foundation"))
    assert "lipstick" in m and "foundation" in m
    assert open_loop_delta_e(m["foundation"],
                             rgb2lab((SKIN / 255.0)[None, :])[0]) < 8
    assert m["lipstick"][0] < m["foundation"][0]        # 红唇比肤色暗


# ---------------- spec 标定：参考优先，两档兜底 ----------------

def test_spec_from_profiles_replaces_stops_only_for_profiles():
    template = {"layers": [
        {"region": "lipstick", "enabled": True, "opacity": 0.6, "finish": "gloss",
         "color_stops": [{"at": 0.0, "hex": "#111111"}, {"at": 1.0, "hex": "#222222"}]},
        {"region": "blush", "enabled": True, "opacity": 0.4,
         "color_stops": [{"at": 0.0, "hex": "#333333"}]}]}
    out = spec_from_profiles(template, {"lipstick": _ref_profile()})
    lip, blush = out["layers"]
    assert len(lip["color_stops"]) == 3                 # 三档剖面
    assert lip["color_stops"][0]["hex"] != "#111111"
    assert blush["color_stops"] == [{"at": 0.0, "hex": "#333333"}]
    assert out["calibration"]["source"] == "colorfield_profile"
    assert template["layers"][0]["color_stops"][0]["hex"] == "#111111"  # 不改输入


def test_calibrate_spec_prefers_profile_falls_back_to_two_stops(face_canvas):
    img, px = face_canvas
    template = {"layers": [
        {"region": "lipstick", "enabled": True, "opacity": 0.5, "finish": "gloss",
         "color_stops": [{"at": 0.0, "hex": "#111111"}, {"at": 1.0, "hex": "#222222"}]}]}
    prof = {"lipstick": _ref_profile()}
    spec = calibrate_spec(template, img, px=px, profiles=prof)
    assert len(spec["layers"][0]["color_stops"]) == 3    # 剖面路径
    assert "lipstick" in spec["calibration"]["profile_regions"]
    # profiles={} → 显式禁用剖面，退回 extract_makeup_colors 的两档均值
    ext = extract_makeup_colors(img, px=px)
    back = calibrate_spec(template, img, px=px, profiles={})
    assert len(back["layers"][0]["color_stops"]) == 2
    assert back["layers"][0]["opacity"] == ext["lipstick"]["opacity"]


# ---------------- 逐 splat 配对 + 最小二乘自求解 ----------------

def _baked(canonical_cloud, tex=256):
    cloud, landmarks = canonical_cloud
    binding = bind_uv(cloud, landmarks, tex=tex)
    baker = UvMakeupBaker(tex=tex)
    maps = baker.bake(_lip_layers(), intensity=1.0)
    return cloud, landmarks, binding, baker, maps


def test_region_cent_uv_from_lip_and_layer_fields(canonical_cloud):
    _c, _l, _b, _bk, maps = _baked(canonical_cloud)
    lip = region_cent_uv(maps, "lipstick")
    assert lip is not None and float(lip.max()) > 0.9
    assert lip.shape == (maps.tex, maps.tex)
    assert region_cent_uv(maps, "blush") is None        # 未上妆区域无中心场
    maps.layer_fields.append({"region": "blush",
                              "w": np.where(lip > 0.5, 0.5, 0.0)})
    assert region_cent_uv(maps, "blush") is not None    # 逐层字段也能定中心


def test_pair_region_samples_uses_per_splat_reference_color(canonical_cloud):
    """唇线 splat 拿唇缘参考色、唇心 splat 拿唇心参考色（不是全区域一个均值）。"""
    cloud, _l, binding, _bk, maps = _baked(canonical_cloud)
    prof = _ref_profile()
    pair = pair_region_samples(cloud, maps, binding.uv, binding.valid,
                               "lipstick", prof)
    assert pair is not None
    cur, tgt, w = pair
    assert len(cur) == len(tgt) == len(w) > 30
    assert np.abs(cur - SKIN / 255.0).max() < 1e-6        # cur = 真实素颜底色（f32）
    assert np.all((tgt >= 0) & (tgt <= 1))
    assert float(tgt[:, 0].std()) > 1e-4                 # 参考色逐 splat 变化
    # 唇心方向（参考更暗）的样本更暗
    assert float(tgt.min()) < float(tgt.max())
    assert pair_region_samples(cloud, maps, binding.uv, binding.valid,
                               "blush", prof) is None    # 无中心场 → 不配对


def test_solve_region_coeffs_beats_manual_coeff_table():
    """同一组 (素颜底色, 参考色) 上，自求解的 ΔE00 必须优于手工表。"""
    rng = np.random.default_rng(3)
    cur = np.clip(SKIN / 255.0 + rng.normal(0, 0.02, (600, 3)), 0, 1)
    tgt = np.clip(LIP_IN / 255.0 + rng.normal(0, 0.015, (600, 3)), 0, 1)
    w = np.ones(600)
    manual = manual_coeff("lipstick")
    rep = compare_coeffs(cur, tgt, w, manual=manual)
    assert rep["gain"] > 0 and rep["delta_e_solved"] < rep["delta_e_manual"]
    assert rep["delta_e_solved"] < 5.0                    # 解到接近参考
    kl, ch = solve_region_coeffs(cur, tgt, w)
    (kl_lo, kl_hi), (ch_lo, ch_hi) = COEFF_BOUNDS
    assert kl_lo <= kl <= kl_hi and ch_lo <= ch <= ch_hi
    # 目标色≈素颜（底妆式低饱和）：不该把 chroma 推到爆
    same = np.tile(SKIN / 255.0, (200, 1))
    kl2, ch2 = solve_region_coeffs(same, same, np.ones(200))
    assert ch2 <= 3.0


def test_manual_coeff_region_upgradable_table(tmp_path):
    table = {"lipstick": (1.25, 1.5), "blush": (0.4, 0.9)}
    p = save_coeff_table(table, tmp_path / "coeffs.json")
    import json
    assert json.loads(p.read_text(encoding="utf-8"))["lipstick"] == [1.25, 1.5]


# ---------------- ΔE00 闭环修正 ----------------

def test_loop_delta_sign_and_cap():
    ref = np.array([50.0, 40.0, 20.0])
    dark = loop_delta(np.array([40.0, 30.0, 15.0]), ref)     # 实测偏暗
    assert dark[0] > 0 and dark[1] > 0                       # 目标往更亮/更艳推
    light = loop_delta(np.array([70.0, 55.0, 30.0]), ref)    # 实测偏亮
    assert light[0] < 0
    huge = loop_delta(np.array([-60.0, -60.0, -60.0]), ref)
    assert np.abs(huge).max() == pytest.approx(LOOP_STEP_MAX)  # 单轮封顶


def test_shift_profile_keeps_gradient_shape_and_clips():
    ts, lab = _ref_profile()
    d = np.array([10.0, 5.0, 5.0])
    ts2, lab2 = shift_profile((ts, lab), d)
    assert np.abs(ts2 - ts).max() < 1e-12
    assert np.abs(np.diff(lab2, axis=0) - np.diff(lab, axis=0)).max() < 1e-9
    _ts, pinned = shift_profile((ts, lab), np.array([200.0, 0.0, 0.0]))
    assert pinned[:, 0].max() <= 100.0                       # L 被夹回合法域


def test_nudge_profiles_moves_target_toward_measured_gap():
    profs = {"lipstick": _ref_profile()}
    ref_core = reference_core_lab(profs["lipstick"])
    measured = {"lipstick": ref_core - np.array([8.0, 4.0, 4.0])}   # 渲染偏暗
    moved = _nudge_profiles(profs, measured, lambda *a: None)
    assert moved and moved[0].startswith("lipstick")
    assert profs["lipstick"][1][:, 0].mean() > _ref_profile()[1][:, 0].mean()
    # 已在噪声量级时不再推
    profs2 = {"lipstick": _ref_profile()}
    assert _nudge_profiles(profs2, {"lipstick": ref_core}, lambda *a: None) == []


# ---------------- 端到端接线：系数不依赖语义分割 ----------------

def test_solve_color_coeffs_writes_fields_and_report(canonical_cloud):
    cloud, _l, binding, _bk, maps = _baked(canonical_cloud)
    table: dict = {}
    zones, rep = _solve_color_coeffs(cloud, maps, binding, None,
                                     {"lipstick": _ref_profile()}, lambda *a: None)
    assert zones is not None and "lipstick" in zones.fields.coeffs
    assert rep["lipstick"]["samples"] > 30
    table.update(rep)
    assert table["lipstick"]["delta_e_solved"] < table["lipstick"]["delta_e_manual"]


def test_coeffs_apply_without_semantic_segmentation(canonical_cloud):
    """只有系数、没有语义分割时，迁移系数仍然覆盖几何兜底（kL/chroma 变）。"""
    cloud, _l, binding, baker, maps = _baked(canonical_cloud)
    geom = baker.assignment(cloud, maps, binding.uv, binding.valid)
    fields = ZoneFields(coeffs={"lipstick": (1.6, 1.4)})
    obs = baker.assignment(cloud, maps, binding.uv, binding.valid, zones=fields)
    zone = np.asarray(geom["w"]) > 0.02
    assert zone.sum() > 30
    assert np.allclose(np.asarray(obs["kL"])[zone], 1.6)
    assert np.allclose(np.asarray(obs["chroma"])[zone], 1.4)
    assert np.allclose(np.asarray(obs["w"]), np.asarray(geom["w"]))  # 形状不动


def test_apply_makeup_solves_coeffs_and_changes_shell(canonical_cloud, tmp_path):
    """产品入口接线：color_profiles → 自求解 → 壳层颜色随系数改变 + 报告留痕。

    参考剖面取"深酒红"（远暗于模板唇色）——自求解必须把 kL/chroma 推离手工表，
    壳层颜色因此可见地变化（证明系数真的进了烘焙路径，而不只是报告里的一行）。"""
    cloud, landmarks = canonical_cloud
    prof = (np.array([0.0, 1.0]),
            np.stack([rgb2lab((np.array([110.0, 40.0, 70.0]) / 255.0)[None, :])[0],
                      rgb2lab((np.array([60.0, 20.0, 45.0]) / 255.0)[None, :])[0]]))
    spec = spec_from_profiles({"layers": _lip_layers()}, {"lipstick": prof})
    bare, _cov = apply_makeup_to_asset(cloud, landmarks, spec, tmp_path / "bare",
                                       tex=256)
    cr: dict = {}
    made, _cov2 = apply_makeup_to_asset(
        cloud, landmarks, spec, tmp_path / "solved", tex=256,
        color_profiles={"lipstick": prof}, color_report=cr)
    assert "lipstick" in cr
    assert cr["lipstick"]["delta_e_solved"] <= cr["lipstick"]["delta_e_manual"]
    assert cr["lipstick"]["solved"] != cr["lipstick"]["manual"]
    n_b = len(cloud["xyz"])
    assert np.abs(made["rgba"][:n_b] - cloud["rgba"]).max() < 1e-6   # 底模不动
    # 壳层数量会随系数改变（妆缘梯度不同 → 2×2 分裂的子 splat 数不同），故按
    # 颜色分布（中位）比较而非逐元素对齐：自求解的唇色必须明显不同于兜底系数
    shell_bare = np.median(bare["rgba"][n_b:, :3], axis=0)
    shell_obs = np.median(made["rgba"][n_b:, :3], axis=0)
    assert np.abs(shell_bare - shell_obs).max() > 0.02