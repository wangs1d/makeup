"""test_commercial_delivery — 商业化交付升级离线单测。

覆盖：资产质量分级与门禁（quality）、还原度验收门与自动重标定（calibrate.
fidelity_gate/boost_spec_regions）、背景模板与 alpha 合成（offline_render.
_make_background/_resolve_background/straight_rgba）、环境光 preset
（preset_light_world）、H.264 交付编码回退链（_open_video_writer）、
妆效 preset 库完整性。全部 CPU/numpy 可跑，不依赖 gsplat/CUDA。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parent.parent / "desktop-app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


# ---------------- 质量分级与门禁 ----------------

def test_quality_grades_by_resolution_and_psnr():
    from makeupstudio.face3dgs.appearance.quality import assess_quality

    a = assess_quality((1920, 1080), 25.0, 230_000)
    assert a.grade == "A" and a.min_side == 1080 and not a.reasons

    b = assess_quality((1280, 720), 22.0, 150_000)
    assert b.grade == "B" and not b.reasons

    c = assess_quality((640, 536), 24.0, 200_000)
    assert c.grade == "C"
    assert any("短边" in r for r in c.reasons)

    # PSNR 退化压到 C，即使分辨率够
    d = assess_quality((1920, 1080), 12.0, 200_000)
    assert d.grade == "C" and any("PSNR" in r for r in d.reasons)

    # PSNR 未知（旧资产）不降级；splat 过少单独记原因
    e = assess_quality((1920, 1080), None, 25_000)
    assert e.grade == "A" and any("splat" in r for r in e.reasons)


def test_quality_grade_roundtrip(tmp_path):
    from makeupstudio.face3dgs.appearance.quality import load_grade

    assert load_grade(tmp_path) is None          # 无 report.json
    (tmp_path / "report.json").write_text(json.dumps(
        {"quality": {"grade": "C", "min_side": 536, "psnr": 24.0,
                     "splats": 200_000, "reasons": []}}), encoding="utf-8")
    assert load_grade(tmp_path) == "C"
    (tmp_path / "report.json").write_text("{broken", encoding="utf-8")
    assert load_grade(tmp_path) is None


def test_deliver_gate_blocks_grade_c(tmp_path):
    """C 级资产默认拒绝交付；force 越过后在 CUDA/gsplat 缺失处失败（而非门禁处）。"""
    from makeupstudio.face3dgs.appearance.offline_render import deliver

    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "report.json").write_text(json.dumps(
        {"quality": {"grade": "C", "min_side": 536, "psnr": 24.0,
                     "splats": 200_000, "reasons": ["低清源"]}}), encoding="utf-8")
    ply = asset / "madeup.ply"
    ply.write_bytes(b"ply")
    with pytest.raises(RuntimeError, match="质量分级 C"):
        deliver(ply, tmp_path / "out")
    with pytest.raises((RuntimeError, ValueError)) as ei:
        deliver(ply, tmp_path / "out", force=True)
    assert "质量分级 C" not in str(ei.value)      # 越过门禁后才撞上 CUDA/资产错误


# ---------------- 还原度验收门 ----------------

def test_fidelity_gate_thresholds():
    from makeupstudio.face3dgs.appearance.calibrate import (
        FIDELITY_BUDGET, fidelity_gate)

    assert fidelity_gate({})["status"] == "passed"          # 无度量 → 放行
    assert fidelity_gate({"_mean": 99.0})["status"] == "passed"   # 汇总键忽略
    ok = fidelity_gate({"lipstick": FIDELITY_BUDGET["lipstick"] - 0.1,
                        "eyeshadow": 10.0})
    assert ok["status"] == "passed"
    over = fidelity_gate({"lipstick": FIDELITY_BUDGET["lipstick"] + 0.1,
                          "blush": 99.0})
    assert over["status"] == "over" and over["over"] == ["lipstick", "blush"]


def test_exclude_region_overlap_keeps_features_out_of_foundation():
    from makeupstudio.face3dgs.appearance.calibrate import exclude_region_overlap

    t = 16
    foundation = np.zeros((t, t), np.float32); foundation[:, :] = 0.9
    lip = np.zeros((t, t), np.float32); lip[2:5, 2:8] = 1.0
    eye = np.zeros((t, t), np.float32); eye[8:10, 2:6] = 0.8
    masks = {"foundation": foundation, "lipstick": lip,
             "eyeshadow": eye, "blush": np.full((t, t), 0.5, np.float32)}
    out = exclude_region_overlap(masks)
    # 底妆区扣除唇/眼特征；腮红不在排除列表，保持不变
    assert out["foundation"][3, 4] == 0.0            # 唇区像素被扣除
    assert out["foundation"][9, 3] == pytest.approx(0.1)  # 0.9-0.8
    assert out["foundation"][12, 12] == pytest.approx(0.9)
    assert np.allclose(out["blush"], 0.5) and np.allclose(out["lipstick"], lip)
    assert np.allclose(foundation[:, :], 0.9)        # 输入不被改
    # 无 foundation 键时原样透传
    assert "lipstick" in exclude_region_overlap({"lipstick": lip})


def test_boost_spec_regions_scales_opacity_only():
    from makeupstudio.face3dgs.appearance.calibrate import boost_spec_regions

    spec = {"layers": [
        {"region": "lipstick", "opacity": 0.95,
         "color_stops": [{"at": 0.0, "hex": "#C4788A"}]},
        {"region": "blush", "opacity": 0.5,
         "color_stops": [{"at": 0.0, "hex": "#DFA0A8"}]},
        {"region": "lipstick", "opacity": 0.4,
         "color_stops": [{"at": 0.0, "hex": "#111111"}]},
    ]}
    out = boost_spec_regions(spec, ["lipstick"], factor=1.25)
    assert out["layers"][0]["opacity"] == 1.0               # 封顶
    assert out["layers"][2]["opacity"] == 0.5               # 0.4×1.25
    assert out["layers"][1]["opacity"] == 0.5               # 未选区域不动
    # 只动浓度不动颜色
    assert out["layers"][0]["color_stops"] == spec["layers"][0]["color_stops"]
    assert spec["layers"][0]["opacity"] == 0.95             # 原spec不被改


# ---------------- 背景模板与 alpha 合成 ----------------

def test_make_background_presets():
    from makeupstudio.face3dgs.appearance.offline_render import (
        BG_PRESETS, _make_background)

    for name in BG_PRESETS:
        bg = _make_background(name, 32)
        assert bg.shape == (32, 32, 3) and bg.dtype == np.float32
        assert bg.min() >= 0.0 and bg.max() <= 1.0 + 1e-6
    white = _make_background("white", 16)
    assert np.allclose(white, 1.0)
    grad = _make_background("studio", 64)
    assert grad[0, 0, 0] > grad[-1, 0, 0]                   # 顶亮底暗渐变
    assert np.allclose(grad[0], grad[0, 0])                 # 每行恒定
    with pytest.raises(ValueError):
        _make_background("nope", 16)


def test_resolve_background_tuple_vs_preset():
    from makeupstudio.face3dgs.appearance.offline_render import (
        _make_background, _resolve_background)

    col, arr = _resolve_background((1.0, 0.5, 0.0), 16)
    assert col == (1.0, 0.5, 0.0) and arr is None           # 常量走 GPU 混合
    col, arr = _resolve_background("warm", 16)
    assert col == (0.0, 0.0, 0.0) and np.allclose(arr, _make_background("warm", 16))
    col, arr = _resolve_background("transparent", 16)
    assert col == (0.0, 0.0, 0.0) and arr is None           # 不复合
    with pytest.raises(ValueError):
        _resolve_background(3.14, 16)


def test_straight_rgba_roundtrip():
    from makeupstudio.face3dgs.appearance.offline_render import straight_rgba

    rng = np.random.default_rng(0)
    color = rng.random((8, 8, 3))
    alpha = np.full((8, 8), 0.25)
    premul = color * alpha[..., None]
    rgba = straight_rgba(premul, alpha)
    assert rgba.shape == (8, 8, 4)
    assert np.allclose(rgba[..., :3], color, atol=1e-9)     # 颜色还原
    assert np.allclose(rgba[..., 3], alpha)
    zero = straight_rgba(np.zeros((4, 4, 3)), np.zeros((4, 4)))
    assert zero.max() <= 1.0 and np.all(zero[..., :3] == 0)  # 全透明不除零


# ---------------- 环境光 preset ----------------

def test_preset_light_world_axes():
    from makeupstudio.face3dgs.appearance.offline_render import (
        LIGHT_PRESETS, preset_light_world)

    up, front = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    assert preset_light_world("capture", up, front) is None  # 采集光不覆盖
    assert preset_light_world(None, up, front) is None
    with pytest.raises(ValueError):
        preset_light_world("sunset", up, front)

    d, s, t = preset_light_world("studio", up, front)
    p = LIGHT_PRESETS["studio"]
    assert abs(np.linalg.norm(d) - 1.0) < 1e-9 and 0.0 < s <= 1.2
    assert np.allclose(d, np.asarray(p["dir"], float) /
                       np.linalg.norm(p["dir"]), atol=1e-9)  # 正交基下 = 原方向
    assert np.allclose(t, p["tint"])

    # 旋转资产（front 转向 x）：光方向应随脸转，且保持在脸前方一侧
    d2, _s2, _t2 = preset_light_world("studio", up, np.array([1.0, 0.0, 0.0]))
    assert d2[0] > 0.5                                      # 前向分量跟随 front
    assert abs(np.linalg.norm(d2) - 1.0) < 1e-9


# ---------------- H.264 交付编码回退链 ----------------

def test_orbit_poses_preserves_real_intrinsics():
    """cam_size 给定时输出 K = 真实 K 等比缩放（主点偏移保留，不被强挪到画心）。"""
    from makeupstudio.face3dgs.appearance.offline_render import orbit_from_poses

    center = np.zeros(3)
    w2cs, Ks = [], []
    for yaw in (-30.0, 0.0, 30.0):
        a = np.radians(yaw)
        eye = np.array([np.sin(a) * 5.0, 0.0, np.cos(a) * 5.0])
        z = -eye / np.linalg.norm(eye)                   # look at origin
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        w2c = np.eye(4)
        w2c[:3, :3] = np.stack([x, y, z])
        w2c[:3, 3] = -w2c[:3, :3] @ eye
        w2cs.append(w2c)
        Ks.append(np.array([[800.0, 0, 320.0], [0, 800.0, 268.0],
                            [0, 0, 1]]))                 # 640×536 采集内参
    cams = orbit_from_poses(w2cs, Ks, center, n=5, size=1080,
                            cam_size=(640, 536))
    assert len(cams) == 5
    K0 = cams[0][1]
    s = 1080 / 640.0
    assert K0[0, 0] == pytest.approx(800.0 * s)          # 焦距等比
    assert K0[0, 2] == pytest.approx(320.0 * s)          # 主点偏移保留
    assert K0[1, 2] == pytest.approx(268.0 * s)
    # 不传 cam_size：保持旧行为（主点=画布中心）
    cams_legacy = orbit_from_poses(w2cs, Ks, center, n=5, size=1080)
    assert cams_legacy[0][1][0, 2] == pytest.approx(540.0)


def test_prune_off_surface_removes_floaters():
    from makeupstudio.face3dgs.appearance.pipeline import prune_off_surface

    rng = np.random.default_rng(0)
    lm = np.concatenate([rng.random((468, 2)), np.full((468, 1), 3.0)], 1)
    surf = lm + rng.normal(0, 0.01, (468, 3))            # 脸表面 splat
    fl = np.concatenate([rng.random((50, 2)), np.full((50, 1), 1.5)], 1)
    xyz = np.concatenate([surf, fl]).astype(np.float32)
    cloud = {"xyz": xyz, "rgba": np.ones((len(xyz), 4), np.float32)}
    out = prune_off_surface(cloud, lm)
    assert len(out["xyz"]) == 468                        # 贴脸漂浮物全灭
    assert np.allclose(np.sort(out["xyz"], 0), np.sort(surf, 0), atol=0.05)
    lm0 = np.zeros((468, 3))                             # 地标无效 → 原样返回
    assert len(prune_off_surface(cloud, lm0)["xyz"]) == len(xyz)


def test_train_config_mask_shape_oval_wider_than_face_contour():
    """oval 蒙版跟随脸型轮廓（含凹陷），构建不炸即可；hull 仍为默认。"""
    from makeupstudio.face3dgs.appearance.train_base import TrainConfig

    assert TrainConfig().mask_shape == "hull"
    cfg = TrainConfig(mask_shape="oval", hull_margin=1.06)
    assert cfg.mask_shape == "oval" and cfg.hull_margin == 1.06


def test_open_video_writer_produces_playable_file(tmp_path):
    """无 ffmpeg 时回退 cv2 fourcc（mp4v），文件可写可读；有 ffmpeg 时 H.264。"""
    import cv2

    from makeupstudio.face3dgs.appearance.offline_render import _open_video_writer

    mp4 = tmp_path / "t.mp4"
    vw, codec, warn = _open_video_writer(mp4, 64, 10)
    frame = np.full((64, 64, 3), 128, np.uint8)
    for _ in range(6):
        vw.write(frame)
    vw.release()
    assert mp4.exists() and mp4.stat().st_size > 0
    if codec == "mp4v":
        assert warn is not None                             # 兼容性告警必须给出
    back = cv2.VideoCapture(str(mp4))
    assert back.isOpened() and back.get(cv2.CAP_PROP_FRAME_COUNT) >= 4
    back.release()


# ---------------- 妆效 preset 库完整性 ----------------

def test_preset_library_complete_and_distinct():
    import hashlib

    preset_dir = Path(__file__).resolve().parent.parent / "makeup-skill" / "presets"
    files = sorted(preset_dir.glob("*.json"))
    assert len(files) >= 6, "妆效库应 ≥6 个 preset"
    seen_colors = {}
    for f in files:
        spec = json.loads(f.read_text(encoding="utf-8"))
        assert spec["spec_version"] == "1.0" and spec["name"] == f.stem
        assert spec["layers"] and isinstance(spec.get("steps", []), list)
        regions = {l["region"] for l in spec["layers"]}
        assert "lipstick" in regions and "foundation" in regions
        for l in spec["layers"]:
            assert 0.0 < float(l["opacity"]) <= 1.0
            assert l["color_stops"], f"{f.stem}: {l['region']} 缺色带"
        # 以唇妆色带哈希保证 preset 之间色系确实不同
        lip = next(l for l in spec["layers"] if l["region"] == "lipstick")
        h = hashlib.md5(json.dumps(
            [s["hex"] for s in lip["color_stops"]]).encode()).hexdigest()
        assert h not in seen_colors, f"唇色与 {seen_colors.get(h)} 重复"
        seen_colors[h] = f.stem
