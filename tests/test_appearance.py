"""appearance 包测试：表情选帧 / UV 绑定 / UV 妆容合成 / PBR 材质（合成数据，CPU-only）。

CUDA 相关（train_base/optimize 的训练循环）不在离线测试范围：依赖 GPU 与
gsplat 编译产物，环境不齐时自动跳过。
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

from makeupstudio.face3dgs.appearance import pbr  # noqa: E402
from makeupstudio.face3dgs.appearance.frames import (  # noqa: E402
    expression_features,
    select_frames,
)
from makeupstudio.face3dgs.appearance.makeup_uv import (  # noqa: E402
    FINISH_TARGET,
    UvMakeupBaker,
)
from makeupstudio.face3dgs.appearance.uvbind import bind_uv  # noqa: E402

REFS = ROOT / "makeup-skill" / "references"


# ---------------- frames ----------------

def _synthetic_px(mouth_open_px: float = 6.0, mouth_wide_px: float = 96.0,
                  brow_gap_px: float = 30.0, eye_open_px: float = 12.0) -> np.ndarray:
    """在 (478,2) 网格上摆出用于特征计算的合成 landmarks（其余置 0）。"""
    px = np.zeros((478, 2), np.float64)
    px[33], px[263] = (100.0, 200.0), (196.0, 200.0)          # 眼距 96
    px[13], px[14] = (148.0, 300.0), (148.0, 300.0 + mouth_open_px)
    px[61], px[291] = (100.0, 300.0), (100.0 + mouth_wide_px, 300.0)
    px[105], px[159] = (110.0, 200.0 - brow_gap_px), (110.0, 200.0)
    px[334], px[386] = (186.0, 200.0 - brow_gap_px), (186.0, 200.0)
    px[145], px[374] = (110.0, 200.0 + eye_open_px), (186.0, 200.0 + eye_open_px)
    return px


def test_expression_features_normalized():
    f = expression_features(_synthetic_px())
    assert f.shape == (6,)
    io = 96.0
    assert f[0] == pytest.approx(6.0 / io, abs=1e-6)          # mouth_open
    assert f[1] == pytest.approx(96.0 / io, abs=1e-6)         # mouth_wide
    assert f[2] == pytest.approx(30.0 / io, abs=1e-6)         # brow_gap
    assert f[3] == pytest.approx(12.0 / io, abs=1e-6)         # eye_open
    assert f[4] == 0.0 and f[5] == 0.0                        # 虹膜缺失 → gaze=0


def test_expression_features_gaze_follows_iris():
    """虹膜有效时 gaze 反映视线偏移；虹膜置零（缺失）时回 0。"""
    px = _synthetic_px()
    px[133], px[362] = (124.0, 200.0), (172.0, 200.0)   # 内眼角
    px[468] = (148.0, 200.0)          # 左虹膜中心（居中）→ gaze≈0
    px[473] = (148.0, 200.0)
    f0 = expression_features(px)
    assert f0[4] == pytest.approx(0.0, abs=1e-6)
    px_shift = px.copy()
    px_shift[468] = (158.0, 200.0)    # 虹膜向右移 10px
    px_shift[473] = (158.0, 200.0)
    f1 = expression_features(px_shift)
    assert f1[4] > f0[4] + 0.05       # gaze_x 明显增大


def test_select_frames_picks_dominant_cluster(tmp_path):
    """48 帧里 36 帧同一表情、12 帧夸张张嘴 → 主簇入选，离群被拒。"""
    import cv2
    names = [f"frame_{i:05d}.jpg" for i in range(48)]
    for n in names:                                        # 真写小图（imread 需要文件存在）
        cv2.imwrite(str(tmp_path / n), np.zeros((8, 8, 3), np.uint8))
    feats = {n: expression_features(_synthetic_px()) for n in names}
    for n in names[::4]:                                       # 12 帧张嘴离群
        feats[n] = expression_features(_synthetic_px(mouth_open_px=60.0))
    queue = sorted(names)

    def detect(img, t_ms):
        name = queue.pop(0)
        open_px = 60.0 if name in names[::4] else 6.0
        return {"px": _synthetic_px(mouth_open_px=open_px),
                "pose": [0.0, 0.0, 0.0]}

    sel = select_frames(tmp_path, names, detect, min_frames=24, max_frames=48)
    assert len(sel) == 36                                  # 张嘴帧被 mouth_open_cap 提前滤除
    assert all(n not in names[::4] for n in sel.names)
    assert sel.ref in sel.names
    assert len(sel.rejected) == 0                          # MAD 离群 = 0（张嘴帧不算簇内离群）


# ---------------- uvbind / makeup_uv ----------------

@pytest.fixture(scope="module")
def canonical_cloud():
    """canonical 表面采样点云（自带 3D 一致性，供 UV 绑定）。"""
    from makeupstudio.face3dgs.fit_makeup import FaceMakeupFitter, _sample_tris
    f = FaceMakeupFitter(tracker_factory=lambda: (_ for _ in ()).throw(RuntimeError))
    rng = np.random.default_rng(11)
    P, _b, _rows = _sample_tris(f.model.base, f.model.tris, 6000, rng)
    cloud = {
        "xyz": P.astype(np.float32),
        "scale": np.full((len(P), 3), 0.002, np.float32),
        "rot": np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (len(P), 1)),
        "rgba": np.concatenate([np.full((len(P), 3), 0.6), np.ones((len(P), 1))], 1),
    }
    return cloud, f.model.base[:468]


def test_select_frames_closed_lips_preference(tmp_path):
    """闭嘴 canonical：微张（0.04）与半开（0.10）两簇并存时优先选闭嘴簇，
    不让半开帧进训练集（唇区单模态 = 唇纹清晰、无牙齿烤入）。"""
    import cv2
    names = [f"frame_{i:05d}.jpg" for i in range(40)]
    for n in names:
        cv2.imwrite(str(tmp_path / n), np.zeros((8, 8, 3), np.uint8))
    closed = names[:20]                            # 闭嘴簇（mouth_open=4px）
    queue = sorted(names)

    def detect(img, t_ms):
        name = queue.pop(0)
        open_px = 4.0 if name in closed else 10.0
        return {"px": _synthetic_px(mouth_open_px=open_px),
                "pose": [0.0, 0.0, 0.0]}

    sel = select_frames(tmp_path, names, detect, min_frames=12, max_frames=48)
    assert len(sel) == 20
    assert all(n in closed for n in sel.names)     # 半开簇整簇被拒


def test_select_frames_closed_lips_fallback(tmp_path):
    """闭嘴簇帧数不足 min_frames 时回退中位数带（旧行为），不硬饿死。"""
    import cv2
    names = [f"frame_{i:05d}.jpg" for i in range(40)]
    for n in names:
        cv2.imwrite(str(tmp_path / n), np.zeros((8, 8, 3), np.uint8))
    closed = names[:6]                             # 只有 6 帧闭嘴（< min_frames）
    queue = sorted(names)

    def detect(img, t_ms):
        name = queue.pop(0)
        open_px = 4.0 if name in closed else 10.0
        return {"px": _synthetic_px(mouth_open_px=open_px),
                "pose": [0.0, 0.0, 0.0]}

    sel = select_frames(tmp_path, names, detect, min_frames=24, max_frames=48)
    assert len(sel) >= 24                          # 回退中位数带，帧数保住
    assert sel.ref in sel.names


def test_bind_uv_and_bake(canonical_cloud):
    cloud, landmarks = canonical_cloud
    binding = bind_uv(cloud, landmarks, tex=256)
    assert binding.valid.mean() > 0.9                          # 表面点几乎全部有效
    cov = binding.cov["lipstick"]
    assert cov.shape == (256, 256)
    assert 0.0 <= float(cov.max()) <= 1.0

    baker = UvMakeupBaker(tex=256)
    layers = [
        {"id": "lip", "region": "lipstick", "enabled": True, "opacity": 0.9,
         "finish": "gloss", "color_stops": [{"at": 0.0, "hex": "#C21858"},
                                            {"at": 1.0, "hex": "#E91E63"}]},
        {"id": "blush", "region": "blush", "enabled": True, "opacity": 0.5,
         "finish": "matte", "color_stops": [{"at": 0.0, "hex": "#F48FB1"}]},
        {"id": "off", "region": "eyeshadow", "enabled": False, "opacity": 0.5,
         "finish": "matte", "color_stops": [{"at": 0.0, "hex": "#888888"}]},
    ]
    maps = baker.bake(layers, intensity=1.0)
    assert float(maps.w.max()) > 0.2          # 非唇区域（腮红/底妆）
    assert maps.lip_w is not None and float(maps.lip_w.max()) > 0.3   # 唇妆独立通道
    assert maps.lip_stops is not None
    assert (maps.w > 0.05).mean() < 0.5                        # 妆区只占部分 UV
    # 唇釉清漆/SSS 走 3D(或兜底)覆盖，不再进全局场 → 用 apply 后的材质验证
    # 粉状腮红区域 rough 高于唇釉
    mask_blush = (maps.w > 0.4) & (maps.channels["coat"] < 0.1)
    if mask_blush.any():
        assert maps.channels["rough"][mask_blush].mean() > 0.55

    made = baker.apply_to_cloud(cloud, maps, binding.uv, binding.valid)
    w = made["makeup_w"]
    assert w.shape == (len(cloud["xyz"]),)
    moved = np.abs(made["rgba"][:, :3] - cloud["rgba"][:, :3]).sum(1)
    assert (moved[w > 0.5] > 0.02).mean() > 0.8                # 高覆盖处颜色确实变了
    mat = made["material"]
    assert mat.coat[w > 0.5].mean() > mat.coat[w < 0.05].mean()


def test_lip_band_uv_raster_excludes_mouth():
    """唇带光栅不越内环：口腔开口面（拓扑 mouth）texel 不被涂口红。"""
    baker = UvMakeupBaker(tex=256)
    cov, cent = baker.raster_lip_band()
    assert cov.max() > 0.5
    assert (cov > 0.2).mean() < 0.3                            # 唇是小的


def test_finish_targets_cover_schema():
    for f in ("matte", "satin", "dewy", "gloss"):
        rough, coat, sheen = FINISH_TARGET[f]
        assert 0.03 <= rough <= 1.0 and 0.0 <= coat <= 1.0 and 0.0 <= sheen <= 1.0
    assert FINISH_TARGET["gloss"][1] > FINISH_TARGET["matte"][1]


# ---------------- pbr ----------------

def test_pbr_shade_ranges_and_gloss_flow():
    n = 512
    rng = np.random.default_rng(5)
    rgba = np.concatenate([rng.random((n, 3)) * 0.5 + 0.3, np.ones((n, 1))], 1)
    normal = rng.normal(size=(n, 3))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    mat = pbr.Material(np.full(n, 0.2, np.float32), np.linspace(0, 1, n).astype(np.float32),
                       np.zeros(n, np.float32), np.full(n, 0.1, np.float32))
    light = np.array([0.3, 0.4, 1.0])
    view = np.array([0.0, 0.0, 1.0])
    out = pbr.shade_points(rgba, normal, view, light, np.ones(3), mat)
    assert out.shape == (n, 3)
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0
    # 正对半向量的点获得更高光：法线 = normalize(L+V) 处 coat=1 应最亮
    H = light / np.linalg.norm(light) + view
    H /= np.linalg.norm(H)
    i = int(np.argmax((normal @ H)))
    j = int(np.argmin((normal @ H)))
    assert out[i].mean() >= out[j].mean()


def test_rough_to_shin_monotonic():
    r = np.array([0.05, 0.2, 0.5, 1.0])
    s = pbr.rough_to_shin(r)
    assert (np.diff(s) < 0).all()                              # 越粗糙指数越低


def test_mat_from_cloud_legacy_compat():
    from makeupstudio.face3dgs.appearance.render_pbr import _mat_from_cloud
    n = 4
    cloud = {"xyz": np.zeros((n, 3)), "gloss": np.array([0.0, 0.5, 1.0, 0.2], np.float32),
             "shin": np.full(n, 60.0, np.float32)}
    mat = _mat_from_cloud(cloud)
    assert mat.coat[2] > mat.coat[0]                           # gloss 高 → coat 高


# ---------------- CUDA 链路（可选） ----------------

def test_train_base_smoke_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("无 CUDA 设备")
    gsplat = pytest.importorskip("gsplat")
    from gsplat import rasterization  # noqa: F401
    n = 2000
    means = torch.randn(n, 3, device="cuda") * 0.3
    means[:, 2] += 5
    quats = torch.randn(n, 4, device="cuda")
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales = torch.rand(n, 3, device="cuda") * 0.05 + 0.01
    colors = torch.rand(n, 3, device="cuda")
    c2w = torch.eye(4, device="cuda")[None]
    c2w[0, 2, 3] = -5.0
    K = torch.tensor([[[300.0, 0, 240], [0, 300.0, 240], [0, 0, 1]]], device="cuda")
    out, _a, _i = rasterization(means, quats, scales, torch.full((n,), 0.5, device="cuda"),
                                colors[None], c2w, K, 480, 480)
    assert torch.isfinite(out).all()
