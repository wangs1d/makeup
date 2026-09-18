"""写实链路语义层测试：UV 归属鲁棒化、离群剔除、唇拓扑、3D 唇带权重
（张嘴口腔排除/颜色门控）、guidance 环境门控。全部合成数据，
不依赖 GPU / mediapipe / Stable-Makeup 环境。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "desktop-app"
sys.path.insert(0, str(APP))

from makeupstudio.face3dgs.fit_makeup import FaceMakeupFitter  # noqa: E402
from makeupstudio.face3dgs import guidance  # noqa: E402


@pytest.fixture(scope="module")
def fitter():
    return FaceMakeupFitter()


def _user_cloud(fitter, n=1200, jitter=0.004, seed=9):
    """canonical 表面 + 已知相似变换的合成"用户脸"（与既有 fit 测试同思路）。"""
    rng = np.random.default_rng(seed)
    ang = 0.3
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0],
                  [-np.sin(ang), 0, np.cos(ang)]])
    s, t = 1.4, np.array([0.3, -0.1, 2.0])
    V, tris = fitter.model.base, fitter.model.tris
    area = 0.5 * np.linalg.norm(np.cross(V[tris[:, 1]] - V[tris[:, 0]],
                                         V[tris[:, 2]] - V[tris[:, 0]]), axis=1)
    rows = rng.choice(len(tris), size=n, p=area / area.sum())
    b = rng.dirichlet([1, 1, 1], size=n)
    P = (V[tris[rows, 0]] * b[:, :1] + V[tris[rows, 1]] * b[:, 1:2]
         + V[tris[rows, 2]] * b[:, 2:]) + rng.normal(0, jitter, (n, 3))
    xyz = (P @ (s * R).T) + t
    cloud = {"xyz": xyz.astype(np.float32),
             "scale": np.full((n, 3), 0.01, np.float32),
             "rot": np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32),
             "rgba": np.full((n, 4), 0.75, np.float32)}
    landmarks = V[:468] @ (s * R).T + t
    return cloud, landmarks


def _with_hair(cloud, s=1.4, t=np.array([0.3, -0.1, 2.0]), n=400, seed=5):
    """在脸后方加一圈深色"头发壳" splat（距 canonical 表面极远）。"""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    ph = rng.uniform(-0.4, 1.2, n)
    r = 0.42 + rng.normal(0, 0.02, n)
    shell = np.stack([r * np.sin(ph) * np.cos(th), r * np.cos(ph) * 1.1,
                      -r * np.sin(ph) * 0.7 - 0.1], axis=1)
    xyz = (shell @ (s * np.eye(3)).T) + t
    return {"xyz": np.concatenate([cloud["xyz"], xyz.astype(np.float32)]),
            "scale": np.concatenate([cloud["scale"],
                                     np.full((n, 3), 0.012, np.float32)]),
            "rot": np.concatenate([cloud["rot"],
                                   np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))]).astype(np.float32),
            "rgba": np.concatenate([cloud["rgba"],
                                    np.full((n, 4), 0.2, np.float32)])}


# ---------------- 离群剔除（kNN UV 归属） ----------------

def test_outlier_hair_never_painted(fitter):
    """头发/背景壳 splat 距 canonical 表面超阈值 → valid=False，永不被涂妆。"""
    from makeupstudio.face3dgs.appearance.uvbind import bind_uv
    cloud, landmarks = _user_cloud(fitter)
    full = _with_hair(cloud)
    binding = bind_uv(full, landmarks, tex=256)
    n_face = len(cloud["xyz"])
    assert binding.valid[:n_face].mean() > 0.9        # 脸部几乎全部有效
    assert binding.valid[n_face:].mean() < 0.05       # 头发壳判为离群


# ---------------- guidance 环境门控（Stable-Makeup 适配器） ----------------

def test_guidance_gate_reports_missing_env(monkeypatch):
    monkeypatch.setattr(guidance, "REPO_DIR", Path("/nonexistent/Stable-Makeup"))
    st = guidance.status()
    assert not st.ok and "git clone" in st.missing_hint
    with pytest.raises(RuntimeError, match="环境不完整"):
        guidance.generate("a.png", "b.png", "c.png")


# ---------------- L0：3D 唇红带权重（张嘴口红不再糊满嘴） ----------------

def _tri_centroids(V, tris, jit=0.0005, seed=5):
    rng = np.random.default_rng(seed)
    P = (V[tris[:, 0]] + V[tris[:, 1]] + V[tris[:, 2]]) / 3.0
    return P + rng.normal(0, jit, P.shape)


def test_lip_topology_band_and_mouth_disjoint(fitter):
    topo = fitter.lip_topology()
    assert len(topo["band"]) >= 20 and len(topo["mouth"]) >= 5
    assert not (set(topo["band"].tolist()) & set(topo["mouth"].tolist()))
    # 唇红带三角形顶点全部落在唇环附近（拓扑上不可能越唇缘一步）
    V = fitter.model.base
    band_tris = fitter.model.tris[topo["band"]]
    do = np.linalg.norm(V[:, None] - V[topo["outer"]][None], axis=2).min(1)
    di = np.linalg.norm(V[:, None] - V[topo["inner"]][None], axis=2).min(1)
    for tri in band_tris:
        assert all(min(do[v], di[v]) < 0.03 for v in tri)


def test_lip_weight_zero_on_open_mouth_surfaces(fitter):
    """张嘴 landmark 网格：口腔开口面上的 splat 唇带权重必须为 0（mouth tag 路径）。"""
    V = fitter.model.pose_explicit(0, 0, 0, mouth_k=1.0, smile_k=0.0)
    topo = fitter.lip_topology()
    tris = fitter.model.tris

    def cloud_of(P):
        n = len(P)
        return {"xyz": P.astype(np.float32),
                "scale": np.full((n, 3), 0.004, np.float32),
                "rot": np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32),
                "rgba": np.full((n, 4), 0.6, np.float32)}

    mouth_pts = _tri_centroids(V, tris[topo["mouth"]])
    band_pts = _tri_centroids(V, tris[topo["band"]])
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    w_m, _ = fitter._lip_band_weight(cloud_of(mouth_pts), V[:468], s, R, t,
                                     mouth=np.ones(len(mouth_pts), bool))
    assert float(w_m.max()) == 0.0                                   # 口腔零权重
    w_b, _ = fitter._lip_band_weight(cloud_of(band_pts), V[:468], s, R, t,
                                     mouth=np.zeros(len(band_pts), bool))
    assert (w_b > 0.3).mean() > 0.3                                  # 唇红带高权重


def test_lip_weight_generic_path_excludes_teeth_behind_strip(fitter):
    """通用路径（无 uv/无 tag）：口腔线条面法向背面的"牙齿" splat 被侧别测试排除。"""
    V = fitter.model.pose_explicit(0, 0, 0, mouth_k=1.0, smile_k=0.0)
    topo = fitter.lip_topology()
    tris = fitter.model.tris
    mouth_pts = _tri_centroids(V, tris[topo["mouth"]], seed=7)
    band_pts = _tri_centroids(V, tris[topo["band"]], seed=8)
    # 牙齿样点：沿口腔面条法线反方向（进入口腔）推入口腔内（0.05 脸高，
    # 明确越过 0.5τ 侧别阈值——真实数据里的唇后牙齿/暗腔在更深处）
    p0, p1, p2 = V[tris[topo["mouth"]][:, 0]], V[tris[topo["mouth"]][:, 1]], \
        V[tris[topo["mouth"]][:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    teeth = mouth_pts - fn * 0.05
    P = np.concatenate([band_pts, teeth]).astype(np.float32)
    n = len(P)
    cloud = {"xyz": P,
             "scale": np.full((n, 3), 0.004, np.float32),
             "rot": np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32),
             "rgba": np.full((n, 4), 0.6, np.float32)}
    cloud["uv"] = np.zeros((n, 2))     # 触发地标路径：Vw = 真实张嘴地标（与 splat 同几何）
    w, _ = fitter._lip_band_weight(cloud, V[:468], 1.0, np.eye(3), np.zeros(3))
    assert (w[:len(band_pts)] > 0.3).mean() > 0.3                    # 唇红带高权重
    assert (w[len(band_pts):] > 0.1).mean() < 0.2                    # 牙齿几乎零权重
