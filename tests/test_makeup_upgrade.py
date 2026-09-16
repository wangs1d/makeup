"""P0/P1/P2 妆容管线升级的测试：UV 归属鲁棒化、离群剔除、密度增强、
Blinn-Phong 镜面、guidance 多视角投影采样、canonical 入口（FLAME 底座接缝）。
全部合成数据，不依赖 GPU / mediapipe / Stable-Makeup 环境。"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "desktop-app"
sys.path.insert(0, str(APP))

_spec = importlib.util.spec_from_file_location(
    "preview_render_core", ROOT / "makeup-skill" / "scripts" / "preview_render.py")
prc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prc)

from makeupstudio.face3dgs.fit_makeup import (FaceMakeupFitter, blinn_phong_spec,  # noqa: E402
                                              render_cloud)
from makeupstudio.face3dgs import guidance  # noqa: E402
from makeupstudio.face3dgs.splat_io import export_splat  # noqa: E402


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


# ---------------- P0：离群剔除 + kNN UV ----------------

def test_outlier_hair_never_painted(fitter):
    cloud, landmarks = _user_cloud(fitter)
    full = _with_hair(cloud)
    presets = json.loads((ROOT / "makeup-skill" / "presets" / "date-rose.json")
                         .read_text(encoding="utf-8"))
    made = fitter.apply_makeup(full, presets["layers"], landmarks, intensity=0.8)
    n_face = len(cloud["xyz"])
    assert len(made["xyz"]) == len(full["xyz"])          # 保数量（densify 默认关）
    assert not np.allclose(made["rgba"][:n_face], full["rgba"][:n_face])   # 脸变了
    assert np.allclose(made["rgba"][n_face:], full["rgba"][n_face:])       # 头发没被涂


def test_gloss_baked_into_cloud(fitter):
    cloud, landmarks = _user_cloud(fitter, n=800)
    lip = {"id": "lip", "region": "lipstick", "enabled": True,
           "color_stops": [{"at": 0.0, "hex": "#B03040"}, {"at": 1.0, "hex": "#D04858"}],
           "opacity": 0.9, "finish": "gloss"}
    made = fitter.apply_makeup(cloud, [lip], landmarks, intensity=0.9)
    assert "gloss" in made and "shin" in made
    assert float(made["gloss"].max()) > 0.3              # gloss finish 烘出了镜面强度
    assert made["gloss"].shape[0] == len(made["xyz"])


# ---------------- P0：密度增强 ----------------

def test_densify_adds_small_sigma_clones(fitter, tmp_path):
    cloud, landmarks = _user_cloud(fitter, n=1500)
    lip = {"id": "lip", "region": "lipstick", "enabled": True,
           "color_stops": [{"at": 0.0, "hex": "#B03040"}, {"at": 1.0, "hex": "#D04858"}],
           "opacity": 0.9, "finish": "satin"}
    plain = fitter.apply_makeup(cloud, [lip], landmarks)
    dense = fitter.apply_makeup(cloud, [lip], landmarks, densify=True)
    assert len(dense["xyz"]) > len(plain["xyz"])         # 克隆发生了
    assert len(dense["xyz"]) <= int(len(cloud["xyz"]) * 1.35) + 1
    tail = len(plain["xyz"])
    assert np.all(dense["scale"][tail:, 0] <= plain["scale"].max() * 0.56)   # 子高斯更小
    export_splat(dense, tmp_path / "dense.splat")        # 32B 定长导出不炸
    assert (tmp_path / "dense.splat").stat().st_size == 32 * len(dense["xyz"])


# ---------------- P0：Blinn-Phong 镜面 ----------------

def test_blinn_phong_spec_view_dependent():
    n = 6
    cloud = {"xyz": np.zeros((n, 3), np.float32),
             "rot": np.tile([0.0, 1.0, 0.0, 0.0], (n, 1)).astype(np.float32),
             "gloss": np.full(n, 0.9, np.float32),
             "shin": np.full(n, 8.0, np.float32)}       # 宽高光；rot=(0,1,0,0)→法线(0,0,-1)朝相机
    eye = np.eye(3, dtype=np.float64)
    head = np.array([0.0, 0.0, -1.0])                    # 顺光头灯（世界系）
    s0 = blinn_phong_spec(cloud, eye, light_dir=head)
    assert s0.shape == (n, 3) and float(np.abs(s0).sum()) > 0.05
    # 法线随相机转 86°：高光应明显变化（视角相关性）
    c, si = np.cos(1.5), np.sin(1.5)
    Ry = np.array([[c, 0, si], [0, 1, 0], [-si, 0, c]])
    s1 = blinn_phong_spec(cloud, Ry, light_dir=head)
    assert np.abs(s1 - s0).max() > 0.01
    # 素颜云（无 gloss 键）恒为 0
    bare = {"xyz": np.zeros((n, 3), np.float32),
            "rot": cloud["rot"].copy()}
    assert float(np.abs(blinn_phong_spec(bare, eye)).sum()) == 0.0


def test_render_cloud_adds_spec_highlights(fitter):
    cloud, landmarks = _user_cloud(fitter, n=900)
    lip = {"id": "lip", "region": "lipstick", "enabled": True,
           "color_stops": [{"at": 0.0, "hex": "#B03040"}, {"at": 1.0, "hex": "#D04858"}],
           "opacity": 0.95, "finish": "gloss"}
    made = fitter.apply_makeup(cloud, [lip], landmarks, intensity=0.95)
    made["rot"] = np.tile([0.0, 1.0, 0.0, 0.0], (len(made["xyz"]), 1)).astype(np.float32)
    made["gloss"] = np.full(len(made["xyz"]), 0.9, np.float32)
    made["shin"] = np.full(len(made["xyz"]), 8.0, np.float32)   # 宽高光，保证像素可见

    class _Cam:
        cam_id, width, height = 1, 640, 480
        params = np.array([500.0, 320.0, 240.0])

    R, t = np.eye(3), np.zeros(3)
    img_spec = render_cloud(made, R, t, _Cam, w=160, h=120,
                            light_dir=np.array([0.0, 0.0, -1.0]))
    no_spec = {k: v for k, v in made.items() if k not in ("gloss", "shin")}
    img_plain = render_cloud(no_spec, R, t, _Cam, w=160, h=120,
                             light_dir=np.array([0.0, 0.0, -1.0]))
    assert img_spec.shape == (120, 160, 3)
    assert np.abs(img_spec.astype(int) - img_plain.astype(int)).max() >= 1   # 高光可见


# ---------------- P1：guidance 多视角投影采样 ----------------

def test_apply_guidance_samples_reference_colors(fitter):
    cloud, landmarks = _user_cloud(fitter, n=900)
    lip = {"id": "lip", "region": "lipstick", "enabled": True,
           "color_stops": [{"at": 0.0, "hex": "#B03040"}, {"at": 1.0, "hex": "#D04858"}],
           "opacity": 0.9, "finish": "gloss"}

    class _Cam:
        cam_id, width, height = 1, 320, 240
        params = np.array([250.0, 160.0, 120.0])

    # guidance 图：绿色。（唇区 splat 投影落在图内 → 被染绿；离群/图外不染）
    img = np.full((240, 320, 3), (40, 180, 60), np.uint8)     # BGR 绿
    views = [{"R": np.eye(3), "t": np.zeros(3), "cam": _Cam, "img": img}
             for _ in range(3)]
    made = fitter.apply_guidance(cloud, [lip], views, landmarks, intensity=0.9)
    g = made["rgba"][:, 1] - made["rgba"][:, 0]
    painted = g > 0.15
    assert 0 < painted.sum() < len(painted) * 0.6             # 部分 splat 吃到绿色
    # 无 valid 视角时回退不染色
    far_views = [{"R": np.eye(3), "t": np.array([50.0, 0, 0]), "cam": _Cam, "img": img}]
    made2 = fitter.apply_guidance(cloud, [lip], far_views, landmarks, intensity=0.9)
    assert np.allclose(made2["rgba"], cloud["rgba"], atol=0.05)


def test_guidance_gate_reports_missing_env(monkeypatch):
    monkeypatch.setattr(guidance, "REPO_DIR", Path("/nonexistent/Stable-Makeup"))
    st = guidance.status()
    assert not st.ok and "git clone" in st.missing_hint
    with pytest.raises(RuntimeError, match="环境不完整"):
        guidance.generate("a.png", "b.png", "c.png")


# ---------------- P2：canonical 入口（FLAME 底座接缝） ----------------

def test_fit_canonical_identity_landmarks(fitter, tmp_path):
    """FlashAvatar 导出的 canonical 点云：landmarks = model.base[:468]，零配准直接上妆。"""
    rng = np.random.default_rng(3)
    V, tris = fitter.model.base, fitter.model.tris
    rows = rng.choice(len(tris), size=900)
    P = V[tris[rows, 0]]
    cloud = {"xyz": P.astype(np.float32),
             "scale": np.full((900, 3), 0.008, np.float32),
             "rot": np.tile([0.0, 0.0, 0.0, 1.0], (900, 1)).astype(np.float32),
             "rgba": np.full((900, 4), 0.75, np.float32)}
    presets = json.loads((ROOT / "makeup-skill" / "presets" / "daily-natural.json")
                         .read_text(encoding="utf-8"))
    made = fitter.fit_canonical(cloud, presets, tmp_path, intensity=0.8)
    assert (tmp_path / "madeup.ply").exists()
    report = json.loads((tmp_path / "fit_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "canonical" and report["scale"] == 1.0
    assert len(made["xyz"]) >= len(cloud["xyz"])              # 妆容壳层已并入
    assert not np.allclose(made["rgba"][:len(cloud["xyz"])], cloud["rgba"])


# ---------------- L0：3D 唇红带权重（张嘴口红不再糊满嘴） ----------------

LIP = {"id": "lip", "region": "lipstick", "enabled": True,
       "color_stops": [{"at": 0.0, "hex": "#B03040"}, {"at": 1.0, "hex": "#D04858"}],
       "opacity": 0.9, "finish": "gloss"}


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


def test_lipstick_never_paints_open_mouth_surfaces(fitter):
    """张嘴 landmark 网格：口腔开口面上的稠密 splat 口红权重必须为 0（tag 路径）。"""
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
    made_m = fitter.apply_makeup(cloud_of(mouth_pts), [LIP], V[:468],
                                 mouth=np.ones(len(mouth_pts), bool))
    assert np.allclose(made_m["rgba"], cloud_of(mouth_pts)["rgba"])   # 口腔零涂色

    band_pts = _tri_centroids(V, tris[topo["band"]])
    made_b = fitter.apply_makeup(cloud_of(band_pts), [LIP], V[:468],
                                 mouth=np.zeros(len(band_pts), bool))
    d = np.abs(made_b["rgba"] - cloud_of(band_pts)["rgba"]).max(1)
    assert (d > 0.1).mean() > 0.3                                     # 唇红带大面积上色


def test_lipstick_generic_path_excludes_mouth_behind_strip(fitter):
    """通用路径（无 uv/无 tag）：口腔线条面法向背面的"牙齿" splat 被侧别测试排除。"""
    rng = np.random.default_rng(9)
    s, ang, t = 1.4, 0.3, np.array([0.3, -0.1, 2.0])
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0],
                  [-np.sin(ang), 0, np.cos(ang)]])
    V, tris = fitter.model.base, fitter.model.tris
    topo = fitter.lip_topology()
    mouth_pts = _tri_centroids(V, tris[topo["mouth"]], seed=7)
    band_pts = _tri_centroids(V, tris[topo["band"]], seed=8)
    # 牙齿样点：沿口腔面条法线反方向（进入口腔）推后几个 τ
    p0, p1, p2 = V[tris[topo["mouth"]][:, 0]], V[tris[topo["mouth"]][:, 1]], \
        V[tris[topo["mouth"]][:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    teeth = mouth_pts - fn * 0.02
    P = np.concatenate([band_pts, teeth])
    Pw = (P @ (s * R).T) + t
    n = len(Pw)
    cloud = {"xyz": Pw.astype(np.float32),
             "scale": np.full((n, 3), 0.004, np.float32),
             "rot": np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32),
             "rgba": np.full((n, 4), 0.6, np.float32)}
    landmarks = V[:468] @ (s * R).T + t
    made = fitter.apply_makeup(cloud, [LIP], landmarks, intensity=0.9)
    d = np.abs(made["rgba"][:len(band_pts)] - cloud["rgba"][:len(band_pts)]).max(1)
    assert (d > 0.1).mean() > 0.3                        # 唇红带上色
    d_teeth = np.abs(made["rgba"][len(band_pts):] - cloud["rgba"][len(band_pts):]).max(1)
    assert (d_teeth > 0.1).mean() < 0.2                  # 牙齿几乎不上色


# ---------------- L1：妆容壳层 ----------------

def test_build_makeup_shell_ranges_and_merge(fitter, tmp_path):
    cloud, landmarks = _user_cloud(fitter, n=900)
    # 壳层锚定"观测唇点"（颜色门控权重场）；合成云颜色均匀时走纯几何窄核，
    # 在唇红带附近补一批候选点保证壳层有可锚定的 splat
    topo = fitter.lip_topology()
    band_pts = _tri_centroids(fitter.model.base, fitter.model.tris[topo["band"]],
                              jit=0.002, seed=11)
    m = len(band_pts)
    cloud["xyz"] = np.concatenate([cloud["xyz"], band_pts.astype(np.float32)])
    cloud["scale"] = np.concatenate([cloud["scale"], np.full((m, 3), 0.004, np.float32)])
    cloud["rot"] = np.concatenate([cloud["rot"], np.tile([0.0, 0.0, 0.0, 1.0], (m, 1))
                                   .astype(np.float32)])
    cloud["rgba"] = np.concatenate([cloud["rgba"], np.full((m, 4), 0.75, np.float32)])
    presets = json.loads((ROOT / "makeup-skill" / "presets" / "date-rose.json")
                         .read_text(encoding="utf-8"))
    layers = [dict(l) for l in presets["layers"] if l.get("enabled", True)]
    shell, ranges = fitter.build_makeup_shell(cloud, layers, landmarks, intensity=0.8)
    assert len(shell["xyz"]) >= 500
    assert sum(r["count"] for r in ranges) == len(shell["xyz"])
    regions = {r["region"] for r in ranges}
    assert {"lipstick", "eyeliner", "eyebrow"} <= regions      # 锐利层全部进壳层
    a = shell["rgba"][:, 3]
    assert ((a > 0) & (a <= 1)).all()                          # alpha 有效
    assert shell["gloss"].min() >= 0 and shell["shin"].min() >= 0
    assert shell["gloss"].max() > 0.3                          # 口红 gloss 已烘焙
    # 壳层口红 splat 已带妆色（与素颜灰底显著不同）
    lip_rng = ranges[[r["region"] for r in ranges].index("lipstick")]
    seg = shell["rgba"][lip_rng["start"]:lip_rng["start"] + lip_rng["count"]]
    assert np.abs(seg[:, :3] - 0.6).max() > 0.15
    # 合并 + ply 往返
    made = fitter.apply_makeup(cloud, layers, landmarks, intensity=0.8)
    merged = fitter.merge_shell(made, shell)
    assert len(merged["xyz"]) == len(made["xyz"]) + len(shell["xyz"])
    from makeupstudio.face3dgs.splat_io import read_ply, write_ply  # noqa: E402
    write_ply(merged, tmp_path / "merged.ply")
    back = read_ply(tmp_path / "merged.ply")
    assert len(back["xyz"]) == len(merged["xyz"])
    # 壳层锚定真实张嘴几何：landmark 网格张嘴后壳层点跟随上/下唇分离
    Vopen = fitter.model.pose_explicit(0, 0, 0, mouth_k=1.0, smile_k=0.0)
    shell_o, _ = fitter.build_makeup_shell(cloud, layers, Vopen[:468], intensity=0.8)
    assert len(shell_o["xyz"]) > 0


def test_shell_ranges_manifest_written_by_fit_canonical(fitter, tmp_path):
    rng = np.random.default_rng(3)
    V, tris = fitter.model.base, fitter.model.tris
    rows = rng.choice(len(tris), size=800)
    cloud = {"xyz": V[tris[rows, 0]].astype(np.float32),
             "scale": np.full((800, 3), 0.008, np.float32),
             "rot": np.tile([0.0, 0.0, 0.0, 1.0], (800, 1)).astype(np.float32),
             "rgba": np.full((800, 4), 0.75, np.float32)}
    presets = json.loads((ROOT / "makeup-skill" / "presets" / "date-rose.json")
                         .read_text(encoding="utf-8"))
    fitter.fit_canonical(cloud, presets, tmp_path, intensity=0.8)
    man = json.loads((tmp_path / "shell_manifest.json").read_text(encoding="utf-8"))
    assert man["shell_splats"] > 0
    assert man["face_splats"] + man["shell_splats"] == man["face_splats"] + \
        sum(r["count"] for r in man["ranges"])
    report = json.loads((tmp_path / "fit_report.json").read_text(encoding="utf-8"))
    assert report["shell_splats"] == man["shell_splats"]
