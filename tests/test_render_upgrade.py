"""渲染效果升级（R+）单测：法线 / 主光估计 / SH 导出贯穿 / relight 门控 / UV 高通。

对应 2026-09 渲染质量升级：min-scale 轴法线 + kNN 平滑、SH degree2 训练导出、
MKLT1 主光 sidecar、纯叠加 PBR（relight=0 默认）、底模 albedo 高通微观纹理。
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "desktop-app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from makeupstudio.face3dgs.appearance import pbr  # noqa: E402
from makeupstudio.face3dgs.appearance.normals import axis_normals, smooth_normals  # noqa: E402


def quat_xyzw(axis, angle):
    ax = np.asarray(axis, np.float64)
    ax = ax / np.linalg.norm(ax)
    s, c = np.sin(angle / 2), np.cos(angle / 2)
    return np.array([*ax * s, c])


# ---------------- normals ----------------

def test_axis_normals_picks_min_scale_axis():
    # scale_x 最小 → 薄轴 = 局部 X；旋转 90°(绕 Z) 后 X→世界 Y
    n = 5
    rot = np.tile(quat_xyzw([0, 0, 1], np.pi / 2), (n, 1))
    scale = np.tile([0.01, 0.2, 0.3], (n, 1))
    nm = axis_normals(rot, scale)
    assert nm.shape == (n, 3)
    assert np.allclose(np.abs(nm[:, 1]), 1.0, atol=1e-9)   # 世界 Y
    # 各向同性退化为 quat 假设路径：scale_z 最小 → Z 列
    scale2 = np.tile([0.2, 0.3, 0.01], (n, 1))
    nm2 = axis_normals(rot, scale2)
    assert np.allclose(nm2, [0, 0, 1], atol=1e-9)


def test_smooth_normals_converges_planar_and_keeps_outliers():
    rng = np.random.default_rng(3)
    n = 300
    xyz = rng.random((n, 3)).astype(np.float32) * 0.1      # 密集平面片
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))
    noisy = normals + rng.normal(0, 0.6, (n, 3))
    noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
    out = smooth_normals(xyz, noisy, k=8, iters=2)
    # 平滑后与平面法线夹角显著小于噪声输入
    cos_in = noisy @ normals[0]
    cos_out = out @ normals[0]
    assert cos_out.mean() > cos_in.mean() + 0.2
    # 孤立飞点保持原法线（不扩散噪声）
    iso_xyz = np.array([[100.0, 100.0, 100.0]], np.float32)
    iso_n = np.array([[0.0, 1.0, 0.0]])
    out_iso = smooth_normals(np.vstack([xyz, iso_xyz]), np.vstack([noisy, iso_n]),
                             k=8, iters=2)
    assert np.allclose(out_iso[-1], iso_n[0], atol=1e-6)


# ---------------- 主光估计 + MKLT1 ----------------

def _view(img, K=None, w2c=None):
    from makeupstudio.face3dgs.appearance.train_base import View
    h, w = img.shape[:2]
    if K is None:
        K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64)
    if w2c is None:
        w2c = np.eye(4)
    mask = np.ones((h, w), np.float32)
    return View(name="v", w2c=w2c, K=K, img=img, mask=mask)


def test_estimate_light_dir_faces_bright_region():
    from makeupstudio.face3dgs.appearance.train_base import estimate_light_dir
    w = h = 64
    img = np.full((h, w, 3), 0.2, np.float32)
    img[:, 48:] = 0.95                      # 亮区在相机 +x 侧
    # 窄 FOV（fx=32）让偏轴亮区的反投影方向 x 分量显著
    K = np.array([[32, 0, w / 2], [0, 32, h / 2], [0, 0, 1]], np.float64)
    d, strength, tint = estimate_light_dir([_view(img, K), _view(img, K)])
    assert d[0] > 0.5                       # 主光指向亮侧
    assert d[2] > 0.0                       # 且在相机前方
    assert abs(np.linalg.norm(d) - 1.0) < 1e-9
    assert 0.0 < strength <= 1.0
    assert tint.shape == (3,) and tint.max() <= 1.0


def test_write_light_bin_layout_roundtrip():
    from makeupstudio.face3dgs.appearance.train_base import write_light_bin
    d = np.array([0.0, 0.6, 0.8])
    s = 0.42
    t = np.array([0.9, 0.8, 0.7])
    import tempfile
    from pathlib import Path
    p = Path(tempfile.mkdtemp()) / "light.bin"
    write_light_bin(p, d, s, t)
    raw = p.read_bytes()
    assert raw[:4] == b"MKLT"
    assert struct.unpack("<H", raw[4:6])[0] == 1
    assert len(raw) == 44                   # 16 头 + 12 dir + 4 strength + 12 tint
    vals = struct.unpack("<7f", raw[16:])
    assert np.allclose(vals[0:3], d, atol=1e-6) and abs(vals[3] - s) < 1e-6
    assert np.allclose(vals[4:7], t, atol=1e-6)


# ---------------- PLY SH 贯穿 ----------------

def _mk_cloud(n=40, with_sh=True):
    rng = np.random.default_rng(7)
    cloud = {
        "xyz": rng.random((n, 3)).astype(np.float32),
        "scale": rng.random((n, 3)).astype(np.float32) * 0.01 + 0.005,
        "rot": rng.normal(size=(n, 4)).astype(np.float32),
        "rgba": np.concatenate([rng.random((n, 3)) * 0.5 + 0.25,
                                np.full((n, 1), 0.9, np.float32)], 1).astype(np.float32),
    }
    if with_sh:
        cloud["sh_rest"] = (rng.normal(0, 0.05, (n, 8, 3))).astype(np.float32)
    return cloud


def test_ply_sh_rest_roundtrip(tmp_path):
    from makeupstudio.face3dgs.splat_io import read_ply
    from makeupstudio.splat3d import SplatCloudBuilder
    cloud = _mk_cloud()
    p = tmp_path / "base.ply"
    SplatCloudBuilder.export_ply(cloud, p)
    assert "property float f_rest_23" in p.read_text(errors="ignore")
    out = read_ply(p)
    assert out["sh_rest"].shape == (40, 8, 3)
    assert np.allclose(out["sh_rest"], cloud["sh_rest"], atol=1e-6)
    # 通道主序：f_rest_{c*8+k} ↔ sh_rest[:, k, c]
    assert np.allclose(out["sh_rest"][:, 0, 1], cloud["sh_rest"][:, 0, 1], atol=1e-6)
    # DC/几何字段不受影响
    assert np.allclose(out["rgba"][:, :3], cloud["rgba"][:, :3], atol=1e-5)
    # 无 SH 的云 → 读回无 sh_rest 键（向后兼容）
    p2 = tmp_path / "bare.ply"
    SplatCloudBuilder.export_ply(_mk_cloud(with_sh=False), p2)
    assert "sh_rest" not in read_ply(p2)


# ---------------- relight 门控（纯叠加默认） ----------------

def test_pbr_relight_gate_default_additive_only():
    n = 256
    rng = np.random.default_rng(11)
    rgba = np.concatenate([np.full((n, 3), 0.6), np.ones((n, 1))], 1)
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))            # 全部正对相机
    mat = pbr.Material(np.full(n, 0.5), np.full(n, 0.0),
                       np.zeros(n), np.zeros(n))
    # relight=0（默认）：光照方向不影响输出（无高光通道时）
    a = pbr.shade_points(rgba, normals, [0, 0, 1], [0, 0, 1], np.ones(3), mat)
    b = pbr.shade_points(rgba, normals, [0, 0, 1], [0, 1, 0], np.ones(3), mat)
    assert np.allclose(a, b, atol=1e-9)
    assert np.allclose(a, 0.6, atol=1e-6)                 # 与烘焙色一致，不改明暗
    # relight=1：背光面被压暗（合成 wrap diffuse 生效）
    back = np.tile([0.0, 0.0, -1.0], (n, 1))
    c = pbr.shade_points(rgba, back, [0, 0, 1], [0, 0, 1], np.ones(3), mat,
                         relight=1.0)
    assert c.mean() < 0.6 * 0.45                          # 0.30 底光以下


# ---------------- 会话路径：AvatarData 携带 sh_rest 贯穿 ----------------

def test_avatar_io_sh_rest_survives_session(tmp_path, monkeypatch):
    """带 f_rest 的画像进会话：load → (归一化) → 抽稀 → save → load，
    sh_rest 形状与数值保持（Unity 端因此拿到 SH 视角色）。"""
    sys.path.insert(0, str(ROOT / "makeup-skill" / "scripts"))
    try:
        from avatar_io import AvatarData, load_avatar, save_avatar
    finally:
        sys.path.pop(0)
    rng = np.random.default_rng(9)
    n = 500
    means = rng.random((n, 3)).astype(np.float32)
    av = AvatarData(
        means, rng.random((n, 3)).astype(np.float32) * 0.01,
        np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
        rng.random((n, 3)).astype(np.float32) * 0.5 + 0.25,
        np.full(n, 0.9, np.float32), face_height=1.0,
        sh_rest=rng.normal(0, 0.05, (n, 8, 3)).astype(np.float32))
    p = tmp_path / "av.ply"
    save_avatar(av, p)
    assert "property float f_rest_23" in p.read_text(errors="ignore")
    av2 = load_avatar(p, max_count=300)          # 归一化 + 抽稀
    assert av2.sh_rest is not None and av2.sh_rest.shape == (300, 8, 3)
    assert np.isfinite(av2.sh_rest).all()
    # 抽稀保留的是同一批 splat：数值与原（归一化后）对应行一致
    p2 = tmp_path / "av2.ply"
    save_avatar(av2, p2)
    av3 = load_avatar(p2)
    assert np.allclose(av3.sh_rest, av2.sh_rest, atol=1e-6)
    # DC-only 资产不受影响
    av.sh_rest = None
    save_avatar(av, p2)
    assert load_avatar(p2).sh_rest is None


# ---------------- 导出往返回归（历史 bug 防复发） ----------------

def test_ply_export_roundtrip_opacity_and_quat(tmp_path):
    """export_ply→read_ply 往返必须无损：opacity 用标准 logit（旧 √ 编码会系统性
    压透明，脸面渗底色）；rot PLY(wxyz)↔内部(xyzw) 转换严格互逆。"""
    from makeupstudio.face3dgs.splat_io import read_ply
    from makeupstudio.splat3d import SplatCloudBuilder
    rng = np.random.default_rng(13)
    n = 100
    a = np.clip(rng.uniform(0.05, 0.99, (n, 1)), 0.02, 0.98).astype(np.float32)
    cloud = {
        "xyz": rng.random((n, 3)).astype(np.float32),
        "scale": rng.random((n, 3)).astype(np.float32) * 0.01 + 0.003,
        "rot": rng.normal(size=(n, 4)).astype(np.float32),   # 内部 xyzw
        "rgba": np.concatenate([np.full((n, 3), 0.5, np.float32), a], 1),
    }
    p = tmp_path / "rt.ply"
    SplatCloudBuilder.export_ply(cloud, p)
    out = read_ply(p)
    assert np.allclose(out["rgba"][:, 3], a[:, 0], atol=5e-3), "opacity 往返损失过大"
    assert np.allclose(out["rot"], cloud["rot"], atol=1e-4), "四元数往返错位"
    assert np.allclose(out["scale"], cloud["scale"], rtol=1e-3)


def test_mip3d_filter_floors_small_splats():
    from makeupstudio.face3dgs.appearance.train_base import mip3d_filter
    # 3×3×3 网格点，间距 0.1：一个 splat 尺度远小于间距 → 被 γ·nn 抬底
    g = np.mgrid[0:3, 0:3, 0:3].reshape(3, -1).T * 0.1
    cloud = {"xyz": g.astype(np.float32),
             "scale": np.full((27, 3), 0.001, np.float32)}
    mip3d_filter(cloud, gamma=0.3)
    assert cloud["scale"].min() >= 0.3 * 0.1 * 0.99
    # γ=0 不动
    cloud2 = {"xyz": g.astype(np.float32),
              "scale": np.full((27, 3), 0.001, np.float32)}
    mip3d_filter(cloud2, gamma=0.0)
    assert np.allclose(cloud2["scale"], 0.001)


# ---------------- 离线渲染交付（相机几何，CPU-only） ----------------

def _synthetic_face():
    """椭球脸 + 鼻尖：用于轴向/相机几何测试（归一化脸高≈2）。"""
    rng = np.random.default_rng(21)
    n = 3000
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = dirs * np.array([0.7, 1.0, 0.45])                # y=上下轴，z=正背
    pts = pts[rng.random(n) < (0.7 + 0.3 * (dirs[:, 2] > 0))]  # 正面略密
    nose = np.array([[0.0, -0.05, 0.52]])
    return np.vstack([pts, nose]).astype(np.float64)


def test_infer_axes_landmarks_and_pca_fallback():
    from makeupstudio.face3dgs.appearance.offline_render import infer_axes
    xyz = _synthetic_face()
    center = xyz.mean(0)
    # 地标路径：L10 额顶 / L152 颏底 / L1 鼻尖
    lm = np.zeros((468, 3))
    lm[10] = [0, 0.95, 0]; lm[152] = [0, -0.95, 0]; lm[1] = [0, -0.05, 0.55]
    c, up, front = infer_axes(xyz, lm)
    assert np.allclose(up, [0, 1, 0], atol=1e-6)
    assert front[2] > 0.9 and abs(front[1]) < 0.2          # 朝 +z，已去 up 分量
    # PCA 回退不崩、输出都是单位向量
    c2, up2, front2 = infer_axes(xyz, None)
    assert abs(np.linalg.norm(up2) - 1) < 1e-9
    assert abs(np.linalg.norm(front2) - 1) < 1e-9
    assert np.isfinite(c2).all()


def test_orbit_synthetic_and_poses_geometry():
    from makeupstudio.face3dgs.appearance.offline_render import (
        orbit_from_poses, orbit_synthetic)
    center = np.zeros(3)
    up, front = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    cams = orbit_synthetic(center, up, front, fh=2.0, n=9, yaw_range_deg=24.0)
    assert len(cams) == 9
    for w2c, K, yaw in cams:
        eye = -w2c[:3, :3].T @ w2c[:3, 3]
        assert np.linalg.norm(eye) > 1.0                    # 在脸外
        # 相机朝向中心：w2c 把 center 变换到 +z 轴上（正前）
        c_cam = w2c[:3, :3] @ center + w2c[:3, 3]
        assert c_cam[2] > 0 and abs(c_cam[0]) < 1e-6 and abs(c_cam[1]) < 1e-6
        # 正交 + 右手 + 图像正立（y_cam 指向 -up = 画面下方）
        assert np.allclose(w2c[:3, :3] @ w2c[:3, :3].T, np.eye(3), atol=1e-9)
        assert np.linalg.det(w2c[:3, :3]) > 0
        assert w2c[1, :3] @ up < 0
        assert K[0, 0] == K[1, 1] > 0 and K[0, 2] == 512.0
    yaws = [y for _, _, y in cams]
    assert yaws[0] == -24.0 and yaws[-1] == 24.0

    # 位姿环绕：sweep 覆盖真实相机方位角范围，焦距取真实中位数
    w2cs, Ks = [], []
    for yaw_deg in (-30.0, 0.0, 30.0):
        a = np.radians(yaw_deg)
        eye = np.array([np.sin(a) * 10.0, 0.0, np.cos(a) * 10.0])
        zc = -eye / np.linalg.norm(eye)
        xc = np.cross([0, 1, 0], zc)
        xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        w = np.eye(4)
        w[:3, :3] = np.stack([xc, yc, zc])
        w[:3, 3] = -w[:3, :3] @ eye
        w2cs.append(w)
        Ks.append(np.array([[800.0, 0, 240], [0, 800.0, 240], [0, 0, 1]]))
    pcams = orbit_from_poses(w2cs, Ks, center, n=7, size=512)
    pyaws = [y for _, _, y in pcams]
    assert pyaws[0] <= -30.0 and pyaws[-1] >= 30.0          # 覆盖真实范围 + pad
    for w2c, K, _ in pcams:
        c_cam = w2c[:3, :3] @ center + w2c[:3, 3]
        assert c_cam[2] > 0 and abs(c_cam[0]) < 1e-6
        assert abs(K[0, 0] - 800.0) < 1e-6                  # 焦距 = 真实相机中位数


def test_deliver_requires_cuda(tmp_path):
    """无 CUDA 环境给明确错误而不是崩溃（有 CUDA 时 smoke 由 test_train_base 系列覆盖）。"""
    from makeupstudio.face3dgs.appearance.offline_render import deliver
    try:
        import torch
        if torch.cuda.is_available():
            pytest.skip("有 CUDA，跳过无 GPU 分支")
    except ImportError:
        pass
    p = tmp_path / "a.ply"
    p.write_bytes(b"ply")
    with pytest.raises(RuntimeError):
        deliver(p, tmp_path / "out")




# ---------------- 底模 albedo 高通 ----------------

def test_base_micro_hp_sparse_none_and_grid_finite():
    from makeupstudio.face3dgs.appearance.makeup_uv import UvMakeupBaker
    t = 64
    # 覆盖不足（全部无效）→ None
    cloud0 = {"rgba": np.array([[0.5, 0.5, 0.5, 1.0]], np.float32)}
    tx = np.array([32.0]); ty = np.array([32.0])
    assert UvMakeupBaker._base_micro_hp(cloud0, t, tx, ty,
                                        np.array([False])) is None
    # 网格均匀覆盖、有明暗纹理 → 有限值且被钳到 ±1
    rng = np.random.default_rng(5)
    n = 2000
    xs = rng.integers(8, 56, n).astype(np.float64)
    ys = rng.integers(8, 56, n).astype(np.float64)
    lum = 0.3 + 0.4 * rng.random(n)
    cloud = {"rgba": np.stack([lum, lum, lum, np.ones(n)], 1).astype(np.float32)}
    hp = UvMakeupBaker._base_micro_hp(cloud, t, xs, ys, np.ones(n, bool))
    assert hp is not None and hp.shape == (n,)
    assert np.isfinite(hp).all() and np.abs(hp).max() <= 1.0 + 1e-6
    # 平坦颜色 → 高通幅度≈0 → None（退回程序噪声路径）
    cloud_flat = {"rgba": np.full((n, 4), 0.5, np.float32)}
    assert UvMakeupBaker._base_micro_hp(cloud_flat, t, xs, ys,
                                        np.ones(n, bool)) is None
