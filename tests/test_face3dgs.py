"""face3dgs 测试：引擎发现、采集引导、重建编排（模拟引擎）、COLMAP 读写、
脸部隔离与妆容贴合（合成数据，不依赖 GPU / mediapipe / 真实引擎）。"""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "desktop-app"
sys.path.insert(0, str(APP))

_spec = importlib.util.spec_from_file_location("preview_render_core", ROOT / "makeup-skill" / "scripts" / "preview_render.py")
prc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prc)

from makeupstudio.face3dgs import colmap_io, engines  # noqa: E402
from makeupstudio.face3dgs.capture import BUCKET_MIN_SECONDS, OrbitCaptureSession  # noqa: E402
from makeupstudio.face3dgs.fit_makeup import (FaceMakeupFitter, kabsch_similarity,  # noqa: E402
                                              triangulate_dlt)
from makeupstudio.face3dgs.isolate import isolate_face  # noqa: E402
from makeupstudio.face3dgs.reconstruct import (LocalEngineBackend, ReconJob,  # noqa: E402
                                               ReconResult)
from makeupstudio.face3dgs.splat_io import read_ply, write_ply  # noqa: E402
from makeupstudio.guide_overlay import draw_capture_guidance  # noqa: E402


# ---------------- 引擎发现 ----------------

def test_engine_discovery_priority(tmp_path, monkeypatch):
    d = tmp_path / "engines"
    d.mkdir()
    (d / "colmap.exe").write_bytes(b"fake")
    monkeypatch.setenv("OOOSPLAT_ENGINE_DIR", str(d))
    monkeypatch.delenv("OOOSPLAT_COLMAP", raising=False)
    monkeypatch.delenv("PATH")                      # 排除系统 PATH 干扰
    assert engines.find_engine("colmap") == d / "colmap.exe"
    # 显式环境变量优先于 ENGINE_DIR
    other = tmp_path / "other.exe"
    other.write_bytes(b"fake")
    monkeypatch.setenv("OOOSPLAT_COLMAP", str(other))
    assert engines.find_engine("colmap") == other
    monkeypatch.delenv("OOOSPLAT_ENGINE_DIR")
    assert engines.find_engine("colmap") == other


# ---------------- 采集引导 ----------------

class FakeTracker:
    """按调用次数吐出预设 yaw；px 横跨 60% 画面（脸足够大）。"""

    def __init__(self, yaws):
        self.yaws = list(yaws)
        self.i = 0

    def detect(self, frame, ts):
        if self.i >= len(self.yaws):
            yaw = self.yaws[-1]
        else:
            yaw = self.yaws[self.i]
        self.i += 1
        h, w = frame.shape[:2]
        px = np.stack([np.linspace(0.2 * w, 0.8 * w, 478),
                       np.full(478, 0.5 * h)], axis=1).astype(np.float32)
        return {"px": px, "norm": None, "pose": (yaw, 0.0, 0.0)}


def _feed(session: OrbitCaptureSession, yaw_seq, dt_ms=200.0, luma=120, t0=0.0):
    frame = np.full((240, 320, 3), luma, np.uint8)
    t = t0
    out = []
    for yaw in yaw_seq:
        t += dt_ms
        out.append(session.process(frame, t))
    return out


def test_capture_guidance_buckets_and_done(tmp_path):
    tr = FakeTracker([-40.0] * 12 + [0.0] * 12 + [40.0] * 13)
    s = OrbitCaptureSession(tmp_path / "c.mp4", tracker=tr, frame_size=(320, 240))
    gs = _feed(s, [-40.0] * 12)                     # 2.4s 左侧
    assert gs[-1].bucket_seconds["left"] >= BUCKET_MIN_SECONDS
    assert not gs[-1].done
    _feed(s, [0.0] * 12, t0=2.6)                    # 2.4s 正面
    _feed(s, [40.0] * 12, t0=5.2)                   # 2.4s 右侧
    g = s.process(np.full((240, 320, 3), 120, np.uint8), 10000.0)
    assert g.done and g.progress == 1.0
    res = s.finish()
    assert res.video_path.exists() and res.video_path.stat().st_size > 0
    report = json.loads(res.video_path.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert report["frames"] > 0 and report["bucket_seconds"]["left"] > 0


def test_capture_rejects_dark_and_no_face(tmp_path):
    s = OrbitCaptureSession(tmp_path / "c.mp4", tracker=None, frame_size=(320, 240))
    g = _feed(s, [0.0] * 3, luma=20)[-1]
    assert not g.ok and any("暗" in m for m in g.messages)
    s2 = OrbitCaptureSession(tmp_path / "c2.mp4", tracker=FakeTracker([0.0]), frame_size=(320, 240))
    g2 = s2.process(np.zeros((10, 10, 3), np.uint8), 200.0)   # 画面太小 → 脸占比检查
    # 无脸与过暗判定仍会拒绝写入
    s3 = OrbitCaptureSession(tmp_path / "c3.mp4", tracker=None, frame_size=(320, 240))
    g3 = s3.process(np.zeros((240, 320, 3), np.uint8), 200.0)
    assert not g3.ok and g3.total_seconds == 0.0


# ---------------- 采集引导（帧内 HUD 动画） ----------------

def test_capture_guidance_carries_face_box(tmp_path):
    """判定结果需携带脸框与姿态，供画面内引导动画使用。"""
    s = OrbitCaptureSession(tmp_path / "c.mp4", tracker=FakeTracker([-30.0]),
                            frame_size=(320, 240))
    g = s.process(np.full((240, 320, 3), 120, np.uint8), 200.0)
    assert g.face_box is not None and g.face_box[0] < g.face_box[2]
    assert g.yaw == pytest.approx(-30.0) and g.pitch == pytest.approx(0.0)
    s2 = OrbitCaptureSession(tmp_path / "c2.mp4", tracker=None, frame_size=(320, 240))
    g2 = s2.process(np.full((240, 320, 3), 120, np.uint8), 200.0)
    assert g2.face_box is None


def test_capture_overlay_draws_animation(tmp_path):
    """画面内引导：不同状态下都应产出同尺寸 HUD 帧，且确实画了东西。"""
    tr = FakeTracker([-40.0] * 2)
    s = OrbitCaptureSession(tmp_path / "c.mp4", tracker=tr, frame_size=(320, 240))
    frame = np.full((240, 320, 3), 120, np.uint8)
    g = s.process(frame, 200.0)
    for g_now in (g, s.process(frame, 400.0)):
        vis = draw_capture_guidance(frame, g_now, t=0.5)
        assert vis.shape == frame.shape and vis.dtype == np.uint8
        assert not np.array_equal(vis, frame)          # HUD 已叠加
        assert int((vis[:, :, 1] > 180).sum()) > 50    # 存在引导色（绿色系）像素
    # 无脸状态也要能画（红色提示，不崩溃）
    s2 = OrbitCaptureSession(tmp_path / "c2.mp4", tracker=None, frame_size=(320, 240))
    g2 = s2.process(frame, 200.0)
    vis = draw_capture_guidance(frame, g2, t=0.5)
    assert vis.shape == frame.shape and not np.array_equal(vis, frame)
    # 完成态
    g2.done = True
    assert draw_capture_guidance(frame, g2, t=0.5).shape == frame.shape


# ---------------- 重建编排（模拟引擎） ----------------

def _fake_run_factory(brush_ok=True):
    calls = []

    def fake_run(self, cmd, cb, stage, frac, check=True):
        name = Path(str(cmd[0])).stem.lower()
        calls.append((stage, name))
        project = None
        if name == "brush":
            # brush v0.3：第一个位置参数 = 数据集根
            project = Path(str(cmd[1]))
        if project is None:
            for a in cmd[1:]:
                a = str(a)
                if a.startswith("--") or not ("images" in a or "colmap" in a):
                    continue
                root = a.split("images")[0].split("colmap")[0]
                if root.strip():
                    project = Path(root.rstrip("/\\") or a)
                    break
        if project is None:
            for a in cmd:
                if str(a).endswith(".db"):
                    project = Path(str(a)).parent.parent
                    break
        if project is None:
            project = Path(str(cmd[-1])).parent
        if name == "ffmpeg":
            imgs = project / "images"
            imgs.mkdir(parents=True, exist_ok=True)
            for i in range(1, 16):
                cv2.imwrite(str(imgs / f"frame_{i:05d}.jpg"),
                            np.full((60, 80, 3), 128, np.uint8))
        elif name == "colmap":
            sparse = None
            for a in cmd:
                if "--output_path" in str(a):
                    sparse = Path(str(a))
            sparse = sparse or project / "colmap" / "sparse"
            (sparse / "0").mkdir(parents=True, exist_ok=True)
            _write_synthetic_sparse(sparse / "0", n_images=15, size=(80, 60))
        elif name == "brush" and brush_ok:
            (project / "final.ply").write_bytes(b"x")
        return 0

    return fake_run, calls


def _write_synthetic_sparse(out_dir: Path, n_images: int, size):
    """单 PINHOLE 相机、绕 y 轴旋转的环形机位（z 朝向原点）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size
    f = 100.0
    with open(out_dir / "cameras.bin", "wb") as fc:
        fc.write(struct.pack("<Q", 1))
        fc.write(struct.pack("<i", 1))                     # cam_id=1
        fc.write(struct.pack("<i", 0))                     # SIMPLE_PINHOLE
        fc.write(struct.pack("<Q", w))
        fc.write(struct.pack("<Q", h))
        fc.write(struct.pack("<3d", f, w / 2, h / 2))      # 无 n_params 字段（真实格式）
    with open(out_dir / "images.bin", "wb") as fi:
        fi.write(struct.pack("<Q", n_images))
        for k in range(n_images):
            ang = 2 * np.pi * k / n_images
            # 相机在半径 2 的圆上看向原点：z 轴（光轴）指向原点
            R = np.array([[np.cos(ang), 0, np.sin(ang)],
                          [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
            R_cam = R @ np.diag([1.0, 1.0, -1.0])           # 翻转 z：光轴朝原点
            C = R @ np.array([0, 0, 2.0])
            q = rot_to_quat(R_cam.T)                        # world→cam 四元数 (w,x,y,z)
            t = -R_cam.T @ C
            fi.write(struct.pack("<i", k + 1))
            fi.write(struct.pack("<4d", *q))
            fi.write(struct.pack("<3d", *t))
            fi.write(struct.pack("<i", 1))
            name = f"frame_{k + 1:05d}.jpg"
            fi.write(name.encode() + b"\x00")
            fi.write(struct.pack("<Q", 0))                  # 无 2D 点


def rot_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    return np.array([1.0, 0, 0, 0])


def test_reconstruction_pipeline_with_mock_engines(tmp_path, monkeypatch):
    fake_run, calls = _fake_run_factory()
    monkeypatch.setattr(LocalEngineBackend, "_run", fake_run)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    backend = LocalEngineBackend()
    # 引擎发现与机器环境相关（.engines / OOOSplat 安装目录都会被搜索），
    # 这里显式置空再断言“缺失即拒绝”，不依赖测试机的安装状态
    monkeypatch.setattr("makeupstudio.face3dgs.reconstruct.find_engine", lambda n: None)
    assert all(backend.availability().values()) is False  # 真实引擎缺失 → run 会先拒绝
    monkeypatch.setattr("makeupstudio.face3dgs.reconstruct.find_engine",
                        lambda n: Path(f"C:/fake/{n}.exe"))
    res = backend.run(ReconJob(video, tmp_path / "proj", "draft"))
    assert res.ply_path.exists() and res.frames == 15
    stages = [c[0] for c in calls]
    assert stages.index("extract") < stages.index("colmap") < stages.index("train")
    model = colmap_io.read_sparse(res.sparse_dir)          # 产物可被下游读取
    assert len(model.images) == 15


def test_cv2_frame_extraction_fallback(tmp_path):
    """无 ffmpeg 时 OpenCV 抽帧回退可用（真实小视频端到端的前置）。"""
    import cv2
    video = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(30):
        vw.write(np.full((48, 64, 3), i * 8 % 255, np.uint8))
    vw.release()
    backend = LocalEngineBackend()
    images = tmp_path / "images"
    images.mkdir()
    n = backend._extract(video, images, fps=5.0, cb=lambda *a: None)
    assert n >= 14                                          # 3s @5fps ≈ 15 帧


def test_reconstruction_missing_engine_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("makeupstudio.face3dgs.reconstruct.find_engine", lambda n: None)
    with pytest.raises(Exception, match="缺少引擎"):
        LocalEngineBackend().run(ReconJob(tmp_path / "v.mp4", tmp_path / "p"))


# ---------------- COLMAP 读写与投影 ----------------

def test_colmap_roundtrip_and_projection(tmp_path):
    d = tmp_path / "sparse"
    _write_synthetic_sparse(d, n_images=4, size=(80, 60))
    model = colmap_io.read_sparse(d)
    cam = model.camera
    assert cam.model == "SIMPLE_PINHOLE" and cam.params[0] == 100.0
    im = model.images["frame_00001.jpg"]
    # 相机 0 位于 (0,0,2) 看向原点 → 原点应投在主点
    p = cam.project(np.array([[0.0, 0.0, 0.0]]), im["qvec"], im["tvec"])
    assert np.allclose(p[0], [40.0, 30.0], atol=1e-6)


# ---------------- 脸部隔离 ----------------

def test_isolate_face_by_reprojection_vote(tmp_path):
    proj = tmp_path / "proj"
    (proj / "images").mkdir(parents=True)
    _write_synthetic_sparse(proj / "colmap" / "sparse" / "0", n_images=4, size=(80, 60))
    for k in range(1, 5):
        cv2.imwrite(str(proj / "images" / f"frame_{k:05d}.jpg"),
                    np.full((60, 80, 3), 128, np.uint8))
    rng = np.random.default_rng(3)
    face = rng.normal([0, 0, 0], 0.08, (500, 3))           # 原点附近 = 脸
    bg = np.stack([rng.uniform(-3, 3, 300), rng.uniform(-2, 2, 300),
                   rng.uniform(-1, 1, 300)], axis=1)
    bg += np.array([0, 0, -2.5])                            # 相机背后/画面外 = 背景
    xyz = np.concatenate([face, bg]).astype(np.float32)
    cloud = {"xyz": xyz, "scale": np.full((len(xyz), 3), 0.001, np.float32),
             "rot": np.tile([0, 0, 0, 1.0], (len(xyz), 1)).astype(np.float32),
             "rgba": np.full((len(xyz), 4), 0.8, np.float32)}
    ply = proj / "final.ply"
    write_ply(cloud, ply)

    def bbox_provider(frame):
        return (0.25, 0.1, 0.75, 0.9)                       # 固定中央脸框（归一化）

    result = ReconResult(project_dir=proj, ply_path=ply,
                         sparse_dir=proj / "colmap" / "sparse" / "0",
                         images_dir=proj / "images", seconds=0.0)
    stats = isolate_face(result, proj / "face.ply", bbox_provider=bbox_provider,
                         max_views=4)
    assert stats["views_with_face"] == 4
    kept = read_ply(proj / "face.ply")
    assert 300 < len(kept["xyz"]) <= 505                    # 脸保留、背景剔除
    assert np.linalg.norm(kept["xyz"], axis=1).max() < 0.5


# ---------------- 妆容贴合 ----------------

def test_kabsch_similarity_recovers_transform():
    rng = np.random.default_rng(5)
    src = rng.normal(0, 1, (100, 3))
    ang = 0.7
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0],
                  [-np.sin(ang), 0, np.cos(ang)]])
    s_true, t_true = 1.7, np.array([4.0, -2.0, 1.0])
    dst = src @ (s_true * R).T + t_true
    s, Rr, t = kabsch_similarity(src, dst)
    assert abs(s - s_true) < 1e-9
    assert np.allclose(Rr, R, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-9)


def test_triangulate_dlt_two_views():
    f, cx, cy = 100.0, 40.0, 30.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    P0 = K @ np.hstack([np.eye(3), [[0], [0], [0]]])          # 相机 0 在原点
    R1 = np.array([[1, 0, 0.1], [0, 1, 0], [-0.1, 0, 1]])
    R1 /= np.linalg.det(R1) ** (1 / 3)
    C1 = np.array([0.5, 0.0, 0.0])
    P1 = K @ np.hstack([R1, -R1 @ C1[:, None]])
    X = np.array([0.1, -0.2, 3.0])
    uv0 = (P0 @ np.append(X, 1))[:2] / (P0 @ np.append(X, 1))[2]
    uv1 = (P1 @ np.append(X, 1))[:2] / (P1 @ np.append(X, 1))[2]
    Xr = triangulate_dlt([P0, P1], [uv0, uv1])
    assert np.allclose(Xr, X, atol=1e-6)


@pytest.fixture(scope="module")
def fitter():
    return FaceMakeupFitter()


def _synthetic_user_cloud(fitter, jitter=0.004, n=900):
    """用已知相似变换把 canonical 表面散成"用户脸"点云 + 完美 3D 地标。"""
    rng = np.random.default_rng(9)
    ang = 0.3
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0],
                  [-np.sin(ang), 0, np.cos(ang)]])
    s, t = 1.4, np.array([0.3, -0.1, 2.0])
    V = fitter.model.base
    tris = fitter.model.tris
    rows = rng.choice(len(tris), size=n, p=None)
    area = 0.5 * np.linalg.norm(np.cross(V[tris[:, 1]] - V[tris[:, 0]],
                                         V[tris[:, 2]] - V[tris[:, 0]]), axis=1)
    rows = rng.choice(len(tris), size=n, p=area / area.sum())
    b = rng.dirichlet([1, 1, 1], size=n)
    P = (V[tris[rows, 0]] * b[:, :1] + V[tris[rows, 1]] * b[:, 1:2]
         + V[tris[rows, 2]] * b[:, 2:])
    P = P + rng.normal(0, jitter, P.shape)
    xyz = (P @ (s * R).T) + t
    # 法线朝 +z（近似），薄片 rot
    rot = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    landmarks = V[:468] @ (s * R).T + t
    return {"xyz": xyz.astype(np.float32), "scale": np.full((n, 3), 0.01, np.float32),
            "rot": rot.astype(np.float32),
            "rgba": np.full((n, 4), 0.75, np.float32)}, landmarks
