"""test_lam_adapter — P4 LAM 单图入口：门控 / canonical 配准 / 照片视角 / 语义提升。

覆盖：环境门控（缺失时给可执行提示、predict 抛可执行错误而不阻断视频链路）、
canonical 模板 → 点云帧的相似配准（fit_similarity：已知 (s,R,t) 必须被还原；
离面离群点与正背向翻转下仍收敛）、弱透视拟合（模板 → 照片 2D 地标的复投影精度）、
照片视角相机（photo_camera：在点云帧里把地标投回照片像素）、
load_canonical（landmarks.npy 与点云同帧 + 配准信息）、
photo_zones（单视角 → single_seg，唇区高斯被照片蒙版命中、额区不被命中；
face-parsing 缺席时优雅返回 None），以及 build_asset_from_image 单图全链路
（base_source="LAM" + base/madeup/landmarks/report 齐备）。

全部 CPU/numpy；LAM 推理与 mediapipe 均不参与（parser / px / normals 注入）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "desktop-app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from makeupstudio.face3dgs.appearance import lam_adapter as lam  # noqa: E402
from makeupstudio.face3dgs.appearance.calibrate import LIPS_OUTER  # noqa: E402
from makeupstudio.face3dgs.appearance.pipeline import (  # noqa: E402
    build_asset_from_image,
)
from makeupstudio.face3dgs.appearance.semantics import project_points  # noqa: E402
from makeupstudio.face3dgs.fit_makeup import (  # noqa: E402
    FaceMakeupFitter,
    _sample_tris,
)
from makeupstudio.face3dgs.splat_io import read_ply, write_ply  # noqa: E402

SKIN = np.array([205.0, 185.0, 170.0])       # 素颜肤色（RGB 0..255）


# ---------------- 合成数据 ----------------

def _fitter() -> FaceMakeupFitter:
    return FaceMakeupFitter(tracker_factory=lambda: (_ for _ in ()).throw(
        RuntimeError("test 不需要 tracker")))


@pytest.fixture(scope="module")
def template():
    """canonical 468 地标（测试里的"LAM 帧模板"）。"""
    return np.asarray(_fitter().model.base[:lam.N_LM], np.float64)


@pytest.fixture(scope="module")
def surface_points():
    """canonical 面部表面均匀采样（充当 LAM 回归出的高斯位置）。"""
    f = _fitter()
    rng = np.random.default_rng(7)
    P, _b, _rows = _sample_tris(f.model.base, f.model.tris, 24000, rng)
    return np.asarray(P, np.float64)


def _known_srt(seed: int = 3) -> tuple[float, np.ndarray, np.ndarray]:
    """已知相似变换：绕轴小角度旋转 + 尺度 + 平移（模拟 LAM 的 canonical 帧）。"""
    rng = np.random.default_rng(seed)
    ax = rng.normal(size=3)
    ax = ax / np.linalg.norm(ax)
    ang = np.radians(24.0)
    K = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return 2.5, R, np.array([1.0, -0.5, 0.7])


def _cloud_from(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray,
                hair: int = 0, seed: int = 5) -> np.ndarray:
    """表面点 → 目标帧；hair>0 时额外撒头发/背景离群点（配准的鲁棒性压力）。"""
    out = points @ (s * R).T + t
    if hair:
        rng = np.random.default_rng(seed)
        span = float(np.linalg.norm(np.ptp(out, axis=0)))
        out = np.concatenate([out, out.mean(0, keepdims=True)
                              + rng.normal(0, 1.6 * span, (hair, 3))], 0)
    return out


def _splat_cloud(xyz: np.ndarray) -> dict:
    n = len(xyz)
    return {"xyz": np.asarray(xyz, np.float32),
            "scale": np.full((n, 3), 0.002, np.float32),
            "rot": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (n, 1)),
            "rgba": np.concatenate([np.tile(SKIN / 255.0, (n, 1)).astype(np.float32),
                                    np.ones((n, 1), np.float32)], 1),
            "sh_rest": np.zeros((n, 3, 3), np.float32)}


class _FakeParser:
    """face-parsing 替身：直接给出照片像素空间的语义蒙版（免 torch 权重）。"""

    def __init__(self, masks: dict[str, np.ndarray]):
        self._masks = masks
        self.load_error = None

    def available(self) -> bool:
        return True

    def parse(self, _frame_bgr):
        return self._masks


class _DeadParser:
    def available(self) -> bool:
        return False

    load_error = "test: 权重未缓存"

    def parse(self, _frame_bgr):
        return None


@pytest.fixture(scope="module")
def lam_status():
    return lam.status()


# ---------------- 门控 ----------------

def test_status_reports_missing_and_hint(lam_status):
    assert isinstance(lam_status.ok, bool)
    assert isinstance(lam_status.missing_hint, str)
    if not lam_status.ok:                        # 开发机常态：给出可执行指引
        assert lam_status.missing_hint


def test_predict_raises_actionable_hint(lam_status, tmp_path):
    """环境不完整时 predict 抛可执行错误（调用方据此降级，不阻断视频链路）。"""
    if lam_status.ok:
        pytest.skip("本机 LAM 环境已就绪，门控分支不可达")
    with pytest.raises(RuntimeError, match="LAM 环境不完整"):
        lam.predict(tmp_path / "none.jpg", tmp_path / "out")


def test_build_asset_from_image_gated_without_env(lam_status, tmp_path):
    img = tmp_path / "photo.png"
    cv2.imwrite(str(img), np.full((64, 64, 3), 200, np.uint8))
    if lam_status.ok:
        pytest.skip("本机 LAM 环境已就绪，门控分支不可达")
    with pytest.raises(RuntimeError, match="LAM 环境不完整"):
        build_asset_from_image(img, tmp_path / "asset", spec=None)


def test_find_ply_falls_back_to_newest(tmp_path):
    lam_dir = tmp_path / "lam"
    (lam_dir / "sub").mkdir(parents=True)
    (lam_dir / "sub" / "a.ply").write_bytes(b"ply\n")
    assert lam._find_ply(lam_dir).name == "a.ply"
    (lam_dir / lam.LAM_PLY_NAME).write_bytes(b"ply\n")   # 约定名优先
    assert lam._find_ply(lam_dir).name == lam.LAM_PLY_NAME


# ---------------- canonical 配准 ----------------

def test_fit_similarity_recovers_known_transform(template, surface_points):
    s, R, t = _known_srt()
    cloud = _cloud_from(surface_points, s, R, t)
    s_hat, R_hat, t_hat, rmse = lam.fit_similarity(cloud, template)
    span = float(np.linalg.norm(np.ptp(cloud, axis=0)))
    assert rmse < 0.02 * span                                  # 配准残差 ≈ 0
    assert abs(s_hat - s) / s < 0.05                           # 尺度还原
    truth = template @ (s * R).T + t
    got = template @ (s_hat * R_hat).T + t_hat
    assert np.median(np.linalg.norm(got - truth, axis=1)) < 0.03 * span


def test_fit_similarity_robust_to_off_face_points(template, surface_points):
    """头发/背景离群（16% 体积、距离脸 1.6× 尺度）不能把配准拉走。"""
    s, R, t = _known_srt()
    cloud = _cloud_from(surface_points, s, R, t, hair=4000)
    _s, _r_hat, _t_hat, rmse = lam.fit_similarity(cloud, template)
    span = float(np.linalg.norm(np.ptp(cloud, axis=0)))
    assert rmse < 0.05 * span


def test_fit_similarity_handles_flipped_front(template, surface_points):
    """正背向翻转（绕 up 轴 180°）：PCA 定不了 front 符号，两向试解必须收敛。"""
    s, _R0, t = _known_srt()
    R = np.diag([-1.0, 1.0, -1.0]) @ _R0
    cloud = _cloud_from(surface_points, s, R, t)
    s_hat, R_hat, t_hat, _rmse = lam.fit_similarity(cloud, template)
    span = float(np.linalg.norm(np.ptp(cloud, axis=0)))
    truth = template @ (s * R).T + t
    got = template @ (s_hat * R_hat).T + t_hat
    assert np.median(np.linalg.norm(got - truth, axis=1)) < 0.03 * span


def test_landmarks_in_frame_are_consistent(template, surface_points):
    s, R, t = _known_srt()
    cloud = _cloud_from(surface_points, s, R, t)
    lm, (s_hat, _R_hat, _t_hat), rmse = lam.landmarks_in_frame(cloud, template)
    assert lm.shape == (lam.N_LM, 3)
    assert rmse < 0.02 * float(np.linalg.norm(np.ptp(cloud, axis=0)))
    # 与已知变换下的真值一致（下游 bind_uv.register 的残差由此 ≈ 0）
    truth = template @ (s * R).T + t
    span = float(np.linalg.norm(np.ptp(cloud, axis=0)))
    assert np.median(np.linalg.norm(lm - truth, axis=1)) < 0.03 * span
    assert abs(s_hat - s) / s < 0.05


def test_load_canonical_writes_landmarks(template, surface_points, tmp_path):
    s, R, t = _known_srt()
    ply = tmp_path / "gaussian.ply"
    write_ply(_splat_cloud(_cloud_from(surface_points, s, R, t)), ply)
    out_lm = tmp_path / "landmarks.npy"
    cloud, lm, info = lam.load_canonical(ply, out_lm)
    assert len(cloud["xyz"]) == len(surface_points)
    assert out_lm.is_file() and np.load(out_lm).shape == (lam.N_LM, 3)
    assert info["landmarks"] == lam.N_LM and info["landmarks_rmse"] >= 0.0
    assert abs(np.load(out_lm) - lm).max() < 1e-6


def test_canonical_template_prefers_explicit_path(template, tmp_path):
    p = tmp_path / "tpl.npy"
    np.save(p, template + 1.0)
    got = lam.canonical_template(p)
    assert got.shape == (lam.N_LM, 3)
    assert np.abs(got - template - 1.0).max() < 1e-9


# ---------------- 照片视角 ----------------

def _ortho_px(template, s, R, t, noise=0.0, seed=1):
    px = s * (template @ np.asarray(R, np.float64).T)[:, :2] + np.asarray(t)
    if noise:
        px = px + np.random.default_rng(seed).normal(0, noise, px.shape)
    return px


FRONT_VIEW = np.diag([-1.0, 1.0, -1.0])      # 相机在 +z 侧（看到脸的正面）


def test_weak_perspective_recovers_projection(template):
    px = _ortho_px(template, 320.0, FRONT_VIEW, [256.0, 256.0], noise=0.6)
    s, R, t = lam.weak_perspective_fit(template, px)
    err = np.linalg.norm(s * (template @ R.T)[:, :2] + t - px, axis=1)
    assert abs(s - 320.0) / 320.0 < 0.05
    assert float(np.median(err)) < 1.0                    # 亚像素复投影


def test_photo_camera_projects_landmarks_back_to_photo(template):
    """点云帧下的照片相机：把地标投回照片像素（弱透视近似误差 <3px）。"""
    lm = template.copy()                                  # 点云帧 == 模板帧
    px = _ortho_px(template, 300.0, FRONT_VIEW, [256.0, 256.0])
    R, t, K = lam.photo_camera(lm, px, (1.0, np.eye(3), np.zeros(3)), template)
    xy, ok = project_points(lm, R, t, K)
    assert ok.all()
    assert float(np.median(np.linalg.norm(xy - px, axis=1))) < 3.0


def test_photo_camera_composes_template_to_cloud_frame(template, surface_points):
    """模板帧 ≠ 点云帧时仍成立：配准后的地标经相机投影 = 照片像素。"""
    s1, R1, t1 = _known_srt()
    lm = template @ (s1 * R1).T + t1
    px = _ortho_px(template, 300.0, FRONT_VIEW, [256.0, 256.0])
    R, t, K = lam.photo_camera(lm, px, (s1, R1, t1), template)
    xy, ok = project_points(lm, R, t, K)
    assert ok.all()
    assert float(np.median(np.linalg.norm(xy - px, axis=1))) < 3.0


# ---------------- 语义观测妆区（单图 → single_seg） ----------------

def _photo_with_lips(template, size=640, scale=340.0, pad=0.14):
    """合成照片 + 照片像素空间里的"lips"蒙版（唇外轮廓多边形）。"""
    px = _ortho_px(template, scale, FRONT_VIEW, [size / 2, size / 2])
    img = np.full((size, size, 3), 200, np.uint8)
    lip = px[list(LIPS_OUTER)].astype(np.int32)
    cv2.fillPoly(img, [lip], (60, 60, 190), lineType=cv2.LINE_AA)
    mask = np.zeros((size, size), np.float32)
    cv2.fillPoly(mask, [lip], 1.0, lineType=cv2.LINE_AA)
    k = max(3, int(round(pad * scale * 0.1)) | 1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return img, px, np.clip(mask, 0.0, 1.0)


def _radial_normals(xyz):
    n = np.asarray(xyz, np.float64) - np.asarray(xyz, np.float64).mean(0)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def test_photo_zones_single_view_marks_lip_splats(template, surface_points, tmp_path):
    """单张照片 → 唇区高斯被语义蒙版命中，额区不被命中；来源如实标 single_seg。"""
    cloud_xyz = surface_points
    img, px, lips = _photo_with_lips(template)
    photo = tmp_path / "photo.png"
    cv2.imwrite(str(photo), img)
    zones = lam.photo_zones(_splat_cloud(cloud_xyz), template, photo,
                            ("lipstick",), px=px, parser=_FakeParser({"lips": lips}),
                            srt=(1.0, np.eye(3), np.zeros(3)),
                            normals=_radial_normals(cloud_xyz),
                            region_scale={"lipstick": 0.9})
    assert zones is not None
    rec = zones.report["lipstick"]
    assert rec["level"] == "single_seg"          # 单帧观测不冒充多视角投票
    assert rec["mode"] == "replace" and rec["views_used"] == 1
    assert rec["splats"] >= 40
    p = zones.fields.seg["lipstick"].p
    assert p is not None and p.shape == (len(cloud_xyz),)
    # 唇区高斯：投影落在唇多边形内 → 概率高；额顶高斯：概率低
    xy, ok = project_points(cloud_xyz, *(lam.photo_camera(
        template, px, (1.0, np.eye(3), np.zeros(3)), template)))
    in_lip = (lips[np.clip(np.round(xy[:, 1]).astype(int), 0, 639),
                   np.clip(np.round(xy[:, 0]).astype(int), 0, 639)] > 0.5) & ok
    assert float(np.median(p[in_lip])) > 0.8
    top = cloud_xyz[:, 1] > np.quantile(cloud_xyz[:, 1], 0.95)
    assert float(np.max(p[top])) < 0.5


def test_photo_zones_gated_without_parser(template, surface_points, tmp_path):
    img, px, lips = _photo_with_lips(template)
    photo = tmp_path / "photo.png"
    cv2.imwrite(str(photo), img)
    zones = lam.photo_zones(_splat_cloud(surface_points), template, photo,
                            ("lipstick",), px=px, parser=_DeadParser(),
                            srt=(1.0, np.eye(3), np.zeros(3)))
    assert zones is None                          # 门控：保持几何兜底，不阻断


def test_photo_zones_missing_photo_returns_none(template, surface_points, tmp_path):
    zones = lam.photo_zones(_splat_cloud(surface_points), template,
                            tmp_path / "nope.png", ("lipstick",), px=template[:, :2],
                            parser=_FakeParser({"lips": np.ones((8, 8), np.float32)}))
    assert zones is None


# ---------------- 单图全链路 ----------------

def test_build_asset_from_image_end_to_end(template, surface_points, tmp_path):
    """--ply 路径（跳过 LAM 推理）：资产 + 上妆 + report 全链路可离线跑通。"""
    s, R, t = _known_srt()
    ply = tmp_path / "lam" / lam.LAM_PLY_NAME
    ply.parent.mkdir(parents=True)
    write_ply(_splat_cloud(_cloud_from(surface_points, s, R, t)), ply)
    img = np.full((512, 512, 3), 200, np.uint8)
    photo = tmp_path / "photo.png"
    cv2.imwrite(str(photo), img)
    spec = {"layers": [{"id": "lip", "region": "lipstick", "enabled": True,
                        "opacity": 0.8,
                        "color_stops": [{"at": 0.0, "hex": "#C21858"},
                                        {"at": 1.0, "hex": "#E91E63"}]}]}
    out = tmp_path / "asset"
    base_xyz, landmarks, report = build_asset_from_image(
        photo, out, spec=spec, ply=ply, tex=512)
    assert report["base_source"] == "LAM"
    assert report["frames_selected"] == 1
    assert report["lam"]["landmarks_rmse"] >= 0.0
    for name in ("base.ply", "madeup.ply", "landmarks.npy", "asset_report.json"):
        assert (out / name).exists(), name
    assert np.load(out / "landmarks.npy").shape == (lam.N_LM, 3)
    made = read_ply(out / "madeup.ply")
    assert len(made["xyz"]) > len(base_xyz["xyz"])          # 妆容壳层已合并
    assert (out / "makeup_maps.npz").exists()
    disk = json.loads((out / "asset_report.json").read_text(encoding="utf-8"))
    assert disk["base_source"] == "LAM" and disk["splats"] == len(base_xyz["xyz"])
    assert disk["makeup_layer_splats"] > 0


def test_build_asset_from_image_without_spec_writes_base_only(
        template, surface_points, tmp_path):
    s, R, t = _known_srt()
    ply = tmp_path / "g.ply"
    write_ply(_splat_cloud(_cloud_from(surface_points, s, R, t)), ply)
    photo = tmp_path / "photo.png"
    cv2.imwrite(str(photo), np.full((256, 256, 3), 190, np.uint8))
    out = tmp_path / "asset"
    _c, _lm, report = build_asset_from_image(photo, out, ply=ply, tex=256)
    assert report["base_source"] == "LAM" and (out / "base.ply").exists()
    assert not (out / "madeup.ply").exists()
    assert (out / "asset_report.json").exists()