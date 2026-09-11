"""追踪协议 v2 / 姿态求解 / 1€ 滤波 / 帧中继分片 的单元测试（纯 numpy，无 mediapipe）。"""
from __future__ import annotations

import math

import numpy as np
import pytest

import tracking_protocol as tp

W, H = 640, 480
F = tp.DEFAULT_FOCAL_RATIO * W


def rot_ypr(yaw, pitch, roll):
    y, p, r = map(math.radians, (yaw, pitch, roll))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


@pytest.fixture(scope="module")
def canonical():
    return tp.load_canonical(tp.default_canonical_path())


def test_canonical_shape_and_units(canonical):
    assert canonical.shape == (468, 3)
    # 厘米量级：脸高 ~17-18cm
    assert 15 < canonical[:, 1].max() - canonical[:, 1].min() < 20


@pytest.mark.parametrize("yaw,pitch,roll,depth", [
    (0, 0, 0, 50), (25, -10, 5, 45), (-30, 15, -8, 60), (40, 20, 10, 40), (-45, -20, 0, 30)])
def test_solve_pose_recovers_synthetic(canonical, yaw, pitch, roll, depth):
    R = rot_ypr(yaw, pitch, roll)
    T = np.array([3.0, -2.0, -depth])
    img = tp.project_canonical(canonical, R, T, W, H, F)
    pose, local = tp.solve_pose(img, canonical, W, H, F)
    rot_err = math.degrees(math.acos(min(1.0, (np.trace(R.T @ pose.rotation) - 1) / 2)))
    assert rot_err < 2.5, f"旋转误差 {rot_err:.2f}°"
    assert abs(pose.depth_cm - depth) / depth < 0.05
    assert np.allclose(pose.translation[:2], T[:2], atol=0.4)
    # 表情层：刚性点应回到 canonical 附近
    err = np.abs(local[tp.RIGID_LANDMARKS] - canonical[tp.RIGID_LANDMARKS]).mean()
    assert err < 0.3
    assert abs(np.linalg.det(pose.rotation) - 1) < 1e-6


def test_pack_unpack_roundtrip(canonical):
    R = rot_ypr(10, 5, 0)
    img = tp.project_canonical(canonical, R, np.array([0, 0, -50.0]), W, H, F)
    pose, local = tp.solve_pose(img, canonical, W, H, F)
    pk = tp.TrackPacket(seq=42, t=12.5, w=W, h=H, focal_px=F, face_ok=True,
                        pose=pose.matrix(), local=local, img=img[:, :2], relay=True)
    data = tp.pack_v2(pk)
    assert tp.is_v2(data)
    assert len(data) == tp.HEADER_SIZE + 64 + 468 * 12 + 468 * 8
    q = tp.unpack_v2(data)
    assert q.seq == 42 and q.w == W and q.h == H and q.face_ok and q.relay
    assert abs(q.focal_px - F) < 1e-3
    assert np.allclose(q.pose, pose.matrix(), atol=1e-4)
    assert np.allclose(q.local, local, atol=1e-3)
    assert np.allclose(q.img, img[:, :2], atol=1e-5)


def test_pack_noface_is_header_only():
    data = tp.pack_v2(tp.TrackPacket(seq=1, t=0.0, w=W, h=H, focal_px=F, face_ok=False))
    assert len(data) == tp.HEADER_SIZE
    q = tp.unpack_v2(data)
    assert not q.face_ok and q.local is None and q.pose is None


def test_unpack_rejects_garbage():
    with pytest.raises(ValueError):
        tp.unpack_v2(b"NOPE" + b"\0" * 40)
    with pytest.raises(ValueError):
        tp.unpack_v2(tp.MAGIC_V2 + b"\0" * 10)


def test_pack_v1_legacy_json(canonical):
    img = tp.project_canonical(canonical, np.eye(3), np.array([0, 0, -50.0]), W, H, F)
    data = tp.pack_v1(True, W, H, 1.0, img)
    import json
    j = json.loads(data)
    assert j["ok"] and len(j["pts"]) == 468 and j["w"] == W
    assert not tp.is_v2(data)


def test_gl_to_unity_matrix_convention():
    """App 端 GlToUnity = F·M·F（F=diag(1,1,-1)），平移 z 变号、厘米→米。用 Python 复现并检验性质。"""
    R = rot_ypr(20, -10, 5)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [3.0, -2.0, -50.0]
    Fm = np.diag([1.0, 1.0, -1.0, 1.0])
    U = Fm @ M @ Fm
    # 与 C# 逐元素规则一致：m02,m12,m20,m21,m23 变号
    expect = M.copy()
    for i, j in [(0, 2), (1, 2), (2, 0), (2, 1), (2, 3)]:
        expect[i, j] = -M[i, j]
    assert np.allclose(U, expect)
    # 仍是刚体：旋转部分正交、行列式 +1；相机前方 z 为正（Unity）
    assert abs(np.linalg.det(U[:3, :3]) - 1) < 1e-9
    assert U[2, 3] > 0


def test_one_euro_static_noise_and_step_response():
    """机理验证：静止点强降噪；阶跃在 ~0.3s 内跟上（无无限拖影）。参数与数据尺度相关：
    C# 端点数据是米（beta≈75 Hz/(m/s)），此处用同尺度数据验证算法行为。"""
    rng = np.random.default_rng(3)
    dt = 1 / 30
    t = np.arange(0, 3.0, dt)
    # 1) 静止点 + 抖动：1€ 应显著降噪（tau = 1/(2π·0.8) → α≈0.13 @30fps）
    noisy = np.zeros((len(t), 3)) + rng.normal(0, 5e-4, (len(t), 3))   # ±0.5mm 抖动
    filt = tp.OneEuroFilter(min_cutoff=0.8, beta=75.0)
    out = np.array([filt(noisy[i], float(t[i])) for i in range(len(t))])
    err_raw = np.abs(noisy[30:] - noisy[30:].mean(axis=0)).mean()
    err_f = np.abs(out[30:] - out[30:].mean(axis=0)).mean()
    assert err_f < err_raw * 0.4, (err_f, err_raw)
    # 2) 阶跃 1cm：0.3s 内到达 90%（无无限滞后）
    step = np.where((t >= 1.0)[:, None], 0.01, 0.0)
    filt2 = tp.OneEuroFilter(min_cutoff=0.8, beta=75.0)
    out2 = np.array([filt2(step[i], float(t[i])) for i in range(len(t))])
    after = out2[t >= 1.4]
    assert np.allclose(after, 0.01, atol=0.001)


def test_frame_chunk_assemble_roundtrip():
    payload = bytes(range(256)) * 300      # 76.8KB → 5 片
    chunks = tp.chunk_frame(7, payload, 640, 360)
    assert len(chunks) == 5
    asm = tp.FrameAssembler()
    got = None
    for c in chunks[::-1]:                  # 乱序送达也能重组
        got = asm.feed(c) or got
    assert got is not None
    fid, w, h, jpeg = got
    assert (fid, w, h) == (7, 640, 360) and jpeg == payload


def test_frame_assembler_drops_stale_incomplete():
    asm = tp.FrameAssembler(max_pending=2)
    a = tp.chunk_frame(1, b"x" * 40000, 64, 48)   # 3 片，只送 1 片
    asm.feed(a[0])
    for fid in (2, 3, 4):
        for c in tp.chunk_frame(fid, b"y" * 100, 64, 48):
            asm.feed(c)
    assert asm.dropped >= 1
    assert asm.feed(a[1]) is None and asm.feed(a[2]) is None   # 已被淘汰，不会拼出脏帧
