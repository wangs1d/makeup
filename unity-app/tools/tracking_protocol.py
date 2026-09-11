#!/usr/bin/env python3
"""tracking_protocol — sidecar ↔ App 的追踪数据协议与姿态求解（纯 numpy，无 mediapipe 依赖）。

## 追踪包 v2（二进制，UDP，每帧一包，≈10KB）

    header  <4sHIdHHfHH   magic=b"MKT2", flags, seq, t(秒), w, h, focal_px, n_pts, reserved
    pose    16 × f32      4×4 行主序，厘米；OpenGL 相机系：x 右、y 上、z 朝向相机（相机看 -z）
    pts     n_pts×3 × f32 头部局部坐标（canonical 系，厘米）= 表情层，App 直接作网格顶点
    img     n_pts×2 × f32 归一化图像坐标 (x,y)，y 向下（2D 叠加/兜底用）

    flags: bit0 face_ok   bit1 has_pose   bit2 relay_source   bit3 smoothed

无脸帧只发 header（flags.face_ok=0，n_pts=0）。旧版 v1 JSON 见 pack_v1（App 端仍兼容）。

## 帧中继（App → sidecar，UDP 分片）

    chunk header <4sIHHHHI  magic=b"MKF1", frame_id, chunk_idx, chunk_count, w, h, total_len
    payload ≤ CHUNK_PAYLOAD 字节的 JPEG 片段；sidecar 按 frame_id 重组。

## 姿态求解

MediaPipe 只给归一化 (x, y, 相对 z)。这里用 canonical 网格（厘米）做弱透视 Umeyama 相似变换拟合：
    S_i ≈ s·R·C_i + t     S = 屏幕像素系 (x右, y上, z朝相机)
深度 Z = focal_px / s，平移 T = (t_x·Z/f, t_y·Z/f, -Z)。表情层 L_i = Rᵀ(S_i − t)/s。
姿态与表情解耦后，App 端根节点放 M=[R|T]，顶点放 L。
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC_V2 = b"MKT2"
MAGIC_FRAME = b"MKF1"
HEADER_FMT = "<4sHIdHHfHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)          # 30
POSE_FMT = "<16f"
CHUNK_FMT = "<4sIHHHHI"
CHUNK_HEADER_SIZE = struct.calcsize(CHUNK_FMT)     # 20
CHUNK_PAYLOAD = 16000

FLAG_FACE = 1 << 0
FLAG_POSE = 1 << 1
FLAG_RELAY = 1 << 2
FLAG_SMOOTHED = 1 << 3

DEFAULT_TRACK_PORT = 8766
DEFAULT_RELAY_PORT = 8767
DEFAULT_FOCAL_RATIO = 0.85   # focal_px = ratio × 宽；≈ 61° 水平视场（普通笔记本/USB 摄像头）

# 姿态拟合用的刚性关键点（避开会随表情大幅变形的唇/眼睑）：
# 脸轮廓、鼻梁鼻尖、眼角、额头、颧骨、下巴
RIGID_LANDMARKS = sorted({
    # face oval
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    # nose bridge / tip / base
    6, 197, 195, 5, 4, 1, 19, 94, 2, 168, 8, 9, 151,
    # eye corners
    33, 133, 362, 263, 130, 243, 463, 359,
    # cheekbones
    50, 101, 118, 117, 111, 116, 123, 280, 330, 347, 346, 340, 345, 352,
    # chin / jaw center
    199, 175, 200, 18, 421, 201,
})


# ---------------- canonical 模型 ----------------

def load_canonical(obj_path: str | Path) -> np.ndarray:
    """canonical_face_model.obj → (468,3) 厘米，右手系，+z 朝观察者。"""
    verts = []
    for line in Path(obj_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
    return np.asarray(verts, np.float64)


def default_canonical_path() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "Assets" / "StreamingAssets" / "canonical_face_model.obj",
                 here.parent.parent / "makeup-skill" / "references" / "canonical_face_model.obj"):
        if cand.exists():
            return cand
    raise FileNotFoundError("找不到 canonical_face_model.obj")


# ---------------- 姿态求解 ----------------

def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """相似变换拟合 dst ≈ s·R·src + t（Umeyama 1991）。返回 (s, R(3,3), t(3,))。"""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / n
    u, d, vt = np.linalg.svd(cov)
    sgn = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sgn[2, 2] = -1.0
    r = u @ sgn @ vt
    var_s = (xs ** 2).sum() / n
    s = float(np.trace(np.diag(d) @ sgn) / max(var_s, 1e-12))
    t = mu_d - s * (r @ mu_s)
    return s, r, t


@dataclass
class Pose:
    rotation: np.ndarray            # (3,3) canonical → 相机系
    translation: np.ndarray         # (3,) 厘米，OpenGL 相机系
    scale_px_per_cm: float
    depth_cm: float
    rms_px: float = 0.0             # 拟合残差（像素）

    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.translation
        return m

    def euler_deg(self) -> tuple[float, float, float]:
        """(yaw, pitch, roll) 度，便于日志/测试。"""
        r = self.rotation
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[1, 2]))))
        yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
        roll = math.degrees(math.atan2(r[1, 0], r[1, 1]))
        return yaw, pitch, roll


def image_to_screen(img_pts: np.ndarray, w: int, h: int) -> np.ndarray:
    """归一化 (x,y,z) → 屏幕像素系 (x右, y上, z朝相机)。MediaPipe z 单位≈x（按宽归一），越小越近。"""
    p = np.asarray(img_pts, np.float64)
    out = np.empty_like(p)
    out[:, 0] = (p[:, 0] - 0.5) * w
    out[:, 1] = (0.5 - p[:, 1]) * h
    out[:, 2] = -p[:, 2] * w
    return out


def solve_pose(img_pts: np.ndarray, canonical: np.ndarray, w: int, h: int,
               focal_px: float, iters: int = 3) -> tuple[Pose, np.ndarray]:
    """弱透视姿态拟合 + 透视校正迭代。返回 (Pose, 头部局部坐标 (N,3) 厘米)。

    每轮用上一轮的逐点深度 Z_i 把透视像素坐标还原为正交坐标 x·Z_i/Z 再重拟合，
    大角度（>30°）下旋转误差从 ~4° 降到 ~1°。
    """
    n = min(len(img_pts), len(canonical))
    screen0 = image_to_screen(img_pts[:n], w, h)
    idx = [i for i in RIGID_LANDMARKS if i < n]
    rig = canonical[idx]
    screen = screen0.copy()
    s, r, t = 1.0, np.eye(3), np.zeros(3)
    for _ in range(max(1, iters)):
        s, r, t = umeyama(rig, screen[idx])
        s = max(s, 1e-6)
        depth = focal_px / s
        zrot = (canonical[:n] @ r.T)[:, 2]
        zi = np.clip(depth - zrot, depth * 0.5, depth * 1.5)
        screen[:, :2] = screen0[:, :2] * (zi / depth)[:, None]
    depth = focal_px / s
    fitted = s * (rig @ r.T) + t
    rms = float(np.sqrt(((fitted[:, :2] - screen[idx, :2]) ** 2).sum(1).mean()))
    translation = np.array([t[0] * depth / focal_px, t[1] * depth / focal_px, -depth])
    local = ((screen - t) @ r) / s          # Rᵀ(S − t)/s
    return Pose(r, translation, s, depth, rms), local


def project_canonical(canonical: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
                      w: int, h: int, focal_px: float) -> np.ndarray:
    """透视投影（测试/合成模式用）：厘米姿态 → MediaPipe 风格归一化 (x, y, z)。"""
    cam = canonical @ rotation.T + translation           # OpenGL 相机系，z<0 在前方
    z = -cam[:, 2]
    x = cam[:, 0] * focal_px / z
    y = cam[:, 1] * focal_px / z
    out = np.empty_like(cam)
    out[:, 0] = x / w + 0.5
    out[:, 1] = 0.5 - y / h
    # MediaPipe z：以脸中心为 0、越近越小，且"与 x 同尺度"——即同样带 f/Z 透视缩放后按宽归一
    out[:, 2] = (z - z.mean()) * (focal_px / z.mean()) / w
    return out


# ---------------- One Euro 滤波（参考实现；C# 端 OneEuroFilter.cs 同算法） ----------------

@dataclass
class OneEuroFilter:
    min_cutoff: float = 1.2
    beta: float = 0.02
    d_cutoff: float = 1.0
    _x: np.ndarray | None = field(default=None, repr=False)
    _dx: np.ndarray | None = field(default=None, repr=False)
    _t: float | None = field(default=None, repr=False)

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = self._dx = None
        self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self._t
        self._t = t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x


# ---------------- 追踪包编解码 ----------------

@dataclass
class TrackPacket:
    seq: int
    t: float
    w: int
    h: int
    focal_px: float
    face_ok: bool
    pose: np.ndarray | None = None      # (4,4)
    local: np.ndarray | None = None     # (N,3) 厘米
    img: np.ndarray | None = None       # (N,2) 归一化
    relay: bool = False
    smoothed: bool = False

    @property
    def n_pts(self) -> int:
        return 0 if self.local is None else len(self.local)


def pack_v2(p: TrackPacket) -> bytes:
    flags = (FLAG_FACE if p.face_ok else 0) | (FLAG_POSE if p.pose is not None else 0) \
        | (FLAG_RELAY if p.relay else 0) | (FLAG_SMOOTHED if p.smoothed else 0)
    n = p.n_pts if p.face_ok else 0
    head = struct.pack(HEADER_FMT, MAGIC_V2, flags, p.seq & 0xFFFFFFFF, float(p.t),
                       p.w, p.h, float(p.focal_px), n, 0)
    if not p.face_ok or n == 0:
        return head
    pose = p.pose if p.pose is not None else np.eye(4)
    body = [head, np.asarray(pose, np.float32).reshape(16).tobytes(),
            np.asarray(p.local[:n], np.float32).reshape(-1).tobytes()]
    img = p.img[:n] if p.img is not None else np.zeros((n, 2), np.float32)
    body.append(np.asarray(img, np.float32).reshape(-1).tobytes())
    return b"".join(body)


def unpack_v2(data: bytes) -> TrackPacket:
    if len(data) < HEADER_SIZE or data[:4] != MAGIC_V2:
        raise ValueError("not a MKT2 packet")
    magic, flags, seq, t, w, h, focal, n, _ = struct.unpack_from(HEADER_FMT, data, 0)
    pk = TrackPacket(seq=seq, t=t, w=w, h=h, focal_px=focal, face_ok=bool(flags & FLAG_FACE),
                     relay=bool(flags & FLAG_RELAY), smoothed=bool(flags & FLAG_SMOOTHED))
    if not pk.face_ok or n == 0:
        return pk
    off = HEADER_SIZE
    need = off + 64 + n * 12 + n * 8
    if len(data) < need:
        raise ValueError(f"MKT2 packet truncated: {len(data)} < {need}")
    pose = np.frombuffer(data, np.float32, 16, off).reshape(4, 4).astype(np.float64)
    off += 64
    pk.pose = pose if flags & FLAG_POSE else None
    pk.local = np.frombuffer(data, np.float32, n * 3, off).reshape(n, 3).astype(np.float64)
    off += n * 12
    pk.img = np.frombuffer(data, np.float32, n * 2, off).reshape(n, 2).astype(np.float64)
    return pk


def pack_v1(face_ok: bool, w: int, h: int, t: float, img_pts: np.ndarray | None) -> bytes:
    """旧版 JSON（App 端 UdpLandmarkReceiver 的兼容路径）。"""
    payload = {"ok": bool(face_ok), "w": int(w), "h": int(h), "t": round(float(t), 3)}
    if face_ok and img_pts is not None:
        payload["pts"] = [[round(float(x), 5), round(float(y), 5), round(float(z), 5)]
                          for x, y, z in img_pts]
    return json.dumps(payload, separators=(",", ":")).encode()


def is_v2(data: bytes) -> bool:
    return len(data) >= 4 and data[:4] == MAGIC_V2


# ---------------- 帧中继分片 ----------------

def chunk_frame(frame_id: int, jpeg: bytes, w: int, h: int) -> list[bytes]:
    total = len(jpeg)
    count = max(1, (total + CHUNK_PAYLOAD - 1) // CHUNK_PAYLOAD)
    out = []
    for i in range(count):
        part = jpeg[i * CHUNK_PAYLOAD:(i + 1) * CHUNK_PAYLOAD]
        out.append(struct.pack(CHUNK_FMT, MAGIC_FRAME, frame_id & 0xFFFFFFFF, i, count, w, h, total) + part)
    return out


class FrameAssembler:
    """按 frame_id 重组分片；新帧完成即丢弃更旧的未完成帧。"""

    def __init__(self, max_pending: int = 4) -> None:
        self._pending: dict[int, dict] = {}
        self.max_pending = max_pending
        self.dropped = 0

    def feed(self, datagram: bytes) -> tuple[int, int, int, bytes] | None:
        """返回完整帧 (frame_id, w, h, jpeg) 或 None。"""
        if len(datagram) < CHUNK_HEADER_SIZE or datagram[:4] != MAGIC_FRAME:
            return None
        _, fid, idx, count, w, h, total = struct.unpack_from(CHUNK_FMT, datagram, 0)
        slot = self._pending.get(fid)
        if slot is None:
            slot = {"parts": {}, "count": count, "w": w, "h": h, "total": total}
            self._pending[fid] = slot
            while len(self._pending) > self.max_pending:
                oldest = min(self._pending)
                if oldest == fid:
                    break
                self._pending.pop(oldest)
                self.dropped += 1
        slot["parts"][idx] = datagram[CHUNK_HEADER_SIZE:]
        if len(slot["parts"]) < slot["count"]:
            return None
        jpeg = b"".join(slot["parts"][i] for i in range(slot["count"]))
        # 完成：丢弃所有更旧的帧
        for k in [k for k in self._pending if k <= fid]:
            self._pending.pop(k, None)
        if len(jpeg) != slot["total"]:
            self.dropped += 1
            return None
        return fid, slot["w"], slot["h"], jpeg
