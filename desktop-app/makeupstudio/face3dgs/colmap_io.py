"""colmap_io — COLMAP 稀疏模型（bin）的最小读写，供 isolate 做重投影。

只实现 isolate 需要的子集：SIMPLE_PINHOLE / SIMPLE_RADIAL / PINHOLE / OPENCV
相机模型、images.bin（四元数+平移+关键点名）。格式参考 COLMAP 官方
scripts/python/read_write_model.py（BSD 授权）。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    cam_id: int
    model: str
    width: int
    height: int
    params: np.ndarray          # 依模型而定

    def project(self, xyz_world: np.ndarray, qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """世界点经相机位姿（world→cam）投影到像素 (N,2)；畸变忽略（人脸近轴可接受）。"""
        R = quat_to_rotmat(qvec)
        cam = (R @ xyz_world.T).T + tvec
        z = cam[:, 2]
        f, cx, cy = self.params[0], self.params[1], self.params[2]
        return np.stack([f * cam[:, 0] / z + cx, f * cam[:, 1] / z + cy], axis=1)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """COLMAP 四元数 (w, x, y, z) → 3×3 旋转。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], np.float64)


@dataclass
class SparseModel:
    cameras: dict[int, Camera]
    images: dict[str, dict]      # name → {qvec, tvec, cam_id}

    @property
    def camera(self) -> Camera:
        return next(iter(self.cameras.values()))    # single_camera=1


def read_images_bin(path: Path) -> dict[str, dict]:
    images: dict[str, dict] = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _img_id = struct.unpack("<i", f.read(4))[0]
            q = np.array(struct.unpack("<4d", f.read(32)), np.float64)
            t = np.array(struct.unpack("<3d", f.read(24)), np.float64)
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.seek(n_pts * 24, 1)                    # 跳过 2D 点与三维点 id
            images[name.decode("utf-8")] = {"qvec": q, "tvec": t, "cam_id": cam_id}
    return images


_STANDARD = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 9,
             8: 8, 9: 9, 10: 12, 11: 18, 12: 3}
_FORK = {0: 3, 1: 4, 2: 8, 3: 4, 4: 5, 5: 12, 6: 12, 7: 9,
         8: 8, 9: 15, 10: 15, 11: 12, 12: 12}
_MODEL_NAME_STD = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                   3: "RADIAL", 4: "OPENCV"}
_MODEL_NAME_FORK = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 3: "SIMPLE_RADIAL",
                    4: "RADIAL", 8: "OPENCV"}


def _parse_cameras_bin(path: Path, n_params_of: dict, model_name_of: dict):
    cameras: dict[int, Camera] = {}
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise ValueError(f"{path} 过短")
        n = struct.unpack("<Q", head)[0]
        for _ in range(n):
            cam_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            n_params = n_params_of.get(model_id)
            if n_params is None:
                raise ValueError(f"未知相机模型 id {model_id}")
            params = np.array(struct.unpack(f"<{n_params}d", f.read(8 * n_params)), np.float64)
            model = model_name_of.get(model_id, f"MODEL{model_id}")
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
                params = np.array([(fx + fy) / 2, cx, cy])
            elif model == "SIMPLE_PINHOLE":
                params = params[:3]
            elif model in ("SIMPLE_RADIAL", "RADIAL"):
                params = params[:3]          # f, cx, cy（忽略畸变）
            elif model == "OPENCV":
                fx, fy, cx, cy = params[:4]
                params = np.array([(fx + fy) / 2, cx, cy])
            else:
                raise ValueError(f"暂不支持的相机模型 {model}")
            cameras[cam_id] = Camera(cam_id, model, w, h, params)
        consumed = f.tell()
    return cameras, consumed


def read_cameras_bin(path: Path, dialect: str = "auto") -> dict[int, Camera]:
    """COLMAP cameras.bin：id(i32) model(i32) w(Q) h(Q) params(f64×N，无 n_params 字段)。

    相机模型 id 的枚举顺序有两种方言：
        "standard"：标准 COLMAP 3/4（pycolmap、官方 colmap CLI 写出）；
        "fork"：本仓库 COLMAP 3.11 fork 的顺序（App 历史契约）。
    "auto"（默认）：先按 standard 解析并用"文件必须恰好耗尽"校验，失败回退 fork。
    """
    if dialect in ("standard", "fork"):
        n_map = _STANDARD if dialect == "standard" else _FORK
        cameras, _ = _parse_cameras_bin(path, n_map,
                                        _MODEL_NAME_STD if dialect == "standard"
                                        else _MODEL_NAME_FORK)
        return cameras
    try:
        cameras, consumed = _parse_cameras_bin(path, _STANDARD, _MODEL_NAME_STD)
        if consumed == Path(path).stat().st_size:
            return cameras
    except Exception:
        pass
    cameras, _ = _parse_cameras_bin(path, _FORK, _MODEL_NAME_FORK)
    return cameras


def read_sparse(sparse_dir: str | Path, dialect: str = "auto") -> SparseModel:
    d = Path(sparse_dir)
    return SparseModel(cameras=read_cameras_bin(d / "cameras.bin", dialect=dialect),
                       images=read_images_bin(d / "images.bin"))
