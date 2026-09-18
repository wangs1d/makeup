"""sfm — 扫描 → 位姿：视频/图像目录的增量 SfM（pycolmap）。

产品链路的第一环：用户上传视频 或 摄像头环绕扫描（face3dgs.capture）→
抽帧 → pycolmap 特征/匹配/增量重建 → 选最大连通子模型 → 输出
colmap/sparse/<best>（train_base 的位姿来源）。COLMAP 二进制不再必需，
Brush 训练器也被 gsplat 取代——旧 reconstruct 引擎链退位。
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ProgressCB = Callable[[float, str], None]


@dataclass
class SfmResult:
    images_dir: Path
    sparse_dir: Path            # 最大子模型（train_base 消费）
    frames: int
    registered: int


def extract_frames(video: str | Path, images_dir: str | Path,
                   fps: float = 10.0, max_frames: int = 200,
                   on_progress: ProgressCB | None = None) -> int:
    """视频 → 抽帧（jpg）。fps 控制密度；短视频自动提高抽帧率保帧数。"""
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    for stale in images_dir.glob("frame_*.jpg"):
        stale.unlink()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(int(round(src_fps / fps)), 1)
    n, idx = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cv2.imwrite(str(images_dir / f"frame_{n:05d}.jpg"), frame)
            n += 1
            if n >= max_frames:
                break
        idx += 1
        if on_progress and total:
            on_progress(idx / total, f"抽帧 {n} 帧")
    cap.release()
    if n < 10:
        raise RuntimeError(f"抽帧仅得 {n} 帧，视频太短或损坏")
    return n


def run_sfm(images_dir: str | Path, project_dir: str | Path,
            on_progress: ProgressCB | None = None) -> SfmResult:
    """图像目录 → 增量 SfM → 最大子模型。依赖 pycolmap（pip install pycolmap）。"""
    cb = on_progress or (lambda *a: None)
    images_dir, project_dir = Path(images_dir), Path(project_dir)
    try:
        import pycolmap
    except ImportError as e:
        raise RuntimeError("缺少 pycolmap：pip install pycolmap") from e

    colmap_dir = project_dir / "colmap"
    db = colmap_dir / "db.db"
    sparse_out = colmap_dir / "sparse"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    if sparse_out.exists():
        shutil.rmtree(sparse_out)
    sparse_out.mkdir(parents=True)

    cb(0.2, "特征提取…")
    pycolmap.extract_features(db_path=db, image_path=images_dir,
                              camera_mode=pycolmap.CameraMode.SINGLE)
    cb(0.5, "特征匹配…")
    pycolmap.match_exhaustive(db_path=db)
    cb(0.7, "增量重建…")
    maps = pycolmap.incremental_mapping(db_path=db, image_path=images_dir,
                                        output_path=sparse_out)

    best, best_n = None, -1
    for d in sorted(sparse_out.glob("[0-9]*")):
        rec = pycolmap.Reconstruction(d)
        if rec.num_reg_images() >= 5 and rec.num_reg_images() > best_n:
            best, best_n = d, rec.num_reg_images()
    if best is None:
        raise RuntimeError("SfM 无可用子模型（≥5 张配准）；检查采集质量/光照")
    cb(1.0, f"SfM 子模型 {best.name}：{best_n} 张配准")
    return SfmResult(images_dir=images_dir, sparse_dir=best,
                     frames=len(list(images_dir.glob("*.jpg"))),
                     registered=best_n)
