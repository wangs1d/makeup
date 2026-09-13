"""reconstruct — 视频 → 3DGS 重建编排（ooosplat 同款管线）。

项目目录布局与 ooosplat 一致，便于互操作/排查：
    <project>/images/frame_00001.jpg      FFmpeg 抽帧
    <project>/colmap/db.db                COLMAP 特征库
    <project>/colmap/sparse/0/*.bin       相机内外参（供 isolate 做重投影）
    <project>/final.ply                   Brush 训练产物

后端抽象：LocalEngineBackend 调本机引擎；服务器部署时实现 RemoteBackend
（同一 run() 契约：提交视频 → 轮询 → 取回 final.ply + colmap 相机），
capture / isolate / fit_makeup 完全不用改。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .engines import find_engine

QUALITY = {
    # 抽帧密度（每秒帧数）与 Brush 训练迭代数——ooosplat 同名档位的本地近似
    "draft": {"fps": 5.0, "iters": 6000, "max_frames": 150},
    "standard": {"fps": 10.0, "iters": 15000, "max_frames": 300},
    "high": {"fps": 15.0, "iters": 30000, "max_frames": 500},
}

ProgressCB = Callable[[str, float, str], None]   # (stage, 0..1, message)


class ReconstructionError(RuntimeError):
    pass


@dataclass
class ReconJob:
    video_path: Path
    project_dir: Path
    quality: str = "standard"

    def __post_init__(self):
        if self.quality not in QUALITY:
            raise ValueError(f"quality 须为 {tuple(QUALITY)}，收到 {self.quality!r}")
        self.video_path = Path(self.video_path)
        self.project_dir = Path(self.project_dir)


@dataclass
class ReconResult:
    project_dir: Path
    ply_path: Path
    sparse_dir: Path          # colmap/sparse/0（isolate 需要）
    images_dir: Path
    seconds: float
    frames: int = 0


class ReconstructionBackend(ABC):
    """重建后端契约：本地引擎或远程服务实现同一 run()。"""

    @abstractmethod
    def run(self, job: ReconJob, on_progress: ProgressCB | None = None) -> ReconResult: ...

    @abstractmethod
    def availability(self) -> dict[str, bool]: ...


class LocalEngineBackend(ReconstructionBackend):
    """直接编排 FFmpeg → COLMAP → Brush（引擎发现规则见 engines.py）。

    命令模板可通过环境变量覆盖（空格分隔）：
        OOOSPLAT_COLMAP_MAPPER_ARGS / OOOSPLAT_BRUSH_TRAIN_ARGS
    """
    timeout_s = 3600.0

    def availability(self) -> dict[str, bool]:
        # ffmpeg 可选：缺失时用 OpenCV 内置解码抽帧
        return {n: find_engine(n) is not None for n in ("colmap", "brush")}

    # ---------------- 管线 ----------------

    def run(self, job: ReconJob, on_progress: ProgressCB | None = None) -> ReconResult:
        cb = on_progress or (lambda *a: None)
        t0 = time.time()
        avail = self.availability()
        missing = [k for k, v in avail.items() if not v]
        if missing:
            raise ReconstructionError(
                f"缺少引擎: {missing}。安装 OOOSplat 桌面版或设置 OOOSPLAT_ENGINE_DIR / "
                f"OOOSPLAT_COLMAP 等环境变量后重试。")
        p = job.project_dir
        images = p / "images"
        colmap_dir = p / "colmap"
        sparse = colmap_dir / "sparse" / "0"
        ply = p / "final.ply"
        for d in (images, colmap_dir):
            d.mkdir(parents=True, exist_ok=True)
        # 重跑清理：残留的 db.db 会让 COLMAP 对已入库图片报 IMAGE_EXISTS，
        # 旧抽帧与新高 fps 帧混在同一目录 → 稀疏重建退化成几个点
        for stale in images.glob("frame_*.jpg"):
            stale.unlink()
        for stale in (colmap_dir / "db.db",):
            if stale.exists():
                stale.unlink()
        if (colmap_dir / "sparse").exists():
            shutil.rmtree(colmap_dir / "sparse")
        if ply.exists():
            ply.unlink()
        shutil.copy2(job.video_path, p / "video.mp4")

        q = QUALITY[job.quality]
        frames = self._extract(job.video_path, images, q["fps"], cb)
        if frames < 10:
            raise ReconstructionError(f"抽帧仅得 {frames} 帧，视频太短或损坏")
        self._colmap(images, colmap_dir, sparse, cb)
        self._prune_sparse_points(sparse, cb)
        self._brush(colmap_dir, images, ply, q["iters"], cb)
        if not ply.exists():
            raise ReconstructionError("Brush 未产出 final.ply")
        cb("done", 1.0, f"重建完成，用时 {time.time() - t0:.0f}s")
        return ReconResult(project_dir=p, ply_path=ply, sparse_dir=sparse,
                           images_dir=images, seconds=time.time() - t0, frames=frames)

    # ---------------- 各阶段 ----------------

    def _extract(self, video: Path, images: Path, fps: float, cb) -> int:
        ffmpeg = find_engine("ffmpeg")
        if ffmpeg is not None:
            cb("extract", 0.0, f"FFmpeg 抽帧 {fps} fps")
            self._run([ffmpeg, "-y", "-i", video, "-vf", f"fps={fps}",
                       "-q:v", "1", str(images / "frame_%05d.jpg")],
                      cb, "extract", 0.5)
        else:
            # 回退：OpenCV 自带 FFmpeg 解码，抽帧无需独立 ffmpeg 引擎
            cb("extract", 0.0, f"OpenCV 抽帧 {fps} fps（未找到独立 ffmpeg）")
            import cv2
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise ReconstructionError(f"无法打开视频 {video}")
            step = 1.0 / fps
            n = 0
            next_t, i = 0.0, 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = i / cap.get(cv2.CAP_PROP_FPS)
                if t + 1e-6 >= next_t:
                    cv2.imwrite(str(images / f"frame_{n + 1:05d}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 98])
                    n += 1
                    next_t += step
                i += 1
            cap.release()
        frames = len(list(images.glob("frame_*.jpg")))
        cb("extract", 1.0, f"抽帧完成 {frames} 帧")
        return frames

    def _colmap(self, images: Path, colmap_dir: Path, sparse: Path, cb) -> None:
        db = colmap_dir / "db.db"
        colmap = find_engine("colmap")
        cb("colmap", 0.05, "COLMAP 特征提取")
        # 纯 CPU SIFT（use_gpu=0）；estimate_affine_shape/DSP 在 CPU 上慢一个量级且
        # 对验证无增益，默认不开（可用 OOOSPLAT_COLMAP_EXTRACT_ARGS 覆盖）。
        # 3.x 旗标为 --SiftExtraction.use_gpu，4.x 改名 --FeatureExtraction.*；先新后旧
        rc = self._run([colmap, "feature_extractor",
                        "--database_path", db, "--image_path", images,
                        "--ImageReader.camera_model", "OPENCV",
                        "--ImageReader.single_camera", "1",
                        "--FeatureExtraction.use_gpu", "0"], cb, "colmap", 0.2,
                       check=False)
        if rc != 0:
            self._run([colmap, "feature_extractor",
                       "--database_path", db, "--image_path", images,
                       "--ImageReader.camera_model", "OPENCV",
                       "--ImageReader.single_camera", "1",
                       "--SiftExtraction.use_gpu", "0"], cb, "colmap", 0.2)
        cb("colmap", 0.35, "COLMAP 特征匹配")
        # 3.x 旗标为 --SiftMatching.use_gpu，4.x 改名 --FeatureMatching.*；先新后旧
        flag_new, flag_old = "--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"
        # 人脸特写序列纹理弱，序列匹配的图太稀会导致 mapper 只长出几个点。
        # 帧数 ≤120 时用穷举匹配（CPU 可承受），显著提高初始化与三角化成功率。
        n_frames = len(list(images.glob("frame_*.jpg")))
        matcher = "exhaustive_matcher" if n_frames <= 120 else "sequential_matcher"
        rc = self._run([colmap, matcher, "--database_path", db, flag_new, "0"],
                       cb, "colmap", 0.4, check=False)
        if rc != 0:
            self._run([colmap, matcher, "--database_path", db, flag_old, "0"],
                      cb, "colmap", 0.45)
        if self._max_verification_inliers(db) == 0:
            # 环境韧性：个别 COLMAP 构建（本机实测 3.11/4.2 nocuda）匹配结果正常
            # 但几何验证全零。回退用 OpenCV RANSAC 重做验证并写回数据库。
            cb("colmap", 0.45, "COLMAP 验证为空 → OpenCV 几何验证回退")
            from .verify_cv2 import verify_matches_with_cv2
            n_ok = verify_matches_with_cv2(db)
            cb("colmap", 0.5, f"OpenCV 验证完成 {n_ok} 对")
        cb("colmap", 0.55, "COLMAP 稀疏重建")
        sparse.parent.mkdir(parents=True, exist_ok=True)   # COLMAP 4.2 要求目录已存在
        self._run(self._mapper_args(colmap, db, images, colmap_dir / "sparse"), cb, "colmap", 0.8)
        if not sparse.exists():
            # mapper 可能输出 sparse/1...：取含图像数最多的子目录
            best, best_n = None, -1
            for cand in sorted((colmap_dir / "sparse").glob("*")):
                img_bin = cand / "images.bin"
                if img_bin.exists():
                    n = img_bin.stat().st_size // 120    # 粗略代理：文件越大图越多
                    if n > best_n:
                        best, best_n = cand, n
            if best is not None and best.name != "0":
                shutil.copytree(best, sparse, dirs_exist_ok=True)
        if not (sparse / "images.bin").exists():
            raise ReconstructionError("COLMAP 重建失败（sparse/0 不完整）。"
                                      "建议：光照均匀、转动更慢、重拍。")
        cb("colmap", 1.0, "相机重建完成")

    @staticmethod
    def _max_verification_inliers(db_path: Path) -> int:
        """返回验证内点最大值；表不存在返回 -1（无法判断，不触发回退）。"""
        import sqlite3
        try:
            con = sqlite3.connect(db_path)
            best = -1
            for (blob,) in con.execute("SELECT rows FROM two_view_geometries"):
                best = max(best, 0)
                if isinstance(blob, bytes) and len(blob) // 4 > best:
                    best = len(blob) // 4
            con.close()
            return best
        except sqlite3.Error:
            return -1

    @staticmethod
    def _mapper_args(colmap, db, images, sparse_out) -> list:
        custom = os.environ.get("OOOSPLAT_COLMAP_MAPPER_ARGS")
        if custom:
            return [colmap, *custom.split()]
        # 关闭畸变与焦距自标定：人脸特写纹理弱、视场窄，BA 的焦距极易发散
        # （实测可优化到 27000px+，模型彻底不可用）；webcam 焦距用初始估计即可
        return [colmap, "mapper", "--database_path", db, "--image_path", images,
                "--output_path", sparse_out,
                "--Mapper.ba_refine_extra_params", "0",
                "--Mapper.ba_refine_focal_length", "0"]

    @staticmethod
    def _prune_sparse_points(sparse: Path, cb, keep_frac: float = 0.7) -> None:
        """Brush 前裁剪稀疏点：远离相机轨迹质心的背景点会把 Brush 的场景归一化
        搞崩（实测整场训练收敛成几个零透明度漂浮点）。保留最近 keep_frac 的点。"""
        import struct
        f = sparse / "points3D.bin"
        if not f.exists():
            return
        data = f.read_bytes()
        off = 0
        (n,) = struct.unpack_from("<Q", data, off)
        off += 8
        rows, tracks = [], []
        xyz_all = []
        for _ in range(n):
            pid, x, y, z, r, g, b, err = struct.unpack_from("<QdddBBBd", data, off)
            off += 43
            (tl,) = struct.unpack_from("<Q", data, off)
            off += 8
            track = data[off:off + 8 * tl]
            off += 8 * tl
            rows.append((pid, (x, y, z), (r, g, b), err))
            tracks.append((tl, track))
            xyz_all.append((x, y, z))
        if n < 50:
            return
        # COLMAP 世界系非度量：脸部点 = 离所有相机中心质心最近的那簇
        import numpy as np
        from . import colmap_io
        model = colmap_io.read_sparse(sparse)
        centers = []
        for im in model.images.values():
            R = colmap_io.quat_to_rotmat(im["qvec"])
            centers.append(-(R.T @ np.asarray(im["tvec"])))
        cc = np.mean(centers, axis=0)
        d = np.linalg.norm(np.asarray(xyz_all) - cc, axis=1)
        thr = np.quantile(d, keep_frac)
        out = bytearray(struct.pack("<Q", int((d <= thr).sum())))
        for row, dd, (tl, track) in zip(rows, d, tracks):
            if dd > thr or tl < 2:
                continue
            pid, (x, y, z), (r, g, b), err = row
            out += struct.pack("<QdddBBBd", pid, x, y, z, r, g, b, err)
            out += struct.pack("<Q", tl) + track
        f.write_bytes(bytes(out))

    def _brush(self, colmap_dir: Path, images: Path, ply: Path, iters: int, cb) -> None:
        brush = find_engine("brush")
        cb("train", 0.0, f"Brush 训练 {iters} 迭代")
        custom = os.environ.get("OOOSPLAT_BRUSH_TRAIN_ARGS")
        if custom:
            args = [brush, *custom.split()]
        else:
            # Brush v0.3 CLI：brush_app <数据集根> --total-steps N --export-path D --export-name F
            # 数据集根 = 含 images/ 与 colmap/sparse/0 的目录
            args = [brush, str(colmap_dir.parent), "--total-steps", str(iters),
                    "--export-path", str(ply.parent), "--export-name", ply.name,
                    "--export-every", str(iters)]
        self._run(args, cb, "train", 0.95, check=False)
        if not ply.exists():
            # 某些版本按 export_{iter}.ply 命名——回退取最新的 export_*.ply
            exports = sorted(ply.parent.glob("export_*.ply"),
                             key=lambda p: p.stat().st_mtime)
            if exports:
                shutil.move(str(exports[-1]), ply)
        if not ply.exists():
            raise ReconstructionError("Brush 训练未产出 final.ply（详见日志）")

    # ---------------- 工具 ----------------

    def _run(self, cmd: list, cb, stage: str, frac: float, check: bool = True) -> int:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        tail: list[str] = []
        try:
            for line in proc.stdout:            # type: ignore[union-attr]
                line = line.rstrip()
                if line:
                    tail.append(line)
                    if len(tail) > 40:
                        tail.pop(0)
                    cb(stage, frac, line[-160:])
            rc = proc.wait(timeout=self.timeout_s)
        except subprocess.TimeoutExpired as e:
            proc.kill()
            raise ReconstructionError(f"{stage} 超时") from e
        if check and rc != 0:
            log = "\n".join(tail[-12:])
            raise ReconstructionError(f"{stage} 失败（rc={rc}）：\n{log}")
        return rc

    @property
    def name(self) -> str:
        return "local-engines"


class RemoteBackend(ReconstructionBackend):
    """服务器版后端占位：部署阶段实现。

    契约：POST <server>/jobs 提交视频 → 返回 job_id；GET <server>/jobs/<id>
    轮询进度；GET <server>/jobs/<id>/final.ply 与 .../sparse.zip 取回产物。
    与 LocalEngineBackend 产出相同的 ReconResult，调用方无感切换。
    """

    def __init__(self, server: str):
        self.server = server.rstrip("/")

    def availability(self) -> dict[str, bool]:
        raise NotImplementedError("RemoteBackend 在服务器部署阶段接入")

    def run(self, job: ReconJob, on_progress: ProgressCB | None = None) -> ReconResult:
        raise NotImplementedError("RemoteBackend 在服务器部署阶段接入")


def run_reconstruction(video_path: str | Path, project_dir: str | Path,
                       quality: str = "standard",
                       backend: ReconstructionBackend | None = None,
                       on_progress: ProgressCB | None = None) -> ReconResult:
    backend = backend or LocalEngineBackend()
    return backend.run(ReconJob(Path(video_path), Path(project_dir), quality), on_progress)
