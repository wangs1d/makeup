"""capture — 引导用户完成"环绕拍摄"式脸部视频采集。

重建质量的上游决定因素是采集质量：人脸足够大、光照均匀、头部左右转动覆盖
（左/正/右三个偏航区间都有足够时长）。本模块只做纯逻辑（帧进 → 指导状态出 +
写 MP4），摄像头读取由调用方（Qt CameraWorker 或 CLI）负责，便于测试与
未来搬到浏览器端（getUserMedia + 同一套判定规则跑在服务端）。
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# 偏航覆盖区间（度）：ooosplat/COLMAP 对环绕序列最稳的是 ±60° 缓慢转动。
# 区间间不留死区（旧版 ±15~25 之间不计时时，用户容易一直停在里面，center 秒数为 0）
YAW_BUCKETS = {"left": (-90.0, -16.0), "center": (-16.0, 16.0), "right": (16.0, 90.0)}
BUCKET_MIN_SECONDS = 2.0        # 每个区间至少累计采集时长
FACE_MIN_WIDTH_RATIO = 0.22     # 人脸宽 / 帧宽 下限（太小重建精度差）
LUMA_RANGE = (60.0, 200.0)      # 可用光照的灰度均值范围
MAX_SECONDS = 60.0              # 单次采集上限（防止无限录制）


@dataclass
class CaptureGuidance:
    """单帧指导状态（UI 直接渲染）。"""
    ok: bool
    messages: list[str]                 # 需要纠正的问题（空列表=可以拍）
    bucket: str | None                  # 当前偏航区间 left/center/right
    bucket_seconds: dict[str, float]    # 各区间已累计的有效时长
    total_seconds: float
    done: bool
    face_box: tuple[float, float, float, float] | None = None   # 脸框 x0,y0,x1,y1（像素）
    yaw: float = 0.0                    # 头部偏航（度，负=向左）
    pitch: float = 0.0                  # 头部俯仰（度）

    @property
    def progress(self) -> float:
        have = sum(min(v, BUCKET_MIN_SECONDS) for v in self.bucket_seconds.values())
        return min(1.0, have / (BUCKET_MIN_SECONDS * len(YAW_BUCKETS)))


@dataclass
class CaptureResult:
    video_path: Path
    frames: int
    duration: float
    bucket_seconds: dict[str, float]
    guidance_frames: int = 0


def _bucket_of(yaw: float) -> str | None:
    for name, (lo, hi) in YAW_BUCKETS.items():
        if lo <= yaw <= hi:
            return name
    return None


def _open_writer(path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """MP4 写出器：优先 H.264（avc1，同码率下清晰度远高于 mp4v），不可用则回退。"""
    for cc in ("avc1", "H264", "mp4v"):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cc), fps, size)
        if vw.isOpened():
            return vw
        vw.release()
    raise RuntimeError(f"无法创建视频文件: {path}")


def open_camera(cam_id: int = 0,
                preferred: tuple[tuple[int, int], ...] = ((3840, 2160), (2560, 1440),
                                                          (1920, 1080), (1280, 720))):
    """打开摄像头并按清晰度优先级请求分辨率：驱动不接受则逐级回退到实际支持的最大档。

    Windows 优先 DSHOW 后端：MSMF 对部分设备会停在半开状态（选流失败），
    之后读取/释放都会异常甚至原生崩溃；DSHOW 打不开再回退 MSMF。
    """
    backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32"
                else [cv2.CAP_ANY])
    cap = None
    for backend in backends:
        candidate = cv2.VideoCapture(cam_id, backend)
        if candidate.isOpened() and candidate.read()[0]:
            cap = candidate
            break
        candidate.release()
    if cap is None:
        return None
    for w, h in preferred:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ok, frame = cap.read()
        if ok and frame.shape[1] >= int(w * 0.9):    # 驱动接受了该档位
            return cap
    return cap                                       # 各档都不达预期 → 用相机默认分辨率


class OrbitCaptureSession:
    """消费摄像头帧，输出实时指导并录制有效片段。

    tracker: 提供 detect(frame_bgr, timestamp_ms) -> {"pose": (yaw, pitch, roll), ...} | None
    的对象（makeupstudio.tracker.FaceTracker 兼容）；测试可注入假 tracker。
    """

    def __init__(self, out_path: str | Path, tracker=None, fps: float = 25.0,
                 frame_size: tuple[int, int] = (1280, 720)):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker
        self.fps = fps
        self.frame_size = frame_size
        self._writer = _open_writer(self.out_path, fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {self.out_path}")
        self._bucket_seconds = {k: 0.0 for k in YAW_BUCKETS}
        self._total = 0.0
        self._frames = 0
        self._t_last: float | None = None
        self._last_yaw = 0.0

    # ---------------- 主入口 ----------------

    def process(self, frame_bgr: np.ndarray, timestamp_ms: float | None = None) -> CaptureGuidance:
        """处理一帧：判定质量 → 有效则写入视频 → 返回指导状态。"""
        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000.0
        dt = 0.0 if self._t_last is None else (timestamp_ms - self._t_last) / 1000.0
        self._t_last = timestamp_ms

        msgs: list[str] = []
        det = None
        face_box: tuple[float, float, float, float] | None = None
        yaw = self._last_yaw
        pitch = 0.0
        if self.tracker is not None:
            try:
                det = self.tracker.detect(frame_bgr, timestamp_ms)
            except Exception:
                det = None
        if det is None:
            msgs.append("未检测到人脸，请正对摄像头")
        else:
            yaw, pitch, _ = det["pose"]
            self._last_yaw = yaw
            h, w = frame_bgr.shape[:2]
            px = det["px"]                                # 像素坐标 (478,2)
            face_box = (float(px[:, 0].min()), float(px[:, 1].min()),
                        float(px[:, 0].max()), float(px[:, 1].max()))
            face_w = float(px[:, 0].max() - px[:, 0].min())
            if face_w < FACE_MIN_WIDTH_RATIO * w:
                msgs.append("请靠近一点，让脸占画面 1/3 以上")
            if not (-20.0 <= pitch <= 25.0):
                msgs.append("请平视摄像头，不要低头/仰头")

        luma = float(frame_bgr.mean())
        if luma < LUMA_RANGE[0]:
            msgs.append("环境太暗，请增加光照")
        elif luma > LUMA_RANGE[1]:
            msgs.append("环境过曝，请减弱直射光")

        bucket = _bucket_of(yaw)
        acceptable = not msgs
        if acceptable and bucket is not None and 0.0 < dt < 0.5:
            self._bucket_seconds[bucket] += dt
            self._total += dt
            resized = frame_bgr if frame_bgr.shape[1::-1] == self.frame_size else \
                cv2.resize(frame_bgr, self.frame_size)
            self._writer.write(resized)
            self._frames += 1

        if self._total >= MAX_SECONDS:
            msgs = ["时长已足够，可以停止了"]
        done = self._coverage_complete() or self._total >= MAX_SECONDS
        if acceptable and not done:
            msgs = self._missing_bucket_hint()
        return CaptureGuidance(
            ok=acceptable, messages=msgs, bucket=bucket,
            bucket_seconds=dict(self._bucket_seconds),
            total_seconds=self._total, done=done,
            face_box=face_box, yaw=float(yaw), pitch=float(pitch))

    def finish(self) -> CaptureResult:
        self._writer.release()
        # 实际写入帧率 = 有效帧数 / 有效时长（被拒写的帧不进视频）。与标称 fps
        # 差异大时按真实时长重设时间轴，否则视频被“压缩”，头转动速度失真，
        # COLMAP 序列匹配与 landmark 三角化都会受影响。
        try:
            real_fps = self._frames / self._total if self._total > 0 else self.fps
            if self._frames >= 2 and abs(real_fps - self.fps) / self.fps > 0.15 \
                    and self.out_path.exists():
                tmp = self.out_path.with_suffix(".retime.mp4")
                cap = cv2.VideoCapture(str(self.out_path))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vw = _open_writer(tmp, real_fps, (w, h))
                while vw.isOpened():
                    ok, fr = cap.read()
                    if not ok:
                        break
                    vw.write(fr)
                cap.release()
                vw.release()
                if tmp.exists() and tmp.stat().st_size > 0:
                    tmp.replace(self.out_path)
                else:
                    tmp.unlink(missing_ok=True)
        except Exception:
            pass                                              # 重定时失败不影响采集产物
        report = {
            "frames": self._frames,
            "duration": round(self._total, 2),
            "bucket_seconds": {k: round(v, 2) for k, v in self._bucket_seconds.items()},
            "fps": self.fps,
            "video_fps": round(real_fps, 2) if self._total > 0 else self.fps,
            "size": list(self.frame_size),
        }
        self.out_path.with_suffix(".report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return CaptureResult(video_path=self.out_path, frames=self._frames,
                             duration=self._total, bucket_seconds=dict(self._bucket_seconds))

    # ---------------- 内部 ----------------

    def _coverage_complete(self) -> bool:
        return all(v >= BUCKET_MIN_SECONDS for v in self._bucket_seconds.values())

    def _missing_bucket_hint(self) -> list[str]:
        missing = [n for n, v in self._bucket_seconds.items() if v < BUCKET_MIN_SECONDS]
        if not missing:
            return []
        zh = {"left": "向左转头", "center": "回正", "right": "向右转头"}
        return ["请缓慢" + "、".join(zh[m] for m in missing) + "，保持匀速"]


def capture_from_camera(out_path: str | Path, cam_id: int = 0,
                        on_guidance: Callable[[CaptureGuidance, np.ndarray], bool] | None = None,
                        tracker=None) -> CaptureResult:
    """一体化采集：打开摄像头循环处理，直到覆盖完成 / 调用方回调返回 False / 按 q。"""
    if tracker is None:
        from ..tracker import FaceTracker
        try:
            tracker = FaceTracker()
        except Exception:
            tracker = None
    cap = open_camera(cam_id)                        # 4K 优先，逐级回退到相机支持的最大档
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {cam_id}")
    session: OrbitCaptureSession | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if session is None:
                h, w = frame.shape[:2]
                session = OrbitCaptureSession(out_path, tracker=tracker, fps=25.0, frame_size=(w, h))
            g = session.process(frame)
            if on_guidance is not None and not on_guidance(g, frame):
                break
            if g.done:
                break
            try:                                 # CLI 也给画面内引导（惰性导入避免环）
                from ..guide_overlay import draw_capture_guidance
                cv2.imshow("3DGS 采集引导", draw_capture_guidance(frame, g))
            except cv2.error:
                pass                             # 无显示环境（CI / 远程）
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if session is None:
            raise RuntimeError("摄像头未产出任何帧")
        return session.finish()
