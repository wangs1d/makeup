"""face3dgs_tab — 主窗口"我的 3D 脸"视图。

流程：环绕采集（引导）→ 后台重建（引擎缺失时给出安装指引）→ 隔离脸部 →
套用当前妆容。所有耗时步骤在 QThread 中执行，UI 只收信号。

布局：左侧预览画布，右侧一张步骤卡片（三步走：采集 → 重建 → 贴妆），
每步一枚状态圆点 + 标题 + 一行状态文字 + 操作按钮。
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QVBoxLayout, QWidget)

from .face3dgs import engines
from .face3dgs.capture import OrbitCaptureSession, open_camera
from .guide_overlay import draw_capture_guidance

OUT_DIR = Path(__file__).resolve().parents[2] / "out" / "face3dgs"   # 与 CLI 共享（仓库根 out/）
VIEWER_SPLAT_DIR = Path(__file__).resolve().parents[1] / "out" / "splat"


class _FakeTracker:
    """tracker 不可用时的降级：仅录光照/时长，不做姿态判定。"""
    detect = None


class CaptureWorker(QThread):
    guidance = Signal(np.ndarray, object)     # frame, CaptureGuidance
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, out_path: Path, parent=None):
        super().__init__(parent)
        self.out_path = out_path
        self._run = True

    def stop(self):
        self._run = False

    def run(self):
        try:
            from .tracker import FaceTracker
            try:
                tracker: object | None = FaceTracker()
            except Exception:
                tracker = None
            cap = open_camera(0)                     # 4K 优先，逐级回退到相机支持的最大档
            if cap is None or not cap.isOpened():
                self.failed.emit("无法打开摄像头")
                return
            session: OrbitCaptureSession | None = None
            while self._run:
                ok, frame = cap.read()
                if not ok:
                    break
                if session is None:
                    h, w = frame.shape[:2]
                    session = OrbitCaptureSession(self.out_path, tracker=tracker,
                                                  fps=25.0, frame_size=(w, h))
                g = session.process(frame)
                self.guidance.emit(draw_capture_guidance(frame, g), g)
                if g.done:
                    break
            cap.release()
            self.done.emit(str(session.finish().video_path) if session else "")
        except Exception:
            self.failed.emit(traceback.format_exc())


class ReconWorker(QThread):
    stage = Signal(str, float, str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, video: str, project: Path, quality: str = "standard", parent=None):
        super().__init__(parent)
        self.video = video
        self.project = project
        self.quality = quality

    def run(self):
        try:
            from .reconstruct import run_reconstruction
            res = run_reconstruction(self.video, self.project, self.quality,
                                     on_progress=lambda s, f, m: self.stage.emit(s, f, m))
            self.done.emit(str(res.ply_path))
        except Exception as e:
            self.failed.emit(str(e) or traceback.format_exc())


class FitWorker(QThread):
    done = Signal(str, str)      # ply_path, preview_png
    failed = Signal(str)

    def __init__(self, project: Path, spec: dict, intensity: float, parent=None):
        super().__init__(parent)
        self.project = project
        self.spec = spec
        self.intensity = intensity

    def run(self):
        try:
            from .fit_makeup import FaceMakeupFitter
            from .isolate import isolate_face
            from .reconstruct import ReconResult
            proj = self.project
            result = ReconResult(project_dir=proj, ply_path=proj / "final.ply",
                                 sparse_dir=proj / "colmap" / "sparse" / "0",
                                 images_dir=proj / "images", seconds=0.0)
            # 先裁出脸部点云：final.ply 含背景/肩颈，直接贴合会把妆上到背景上，
            # 预览也全是漂浮噪点。face.ply 比 final.ply 旧时自动重裁。
            face_ply = proj / "face.ply"
            if not face_ply.exists() or \
                    face_ply.stat().st_mtime < result.ply_path.stat().st_mtime:
                isolate_face(result, face_ply)
            fitter = FaceMakeupFitter()
            fit = fitter.fit(result, self.spec, OUT_DIR / "fitted",
                             face_ply=face_ply, intensity=self.intensity)
            self.done.emit(str(fit.ply_path), str(fit.previews[0]) if fit.previews else "")
        except Exception:
            self.failed.emit(traceback.format_exc())


class Face3DgsTab(QWidget):
    """预览画布（嵌入主内容区）；步骤控件经 build_panel() 嵌入左侧栏。

    流程：采集 → 重建 → 贴妆。所有耗时步骤在 QThread 中执行，UI 只收信号。
    """

    STEPS = [
        ("环绕采集", "正对摄像头，缓慢左右转头"),
        ("重建点云", "从采集视频重建 3D 高斯点云"),
        ("贴合妆容", "把当前妆容贴到你的 3D 脸"),
    ]

    def __init__(self, get_layers, get_intensity, parent=None):
        super().__init__(parent)
        self.get_layers = get_layers          # () -> spec layers（当前妆容）
        self.get_intensity = get_intensity    # () -> float
        self.on_published = None              # 贴妆发布成功后的回调（启用查看器按钮）
        self.capture_worker: CaptureWorker | None = None
        self.recon_worker: ReconWorker | None = None
        self.fit_worker: FitWorker | None = None
        self._active_step = -1
        self._build_ui()
        self.build_panel()                    # 先构建，保证引用存在（稍后由主窗口取走）

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        self.preview = QLabel("对准摄像头完成一次环绕采集，\n即可生成你的 3D 脸并试妆")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setObjectName("videoArea")
        v.addWidget(self.preview)

    def build_panel(self, on_published=None) -> QWidget:
        """构建左侧栏的步骤卡片（采集 / 重建 / 贴妆）。"""
        if on_published is not None:
            self.on_published = on_published
        if hasattr(self, "panel"):
            return self.panel

        self.panel = QWidget()
        pv = QVBoxLayout(self.panel)
        pv.setContentsMargins(0, 0, 0, 0)

        card = QFrame()
        card.setObjectName("sideCard")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(14)

        self.engine_hint = QLabel(self._engine_hint())
        self.engine_hint.setObjectName("hint")
        self.engine_hint.setWordWrap(True)
        v.addWidget(self.engine_hint)

        self.step_dots: list[QLabel] = []
        self.step_status: list[QLabel] = []
        self.buttons: list[QPushButton] = []
        labels = ["开始采集", "开始重建", "开始贴妆"]
        for i, (title, desc) in enumerate(self.STEPS):
            row = QHBoxLayout()
            row.setSpacing(12)
            dot = QLabel(str(i + 1))
            dot.setObjectName("stepDot")
            dot.setAlignment(Qt.AlignCenter)
            dot.setFixedSize(26, 26)
            dot.setProperty("state", "pending")
            row.addWidget(dot, 0, Qt.AlignTop)
            self.step_dots.append(dot)

            text = QVBoxLayout()
            text.setSpacing(2)
            title_label = QLabel(title)
            title_label.setObjectName("stepTitle")
            text.addWidget(title_label)
            status = QLabel(desc)
            status.setObjectName("stepStatus")
            status.setWordWrap(True)
            text.addWidget(status)
            self.step_status.append(status)
            row.addLayout(text, 1)

            btn = QPushButton(labels[i])
            btn.setCursor(Qt.PointingHandCursor)
            btn.setEnabled(i == 0)
            if i == 0:
                btn.setProperty("primary", True)
                btn.clicked.connect(self._toggle_capture)
            elif i == 1:
                btn.clicked.connect(self._start_recon)
            else:
                btn.clicked.connect(self._start_fit)
            row.addWidget(btn, 0, Qt.AlignTop)
            self.buttons.append(btn)
            v.addLayout(row)
        self.btn_capture, self.btn_rebuild, self.btn_fit = self.buttons

        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.hide()
        v.addWidget(self.bar)
        v.addStretch(1)
        pv.addWidget(card)
        return self.panel

    @staticmethod
    def _engine_hint() -> str:
        st = engines.status_all()
        missing = [n for n, s in st.items() if not s.ok]
        if not missing:
            return "重建引擎就绪（FFmpeg / COLMAP / Brush）"
        return "重建引擎缺失: " + "、".join(missing) + " — 安装 OOOSplat 桌面版（自带全部引擎）" \
            "后重启，或设置 OOOSPLAT_ENGINE_DIR 指向引擎目录。"

    # ---------------- 步骤状态 ----------------

    def _set_step(self, i: int, state: str, status: str | None = None, err: bool = False):
        """state: pending / active / done。"""
        dot = self.step_dots[i]
        dot.setProperty("state", state)
        dot.setText("✓" if state == "done" else str(i + 1))
        dot.style().unpolish(dot)
        dot.style().polish(dot)
        if status is not None:
            lab = self.step_status[i]
            lab.setText(status)
            lab.setProperty("err", err)
            lab.style().unpolish(lab)
            lab.style().polish(lab)
        if state == "active":
            self._active_step = i

    def _set_busy(self, busy: bool, indeterminate: bool = False):
        if busy:
            if indeterminate:
                self.bar.setRange(0, 0)
            else:
                self.bar.setRange(0, 100)
            self.bar.show()
        else:
            self.bar.setRange(0, 100)
            self.bar.setValue(0)
            self.bar.hide()

    # ---------------- 采集 ----------------

    def _toggle_capture(self):
        if self.capture_worker is not None:          # 正在采集 → 停止
            self.capture_worker.stop()
            self.capture_worker.wait(3000)
            self.capture_worker = None
            self.btn_capture.setText("开始采集")
            return
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.capture_worker = CaptureWorker(OUT_DIR / "capture.mp4")
        self.capture_worker.guidance.connect(self._on_capture_frame)
        self.capture_worker.done.connect(self._on_capture_done)
        self.capture_worker.failed.connect(self._on_failed)
        self.btn_capture.setText("停止采集")
        self._set_step(0, "active")
        self.capture_worker.start()

    def _on_capture_frame(self, frame: np.ndarray, g):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.preview.setPixmap(QPixmap.fromImage(img).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        msg = g.messages[0] if g.messages else "保持缓慢转头"
        self._set_step(0, "active",
                       f"{g.total_seconds:.1f}s · 左 {g.bucket_seconds['left']:.0f} "
                       f"正 {g.bucket_seconds['center']:.0f} 右 {g.bucket_seconds['right']:.0f} · {msg}")
        self.bar.setValue(int(g.progress * 100))
        if not self.bar.isVisible():
            self._set_busy(True)
        if g.done:
            self._toggle_capture()

    def _on_capture_done(self, video: str):
        self._set_step(0, "done", "采集完成")
        self._set_step(1, "active", "等待重建")
        self.btn_rebuild.setEnabled(True)
        self.btn_rebuild.setProperty("primary", True)
        self.btn_rebuild.style().unpolish(self.btn_rebuild)
        self.btn_rebuild.style().polish(self.btn_rebuild)
        self._set_busy(False)

    # ---------------- 重建 ----------------

    def _start_recon(self):
        st = engines.status_all()
        missing = [n for n, s in st.items() if not s.ok]
        if missing:
            self.engine_hint.setText(self._engine_hint())
            return
        self.btn_rebuild.setEnabled(False)
        self._set_step(1, "active", "正在重建…")
        self._set_busy(True)
        self.recon_worker = ReconWorker(str(OUT_DIR / "capture.mp4"), OUT_DIR / "proj")
        self.recon_worker.stage.connect(self._on_recon_stage)
        self.recon_worker.done.connect(self._on_recon_done)
        self.recon_worker.failed.connect(self._on_failed)
        self.recon_worker.start()

    def _on_recon_stage(self, stage: str, frac: float, msg: str):
        self.bar.setValue(int(frac * 100))
        self._set_step(1, "active", f"[{stage}] {msg[:60]}")

    def _on_recon_done(self, ply: str):
        self._set_step(1, "done", "重建完成")
        self._set_step(2, "active", "等待贴妆")
        self.btn_fit.setEnabled(True)
        self.btn_fit.setProperty("primary", True)
        self.btn_fit.style().unpolish(self.btn_fit)
        self.btn_fit.style().polish(self.btn_fit)
        self._set_busy(False)

    # ---------------- 贴合 ----------------

    def _start_fit(self):
        spec_layers = self.get_layers()
        spec = {"layers": spec_layers}
        self.btn_fit.setEnabled(False)
        self._set_step(2, "active", "正在贴合…")
        self._set_busy(True, indeterminate=True)
        self.fit_worker = FitWorker(OUT_DIR / "proj", spec, self.get_intensity())
        self.fit_worker.done.connect(self._on_fit_done)
        self.fit_worker.failed.connect(self._on_failed)
        self.fit_worker.start()

    def _on_fit_done(self, ply: str, preview: str):
        ok = self._publish_user_splat(Path(ply).with_suffix(".splat"))
        self._set_step(2, "done",
                       "已发布到「妆效预览」" if ok else "贴合完成（发布查看器失败）",
                       err=not ok)
        self.btn_fit.setEnabled(True)
        self._set_busy(False)
        if preview:
            self.preview.setPixmap(QPixmap(preview).scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _publish_user_splat(self, splat: Path) -> bool:
        """把贴合结果发布到「妆效预览」查看器：复制为 out/splat/user.splat 并在
        version.json 登记 user 条目（viewer 轮询到新版本后可点「我的 3D 脸」）。"""
        try:
            import shutil
            VIEWER_SPLAT_DIR.mkdir(parents=True, exist_ok=True)
            dst = VIEWER_SPLAT_DIR / "user.splat"
            shutil.copyfile(splat, dst)
            ver_path = VIEWER_SPLAT_DIR / "version.json"
            try:
                prev = json.loads(ver_path.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
            prev.setdefault("version", str(int(time.time())))
            prev["ready"] = True
            prev["user"] = {"version": str(int(time.time())),
                            "file": "user.splat",
                            "points": dst.stat().st_size // 32}
            ver_path.write_text(json.dumps(prev), encoding="utf-8")
        except Exception:
            return False
        if self.on_published is not None:
            self.on_published()
        return True

    def _on_failed(self, msg: str):
        if self._active_step >= 0:
            self._set_step(self._active_step, "active",
                           f"失败：{msg.splitlines()[-1][:80] if msg else '未知错误'}", err=True)
        self._set_busy(False)
        self.btn_capture.setText("开始采集")
        self.btn_rebuild.setEnabled(True)
        self.btn_fit.setEnabled(True)
        self.capture_worker = None
