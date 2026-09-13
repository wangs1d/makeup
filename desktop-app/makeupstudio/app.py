#!/usr/bin/env python3
"""app — MakeupStudio 主窗口（PySide6）。

布局（三层结构，追求简洁）：
  顶栏：应用名 + 状态；
  左侧栏：三个视图切换按钮（实时化妆镜 / 妆效预览 / 我的 3D 脸）+
          当前视图的控制项（妆容库 / 查看器控制 / 3D 脸三步流程）；
  内容区：QStackedWidget 承载三个视图。
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (QMutex, QSize, QThread, QTimer, QUrl, Qt,
                            QWaitCondition, Signal)
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QDialog, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QSlider, QStackedWidget,
                               QVBoxLayout, QWidget)

from .coach import speak, steps_from_layers
from .compositor import WebcamCompositor
from .face3dgs.capture import open_camera
from .server import ViewerServer
from .theme import apply_theme
from .tracker import FaceTracker

ROOT = Path(__file__).resolve().parent.parent.parent
REFS = ROOT / "makeup-skill" / "references"
PRESETS = ROOT / "makeup-skill" / "presets"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"

VIEWS = ["实时化妆镜", "妆效预览", "我的 3D 脸"]
ENVS = ["neutral", "warm", "cool", "dim"]
ENV_LABELS = [("neutral", "自然"), ("warm", "暖光"), ("cool", "冷光"), ("dim", "夜间")]


class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray, object, float)
    failed = Signal(str)

    def __init__(self, cam_id: int = 0, compositor=None, parse_worker=None, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.compositor = compositor              # 妆容合成在本线程跑，GUI 线程只负责绘制
        self.parse_worker = parse_worker
        self.intensity = 0.8                      # GUI 线程写 / 本线程读（GIL 下属性赋值原子）
        self._pending_look: list[dict] | None = None
        self._pending_env: str | None = None
        self._run = True
        self.tracker = None

    def set_look(self, layers: list[dict]):
        self._pending_look = list(layers)         # 下帧循环里在相机线程应用（免 GUI 卡顿/竞态）

    def set_env(self, env: str):
        self._pending_env = env

    def run(self):
        # 化妆镜是实时预览：请求 1080p 平衡清晰度与帧率（合成/追踪耗时随分辨率上升）
        cap = open_camera(self.cam_id, preferred=((1920, 1080), (1280, 720)))
        if cap is None or not cap.isOpened():
            self.failed.emit("无法打开摄像头（可能被其他程序占用）")
            return
        try:
            self.tracker = FaceTracker()
        except Exception as e:                       # mediapipe 不可用时降级为纯摄像头
            self.tracker = None
            self.tracker_error = str(e)
        t0 = time.time()
        fps, fps_t, fps_n = 0.0, time.time(), 0
        while self._run:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)               # 镜像：化妆镜体验
            if self._pending_look is not None:
                self.compositor.set_look(self._pending_look)
                self._pending_look = None
            if self._pending_env is not None:
                self.compositor.set_env(self._pending_env)
                self._pending_env = None
            track = None
            if self.tracker is not None:
                track = self.tracker.detect(frame, (time.time() - t0) * 1000.0)
            out = frame
            if track is not None:
                if self.parse_worker is not None:
                    self.parse_worker.submit(frame, track["px"])   # 内部会拷贝，须在合成前
                out = self.compositor.process(frame, track["px"], self.intensity)
            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps, fps_t, fps_n = fps_n / (time.time() - fps_t), time.time(), 0
            self.frame_ready.emit(out, track, fps)
        cap.release()
        if self.tracker is not None:
            self.tracker.close()

    def stop(self):
        self._run = False
        return self.wait(8000)     # 追踪器冷启动收尾可能耗时数秒，等它真正退出


class ParseWorker(QThread):
    """低频人脸解析：像素级蒙版（唇/皮肤…）供 compositor 贴合用。

    只保留最新一帧（解析比相机慢，排队没有意义）；模型惰性加载，
    不可用时静默退出（compositor 自动走 landmark 蒙版回退）。
    """
    parsed = Signal(object, object)               # masks, landmarks

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self._lock = QMutex()
        self._wake = QWaitCondition()
        self._run = True

    def submit(self, frame: np.ndarray, lm: np.ndarray):
        self._lock.lock()
        self._pending = (frame.copy(), lm.copy())   # 相机帧会被就地合成，必须拷贝
        self._wake.wakeOne()
        self._lock.unlock()

    def stop(self):
        self._run = False
        self._lock.lock()
        self._wake.wakeOne()
        self._lock.unlock()
        self.wait(3000)

    def run(self):
        try:
            from .parser import FaceParser
            parser = FaceParser()
            ok = parser.available()
        except Exception:
            ok = False
        if not ok:
            return
        while self._run:
            self._lock.lock()
            while self._run and self._pending is None:
                self._wake.wait(self._lock)
            job, self._pending = self._pending, None
            self._lock.unlock()
            if job is None:
                break
            frame, lm = job
            try:
                masks = parser.parse(frame)
            except Exception:
                masks = None
            if masks:
                self.parsed.emit(masks, lm)


class SplatWorker(QThread):
    done = Signal(str, float, int, int)              # version, seconds, n_makeup, n_bare
    failed = Signal(str)

    def __init__(self, out_dir: Path, parent=None):
        super().__init__(parent)
        self.out_dir = out_dir
        self.layers: list[dict] = []
        self.intensity = 0.8
        self.env = "neutral"
        self.dirty = False
        self._busy = False

    def submit(self, layers: list[dict], intensity: float, env: str):
        self.layers = copy.deepcopy(layers)
        self.intensity = float(intensity)
        self.env = env
        if self._busy:
            self.dirty = True
            return
        self.start()

    def run(self):
        self._busy = True
        try:
            while True:
                self._build()
                if not self.dirty:
                    break
                self.dirty = False
        finally:
            self._busy = False

    def _build(self):
        try:
            t0 = time.time()
            from .splat3d import SplatCloudBuilder
            builder = SplatCloudBuilder(env=self.env)
            splat_dir = self.out_dir / "splat"
            splat_dir.mkdir(parents=True, exist_ok=True)
            bare = builder.build([], intensity=0.0, n_base=60000, n_makeup=0)
            made = builder.build(self.layers, intensity=self.intensity,
                                 n_base=60000, n_makeup=35000)
            version = str(int(time.time()))
            SplatCloudBuilder.export_splat(made, splat_dir / "makeup.splat")
            SplatCloudBuilder.export_splat(bare, splat_dir / "bare.splat")
            SplatCloudBuilder.export_ply(made, splat_dir / "makeup.ply")
            # 保留 face3dgs_tab 发布的 user 条目（我的3D脸点云）
            ver_path = splat_dir / "version.json"
            try:
                prev = json.loads(ver_path.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
            doc = {"version": version, "ready": True}
            if prev.get("user"):
                doc["user"] = prev["user"]
            ver_path.write_text(json.dumps(doc), encoding="utf-8")
            self.done.emit(version, time.time() - t0, len(made["xyz"]), len(bare["xyz"]))
        except Exception as e:
            import traceback
            self.failed.emit(traceback.format_exc())


class TransferWorker(QThread):
    """妆效迁移后台线程（EleGANt 首次加载模型需数秒，不能卡 GUI）。"""
    done = Signal(str, str)                            # 对比图路径, 结果图路径
    failed = Signal(str)

    def __init__(self, source: str, reference: str, out_dir: Path, parent=None):
        super().__init__(parent)
        self.source = source
        self.reference = reference
        self.out_dir = out_dir

    def run(self):
        try:
            from .transfer import imread_unicode, transfer_and_save
            src = imread_unicode(self.source)
            ref = imread_unicode(self.reference)
            cmp_path, res_path = transfer_and_save(src, ref, self.out_dir)
            self.done.emit(str(cmp_path), str(res_path))
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, cam_id: int = 0):
        super().__init__()
        self.setWindowTitle("MakeupStudio")
        self.resize(1280, 800)

        self.compositor = WebcamCompositor(REFS / "landmark-regions.json")
        self.layers: list[dict] = []
        self.intensity = 0.8
        self.env = "neutral"
        self.last_track = None
        self.transfer_src: str | None = None           # 参考图迁移的两张输入图
        self.transfer_ref: str | None = None
        self.transfer_worker: TransferWorker | None = None
        self.web = None                                    # 惰性创建（首次切到妆效预览）

        self.server = ViewerServer(OUT_DIR, port=8791)
        self.server.start()
        self.splat_worker = SplatWorker(OUT_DIR)
        self.splat_worker.done.connect(self._on_splat_done)
        self.splat_worker.failed.connect(self._on_splat_failed)
        self._splat_timer = QTimer(self)
        self._splat_timer.setSingleShot(True)
        self._splat_timer.setInterval(800)
        self._splat_timer.timeout.connect(self._submit_splat)

        self.cam_id = cam_id
        self.cam: CameraWorker | None = None           # 按需启动：点「开启摄像头」才占用设备
        # 须先于 _build_ui/_load_presets：预设加载会触发 _on_look_changed 访问 self.cam

        self._build_ui()
        self._load_presets()

        self.parse_worker = ParseWorker(self)          # 像素级人脸解析（低频异步）
        self.parse_worker.parsed.connect(self.compositor.set_parse)

        # 陪练状态
        self.coach_steps: list[str] = []
        self.coach_idx = 0
        self.coach_timer = QTimer(self)
        self.coach_timer.setInterval(45000)
        self.coach_timer.timeout.connect(self._coach_next)

    # ---------------- UI ----------------

    def _build_ui(self):
        apply_theme(self)
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_live_page())
        self.stack.addWidget(QWidget())                    # 妆效预览
        from .face3dgs_tab import Face3DgsTab
        self.face3d_tab = Face3DgsTab(lambda: self.layers, lambda: self.intensity)
        self.stack.addWidget(self.face3d_tab)
        body.addWidget(self._build_sidebar())              # 左侧栏：随视图切换控制项
        body.addWidget(self.stack, 1)

        root.addLayout(body, 1)
        self.setCentralWidget(central)
        # Web 视图在启动时预创建：此刻摄像头必然未开启，避免与相机 GL/设备
        # 上下文冲突（摄像头线程存活时初始化 WebEngine 会原生崩溃）
        self._ensure_web()

    def _build_header(self) -> QWidget:
        """顶栏：应用名 + 状态（视图切换在左侧栏）。"""
        header = QWidget()
        header.setObjectName("header")
        h = QHBoxLayout(header)
        h.setContentsMargins(20, 10, 20, 10)
        h.setSpacing(10)
        title = QLabel("MakeupStudio")
        title.setObjectName("appTitle")
        h.addWidget(title)
        sub = QLabel("PC 化妆助手")
        sub.setObjectName("appSubtitle")
        h.addWidget(sub)
        h.addStretch(1)

        self.status_hint = QLabel("摄像头未开启 · 点击「开启摄像头」开始试妆")
        self.status_hint.setObjectName("hint")
        h.addWidget(self.status_hint)
        return header

    def _build_live_page(self) -> QWidget:
        """实时化妆镜：视频画布 + 陪练卡片。摄像头默认不开启，按需手动打开。"""
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        self.cam_area = QStackedWidget()
        holder = QWidget()
        hv = QVBoxLayout(holder)
        hv.setAlignment(Qt.AlignCenter)
        hv.setSpacing(16)
        self.cam_tip = QLabel("摄像头未开启\n打开后进入实时试妆")
        self.cam_tip.setAlignment(Qt.AlignCenter)
        self.cam_tip.setObjectName("hint")
        hv.addWidget(self.cam_tip)
        self.cam_toggle_btn = QPushButton("开启摄像头")
        self.cam_toggle_btn.setProperty("primary", True)
        self.cam_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.cam_toggle_btn.setFixedWidth(180)
        self.cam_toggle_btn.clicked.connect(self._toggle_camera)
        hv.addWidget(self.cam_toggle_btn, 0, Qt.AlignHCenter)
        self.cam_area.addWidget(holder)

        self.cam_label = QLabel("正在启动摄像头…")
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setMinimumSize(640, 480)
        self.cam_label.setObjectName("videoArea")
        self.cam_area.addWidget(self.cam_label)
        self.cam_area.setCurrentIndex(0)
        v.addWidget(self.cam_area, 1)

        card = QFrame()
        card.setObjectName("coachCard")
        h = QHBoxLayout(card)
        h.setContentsMargins(14, 9, 14, 9)
        h.setSpacing(12)
        self.coach_btn = QPushButton("开始陪练")
        self.coach_btn.setProperty("primary", True)
        self.coach_btn.setCursor(Qt.PointingHandCursor)
        self.coach_btn.clicked.connect(self._toggle_coach)
        h.addWidget(self.coach_btn)
        self.coach_label = QLabel("按当前妆容的步骤顺序计时提醒（可语音）")
        self.coach_label.setObjectName("hint")
        h.addWidget(self.coach_label, 1)
        self.coach_progress = QProgressBar()
        self.coach_progress.setMaximumWidth(160)
        self.coach_progress.setTextVisible(False)
        h.addWidget(self.coach_progress)
        v.addWidget(card)
        return page

    def _build_sidebar(self) -> QWidget:
        """左侧栏：顶部三个视图切换按钮 + 当前视图的控制项。"""
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(300)
        v = QVBoxLayout(side)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        nav = QWidget()
        nv = QVBoxLayout(nav)
        nv.setContentsMargins(12, 12, 12, 10)
        nv.setSpacing(3)
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        for i, name in enumerate(VIEWS):
            b = QPushButton(name)
            b.setObjectName("navBtn")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda checked=False, idx=i: self._switch_page(idx))
            self.view_group.addButton(b, i)
            nv.addWidget(b)
        self.view_group.button(0).setChecked(True)
        v.addWidget(nav)
        v.addWidget(self._hairline())

        self.side_stack = QStackedWidget()
        self.side_stack.addWidget(self._build_makeup_panel())
        self.side_stack.addWidget(self._build_viewer_panel())
        self.side_stack.addWidget(
            self.face3d_tab.build_panel(on_published=self._on_user_published))
        v.addWidget(self.side_stack, 1)
        return side

    def _build_makeup_panel(self) -> QWidget:
        """妆容控制（仅实时化妆镜视图）：妆容库 / 强度 / 环境光。"""
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(16, 14, 16, 12)
        v.setSpacing(10)

        v.addWidget(self._section_label("妆容库"))
        self.preset_list = QListWidget()
        self.preset_list.currentRowChanged.connect(self._apply_preset)
        v.addWidget(self.preset_list, 1)
        v.addWidget(self._hairline())

        v.addWidget(self._section_label("强度"))
        irow = QHBoxLayout()
        irow.setSpacing(10)
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(80)
        self.intensity_slider.valueChanged.connect(self._on_intensity)
        irow.addWidget(self.intensity_slider, 1)
        self.intensity_value = QLabel("0.80")
        self.intensity_value.setObjectName("hint")
        self.intensity_value.setFixedWidth(30)
        self.intensity_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        irow.addWidget(self.intensity_value)
        v.addLayout(irow)

        v.addWidget(self._section_label("环境光"))
        v.addWidget(self._segment([(value, label, value == self.env)
                                   for value, label in ENV_LABELS], self._on_env))
        v.addStretch(1)
        return page

    def _build_viewer_panel(self) -> QWidget:
        """妆效预览控制：视角 / 模式 / 导出（经 runJavaScript 驱动查看器）。"""
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(16, 14, 16, 12)
        v.setSpacing(10)

        v.addWidget(self._section_label("视角"))
        vrow = QHBoxLayout()
        vrow.setSpacing(6)
        for name, fn in [("正面", "front"), ("¾ 侧", "three"), ("侧面", "side")]:
            b = QPushButton(name)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda checked=False, f=fn: self._viewer_js(f"{f}()"))
            vrow.addWidget(b)
        v.addLayout(vrow)

        v.addWidget(self._section_label("模式"))
        mrow = QHBoxLayout()
        mrow.setSpacing(6)
        self.btn_spin = QPushButton("自动旋转")
        self.btn_spin.setCheckable(True)
        self.btn_spin.setCursor(Qt.PointingHandCursor)
        self.btn_spin.clicked.connect(
            lambda checked: self._viewer_js(f"viewer3d.spin({str(checked).lower()})"))
        mrow.addWidget(self.btn_spin)
        self.btn_split = QPushButton("素颜对比")
        self.btn_split.setCheckable(True)
        self.btn_split.setCursor(Qt.PointingHandCursor)
        self.btn_split.clicked.connect(
            lambda checked: self._viewer_js(f"viewer3d.split({str(checked).lower()})"))
        mrow.addWidget(self.btn_split)
        v.addLayout(mrow)

        v.addWidget(self._section_label("导出"))
        orow = QHBoxLayout()
        orow.setSpacing(6)
        b_snap = QPushButton("快照导出")
        b_snap.setCursor(Qt.PointingHandCursor)
        b_snap.clicked.connect(lambda: self._viewer_js("viewer3d.snap()"))
        orow.addWidget(b_snap)
        self.btn_user = QPushButton("我的 3D 脸")
        self.btn_user.setEnabled(False)                # 贴妆发布成功后启用
        self.btn_user.setCursor(Qt.PointingHandCursor)
        self.btn_user.clicked.connect(lambda: self._viewer_js("viewer3d.user()"))
        orow.addWidget(self.btn_user)
        v.addLayout(orow)

        v.addWidget(self._hairline())
        self._build_transfer_panel(v)

        hint = QLabel("拖动旋转 · 滚轮缩放 · 双击复位")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return page

    def _build_transfer_panel(self, v: QVBoxLayout):
        """参考图迁移：素颜照 + 真实妆效参考图 → EleGANt 迁移对比图。"""
        v.addWidget(self._section_label("参考图迁移"))
        prow = QHBoxLayout()
        prow.setSpacing(6)
        b_src = QPushButton("素颜照…")
        b_src.setCursor(Qt.PointingHandCursor)
        b_src.clicked.connect(lambda: self._pick_transfer_image("source"))
        prow.addWidget(b_src)
        b_ref = QPushButton("参考图…")
        b_ref.setCursor(Qt.PointingHandCursor)
        b_ref.clicked.connect(lambda: self._pick_transfer_image("reference"))
        prow.addWidget(b_ref)
        v.addLayout(prow)

        self.transfer_hint = QLabel("选素颜照与妆效参考图，AI 生成上妆对比")
        self.transfer_hint.setObjectName("hint")
        self.transfer_hint.setWordWrap(True)
        v.addWidget(self.transfer_hint)

        self.transfer_run = QPushButton("开始迁移")
        self.transfer_run.setProperty("primary", True)
        self.transfer_run.setCursor(Qt.PointingHandCursor)
        self.transfer_run.clicked.connect(self._start_transfer)
        v.addWidget(self.transfer_run)
        v.addStretch(1)

    def _segment(self, options, on_pick) -> QWidget:
        """胶囊分段控件；options: [(value, label, checked), ...]。"""
        seg = QWidget()
        seg.setObjectName("segment")
        lay = QHBoxLayout(seg)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        group = QButtonGroup(self)
        group.setExclusive(True)
        for value, label, checked in options:
            b = QPushButton(label)
            b.setObjectName("segBtn")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setChecked(checked)
            b.clicked.connect(lambda checked=False, val=value: on_pick(val))
            group.addButton(b)
            lay.addWidget(b)
        return seg

    def _viewer_js(self, script: str):
        """在查看器页面执行一段 viewer3d.* 调用（WebEngine 内嵌时生效）。"""
        if self.web is not None and hasattr(self.web, "page"):
            self.web.page().runJavaScript(script)

    def _on_user_published(self):
        self.btn_user.setEnabled(True)

    @staticmethod
    def _section_label(text: str) -> QLabel:
        t = QLabel(text)
        t.setObjectName("sectionTitle")
        return t

    @staticmethod
    def _hairline() -> QFrame:
        line = QFrame()
        line.setObjectName("hairline")
        line.setFrameShape(QFrame.NoFrame)
        line.setFixedHeight(1)
        return line

    # ---------------- 视图切换 ----------------

    def _switch_page(self, idx):
        if idx != 0:
            self._stop_camera()            # 离开化妆镜先释放设备（Web 视图/3D 采集也要用）
        if idx == 1:
            self._ensure_web()
        self.view_group.button(idx).setChecked(True)   # 程序化切页时同步高亮
        self.stack.setCurrentIndex(idx)
        self.side_stack.setCurrentIndex(idx)           # 侧边栏随视图切换

    def _ensure_web(self):
        if self.web is not None:
            return
        page = self.stack.widget(1)
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            self.web = QWebEngineView()
            self.web.setUrl(QUrl(self.server.url + "?embed=1"))   # 隐藏页内按钮条
            lay = QVBoxLayout(page)
            lay.setContentsMargins(16, 16, 16, 16)
            lay.addWidget(self.web)
        except ImportError:
            # 无 WebEngine 内核时回退到系统浏览器（浏览器内保留页内按钮）
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl(self.server.url))
            self.web = QLabel(f"已在系统浏览器打开 {self.server.url}")
            self.web.setAlignment(Qt.AlignCenter)
            lay = QVBoxLayout(page)
            lay.addWidget(self.web)

    # ---------------- 妆容控制 ----------------

    def _load_presets(self):
        self.presets = []
        for p in sorted(PRESETS.glob("*.json")):
            spec = json.loads(p.read_text(encoding="utf-8"))
            self.presets.append(spec)
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, 60))
            row = QWidget()
            row.setObjectName("presetRow")
            rv = QVBoxLayout(row)
            rv.setContentsMargins(8, 5, 8, 5)
            rv.setSpacing(2)
            name = QLabel(spec.get("name", p.stem))
            name.setObjectName("presetName")
            desc = QLabel(spec.get("description", ""))
            desc.setObjectName("presetDesc")
            desc.setWordWrap(True)
            rv.addWidget(name)
            rv.addWidget(desc)
            self.preset_list.addItem(item)
            self.preset_list.setItemWidget(item, row)
        self.preset_list.setCurrentRow(0)                  # 触发 _apply_preset(0)

    def _apply_preset(self, row):
        if row < 0 or row >= len(self.presets):
            return
        spec = self.presets[row]
        self.layers = copy.deepcopy(spec.get("layers", []))
        self.intensity = float(spec.get("intensity", 0.8))
        self.intensity_slider.setValue(int(self.intensity * 100))
        self._on_look_changed()

    def _on_intensity(self, v):
        self.intensity = v / 100.0
        self.intensity_value.setText(f"{self.intensity:.2f}")
        self._on_look_changed()

    def _on_env(self, env):
        self.env = env
        self._on_look_changed()

    def _on_look_changed(self):
        if self.cam is not None:                       # 外观参数在相机线程生效（不卡 GUI）
            self.cam.set_look(self.layers)
            self.cam.intensity = self.intensity
            self.cam.set_env(self.env)
        self._splat_timer.start()

    # ---------------- 摄像头开关（按需占用设备） ----------------

    def _toggle_camera(self):
        if self.cam is not None:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if self.cam is not None:
            return
        self.cam_label.setText("正在启动摄像头…")
        self.cam_area.setCurrentIndex(1)
        self.cam_toggle_btn.setText("关闭摄像头")
        self.cam = CameraWorker(self.cam_id, self.compositor, self.parse_worker)
        self.cam.set_look(self.layers)                 # 开机前的选妆/强度/环境光在首帧生效
        self.cam.intensity = self.intensity
        self.cam.set_env(self.env)
        self.cam.frame_ready.connect(self._on_frame)
        self.cam.failed.connect(self._on_camera_failed)
        self.cam.start()
        self.parse_worker.start()

    def _stop_camera(self, note: str = "摄像头已关闭"):
        if self.cam is None:
            return
        worker, self.cam = self.cam, None
        try:
            worker.frame_ready.disconnect(self._on_frame)
        except (RuntimeError, TypeError):
            pass
        worker.stop()                                  # 释放设备后再继续（3D 采集需要独占摄像头）
        if worker.isFinished():
            worker.deleteLater()
        else:
            # 线程还没退完：绝不能销毁运行中的 QThread（原生崩溃），
            # 挂到 finished 信号上等它真正退出后由事件循环回收
            worker.finished.connect(worker.deleteLater)
        self.parse_worker.stop()
        self.compositor.set_parse(None, None)
        self.cam_area.setCurrentIndex(0)
        self.cam_toggle_btn.setText("开启摄像头")
        self.status_hint.setText(note)

    def _on_camera_failed(self, msg: str):
        worker, self.cam = self.cam, None
        if worker is not None:
            worker.stop()
            worker.deleteLater()
        self.cam_area.setCurrentIndex(0)
        self.cam_toggle_btn.setText("开启摄像头")
        self.cam_tip.setText(f"摄像头启动失败\n{msg}")
        self.status_hint.setText("摄像头不可用")

    # ---------------- 摄像头帧 ----------------

    def _on_frame(self, frame, track, fps):
        # 帧已在相机线程完成妆容合成，这里只做绘制（GUI 不再被合成卡住）
        self.last_track = track
        note = "" if track is not None else " · 未检测到人脸"
        h, w = frame.shape[:2]
        img = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        self.cam_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        text = f"相机 {fps:.0f} fps{note}"
        if self._last_splat_info:
            text += f" · 3DGS {self._last_splat_info[0]:.1f}s / {self._last_splat_info[1]:,} 点"
        self.status_hint.setText(text)

    _last_splat_info = None

    # ---------------- 3DGS ----------------

    def _submit_splat(self):
        enabled = [dict(l) for l in self.layers if l.get("enabled", True)]
        self.splat_worker.submit(enabled, self.intensity, self.env)

    def _on_splat_done(self, version, seconds, n_made, n_bare):
        self._last_splat_info = (seconds, n_made)
        self.status_hint.setText(f"3DGS 已更新 · {n_made:,} 点 · {seconds:.1f}s")

    def _on_splat_failed(self, tb):
        self.status_hint.setText("3DGS 生成失败：见控制台")
        print(tb)

    # ---------------- 参考图迁移（妆效预览页） ----------------

    _TRANSFER_FILTER = "图片 (*.jpg *.jpeg *.png *.bmp *.webp)"

    def _pick_transfer_image(self, kind: str):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择素颜照" if kind == "source" else "选择妆效参考图",
            "", self._TRANSFER_FILTER)
        if not path:
            return
        if kind == "source":
            self.transfer_src = path
        else:
            self.transfer_ref = path
        name = Path(path).name
        text = {"source": f"素颜：{name}", "reference": f"参考：{name}"}[kind]
        if self.transfer_src and self.transfer_ref:
            text += " · 可以开始迁移"
        self.transfer_hint.setText(text)

    def _start_transfer(self):
        if not (self.transfer_src and self.transfer_ref):
            QMessageBox.information(self, "参考图迁移", "请先选择素颜照与妆效参考图")
            return
        # 就绪检查（含 torch 导入）放后台线程做，避免卡 GUI；未就绪时经 failed 信号报缺
        self.transfer_run.setEnabled(False)
        self.transfer_hint.setText("迁移中… 首次运行需加载模型（数十秒）")
        self.status_hint.setText("妆效迁移进行中…")
        self.transfer_worker = TransferWorker(
            self.transfer_src, self.transfer_ref, OUT_DIR / "transfer")
        self.transfer_worker.done.connect(self._on_transfer_done)
        self.transfer_worker.failed.connect(self._on_transfer_failed)
        self.transfer_worker.start()

    def _on_transfer_done(self, cmp_path: str, res_path: str):
        self.transfer_run.setEnabled(True)
        self.transfer_hint.setText("迁移完成 · 对比图已保存")
        self.status_hint.setText(f"妆效迁移完成：{cmp_path}")
        dlg = QDialog(self)
        dlg.setWindowTitle("妆效迁移结果")
        v = QVBoxLayout(dlg)
        v.setContentsMargins(16, 16, 16, 12)
        v.setSpacing(10)
        img = QLabel()
        img.setAlignment(Qt.AlignCenter)
        pix = QPixmap(cmp_path)
        img.setPixmap(pix.scaled(1080, 620, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        v.addWidget(img)
        note = QLabel(f"对比图 {cmp_path}\n结果图 {res_path}")
        note.setObjectName("hint")
        note.setWordWrap(True)
        v.addWidget(note)
        h = QHBoxLayout()
        b_open = QPushButton("打开所在文件夹")
        b_open.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(cmp_path).parent))))
        h.addWidget(b_open)
        h.addStretch(1)
        b_close = QPushButton("关闭")
        b_close.setProperty("primary", True)
        b_close.clicked.connect(dlg.accept)
        h.addWidget(b_close)
        v.addLayout(h)
        dlg.exec()

    def _on_transfer_failed(self, msg: str):
        self.transfer_run.setEnabled(True)
        self.transfer_hint.setText("迁移失败 · 可重试")
        self.status_hint.setText("妆效迁移失败")
        if "未就绪" in msg:
            QMessageBox.warning(self, "EleGANt 未就绪", msg)
        else:
            QMessageBox.warning(self, "迁移失败",
                                f"{msg}\n\n请确认两张照片都是清晰、正面的单人人脸后重试。")

    # ---------------- 陪练 ----------------

    def _toggle_coach(self):
        if self.coach_timer.isActive():
            self.coach_timer.stop()
            self.coach_btn.setText("开始陪练")
            self.coach_label.setText("陪练已暂停")
            return
        self.coach_steps = steps_from_layers(self.layers)
        if not self.coach_steps:
            self.coach_label.setText("当前妆容没有启用任何单品")
            return
        self.coach_idx = 0
        self.coach_btn.setText("停止陪练")
        self.coach_timer.start()
        self._show_coach_step()

    def _show_coach_step(self):
        n = len(self.coach_steps)
        self.coach_progress.setValue(int(100 * self.coach_idx / max(n - 1, 1)))
        text = f"步骤 {self.coach_idx + 1}/{n}：{self.coach_steps[self.coach_idx]}"
        self.coach_label.setText(text)
        speak(f"第{self.coach_idx + 1}步。{self.coach_steps[self.coach_idx]}")

    def _coach_next(self):
        self.coach_idx = (self.coach_idx + 1) % len(self.coach_steps)
        self._show_coach_step()

    def closeEvent(self, e):
        if self.cam is not None:
            self._stop_camera()
        self.parse_worker.stop()
        w = getattr(self, "transfer_worker", None)     # 迁移线程在跑时等它落盘完再退
        if w is not None and w.isRunning():
            w.wait(120000)
        self.splat_worker.dirty = False
        self.coach_timer.stop()
        self.server.stop()
        super().closeEvent(e)
