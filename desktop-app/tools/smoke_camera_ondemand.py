#!/usr/bin/env python3
"""冒烟测试：主窗口启动时不占用摄像头，开关/失败/切页路径无异常（离屏渲染）。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)
from makeupstudio.app import MainWindow

win = MainWindow(cam_id=0)
win.show()
app.processEvents()

# 1) 启动后不创建摄像头线程
assert win.cam is None, "启动即创建了 CameraWorker，摄像头仍被默认占用"
assert win.cam_area.currentIndex() == 0, "未显示「摄像头未开启」占位页"
assert win.cam_toggle_btn.text() == "开启摄像头"
print("OK: 启动不开摄像头")

# 2) 打开失败路径（不触碰真实硬件：直接调用失败槽）
win._on_camera_failed("测试：设备不可用")
assert win.cam is None
assert win.cam_area.currentIndex() == 0
assert "摄像头启动失败" in win.cam_tip.text()
assert win.cam_toggle_btn.text() == "开启摄像头"
print("OK: 失败路径回占位页")

# 3) 停止路径（未开启时为空操作）
win._stop_camera()
assert win.cam is None
print("OK: 未开启时停止是空操作")

# 4) 切到 3D 脸视图再切回：不崩溃，且回到占位页（不自动开摄像头）
win._switch_page(2)
app.processEvents()
win._switch_page(0)
app.processEvents()
assert win.cam is None, "切回化妆镜后自动开启了摄像头"
assert win.cam_area.currentIndex() == 0
print("OK: 切页释放/回归占位页")

win.close()
print("SMOKE_OK")
