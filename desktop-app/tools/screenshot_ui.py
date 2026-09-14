#!/usr/bin/env python3
"""离屏渲染主窗口三个视图并截图（用于 UI 预览，不影响应用逻辑）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

app = QApplication(sys.argv)
from makeupstudio.app import MainWindow

win = MainWindow(cam_id=0)
win.resize(1280, 800)
win.show()
win._start_camera()                                # 摄像头默认按需开启，截图前手动打开

out_dir = ROOT / "out" / "ui-preview"
out_dir.mkdir(parents=True, exist_ok=True)

state = {"phase": 0}

def snap():
    if state["phase"] == 0:
        win.grab().save(str(out_dir / "1-live.png"))
        state["phase"] = 1
        win._switch_page(1)                     # 妆效预览（WebEngine）
        QTimer.singleShot(6000, snap)
    elif state["phase"] == 1:
        win._viewer_js("viewer3d.three()")      # 验证侧边栏按钮驱动查看器旋转
        QTimer.singleShot(1200, lambda: (win.grab().save(str(out_dir / "2-viewer.png")),
                                         state.__setitem__("phase", 2),
                                         win._switch_page(2),
                                         QTimer.singleShot(600, snap)))
    elif state["phase"] == 2:
        win.grab().save(str(out_dir / "3-face3d.png"))
        finish()
    else:
        finish()

def finish():
    app.quit()

QTimer.singleShot(2500, snap)                   # 等摄像头出首帧
app.exec()
win.close()
print("saved:", sorted(p.name for p in out_dir.glob("*.png")))
