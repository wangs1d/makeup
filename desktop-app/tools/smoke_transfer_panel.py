#!/usr/bin/env python3
"""冒烟：妆效预览页「参考图迁移」面板渲染 + 迁移后台线程全链路。

窗口模式运行（offscreen 下 WebEngine/CUDA 原生崩溃，与面板逻辑无关）；
不占摄像头、不弹结果对话框（TransferWorker 直连退出）：
    py tools/smoke_transfer_panel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout  # noqa: E402

app = QApplication(sys.argv)
import makeupstudio.app as appmod  # noqa: E402


class SmokeMainWindow(appmod.MainWindow):
    def _ensure_web(self):                         # 冒烟不依赖查看器内容，占位避免 WebEngine 噪声
        if self.web is not None:
            return
        self.web = QLabel("（冒烟：跳过 WebEngine）")
        lay = QVBoxLayout(self.stack.widget(1))
        lay.addWidget(self.web)


win = SmokeMainWindow(cam_id=0)
win.resize(1280, 800)
win.show()
win._switch_page(1)                                # 妆效预览页

out_dir = ROOT / "out" / "ui-preview"
out_dir.mkdir(parents=True, exist_ok=True)

src = ROOT / "out" / "transfer" / "demo_src.png"
ref = ROOT / "out" / "transfer" / "demo_ref.png"
result: dict[str, str] = {}


def on_done(cmp_path: str, res_path: str):
    result["cmp"], result["res"] = cmp_path, res_path
    app.quit()


def on_failed(msg: str):
    result["err"] = msg
    app.quit()


def run():
    win.grab().save(str(out_dir / "viewer-with-transfer-panel.png"))
    if not (src.is_file() and ref.is_file()):
        result["err"] = f"缺 demo 输入图：{src} / {ref}（先跑 makeup_transfer CLI 生成）"
        app.quit()
        return
    worker = appmod.TransferWorker(str(src), str(ref), ROOT / "out" / "transfer")
    worker.done.connect(on_done)
    worker.failed.connect(on_failed)
    worker.finished.connect(worker.deleteLater)
    worker.start()
    result["worker"] = "started"


def timeout():
    result["err"] = "迁移超时（300s）"
    app.quit()


QTimer.singleShot(3000, run)                       # 等窗口/WebEngine 就绪
QTimer.singleShot(300000, timeout)
app.exec()
win.close()
print("result:", result)
raise SystemExit(1 if "err" in result else 0)
