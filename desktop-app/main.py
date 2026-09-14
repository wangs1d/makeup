#!/usr/bin/env python3
"""MakeupStudio 入口。

用法：
    python main.py [--camera 0]

三个目标界面：
    ① 实时化妆镜 —— 摄像头 + MediaPipe 478 点 + 实时妆容叠加 + 步骤陪练
       （摄像头默认不开启：启动后点「开启摄像头」按需打开，切走视图自动释放）
    ② 妆效预览 —— 高密度高斯泼溅点云（可旋转/缩放/素颜对比/快照导出）
    ③ 我的 3D 脸 —— 环绕采集→重建→贴妆（采集时才使用摄像头）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication

from makeupstudio.app import MainWindow


def main():
    ap = argparse.ArgumentParser(description="MakeupStudio 桌面应用")
    ap.add_argument("--camera", type=int, default=0, help="摄像头编号（默认 0）")
    args = ap.parse_args()
    app = QApplication(sys.argv)
    win = MainWindow(cam_id=args.camera)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
