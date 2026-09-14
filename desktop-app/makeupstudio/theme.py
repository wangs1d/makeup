#!/usr/bin/env python3
"""theme — MakeupStudio 全局设计系统（QSS）。

设计原则（借鉴 Apple HIG，追求简洁）：
  · 结构只留三层：顶栏（标题 + 视图切换 + 状态）→ 左侧栏（随视图切换控制项）→ 内容区；
  · 浅色底 #f5f5f7，顶栏/侧栏白色，发丝级分隔线，无重投影；
  · 单一强调色 iOS systemGreen，仅用于选中态、滑条、进度与主按钮；
  · 组件一律胶囊/圆角、细边框；文字两级：墨色 #1d1d1f 与辅助灰 #86868b。
"""
from __future__ import annotations

APP_QSS = """
* {
    font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI", sans-serif;
    font-size: 13px;
    color: #1d1d1f;
}
QMainWindow, QDialog { background: #f5f5f7; }

/* ---- 顶栏 ---- */
QWidget#header { background: #ffffff; border-bottom: 1px solid #e8e8ed; }
QLabel#appTitle { font-size: 15px; font-weight: 700; }
QLabel#appSubtitle { color: #86868b; font-size: 12px; }
QLabel#hint { color: #86868b; font-size: 12px; }

/* ---- 分段控件（环境光等行内选择） ---- */
QWidget#segment { background: #ececf0; border-radius: 9px; }
QPushButton#segBtn {
    background: transparent; border: none; border-radius: 6px;
    padding: 5px 16px; color: #6e6e73; font-weight: 500;
}
QPushButton#segBtn:hover:!checked { color: #1d1d1f; }
QPushButton#segBtn:checked { background: #ffffff; color: #1d1d1f; font-weight: 600; }

/* ---- 侧栏导航（视图切换） ---- */
QPushButton#navBtn {
    background: transparent; border: none; border-radius: 9px;
    padding: 9px 14px; color: #6e6e73; text-align: left; font-weight: 500;
}
QPushButton#navBtn:hover:!checked { background: #f2f2f5; color: #1d1d1f; }
QPushButton#navBtn:checked { background: #e9f7ee; color: #1d9e4b; font-weight: 600; }

/* ---- 左侧栏 ---- */
QWidget#sidebar { background: #ffffff; border-right: 1px solid #e8e8ed; }
QLabel#sectionTitle {
    color: #86868b; font-size: 12px; font-weight: 600; letter-spacing: 1px;
}
QFrame#hairline { background: #ececf0; border: none; min-height: 1px; max-height: 1px; }
QWidget#presetRow, QWidget#presetRow QLabel { background: transparent; }
QLabel#presetName { font-weight: 600; }
QLabel#presetDesc { color: #86868b; font-size: 11px; }

/* ---- 卡片 ---- */
QFrame#card, QFrame#coachCard, QFrame#sideCard {
    background: #ffffff; border: 1px solid #e8e8ed; border-radius: 12px;
}
QLabel#stepTitle { font-weight: 600; }
QLabel#stepStatus { color: #86868b; font-size: 12px; }
QLabel#stepStatus[err="true"] { color: #ff3b30; }
QLabel#stepDot {
    background: #ececf0; color: #86868b; border-radius: 13px;
    font-size: 12px; font-weight: 600;
}
QLabel#stepDot[state="active"] {
    background: #e9f7ee; color: #1d9e4b; border: 1px solid #34c759;
}
QLabel#stepDot[state="done"] {
    background: #34c759; color: #ffffff; border: 1px solid #34c759;
}
QLabel#videoArea {
    background: #101013; color: #8d8378; font-size: 14px; border-radius: 12px;
}

/* ---- 按钮 ---- */
QPushButton {
    background: #ffffff; color: #1d1d1f;
    border: 1px solid #d9d9de; border-radius: 16px;
    padding: 7px 18px; font-weight: 500;
}
QPushButton:hover { background: #f2f2f5; }
QPushButton:pressed { background: #e9e9ee; }
QPushButton:disabled { background: #fafafa; color: #b9b9bf; border-color: #ececf0; }
QPushButton[primary="true"] {
    background: #34c759; color: #ffffff; border: none; font-weight: 600;
    padding: 8px 22px;
}
QPushButton[primary="true"]:hover { background: #2eb350; }
QPushButton[primary="true"]:pressed { background: #28a448; }
QPushButton[primary="true"]:disabled { background: #a9e8ba; color: #f3fcf5; }

/* ---- 列表（妆容库） ---- */
QListWidget {
    background: transparent; border: none; outline: 0;
}
QListWidget::item { border-radius: 10px; }
QListWidget::item:hover:!selected { background: #f2f2f5; }
QListWidget::item:selected { background: #e9f7ee; }

/* ---- 滑条 ---- */
QSlider { background: transparent; min-height: 24px; }
QSlider::groove:horizontal { height: 4px; border-radius: 2px; background: #e5e5ea; }
QSlider::sub-page:horizontal { background: #34c759; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 18px; height: 18px; margin: -8px 0;
    border-radius: 9px; background: #ffffff; border: 1px solid #d5d5da;
}
QSlider::handle:horizontal:hover { border: 1px solid #34c759; }

/* ---- 进度条 ---- */
QProgressBar {
    background: #e5e5ea; border: none; border-radius: 4px;
    min-height: 6px; max-height: 6px; text-align: center; color: transparent;
}
QProgressBar::chunk { background: #34c759; border-radius: 3px; }

/* ---- 滚动条 ---- */
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical { background: #d3d3d8; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #b9b9c0; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""


def apply_theme(widget) -> None:
    """把全局 QSS 应用到主窗口（其所有子组件自动继承）。"""
    widget.setStyleSheet(APP_QSS)
