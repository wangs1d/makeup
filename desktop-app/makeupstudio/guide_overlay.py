#!/usr/bin/env python3
"""guide_overlay — 把采集引导动画直接画到视频帧上（帧内 HUD）。

设计借鉴 Apple HIG：
  · 材质——玻璃拟态：控件先对背景视频做真实高斯模糊再叠白色半透明（vibrancy），
    配柔和投影建立层次，而不是靠描边；
  · 形状——连续圆角胶囊、圆头线帽，全部 LINE_AA；
  · 色彩——iOS 系统色盘：白为底、墨色文字，仅用系统绿/橙/红做状态与进度强调；
  · 布局——顶部一枚分组式分段控件（左/正/右 + 迷你进度条），底部一枚状态胶囊；
  · 动效——缓慢呼吸（余弦缓动）、进度弧圆头生长，无闪烁无突兀位移。
只负责画，不做判定；纯 cv2+Pillow，GUI 线程与 CLI 共用。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .face3dgs.capture import BUCKET_MIN_SECONDS, YAW_BUCKETS

WHITE = (255, 255, 255)        # BGR
SOFT = (244, 244, 244)         # 弱化的白（引导椭圆细线）
TRACK = (222, 222, 226)        # iOS gray5：进度条轨道
INK = (30, 28, 28)             # Apple label：玻璃上的墨色文字
GREEN = (89, 199, 89)          # iOS systemGreen (BGR)
ORANGE = (0, 149, 255)         # iOS systemOrange
RED = (48, 59, 255)            # iOS systemRed

_FONTS: dict[int, object] = {}


def _font(size: int):
    """中文可渲染字体（Windows 优先微软雅黑），进程内缓存。"""
    if size in _FONTS:
        return _FONTS[size]
    from PIL import ImageFont
    f = None
    if sys.platform == "win32":
        win = Path("C:/Windows/Fonts")
        for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc"):
            p = win / name
            if p.exists():
                try:
                    f = ImageFont.truetype(str(p), size)
                    break
                except OSError:
                    continue
    if f is None:
        try:
            f = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except OSError:
            f = ImageFont.load_default()
    _FONTS[size] = f
    return f


# ---------------- 基础绘制 ----------------

def _blend_mask(img: np.ndarray, mask: np.ndarray, color, alpha: float):
    """按 mask（uint8 强度 0-255 或 bool）做逐像素 alpha 合成。"""
    m = mask.astype(np.float32) / 255.0 * float(alpha)
    region = img.astype(np.float32)
    col = np.asarray(color, np.float32)
    blended = (region * (1.0 - m[..., None]) + col * m[..., None]).astype(np.uint8)
    np.copyto(img, blended, where=(m > 0)[..., None])


def _rounded_mask(shape, x0: int, y0: int, x1: int, y1: int, r: int) -> np.ndarray:
    """连续圆角矩形 mask（矩形主体 + 四角内切圆）。"""
    r = max(1, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    m = np.zeros(shape[:2], np.uint8)
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def _glass_pill(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, unit: float,
                alpha: float = 0.82, tint=WHITE):
    """玻璃胶囊：柔和投影 → 背景真实模糊 → 白色半透明叠加。"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    # 投影（偏移 + 高斯羽化的黑色剪影）
    dy = max(2, int(4 * unit))
    sm = np.zeros((h, w), np.uint8)
    src = _rounded_mask((h, w), x0, y0, x1, y1, (y1 - y0) // 2)
    sm[dy:, :] = src[:h - dy, :]
    _blend_mask(img, sm, (0, 0, 0), 0.18)
    # 背景模糊 + 白色 tint
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=max(4.0, 7.0 * unit))
    mask = _rounded_mask((h, w), x0, y0, x1, y1, (y1 - y0) // 2).astype(np.float32) / 255.0
    mask = mask[..., None]
    comp = blur.astype(np.float32) * (1 - alpha) + np.asarray(tint, np.float32) * alpha
    np.copyto(img, comp.astype(np.uint8), where=mask > 0)


def _round_line(img, p0, p1, color, th: int):
    """圆头线段（两端补实心圆帽）。"""
    p0 = (int(p0[0]), int(p0[1]))
    p1 = (int(p1[0]), int(p1[1]))
    cv2.line(img, p0, p1, color, th, cv2.LINE_AA)
    r = max(1, th // 2)
    for p in (p0, p1):
        cv2.circle(img, p, r, color, -1, cv2.LINE_AA)


def _face_state(g) -> str:
    """ok（白）/ warn（橙）/ none（红）。"""
    if getattr(g, "face_box", None) is None:
        return "none"
    return "ok" if g.ok else "warn"


def _state_color(g):
    return {"ok": WHITE, "warn": ORANGE, "none": RED}[_face_state(g)]


# ---------------- 元素 ----------------

def _brackets(img, box, color, t, frac=0.17):
    """Face ID 式四角括号：圆头线帽自然形成圆角，缓慢呼吸开合。"""
    x0, y0, x1, y1 = (int(v) for v in box)
    grow = int(3 * (0.5 + 0.5 * np.cos(t * 2.2)))
    x0, y0, x1, y1 = x0 - grow, y0 - grow, x1 + grow, y1 + grow
    w, h = x1 - x0, y1 - y0
    ln = max(10, min(int(min(w, h) * frac), w // 2, h // 2))
    th = 3
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                           (x0, y1, 1, -1), (x1, y1, -1, -1)):
        _round_line(img, (cx + dx * ln, cy), (cx, cy), color, th)
        _round_line(img, (cx, cy), (cx, cy + dy * ln), color, th)


def _chevrons(img, cx, cy, direction, t, unit, n=2):
    """SF chevron 风格转向箭头：细、圆头、余弦缓动脉冲。"""
    pulse = 0.5 + 0.5 * np.cos(t * 2.8)
    gap = int(24 * unit)
    arm = int(12 * unit)
    th = max(2, int(3 * unit))
    shade = int(170 + 85 * pulse)
    for i in range(n):
        x = cx + direction * (i * gap + int(7 * unit * pulse) - 8 * unit)
        _round_line(img, (x - direction * arm, cy - arm), (x, cy),
                    (shade, shade, shade), th)
        _round_line(img, (x, cy), (x - direction * arm, cy + arm),
                    (shade, shade, shade), th)


def _return_to_center(img, cx, cy, t, unit):
    """一对向心圆头箭头（提示回正）。"""
    pulse = 0.5 + 0.5 * np.cos(t * 2.8)
    arm = int(13 * unit)
    th = max(2, int(3 * unit))
    shade = int(170 + 85 * pulse)
    for side in (-1, 1):
        x = cx + side * (int(44 * unit) + int(6 * unit * pulse))
        _round_line(img, (x + side * arm, cy - arm), (x, cy), (shade,) * 3, th)
        _round_line(img, (x, cy), (x + side * arm, cy + arm), (shade,) * 3, th)


def _segmented_control(img, g, w, unit) -> list[tuple]:
    """iOS 分段控件：一枚玻璃容器承载 左/正/右 三段，每段底部迷你进度条。"""
    seg, pad = int(104 * unit), int(16 * unit)
    bar_w = seg - int(28 * unit)
    ch = int(64 * unit)
    x0 = w // 2 - (3 * seg + 4 * pad) // 2
    y0 = int(22 * unit)
    _glass_pill(img, x0, y0, x0 + 3 * seg + 4 * pad, y0 + ch, unit, alpha=0.80)
    zh = {"left": "左", "center": "正", "right": "右"}
    texts = []
    for i, name in enumerate(YAW_BUCKETS):
        ratio = min(1.0, g.bucket_seconds.get(name, 0.0) / BUCKET_MIN_SECONDS)
        ccx = x0 + pad + i * (seg + pad) + seg // 2
        texts.append((ccx, y0 + int(24 * unit), zh[name], INK, int(17 * unit)))
        # 迷你进度条：轨道 gray5，填充 systemGreen，圆头
        by = y0 + ch - int(14 * unit)
        bx0, bx1 = ccx - bar_w // 2, ccx + bar_w // 2
        _round_line(img, (bx0, by), (bx1, by), TRACK, max(3, int(4 * unit)))
        if ratio > 0.02:
            _round_line(img, (bx0, by), (bx0 + int(bar_w * ratio), by), GREEN,
                        max(3, int(4 * unit)))
    return texts


def _banner(img, g, w, h, unit) -> tuple[int, int, int, int, str, tuple, int]:
    """底部状态玻璃胶囊：状态圆点 + 墨色文字。"""
    if g.done:
        text, dot = "采集完成", GREEN
    elif g.messages:
        text = " ｜ ".join(g.messages)
        dot = {"ok": GREEN, "warn": ORANGE, "none": RED}[_face_state(g)]
    else:
        text, dot = "很好，保持缓慢转头", GREEN
    size = int(19 * unit)
    f = _font(size)
    bbox = f.getbbox(text)
    tw = bbox[2] - bbox[0]
    dot_r = int(4.5 * unit)
    pad_x, pad_y = int(22 * unit), int(13 * unit)
    pw = tw + dot_r * 2 + pad_x * 2
    ph = size + pad_y * 2 + int(6 * unit)
    x0 = max(int(12 * unit), w // 2 - pw // 2)
    y0 = h - ph - int(24 * unit)
    return x0, y0, x0 + pw, y0 + ph, text, dot, size


# ---------------- 主入口 ----------------

def draw_capture_guidance(frame_bgr: np.ndarray, g, t: float | None = None) -> np.ndarray:
    """在采集帧上绘制画面内引导动画，返回带 HUD 的副本（不影响录制帧）。"""
    if t is None:
        t = time.time()
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    unit = max(0.45, w / 1280.0)
    state = _face_state(g)
    accent = GREEN if g.done else _state_color(g)

    # 1) 引导椭圆（细白）+ 进度弧（systemGreen，圆头生长）
    cx, cy = w // 2, int(h * 0.46)
    axes = (int(w * 0.155), int(h * 0.33))
    cv2.ellipse(out, (cx, cy), axes, 0, 0, 360, SOFT, max(1, int(1.5 * unit)), cv2.LINE_AA)
    prog = float(g.progress)
    if prog > 0.004:
        end_deg = -90 + int(360 * prog)
        th = max(3, int(4 * unit))
        cv2.ellipse(out, (cx, cy), (axes[0] + int(7 * unit), axes[1] + int(7 * unit)),
                    0, -90, end_deg, GREEN, th, cv2.LINE_AA)
        rad = np.radians(end_deg)                        # 圆头端帽
        ex = cx + int((axes[0] + 7 * unit) * np.cos(rad))
        ey = cy + int((axes[1] + 7 * unit) * np.sin(rad))
        cv2.circle(out, (ex, ey), th // 2 + 1, GREEN, -1, cv2.LINE_AA)

    # 2) 脸部四角括号（状态色，Face ID 式呼吸）
    if getattr(g, "face_box", None) is not None:
        _brackets(out, g.face_box, accent, t)

    # 3) 转向 / 回正箭头（无脸时先引导入镜，不显示）
    if not g.done and state != "none":
        missing = [n for n, v in g.bucket_seconds.items() if v < BUCKET_MIN_SECONDS]
        if "left" in missing:
            _chevrons(out, int(w * 0.085), cy, -1, t, unit)
        if "right" in missing:
            _chevrons(out, int(w * 0.915), cy, +1, t, unit)
        if "left" not in missing and "right" not in missing and "center" in missing:
            _return_to_center(out, cx, cy, t, unit)

    texts = _segmented_control(out, g, w, unit)

    # 4) 底部状态玻璃胶囊（圆点 + 墨色文字）
    bx0, by0, bx1, by1, text, dot, tsize = _banner(out, g, w, h, unit)
    _glass_pill(out, bx0, by0, bx1, by1, unit, alpha=0.86)
    dcx, dcy = bx0 + int(22 * unit), (by0 + by1) // 2
    cv2.circle(out, (dcx, dcy), int(4.5 * unit), dot, -1, cv2.LINE_AA)
    texts.append((dcx + int(15 * unit), dcy, text, INK, tsize, "lm"))

    _draw_texts(out, texts)
    return out


def _draw_texts(img_bgr: np.ndarray, items: list[tuple]):
    """统一走 PIL 画中文（RGBA 图层 + 极淡文字投影提升可读性）。
    items: (x, y, text, bgr_color[, size[, anchor]])，锚点默认居中。"""
    from PIL import Image, ImageDraw
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for item in items:
        x, y, text, bgr = item[:4]
        size = item[4] if len(item) > 4 else 18
        anchor = item[5] if len(item) > 5 else "mm"
        f = _font(size)
        draw.text((x, y + max(1, size // 14)), text, font=f,
                  fill=(0, 0, 0, 60), anchor=anchor)          # 柔和投影
        fill = (int(bgr[2]), int(bgr[1]), int(bgr[0]), 255)
        draw.text((x, y), text, font=f, fill=fill, anchor=anchor)
    base.alpha_composite(layer)
    img_bgr[:] = cv2.cvtColor(np.asarray(base.convert("RGB")), cv2.COLOR_RGB2BGR)
