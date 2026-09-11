#!/usr/bin/env python3
"""render_demo_video — 离线渲染演示视频。

读取 run_demo_session 录制的真实 Bridge 会话事件（events.json），
用 Python 预览渲染器（preview_render，与 Unity App 同一套 spec/蒙版/溅射/光照逻辑）
逐帧复现整个使用过程，叠加镜像 UI（状态栏/强度滑杆/agent 会话面板/指导字幕/
步骤进度条/章节标题），编码为 MP4。

用法：python render_demo_video.py [--fps 24] [--env warm] [--out ../out/demo/makeup-demo.mp4]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_render import ENVS, FaceRenderer, build_splats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STREAM = ROOT / "unity-app" / "Assets" / "StreamingAssets"
EVENTS = Path(__file__).resolve().parent / "events.json"

W, H = 1280, 720
VX, VY, VW, VH = 24, 24, 800, 672            # 镜像视口
PX, PW = 848, 408                             # 右侧面板
RENDER_SCALE = 0.8                            # 内部渲染分辨率比例（提速）
FPS_DEFAULT = 24
FADE_S = 0.6

CN = {"foundation": "粉底", "concealer": "遮瑕", "contour": "修容", "eyebrow": "眉",
      "eyeshadow": "眼影", "eyeliner": "眼线", "lashes": "睫毛", "blush": "腮红",
      "highlight": "高光", "lipstick": "唇"}

CHAPTERS = [
    (0.3, 2.4, "MakeupMirror · 实时试妆演示（Python 预览渲染 · 环境光合成）"),
    (2.4, 4.4, "① 选妆试妆 —— 摄像头里实时换妆，转头跟随"),
    (16.8, 18.8, "② 单品试妆 —— 只换唇妆（高斯溅射唇釉）"),
    (21.4, 23.4, "③ 实时化妆协助 —— AI 教练对比目标妆参考图分步指导"),
    (42.6, 44.6, "④ 妆容素材解析 —— 上传照片/视频 → 自动烘焙 → 同款"),
    (47.9, 49.9, "一键卸妆还原"),
]
COACH_WINDOW = (21.4, 42.6)
END_CARD_T = 50.5


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)


class Timeline:
    """把事件流变成逐帧状态查询。"""

    def __init__(self, data: dict):
        self.applies = []    # (t, key, spec, intensity, splats, assets)
        self.clears = []
        self.coaches = []    # (t, msg dict, is_agent_note)
        self.duration = data.get("duration", 54)
        lm = json.loads((STREAM / "landmark-regions.json").read_text(encoding="utf-8"))
        self.lm = lm["regions"]
        assets = {}
        for ev in data["events"]:
            if ev["kind"] == "apply_spec":
                msg = ev["msg"]
                spec = msg["spec"]
                for fn, b64 in (msg.get("assets") or {}).items():
                    assets[fn] = base64.b64decode(b64)
                key = f"{spec.get('name')}#{'#'.join(sorted(l['id'] for l in spec['layers']))}"
                intensity = msg.get("intensity", spec.get("intensity", 0.8))
                splats = spec.get("splat_layers") or build_splats(spec["layers"], self.lm)
                self.applies.append((ev["t"], key, spec, float(intensity), splats, assets))
            elif ev["kind"] == "clear_makeup":
                self.clears.append(ev["t"])
            elif ev["kind"] == "coaching":
                msg = ev["msg"]
                self.coaches.append((ev["t"], msg, msg.get("note") == "agent_note"))
        self.applies.sort(key=lambda a: a[0])

    def state_at(self, t: float):
        """返回 (cur, prev, fade_w, intensity, presence)。卸妆后 presence 在 FADE_S 内降到 0。"""
        cur = None
        for ap in self.applies:
            if ap[0] <= t:
                cur = ap
        presence = 1.0
        if cur is not None:
            for ct in self.clears:
                if ct > cur[0] and ct <= t:
                    presence = max(0.0, 1.0 - (t - ct) / FADE_S)
        prev = None
        if cur is not None:
            older = [ap for ap in self.applies if ap[0] < cur[0]]
            if older:
                prev = older[-1]
        fade = 1.0
        if cur is not None:
            fade = min(1.0, (t - cur[0]) / FADE_S)
        intensity = cur[3] if cur else 1.0
        if presence <= 0.0:
            cur = None
        return cur, prev, fade, intensity, presence

    def active_coach(self, t: float):
        """最近 4.5s 内的教练提醒（非 agent 说明）。"""
        best = None
        for ct, msg, is_note in self.coaches:
            if is_note or not (ct <= t <= ct + 4.5):
                continue
            best = (msg.get("text", ""), msg.get("priority", "info"), (t - ct) / 4.5)
        return best

    def progress_at(self, t: float):
        """最近一条带 progress 的 coaching → (step, steps, step_name, progress)。"""
        best = None
        for ct, msg, is_note in self.coaches:
            if is_note or ct > t or msg.get("progress") is None:
                continue
            best = (int(msg.get("step") or 0), int(msg.get("steps") or 0),
                    msg.get("step_name") or "", float(msg.get("progress") or 0.0))
        return best

    def bubbles(self, t: float):
        out = []
        for ct, msg, is_note in self.coaches:
            if ct <= t:
                out.append((ct, msg.get("text", ""), msg.get("priority", "info"), is_note))
        return out[-7:]


# ---------------- UI ----------------

BG_PANEL = (22, 24, 30, 255)
FG = (232, 234, 238, 255)
FG_DIM = (150, 155, 165, 255)
ACCENT = (240, 110, 140, 255)
WARN = (255, 160, 60, 255)
GREEN = (110, 220, 130, 255)


def rounded(d: ImageDraw.ImageDraw, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def glyph_safe(text: str) -> str:
    """msyh 缺 U+2713（✓）等字形，替换为可渲染的近似符号。"""
    return str(text).replace("✓", "√")


def draw_ui(frame_bgr: np.ndarray, tl: Timeline, t: float, intensity: float,
            spec_name: str | None, env: str) -> np.ndarray:
    img = Image.fromarray(frame_bgr[..., ::-1]).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    f_s, f_m, f_l = font(20), font(24), font(30)

    # 右侧面板
    d.rectangle([PX, 0, W, H], fill=BG_PANEL)
    d.text((PX + 24, 26), "Agent 实时会话", font=f_l, fill=FG)
    d.text((PX + 24, 66), "bridge 1.1 · 精确路由 · 资产 HTTP 侧车", font=f_s, fill=FG_DIM)
    d.line([PX + 20, 104, W - 20, 104], fill=(60, 64, 74, 255), width=2)

    y = 124
    bubbles = tl.bubbles(t)
    in_coach = COACH_WINDOW[0] <= t <= COACH_WINDOW[1]
    if in_coach:
        bubbles = bubbles[-4:]  # 指导章节给底部步骤条留空间
    for bt, text, prio, is_note in bubbles:
        who = "agent" if is_note else "coach"
        text = glyph_safe(text)
        col = ACCENT if is_note else (WARN if prio == "warn" else (120, 200, 255, 255))
        d.text((PX + 24, y), f"{who} · {bt:5.1f}s", font=f_s, fill=col)
        y += 28
        line, lines = "", []
        for ch in text:
            if d.textlength(line + ch, font=f_m) > PW - 72:
                lines.append(line)
                line = ch
            else:
                line += ch
        if line:
            lines.append(line)
        hgt = len(lines) * 30 + 18
        rounded(d, [PX + 20, y, W - 20, y + hgt], 10, fill=(38, 42, 52, 255))
        for i, ln in enumerate(lines):
            d.text((PX + 34, y + 8 + i * 30), ln, font=f_m, fill=FG)
        y += hgt + 14
        if y > H - 150:
            break

    # 镜像视口边框 + 状态栏
    rounded(d, [VX - 2, VY - 2, VX + VW + 2, VY + VH + 2], 14,
            outline=(70, 74, 86, 255), width=3)
    d.rectangle([VX + 14, VY + 12, VX + 14 + 12, VY + 24], fill=GREEN)
    d.text((VX + 38, VY + 10), "bridge: connected", font=f_s, fill=FG)
    d.text((VX + 220, VY + 10), f"tracking: ok · 30 fps · 468 pts · pose √ · env {env}", font=f_s, fill=FG)
    d.text((VX + VW - 16, VY + VH - 34), "Python 预览渲染（Unity App 同管线：蒙版/溅射/光照）",
           font=font(18), fill=(200, 202, 210, 200), anchor="ra")

    # 强度滑杆
    sx0, sy = VX + 24, VY + VH - 66
    d.text((sx0, sy - 34), f"妆感 {int(intensity * 100)}%", font=f_m, fill=FG)
    rounded(d, [sx0, sy, sx0 + 320, sy + 14], 7, fill=(52, 56, 66, 255))
    rounded(d, [sx0, sy, sx0 + int(320 * intensity), sy + 14], 7, fill=ACCENT)
    d.ellipse([sx0 + int(320 * intensity) - 12, sy - 5,
               sx0 + int(320 * intensity) + 12, sy + 19], fill=(245, 245, 248, 255))
    if spec_name:
        d.text((sx0 + 150, sy - 34), f"当前妆容：{spec_name}", font=f_s, fill=FG_DIM)

    # 教练大字幕（视口底部居中）
    coach = tl.active_coach(t)
    if coach:
        text, prio, prog = coach
        alpha = int(255 * min(1.0, (1 - prog) * 3) if prog > 0.8 else 255)
        col = WARN if prio == "warn" else (255, 255, 255, alpha)
        d.text((VX + VW // 2, VY + VH - 118), glyph_safe(text), font=font(28), fill=col, anchor="mm")

    # 步骤进度条（指导章节期间，由 coaching 消息的 step/steps/progress 驱动）
    if in_coach:
        pr = tl.progress_at(t)
        x0, y0 = PX + 24, H - 58
        rounded(d, [PX + 12, y0 - 48, W - 12, y0 + 30], 10, fill=(16, 17, 22, 220))
        d.text((x0, y0 - 36), "化妆步骤（live_coach 推送）", font=font(18), fill=FG_DIM)
        if pr:
            step, steps, name, progress = pr
            d.text((x0, y0 - 6), f"步骤 {step}/{steps} · {name} · {int(progress * 100)}%", font=f_m, fill=FG)
            bw = PW - 48
            rounded(d, [x0, y0 + 20, x0 + bw, y0 + 28], 4, fill=(52, 56, 66, 255))
            done_w = int(bw * ((step - 1 + progress) / max(steps, 1)))
            rounded(d, [x0, y0 + 20, x0 + max(done_w, 4), y0 + 28], 4, fill=GREEN if progress >= 1 else ACCENT)
        else:
            d.text((x0, y0 - 6), "正在生成目标妆参考图 …", font=f_m, fill=FG_DIM)

    # 章节字幕
    for t0, t1, text in CHAPTERS:
        if t0 <= t <= t1:
            fade = min(1.0, (t - t0) / 0.4, (t1 - t) / 0.4)
            a = int(230 * max(0, fade))
            tw = d.textlength(text, font=f_l)
            rounded(d, [VX + VW // 2 - tw / 2 - 18, VY + 18,
                        VX + VW // 2 + tw / 2 + 18, VY + 64], 10, fill=(10, 10, 14, a))
            d.text((VX + VW // 2, VY + 41), text, font=f_l, fill=(255, 255, 255, a), anchor="mm")
            break

    # 结尾卡
    if t >= END_CARD_T:
        a = int(min(1.0, (t - END_CARD_T) / 0.8) * 235)
        d.rectangle([0, 0, W, H], fill=(12, 12, 16, a))
        if a > 40:
            d.text((W // 2, 200), "化妆助手 · 三大能力", font=font(40), fill=FG, anchor="mm")
            for i, line in enumerate([
                    "①  选妆试妆 —— 6DoF 姿态贴脸、环境光合成、高斯溅射唇釉",
                    "②  妆容素材解析 —— 上传照片/视频，VLM 解析成同款 + 效果预览图",
                    "③  实时化妆协助 —— 目标妆参考图视觉对比、滑窗防抖、进度/评分/语音"]):
                d.text((W // 2, 300 + i * 56), line, font=font(26), fill=(210, 214, 222, a),
                       anchor="mm")
            d.text((W // 2, 510), "本视频为真实 Bridge 协议会话的离线复现（无 VLM key，指导为 live_coach --test 离线剧本）",
                   font=f_s, fill=(150, 155, 165, a), anchor="mm")
            d.text((W // 2, 542), "makeup-skill/ ── SKILL.md 能力包 · unity-app/ ── Unity 试妆 App",
                   font=f_s, fill=(150, 155, 165, a), anchor="mm")

    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))[..., ::-1].copy()


def ease(x):
    return x * x * (3 - 2 * x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=FPS_DEFAULT)
    ap.add_argument("--out", default=str(ROOT / "out" / "demo" / "makeup-demo.mp4"))
    ap.add_argument("--env", default="warm", choices=sorted(ENVS), help="环境光预设")
    ap.add_argument("--duration", type=float, default=0, help="只渲染前 N 秒（0=全部，调试用）")
    ap.add_argument("--start", type=float, default=0, help="从第 N 秒开始渲染（调试用）")
    args = ap.parse_args()

    data = json.loads(EVENTS.read_text(encoding="utf-8"))
    tl = Timeline(data)
    duration = tl.duration if not args.duration else min(tl.duration, args.start + args.duration)

    renderer = FaceRenderer(int(VW * RENDER_SCALE), int(VH * RENDER_SCALE),
                            STREAM / "canonical_face_model.obj",
                            STREAM / "landmark-regions.json")
    renderer.set_env(args.env)
    tex_cache: dict = {}

    def textures(key, spec):
        if key not in tex_cache:
            tex_cache[key] = renderer.makeup_for(key, spec["layers"])
        return tex_cache[key]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not vw.isOpened():
        raise SystemExit("VideoWriter 打开失败")

    n_frames = int((duration - args.start) * args.fps)
    t_start = time.time()
    last_target, last_change, shown_i = None, 0.0, None
    for fi in range(n_frames):
        t = args.start + fi / args.fps
        cur, prev, fade, target_i, presence = tl.state_at(t)

        # 强度滑杆平滑动画：目标变化时从旧值缓动到新值
        if last_target is None or abs(target_i - last_target) > 1e-6:
            last_target, last_change, shown_i = target_i, t, shown_i
        k = ease(min(1.0, (t - last_change) / 0.6))
        intensity = shown_i + (target_i - shown_i) * k if shown_i is not None else target_i
        if cur is not None:
            shown_i = intensity

        mk = prev_mk = None
        splats = None
        name = None
        if cur is not None:
            _, key, spec, _, splats, _assets = cur
            mk = textures(key, spec)
            name = spec.get("name", "")
            if prev is not None and fade < 1.0:
                prev_mk = textures(prev[1], prev[2])
            splats = splats if intensity > 0.02 else None
        face = renderer.render(t, mk, prev_mk, ease(fade), intensity, splats, presence=presence)
        if RENDER_SCALE != 1.0:
            face = cv2.resize(face, (VW, VH), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:] = (30, 30, 34)
        canvas[VY:VY + VH, VX:VX + VW] = face
        frame = draw_ui(canvas, tl, t, intensity, name, args.env)
        vw.write(frame)
        if fi % 120 == 0:
            el = time.time() - t_start
            print(f"[render] {fi}/{n_frames} 帧  {t:5.1f}s  已用 {el:5.0f}s"
                  f"  预计剩余 {el / max(fi, 1) * (n_frames - fi):5.0f}s", flush=True)

    vw.release()
    print(f"[render] 完成 → {out_path}（{out_path.stat().st_size / 1024:.0f} KB，"
          f"总耗时 {(time.time() - t_start) / 60:.1f} 分钟）")


if __name__ == "__main__":
    main()
