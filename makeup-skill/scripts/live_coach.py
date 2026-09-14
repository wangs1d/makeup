#!/usr/bin/env python3
"""能力三：实时化妆协助。对比目标妆容与用户当前妆面，分步指导推送到 App。

流程循环：
    启动：渲染目标妆参考图（preview_render，与 App 同管线）
    每轮：向 App 请求摄像头帧 → VLM 对比（参考图 + 当前帧 + 当前步骤）→ JSON 结论 →
          滑动窗投票推进步骤 → 新提醒推送 coaching（文字+进度+语音）→ 等待 interval
    结束：会话报告 session_report.md（步骤完成情况 + 各部位评分 + 建议）

用法：
    python live_coach.py --spec presets/daily-natural.json
    python live_coach.py --spec out/parsed/makeup_spec.json --interval 4 --duration 300
    python live_coach.py --spec look.json --steps base,lip      # 只陪练部分步骤
    python live_coach.py --spec look.json --test                # 离线 canned 模式（不调 VLM）
    python live_coach.py --spec look.json --no-reference        # 不给 VLM 参考图
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vlm  # noqa: E402
from bridge_common import BridgeClient, BridgeError, fail  # noqa: E402

SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = SKILL_DIR / "references" / "prompt-templates" / "live_coach.md"

AREA_CN = {"base": "底妆", "eyebrow": "眉", "eye": "眼妆", "blush": "腮红/高光", "lip": "唇"}


class CoachDone(Exception):
    """全部步骤完成，结束循环（协程里不能用 StopIteration）。"""


def load_prompt() -> tuple[str, str]:
    """返回 (system, user 模板)。"""
    try:
        md = TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError:
        fail(f"提示词模板缺失：{TEMPLATE_PATH}")
    system = user = None
    for section in md.split("\n## "):
        m = re.search(r"```\n(.*?)\n```", section, re.DOTALL)
        if not m:
            continue
        if section.strip().startswith("system"):
            system = m.group(1).strip()
        elif section.strip().startswith("user"):
            user = m.group(1).strip()
    if not system or not user:
        fail("模板中缺少 system/user 提示词块")
    return system, user


def spec_digest(spec: dict) -> str:
    """给 VLM 的目标妆容精简描述。"""
    lines = [f"妆容：{spec.get('description', spec.get('name', ''))}，"
             f"整体浓度 {spec.get('intensity', 0.8):.0%}"]
    for l in spec.get("layers", []):
        if not l.get("enabled", True):
            continue
        colors = "→".join(c["hex"] for c in l.get("color_stops", []))
        line = f"- {l['region']}/{l.get('side', 'both')}：{l.get('finish')}，显色 {l.get('opacity', 0.7):.0%}，颜色 {colors}"
        if l.get("notes"):
            line += f"（{l['notes']}）"
        lines.append(line)
    return "\n".join(lines)


def build_steps(spec: dict, only: list[str] | None) -> list[dict]:
    steps = spec.get("steps") or []
    if not steps:  # 无 steps 的 spec：按层序生成占位步骤
        regions = [l["region"] for l in spec.get("layers", []) if l.get("enabled", True)]
        steps = [{"order": i + 1, "area": r, "regions": [r], "instruction": f"完成{r}"}
                 for i, r in enumerate(dict.fromkeys(regions))]
    if only:
        wanted = set(only)
        steps = [s for s in steps if s.get("area") in wanted or set(s.get("regions", [])) & wanted]
    return sorted(steps, key=lambda s: s.get("order", 0))


def assess_lighting(frame_bytes: bytes) -> str | None:
    """本地亮度粗检，不耗 VLM 调用。返回提醒文案或 None。"""
    try:
        img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        mean = float(img.mean())
        if mean < 55:
            return "现在光线太暗啦，开一盏正面灯再继续，不然看不清妆效也会画歪"
        if mean > 215:
            return "光线太亮有点过曝了，稍微避开直射光"
    except Exception:
        pass
    return None


# ---------------- 离线 canned 模式（--test，不调 VLM） ----------------

class CannedVerdicts:
    """离线剧本：每步 in_progress ×2 → done ×2（滑窗投票需要连续两次 done 才推进）。
    提示语按步骤 area 取真实口吻的示例，便于无 VLM 环境演示/联调。"""

    TIPS = {
        "base": (["先上粉底，由面中向外轻轻拍开", "鼻翼和下巴边缘再补一点，别留分界线"],
                 "底妆完成，均匀通透 ✓"),
        "eyebrow": (["眉毛顺毛流描，眉尾略深眉头浅", "左眉峰再往上提 1 毫米，两边就对称了"],
                    "眉毛完成 ✓"),
        "eye": (["浅棕眼影铺满眼窝，眼尾三角区再加深一次", "眼线贴睫毛根部，眼尾平拉 2 毫米"],
                "眼妆完成 ✓"),
        "blush": (["腮红斜向上扫在颧骨上方，少量叠两次", "右脸再补一点，两边浓度一致"],
                  "腮红完成 ✓"),
        "lip": (["唇釉先涂内侧，再用指腹向外晕开做渐变", "唇峰描清楚一点，下唇中央再叠一层"],
                "唇妆完成 ✓"),
    }

    def __init__(self, steps: list[dict]) -> None:
        self.steps = steps
        self.n_steps = max(1, len(steps))
        self.round = 0

    def next(self) -> dict:
        self.round += 1
        k = (self.round - 1) % 4          # 0,1 in_progress；2,3 done
        step_idx = ((self.round - 1) // 4) % self.n_steps
        last = step_idx == self.n_steps - 1
        area = self.steps[step_idx].get("area", "other") if self.steps else "other"
        tips, done_text = self.TIPS.get(area, ([f"继续第 {step_idx + 1} 步", "快完成了，保持手法"], "本步完成 ✓"))
        base = {
            "face_visible": True,
            "lighting": {"ok": True},
            "progress": round(0.35 + 0.3 * k, 2) if k < 2 else 1.0,
            "actions": [],
            "done_summary": "",
        }
        if k < 2:
            base["step_status"] = "in_progress"
            base["actions"] = [{"area": area, "text": tips[k], "urgency": "info"}]
        else:
            base["step_status"] = "done"
            nxt = self.steps[step_idx + 1].get("area") if not last else None
            base["done_summary"] = done_text + (f" 接下来{AREA_CN.get(nxt, nxt)}" if nxt else "")
            if last and k == 3:
                base["scores"] = self.final_scores()
        return base

    def final_scores(self) -> list[dict]:
        return [{"area": a, "score": s, "advice": "离线演示评分"}
                for a, s in [("base", 90), ("eyebrow", 88), ("eye", 85), ("blush", 92), ("lip", 95)]]


# ---------------- Coach ----------------

class Coach:
    def __init__(self, spec: dict, interval: float, args) -> None:
        self.spec = spec
        self.interval = interval
        self.args = args
        self.system_prompt, self.user_template = load_prompt()
        self.steps = build_steps(spec, None)
        self.step_idx = 0
        self.status_window: list[str] = []      # 最近 3 轮 step_status（投票防抖）
        self.last_actions: list[str] = []
        self.history: list[str] = []
        self.log: list[tuple] = []              # (round, time, kind, text)
        self.rounds = 0
        self.progress = 0.0
        self.tracking_ok = True
        self.canned = CannedVerdicts(self.steps) if args.test else None
        self.reference_b64: str | None = None

    # ---- 参考图 ----

    def make_reference(self, out_dir: Path) -> None:
        if self.args.no_reference:
            return
        try:
            # 画像妆容台流程：参考图来自画像编译产物（avatar_session preview 产出）
            look_dir = getattr(self.args, "avatar_look", None)
            if look_dir:
                p = Path(look_dir) / "reference.jpg"
                if not p.exists():
                    raise FileNotFoundError(f"缺少画像编译参考图：{p}")
                from avatar_render import imread
                png = imread(p)                       # Unicode 安全读取
                if png is None:
                    raise IOError(f"参考图读取失败：{p}")
                self.reference_b64 = base64.b64encode(
                    cv2.imencode(".jpg", png, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()).decode()
                print(f"[coach] 目标妆参考图（画像渲染）← {p}（随每轮发给 VLM 对比）")
                return
            from preview_render import render_reference
            png = render_reference(self.spec, out_dir / "reference.jpg",
                                   size=512, env=self.args.env)
            self.reference_b64 = base64.b64encode(
                cv2.imencode(".jpg", png, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()).decode()
            print(f"[coach] 已生成目标妆参考图 → {out_dir / 'reference.jpg'}（随每轮发给 VLM 对比）")
        except Exception as e:  # noqa: BLE001
            print(f"[coach] 参考图生成失败（继续用文字描述）：{e}")

    # ---- 状态机 ----

    def current_step(self) -> dict:
        return self.steps[min(self.step_idx, len(self.steps) - 1)]

    def on_tracking(self, msg: dict) -> None:
        self.tracking_ok = bool(msg.get("ok", True))

    def vote_and_advance(self, status: str) -> None:
        """最近 3 轮投票：≥2 个 done 才推进（单帧误判不跳步）。"""
        self.status_window.append(status)
        if len(self.status_window) > 3:
            self.status_window.pop(0)
        if status == "done" and self.step_idx < len(self.steps) - 1 \
                and self.status_window.count("done") >= 2:
            self.step_idx += 1
            self.status_window.clear()
            self.progress = 0.0

    def history_text(self) -> str:
        return "；".join(self.history[-3:]) if self.history else "（暂无）"

    # ---- 单轮 ----

    async def round(self, conn: BridgeClient) -> None:
        await conn.ensure_connected()
        self.rounds += 1
        step = self.current_step()
        try:
            frame = await conn.request_frame(quality=80)
        except (BridgeError, Exception) as e:
            print(f"[coach] 取帧失败：{e}")
            await asyncio.sleep(self.interval * 2)
            return

        if not self.tracking_ok:
            await self.push(conn, "先把整张脸对准屏幕中间的框，保持 30~40cm 距离", "other", "warn")
            return

        light_hint = assess_lighting(frame)
        if light_hint:
            await self.push(conn, light_hint, "other", "warn", note="light")
            return

        frame_b64 = base64.b64encode(frame).decode()
        target = f"目标妆容：\n{spec_digest(self.spec)}"
        step_line = (f"当前步骤：{self.step_idx + 1}/{len(self.steps)} "
                     f"{AREA_CN.get(step['area'], step['area'])} — {step['instruction']}"
                     f"（涉及 {','.join(step.get('regions', []))}）")
        user_msg = self.user_template \
            .replace("{TARGET}", target) \
            .replace("{STEP}", step_line) \
            .replace("{HISTORY}", self.history_text()) \
            .replace("{REF}", "第 1 张图是目标妆容的效果参考图，第 2 张是用户当前画面" if self.reference_b64
                     else "附图是用户当前画面")

        try:
            if self.canned is not None:
                verdict = self.canned.next()
            else:
                content = []
                if self.reference_b64:
                    content.append({"type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{self.reference_b64}"}})
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})
                content.append({"type": "text", "text": f"{self.system_prompt}\n\n{user_msg}"})
                raw = vlm.chat([{"role": "user", "content": content}],
                               temperature=0.3, max_tokens=1024)
                verdict = vlm.extract_json(raw)
        except (RuntimeError, ValueError) as e:
            print(f"[coach] 第 {self.rounds} 轮 VLM 失败：{e}")
            await asyncio.sleep(self.interval)
            return

        if not verdict.get("face_visible", True):
            await self.push(conn, "摄像头里没看到脸，稍微挪一下位置", "other", "warn")
            return

        actions = [a for a in (verdict.get("actions") or []) if isinstance(a, dict)]
        status = verdict.get("step_status", "in_progress")
        try:
            progress = float(verdict.get("progress"))
            progress = min(max(progress, 0.0), 1.0)
        except (TypeError, ValueError):
            progress = None
        if progress is not None:
            self.progress = max(self.progress, progress) if status != "done" else 1.0

        prefix = f"[{self.rounds:03d} {time.strftime('%H:%M:%S')}] " \
                 f"步骤{self.step_idx + 1}/{len(self.steps)} {AREA_CN.get(step['area'], step['area'])}"
        if progress is not None:
            prefix += f"（{self.progress:.0%}）"

        fresh = []
        for a in actions:
            text = str(a.get("text", "")).strip()
            if text and text not in self.last_actions:
                fresh.append((str(a.get("area", "other")), text, str(a.get("urgency", "info"))))
        self.last_actions = [str(a.get("text", "")).strip() for a in actions]

        if verdict.get("lighting") and isinstance(verdict["lighting"], dict) \
                and not verdict["lighting"].get("ok", True) and verdict["lighting"].get("hint"):
            fresh.append(("other", str(verdict["lighting"]["hint"]), "warn"))

        before = self.step_idx
        self.vote_and_advance(status)
        advanced = self.step_idx > before

        if status == "done" or advanced:
            done_msg = str(verdict.get("done_summary") or "").strip()
            text = done_msg or f"{AREA_CN.get(step['area'], step['area'])}完成 ✓"
            print(f"{prefix} ✓ {text}")
            await self.push(conn, text, "other", "info", note="done")
            self.history.append(f"完成：{AREA_CN.get(step['area'], step['area'])}")
            self.log.append((self.rounds, time.strftime("%H:%M:%S"), "done", text))
        elif fresh:
            for area, text, urgency in fresh:
                print(f"{prefix} {'⚠' if urgency == 'warn' else '·'} {text}")
                await self.push(conn, text, area, urgency)
                self.log.append((self.rounds, time.strftime("%H:%M:%S"), "warn" if urgency == "warn" else "tip", text))
            self.history.extend(t for _, t, _ in fresh[:2])
            self.history = self.history[-5:]
        else:
            print(f"{prefix} （保持当前动作）")

        # 全部步骤完成 → 评分 + 结束
        if self.step_idx >= len(self.steps) - 1 and status == "done" \
                and self.current_step() is self.steps[-1] and self.status_window.count("done") >= 2:
            scores = verdict.get("scores")
            if not scores and self.canned is not None:
                scores = self.canned.final_scores()
            await self.finish(conn, scores)
            raise CoachDone

    async def finish(self, conn: BridgeClient, scores) -> None:
        if not scores and self.canned is None:
            try:
                scores = self.ask_scores()
            except Exception as e:  # noqa: BLE001
                print(f"[coach] 评分调用失败（跳过）：{e}")
        summary = "全妆完成，今天也美美的！"
        top = None
        if scores:
            try:
                lowest = sorted((s for s in scores if isinstance(s.get("score"), (int, float))),
                                key=lambda s: s["score"])
                if lowest and lowest[0]["score"] < 90:
                    top = lowest[0]
                    summary = (f"全妆完成！{AREA_CN.get(top.get('area'), top.get('area', ''))}还可以再补一补：{top.get('advice', '')}"
                               if top.get("advice") else summary)
            except Exception:  # noqa: BLE001
                pass
        await self.push(conn, summary, "other", "info", note="done")
        print(f"[coach] 所有步骤完成 ✓ {summary}")
        self.write_report(scores)

    def ask_scores(self) -> list[dict]:
        """结束轮：VLM 根据过程记录按部位打分（纯文本调用）。"""
        record = "\n".join(f"r{r} {t} [{kind}] {text}" for r, t, kind, text in self.log[-15:])
        raw = vlm.chat([{"role": "user", "content":
            "你是化妆教练。根据以下指导过程记录，对这次化妆各部位完成度打分（0-100）并给一句建议。\n"
            "输出严格 JSON 数组：[{\"area\":\"base|eyebrow|eye|blush|lip|other\",\"score\":0-100,\"advice\":\"一句话\"}]\n"
            f"过程记录：\n{record}"}],
            temperature=0.2, max_tokens=512)
        data = vlm.extract_json(raw)
        return data if isinstance(data, list) else []

    def write_report(self, scores) -> None:
        out_dir = Path(self.args.session_dir) if self.args.session_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# 化妆陪练会话报告",
            "",
            f"- 目标妆容：{self.spec.get('name')}（{self.spec.get('description', '')}）",
            f"- 时长/轮数：{self.rounds} 轮（间隔 {self.interval:g}s）",
            f"- 步骤完成：{self.step_idx + 1}/{len(self.steps)}"
            + ("（全部完成 ✓）" if self.step_idx >= len(self.steps) - 1 else ""),
            "",
            "## 过程",
            "",
        ]
        for r, t, kind, text in self.log:
            mark = {"done": "✓", "warn": "⚠", "tip": "·"}.get(kind, "·")
            lines.append(f"- `{t}` {mark} {text}")
        if scores:
            lines += ["", "## 完成度评分", "", "| 部位 | 得分 | 建议 |", "|---|---|---|"]
            for s in scores:
                lines.append(f"| {AREA_CN.get(s.get('area'), s.get('area', '?'))} "
                             f"| {s.get('score', '-')} | {s.get('advice', '-')} |")
        p = out_dir / "session_report.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"[coach] 会话报告 → {p}")

    async def push(self, conn: BridgeClient, text: str, area: str, priority: str,
                   note: str = "") -> None:
        step = self.current_step()
        msg = {"type": "coaching", "text": text, "area": area,
               "priority": priority, "speak": True, "note": note,
               "step": self.step_idx + 1, "steps": len(self.steps),
               "step_name": AREA_CN.get(step["area"], step["area"]),
               "progress": round(self.progress, 2)}
        try:
            await conn.send(msg)
        except Exception:
            pass


async def run(args: argparse.Namespace) -> None:
    spec_path = Path(args.spec)
    if not spec_path.exists():
        fail(f"spec 不存在：{spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not args.test and not vlm.is_configured():
        fail("VLM 未配置：实时指导需要设置 MAKEUP_VLM_API_KEY（见 SKILL.md）；离线联调用 --test")

    try:
        conn = await BridgeClient.connect(role="agent", client_id="live-coach", retries=8, backoff=0.5)
    except BridgeError as e:
        fail(str(e))

    session_dir = Path(args.session_dir) if args.session_dir else Path("out/coach-sessions") / time.strftime("%Y%m%d-%H%M%S")
    coach = Coach(spec, args.interval, args)
    coach.steps = build_steps(spec, [s.strip() for s in args.steps.split(",")] if args.steps else None)
    if args.test:
        coach.canned = CannedVerdicts(coach.steps)   # 步骤过滤后重建剧本
    conn.on_tracking = coach.on_tracking
    st = await conn.status()
    if st.get("apps", 0) == 0:
        fail("没有试妆 App 连接：实时指导需要 App 回传摄像头画面（references/app-setup.md）")
    coach.make_reference(session_dir)
    if args.station:
        # 妆容台联动：让 App 切到 station 布局（画像侧栏参照，真脸不渲染妆容）
        try:
            await conn.request({"type": "enter_station", "look":
                                {"name": spec.get("name", "look"),
                                 "intensity": spec.get("intensity", 0.8)}}, timeout=10.0)
            print("[coach] 已让 App 进入妆容台布局 ✓")
        except Exception as e:  # noqa: BLE001
            print(f"[coach] enter_station 失败（App 可能未加载画像，继续普通模式）：{e}")

    areas = " → ".join(AREA_CN.get(s["area"], s["area"]) for s in coach.steps)
    mode = "离线 canned" if args.test else f"VLM {vlm.VLM_MODEL}"
    print(f"[coach] 开始陪练：{spec.get('name')}（{len(coach.steps)} 步：{areas}；"
          f"每 {args.interval:g}s 采样一轮，{mode}，Ctrl+C 结束）")

    started = time.monotonic()
    try:
        while True:
            if args.duration and time.monotonic() - started > args.duration:
                print("[coach] 达到时长上限，结束指导")
                coach.write_report(None)
                break
            try:
                await coach.round(conn)
            except CoachDone:
                break
            await asyncio.sleep(args.interval)
    finally:
        if args.station:
            try:
                await conn.send({"type": "leave_station"})
            except Exception:  # noqa: BLE001
                pass
        await conn.close()
        print(f"[coach] 已结束（共 {coach.rounds} 轮，推进到第 {coach.step_idx + 1} 步）")


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)   # 被 agent 以管道方式运行时也能实时看到每轮输出
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="实时化妆指导")
    ap.add_argument("--spec", required=True, help="目标妆容 makeup_spec.json")
    ap.add_argument("--interval", type=float, default=5.0, help="采样间隔秒（默认 5）")
    ap.add_argument("--duration", type=float, default=0, help="最长运行秒数（0=不限）")
    ap.add_argument("--steps", help="只陪练部分步骤，逗号分隔 area 名（base,eyebrow,eye,blush,lip）")
    ap.add_argument("--test", action="store_true", help="离线 canned 模式，不调 VLM（联调演示用）")
    ap.add_argument("--no-reference", action="store_true", help="不给 VLM 发目标妆参考图")
    ap.add_argument("--avatar-look", default=None,
                    help="画像编译目录（avatar_session preview 产出），参考图改用画像渲染效果")
    ap.add_argument("--station", action="store_true",
                    help="开始时让 App 进妆容台布局、结束时退出（画像试妆流程）")
    ap.add_argument("--env", default="neutral", help="参考图渲染环境光（neutral/warm/cool/dim）")
    ap.add_argument("--session-dir", default=None, help="参考图与会话报告目录（默认 out/coach-sessions/<时间戳>）")
    args = ap.parse_args()
    if args.interval < 2:
        fail("--interval 最小 2 秒（VLM 调用有延迟，太密会排队）")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[coach] 用户中断，已退出")


if __name__ == "__main__":
    main()
