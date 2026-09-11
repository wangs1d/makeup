#!/usr/bin/env python3
"""run_demo_session — 用真实系统跑一遍演示会话，录制事件流。

启动真实的 bridge_server（含资产侧车）+ recorder_app，然后按剧本时间用**真实脚本**驱动：
apply_spec.py 子进程（带烘焙资产：内嵌/HTTP 侧车两种形态都会走到）、
live_coach.py --test 子进程（真实 Bridge 取帧 → 离线剧本 → coaching 推送，含进度/光线提醒/评分），
产出 preview/events.json 给离线渲染器。

用法：python run_demo_session.py [--port 8865] [--asset-port 8868]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "makeup-skill"
OUT = Path(__file__).resolve().parent / "events.json"
DURATION = 54.0

# 剧本：(相对时间秒, 动作)
#   ("apply", spec路径, 额外参数, agent 说明) | ("coach_start", 参数列表) | ("clear",)
SCRIPT = [
    (2.0,  ("apply", "presets/daily-natural.json", ["--assets", "../out/baked/daily-natural"],
            "已下发妆容「清透日常妆」√ 8 个部位 · 浓度 75% · 附烘焙资产 18 个")),
    (9.0,  ("apply", "presets/daily-natural.json", ["--intensity", "1.0", "--assets", "../out/baked/daily-natural"],
            "把妆感调到 100% 看看完整效果")),
    (13.0, ("apply", "presets/daily-natural.json", ["--intensity", "0.5", "--assets", "../out/baked/daily-natural"],
            "淡一点，50% 更日常")),
    (17.0, ("apply", "presets/date-rose.json", ["--only", "lipstick", "--intensity", "0.9",
                                                "--assets", "../out/baked/date-rose"],
            "单件试妆：只换约会妆的镜面水红唇釉 90%")),
    (21.5, ("coach_start", ["--steps", "base,lip", "--interval", "2", "--test",
                            "--session-dir", str(ROOT / "out" / "demo" / "coach-session")])),
    (43.0, ("apply", "../out/parsed-test/makeup_spec.json", ["--bake"],
            "VLM 已解析你上传的妆容素材（演示为 canned 输出）→ 自动烘焙 → 同款已下发")),
    (48.0, ("clear",)),
]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8865, help="bridge 端口（默认 8865，避开常见占用）")
    ap.add_argument("--asset-port", type=int, default=8868)
    ap.add_argument("--inline-max", type=int, default=200 * 1024,
                    help="资产内嵌上限字节（默认 200KB → 演示里 daily 走内嵌、date-rose/解析妆走 HTTP 侧车）")
    args = ap.parse_args()

    url = f"ws://127.0.0.1:{args.port}"
    env = dict(os.environ, MAKEUP_BRIDGE_URL=url, MAKEUP_ASSET_INLINE_MAX=str(args.inline_max),
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")

    bridge_proc = subprocess.Popen(
        [sys.executable, str(SKILL / "scripts" / "bridge_server.py"), "--quiet",
         "--port", str(args.port), "--asset-port", str(args.asset_port)],
        cwd=SKILL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    await asyncio.sleep(0.8)
    rec_proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "recorder_app.py"),
         "--out", str(OUT), "--duration", str(DURATION), "--bridge", url],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=env)

    sys.path.insert(0, str(SKILL / "scripts"))
    from bridge_common import BridgeClient

    # 等 recorder 上线
    conn = None
    for _ in range(40):
        try:
            conn = await BridgeClient.connect(role="agent", client_id="demo-driver", timeout=2.0, url=url)
            st = await conn.status()
            if st.get("apps", 0) >= 1:
                break
            await conn.close()
            conn = None
        except Exception:
            pass
        await asyncio.sleep(0.25)
    if conn is None:
        print("recorder 未上线，退出", file=sys.stderr)
        bridge_proc.terminate()
        sys.exit(1)
    print(f"[driver] recorder 已就绪（{url}），开始剧本 …", flush=True)

    t0 = time.monotonic()
    coach_proc: subprocess.Popen | None = None

    async def at(t: float, coro):
        await asyncio.sleep(max(0.0, t - (time.monotonic() - t0)))
        await coro

    async def apply(spec: str, extra: list[str], note: str):
        r = await asyncio.create_subprocess_exec(
            sys.executable, str(SKILL / "scripts" / "apply_spec.py"),
            "--spec", str(SKILL / spec), *extra,
            cwd=SKILL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env)
        out, _ = await r.communicate()
        ok = "√" if r.returncode == 0 else "✗"
        print(f"[driver] apply {spec} {' '.join(extra)} → {ok} "
              f"{out.decode('utf-8', 'replace').strip().replace(chr(10), ' | ')[:160]}", flush=True)
        await conn.send({"type": "coaching", "text": f"〔agent〕{note}",
                         "area": "other", "priority": "info", "speak": False,
                         "note": "agent_note"})

    async def coach_start(extra: list[str]):
        nonlocal coach_proc
        coach_proc = subprocess.Popen(
            [sys.executable, str(SKILL / "scripts" / "live_coach.py"),
             "--spec", str(SKILL / "presets" / "daily-natural.json"), *extra],
            cwd=SKILL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=env)
        print("[driver] live_coach --test 已启动（真实 Bridge 取帧 + 离线剧本）", flush=True)
        await conn.send({"type": "coaching", "text": "〔agent〕开始实时陪练：底妆 → 唇（每 2s 取帧对比目标妆参考图）",
                         "area": "other", "priority": "info", "speak": False, "note": "agent_note"})

    async def clear():
        r = await asyncio.create_subprocess_exec(
            sys.executable, str(SKILL / "scripts" / "apply_spec.py"), "--clear",
            cwd=SKILL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env)
        await r.communicate()
        print(f"[driver] clear → {'√' if r.returncode == 0 else '✗'}", flush=True)

    tasks = []
    for t, action in SCRIPT:
        if action[0] == "apply":
            _, spec, extra, note = action
            tasks.append(asyncio.create_task(at(t, apply(spec, extra, note))))
        elif action[0] == "coach_start":
            tasks.append(asyncio.create_task(at(t, coach_start(action[1]))))
        elif action[0] == "clear":
            tasks.append(asyncio.create_task(at(t, clear())))

    await asyncio.sleep(DURATION + 0.5)
    for task in tasks:
        task.cancel()
    await conn.close()
    if coach_proc is not None:
        try:
            cout, _ = coach_proc.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            coach_proc.kill()
            cout, _ = coach_proc.communicate()
        print("[driver] live_coach 输出：\n  " + "\n  ".join(cout.strip().splitlines()[-14:]), flush=True)
    rec_proc.wait(timeout=10)
    rout = rec_proc.stdout.read() if rec_proc.stdout else ""
    print("[driver] recorder：\n  " + "\n  ".join(rout.strip().splitlines()[-8:]), flush=True)
    bridge_proc.terminate()
    data = json.loads(OUT.read_text(encoding="utf-8"))
    kinds = {}
    for ev in data["events"]:
        kinds[ev["kind"]] = kinds.get(ev["kind"], 0) + 1
    print(f"[driver] 会话结束，事件已写入 {OUT}：{kinds}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
