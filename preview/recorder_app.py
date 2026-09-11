#!/usr/bin/env python3
"""recorder_app — 演示录制用 app 角色客户端。

与 makeup bridge_server 建立真实 WebSocket 会话，按协议 ack 所有指令，
把收到的每条消息连同相对时间戳记录下来（含 spec 全文；资产按 assets_url 取回后以 base64 内联保存），
供离线渲染器逐帧复现。request_frame 回一张合成帧（frame 带 ref），使 live_coach --test 可闭环。
这是对 Unity App 行为的最小化等价实现。

用法：python recorder_app.py --out events.json [--duration 45]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "makeup-skill" / "scripts"))
from bridge_common import BridgeClient, download_url  # noqa: E402


def synthetic_frame(dark: bool = False) -> bytes:
    """合成一张"用户画面"；dark=True 给过暗帧，用于触发 live_coach 的本地光线提醒（演示用）。"""
    from PIL import Image, ImageDraw
    bg, skin = ((196, 178, 166), (236, 200, 182)) if not dark else ((28, 24, 22), (46, 36, 32))
    img = Image.new("RGB", (640, 480), bg)
    d = ImageDraw.Draw(img)
    d.ellipse([200, 90, 440, 390], fill=skin,
              outline=(200, 160, 140) if not dark else (60, 50, 46), width=2)
    d.ellipse([255, 190, 295, 212], fill=(90, 70, 70))
    d.ellipse([345, 190, 385, 212], fill=(90, 70, 70))
    d.arc([275, 280, 365, 330], 10, 170, fill=(190, 110, 110), width=4)
    d.text((16, 16), "RECORDER FRAME", fill=(120, 90, 80))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--bridge", default=None)
    args = ap.parse_args()

    t0 = time.monotonic()
    events: list[dict] = []
    stop = asyncio.Event()
    frame_count = [0]

    conn = await BridgeClient.connect(role="app", client_id="recorder-app",
                                      caps=["render", "frame", "coaching", "assets_url", "progress"],
                                      url=args.bridge)
    events.append({"t": round(time.monotonic() - t0, 3), "kind": "app_ready"})
    print(f"[recorder] 已连接 bridge，开始录制 {args.duration}s …", flush=True)

    def on_message(msg: dict) -> None:
        asyncio.get_running_loop().create_task(handle(msg))

    async def handle(msg: dict) -> None:
        t = round(time.monotonic() - t0, 3)
        mtype = msg.get("type")
        ref = msg.get("ref")
        if mtype == "apply_spec" and msg.get("assets_url"):
            # 资产侧车 → 取回并内联，离线渲染器不依赖 bridge 在线
            inl = dict(msg.get("assets") or {})
            for k, url in msg["assets_url"].items():
                try:
                    inl[k] = base64.b64encode(download_url(url)).decode()
                except Exception as e:  # noqa: BLE001
                    print(f"[recorder] 资产下载失败 {k}: {e}", flush=True)
            msg = dict(msg, assets=inl)
            msg.pop("assets_url", None)
        if mtype in {"apply_spec", "set_intensity", "clear_makeup", "coaching", "ping",
                     "request_frame"}:
            events.append({"t": t, "kind": mtype, "msg": msg})
            await conn.send({"type": "ack", "ref": ref, "status": "ok"})
            if mtype == "request_frame":
                frame_count[0] += 1
                dark = frame_count[0] == 3      # 第 3 次取帧给暗帧 → 演示光线提醒
                await conn.send({"type": "frame", "ref": ref,
                                 "data": base64.b64encode(synthetic_frame(dark)).decode(),
                                 "w": 640, "h": 480})
            summary = {"apply_spec": lambda m: f"spec={m.get('spec', {}).get('name')} "
                                              f"assets={len(m.get('assets', {}))} "
                                              f"splats={len(m.get('spec', {}).get('splat_layers') or [])}",
                       "coaching": lambda m: f"text={m.get('text')!r}"
                                             + (f" progress={m.get('progress')}" if m.get('progress') is not None else "")}.get(mtype)
            print(f"[recorder] {t:6.2f}s  {mtype:14s} {summary(msg) if summary else ''}",
                  flush=True)

    conn.on_message = on_message
    try:
        await asyncio.wait_for(stop.wait(), timeout=args.duration)
    except asyncio.TimeoutError:
        pass
    finally:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"duration": args.duration, "events": events},
                                  ensure_ascii=False), encoding="utf-8")
        await conn.close()
        print(f"[recorder] 已写出 {len(events)} 条事件 → {out}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
