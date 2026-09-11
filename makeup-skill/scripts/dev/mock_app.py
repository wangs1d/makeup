#!/usr/bin/env python3
"""开发用 Mock App：模拟试妆 App 的 app 角色行为，用于在没有 Unity 端时联调 Bridge 协议。

行为：
- hello(role=app, caps=[render, frame, coaching, assets_url, progress])
- apply_spec（assets 内嵌 / assets_url 下载）/ clear_makeup / set_intensity → 打印摘要并回 ack ok
- request_frame → 回传一张合成 JPEG（有摄像头时优先用真实画面），frame 带 ref
- coaching → 打印（含 progress/step）
- ping → ack

用法：python dev/mock_app.py [--bridge ws://127.0.0.1:8765] [--use-camera] [--client-id mock-app]
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bridge_common import BridgeClient, download_url  # noqa: E402


def synthetic_frame() -> bytes:
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (640, 480), (196, 178, 166))
        d = ImageDraw.Draw(img)
        d.ellipse([200, 100, 440, 380], outline=(200, 160, 140), width=3)
        d.text((16, 16), "MOCK APP FRAME", fill=(120, 90, 80))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        return buf.getvalue()
    except ImportError:
        # 1x1 最小 JPEG 兜底
        return base64.b64decode(
            "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
            "HBwcJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPDs0NDT/wAALCAABAAEBAREA/8QAFAABAAAAAAAA"
            "AAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q==")


async def grab_camera_frame() -> bytes | None:
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None
    except ImportError:
        return None


def fetch_assets(msg: dict) -> dict[str, bytes]:
    """内嵌 base64 与 assets_url 两种资产形态都取回字节。"""
    out: dict[str, bytes] = {}
    for k, v in (msg.get("assets") or {}).items():
        out[k] = base64.b64decode(v)
    for k, url in (msg.get("assets_url") or {}).items():
        try:
            out[k] = download_url(url)
        except Exception as e:  # noqa: BLE001
            print(f"[mock-app] 资产下载失败 {k}: {e}")
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default=None)
    ap.add_argument("--use-camera", action="store_true")
    ap.add_argument("--client-id", default="mock-app")
    args = ap.parse_args()

    conn = await BridgeClient.connect(role="app", client_id=args.client_id,
                                      caps=["render", "frame", "coaching", "assets_url", "progress"],
                                      url=args.bridge)
    print(f"[mock-app] 已连接 bridge（apps 池）；hello={conn.hello}")

    def on_message(msg: dict) -> None:
        asyncio.get_running_loop().create_task(handle(msg))

    async def handle(msg: dict) -> None:
        t = msg.get("type")
        ref = msg.get("ref")
        if t == "apply_spec":
            spec = msg.get("spec", {})
            layers = [f"{l.get('region')}/{l.get('side')}" for l in spec.get("layers", [])]
            assets = fetch_assets(msg)
            print(f"[mock-app] 收到 apply_spec：name={spec.get('name')} "
                  f"intensity={msg.get('intensity', spec.get('intensity'))} layers={layers} "
                  f"assets={len(assets)}（{sum(map(len, assets.values())) // 1024} KB）"
                  f" splat_layers={len(spec.get('splat_layers') or [])}")
            await conn.send({"type": "ack", "ref": ref, "status": "ok"})
        elif t == "clear_makeup":
            print("[mock-app] 收到 clear_makeup（卸妆）")
            await conn.send({"type": "ack", "ref": ref, "status": "ok"})
        elif t == "set_intensity":
            print(f"[mock-app] 收到 set_intensity：{msg.get('value')}")
            await conn.send({"type": "ack", "ref": ref, "status": "ok"})
        elif t == "request_frame":
            frame = None
            if args.use_camera:
                frame = await grab_camera_frame()
            if frame is None:
                frame = synthetic_frame()
            await conn.send({"type": "ack", "ref": ref, "status": "ok"})
            await conn.send({"type": "frame", "ref": ref,
                             "data": base64.b64encode(frame).decode(), "w": 640, "h": 480})
        elif t == "coaching":
            icon = "⚠" if msg.get("priority") == "warn" else "💡"
            prog = f" [{msg.get('step')}/{msg.get('steps')} {msg.get('progress', 0):.0%}]" \
                if msg.get("progress") is not None else ""
            print(f"[mock-app] {icon} 指导{prog}：{msg.get('text')}")
        elif t == "ping":
            await conn.send({"type": "ack", "ref": ref, "status": "ok"})

    conn.on_message = on_message
    try:
        await asyncio.Future()  # run until Ctrl+C
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await conn.close()
        print("[mock-app] 已退出")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
