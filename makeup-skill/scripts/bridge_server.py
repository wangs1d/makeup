#!/usr/bin/env python3
"""Makeup Assistant Bridge — Agent 脚本与试妆 App 之间的本地 WebSocket 消息总线 + 资产 HTTP 侧车。

角色：app（试妆 App，如 Unity 镜面客户端）与 agent（本 skill 的脚本）。
协议见 references/bridge-protocol.md（v1.1）：
    · 精确路由：agent 消息可带 `to`（app client_id）定向投递，缺省广播给所有 app；
      app 的 ack/frame 按 `ref` 回投给发出该请求的 agent（多 agent 并发不再错投）；
    · 资产侧车：http://127.0.0.1:<asset-port>/assets/<key> 支持 PUT 上传 / GET 下载，
      大资产不再 base64 内嵌进单条 WebSocket 消息（hello_ok 里下发 asset_base_url）。
心跳使用 WebSocket 传输层 ping（20s/15s），应用层 ping 消息用于业务探活。

用法：
    python bridge_server.py [--host 127.0.0.1] [--port 8765] [--asset-port 8768] [--quiet]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import re
import sys
import time
from collections import OrderedDict

try:
    from websockets.asyncio.server import serve
    from websockets.exceptions import ConnectionClosed
except ImportError:
    print("缺少依赖：pip install 'websockets>=13.0'", file=sys.stderr)
    sys.exit(1)

BRIDGE_VERSION = "makeup-bridge/1.1"

AGENT_TO_APP_TYPES = {"apply_spec", "clear_makeup", "set_intensity", "coaching", "request_frame", "ping"}
APP_BROADCAST_TYPES = {"tracking_state", "intensity_changed", "error"}
PENDING_TTL = 90.0            # ref → agent 映射保留秒数
LOG_QUIET = False


def log(msg: str) -> None:
    if LOG_QUIET:
        return
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------- 资产存储（内存 LRU） ----------------

class AssetStore:
    def __init__(self, max_bytes: int = 256 * 1024 * 1024) -> None:
        self._items: "OrderedDict[str, bytes]" = OrderedDict()
        self._total = 0
        self.max_bytes = max_bytes

    def put(self, key: str, data: bytes) -> None:
        if key in self._items:
            self._total -= len(self._items.pop(key))
        self._items[key] = data
        self._total += len(data)
        while self._total > self.max_bytes and len(self._items) > 1:
            _, old = self._items.popitem(last=False)
            self._total -= len(old)

    def get(self, key: str) -> bytes | None:
        data = self._items.get(key)
        if data is not None:
            self._items.move_to_end(key)
        return data

    def __len__(self) -> int:
        return len(self._items)

    @property
    def total_bytes(self) -> int:
        return self._total


KEY_RE = re.compile(r"^[A-Za-z0-9_\-./]{1,200}$")


class AssetHttpServer:
    """极简 HTTP/1.1：PUT/GET /assets/<key>、GET /health。仅供 localhost 的 agent/App 使用。"""

    def __init__(self, store: AssetStore, host: str, port: int) -> None:
        self.store = store
        self.host = host
        self.port = port
        self._server: asyncio.AbstractServer | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/assets/"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)

    async def close(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
            lines = head.decode("latin-1").split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            headers = {}
            for ln in lines[1:]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            length = int(headers.get("content-length", "0") or 0)
            body = await asyncio.wait_for(reader.readexactly(length), timeout=60) if length else b""
            status, ctype, payload = self._route(method, path, body)
            writer.write(f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {len(payload)}\r\n"
                         f"Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n".encode() + payload)
            await writer.drain()
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, ValueError, ConnectionError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    def _route(self, method: str, path: str, body: bytes) -> tuple[str, str, bytes]:
        if path == "/health":
            return "200 OK", "application/json", json.dumps({
                "bridge": BRIDGE_VERSION, "assets": len(self.store),
                "bytes": self.store.total_bytes}).encode()
        if not path.startswith("/assets/"):
            return "404 Not Found", "text/plain", b"not found"
        key = path[len("/assets/"):]
        if not KEY_RE.match(key) or ".." in key:
            return "400 Bad Request", "text/plain", b"bad key"
        if method == "PUT" or method == "POST":
            if not body:
                return "400 Bad Request", "text/plain", b"empty body"
            self.store.put(key, body)
            return "201 Created", "application/json", json.dumps(
                {"key": key, "url": self.base_url + key, "bytes": len(body)}).encode()
        if method == "GET" or method == "HEAD":
            data = self.store.get(key)
            if data is None:
                return "404 Not Found", "text/plain", b"no such asset"
            ctype = mimetypes.guess_type(key)[0] or "application/octet-stream"
            return "200 OK", ctype, data if method == "GET" else b""
        return "405 Method Not Allowed", "text/plain", b"method"


# ---------------- Bridge ----------------

class Bridge:
    def __init__(self, asset_base_url: str | None = None) -> None:
        self.clients: dict = {}            # ws -> {"role": str, "client_id": str, "caps": list}
        self.pending: dict = {}            # ref -> (agent ws, timestamp)
        self.frame_requester = None        # 最近请求 frame 的 agent ws（无 ref 兜底）
        self.asset_base_url = asset_base_url

    # ---------- 注册 ----------

    async def register(self, ws, msg: dict) -> None:
        role = msg.get("role")
        if role not in ("app", "agent"):
            await ws.send(json.dumps({"type": "error", "code": "bad_role",
                                      "message": "role must be app|agent"}))
            return
        client_id = msg.get("client_id") or f"peer-{id(ws)}"
        caps = msg.get("caps", [])
        # 同 role+client_id 顶掉旧连接
        for other, meta in list(self.clients.items()):
            if meta["role"] == role and meta["client_id"] == client_id and other is not ws:
                log(f"顶掉同 id 旧连接：{role}/{client_id}")
                self.clients.pop(other, None)
                self._cleanup_meta(other)
                try:
                    await other.close(code=4001, reason="replaced")
                except Exception:
                    pass
        self.clients[ws] = {"role": role, "client_id": client_id, "caps": caps}
        log(f"上线：{role}/{client_id} caps={caps}（apps={self.count('app')}, agents={self.count('agent')}）")
        hello_ok = {
            "type": "hello_ok", "role": role, "client_id": client_id,
            "bridge": BRIDGE_VERSION, "apps": self.count("app"),
            "app_ids": self.ids("app"),
        }
        if self.asset_base_url:
            hello_ok["asset_base_url"] = self.asset_base_url
        await ws.send(json.dumps(hello_ok))

    def _cleanup_meta(self, ws) -> None:
        if self.frame_requester is ws:
            self.frame_requester = None
        for ref in [r for r, (w, _) in self.pending.items() if w is ws]:
            self.pending.pop(ref, None)

    def count(self, role: str) -> int:
        return sum(1 for m in self.clients.values() if m["role"] == role)

    def ids(self, role: str) -> list[str]:
        return [m["client_id"] for m in self.clients.values() if m["role"] == role]

    def meta(self, ws) -> dict:
        return self.clients.get(ws) or {}

    def _prune_pending(self) -> None:
        now = time.monotonic()
        for ref in [r for r, (_, t) in self.pending.items() if now - t > PENDING_TTL]:
            self.pending.pop(ref, None)

    # ---------- 路由 ----------

    async def handle_agent(self, ws, msg: dict) -> None:
        mtype = msg.get("type")
        ref = msg.get("ref")
        to = msg.get("to")
        apps = [w for w, m in self.clients.items() if m["role"] == "app"
                and (to is None or m["client_id"] == to)]
        if mtype == "ping" and not to:
            await ws.send(json.dumps({"type": "ack", "ref": ref, "status": "ok",
                                      "bridge": BRIDGE_VERSION, "apps": self.count("app"),
                                      "app_ids": self.ids("app")}))
            return
        if mtype not in AGENT_TO_APP_TYPES:
            await ws.send(json.dumps({"type": "ack", "ref": ref, "status": "error",
                                      "error": f"unknown_type:{mtype}"}))
            return
        if not apps:
            await ws.send(json.dumps({"type": "ack", "ref": ref, "status": "error",
                                      "error": "no_app_connected" if to is None else f"no_such_app:{to}"}))
            return
        if ref is not None:
            self._prune_pending()
            self.pending[ref] = (ws, time.monotonic())
        if mtype == "request_frame":
            self.frame_requester = ws
        raw = json.dumps(msg)
        for app in apps:
            try:
                await app.send(raw)
            except ConnectionClosed:
                pass

    async def handle_app(self, ws, msg: dict) -> None:
        mtype = msg.get("type")
        ref = msg.get("ref")
        src = self.meta(ws).get("client_id")
        if mtype == "ack":
            target = self.pending.get(ref, (None, 0))[0] if ref is not None else None
            if target is None:
                log(f"丢弃无处投递的 ack：ref={ref}")
                return
            msg = dict(msg, **{"from": src})
            try:
                await target.send(json.dumps(msg))
            except ConnectionClosed:
                pass
        elif mtype == "frame":
            target = self.pending.get(ref, (None, 0))[0] if ref is not None else None
            if target is None:
                target = self.frame_requester
            if target is None:
                log("收到 frame 但没有请求方，丢弃")
                return
            msg = dict(msg, **{"from": src})
            try:
                await target.send(json.dumps(msg))
            except ConnectionClosed:
                pass
        elif mtype in APP_BROADCAST_TYPES:
            raw = json.dumps(dict(msg, **{"from": src}))
            for w, m in list(self.clients.items()):
                if m["role"] == "agent":
                    try:
                        await w.send(raw)
                    except ConnectionClosed:
                        pass
        else:
            log(f"忽略 app 端未知消息类型：{mtype}")

    # ---------- 连接处理 ----------

    async def handler(self, ws) -> None:
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict) or "type" not in msg:
                    continue
                mtype = msg["type"]
                m = self.meta(ws)
                role = m.get("role")
                if mtype == "hello":
                    await self.register(ws, msg)
                    continue
                if role is None:
                    await ws.send(json.dumps({"type": "error", "code": "no_hello",
                                              "message": "send hello first"}))
                    continue
                if mtype == "status":
                    await ws.send(json.dumps({"type": "status", "ref": msg.get("ref"),
                                              "bridge": BRIDGE_VERSION,
                                              "apps": self.count("app"),
                                              "agents": self.count("agent"),
                                              "app_ids": self.ids("app"),
                                              "asset_base_url": self.asset_base_url}))
                elif role == "agent":
                    await self.handle_agent(ws, msg)
                elif role == "app":
                    await self.handle_app(ws, msg)
        except ConnectionClosed:
            pass
        finally:
            meta = self.clients.pop(ws, None)
            self._cleanup_meta(ws)
            if meta:
                log(f"下线：{meta['role']}/{meta['client_id']}"
                    f"（apps={self.count('app')}, agents={self.count('agent')}）")


async def main() -> None:
    global LOG_QUIET
    ap = argparse.ArgumentParser(description="Makeup Assistant Bridge")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--asset-port", type=int, default=8768, help="资产 HTTP 端口（0=关闭）")
    ap.add_argument("--quiet", action="store_true", help="不打印连接日志")
    args = ap.parse_args()
    LOG_QUIET = args.quiet

    asset_http = None
    if args.asset_port:
        asset_http = AssetHttpServer(AssetStore(), args.host, args.asset_port)
        try:
            await asset_http.start()
        except OSError as e:
            print(f"资产 HTTP 端口 {args.asset_port} 不可用（{e}），资产将回退 base64 内嵌", file=sys.stderr)
            asset_http = None

    bridge = Bridge(asset_http.base_url if asset_http else None)
    try:
        async with serve(bridge.handler, args.host, args.port,
                         ping_interval=20, ping_timeout=15, max_size=32 * 1024 * 1024):
            print(f"Makeup Bridge {BRIDGE_VERSION} listening on ws://{args.host}:{args.port}"
                  + (f"  assets: {asset_http.base_url}" if asset_http else ""), flush=True)
            print("等待 app（试妆 App）与 agent（skill 脚本）接入… Ctrl+C 退出", flush=True)
            await asyncio.Future()  # run forever
    finally:
        if asset_http:
            await asset_http.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBridge 已退出")
