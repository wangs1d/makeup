#!/usr/bin/env python3
"""BridgeClient — 本 skill 各脚本共用的 Bridge WebSocket 客户端封装。

Agent 侧脚本（apply_spec / live_coach / mock 客户端）都通过它连接 bridge_server：
处理 hello 握手、ack 应答匹配（按 ref）、frame 接收（按 ref）、tracking_state 事件回调、
断线自动重连（指数退避）、资产 HTTP 上传（bridge 1.1 资产侧车）。

Bridge 地址来源：环境变量 MAKEUP_BRIDGE_URL，默认 ws://127.0.0.1:8765。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
import uuid

try:
    import websockets
except ImportError:
    print("缺少依赖：pip install 'websockets>=13.0'", file=sys.stderr)
    sys.exit(1)

DEFAULT_BRIDGE_URL = os.environ.get("MAKEUP_BRIDGE_URL", "ws://127.0.0.1:8765")


class BridgeError(RuntimeError):
    pass


class BridgeClient:
    def __init__(self, ws, role: str, client_id: str, hello: dict, caps: list, url: str) -> None:
        self.ws = ws
        self.role = role
        self.client_id = client_id
        self.hello = hello
        self.caps = caps
        self.url = url
        self.on_tracking = None        # callable(dict)
        self.on_message = None         # callable(dict)，除 ack/frame 外的所有消息
        self.on_reconnect = None       # callable()，重连成功后回调
        self._acks: dict = {}          # ref -> asyncio.Queue
        self._frames: asyncio.Queue = asyncio.Queue()       # 无 ref 的 frame（兼容）
        self._frames_by_ref: dict = {}                       # ref -> asyncio.Queue
        self._reader_task: asyncio.Task | None = None
        self._closed = False
        self.alive = True
        self.reconnects = 0

    # ---------- 生命周期 ----------

    @classmethod
    async def connect(cls, role: str = "agent", client_id: str | None = None,
                      caps: list | None = None, url: str | None = None,
                      timeout: float = 8.0, retries: int = 0, backoff: float = 1.0) -> "BridgeClient":
        """连接并握手。retries>0 时连接失败按指数退避重试（长期运行脚本用）。"""
        url = url or DEFAULT_BRIDGE_URL
        client_id = client_id or f"{role}-{uuid.uuid4().hex[:8]}"
        caps = caps or []
        attempt = 0
        while True:
            try:
                ws, hello = await cls._open(url, role, client_id, caps, timeout)
                self = cls(ws, role, client_id, hello, caps, url)
                self._reader_task = asyncio.create_task(self._reader())
                return self
            except BridgeError:
                if attempt >= retries:
                    raise
                attempt += 1
                await asyncio.sleep(min(backoff * (2 ** (attempt - 1)), 15.0))

    @staticmethod
    async def _open(url: str, role: str, client_id: str, caps: list, timeout: float):
        try:
            ws = await asyncio.wait_for(websockets.connect(url, max_size=32 * 1024 * 1024), timeout)
        except (asyncio.TimeoutError, OSError) as e:
            raise BridgeError(f"无法连接 Bridge {url}：{e}\n"
                              f"请先启动：python bridge_server.py") from e
        except Exception as e:  # websockets.InvalidStatus 等：端口被别的 HTTP 服务占用
            raise BridgeError(f"Bridge {url} 握手被拒绝（端口可能被其他服务占用，可换 --port 并设 "
                              f"MAKEUP_BRIDGE_URL）：{e}") from e
        await ws.send(json.dumps({"type": "hello", "role": role, "client_id": client_id,
                                  "caps": caps}))
        try:
            hello = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        except (asyncio.TimeoutError, OSError) as e:
            await ws.close()
            raise BridgeError(f"Bridge 握手失败：{e}") from e
        if hello.get("type") != "hello_ok":
            await ws.close()
            raise BridgeError(f"Bridge 握手被拒绝：{hello}")
        return ws, hello

    async def reconnect(self, retries: int = 1_000_000, backoff: float = 1.0) -> None:
        """断线后重连（保留回调与 client_id）。"""
        if self._closed:
            raise BridgeError("客户端已关闭")
        attempt = 0
        while True:
            try:
                ws, hello = await self._open(self.url, self.role, self.client_id, self.caps, 8.0)
                self.ws, self.hello, self.alive = ws, hello, True
                self.reconnects += 1
                self._reader_task = asyncio.create_task(self._reader())
                if self.on_reconnect:
                    await self._emit(self.on_reconnect, None)
                return
            except BridgeError:
                if attempt >= retries or self._closed:
                    raise
                attempt += 1
                await asyncio.sleep(min(backoff * (2 ** (attempt - 1)), 15.0))

    async def ensure_connected(self) -> None:
        if not self.alive and not self._closed:
            await self.reconnect()

    async def _reader(self) -> None:
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = msg.get("type")
                ref = msg.get("ref")
                if t == "frame":
                    q = self._frames_by_ref.pop(ref, None) if ref is not None else None
                    await (q or self._frames).put(msg)
                    continue
                # 带匹配 ref 的回包（ack / status 等）一律解析对应等待者
                if ref is not None and ref in self._acks:
                    q = self._acks.pop(ref)
                    await q.put(msg)
                    if self.on_message:
                        await self._emit(self.on_message, msg)
                    continue
                if t == "tracking_state" and self.on_tracking:
                    await self._emit(self.on_tracking, msg)
                if self.on_message:
                    await self._emit(self.on_message, msg)
        except Exception:
            pass
        finally:
            self.alive = False
            # 唤醒所有等待者，避免挂死
            for q in list(self._acks.values()):
                await q.put({"type": "ack", "status": "error", "error": "connection_lost"})
            self._acks.clear()

    @staticmethod
    async def _emit(fn, msg) -> None:
        try:
            r = fn(msg) if msg is not None else fn()
            if hasattr(r, "__await__"):
                await r
        except Exception:
            pass

    async def close(self) -> None:
        self._closed = True
        if self._reader_task:
            self._reader_task.cancel()
        try:
            await self.ws.close()
        except Exception:
            pass

    # ---------- 属性 ----------

    @property
    def asset_base_url(self) -> str | None:
        return self.hello.get("asset_base_url")

    @property
    def app_ids(self) -> list[str]:
        return list(self.hello.get("app_ids") or [])

    # ---------- 消息 ----------

    async def send(self, msg: dict, to: str | None = None) -> None:
        if to:
            msg = dict(msg, to=to)
        await self.ws.send(json.dumps(msg))

    async def request(self, msg: dict, timeout: float = 10.0, to: str | None = None) -> dict:
        """发送消息并等待 app 端 ack；ref 自动生成。返回 ack 字典。"""
        ref = uuid.uuid4().hex[:12]
        msg = dict(msg, ref=ref)
        q: asyncio.Queue = asyncio.Queue()
        self._acks[ref] = q
        await self.send(msg, to=to)
        try:
            ack = await asyncio.wait_for(q.get(), timeout)
        except asyncio.TimeoutError:
            self._acks.pop(ref, None)
            raise BridgeError(f"等待 {msg['type']} 的 ack 超时（{timeout}s），App 可能未响应") from None
        if ack.get("status") != "ok":
            raise BridgeError(f"{msg['type']} 失败：{ack.get('error', ack)}")
        return ack

    async def request_frame(self, quality: int = 80, timeout: float = 12.0,
                            to: str | None = None) -> bytes:
        """请求一帧摄像头画面，返回 JPEG 字节。frame 按 ref 匹配；老 App 不带 ref 时走兼容队列。"""
        ref = uuid.uuid4().hex[:12]
        fq: asyncio.Queue = asyncio.Queue()
        self._frames_by_ref[ref] = fq
        aq: asyncio.Queue = asyncio.Queue()
        self._acks[ref] = aq
        await self.send({"type": "request_frame", "quality": quality, "ref": ref}, to=to)
        try:
            ack = await asyncio.wait_for(aq.get(), timeout / 2)
        except asyncio.TimeoutError:
            self._acks.pop(ref, None)
            self._frames_by_ref.pop(ref, None)
            raise BridgeError(f"等待 request_frame 的 ack 超时（{timeout / 2:g}s），App 可能未响应") from None
        if ack.get("status") != "ok":
            self._frames_by_ref.pop(ref, None)
            raise BridgeError(f"request_frame 失败：{ack.get('error', ack)}")

        async def first_frame():
            t1 = asyncio.create_task(fq.get())
            t2 = asyncio.create_task(self._frames.get())
            done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()
            return done.pop().result()

        try:
            msg = await asyncio.wait_for(first_frame(), timeout)
        except asyncio.TimeoutError:
            self._frames_by_ref.pop(ref, None)
            raise BridgeError("App 未回传画面帧（可能 caps 不含 frame 或追踪未就绪）") from None
        return base64.b64decode(msg.get("data", ""))

    async def status(self, timeout: float = 5.0) -> dict:
        ref = uuid.uuid4().hex[:12]
        q: asyncio.Queue = asyncio.Queue()
        self._acks[ref] = q
        await self.send({"type": "status", "ref": ref})
        try:
            return await asyncio.wait_for(q.get(), timeout)
        except asyncio.TimeoutError:
            raise BridgeError("Bridge 状态查询超时") from None

    # ---------- 资产侧车 ----------

    @staticmethod
    def asset_key(name: str, data: bytes) -> str:
        return f"{hashlib.sha256(data).hexdigest()[:16]}/{os.path.basename(name)}"

    async def upload_asset(self, name: str, data: bytes) -> str:
        """上传到 bridge 资产 HTTP 侧车，返回可下载 URL。无侧车时抛 BridgeError。"""
        base = self.asset_base_url
        if not base:
            raise BridgeError("Bridge 未开启资产侧车（老版本 bridge 或 --asset-port 0）")
        key = self.asset_key(name, data)
        url = base + key

        def _put():
            req = urllib.request.Request(url, data=data, method="PUT",
                                         headers={"Content-Type": "application/octet-stream"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status

        try:
            status = await asyncio.get_running_loop().run_in_executor(None, _put)
        except (urllib.error.URLError, OSError) as e:
            raise BridgeError(f"资产上传失败 {url}：{e}") from e
        if status not in (200, 201):
            raise BridgeError(f"资产上传返回 {status}：{url}")
        return url


def download_url(url: str, timeout: float = 30.0) -> bytes:
    """同步下载（app 角色的 Python 客户端如 mock_app/recorder_app 用）。"""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def fail(msg: str, code: int = 1) -> None:
    print(f"[makeup] {msg}", file=sys.stderr)
    sys.exit(code)
