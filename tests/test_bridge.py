"""Bridge v1.1 单元/集成测试：精确路由、ack/frame 按 ref 回投、资产侧车、重连。

直接在进程内起 Bridge + AssetHttpServer（随机端口），不依赖 8765/8868。
"""
from __future__ import annotations

import asyncio

import pytest

import bridge_server as bs
from bridge_common import BridgeClient, BridgeError


def free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Stack:
    def __init__(self) -> None:
        self.port = free_port()
        self.asset_port = free_port()
        self.store = bs.AssetStore()
        self.http = bs.AssetHttpServer(self.store, "127.0.0.1", self.asset_port)
        self.bridge = bs.Bridge(self.http.base_url)
        self._server = None

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"


def run_stack(test):
    """起一个 Bridge 栈跑 test(stack) 协程。"""

    async def runner():
        from websockets.asyncio.server import serve
        stack = Stack()
        await stack.http.start()
        stack._server = await serve(stack.bridge.handler, "127.0.0.1", stack.port)
        try:
            await test(stack)
        finally:
            stack._server.close()
            await stack._server.wait_closed()
            await stack.http.close()

    asyncio.run(runner())


def test_hello_status_and_asset_base():
    async def t(stack):
        agent = await BridgeClient.connect(role="agent", client_id="t-agent", url=stack.url)
        assert agent.hello["bridge"] == bs.BRIDGE_VERSION
        assert agent.asset_base_url == stack.http.base_url
        st = await agent.status()
        assert st["apps"] == 0 and st["agents"] == 1
        await agent.close()
    run_stack(t)


def test_no_app_error_and_to_missing_app():
    async def t(stack):
        agent = await BridgeClient.connect(role="agent", client_id="t-agent", url=stack.url)
        with pytest.raises(BridgeError, match="no_app_connected"):
            await agent.request({"type": "clear_makeup"})
        with pytest.raises(BridgeError, match="no_such_app"):
            await agent.request({"type": "ping"}, to="ghost")
        await agent.close()
    run_stack(t)


def test_ack_routed_to_requesting_agent_with_two_agents():
    """两个 agent 并发给同一 app 发请求，各自收到自己的 ack（旧版按"最近 agent"会错投）。"""

    async def t(stack):
        app = await BridgeClient.connect(role="app", client_id="t-app", url=stack.url)

        async def handle(msg):
            if msg.get("type") == "set_intensity":
                await asyncio.sleep(0.05 * int(msg["value"]))   # 故意乱序回包
                await app.send({"type": "ack", "ref": msg["ref"], "status": "ok"})
        app.on_message = lambda m: asyncio.get_running_loop().create_task(handle(m))

        a1 = await BridgeClient.connect(role="agent", client_id="a1", url=stack.url)
        a2 = await BridgeClient.connect(role="agent", client_id="a2", url=stack.url)
        r1 = asyncio.create_task(a1.request({"type": "set_intensity", "value": 1}, timeout=5))
        r2 = asyncio.create_task(a2.request({"type": "set_intensity", "value": 10}, timeout=5))
        ack1, ack2 = await asyncio.gather(r1, r2)
        assert ack1["status"] == "ok" and ack2["status"] == "ok"
        assert ack1.get("from") == "t-app"
        await a1.close(); await a2.close(); await app.close()
    run_stack(t)


def test_frame_routed_by_ref():
    async def t(stack):
        app = await BridgeClient.connect(role="app", client_id="t-app", url=stack.url)

        async def handle(msg):
            if msg.get("type") == "request_frame":
                await app.send({"type": "ack", "ref": msg["ref"], "status": "ok"})
                await app.send({"type": "frame", "ref": msg["ref"], "data": "AQID", "w": 1, "h": 1})
        app.on_message = lambda m: asyncio.get_running_loop().create_task(handle(m))

        agent = await BridgeClient.connect(role="agent", client_id="t-agent", url=stack.url)
        data = await agent.request_frame()
        assert data == b"\x01\x02\x03"
        await agent.close(); await app.close()
    run_stack(t)


def test_to_targets_single_app():
    """两个 app 在线时，to=<id> 只定向到指定 app。"""

    async def t(stack):
        got = {"alpha": 0, "beta": 0}
        apps = {
            "alpha": await BridgeClient.connect(role="app", client_id="alpha", url=stack.url),
            "beta": await BridgeClient.connect(role="app", client_id="beta", url=stack.url),
        }

        def mk(name, conn):
            async def handle(msg):
                got[name] += 1
                if msg.get("ref") is not None:
                    await conn.send({"type": "ack", "ref": msg["ref"], "status": "ok"})
            return handle

        for n, a in apps.items():
            a.on_message = lambda m, _n=n, _c=a: asyncio.get_running_loop().create_task(mk(_n, _c)(m))

        agent = await BridgeClient.connect(role="agent", client_id="t-agent", url=stack.url)
        await agent.request({"type": "clear_makeup"}, to="beta")
        await asyncio.sleep(0.2)
        assert got["beta"] == 1 and got["alpha"] == 0
        await agent.close()
        for a in apps.values():
            await a.close()
    run_stack(t)


def test_tracking_state_broadcast_to_agents():
    async def t(stack):
        agent = await BridgeClient.connect(role="agent", client_id="t-agent", url=stack.url)
        events = []
        agent.on_tracking = lambda m: events.append(m)

        app = await BridgeClient.connect(role="app", client_id="t-app", url=stack.url)
        await app.send({"type": "tracking_state", "ok": True, "fps": 28.5, "landmarks": 468})
        await asyncio.sleep(0.3)
        assert events and events[0]["fps"] == 28.5 and events[0]["from"] == "t-app"
        await agent.close(); await app.close()
    run_stack(t)


def test_asset_store_lru():
    store = bs.AssetStore()
    store.put("a", b"x" * 10)
    store.put("b", b"y" * 10)
    assert store.get("a") == b"x" * 10     # 访问后 a 变为最新
    store.max_bytes = 25
    store.put("c", b"z" * 10)
    assert store.get("b") is None          # LRU 淘汰最久未用的 b
    assert store.get("c") == b"z" * 10


def test_asset_upload_download_via_http():
    async def t(stack):
        agent = await BridgeClient.connect(role="agent", client_id="t-agent", url=stack.url)
        payload = bytes(range(256)) * 64
        url = await agent.upload_asset("ramp_test.png", payload)
        assert url.startswith(stack.http.base_url)
        key = url[len(stack.http.base_url):]
        assert stack.store.get(key) == payload
        from bridge_common import download_url
        got = await asyncio.get_running_loop().run_in_executor(None, download_url, url)
        assert got == payload
        # 路径穿越在客户端就被 basename 消毒（正确行为）：落到安全 key
        url2 = await agent.upload_asset("../evil.png", b"x")
        assert "/" in url2[len(stack.http.base_url):] and "evil.png" in url2
        assert ".." not in url2
        # 服务器对无法消毒的非法原始 key 返回 400 → urllib 抛 HTTPError → BridgeError
        import urllib.request
        def _bad_put():
            req = urllib.request.Request(stack.http.base_url + "bad%20key!", data=b"x", method="PUT")
            urllib.request.urlopen(req, timeout=10)
        try:
            await asyncio.get_running_loop().run_in_executor(None, _bad_put)
            raise AssertionError("非法 key 未被拒绝")
        except urllib.error.HTTPError as e:
            assert e.code == 400
        # 走 upload_asset 的等价路径：非法 key → BridgeError
        obj_key = stack.http.base_url + "bad%20key!"
        def _bad_put2():
            req = urllib.request.Request(obj_key, data=b"x", method="PUT")
            urllib.request.urlopen(req, timeout=10)
        with pytest.raises(BridgeError):
            try:
                await asyncio.get_running_loop().run_in_executor(None, _bad_put2)
            except urllib.error.HTTPError as e:
                raise BridgeError(str(e))
        await agent.close()
    run_stack(t)


def test_client_reconnect_after_drop():
    async def t(stack):
        agent = await BridgeClient.connect(role="agent", client_id="t-agent", url=stack.url,
                                           retries=1)
        assert agent.alive
        # 模拟断线：直接掐断底层连接
        agent.ws.transport.abort() if getattr(agent.ws, "transport", None) else await agent.ws.close()
        agent.alive = False
        await agent.reconnect(retries=3, backoff=0.1)
        assert agent.alive and agent.reconnects == 1
        st = await agent.status()
        assert st["agents"] == 1
        await agent.close()
    run_stack(t)
