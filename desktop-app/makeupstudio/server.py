#!/usr/bin/env python3
"""server — 3DGS 查看器的本地静态服务器。

路由：
    /                → viewer/index.html
    /splat/*         → out_dir 下的 .splat / version.json
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"


class _Handler(BaseHTTPRequestHandler):
    out_dir: Path = Path("out")

    def do_GET(self):
        try:
            if self.path.startswith("/splat/"):
                root = self.out_dir / "splat"
                rel = self.path[len("/splat/"):].split("?")[0] or "version.json"
            else:
                root = _VIEWER_DIR
                route = self.path.split("?")[0]          # 忽略查询参数（如 ?embed=1）
                rel = "index.html" if route in ("/", "") else route.lstrip("/")
            path = (root / rel).resolve()
            if root not in path.parents and path != root / rel:
                self.send_error(403)
                return
            data = path.read_bytes()
            ctype = {".html": "text/html", ".js": "text/javascript", ".json": "application/json",
                     ".splat": "application/octet-stream", ".ply": "application/octet-stream",
                     ".png": "image/png"}.get(path.suffix, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (FileNotFoundError, IsADirectoryError):
            self.send_error(404)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def log_message(self, *args):
        pass


class ViewerServer:
    def __init__(self, out_dir: str | Path, port: int = 8791):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "splat").mkdir(exist_ok=True)
        _Handler.out_dir = self.out_dir
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self.port = port
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self):
        self._thread.start()

    def stop(self):
        self._httpd.shutdown()
