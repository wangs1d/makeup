"""engines — 本地重建引擎（FFmpeg / COLMAP / Brush）的发现与状态检查。

查找顺序（与 ooosplat 的约定兼容）：
    1. 显式环境变量 OOOSPLAT_FFMPEG / OOOSPLAT_COLMAP / OOOSPLAT_BRUSH；
    2. OOOSPLAT_ENGINE_DIR/<name>(.exe)；ooosplat 桌面版把引擎集中在该目录；
    3. 系统 PATH。
服务器部署时后端机只需保证其中一种方式可用。
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ENGINE_NAMES = ("ffmpeg", "ffprobe", "colmap", "brush")


@dataclass
class EngineStatus:
    name: str
    path: str | None
    version: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.path is not None


def _find_in_root(root: Path, name: str) -> Path | None:
    candidates = [root / name, root / f"{name}.exe", root / f"{name}.bat",
                  root / "bin" / f"{name}.exe"]
    candidates += sorted(root.glob(f"*/{name}.bat")) + sorted(root.glob(f"*/bin/{name}.exe"))
    # OOOSplat 0.4.0 布局：<engine_dir>/<dir>/<exe>，目录名可不等于引擎名
    # （ffmpeg/ffprobe.exe、brush/brush_app.exe）；colmap 走 COLMAP.bat
    candidates += sorted(root.glob(f"*/{name}.exe"))
    candidates += sorted(root.glob(f"*/{name}_app.exe"))
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _default_engine_roots() -> list[Path]:
    """无需配置即可用的引擎目录：项目 .engines 与 OOOSplat 桌面版安装位置。"""
    roots = [Path(__file__).resolve().parents[3] / ".engines"]
    pf = os.environ.get("ProgramFiles")
    if pf:
        roots.append(Path(pf) / "OOOSplat" / "engines")
    lad = os.environ.get("LOCALAPPDATA")
    if lad:
        roots += [Path(lad) / "Programs" / "OOOSplat" / "engines",
                  Path(lad) / "OOOSplat" / "engines"]
    return roots


def find_engine(name: str) -> Path | None:
    """按 环境变量 → OOOSPLAT_ENGINE_DIR → 默认位置 → PATH 的顺序解析引擎。"""
    if name not in ENGINE_NAMES:
        raise ValueError(f"未知引擎: {name}")
    env_key = f"OOOSPLAT_{name.upper()}"
    explicit = os.environ.get(env_key)
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    engine_dir = os.environ.get("OOOSPLAT_ENGINE_DIR")
    if engine_dir:
        found = _find_in_root(Path(engine_dir), name)
        if found is not None:
            return found
    for root in _default_engine_roots():
        found = _find_in_root(root, name)
        if found is not None:
            return found
    which = shutil.which(name)
    return Path(which) if which else None


def check_engine(name: str) -> EngineStatus:
    path = find_engine(name)
    if path is None:
        hint = {
            "ffmpeg": "可选（缺失时用 OpenCV 抽帧）；如需安装：`winget install Gyan.FFmpeg`",
            "ffprobe": "随 FFmpeg 一起安装",
            "colmap": "安装 OOOSplat（自带 COLMAP），或从 https://github.com/colmap/colmap/releases 下载（慢速网络可用 gh-proxy.com 镜像）",
            "brush": "安装 OOOSplat，或从 https://github.com/ArthurBrussee/brush/releases 下载 brush-app",
        }[name]
        return EngineStatus(name, None, None, f"未找到（{hint}）")
    try:
        # brush 的 --version 走 stderr；统一合并捕获
        out = subprocess.run([str(path), "--version"], capture_output=True,
                             text=True, timeout=20, encoding="utf-8", errors="replace")
        ver = (out.stdout or "").strip() or (out.stderr or "").strip()
        return EngineStatus(name, str(path), ver.splitlines()[0][:120] if ver else None)
    except (OSError, subprocess.TimeoutExpired) as e:
        return EngineStatus(name, str(path), None, f"无法执行: {e}")


def status_all() -> dict[str, EngineStatus]:
    return {name: check_engine(name) for name in ENGINE_NAMES}
