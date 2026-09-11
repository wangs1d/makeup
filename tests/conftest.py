"""pytest 公共配置：把 skill 脚本与 unity 工具目录加入 sys.path。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "makeup-skill" / "scripts", ROOT / "unity-app" / "tools", ROOT / "preview"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def presets(root) -> dict:
    import json
    out = {}
    for p in sorted((root / "makeup-skill" / "presets").glob("*.json")):
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out
