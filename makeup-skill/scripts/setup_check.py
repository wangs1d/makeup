#!/usr/bin/env python3
"""环境自检：依赖、VLM 配置、Bridge、试妆 App、摄像头、预设可用性。

用法：python setup_check.py [--bridge ws://127.0.0.1:8765] [--json]
退出码：0 全部必需项通过；1 有必需项缺失。可选项目缺失不影响退出码。
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

OK, WARN, BAD = "✓", "!", "✗"


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


async def check_bridge(url: str) -> tuple[bool, int, str]:
    try:
        from bridge_common import BridgeClient
        conn = await BridgeClient.connect(role="agent", client_id="setup-check", url=url,
                                          timeout=4.0)
        try:
            st = await conn.status()
            assets = "，资产侧车 " + ("在线" if st.get("asset_base_url") else "关闭")
            return True, int(st.get("apps", 0)), (f"在线 {st.get('bridge', '')}（app 连接数 {st.get('apps', 0)}，"
                                                 f"agent {st.get('agents', 0)}{assets}）")
        finally:
            await conn.close()
    except Exception as e:  # noqa: BLE001
        text = str(e)
        if "握手被拒绝" in text or "rejected" in text or "403" in text or "404" in text:
            return False, 0, (f"端口被其他服务占用（不是 makeup bridge）：{url}。"
                              f"换端口启动：python scripts/bridge_server.py --port 8865，并 "
                              f"set MAKEUP_BRIDGE_URL=ws://127.0.0.1:8865")
        return False, 0, f"不可用：{text.splitlines()[0]}"


def check_camera() -> tuple[bool, str]:
    if not has_module("cv2"):
        return False, "opencv 未安装，无法检测"
    try:
        import cv2
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
        opened = cap.isOpened()
        ok = False
        if opened:
            ok, _ = cap.read()
        cap.release()
        return (True, "摄像头可用" ) if ok else (False, "摄像头打开失败或无画面（可忽略：帧可由试妆 App 提供）")
    except Exception as e:  # noqa: BLE001
        return False, f"检测异常：{e}"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default=None, help="Bridge 地址（默认 MAKEUP_BRIDGE_URL 或 ws://127.0.0.1:8765）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    results = []  # (level, item, detail)

    # 1. Python 依赖
    for mod, need in [("websockets", True), ("cv2", True), ("numpy", True),
                      ("PIL", True), ("openai", True), ("mediapipe", False)]:
        if has_module(mod):
            results.append((OK, f"依赖 {mod}", "已安装" + ("（可选）" if not need else "")))
        else:
            results.append((BAD if need else WARN, f"依赖 {mod}",
                            "未安装" + ("（必需，pip install -r requirements.txt）" if need
                                        else "（可选：face_tracker sidecar 用）")))

    # 2. VLM
    import vlm
    cfg = vlm.config_report()
    if cfg["api_key_set"]:
        results.append((OK, "VLM 配置", f"{cfg['model']} @ {cfg['base_url']}"))
    else:
        results.append((WARN, "VLM 配置", "未设 MAKEUP_VLM_API_KEY：能力二（解析）/三（实时指导）不可用，"
                                          "能力一（选妆试妆）不受影响"))

    # 3. 预设
    presets = sorted((SKILL_DIR / "presets").glob("*.json"))
    if presets:
        names = ", ".join(p.stem for p in presets)
        results.append((OK, "内置妆容", f"{len(presets)} 个：{names}"))
    else:
        results.append((BAD, "内置妆容", "presets/ 目录为空，安装不完整"))

    # 4. Bridge + App
    from bridge_common import DEFAULT_BRIDGE_URL
    ok, apps, detail = await check_bridge(args.bridge or DEFAULT_BRIDGE_URL)
    if ok:
        results.append((OK, "Bridge", detail))
        if apps > 0:
            results.append((OK, "试妆 App", f"{apps} 个已连接，可以试妆"))
        else:
            results.append((WARN, "试妆 App", "未连接：能力一/三需要 App，见 references/app-setup.md"))
    else:
        results.append((WARN, "Bridge", detail + " —— 需要时运行：python scripts/bridge_server.py"))

    # 5. 本机摄像头（可选，App 端有自己的取流）
    ok, detail = check_camera()
    results.append((OK if ok else WARN, "本机摄像头", detail))

    if args.json:
        print(json.dumps([{"level": l, "item": i, "detail": d} for l, i, d in results],
                         ensure_ascii=False, indent=2))
    else:
        print("Makeup Assistant 环境自检\n" + "=" * 46)
        for level, item, detail in results:
            print(f" {level} {item:<12} {detail}")
        print("=" * 46)
    must = [r for r in results if r[0] == BAD]
    return 1 if must else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
