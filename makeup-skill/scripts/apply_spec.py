#!/usr/bin/env python3
"""能力一/二的收尾动作：把妆容 spec 下发到试妆 App 渲染。

用法：
    python apply_spec.py --spec presets/daily-natural.json
    python apply_spec.py --spec out/parsed/makeup_spec.json --intensity 0.6 --bake
    python apply_spec.py --spec presets/date-rose.json --only lipstick,blush
    python apply_spec.py --clear            # 卸妆还原

选项：
    --intensity 0.0~1.0  覆盖 spec 的整体浓度
    --only region,region 只启用指定部位（单品试妆）
    --clear              不发 spec，直接下发卸妆
    --assets <baked目录>  附带 bake_assets 产物
    --bake               先自动烘焙到 <spec 同级>/baked/<name>/ 再附带（解析出的新妆容推荐）
    --to <client_id>     只发给指定 App（多 App 时）

资产传输：总量 ≤ MAKEUP_ASSET_INLINE_MAX（默认 1MB）时 base64 内嵌随消息；更大时上传到
bridge 资产 HTTP 侧车，消息里只带 assets_url（bridge 1.1+；老 bridge 自动回退内嵌，上限 24MB）。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bridge_common import BridgeClient, BridgeError, fail  # noqa: E402

ASSETS_MAX_BYTES = 24 * 1024 * 1024
INLINE_MAX_BYTES = int(os.environ.get("MAKEUP_ASSET_INLINE_MAX", str(1 * 1024 * 1024)))


def collect_assets(assets_dir: str | None) -> dict[str, bytes]:
    """读取 bake_assets 产物目录，返回 {文件名: bytes}；超过大小上限则拒绝。"""
    if not assets_dir:
        return {}
    d = Path(assets_dir)
    manifest = d / "manifest.json"
    if not manifest.exists():
        fail(f"--assets 目录缺少 manifest.json（先用 bake_assets.py 烘焙）：{d}")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assets: dict[str, bytes] = {}
    total = 0
    for name in manifest_data.get("files", []):
        p = d / name
        if not p.exists():
            fail(f"manifest 引用的资产缺失：{p}")
        raw = p.read_bytes()
        total += len(raw)
        if total > ASSETS_MAX_BYTES:
            fail("资产包过大（>24MB），请检查烘焙输出")
        assets[name] = raw
    print(f"[makeup] 附带资产 {len(assets)} 个（{total / 1024:.0f} KB）")
    return assets


def load_baked_spec(assets_dir: str | None) -> dict | None:
    """烘焙目录里的 baked_spec.json（带 baked 引用与 splat_layers）。"""
    if not assets_dir:
        return None
    p = Path(assets_dir) / "baked_spec.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_spec(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        fail(f"spec 文件不存在：{p}")
    try:
        spec = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        fail(f"spec JSON 解析失败：{e}")
    if str(spec.get("spec_version", "")).split(".")[0] != "1":
        fail(f"不支持的 spec_version：{spec.get('spec_version')!r}（需要 1.x）")
    if not isinstance(spec.get("layers"), list) or not spec["layers"]:
        fail("spec 缺少 layers 或为空")
    return spec


def auto_bake(spec_path: str, spec: dict) -> str:
    """烘焙到 MAKEUP_BAKE_DIR（默认系统临时目录）/makeup-baked/<name>，不污染 spec 所在目录。"""
    base = Path(os.environ.get("MAKEUP_BAKE_DIR") or (Path(tempfile.gettempdir()) / "makeup-baked"))
    out_dir = base / str(spec.get("name", "look"))
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "bake_assets.py"),
                        "--spec", spec_path, "--out", str(out_dir)],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        fail(f"自动烘焙失败：{r.stderr.strip() or r.stdout.strip()}")
    print(f"[makeup] 已烘焙资产 → {out_dir}")
    return str(out_dir)


async def attach_assets(conn: BridgeClient, payload: dict, assets: dict[str, bytes]) -> str:
    """按大小选择内嵌或上传；返回描述文字。"""
    total = sum(len(v) for v in assets.values())
    if total <= INLINE_MAX_BYTES or not conn.asset_base_url:
        payload["assets"] = {k: base64.b64encode(v).decode() for k, v in assets.items()}
        how = "内嵌"
        if total > INLINE_MAX_BYTES:
            how = "内嵌（bridge 无资产侧车，回退）"
    else:
        urls = {}
        for name, data in assets.items():
            urls[name] = await conn.upload_asset(name, data)
        payload["assets_url"] = urls
        how = f"HTTP 侧车（{len(urls)} 个 URL）"
    return how


async def run(args: argparse.Namespace) -> None:
    spec = None
    assets_dir = args.assets
    if not args.clear:
        spec = load_spec(args.spec)
        if args.bake and not assets_dir:
            assets_dir = auto_bake(args.spec, spec)
        baked = load_baked_spec(assets_dir)
        if baked:
            # 烘焙 spec 带 baked 引用与 splat_layers（App 端资产与溅射依赖它）
            spec = dict(baked, intensity=spec.get("intensity", baked.get("intensity", 0.8)))
        if args.only:
            wanted = {r.strip() for r in args.only.split(",") if r.strip()}
            kept = [l for l in spec["layers"] if l.get("region") in wanted]
            if not kept:
                fail(f"--only 的部位没有匹配的 layer。可选 region："
                     f"{sorted({l.get('region') for l in spec['layers']})}")
            spec["layers"] = kept
            if spec.get("splat_layers"):
                spec["splat_layers"] = [s for s in spec["splat_layers"] if s.get("region") in wanted]
            spec["description"] = f"[单品试妆 {'+'.join(sorted(wanted))}] " + spec.get("description", "")
        if args.intensity is not None:
            if not 0.0 <= args.intensity <= 1.0:
                fail("--intensity 取值 0.0~1.0")
            spec["intensity"] = round(args.intensity, 3)

    try:
        conn = await BridgeClient.connect(role="agent", client_id="apply-spec")
    except BridgeError as e:
        fail(str(e))

    try:
        st = await conn.status()
        if st.get("apps", 0) == 0:
            fail("当前没有试妆 App 连接。请按 references/app-setup.md 启动 App 后重试。")
        if args.to and args.to not in (st.get("app_ids") or []):
            fail(f"--to 指定的 App 未连接：{args.to}（在线：{st.get('app_ids')}）")
        if args.clear:
            await conn.request({"type": "clear_makeup"}, to=args.to)
            print("[makeup] 已卸妆还原 ✓")
        else:
            payload = {"type": "apply_spec", "spec": spec}
            if args.intensity is not None:
                payload["intensity"] = spec["intensity"]
            assets = collect_assets(assets_dir)
            how = ""
            if assets:
                how = await attach_assets(conn, payload, assets)
                payload["spec"]["baked"] = True
            await conn.request(payload, timeout=30.0 if assets else 15.0, to=args.to)
            name = spec.get("name", "未命名妆容")
            layers = ", ".join(sorted({l.get("region", "?") for l in spec["layers"]}))
            print(f"[makeup] 已下发妆容「{name}」✓（部位：{layers}；浓度 {spec.get('intensity')}"
                  + (f"；资产 {how}" if how else "") + "）")
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="下发妆容到试妆 App")
    ap.add_argument("--spec", help="makeup_spec.json 路径（--clear 时可省略）")
    ap.add_argument("--intensity", type=float, default=None, help="0.0~1.0 覆盖浓度")
    ap.add_argument("--only", help="只启用指定部位，逗号分隔 region 名")
    ap.add_argument("--assets", help="bake_assets 产物目录，纹理随消息传输")
    ap.add_argument("--bake", action="store_true", help="先自动烘焙再附带资产")
    ap.add_argument("--to", help="只发给指定 App client_id")
    ap.add_argument("--clear", action="store_true", help="卸妆还原")
    args = ap.parse_args()
    if not args.clear and not args.spec:
        ap.error("需要 --spec 或 --clear")
    if args.clear and args.spec:
        ap.error("--clear 与 --spec 不能同时使用")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
