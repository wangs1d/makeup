#!/usr/bin/env python3
"""avatar_session — 3DGS 画像试妆/妆容台会话（Bridge v1.2）。

新流程：用户上传 3DGS 画像 → 选妆在画像上预览 → 用户确认 → 进妆容台辅助化妆。
真脸不再附着妆容渲染（原 apply_spec 真脸链路保留为 legacy）。

用法（按流程顺序）：
    python avatar_session.py register --ply <画像.ply> [--name 我的画像]
    python avatar_session.py preview  --avatar <id> --spec <look.json> [--only lipstick]
    python avatar_session.py preview  --ply <画像.ply> --spec <look.json>   # 未注册直接试
    python avatar_session.py confirm  --avatar <id>
    python avatar_session.py station  --avatar <id> [--spec <look.json>]
    python avatar_session.py leave
    python avatar_session.py status

画像处理：register 时归一化（居中+脸高=1）并抽稀到 --max-gaussians（默认 12 万，
与 App 端 buffer 对齐；tint.bin 按同一 N 编译，App 端不得再抽稀）。
语义标注缓存与编译产物在 --work-dir（默认 ./out/avatars/<id>）。
语义模式：默认 landmarks（正脸渲染+MediaPipe 反投影）；--anchors <json> 用手动锚点
（{"lipstick": [[x,y,z],...], ...}，归一化空间）；已有缓存直接复用。

本地预览：preview 会同时产出 preview.jpg（裸妆|妆后并排）、reference.jpg（妆后单帧，
live_coach --avatar-look 用）与 turntable/，路径打印在输出里。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from avatar_io import load_tint, save_semantics  # noqa: E402
from avatar_render import AvatarRenderer, imwrite, load_add_splats  # noqa: E402
from avatar_semantics import annotate  # noqa: E402
from bridge_common import BridgeClient, BridgeError, fail  # noqa: E402
from makeup_compiler import compile_look  # noqa: E402


def load_spec(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        fail(f"spec 不存在：{p}")
    spec = json.loads(p.read_text(encoding="utf-8"))
    if str(spec.get("spec_version", "")).split(".")[0] != "1":
        fail(f"不支持的 spec_version：{spec.get('spec_version')!r}")
    return spec


def avatar_workdir(args) -> Path:
    base = Path(args.work_dir)
    if getattr(args, "avatar", None):
        return base / args.avatar
    if getattr(args, "ply", None):
        raw = Path(args.ply).read_bytes()
        return base / f"{Path(args.ply).stem}-{hashlib.sha256(raw).hexdigest()[:8]}"
    fail("需要 --avatar <id> 或 --ply <画像.ply>")


def ensure_avatar(work: Path, args) -> tuple[str, "object", dict]:
    """加载（或处理缓存里的）画像；返回 (avatar_id, AvatarData, meta)。"""
    processed = work / "avatar.ply"
    meta_path = work / "meta.json"
    if processed.exists() and meta_path.exists():
        from avatar_io import load_avatar
        return work.name, load_avatar(processed), json.loads(meta_path.read_text(encoding="utf-8"))
    if not getattr(args, "ply", None):
        fail(f"工作目录缺少已处理画像：{processed}（先用 register --ply 上传）")
    from avatar_io import load_avatar, save_avatar
    raw = Path(args.ply).read_bytes()
    av = load_avatar(args.ply, max_count=getattr(args, "max_gaussians", 120000))
    work.mkdir(parents=True, exist_ok=True)
    save_avatar(av, processed)
    meta = {
        "name": getattr(args, "name", None) or Path(args.ply).stem,
        "count": av.n,
        "raw_face_height_m": round(av.face_height, 4),
        "source_sha256_12": hashlib.sha256(raw).hexdigest()[:12],
        "note": "normalized: centered, face height = 1 unit",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[avatar] 画像已处理：{av.n} 高斯（脸高归一 1.0，原 {av.face_height:.3f}m）→ {processed}")
    return work.name, av, meta


async def send(conn: BridgeClient, payload: dict, app: str | None, timeout: float = 30.0) -> dict:
    st = await conn.status()
    if st.get("apps", 0) == 0:
        fail("当前没有试妆 App 连接（references/app-setup.md）")
    if app and app not in (st.get("app_ids") or []):
        fail(f"--to 指定的 App 未连接：{app}（在线：{st.get('app_ids')}）")
    return await conn.request(payload, timeout=timeout, to=app)


# ---------------- 语义 + 编译 + 本地预览（同步，几秒级） ----------------

def prepare_look(work: Path, args) -> tuple[Path, "object", list[dict], dict]:
    avatar_id, av, meta = ensure_avatar(work, args)
    spec = load_spec(args.spec)
    only = {r.strip() for r in args.only.split(",") if r.strip()} if args.only else None

    sem = work / "semantics.bin"
    if args.anchors:
        manual = json.loads(Path(args.anchors).read_text(encoding="utf-8"))
        ids, conf = annotate(av, mode="manual", manual_anchors=manual)
    elif sem.exists():
        ids, conf = annotate(av, mode="cache", cache_path=sem)
    else:
        from avatar_render import front_frame_fn
        try:
            ids, conf = annotate(av, mode="landmarks", render_fn=front_frame_fn(av))
        except Exception as e:  # noqa: BLE001
            fail(f"画像语义标注失败：{e}\n  可改用 --anchors <json> 手动给锚点"
                 f"（模板见 references/avatar-setup.md）")
    if not sem.exists():
        save_semantics(sem, ids, conf)

    compiled = compile_look(av, ids, spec, work / "compiled" / str(spec.get("name", "look")),
                            only=only, avatar_fingerprint=meta.get("source_sha256_12", ""))
    tint = load_tint(compiled / "tint.bin")
    splats = load_add_splats(compiled / "add_splats.json")

    import numpy as np
    r = AvatarRenderer(av, env=args.env)
    inten = float(spec.get("intensity", 0.8))
    r.render_preview_pair(tint, splats, compiled / "preview.jpg",
                          size=args.preview_size, intensity=inten)
    made = r.render(yaw_deg=0, size=args.preview_size, tint=tint,
                    add_splats=splats, intensity=inten)
    imwrite(compiled / "reference.jpg", made[..., ::-1], 90)
    r.render_turntable(compiled / "turntable", tint=tint, add_splats=splats,
                       size=args.preview_size, intensity=inten)
    info = {"spec": spec, "avatar_id": avatar_id, "meta": meta,
            "preview_info": {"preview": str(compiled / "preview.jpg"),
                             "reference": str(compiled / "reference.jpg"),
                             "tinted_gaussians": int((np.asarray(tint)[:, 3] > 1e-4).sum()),
                             "add_splats": len(splats)}}
    print(f"[avatar] 编译完成：tint 覆盖 {info['preview_info']['tinted_gaussians']} 高斯，"
          f"附加溅射 {len(splats)} 个 → {compiled}\n"
          f"  预览图：{compiled / 'preview.jpg'}（请给用户确认）")
    return compiled, tint, splats, info


# ---------------- 子命令 ----------------

async def cmd_register(conn: BridgeClient, args) -> None:
    work = avatar_workdir(args)
    avatar_id, av, meta = ensure_avatar(work, args)
    ply_bytes = (work / "avatar.ply").read_bytes()
    if not conn.asset_base_url:
        fail("bridge 未开启资产侧车（画像体积大，必须走 HTTP 侧车）；"
             "去掉 --asset-port 0 重启 bridge_server")
    assets = {"avatar.ply": await conn.upload_asset(f"{avatar_id}/avatar.ply", ply_bytes)}
    # PBR 材质 + 主光 sidecar（写实管线产物与画像同目录时随注册下发；
    # 缺席则 App 端保持中性材质/默认光）
    for name in ("material.bin", "light.bin"):
        p = work / name
        if p.exists():
            assets[name] = await conn.upload_asset(f"{avatar_id}/{name}", p.read_bytes())
    payload = {
        "type": "avatar_register", "avatar_id": avatar_id, "meta": meta,
        "assets_url": assets,
    }
    await send(conn, payload, args.to, timeout=60.0)
    sidecars = [n for n in ("material.bin", "light.bin") if n in assets]
    print(f"[avatar] 已注册「{meta['name']}」（{avatar_id}，{meta['count']} 高斯"
          + (f"，sidecar: {','.join(sidecars)}" if sidecars else "")
          + f"）→ App 加载中 ✓")


async def cmd_preview(conn: BridgeClient, args) -> None:
    work = avatar_workdir(args)
    compiled, _tint, splats, info = await asyncio.get_event_loop().run_in_executor(
        None, prepare_look, work, args)
    payload = {
        "type": "avatar_preview", "avatar_id": info["avatar_id"],
        "look": {"name": info["spec"].get("name", "look"),
                 "intensity": info["spec"].get("intensity", 0.8),
                 "regions": sorted({l.get("region") for l in info["spec"]["layers"]}),
                 "preview_info": info["preview_info"]},
    }
    if conn.asset_base_url:
        payload["assets_url"] = {
            "tint.bin": await conn.upload_asset(f"{info['avatar_id']}/{compiled.name}/tint.bin",
                                                (compiled / "tint.bin").read_bytes()),
            "add_splats.json": await conn.upload_asset(
                f"{info['avatar_id']}/{compiled.name}/add_splats.json",
                (compiled / "add_splats.json").read_bytes()),
        }
    else:
        payload["makeup"] = {
            "tint_bin_b64": base64.b64encode((compiled / "tint.bin").read_bytes()).decode(),
            "add_splats": splats,
        }
    await send(conn, payload, args.to, timeout=60.0)
    print(f"[avatar] 妆容「{info['spec'].get('name')}」已下发画像预览 ✓"
          f"（确认后 confirm，再 station 进妆容台）")


async def cmd_confirm(conn: BridgeClient, args) -> None:
    avatar_id, _av, _meta = ensure_avatar(avatar_workdir(args), args)
    await send(conn, {"type": "avatar_confirm", "avatar_id": avatar_id}, args.to)
    print(f"[avatar] 用户已确认妆效（{avatar_id}）✓ 可执行 station 进入妆容台")


async def cmd_station(conn: BridgeClient, args) -> None:
    avatar_id, _av, _meta = ensure_avatar(avatar_workdir(args), args)
    payload: dict = {"type": "enter_station", "avatar_id": avatar_id}
    if args.spec:
        spec = load_spec(args.spec)
        payload["look"] = {"name": spec.get("name", "look"),
                           "intensity": spec.get("intensity", 0.8)}
    await send(conn, payload, args.to)
    print("[avatar] 已进入妆容台 ✓ 妆容在侧栏作参照，真脸不再渲染妆容；配合 live_coach 开始指导")


async def cmd_leave(conn: BridgeClient, args) -> None:
    await send(conn, {"type": "leave_station"}, args.to)
    print("[avatar] 已退出妆容台 ✓")


async def cmd_status(conn: BridgeClient, args) -> None:
    print(json.dumps(await conn.status(), ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="3DGS 画像试妆/妆容台会话")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, spec=False):
        p.add_argument("--avatar", help="画像 id（register 产出的工作目录名）")
        p.add_argument("--ply", help="原始 3DGS 画像 ply（首次处理时需要）")
        p.add_argument("--work-dir", default="out/avatars", help="画像缓存根目录（默认 out/avatars）")
        p.add_argument("--to", help="只发给指定 App client_id")
        if spec:
            p.add_argument("--spec", help="妆容 makeup_spec.json")
            p.add_argument("--only", help="只编译指定部位（逗号分隔 region）")
        p.add_argument("--anchors", help="手动语义锚点 JSON（可省，默认 landmarks 自动）")
        p.add_argument("--env", default="neutral", help="预览渲染环境光（neutral/warm/cool/dim）")

    p = sub.add_parser("register", help="上传/注册画像")
    common(p)
    p.add_argument("--name", help="画像显示名")
    p.add_argument("--max-gaussians", type=int, default=120000,
                   help="抽稀上限（默认 12 万，与 App 端 buffer 对齐）")

    p = sub.add_parser("preview", help="选妆在画像上预览")
    common(p, spec=True)
    p.add_argument("--preview-size", type=int, default=512)

    p = sub.add_parser("confirm", help="用户确认妆效")
    common(p)

    p = sub.add_parser("station", help="进入妆容台")
    common(p, spec=True)

    p = sub.add_parser("leave", help="退出妆容台")
    p.add_argument("--to", help="只发给指定 App client_id")

    p = sub.add_parser("status", help="查询 bridge 状态")

    args = ap.parse_args()

    async def run():
        try:
            conn = await BridgeClient.connect(role="agent", client_id="avatar-session")
        except BridgeError as e:
            # preview 允许纯离线（只出本地编译+预览图，不下发 App）
            if args.cmd == "preview":
                print(f"[avatar] bridge 不可用（{e}），仅离线编译与本地预览：")
                work = avatar_workdir(args)
                _compiled, _tint, _splats, info = prepare_look(work, args)
                print(f"[avatar] 离线预览完成 ✓（{info['preview_info']['preview']}；"
                      f"连上 bridge + App 后重跑即可下发画像预览）")
                return
            fail(str(e))
        try:
            await {"register": cmd_register, "preview": cmd_preview, "confirm": cmd_confirm,
                   "station": cmd_station, "leave": cmd_leave, "status": cmd_status}[args.cmd](conn, args)
        finally:
            await conn.close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
