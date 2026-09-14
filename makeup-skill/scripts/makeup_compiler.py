#!/usr/bin/env python3
"""makeup_compiler — 把妆容 spec 编译到画像 Gaussian 上（spec + region-ID → 妆容 buffer）。

两类层两种编译（与 preview_render/RegionMaskBaker 的语义一致，落点从 UV 纹理
换成每-Gaussian 属性）：

1. mesh 层（粉底/腮红/眼影…）→ **tint.bin**：region 命中的 Gaussian 做 alpha-over
   混色。每 Gaussian 产 rgba：rgb = 目标色（ramp 按向心度取色：距区域锚点越近越取
   中心色），a = 覆盖权重（浓淡 intensity 在运行时乘，spec 变浓度不用重编译）。
2. splat 层（唇釉/水光）→ **add_splats.json**：沿 region 高斯的表面（局部 PCA 法线）
   撒附加高斯：位置=点+法线×厚度、长轴沿锚点走向、σ/密度/透明度沿用米制参数表。

产物（compiled/<avatar_id>/<look>/）：
    tint.bin (MKMKP1)   N×rgba f32 —— a=0 的行表示该 Gaussian 无妆容
    add_splats.json     附加高斯列表（画像局部空间，脸高=1）
    manifest.json       元信息（look 名、intensity、计数、画像指纹）

`--only` 等价物：layers 过滤后重编译，tint 全量重写（旧 region 置零）——
单层改妆（换色号）与 apply_spec 的工作流语义对齐。
"""
from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

from avatar_io import save_tint
from avatar_semantics import REGION_IDS, REGION_PROFILES, SurfaceProbe

HEX2RGB = lambda h: np.array([int(h.lstrip("#")[0:2], 16),
                              int(h.lstrip("#")[2:4], 16),
                              int(h.lstrip("#")[4:6], 16)], np.float32) / 255.0


def ramp_color(stops: list[dict], t: np.ndarray | float) -> np.ndarray:
    """color_stops 渐变采样（与 preview_render.sample_ramp 同语义）。t: (...) → (...,3)"""
    pts = sorted(((float(s["at"]), HEX2RGB(s["hex"])) for s in stops), key=lambda p: p[0])
    t = np.clip(np.asarray(t, np.float32), pts[0][0], pts[-1][0])
    out = np.zeros(t.shape + (3,), np.float32)
    out[...] = pts[-1][1]
    for (a0, c0), (a1, c1) in zip(pts, pts[1:]):
        if a1 <= a0:
            continue
        seg = (t >= a0) & (t <= a1)
        f = ((t - a0) / (a1 - a0))[seg][:, None] if seg.any() else None
        if f is not None:
            out[seg] = c0[None, :] * (1 - f) + c1[None, :] * f
    out[t <= pts[0][0]] = pts[0][1]
    return out


def _coverage(dist: np.ndarray, radius: float, falloff: float) -> np.ndarray:
    """中心 1 → 半径 0 的软覆盖（平滑阶梯 + falloff 控制过渡带宽）。"""
    inner = radius * (0.55 - 0.25 * falloff)
    return np.clip((radius - dist) / max(radius - inner, 1e-6), 0.0, 1.0) ** 1.5


def compile_tint(av, ids: np.ndarray, layers: list[dict]) -> np.ndarray:
    """mesh 层 → (N,4) rgba。多层按 spec 顺序 alpha-over；先到者（底层）保留。"""
    n = av.n
    tint = np.zeros((n, 4), np.float32)
    probes: dict[str, np.ndarray] = {}

    for layer in layers:
        if not layer.get("enabled", True):
            continue
        # splat 型层同样参与底色 tint（原管线即"网格底色 + 溅射体积"双渲染），
        # 其附加高斯由 compile_add_splats 另行产出
        region = layer["region"]
        if region not in REGION_PROFILES or region not in REGION_IDS:
            continue
        if ids.max() == 0:
            continue
        sel = np.nonzero(ids == REGION_IDS[region])[0]
        if len(sel) == 0:
            continue
        if region not in probes:                        # region 参考点云（抽稀到 ≤256 控制距离计算量）
            refs = av.means[sel]
            probes[region] = refs if len(refs) <= 256 else refs[::max(1, len(refs) // 256)]
        pts = probes[region]
        sel_pts = av.means[sel]
        dist = np.empty(len(sel), np.float32)
        for i0 in range(0, len(sel), 4096):             # 分块算最近参考点距（大 region 防 OOM）
            chunk = sel_pts[i0:i0 + 4096]
            dist[i0:i0 + 4096] = np.sqrt(
                ((chunk[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(1)
        prof = REGION_PROFILES[region]
        radius = float(prof["radius"]) * (0.85 + 0.3 * float((layer.get("shape") or {}).get("spread", 0.5) or 0.5))
        falloff = float((layer.get("shape") or {}).get("falloff", 0.65) or 0.65)
        cov = _coverage(dist, radius, falloff)
        side = layer.get("side", "both")
        if side in ("left", "right"):                   # 单侧层：只保留对应 x 半边
            sign = 1.0 if side == "left" else -1.0
            cov = cov * (np.sign(av.means[sel, 0]) == sign)
        cent = 1.0 - dist / max(radius, 1e-6)           # 向心度 → ramp 取色
        col = ramp_color(layer["color_stops"], cent)
        grain = float(layer.get("texture_strength", 0.3) or 0)
        if grain > 0.01:
            seed = int(zlib.crc32(layer["id"].encode()) & 0xFFFF)
            g = np.random.default_rng(seed).normal(1.0, 0.12 * grain, len(sel))
            col = np.clip(col * g[:, None], 0, 1)
        a = np.clip(cov * float(layer.get("opacity", 0.7)), 0, 1)

        out_a = tint[sel, 3]
        new_a = np.clip(a + out_a * (1 - a), 0, 1)
        w_new = np.where(new_a > 1e-6, a / np.maximum(new_a, 1e-6), 0.0)
        w_old = np.where(new_a > 1e-6, out_a * (1 - a) / np.maximum(new_a, 1e-6), 0.0)
        tint[sel, :3] = col * w_new[:, None] + tint[sel, :3] * w_old[:, None]
        tint[sel, 3] = new_a
    return np.clip(tint, 0, 1)


def compile_add_splats(av, ids: np.ndarray, layers: list[dict],
                       probe: SurfaceProbe | None = None) -> list[dict]:
    """splat 层 → 附加高斯列表（画像局部空间）。参数语义与 preview_render.build_splats 对齐：
    thickness（米）→ σ 与法向抬升；density → 锚点撒点密度；渐变沿锚点序取色。"""
    probe = probe or SurfaceProbe(av.means)
    out: list[dict] = []
    for layer in layers:
        if not layer.get("enabled", True) or (layer.get("render", {}) or {}).get("type") != "splat":
            continue
        region = layer["region"]
        sel = np.nonzero(ids == REGION_IDS.get(region, 0))[0]
        if len(sel) == 0:
            continue
        splat = layer.get("render", {}).get("splat") or {}
        th = float(splat.get("thickness", 0.0012))
        dens = float(splat.get("density", 0.7))
        stride = max(1, int(round(1.0 / max(dens, 0.05))))
        stride = max(stride, int(np.ceil(len(sel) / 1200)))   # 单层附加高斯 ≤1200（App 实例化批预算）
        pts_all = av.means[sel]
        scale_all = av.scales[sel]
        pts = pts_all[::stride]
        stops = layer["color_stops"]
        span = max(len(pts) - 1, 1)
        k = 0
        for p, sc in zip(pts, scale_all[::stride]):
            _c, nrm = probe.plane_at(p, radius=float(sc.max() * 6) + 0.02)
            # 切向长轴：取与法线垂直、水平分量最大的方向（唇纹/闪片走向的稳定近似）
            t = np.cross(nrm, np.array([0.0, 1.0, 0.0]))
            if np.linalg.norm(t) < 1e-6:
                t = np.cross(nrm, np.array([1.0, 0.0, 0.0]))
            t /= np.linalg.norm(t) + 1e-9
            frac = k / span
            col = ramp_color(stops, np.array([1.0 - frac], np.float32))[0]
            out.append({
                "pos": [round(float(v), 5) for v in (p + nrm * th)],
                "normal": [round(float(v), 5) for v in nrm],
                "axis": [round(float(v), 5) for v in t],
                "sigma": [round(th * 1.6, 6), round(th * 1.2, 6)],
                "color": "#{:02X}{:02X}{:02X}".format(*(int(round(c * 255)) for c in col)),
                "alpha": round(min(float(layer.get("opacity", 0.7)) * dens, 1.0), 3),
            })
            k += 1
    return out


def compile_look(av, ids: np.ndarray, spec: dict, out_dir: str | Path,
                 only: set[str] | None = None,
                 avatar_fingerprint: str = "") -> Path:
    """完整编译：spec（可 only 过滤）→ tint.bin + add_splats.json + manifest.json。"""
    layers = [dict(l) for l in spec.get("layers", []) if l.get("enabled", True)]
    if only:
        layers = [l for l in layers if l.get("region") in only]

    tint = compile_tint(av, ids, layers)
    if only:                                            # 未选中 region 的旧妆清零（单品试妆语义）
        keep_ids = {REGION_IDS[r] for r in only if r in REGION_IDS}
        drop = ~np.isin(ids, list(keep_ids))
        tint[drop, 3] = 0.0
    add = compile_add_splats(av, ids, layers)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tint(out_dir / "tint.bin", tint)
    (out_dir / "add_splats.json").write_text(
        json.dumps({"splats": add, "space": "avatar-local",
                    "face_height": 1.0}, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "look": spec.get("name", "look"),
        "intensity": spec.get("intensity", 0.8),
        "regions": sorted({l["region"] for l in layers}),
        "tinted_gaussians": int((tint[:, 3] > 1e-4).sum()),
        "add_splats": len(add),
        "avatar_fingerprint": avatar_fingerprint,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return out_dir
