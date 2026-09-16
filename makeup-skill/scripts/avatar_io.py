#!/usr/bin/env python3
"""avatar_io — 标准 3DGS 画像 PLY 的解析/写出/归一化。

支持 Gaussian Splatting 生态的标准导出格式（LAM / GaussianAvatars / Postshot 等）：
    二进制 little_endian，vertex 元素含
        x y z                     位置
        f_dc_0..2                 SH 直流分量 → 基色（高阶 SH 忽略，试妆为漫反射色即可）
        opacity                   透射 logit → sigmoid
        scale_0..2                对数尺度 → exp
        rot_0..3                  四元数 (w x y z)
    可选 nx ny nz（法线，部分导出器带，忽略）；f_rest_* 高阶球谐忽略。

归一化：中心平移到原点、按"脸高"估计缩放到 1 世界单位（与 preview_render 的
FACE_HEIGHT_M 约定一致，溅射 σ/offset 直接沿用米制参数）。脸高取 y 方向 1%~99%
分位跨度（对未裁干净背景的画像比 min/max 稳健）。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814

# 标准 3DGS 属性表（顺序即导出器约定；解析按名字查偏移，不依赖顺序）
_BASE_PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
               "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


@dataclass
class AvatarData:
    """解析后的画像。所有数组行对齐；位置/尺度为归一化后世界单位。"""
    means: np.ndarray            # (N,3) float32
    scales: np.ndarray           # (N,3) float32，半轴 σ
    quats: np.ndarray            # (N,4) float32，(w,x,y,z) 单位化
    colors: np.ndarray           # (N,3) float32 sRGB 0..1
    opacities: np.ndarray        # (N,)  float32 0..1
    face_height: float = 1.0     # 归一化前原始脸高（用于把米制参数换算回原始尺度）
    raw_min: np.ndarray | None = None
    raw_max: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    def decimated(self, max_count: int) -> "AvatarData":
        """随机抽稀（保数量）。等距抽稀会保留斐波那契/扫描点的相干结构，渲染出摩尔纹；
        随机子采样打散相干性，σ 重叠不变。种子固定保证 register/compile 两端一致。"""
        if self.n <= max_count:
            return self
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(self.n, max_count, replace=False))
        return AvatarData(self.means[idx], self.scales[idx], self.quats[idx],
                          self.colors[idx], self.opacities[idx],
                          self.face_height, self.raw_min, self.raw_max, self.meta)


# ---------------- PLY 解析 ----------------

def _parse_header(raw: bytes) -> tuple[dict, int]:
    """返回 (元素描述, 数据起始偏移)。元素描述: {name: {count, props: [(名, 类型字节)]}}"""
    if not raw.startswith(b"ply"):
        raise ValueError("不是 PLY 文件（缺 magic）")
    fmt = None
    elements: list[tuple[str, int, list[tuple[str, str]]]] = []
    offset = raw.index(b"end_header\n") + len(b"end_header\n")
    for line in raw[:offset].decode("ascii", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property" and elements:
            elements[-1][2].append((parts[-1], parts[1]))
    if fmt != "binary_little_endian":
        raise ValueError(f"仅支持 binary_little_endian PLY，得到 {fmt!r}（ascii 请先转换）")
    return elements, offset


def load_avatar(path: str | Path, max_count: int | None = None) -> AvatarData:
    """解析 3DGS PLY → AvatarData（归一化：居中 + 脸高=1）。"""
    raw = Path(path).read_bytes()
    elements, offset = _parse_header(raw)
    vert = next((e for e in elements if e[0] == "vertex"), None)
    if vert is None:
        raise ValueError("PLY 缺少 vertex 元素")
    _, count, props = vert
    missing = [p for p in _BASE_PROPS if p not in {n for n, _ in props}]
    if missing:
        raise ValueError(f"PLY 缺少 3DGS 必需属性：{missing}（确认是 3DGS 导出而非普通网格）")

    sizes = {"float": 4, "double": 8, "uchar": 1, "char": 1, "int": 4, "uint": 4,
             "short": 2, "ushort": 2, "float16": 2}
    dtypes = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1", "int": "i4",
              "uint": "u4", "short": "i2", "ushort": "u2", "float16": "f2"}
    # 结构化 dtype 一次映射整行（字段偏移 = 属性顺序，itemsize = 行跨度）
    row_dtype = np.dtype([(name, dtypes[typ]) for name, typ in props])
    assert row_dtype.itemsize == sum(sizes[typ] for _, typ in props)

    need = offset + count * row_dtype.itemsize
    if len(raw) < need:
        raise ValueError(f"PLY 数据区不完整：需要 {need} 字节，只有 {len(raw)}")
    rows = np.frombuffer(buf := raw[offset:need], dtype=row_dtype, count=count)

    def col(name: str) -> np.ndarray:
        return rows[name].astype(np.float32)

    means = np.stack([col("x"), col("y"), col("z")], axis=1)
    raw_dc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1)
    colors = np.clip(0.5 + SH_C0 * raw_dc, 0.0, 1.0) ** (1.0 / 2.2)   # 线性→sRGB 近似
    opacities = 1.0 / (1.0 + np.exp(-col("opacity")))
    scales = np.exp(np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=1))
    quats = np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], axis=1)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-9

    av = normalize(AvatarData(means, scales, quats, colors, opacities))
    if max_count is not None:
        av = av.decimated(max_count)
    return av


# ---------------- 归一化 ----------------

def normalize(av: AvatarData) -> AvatarData:
    """居中（仅 x/y）+ 脸高=1。y 跨度用 1%~99% 分位（抗背景飞点）。

    z 不做平移：前置约定"脸朝 +Z 凸出、椭圆中心为原点"，锚点/溅射的 z 语义
    （正面在 +z 侧）依赖这一点；归一化幂等（再跑一次 span=1、center=0）。
    """
    lo = np.percentile(av.means, 1, axis=0)
    hi = np.percentile(av.means, 99, axis=0)
    height = max(float(hi[1] - lo[1]), 1e-6)
    cx = (lo[0] + hi[0]) / 2
    cy = (lo[1] + hi[1]) / 2
    means = av.means - np.array([cx, cy, 0.0], np.float32)
    means = means / height
    return AvatarData(means.astype(np.float32), av.scales / height, av.quats, av.colors,
                      av.opacities, height, lo, hi, av.meta)


# ---------------- 写出（测试/调试/编译产物自描述） ----------------

def save_avatar(av: AvatarData, path: str | Path, raw_scale: bool = False) -> None:
    """写出标准 3DGS PLY。raw_scale=False 时写归一化坐标（默认）。"""
    means, scales = av.means, av.scales
    if raw_scale:
        h = max(av.face_height, 1e-6)
        means = means * h + (av.raw_min + av.raw_max) / 2
        scales = scales * h
    dc = ((av.colors ** 2.2 - 0.5) / SH_C0).astype(np.float32)
    op = np.log(np.clip(av.opacities, 1e-6, 1 - 1e-6) / np.clip(1 - av.opacities, 1e-6, 1)).astype(np.float32)
    sc = np.log(np.clip(scales, 1e-9, None)).astype(np.float32)

    props: list[tuple[str, str]] = [(n, "float") for n in
                                    ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                                     "opacity", "scale_0", "scale_1", "scale_2",
                                     "rot_0", "rot_1", "rot_2", "rot_3"]]
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {av.n}",
              *[f"property float {n}" for n, _ in props], "end_header", ""]
    cols = [means[:, 0], means[:, 1], means[:, 2], dc[:, 0], dc[:, 1], dc[:, 2],
            op, sc[:, 0], sc[:, 1], sc[:, 2],
            av.quats[:, 0], av.quats[:, 1], av.quats[:, 2], av.quats[:, 3]]
    body = np.empty(av.n, dtype=np.dtype([(n, "<f4") for n, _ in props]))
    for (n, _), c in zip(props, cols):
        body[n] = c.astype("<f4")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("\n".join(header).encode("ascii") + body.tobytes())


# ---------------- 二进制 sidecar（Python 写、C# 读） ----------------

def save_semantics(path: str | Path, region_ids: np.ndarray, confidences: np.ndarray | None = None) -> None:
    """MKSEM1：magic(4)+ver(u16)+flags(u16)+N(u32)+rsv(u32)+regionId(u8×N)[+conf(u8×N，flags=1)]"""
    n = region_ids.shape[0]
    flags = 1 if confidences is not None else 0
    head = struct.pack("<4sHHII", b"MKSM", 1, flags, n, 0)[:16]
    body = region_ids.astype(np.uint8).tobytes()
    if flags:
        body += (np.clip(confidences, 0, 1) * 255).astype(np.uint8).tobytes()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(head + body)


def load_semantics(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    raw = Path(path).read_bytes()
    magic, ver, flags, n, _rsv = struct.unpack_from("<4sHHII", raw, 0)
    if magic != b"MKSM" or ver != 1:
        raise ValueError(f"semantics.bin 格式不符：{magic!r} v{ver}")
    ids = np.frombuffer(raw, np.uint8, n, 16)
    conf = None
    if flags & 1 and len(raw) >= 16 + 2 * n:
        conf = np.frombuffer(raw, np.uint8, n, 16 + n).astype(np.float32) / 255.0
    return ids, conf


def save_tint(path: str | Path, tint: np.ndarray) -> None:
    """MKMKP1：magic(4)+ver(u16)+flags(u16)+N(u32)+rsv(u32) + rgba f32×4×N"""
    n = tint.shape[0]
    head = struct.pack("<4sHHII", b"MKMK", 1, 0, n, 0)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(head + tint.astype("<f4").tobytes())


def load_tint(path: str | Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    magic, ver, _flags, n, _rsv = struct.unpack_from("<4sHHII", raw, 0)
    if magic != b"MKMK" or ver != 1:
        raise ValueError(f"tint.bin 格式不符：{magic!r} v{ver}")
    return np.frombuffer(raw, "<f4", n * 4, 16).reshape(n, 4).astype(np.float32)
