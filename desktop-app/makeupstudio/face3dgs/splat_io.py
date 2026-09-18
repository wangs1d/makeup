"""splat_io — 3DGS .ply 的读取/写出（与 SplatCloudBuilder 导出格式对齐）。

读取容忍 SH 高阶系数（f_rest_*）与额外属性，只取 splat 渲染所需字段；
内部统一为 {xyz, scale(线性), rot(xyzw), rgba(线性 0..1)}。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SH_C0 = 0.28209479112561376


def read_ply(path: str | Path) -> dict[str, np.ndarray]:
    """读取 3DGS 标准二进制 ply → {xyz, scale, rot(xyzw), rgba}。"""
    with open(path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} 不是有效的 ply")
            header += line
        lines = header.decode("ascii", "replace").splitlines()
        n = int(next(l.split()[2] for l in lines if l.startswith("element vertex")))
        props: list[tuple[str, str]] = []
        for line in lines:
            p = line.split()
            if p[:2] == ["property", "float"]:
                props.append((p[2], "f4"))
            elif p[:2] == ["property", "uchar"]:
                props.append((p[2], "u1"))
        dtype = np.dtype(props)
        data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)

    def col(name: str, default: np.ndarray | None = None) -> np.ndarray:
        if name in dtype.names:
            return data[name].astype(np.float32)
        if default is None:
            raise KeyError(f"ply 缺少必需属性 {name}")
        return np.broadcast_to(default, (n,)).astype(np.float32)

    xyz = np.stack([col("x"), col("y"), col("z")], axis=1)
    scale = np.exp(np.stack([col(f"scale_{k}") for k in range(3)], axis=1))
    rot = np.stack([col(f"rot_{k}") for k in range(1, 4)] + [col("rot_0")], axis=1)  # wxyz→xyzw
    rgb = 0.5 + SH_C0 * np.stack([col(f"f_dc_{k}") for k in range(3)], axis=1)
    alpha = 1.0 / (1.0 + np.exp(-col("opacity")))
    rgba = np.concatenate([np.clip(rgb, 0, 1), alpha[:, None]], axis=1)
    out = {"xyz": xyz.astype(np.float32), "scale": scale.astype(np.float32),
           "rot": rot.astype(np.float32), "rgba": rgba.astype(np.float32)}
    # SH 高阶（f_rest_*）存在时按 3DGS 通道主序读回 (n, pc, 3)，随资产贯穿
    # 妆容烘焙/再导出（逐 splat 妆容只改 DC，高阶保持底模视角相关外观）。
    rest_names = sorted((nm for nm in dtype.names if nm.startswith("f_rest_")),
                        key=lambda nm: int(nm.split("_")[-1]))
    if rest_names:
        n_rest = len(rest_names)
        if n_rest % 3 == 0:
            fr = np.stack([data[nm].astype(np.float32) for nm in rest_names], axis=1)
            pc = n_rest // 3
            out["sh_rest"] = np.stack([fr[:, c * pc:(c + 1) * pc] for c in range(3)],
                                      axis=2).astype(np.float32)
    return out


def write_ply(cloud: dict[str, np.ndarray], path: str | Path) -> None:
    """内部 cloud → 3DGS 标准 ply（复用 desktop-app 导出实现，避免格式分叉）。"""
    from ..splat3d import SplatCloudBuilder
    SplatCloudBuilder.export_ply(cloud, path)


def export_splat(cloud: dict[str, np.ndarray], path: str | Path) -> None:
    from ..splat3d import SplatCloudBuilder
    SplatCloudBuilder.export_splat(cloud, path)
