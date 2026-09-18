"""fix_quat_order — 修正 train_base 旧导出 PLY 的四元数循环错位（一次性工具）。

历史 bug：train_base 把 gsplat 的 wxyz 四元数当内部 xyzw 塞进 cloud["rot"]，
export_ply 再按 "xyzw→wxyz" 循环移位一次 → PLY 实际是 (z,w,x,y)。
正确 PLY (wxyz) = 旧 PLY 左移一位：new[i] = old[(i+1)%4]。
train_base 已修复（导出前显式 wxyz→xyzw），本脚本只修旧产物：

    python tools/fix_quat_order.py out/photoreal/d6/base.ply out/photoreal/d6/madeup.ply
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def fix_ply(path: str | Path) -> None:
    p = Path(path)
    raw = p.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii")
    lines = header.splitlines()
    n = int(next(l.split()[2] for l in lines if l.startswith("element vertex")))
    props = [l.split()[-1] for l in lines if l.startswith("property float")]
    assert props[-4:] == ["rot_0", "rot_1", "rot_2", "rot_3"], f"{p.name}: 属性不符"
    dt = np.dtype([(nm, "<f4") for nm in props])
    rows = np.frombuffer(raw[end:end + n * dt.itemsize], dtype=dt, count=n).copy()
    rot = np.stack([rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"]], 1)
    # 备份原四元数列到注释外不落盘；直接左移一位覆盖
    fixed = np.roll(rot, -1, axis=1)               # new[i] = old[(i+1)%4]
    rows["rot_0"], rows["rot_1"] = fixed[:, 0], fixed[:, 1]
    rows["rot_2"], rows["rot_3"] = fixed[:, 2], fixed[:, 3]
    backup = p.with_suffix(p.suffix + ".quatbak")
    if not backup.exists():
        backup.write_bytes(raw)
    with open(p, "wb") as f:
        f.write(raw[:end])
        f.write(rows.tobytes())
    print(f"[fix] {p} 四元数已左移修正（备份 → {backup.name}）")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        fix_ply(arg)
