"""fix_opacity_encoding — 修正旧导出 PLY 的 opacity 编码（一次性工具）。

历史 bug：export_ply 写 log(a/√(1-a))，读取端按 sigmoid(logit) 解码 →
高不透明度 splat 系统性变透明（a=0.95 → 0.81），脸面渗底色。
export_ply 已修复为标准 logit。本脚本把旧 PLY 的 opacity 列原位换算：
    e = exp(file_value);  a_true = e(√(e²+4) − e)/2;  file_value' = logit(a_true)

    python tools/fix_opacity_encoding.py out/photoreal/d6/base.ply ...
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
    names = [l.split()[-1] for l in lines if l.startswith("property float")]
    assert "opacity" in names, f"{p.name}: 无 opacity 属性"
    dt = np.dtype([(nm, "<f4") for nm in names])
    rows = np.frombuffer(raw[end:end + n * dt.itemsize], dtype=dt, count=n).copy()
    fv = rows["opacity"].astype(np.float64)
    e = np.exp(np.clip(fv, -20, 20))
    a_true = e * (np.sqrt(e * e + 4.0) - e) / 2.0        # 反解 log(a/√(1-a)) 的原 opacity
    a_true = np.clip(a_true, 1e-6, 1 - 1e-6)
    rows["opacity"] = np.log(a_true / (1 - a_true)).astype(np.float32)  # 标准 logit
    backup = p.with_suffix(p.suffix + ".opabak")
    if not backup.exists():
        backup.write_bytes(raw)
    with open(p, "wb") as f:
        f.write(raw[:end])
        f.write(rows.tobytes())
    print(f"[fix] {p} opacity 已换算为标准 logit（备份 → {backup.name}）")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        fix_ply(arg)
