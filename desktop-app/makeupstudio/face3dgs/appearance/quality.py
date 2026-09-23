"""quality — 采集/资产质量分级与交付门禁。

背景：低清源（<720p）直接进管线会照单全收，交付质量不可控（SR 过渡路径
已被实测否定——高光/噪声被放大烘进 splat）。本模块把"画质能否支撑商业交付"
变成可判定的分级 + 可执行的门禁，而不是玄学：

    A（可交付）  源帧短边 ≥ 1000px 且底模 PSNR 达标
    B（可用）    源帧短边 ≥ 720px 且底模 PSNR 达标
    C（仅诊断）  其余（低清源 / 训练退化）——不出定妆照，只出诊断图

分级纯 numpy 可算，无 GPU 依赖；deliver/render 端据此拦截（--force 可越过，
拦截结论写进 renders.json 留痕）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# 分级阈值：分辨率短边（px）与底模留出帧 PSNR（dB）
GRADE_MIN_SIDE = {"A": 1000, "B": 720}
GRADE_MIN_PSNR = 20.0


@dataclass
class QualityReport:
    grade: str                     # A / B / C
    min_side: int                  # 源帧短边
    psnr: float                    # 底模留出帧脸区 PSNR（-1 = 未知）
    splats: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"grade": self.grade, "min_side": self.min_side,
                "psnr": self.psnr, "splats": self.splats,
                "reasons": list(self.reasons)}


def assess_quality(camera_wh: tuple[int, int], psnr: float | None,
                   splats: int) -> QualityReport:
    """相机分辨率 + 底模 PSNR + splat 数 → 质量分级（A/B/C）。

    psnr 未知（旧资产缺 train_report）不降级——分辨率是硬前提，PSNR 只是
    训练退化信号；splat 数过少（<3 万）单独记原因（densify 没起飞）。"""
    w, h = int(camera_wh[0]), int(camera_wh[1])
    min_side = min(w, h)
    psnr_v = float(psnr) if psnr is not None else -1.0
    reasons: list[str] = []

    if min_side >= GRADE_MIN_SIDE["A"]:
        res_grade = "A"
    elif min_side >= GRADE_MIN_SIDE["B"]:
        res_grade = "B"
    else:
        res_grade = "C"
        reasons.append(f"源帧短边 {min_side}px < {GRADE_MIN_SIDE['B']}px（低清源，"
                       "SR 过渡不可商用，建议 ≥1080p 重录）")

    psnr_ok = psnr_v < 0 or psnr_v >= GRADE_MIN_PSNR
    if not psnr_ok:
        reasons.append(f"底模 PSNR {psnr_v:.1f}dB < {GRADE_MIN_PSNR}dB（训练退化）")

    if splats < 30_000:
        reasons.append(f"splat 数 {splats} < 3 万（densify 未起飞）")

    grade = res_grade if psnr_ok else "C"
    return QualityReport(grade=grade, min_side=min_side, psnr=psnr_v,
                         splats=int(splats), reasons=reasons)


def load_grade(asset_dir) -> str | None:
    """读资产目录 report.json 里缓存的质量分级（无则 None）。"""
    import json
    from pathlib import Path

    p = Path(asset_dir) / "report.json"
    if not p.exists():
        return None
    try:
        rep = json.loads(p.read_text(encoding="utf-8"))
        q = rep.get("quality") or {}
        g = q.get("grade")
        return g if g in ("A", "B", "C") else None
    except (OSError, ValueError):
        return None
