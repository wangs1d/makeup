"""colorfield — 参考驱动的色彩场提取 + 迁移系数自求解（P2）。

旧路径的两个"我的期待"：
    · 区域色 = 参考图蒙版核心的 Lab 均值 + 45/85 分位明度两档 hex——"唇内→唇外
      渐变 / 腮红落点 / 眼影层次"被压成两个色阶，且色相被单一均值抹平；
    · 迁移系数 kL/chroma 是手工 PHOTOREAL_LAB 表，与参考图无关。

本模块把两者都换成观测：
    region_lab_profile   区域内"边界 → 核心"向心度 t 上的 Lab 剖面（参考图像素
                        按 t 分箱取中位）——色相/色度/明度的空间层次全部保留；
    profile_to_stops     剖面 → 多档 color_stops（喂给现有 spec 烘焙路径）
    solve_region_coeffs  最小二乘自求解 (kL, chroma)：让 _lab_migrate 在真实
                        素颜底色上复现参考色剖面——消解 pigment-safe 阻尼
                        （lab_adapt 最多收到 0.6×）与系数被折叠的欠涂，替代
                        手工表（手工表降级为无参考时的初值）。
纯 numpy/cv2 + makeup_uv 的 Lab 原语，求解器与剖面提取均可离线单测。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .calibrate import ciede2000, image_region_masks
from .makeup_uv import PHOTOREAL_LAB, UvMakeupBaker, lab2rgb, rgb2lab

PROFILE_BINS = 8          # 剖面分箱数（色带档数）
MIN_PROFILE_PX = 60       # 区域有效像素下限
COEFF_BOUNDS = ((0.0, 3.0), (0.5, 3.0))    # (kL, chroma) 搜索域
GRID = 5                  # 粗搜索格点数（每轴）
GN_ITERS = 8              # Levenberg 阻尼高斯-牛顿迭代数
MAX_STOPS = 5             # spec 色带档数上限（多档→样品/序列合成端可读）
LOOP_GAIN = 1.0           # 闭环反向过冲增益
LOOP_STEP_MAX = 20.0      # 单轮 Lab 修正上限（防一次过冲把目标推飞）


# ---------------- 参考图色彩剖面 ----------------

def _detect_px(img_bgr: np.ndarray) -> np.ndarray:
    from ...tracker import FaceTracker
    tracker = FaceTracker(smooth=False)
    try:
        det = tracker.detect(img_bgr, 0.0)
    finally:
        tracker.close()
    if det is None:
        raise RuntimeError("参考图未检测到人脸")
    return det["px"]


def inness_field(mask: np.ndarray) -> np.ndarray:
    """蒙版 → 向心度场 t（0 边界 → 1 最深处），眼部/唇部"内外"结构一致。"""
    hard = (np.asarray(mask, np.float32) >= 0.5).astype(np.uint8)
    if int(hard.sum()) < 16:
        return np.zeros(mask.shape, np.float32)
    d = cv2.distanceTransform(hard, cv2.DIST_L2, 3)
    peak = float(d.max())
    return (d / peak if peak > 1e-6 else d).astype(np.float32)


def profile_from_mask(img_rgb: np.ndarray, mask: np.ndarray,
                      bins: int = PROFILE_BINS
                      ) -> tuple[np.ndarray, np.ndarray] | None:
    """图像空间蒙版 → (t (B,), lab (B,3)) 剖面（已转 0..1 RGB 输入）。"""
    t_all = inness_field(mask)
    sel = np.asarray(mask, np.float32) > 0.35
    if int(sel.sum()) < MIN_PROFILE_PX:
        return None
    ts = t_all[sel]
    lab = rgb2lab(np.asarray(img_rgb, np.float64)[sel])
    edges = np.quantile(ts, np.linspace(0.0, 1.0, bins + 1))
    edges = np.maximum.accumulate(edges)                  # 退化分布时保持单调
    t_out, lab_out = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        take = (ts >= lo) & (ts <= hi)
        if int(take.sum()) < 8:
            continue
        t_out.append(float(np.median(ts[take])))
        lab_out.append(np.median(lab[take], axis=0))
    if len(t_out) < 2:
        return None
    return np.asarray(t_out, np.float64), np.asarray(lab_out, np.float64)


def region_lab_profile(img_bgr: np.ndarray, region: str,
                       px: np.ndarray | None = None, bins: int = PROFILE_BINS
                       ) -> tuple[np.ndarray, np.ndarray] | None:
    """参考图某妆区 → (t (B,), lab (B,3))：向心度上的 Lab 剖面。

    t=0 是区域边界（唇线/腮红外缘），t=1 是区域核心（唇中心/腮红落点）；
    分箱按蒙版内像素的 t 分位切（每箱像素数均衡），每箱取 Lab 中位——中位
    而非均值是为了让"唇内一侧的镜面高光"这类离群像素不污染代表色。"""
    h, w = img_bgr.shape[:2]
    if px is None:
        px = _detect_px(img_bgr)
    m = image_region_masks(px, w, h, regions=(region,)).get(region)
    if m is None:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    return profile_from_mask(rgb, m, bins)


def extract_profiles(img_bgr: np.ndarray, regions, px: np.ndarray | None = None,
                     bins: int = PROFILE_BINS) -> dict[str, tuple]:
    """参考妆照 → 逐区域 Lab 剖面（地标只检一次，供全区域共用）。

    这是"颜色从参考来"的输入端：比 extract_makeup_colors 的两档均值多保留了
    区域内空间层次（唇内→唇外渐变 / 眼影深浅 / 腮红落点），且全部取自参考图
    真实像素的 Lab 中位，不经过任何手工色号表。"""
    regions = tuple(dict.fromkeys(regions))
    if not regions:
        return {}
    h, w = img_bgr.shape[:2]
    if px is None:
        px = _detect_px(img_bgr)
    masks = image_region_masks(px, w, h, regions=regions)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    out: dict[str, tuple] = {}
    for region in regions:
        m = masks.get(region)
        if m is None:
            continue
        prof = profile_from_mask(rgb, m, bins)
        if prof is not None:
            out[region] = prof
    return out


def sample_profile_lab(profile: tuple[np.ndarray, np.ndarray],
                       t: np.ndarray) -> np.ndarray:
    """剖面在任意向心度 t 处的 Lab（分段线性，端点外取端点值）。"""
    ts, lab = profile
    t = np.clip(np.asarray(t, np.float64), ts[0], ts[-1])
    out = np.empty(t.shape + (3,), np.float64)
    for k in range(3):
        out[..., k] = np.interp(t, ts, lab[:, k])
    return out


def profile_to_stops(profile: tuple[np.ndarray, np.ndarray],
                     max_stops: int = 5) -> list[dict]:
    """剖面 → color_stops（at = 向心度 t，与 sample_ramp 的 0=边缘/1=中心一致）。"""
    ts, lab = profile
    if len(ts) > max_stops:
        idx = np.unique(np.linspace(0, len(ts) - 1, max_stops).round().astype(int))
        ts, lab = ts[idx], lab[idx]
    stops = []
    for t, l in zip(ts, lab):
        c = np.clip(lab2rgb(np.asarray([l], np.float64))[0], 0.0, 1.0)
        stops.append({"at": round(float(t), 4),
                      "hex": "#%02X%02X%02X" % tuple(int(round(x * 255)) for x in c)})
    return stops


def spec_from_profiles(template: dict, profiles: dict[str, tuple],
                       max_stops: int = MAX_STOPS) -> dict:
    """模板 spec × 参考图 Lab 剖面 → 标定后的 spec（逐区域多档色带）。

    与 calibrate.calibrate_spec 同一"只有颜色来自观测"的边界，区别是色带不再
    压成两档：唇的内深外浅、眼影层次、腮红落点都保留在 at=向心度上，烘焙端
    sample_ramp 直接按多档插值。shape/finish/层结构仍沿用模板。"""
    import copy

    spec = copy.deepcopy(template)
    hit: dict[str, int] = {}
    for layer in spec.get("layers", []):
        prof = profiles.get(layer.get("region"))
        if prof is None:
            continue
        stops = profile_to_stops(prof, max_stops=max_stops)
        layer["color_stops"] = stops
        hit[layer["region"]] = len(stops)
    if hit:
        spec.setdefault("calibration", {})
        spec["calibration"] = {"regions": list(hit), "stops": hit,
                               "source": "colorfield_profile"}
    return spec


# ---------------- ΔE00 闭环（求解 → 重烘 → 重测，≤2 轮） ----------------

def shift_profile(profile: tuple[np.ndarray, np.ndarray],
                  delta_lab) -> tuple[np.ndarray, np.ndarray]:
    """剖面整体平移 ΔLab（闭环修正目标色用；保持区域内梯度形状不变）。"""
    ts, lab = profile
    out = np.asarray(lab, np.float64) + np.asarray(delta_lab, np.float64)
    out[:, 0] = np.clip(out[:, 0], 0.0, 100.0)
    out[:, 1:] = np.clip(out[:, 1:], -128.0, 127.0)
    return ts, out


def loop_delta(measured: np.ndarray, ref: np.ndarray,
               gain: float = LOOP_GAIN) -> np.ndarray:
    """闭环一步修正量：参考 Lab 与渲染实测 Lab 之差（反向过冲的步长）。

    链路欠涂（实测比参考淡）→ Δ 为正 → 目标色沿该方向过冲，抵消 pigment-safe
    阻尼与壳层 alpha 合成的浓度损失；单步封顶 LOOP_STEP_MAX 防发散。"""
    d = (np.asarray(ref, np.float64) - np.asarray(measured, np.float64)) * float(gain)
    d[0] = np.clip(d[0], -LOOP_STEP_MAX, LOOP_STEP_MAX)
    d[1:] = np.clip(d[1:], -LOOP_STEP_MAX, LOOP_STEP_MAX)
    return d


# ---------------- 迁移系数最小二乘自求解 ----------------

def _residual(cur: np.ndarray, tgt: np.ndarray, w: np.ndarray,
              kL: float, chroma: float) -> np.ndarray:
    """_lab_migrate 输出与参考色 tgt 的 Lab 残差（展平，供最小二乘）。"""
    col = UvMakeupBaker._lab_migrate(cur, tgt, np.full(len(cur), float(kL)),
                                     np.full(len(cur), float(chroma)), w,
                                     full=True)
    return (rgb2lab(np.clip(col, 0, 1)) - rgb2lab(tgt)).ravel()


def solve_region_coeffs(cur: np.ndarray, tgt: np.ndarray, w: np.ndarray,
                        init: tuple[float, float] | None = None
                        ) -> tuple[float, float]:
    """在真实素颜底色 cur 上求 (kL, chroma)，使迁移结果复现参考色 tgt。

    粗网格定初值 + Levenberg 阻尼高斯-牛顿（Lab 残差 2-参数最小二乘）。
    目标函数在色相旋转/色度外推下非凸，单靠梯度易落进"色度保底"平坦区，
    故先粗搜。返回落到 COEFF_BOUNDS 内的解。"""
    cur = np.asarray(cur, np.float64)
    tgt = np.asarray(tgt, np.float64)
    w = np.asarray(w, np.float64)
    if len(cur) < 4:
        return init or (0.25, 1.0)
    (kl_lo, kl_hi), (ch_lo, ch_hi) = COEFF_BOUNDS

    def _cost(p) -> float:
        r = _residual(cur, tgt, w, *p)
        return float(np.mean(r * r))

    if init is None:
        best = None
        for kl in np.linspace(kl_lo, kl_hi, GRID):
            for ch in np.linspace(ch_lo, ch_hi, GRID):
                c = _cost((kl, ch))
                if best is None or c < best[0]:
                    best = (c, (float(kl), float(ch)))
        p = np.array(best[1], np.float64)
    else:
        p = np.clip(np.asarray(init, np.float64),
                    [kl_lo, ch_lo], [kl_hi, ch_hi])

    lam = 1e-2
    cost = _cost(p)
    for _ in range(GN_ITERS):
        r0 = _residual(cur, tgt, w, p[0], p[1])
        jac = np.empty((len(r0), 2), np.float64)
        for k, step in enumerate((0.02, 0.05)):
            q = p.copy()
            q[k] = min(max(q[k] + step, COEFF_BOUNDS[k][0]), COEFF_BOUNDS[k][1])
            h = q[k] - p[k]
            if abs(h) < 1e-9:
                jac[:, k] = 0.0
                continue
            jac[:, k] = (_residual(cur, tgt, w, q[0], q[1]) - r0) / h
        jtj = jac.T @ jac
        jtr = jac.T @ r0
        improved = False
        for _try in range(4):
            try:
                delta = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-9),
                                        -jtr)
            except np.linalg.LinAlgError:
                break
            cand = np.clip(p + delta, [kl_lo, ch_lo], [kl_hi, ch_hi])
            cc = _cost(cand)
            if cc < cost:
                p, cost, lam = cand, cc, max(lam * 0.4, 1e-4)
                improved = True
                break
            lam *= 6.0
        if not improved:
            break
    return float(p[0]), float(p[1])


def compare_coeffs(cur: np.ndarray, tgt: np.ndarray, w: np.ndarray,
                   manual: tuple[float, float] | None = None) -> dict:
    """参考驱动 vs 手工系数的逐区域 ΔE00 对照（P2 验收报告主体）。"""
    cur = np.asarray(cur, np.float64)
    tgt = np.asarray(tgt, np.float64)
    w = np.asarray(w, np.float64)
    manual = manual or (0.25, 1.0)

    def _delta_e(cf: tuple[float, float]) -> float:
        col = UvMakeupBaker._lab_migrate(cur, tgt, np.full(len(cur), cf[0]),
                                         np.full(len(cur), cf[1]), w, full=True)
        return float(np.mean(ciede2000(rgb2lab(np.clip(col, 0, 1)),
                                       rgb2lab(tgt))))

    solved = solve_region_coeffs(cur, tgt, w, init=manual)
    d_manual, d_solved = _delta_e(manual), _delta_e(solved)
    return {"manual": [round(float(x), 3) for x in manual],
            "solved": [round(float(x), 3) for x in solved],
            "delta_e_manual": round(d_manual, 2),
            "delta_e_solved": round(d_solved, 2),
            "gain": round(d_manual - d_solved, 2)}


# ---------------- UV 空间向心度 → 逐 splat 配对 ----------------

def region_cent_uv(maps, region: str) -> np.ndarray | None:
    """该区域在 UV 空间的向心度场（0 边界 → 1 核心）。

    与参考图剖面同一定义，因此 splat 的 uv 一采样就把"它在这个区域的哪一层"
    （唇内/唇线、腮红落点/外缘）对齐到参考图像素上。"""
    w = None
    if region == "lipstick":
        w = getattr(maps, "lip_w", None)
    if w is None:
        for lf in (getattr(maps, "layer_fields", None) or []):
            if lf.get("region") == region:
                w = lf.get("w")
    if w is None or float(np.max(w)) <= 1e-4:
        return None
    return inness_field(w > 0.25 * float(np.max(w)))


def pair_region_samples(cloud: dict, maps, uv: np.ndarray, valid: np.ndarray,
                        region: str, profile: tuple[np.ndarray, np.ndarray],
                        min_t: float = 0.05, cap: int = 4000
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(素颜底色 cur, 参考目标色 tgt, 权重 w) 配对样本——求解器的输入。

    tgt 由参考图剖面在该 splat 的向心度处采样：同一区域里"唇线"的 splat 拿到
    参考图唇线的像素色、"唇中心"的拿到唇中心的。参考色的空间层次因此逐 splat
    落到资产上，而不是全区域一个均值。"""
    from .makeup_pack import sample_uv
    cent = region_cent_uv(maps, region)
    if cent is None:
        return None
    uv = np.asarray(uv, np.float64)
    t = sample_uv(cent, uv[:, 0], uv[:, 1])
    mem = np.asarray(valid, bool) & (t > min_t)
    if int(mem.sum()) < 30:
        return None
    idx = np.nonzero(mem)[0]
    if len(idx) > cap:
        idx = idx[np.linspace(0, len(idx) - 1, cap).round().astype(int)]
    cur = np.clip(np.asarray(cloud["rgba"], np.float64)[idx, :3], 0, 1)
    tgt = np.clip(lab2rgb(sample_profile_lab(profile, t[idx])), 0, 1)
    return cur, tgt, np.ones(len(idx), np.float64)


# ---------------- 渲染帧实测（闭环验证） ----------------

def measure_region_lab(img_rgb: np.ndarray, px: np.ndarray,
                       regions: tuple[str, ...], mask_thr: float = 0.6
                       ) -> dict[str, np.ndarray]:
    """渲染帧妆区实测 Lab（均值）——闭环里"重测"那一步的观测端。

    与 region_delta_e 同一套蒙版语义（该帧观测地标栅格化），但返回实测 Lab
    本身而非 ΔE，便于与参考剖面直接对照。"""
    h, w = img_rgb.shape[:2]
    masks = image_region_masks(px, w, h, regions=regions)
    rgb = np.asarray(img_rgb, np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    out: dict[str, np.ndarray] = {}
    for r, m in masks.items():
        sel = np.asarray(m, np.float32) > mask_thr
        if int(sel.sum()) < 60:
            continue
        out[r] = np.median(rgb2lab(rgb[sel]), axis=0)
    return out


def reference_core_lab(profile: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """剖面的"核心"代表色（t 最大箱的 Lab）——与实测核心色对照。"""
    ts, lab = profile
    return np.asarray(lab[int(np.argmax(ts))], np.float64)


def open_loop_delta_e(measured: np.ndarray, target: np.ndarray) -> float:
    """实测 Lab vs 参考 Lab 的 ΔE00（闭环验证标量）。"""
    return float(ciede2000(np.asarray(measured, np.float64)[None, :],
                           np.asarray(target, np.float64)[None, :])[0])


def manual_coeff(region: str) -> tuple[float, float]:
    """手工 PHOTOREAL_LAB 表值（无参考时的初值 / 对照基线）。"""
    return tuple(PHOTOREAL_LAB.get(region, (0.25, 1.0)))     # type: ignore[return-value]


def save_coeff_table(coeffs: dict[str, tuple[float, float]], path: str | Path
                     ) -> Path:
    """求解出的系数表落盘（可离线升级 PHOTOREAL_LAB 的观测版初值）。"""
    import json
    path = Path(path)
    path.write_text(json.dumps(
        {k: [round(float(a), 4), round(float(b), 4)] for k, (a, b) in coeffs.items()},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return path