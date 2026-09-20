"""calibrate — 妆效参考图取色标定 + 逐区域 ΔE00 还原度指标。

颜色还原此前没有输入端闭环：presets 是手写 hex，"还原到参考妆效"既不能
标定也不能度量。本模块补两端：

    标定（参考图 → spec）
        参考图 MediaPipe 地标 → 图像空间区域蒙版（多边形/椭圆/折线直接由
        地标索引栅格化，不依赖 canonical 3D 配准）→ 逐区域 Lab 统计 →
        与用户素颜底色相减得 pigment → 生成 spec 的 color_stops/opacity。
        EleGANt/Stable-Makeup 的参考图直接可用——妆效还原第一次有可执行定义。

    度量（渲染帧 vs spec → ΔE00）
        CIEDE2000（Sharma 2005 实现，带测试基準值）：渲染帧妆区像素与 spec
        目标色的逐区域平均色差进 report.json——还原度优化从此有标尺。

蒙版索引为 MediaPipe FaceMesh canonical 468 序（与 tracker.detect 输出一致）。
"""
from __future__ import annotations

import numpy as np

from .makeup_uv import hex_to_rgb01, lab2rgb, rgb2lab

# ---------------- FaceMesh 468 索引（图像空间蒙版用） ----------------

FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
LIPS_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185)
# 眼睑/眉折线内→外序与 makeup_uv 共享（3D 锚定同源，保证标定/烘焙语义一致）
from .makeup_uv import BROW_PTS, EYE_INNER, EYE_OUTER, EYE_UPPER  # noqa: E402


# ---------------- CIEDE2000 ----------------

def _hue_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = np.degrees(np.arctan2(b, a))
    return np.mod(h, 360.0)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 色差。lab1/lab2 (n,3)（L 0..100）。Sharma et al. 2005 实现。"""
    L1, a1, b1 = (np.asarray(lab1, np.float64).T)
    L2, a2, b2 = (np.asarray(lab2, np.float64).T)
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.where((a1p == 0) & (b1 == 0), 0.0, _hue_deg(a1p, b1))
    h2p = np.where((a2p == 0) & (b2 == 0), 0.0, _hue_deg(a2p, b2))

    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, dh)
    dh = np.where(dh < -180, dh + 360, dh)
    zero_c = (C1p * C2p) == 0
    dhp = np.where(zero_c, 0.0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hbarp = np.where(np.abs(h1p - h2p) <= 180, 0.5 * hsum,
                     np.where(hsum < 360, 0.5 * (hsum + 360),
                              0.5 * (hsum - 360)))
    hbarp = np.where(zero_c, hsum, hbarp)
    T = (1 - 0.17 * np.cos(np.radians(hbarp - 30))
         + 0.24 * np.cos(np.radians(2 * hbarp))
         + 0.32 * np.cos(np.radians(3 * hbarp + 6))
         - 0.20 * np.cos(np.radians(4 * hbarp - 63)))
    d_theta = 30 * np.exp(-((hbarp - 275) / 25) ** 2)
    RC = 2 * np.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25.0 ** 7))
    SL = 1 + 0.015 * (Lbarp - 50) ** 2 / np.sqrt(20 + (Lbarp - 50) ** 2)
    SC = 1 + 0.045 * Cbarp
    SH = 1 + 0.015 * Cbarp * T
    RT = -np.sin(np.radians(2 * d_theta)) * RC
    return np.sqrt(
        (dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH))


# ---------------- 图像空间区域蒙版 ----------------

def _valid_pts(px: np.ndarray, idx) -> np.ndarray | None:
    pts = np.asarray(px, np.float32)[list(idx)]
    if (np.linalg.norm(pts, axis=1) <= 0).any():
        return None
    return pts


def _fill_polygon(poly: np.ndarray, w: int, h: int, blur: float) -> np.ndarray:
    import cv2
    m = np.zeros((h, w), np.float32)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1.0)
    return cv2.GaussianBlur(m, (0, 0), blur)


def image_region_masks(px: np.ndarray, w: int, h: int,
                       regions: tuple[str, ...] | None = None
                       ) -> dict[str, np.ndarray]:
    """(478,2) 像素地标 → 图像空间区域蒙版（0..1，羽化边缘）。

    只用几何（多边形/椭圆/折线），不依赖任何分割模型；单帧即可，
    用于参考图取色与渲染帧 ΔE 评测两个方向。"""
    import cv2

    px = np.asarray(px, np.float32)
    feather = max(1.0, min(w, h) * 0.008)
    thin = max(1.5, min(w, h) * 0.004)
    out: dict[str, np.ndarray] = {}
    want = set(regions or ("foundation", "lipstick", "eyeshadow", "eyebrow",
                           "eyeliner", "blush"))

    if "foundation" in want:
        oval = _valid_pts(px, FACE_OVAL)
        if oval is not None:
            out["foundation"] = _fill_polygon(oval, w, h, feather * 2)
    if "lipstick" in want:
        lip = _valid_pts(px, LIPS_OUTER)
        if lip is not None:
            out["lipstick"] = _fill_polygon(lip, w, h, feather)

    eye_w = 0.0
    eyes = []
    for side in ("left", "right"):
        up = _valid_pts(px, EYE_UPPER[side])
        br = _valid_pts(px, BROW_PTS[side])
        if up is None:
            continue
        eye_mid = up.mean(0)
        brow_mid = br.mean(0) if br is not None else eye_mid
        eye_w += float(np.linalg.norm(px[EYE_INNER[side]] - px[EYE_OUTER[side]]))
        eyes.append((up, eye_mid, brow_mid, side))
    eye_w = eye_w / max(len(eyes), 1)

    if eyes and "eyeshadow" in want:
        acc = np.zeros((h, w), np.float32)
        for up, eye_mid, brow_mid, _side in eyes:
            lift = (brow_mid - eye_mid)
            nl = np.linalg.norm(lift)
            lift = lift / nl * (0.9 * nl + 0.35 * eye_w) if nl > 1e-3 \
                else np.array([0.0, -0.35 * eye_w], np.float32)
            poly = np.vstack([up, (up + lift)[::-1]])
            acc = np.maximum(acc, _fill_polygon(poly, w, h, feather))
        out["eyeshadow"] = acc
    if eyes and "eyeliner" in want:
        acc = np.zeros((h, w), np.float32)
        for up, _em, _bm, _side in eyes:
            cv2.polylines(acc, [np.round(up).astype(np.int32)], False, 1.0,
                          thickness=max(2, int(round(thin * 1.5))), lineType=cv2.LINE_AA)
        out["eyeliner"] = cv2.GaussianBlur(acc, (0, 0), thin * 0.6)
    if eyes and "eyebrow" in want:
        acc = np.zeros((h, w), np.float32)
        for side in ("left", "right"):
            br = _valid_pts(px, BROW_PTS[side])
            if br is None:
                continue
            cv2.polylines(acc, [np.round(br).astype(np.int32)], False, 1.0,
                          thickness=max(3, int(round(eye_w * 0.16))),
                          lineType=cv2.LINE_AA)
        out["eyebrow"] = cv2.GaussianBlur(acc, (0, 0), feather)
    if eyes and "blush" in want:
        acc = np.zeros((h, w), np.float32)
        for up, eye_mid, _bm, side in eyes:
            outer = px[EYE_UPPER[side][-1]]          # 外眼角（折线末点）
            d = np.array([outer[0] - eye_mid[0], outer[1] - eye_mid[1]], np.float32)
            d = d / (np.linalg.norm(d) + 1e-6)
            c = eye_mid + d * (0.55 * eye_w) + np.array([0.0, 0.62 * eye_w], np.float32)
            axes = (0.52 * eye_w, 0.36 * eye_w)
            cv2.ellipse(acc, (int(round(c[0])), int(round(c[1]))),
                        (int(axes[0]), int(axes[1])), 0, 0, 360, 1.0, -1)
        out["blush"] = cv2.GaussianBlur(acc, (0, 0), feather * 1.5)
    return out


# ---------------- 参考图取色 → spec ----------------

def extract_makeup_colors(img_bgr: np.ndarray, px: np.ndarray | None = None
                          ) -> dict[str, dict]:
    """参考妆照 → 逐区域 {"hex_dark","hex_light","opacity","lab"}。

    区域色 = 蒙版核心（>0.75）像素的 Lab 均值（a/b）与 45/85 分位明度两档
    （色带 dark/light stops）；opacity 由区域 chroma 超出底妆区（肤色基准）
    的幅度估计。px 缺省时用 MediaPipe 现场检测。"""
    import cv2

    h, w = img_bgr.shape[:2]
    if px is None:
        from ...tracker import FaceTracker
        tracker = FaceTracker(smooth=False)
        try:
            det = tracker.detect(img_bgr, 0.0)
        finally:
            tracker.close()
        if det is None:
            raise RuntimeError("参考图未检测到人脸")
        px = det["px"]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
    masks = image_region_masks(px, w, h)
    base_mask = masks.get("foundation")
    base_chroma = 0.0
    if base_mask is not None and (base_mask > 0.75).sum() > 100:
        base_lab = rgb2lab(rgb[base_mask > 0.75])
        base_chroma = float(np.median(np.hypot(base_lab[:, 1], base_lab[:, 2])))
    out: dict[str, dict] = {}
    for region, m in masks.items():
        core = m > 0.75
        if int(core.sum()) < 60:
            continue
        lab = rgb2lab(rgb[core])
        a_m, b_m = float(lab[:, 1].mean()), float(lab[:, 2].mean())
        l_lo, l_hi = np.percentile(lab[:, 0], [45, 85])
        chroma = float(np.hypot(a_m, b_m))
        # opacity：妆区 chroma 相对肤色基准的超出量（底妆几乎无 chroma 差 → 低）
        op = float(np.clip((chroma - base_chroma) / 45.0, 0.30, 0.95))
        if region in ("foundation", "concealer"):
            op = float(np.clip(0.65 - abs(lab[:, 0].mean() - 65) / 160, 0.25, 0.6))
        stops = []
        for lv in (l_lo, l_hi):
            rgb_s = lab2rgb(np.array([[lv, a_m, b_m]]))[0]
            stops.append("#%02X%02X%02X" % tuple(int(round(c * 255)) for c in rgb_s))
        out[region] = {"hex_dark": stops[0], "hex_light": stops[1],
                       "opacity": round(op, 3),
                       "lab": [round(float(x), 2) for x in
                               (float(lab[:, 0].mean()), a_m, b_m)]}
    return out


def calibrate_spec(template: dict, ref_bgr: np.ndarray,
                   px: np.ndarray | None = None,
                   profiles: dict | None = None) -> dict:
    """模板 spec × 参考妆照 → 标定后的 spec（逐区域替换色带与 opacity）。

    形状/finish/层结构沿用模板；只有"颜色与浓度"来自参考图观测。
    P2：颜色优先走参考图 Lab **剖面**（区域内"边界→核心"多档色带，保留唇的
    内深外浅/眼影层次/腮红落点），剖面不可用时退回两档均值（extract_makeup_colors
    的 hex_dark/hex_light）；profiles 可由调用方预先算好复用（避免重复检地标）。"""
    import copy

    if profiles is None:
        from .colorfield import extract_profiles      # 延迟导入避免循环
        prof_all = extract_profiles(
            ref_bgr, [l.get("region") for l in template.get("layers", [])
                      if l.get("region")], px=px)
    else:
        prof_all = profiles

    extracted = extract_makeup_colors(ref_bgr, px=px)
    spec = copy.deepcopy(template)
    hit = []
    for layer in spec.get("layers", []):
        region = layer.get("region")
        if region in prof_all:
            from .colorfield import profile_to_stops
            layer["color_stops"] = profile_to_stops(prof_all[region])
        elif region in extracted:
            e = extracted[region]
            layer["color_stops"] = [{"at": 0.0, "hex": e["hex_dark"]},
                                    {"at": 1.0, "hex": e["hex_light"]}]
        else:
            continue
        if region in extracted:
            layer["opacity"] = extracted[region]["opacity"]
        hit.append(region)
    spec.setdefault("calibration", {})
    spec["calibration"] = {"regions": hit, "extracted": extracted,
                           "profile_regions": list(prof_all)}
    return spec


# ---------------- 还原度度量（渲染帧 vs spec） ----------------

# 验收门（ΔE00 预算）：文档路线图"超阈值自动降级/重标定"的落地。
# 唇是妆面核心 SKU 且色度强，预算更紧；未列出的区域用默认预算。
FIDELITY_BUDGET = {"lipstick": 12.0}
FIDELITY_BUDGET_DEFAULT = 14.0
FIDELITY_RETRY_BOOST = 1.25      # 超预算区域 opacity 提升系数（重烘一次）


def fidelity_gate(delta_e: dict, budget: dict | None = None
                  ) -> dict[str, object]:
    """逐区域 ΔE00 vs 预算 → {"status": passed|over, "over": [region...]}。

    delta_e 空（无妆区可评）视为 passed——门只对"有度量"的区域生效。"""
    bud = dict(FIDELITY_BUDGET)
    if budget:
        bud.update(budget)
    over = []
    for region, v in (delta_e or {}).items():
        if region.startswith("_"):
            continue
        if v > bud.get(region, FIDELITY_BUDGET_DEFAULT):
            over.append(region)
    return {"status": "passed" if not over else "over", "over": over,
            "budget": {k: bud.get(k, FIDELITY_BUDGET_DEFAULT)
                       for k in (delta_e or {}) if not k.startswith("_")}}


def boost_spec_regions(spec: dict, regions: list[str],
                       factor: float = FIDELITY_RETRY_BOOST) -> dict:
    """超预算区域 opacity 上调（封顶 1.0）——还原度自动重标定的单步动作。

    只动浓度不动颜色：色差偏大通常因为 pigment-safe 迁移被"过度保守"，
    提浓度直接缩小 ΔE；改颜色反而破坏已标定的目标色语义。"""
    import copy

    out = copy.deepcopy(spec)
    for layer in out.get("layers", []):
        if layer.get("region") in regions:
            layer["opacity"] = float(min(1.0,
                                         float(layer.get("opacity", 0.7)) * factor))
    return out


def spec_targets(spec: dict) -> dict[str, np.ndarray]:
    """spec → region → 目标 Lab（色带两端均值，即烘焙语义的"妆后目标色"）。"""
    targets: dict[str, np.ndarray] = {}
    for layer in spec.get("layers", []):
        if not layer.get("enabled", True):
            continue
        stops = layer.get("color_stops") or []
        if not stops:
            continue
        rgbs = np.stack([hex_to_rgb01(s["hex"]) for s in stops])
        rgb_mean = np.clip(rgbs.mean(0), 0, 1)
        targets[layer["region"]] = rgb2lab(rgb_mean[None, :])[0]
    return targets


# 图像空间区域蒙版互相包含：foundation 是整脸椭圆（含五官），直接度量的
# "底妆区"会把眼影/唇/眉像素算进去，ΔE 被特征色拉爆（实测全区域假性超预算）。
# 度量前把这些特征妆区从底妆区中扣除。
FOUNDATION_EXCLUDE = ("lipstick", "eyeshadow", "eyebrow", "eyeliner")


def exclude_region_overlap(masks: dict[str, np.ndarray],
                           base: str = "foundation",
                           exclude: tuple[str, ...] = FOUNDATION_EXCLUDE
                           ) -> dict[str, np.ndarray]:
    """返回扣除特征妆区后的蒙版组（不改输入）。非 base 区域原样透传。"""
    out = dict(masks)
    if base not in out:
        return out
    acc = None
    for r in exclude:
        if r in out:
            acc = out[r] if acc is None else np.maximum(acc, out[r])
    if acc is not None:
        out[base] = np.clip(out[base] - acc, 0.0, 1.0)
    return out


def region_delta_e(img_rgb: np.ndarray, masks: dict[str, np.ndarray],
                   targets: dict[str, np.ndarray]) -> dict[str, float]:
    """渲染帧妆区核心像素 vs spec 目标色的逐区域平均 ΔE00（低=还原好）。"""
    out: dict[str, float] = {}
    for region, tgt in targets.items():
        m = masks.get(region)
        if m is None or (m > 0.75).sum() < 60:
            continue
        lab = rgb2lab(img_rgb[m > 0.75])
        d = ciede2000(lab, np.tile(tgt, (len(lab), 1)))
        out[region] = round(float(np.mean(d)), 2)
    return out
