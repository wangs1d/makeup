"""semantics — 妆区语义观测（P1）：把"涂在哪"从几何模板换成图像语义观测。

根因：canonical 模板唇带/眼线带在嘴/眼等高频部位有 ~2% 系统错位，地标三角化
不可靠时锚定带又常退化成 UV 兜底——"涂在哪"一直是几何先验在兜。本模块把
face-parsing 的逐像素分割（唇/皮肤/眉/眼）沿相机投影投到每个 splat 上，
多视角投票出逐 splat 的区域概率，作为比几何模板高一级的观测证据。

四级回退（每级在 report.json 留痕，低一级只在高一级缺席/不可信时兜底）：
    1) multiview_seg   多视角分割投票（≥2 视角观测同一 splat）
    2) single_seg      单图分割（LAM 单图入口 / 仅一帧覆盖）
    3) landmark_band   3D 地标锚定带（fit_makeup 三角化地标，geometry）
    4) uv_template     canonical UV 模板带（纯几何先验，最后的兜底）

融合本身在 makeup_uv._apply_zones（场 → 权重/目标色/系数的唯一入口），本模块
只负责"观测 → 场"。纯 numpy/cv2，投影/投票/吸附/IoU 均可离线单测；face-parsing
依赖经 FaceParser 门控注入，权重缺失时 prepare_views 返回 []（主链路不阻断）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import colmap_io

# region → face-parsing 语义组（None = 无对应语义类别，保持几何兜底）
REGION_GROUPS: dict[str, tuple[str, ...] | None] = {
    "lipstick": ("lips",),
    "foundation": ("skin",),
    "eyebrow": ("brow_l", "brow_r"),
    "eyeshadow": ("eye_l", "eye_r"),
    "eyeliner": ("eye_l", "eye_r"),
    "lashes": ("eye_l", "eye_r"),
    "concealer": None, "contour": None, "blush": None, "highlight": None,
}
# 观测即边界（唇/眉：语义分割的形状就是真相）vs 观测只收边界（地标/模板给
# 形状，语义只防漏涂与越界）——见 makeup_uv._apply_zones 的 mode 语义
REPLACE_REGIONS = ("lipstick", "eyebrow")
# 逐级可信度 / 几何兜底幅度（fallback_ratio：观测缺席处保留多少几何先验）
TRUST = {"multiview_seg": 0.92, "single_seg": 0.72}
FALLBACK_RATIO = {"multiview_seg": 0.12, "single_seg": 0.40}
# multiply 区域（眼线/睫毛/眼影）：眼部分割给的是眼球而非睑线，只能低压收边
MULTIPLY_TRUST = {"eyeliner": 0.45, "lashes": 0.45, "eyeshadow": 0.70,
                  "foundation": 0.85}
MULTIPLY_FALLBACK = {"eyeliner": 0.85, "lashes": 0.85, "eyeshadow": 0.55,
                     "foundation": 0.45}

MULTIVIEW_MIN = 2        # 同一 splat 被多少视角观测才算"多视角"
MIN_ZONE_SPLATS = 40     # 区域命中 splat 少于此值判定观测缺席 → 几何兜底
SNAP_SEARCH = 3          # 唇线吸附搜索半径（像素）
SNAP_REGIONS = ("lipstick", "eyebrow")


# ---------------- 投影 / 采样 ----------------

def camera_matrix(cam, img_w: int) -> np.ndarray:
    """COLMAP 相机 → 该图像像素空间的 3×3 内参（SIMPLE_RADIAL: fx,cx,cy,k1）。

    选帧的观测像素空间是放大帧（SR/超分）空间，与 cam.width 差一个比例——
    直接拿原生内参投影会整体缩放错位，必须按实际图像宽度重标。"""
    s = float(img_w) / float(cam.width)
    fx, cx, cy = (float(cam.params[0]) * s, float(cam.params[1]) * s,
                  float(cam.params[2]) * s)
    return np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], np.float64)


def project_points(xyz: np.ndarray, R: np.ndarray, t: np.ndarray,
                   K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """世界系 (n,3) → 像素 (n,2) + 可见布尔（cheirality z>0）。"""
    cam = np.asarray(xyz, np.float64) @ np.asarray(R, np.float64).T \
        + np.asarray(t, np.float64)
    p = cam @ np.asarray(K, np.float64).T
    z = p[:, 2]
    ok = z > 1e-9
    xy = p[:, :2] / np.maximum(z, 1e-9)[:, None]
    xy[~ok] = -1e4
    return xy, ok


def front_facing(xyz: np.ndarray, R: np.ndarray, t: np.ndarray,
                 normals: np.ndarray) -> np.ndarray:
    """背面剔除：法线朝相机的 splat 才算可见。

    头是凸的——后脑勺 splat 在前视图中投影落在脸部轮廓内（与唇/眼区重叠），
    不做朝向剔除会把后脑勺的观测算进投票，把概率稀释成噪声。"""
    cam_c = -np.asarray(R, np.float64).T @ np.asarray(t, np.float64)
    d = np.asarray(xyz, np.float64) - cam_c
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    return (np.asarray(normals, np.float64) * -d).sum(1) > 0.0


def sample_mask(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """(h,w) 蒙版在像素 xy 的双线性采样；越界返回 0。"""
    h, w = mask.shape[:2]
    m = np.asarray(mask, np.float32)
    x, y = xy[:, 0], xy[:, 1]
    inb = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    gx = np.clip(x, 0.0, w - 1.001)
    gy = np.clip(y, 0.0, h - 1.001)
    x0 = gx.astype(np.int32)
    y0 = gy.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = gx - x0
    fy = gy - y0
    out = (m[y0, x0] * (1 - fx) * (1 - fy) + m[y0, x1] * fx * (1 - fy)
           + m[y1, x0] * (1 - fx) * fy + m[y1, x1] * fx * fy)
    return np.where(inb, out, 0.0).astype(np.float32)


# ---------------- 语义蒙版装配 ----------------

def group_mask(parsed: dict[str, np.ndarray], groups: tuple[str, ...]) -> np.ndarray | None:
    """语义组 → 0/1 蒙版（多组取并）；组全缺返回 None。"""
    ms = [np.asarray(parsed[g], np.float32) for g in groups
          if parsed.get(g) is not None]
    if not ms:
        return None
    m = np.max(np.stack(ms, 0), 0)
    return np.clip(m, 0.0, 1.0) if m.max() > 0.5 else None


def _kernel(px: float) -> np.ndarray:
    k = max(1, int(round(px)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def region_mask(region: str, parsed: dict[str, np.ndarray],
                width: int) -> np.ndarray | None:
    """语义解析 → 该妆区的 0/1 边界蒙版（尚未 edge-snap / 羽化）。

    逐区域形状策略（分割给的是"解剖结构"，妆区是"结构上的薄层"）：
        lipstick/eyebrow/foundation  直接用对应语义结构（唇/眉/皮肤）；
        eyeshadow   眼结构外扩 ~3.5% 帧宽（睑部区域）；
        eyeliner/lashes  眼球内外环带（睑线所在），非眼球本体——眼球本体上
                         画眼线是错的，环带才是收边信息的载体。"""
    groups = REGION_GROUPS.get(region)
    if not groups:
        return None
    m = group_mask(parsed, groups)
    if m is None:
        return None
    if region == "eyeshadow":
        return np.clip(_dilate(m, 0.035 * width), 0.0, 1.0)
    if region in ("eyeliner", "lashes"):
        ring = _dilate(m, 0.030 * width) - _dilate(m, 0.008 * width)
        ring = np.clip(ring, 0.0, 1.0)
        return ring if ring.max() > 0.5 else None
    return m


def _dilate(m: np.ndarray, px: float) -> np.ndarray:
    return cv2.dilate((np.asarray(m, np.float32) > 0.5).astype(np.uint8),
                      _kernel(px)).astype(np.float32)


# ---------------- 唇线 edge-snap ----------------

def snap_to_edge(mask: np.ndarray, gray: np.ndarray,
                 search: int = SNAP_SEARCH, level: float = 0.5) -> np.ndarray:
    """0.5 等值线沿法线吸附到原图 |∇gray| 极值处（可见唇线/眉缘）。

    分割上采样后的边界模糊、模板边界偏 1-3px；原图里唇红与皮肤的过渡有真实
    梯度极值，把等值线拉到那里，边界就贴在可见唇线上。返回硬蒙版（0/1）。"""
    hard = (np.asarray(mask, np.float32) >= level).astype(np.uint8)
    if hard.sum() < 16:
        return hard.astype(np.float32)
    g = cv2.GaussianBlur(np.asarray(gray, np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    h, w = mag.shape[:2]
    contours, hier = cv2.findContours(hard, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return hard.astype(np.float32)
    offs = np.arange(-search, search + 1, dtype=np.float64)
    out = np.zeros((h, w), np.uint8)
    levels = (hier[0][:, 3] if hier is not None else np.full(len(contours), -1))
    for ci, c in enumerate(contours):
        if len(c) < 8:
            out = cv2.drawContours(out, [c], -1, 1 if levels[ci] < 0 else 0, -1)
            continue
        P = c[:, 0, :].astype(np.float64)
        nxt = np.roll(P, -1, axis=0)
        prv = np.roll(P, 1, axis=0)
        tang = nxt - prv
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        # 朝外定向：法线指向轮廓外侧才能把等值线推出/拉入到真实边缘
        cen = P.mean(0)
        nrm *= np.sign(((P - cen) * nrm).sum(1))[:, None]
        samp = P[:, None, :] + offs[None, :, None] * nrm[:, None, :]
        xs = np.clip(samp[..., 0], 0, w - 1).astype(np.int32)
        ys = np.clip(samp[..., 1], 0, h - 1).astype(np.int32)
        gwin = mag[ys, xs]
        gmax = gwin.max(axis=1)
        ghere = mag[np.clip(np.round(P[:, 1]).astype(np.int32), 0, h - 1),
                    np.clip(np.round(P[:, 0]).astype(np.int32), 0, w - 1)]
        # 只在"邻域存在明显更强的边缘"时才移动：原处已在真边缘（gmax≈ghere）
        # 或邻域全平（gmax≈0）都保持不动，避免 argmax 在平区退化成固定偏移
        move = gmax > np.maximum(ghere * 1.10, 0.02)
        best = np.where(move, offs[np.argmax(gwin, axis=1)], 0.0)
        P2 = np.round(P + best[:, None] * nrm).astype(np.int32)
        out = cv2.drawContours(out, [P2.reshape(-1, 1, 2)], -1,
                               1 if levels[ci] < 0 else 0, -1)
    return out.astype(np.float32)


def edge_snap(mask: np.ndarray, gray: np.ndarray | None,
              feather_px: float = 0.0, search: int = SNAP_SEARCH) -> np.ndarray:
    """吸附 + 同宽羽化：硬边界吸附到可见边缘后再用 feather_px 软化。

    羽化为各向同性高斯（σ≈feather/2），与 core.feather_px 的距离整形羽化
    在唇/眉尺度上等价（都是"边界带内线性渐入"），但吸附后的边界已无模板
    偏置，软化只负责抗锯齿。"""
    m = np.asarray(mask, np.float32)
    if gray is not None:
        m = snap_to_edge(m, gray, search=search)
    else:
        m = (m >= 0.5).astype(np.float32)
    if feather_px and feather_px > 1.0:
        m = cv2.GaussianBlur(m, (0, 0), max(float(feather_px) * 0.5, 0.6))
    return np.clip(m, 0.0, 1.0).astype(np.float32)


# ---------------- 观测视角准备（face-parsing 门控） ----------------

def image_masks(img_bgr: np.ndarray, parsed: dict[str, np.ndarray],
                regions: tuple[str, ...],
                feather_px: dict[str, float] | None = None
                ) -> dict[str, np.ndarray]:
    """单图语义解析 → 逐妆区蒙版（区域形状策略 + 唇/眉 edge-snap）。

    视频链路（prepare_views）与 LAM 单图入口（lam_adapter.photo_zones）共用：
    两条链路的"观测妆区"必须同一套边界语义，否则单图入口与视频入口在唇线/
    眉缘上会给出不同的妆区。"""
    w = img_bgr.shape[1]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    masks: dict[str, np.ndarray] = {}
    for region in regions:
        m = region_mask(region, parsed, w)
        if m is None:
            continue
        if region in SNAP_REGIONS:
            m = edge_snap(m, gray, (feather_px or {}).get(region, 0.0))
        masks[region] = m
    return masks


def prepare_views(model, names: list[str], images_dir: Path, parser,
                  regions: tuple[str, ...], max_views: int = 8,
                  feather_px: dict[str, float] | None = None,
                  progress=None) -> list[dict]:
    """选帧图像 → 逐妆区语义蒙版 → 观测视角列表（供 vote_regions）。

    门控：parser 不可用（权重未缓存/无 torch）→ 返回 []，调用方保持几何兜底
    （四级回退的 level 3/4），主链路不阻断。"""
    cb = progress or (lambda *a: None)
    if parser is None or not parser.available():
        hint = getattr(parser, "load_error", None) if parser is not None else None
        cb("semantics", 1.0, f"face-parsing 不可用（保持几何兜底）"
                             f"{'：' + hint if hint else ''}")
        return []
    picks = list(names)[:max(1, int(max_views))]
    views: list[dict] = []
    for i, name in enumerate(picks):
        im = model.images.get(name)
        if im is None:
            continue
        img = cv2.imread(str(images_dir / name))
        if img is None:
            continue
        h, w = img.shape[:2]
        parsed = parser.parse(img)
        if not parsed:
            continue
        masks = image_masks(img, parsed, regions, feather_px)
        if not masks:
            continue
        R = colmap_io.quat_to_rotmat(np.asarray(im["qvec"], np.float64))
        views.append({"name": name, "R": R,
                      "t": np.asarray(im["tvec"], np.float64),
                      "K": camera_matrix(model.camera, w),
                      "masks": masks, "size": (w, h)})
        cb("semantics", (i + 1) / len(picks),
           f"语义观测 {i + 1}/{len(picks)}（{name}，{len(masks)} 区域）")
    return views


# ---------------- 多视角投票 ----------------

def vote_regions(xyz: np.ndarray, views: list[dict], regions: tuple[str, ...],
                 normals: np.ndarray | None = None
                 ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """逐 splat 区域概率 = 可见视角上的分割蒙版采样均值。

    返回 region → (p (n,) 概率, nview (n,) 观测到的可见视角数)。可见 = 相机
    前方（cheirality）+ 落在图像内 + （有法线时）正面朝向。"""
    n = len(xyz)
    acc = {r: np.zeros(n, np.float64) for r in regions}
    cnt = {r: np.zeros(n, np.float64) for r in regions}
    for v in views:
        xy, ok = project_points(xyz, v["R"], v["t"], v["K"])
        w, h = v["size"]
        inb = ok & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) \
            & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
        if normals is not None:
            inb = inb & front_facing(xyz, v["R"], v["t"], normals)
        if not inb.any():
            continue
        for r in regions:
            m = v["masks"].get(r)
            if m is None:
                continue
            acc[r] += sample_mask(m, xy) * inb
            cnt[r] += inb
    out = {}
    for r in regions:
        c = cnt[r]
        out[r] = (np.where(c > 0, acc[r] / np.maximum(c, 1.0), 0.0
                           ).astype(np.float32), c.astype(np.float32))
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """逐 splat 二值妆区 IoU（空并集返回 0）。"""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    u = int((a | b).sum())
    return float((a & b).sum()) / float(u) if u else 0.0


# ---------------- 观测场装配 ----------------

@dataclass
class ObservedZones:
    """观测妆区场 + 逐区域来源留痕（写进 report.json 的 makeup_zones）。"""
    fields: "object" = None                    # ZoneFields（延迟导入避免环）
    report: dict = field(default_factory=dict)

    def summary(self) -> str:
        lv: dict[str, int] = {}
        for r in self.report.values():
            lv[r.get("level", "?")] = lv.get(r.get("level", "?"), 0) + 1
        return " ".join(f"{k}×{v}" for k, v in sorted(lv.items()))


def build_zones(xyz: np.ndarray, views: list[dict], regions: tuple[str, ...],
                region_scale: dict[str, float] | None = None,
                normals: np.ndarray | None = None,
                geom_w: np.ndarray | None = None,
                geometric_level: dict[str, str] | None = None,
                progress=None) -> ObservedZones:
    """多视角语义投票 → ZoneFields（涂在哪）+ 四级回退留痕。

    region_scale：该区域的"上妆强度"（spec opacity × intensity）——replace
    模式下语义概率直接决定权重，必须带上 spec 强度，否则观测会把"淡涂"拉成
    满涂。geom_w：几何兜底权重（用于 IoU 诊断，可空）。"""
    from .makeup_uv import ZoneFields, ZoneSpec

    cb = progress or (lambda *a: None)
    scale = dict(region_scale or {})
    geom_level = dict(geometric_level or {})
    out = ObservedZones(fields=ZoneFields(), report={})
    obs = [r for r in regions if REGION_GROUPS.get(r)]
    for r in regions:
        if not REGION_GROUPS.get(r):
            out.report[r] = {"level": geom_level.get(r, "uv_template"),
                             "mode": "multiply", "reason": "无对应语义类别"}
    if not obs or not views:
        for r in obs:
            out.report[r] = {"level": geom_level.get(r, "uv_template"),
                             "mode": "multiply", "reason": "无语义观测"}
        return out

    voted = vote_regions(xyz, views, tuple(obs), normals=normals)
    for r in obs:
        p, c = voted[r]
        hit = p > 0.5
        n_hit = int(hit.sum())
        if n_hit < MIN_ZONE_SPLATS:
            out.report[r] = {"level": geom_level.get(r, "uv_template"),
                             "mode": "multiply",
                             "reason": f"观测命中过稀（{n_hit} splats）"}
            continue
        nv = float(np.median(c[hit])) if n_hit else 0.0
        level = "multiview_seg" if nv >= MULTIVIEW_MIN else "single_seg"
        mode = "replace" if r in REPLACE_REGIONS else "multiply"
        if mode == "replace":
            trust, fr = TRUST[level], FALLBACK_RATIO[level]
        else:
            trust = MULTIPLY_TRUST.get(r, TRUST[level])
            fr = MULTIPLY_FALLBACK.get(r, FALLBACK_RATIO[level])
        out.fields.seg[r] = ZoneSpec(p=p, mode=mode,
                                     scale=float(scale.get(r, 1.0)),
                                     trust=float(trust), fallback_ratio=float(fr),
                                     level=level)
        rec = {"level": level, "mode": mode, "trust": round(float(trust), 3),
               "fallback_ratio": round(float(fr), 3), "splats": n_hit,
               "obs_views_median": round(nv, 2),
               "views_used": len(views)}
        if geom_w is not None:
            rec["iou_vs_geometry"] = round(
                iou(hit, np.asarray(geom_w) > 0.05), 4)
        out.report[r] = rec
    cb("semantics", 1.0, "语义妆区 " + (out.summary() or "（无）"))
    return out


# ---------------- 嘴/眼特写对比图（验收） ----------------

# MediaPipe 468 拓扑下的嘴/眼外框关键点（crop 用）
_LM = {"mouth": (61, 291, 0, 17), "eye_l": (33, 133, 159, 145),
       "eye_r": (362, 263, 386, 374)}


def save_zone_closeups(px: np.ndarray, ref_bgr: np.ndarray,
                       bare_bgr: np.ndarray, made_bgr: np.ndarray,
                       out_dir: Path, scale: int = 4,
                       pad: float = 0.35) -> list[Path]:
    """嘴/眼 4× 特写对比（真实帧 | 素颜渲染 | 妆后渲染）——边界验收用。

    真实帧与两张渲染图可能不同分辨率，按各自 landmark 像素换算裁剪。"""
    px = np.asarray(px, np.float64)
    ref_bgr = np.asarray(ref_bgr)
    bare_bgr = np.asarray(bare_bgr) if bare_bgr is not None else None
    made_bgr = np.asarray(made_bgr) if made_bgr is not None else None
    size = float(ref_bgr.shape[0])
    sx = float(ref_bgr.shape[1]) / size
    outs: list[Path] = []
    for zone, ids in _LM.items():
        x0, x1 = float(px[ids[0], 0]), float(px[ids[1], 0])
        y0, y1 = float(px[ids[2], 1]), float(px[ids[3], 1])
        cw, chh = max(x1 - x0, 8.0), max(y1 - y0, 8.0)
        cw *= (1 + 2 * pad)
        chh *= (1 + 2 * pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        def _crop(img: np.ndarray) -> np.ndarray | None:
            if img is None:
                return None
            h, w = img.shape[:2]
            s = float(w) / (size * sx)               # 该图相对参考帧的尺度
            bx = int(np.clip((cx - cw / 2) * s, 0, max(w - 8, 0)))
            by = int(np.clip((cy - chh / 2) * s, 0, max(h - 8, 0)))
            bw = int(min(cw * s, w - bx))
            bh = int(min(chh * s, h - by))
            if bw < 4 or bh < 4:
                return None
            crop = img[by:by + bh, bx:bx + bw]
            tgt = (int(min(cw * s, w) * scale), int(min(chh * s, h) * scale))
            return cv2.resize(crop, tgt, interpolation=cv2.INTER_NEAREST)

        tiles = [t for t in (_crop(ref_bgr), _crop(bare_bgr), _crop(made_bgr))
                 if t is not None]
        if not tiles:
            continue
        hh = min(t.shape[0] for t in tiles)
        tiles = [t[:hh] for t in tiles]
        row = np.concatenate(tiles, axis=1)
        p = Path(out_dir) / f"closeup_{zone}.png"
        cv2.imwrite(str(p), row)
        outs.append(p)
    return outs