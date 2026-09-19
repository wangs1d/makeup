"""makeup_uv — 2048² UV 空间妆容合成器（取代"逐 splat 染色 + 程序化壳层"）。

根因对照：
    旧路径把妆容写成 per-splat 颜色混合 → 锐度受 splat 密度限制（densify 35%
    也补不回唇线），跨视角颜色采中位数 → 花斑；壳层贴片 → 油漆感。
    本模块把妆容合成为 UV 空间的"目标场"（albedo/材质/覆盖），锐度由贴图
    分辨率决定；唇红带由 3D 拓扑光栅化进 UV（张嘴口腔面天然排除）；微观纹理
    （唇纹/粉感/珠光闪点）第一次有来源。烘焙回 splat 时用 Lab 部分迁移——
    皮肤纹理与光影保留，只有颜色走向妆色（与 fit_makeup 真实路径同表同语义）。

材质语义（化妆品 = 薄层材质，不只是颜色）：
    finish → (rough, coat, sheen) 目标 + lipstick 附带 sss；烘焙时
    rough_new = mix(皮肤基准, 目标, w)，coat/sheen/sss 取 max 叠加。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..fit_makeup import FaceMakeupFitter, _load_core

FINISH_TARGET = {   # finish → (rough, coat, sheen)
    "matte": (0.62, 0.02, 0.00),
    "satin": (0.48, 0.10, 0.02),
    "dewy": (0.34, 0.38, 0.12),
    "gloss": (0.16, 0.82, 0.15),
    "glitter": (0.30, 0.30, 0.50),   # 珠光/闪片：sheen 通道由 sparkle 闪点图驱动
}
SKIN_ROUGH = 0.52
MAP_KEYS = ("rough", "coat", "sss", "sheen")


# ---------------- 颜色空间（float 精度 Lab，D65） ----------------
# 旧路径走 uint8 cvtColor 往返：低 opacity 底妆的 Lab 增量 < 1/255 时被量化
# 吞掉，妆面出现可见色带。float 全程无量化。

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    """(n,3) sRGB 0..1 → (n,3) Lab（L 0..100，D65）。"""
    lin = _srgb_to_linear(rgb)
    xyz = lin @ np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]], np.float64).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    lab = np.empty_like(f)
    lab[..., 0] = 116 * f[..., 1] - 16
    lab[..., 1] = 500 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200 * (f[..., 1] - f[..., 2])
    return lab


def lab2rgb(lab: np.ndarray) -> np.ndarray:
    """(n,3) Lab → (n,3) sRGB 0..1（D65，越界裁剪）。"""
    lab = np.asarray(lab, np.float64)
    d = 6 / 29
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    finv = lambda t: np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))
    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1)
    xyz *= np.array([0.95047, 1.0, 1.08883])
    lin = xyz @ np.array([[3.2404542, -1.5371385, -0.4985314],
                          [-0.9692660, 1.8760108, 0.0415560],
                          [0.0556434, -0.2040259, 1.0572252]], np.float64).T
    return np.clip(_linear_to_srgb(lin), 0.0, 1.0)


def hex_to_rgb01(hex_str: str) -> np.ndarray:
    h = hex_str.strip().lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)],
                    np.float64) / 255.0


def lab_adapt(cur_L: np.ndarray, tgt_L: np.ndarray) -> np.ndarray:
    """pigment-safe 明度跟随系数：底色/妆色明度差越大越保守。

    深肤底 + 亮妆 hex 时明度全量内插会把 chroma 拖塌成灰调（还原度随肤色
    系统性漂移的根因）；差 → 0 时回到 1（不削弱原有迁移）。下限 0.40。"""
    return np.clip(1.0 - 0.55 * np.abs(np.asarray(tgt_L, np.float64)
                                       - np.asarray(cur_L, np.float64)) / 100.0,
                   0.40, 1.0)

# 写实底模专用的 Lab 迁移系数（底模自带真实光影，系数整体比 fit_makeup 的
# 模板路径保守；底妆只"匀肤"，不提亮——全脸提亮=刷墙）。
# 烘焙时叠加 pigment-safe 自适应（见 _apply_locked）：kL 按底色/妆色明度差
# 收缩 + 色度保底不低于素颜，跨肤色还原一致。
PHOTOREAL_LAB = {
    "foundation": (0.08, 0.35),
    "concealer": (0.12, 0.40),
    "contour": (0.30, 0.55),
    "blush": (0.10, 1.45),
    "eyeshadow": (0.18, 1.30),
    "eyebrow": (0.55, 0.80),
    "eyeliner": (0.55, 0.80),
    "lashes": (0.55, 0.80),
    "lipstick": (0.60, 1.30),
    "highlight": (0.15, 0.20),
}


@dataclass
class UvMakeupMaps:
    """UV 空间妆容目标场。albedo 是"妆后目标色"，w 是覆盖（0=素颜）。

    kL/chroma 逐 texel Lab 迁移系数（按烘焙区域写入）——明度只部分跟随、
    色度推向妆色，皮肤纹理与原生光影保留。"""
    tex: int
    albedo: np.ndarray              # (t,t,3) float 0..1
    w: np.ndarray                   # (t,t) float 0..1（非唇区域）
    kL: np.ndarray                  # (t,t) 明度跟随系数
    chroma: np.ndarray              # (t,t) 色度推向系数
    channels: dict[str, np.ndarray] = field(default_factory=dict)  # rough/coat/sss/sheen
    lip_w: np.ndarray | None = None     # (t,t) 唇妆 UV 兜底通道（3D 路径优先）
    lip_cent: np.ndarray | None = None  # (t,t) 唇带渐变坐标（兜底色带采样）
    lip_albedo: np.ndarray | None = None  # (t,t,3) 唇妆目标色（兜底用，非黑）
    lip_stops: list | None = None       # 唇色带（3D 路径 ramp 用）
    lip_opacity: float = 0.85
    lip_finish: str = "gloss"

    @staticmethod
    def empty(tex: int = 2048) -> "UvMakeupMaps":
        return UvMakeupMaps(
            tex=tex,
            albedo=np.zeros((tex, tex, 3), np.float32),
            w=np.zeros((tex, tex), np.float32),
            kL=np.zeros((tex, tex), np.float32),
            chroma=np.zeros((tex, tex), np.float32),
            channels={k: np.zeros((tex, tex), np.float32) for k in MAP_KEYS})


def _value_noise(tex: int, cell: int, rng: np.random.Generator,
                 octaves: int = 3, gain: float = 0.5) -> np.ndarray:
    """多倍频 value noise (tex,tex) 0..1（唇纹/粉感的微观调制源）。"""
    out = np.zeros((tex, tex), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        g = max(2, int(cell / (2 ** o)))
        grid = rng.random((g, g)).astype(np.float32)
        layer = cv2.resize(grid, (tex, tex), interpolation=cv2.INTER_LINEAR)
        out += amp * layer
        total += amp
        amp *= gain
    return out / total


# ---------------- 高频区域的地标索引（FaceMesh canonical 468 序） ----------------
# 眼线/眉与唇同属高频小区域：canonical 模板带 ~2% 系统错位在唇上不可接受
# （唇已走 3D 锚定），在 2-3mm 宽的眼线上同样肉眼可见。这些折线锚定三角化
# 的真实地标，把"涂在哪"从模板改为观测。折线顺序 = 内→外（渐变坐标方向）。
EYE_INNER, EYE_OUTER = {"left": 133, "right": 362}, {"left": 33, "right": 263}
EYE_UPPER = {   # 内眼角 → 外眼角（上睑缘）
    "left": (133, 173, 157, 158, 159, 160, 161, 246, 33),
    "right": (362, 398, 384, 385, 386, 387, 388, 466, 263),
}
EYE_LOWER = {   # 内眼角 → 外眼角（下睑缘，睫毛/卧蚕参考）
    "left": (133, 155, 154, 153, 145, 144, 163, 7, 33),
    "right": (362, 382, 381, 380, 374, 373, 390, 249, 263),
}
BROW_PTS = {    # 眉头 → 眉尾（5 点折线，Catmull-Rom 重采样）
    "left": (107, 66, 105, 63, 70),
    "right": (300, 293, 334, 296, 336),
}


def _catmull_rom(P: np.ndarray, k: int) -> np.ndarray:
    """折线 Catmull-Rom 重采样 (m,3) → (k,3)（端点复制，C1 连续）。"""
    P = np.asarray(P, np.float64)
    if len(P) < 3:
        t = np.linspace(0, 1, k)[:, None]
        return P[0] * (1 - t) + P[-1] * t
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    seg = np.arange(len(P) - 1)
    t = np.linspace(0, 1, k, endpoint=False)
    s = t * (len(P) - 1)
    i = np.clip(seg.searchsorted(s, "right") - 1, 0, len(seg) - 1)
    u = (s - i)[:, None]
    p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
    return (0.5 * ((2 * p1) + (-p0 + p2) * u
                   + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                   + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))


def _seg_dist_cent(xyz: np.ndarray, a: np.ndarray, b: np.ndarray):
    """点到线段：返回 (dist (n,), t (n,) 线段内参数 0..1)。"""
    ab = b - a
    denom = float(ab @ ab) + 1e-12
    t = np.clip((xyz - a) @ ab / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(xyz - proj, axis=1), t


class UvMakeupBaker:
    """把妆容 spec 合成为 UV 目标场。复用内核 RegionMasks 与唇拓扑，不另起炉灶。"""

    def __init__(self, tex: int = 2048, seed: int = 7):
        self.tex = tex
        self.rng = np.random.default_rng(seed)
        self._fitter: FaceMakeupFitter | None = None

    @property
    def fitter(self) -> FaceMakeupFitter:
        if self._fitter is None:
            self._fitter = FaceMakeupFitter(tracker_factory=lambda: (_ for _ in ()).throw(
                RuntimeError("UvMakeupBaker 不需要 tracker")))
        return self._fitter

    # ---------- 唇红带 3D 拓扑 → UV 光栅化 ----------

    def raster_lip_band(self) -> tuple[np.ndarray, np.ndarray]:
        """唇红带 (cov, cent) 光栅化到 UV。来自 canonical 网格的带三角 +
        顶点到两环距离插值（外圈软、唇线侧饱满，与 3D 路径同羽化语义）。"""
        topo = self.fitter.lip_topology()
        V, tris, uvs = (self.fitter.model.base, self.fitter.model.tris,
                        self.fitter.model.uvs)
        band = topo["band"]
        t = self.tex
        cov = np.zeros((t, t), np.float32)
        cent = np.zeros((t, t), np.float32)
        d_out_all, d_in_all = topo["d_out"], topo["d_in"]
        for tri in band:
            i0, i1, i2 = (int(x) for x in tris[tri])
            p = np.stack([uvs[i0], uvs[i1], uvs[i2]])          # (3,2) uv
            p[:, 1] = 1.0 - p[:, 1]                            # v → 图像行
            do = np.array([d_out_all[i0], d_out_all[i1], d_out_all[i2]])
            di = np.array([d_in_all[i0], d_in_all[i1], d_in_all[i2]])
            mn = np.floor(p.min(0) * (t - 1)).astype(int) - 1
            mx = np.ceil(p.max(0) * (t - 1)).astype(int) + 1
            x0, y0 = max(mn[0], 0), max(mn[1], 0)
            x1, y1 = min(mx[0], t - 1), min(mx[1], t - 1)
            if x1 <= x0 or y1 <= y0:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            P = np.stack([xs, ys], -1).astype(np.float64) / (t - 1)
            a, b, c = p
            det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(det) < 1e-14:
                continue
            l0 = ((b[1] - c[1]) * (P[..., 0] - c[0])
                  + (c[0] - b[0]) * (P[..., 1] - c[1])) / det
            l1 = ((c[1] - a[1]) * (P[..., 0] - c[0])
                  + (a[0] - c[0]) * (P[..., 1] - c[1])) / det
            l2 = 1.0 - l0 - l1
            inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            if not inside.any():
                continue
            do_p = l0 * do[0] + l1 * do[1] + l2 * do[2]
            di_p = l0 * di[0] + l1 * di[1] + l2 * di[2]
            loc = 0.5 * (do_p + di_p) + 1e-6
            # 口红涂满到嘴线：只保留外圈软羽化（口腔面已由拓扑排除，无需内羽化）
            edge = np.clip(do_p / (0.28 * loc), 0, 1)
            cen_p = np.clip(di_p / (do_p + di_p + 1e-6), 0, 1)
            sl_c, sl_e = cov[y0:y1 + 1, x0:x1 + 1], cent[y0:y1 + 1, x0:x1 + 1]
            take = inside & (edge > sl_c)
            sl_c[take] = edge[take]
            sl_e[take] = cen_p[take]
        # 唇缘平滑：粗模板三角的星形锯齿 → 高斯柔化（不改变整体形状）
        cov = cv2.GaussianBlur(cov, (0, 0), 2.5)
        cent = cv2.GaussianBlur(cent, (0, 0), 2.5)
        return cov, cent

    # ---------- 主入口：spec → UV 目标场 ----------

    def bake(self, layers: list[dict], intensity: float = 0.8) -> UvMakeupMaps:
        core = _load_core()
        prev = core.TEX
        core.set_texture_size(self.tex)
        maps = UvMakeupMaps.empty(self.tex)
        try:
            noise_fine = _value_noise(self.tex, 96, self.rng, octaves=3)
            noise_powder = _value_noise(self.tex, 160, self.rng, octaves=2)
            # 闪片用单倍频高频噪声：多倍频平均把分布压向 0.5，阈值永不触发
            noise_flake = _value_noise(self.tex, 384, self.rng, octaves=1)
            for layer in layers:
                if not layer.get("enabled", True):
                    continue
                region = layer["region"]
                opacity = float(layer.get("opacity", 0.7)) * float(intensity)
                if opacity <= 1e-3:
                    continue
                finish = layer.get("finish", "satin")
                sparkle = float(layer.get("sparkle",
                                          1.0 if finish == "glitter" else 0.0))
                stops = layer.get("color_stops") or [
                    {"at": 0.0, "hex": "#B03040"}, {"at": 1.0, "hex": "#D04858"}]
                shape = layer.get("shape") or {}
                if region == "lipstick":
                    cov, cent = self.raster_lip_band()
                    peak = float(cov.max())
                    if peak > 1e-4:
                        cov = cov / peak
                    maps.lip_w = cov * opacity           # 独立通道：3D 路径的兜底
                    maps.lip_stops = stops
                    maps.lip_opacity = opacity
                    maps.lip_finish = finish
                else:
                    m = self.fitter.regions.bake({"region": region, "shape": shape})
                    cov, cent = m[..., 0], m[..., 1]
                    peak = float(cov.max())
                    if peak < 1e-4:
                        continue
                    cov = np.clip(cov / peak, 0, 1)
                w = np.clip(cov * opacity, 0, 1)[..., None]
                col = np.clip(core.sample_ramp(
                    stops, np.clip(cent, 0, 1)), 0, 1).astype(np.float32)
                rough_t, coat_t, sheen_t = FINISH_TARGET.get(finish, FINISH_TARGET["satin"])
                sss_t = 1.0 if region == "lipstick" else 0.0
                micro = (noise_fine if region in ("lipstick", "eyeliner", "eyebrow")
                         else noise_powder)

                # 妆后目标 albedo：色带 × 微观明度调制（唇纹 ±4% / 粉感 ±2.5%）
                if region == "lipstick":
                    micro_col = 1.0 + (micro[..., None] - 0.5) * 0.08
                else:
                    micro_col = 1.0 + (micro[..., None] - 0.5) * 0.05
                tgt = np.clip(col * micro_col, 0, 1)
                if region == "lipstick":
                    # 兜底目标场：3D 锚定缺席时 UV 唇带有真实颜色可用
                    # （此前兜底目标是全零 = 涂黑，唇带失配时唇色发黑）
                    maps.lip_albedo = tgt
                    maps.lip_cent = cent.astype(np.float32)

                # 目标场合成：后层不覆盖先层（w 大者胜），albedo 同步。
                # 唇层不进全局场（唇带与真实唇有 ~2% 错位，走 3D 锚定路径）
                if region != "lipstick":
                    upd = (w[..., 0] > maps.w)
                    maps.albedo[upd] = tgt[upd]
                    maps.w = np.maximum(maps.w, w[..., 0])
                    kL_t, chroma_t = PHOTOREAL_LAB.get(region, (0.25, 1.0))
                    maps.kL[upd] = kL_t
                    maps.chroma[upd] = chroma_t
                    # 微观粗糙度：粉状加橘皮（唇釉材质由 3D 路径在 apply 时覆盖）
                    ch = maps.channels
                    ch["rough"][upd] = (rough_t + (micro[upd] - 0.5)
                                        * (0.05 if finish == "matte" else 0.03))
                    ch["coat"][upd] = coat_t * (0.75 + 0.5 * micro[upd])
                    if sparkle > 0.01:
                        # 珠光闪点：sheen 不再是均匀标量——高频阈值噪声造"密
                        # 底 + 稀疏亮片"，渲染端掠射绒光由此空间变化
                        flake = np.clip((noise_flake - 0.72) * 5.0, 0, 1)
                        ch["sheen"][upd] = np.clip(
                            sheen_t * (0.45 + 2.2 * sparkle * flake[upd]), 0, 1)
                    else:
                        ch["sheen"][upd] = sheen_t
                    ch["sss"][upd] = sss_t
        finally:
            core.set_texture_size(prev)
        return maps

    # ---------- guidance 聚合（P1/R1.1：多视角 guidance → UV albedo） ----------

    def bake_guidance(self, maps: UvMakeupMaps, views: list[dict],
                      model_pts: tuple[np.ndarray, np.ndarray]) -> UvMakeupMaps:
        """多视角 guidance 图 → UV albedo 逐 texel 中位数（AvatarMakeup 的
        "全局 UV map"聚合）。views: [{"R","t","cam","img"(BGR)}]；
        model_pts: (S_xyz (m,3) 世界系表面采样, S_uv (m,2))。
        采到 ≥1 视的 texel 用 guidance 色替换参数化目标（w 保持）。"""
        S_xyz, S_uv = model_pts
        m = len(S_xyz)
        samples = []
        for v in views:
            Rv, tv = np.asarray(v["R"], np.float64), np.asarray(v["t"], np.float64)
            cam = v["cam"]
            cam_xyz = (S_xyz @ Rv.T) + tv
            z = cam_xyz[:, 2]
            f = float(cam.params[0])
            px = cam.params[1] + cam_xyz[:, 0] * f / np.maximum(z, 1e-6)
            py = cam.params[2] + cam_xyz[:, 1] * f / np.maximum(z, 1e-6)
            img = np.asarray(v["img"], np.float32) / 255.0
            h, w_ = img.shape[:2]
            ok = (z > 0.05) & (px >= 0) & (px < w_ - 1) & (py >= 0) & (py < h - 1)
            x0 = np.clip(px, 0, w_ - 1.001).astype(np.int32)
            y0 = np.clip(py, 0, h - 1.001).astype(np.int32)
            fx, fy = (px - x0)[..., None], (py - y0)[..., None]
            rgb = (img[y0, x0, ::-1] * (1 - fx) * (1 - fy)
                   + img[y0, x0 + 1, ::-1] * fx * (1 - fy)
                   + img[y0 + 1, x0, ::-1] * (1 - fx) * fy
                   + img[y0 + 1, x0 + 1, ::-1] * fx * fy)
            samples.append(np.where(ok[:, None], rgb, np.nan).astype(np.float32))
        stack = np.stack(samples) if samples else np.full((1, m, 3), np.nan, np.float32)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(stack, axis=0)
        have = np.isfinite(med).all(1)
        t = maps.tex
        tx = np.clip(S_uv[:, 0] * (t - 1), 0, t - 1.001).astype(np.int32)
        ty = np.clip((1 - S_uv[:, 1]) * (t - 1), 0, t - 1.001).astype(np.int32)
        upd = have & (maps.w[ty, tx] > 0.02)
        maps.albedo[ty[upd], tx[upd]] = med[upd]
        return maps

    # ---------- 唇妆 3D 锚定 ----------

    def lip_band_3d(self, cloud: dict[str, np.ndarray],
                    landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """三角化真实地标构建唇红带 → 逐 splat (权重, 渐变坐标)。

        复用 fit_makeup L0 的观测唇域锚定：Vw 直接取真实地标顶点（非 Kabsch
        模板），颜色门控把权重收进真实唇内——canonical 模板唇带 ~2% 的系统性
        错位在唇这种小高频区域不可接受（UV 唇带因此只做兜底）。"""
        fitter = self.fitter
        s, R, t, _ = fitter.register(landmarks)
        cloud2 = dict(cloud)
        cloud2["uv"] = np.zeros((len(cloud["xyz"]), 2))   # 触发 _world_lip_mesh 地标路径
        w, cent = fitter._lip_band_weight(cloud2, landmarks, s, R, t)
        return w, cent

    # ---------- 眼线/睫毛/眉 3D 地标锚定 ----------

    def landmark_band_3d(self, cloud: dict[str, np.ndarray],
                         landmarks: np.ndarray, region: str,
                         shape: dict | None = None,
                         strength: float = 1.0,
                         ) -> tuple[np.ndarray, np.ndarray]:
        """三角化真实地标 → 眼线/睫毛/眉的逐 splat (权重, 渐变坐标)。

        与唇的 3D 锚定同一动机：canonical 模板在这些 2-5mm 宽的高频区域有
        ~2% 脸高的系统错位，模板 UV 蒙版（bind_uv 路径）只能当兜底。折线
        沿观测地标构建，宽度/眼线拉长由 layer.shape 参数控制；权重随离折线
        距离 smoothstep 衰减，cent 沿折线弧长 0(内)→1(外) 驱动色带。
        landmark 无效（三角化失败）时返回零权重 → 回退 UV 模板蒙版。"""
        shape = shape or {}
        xyz = np.asarray(cloud["xyz"], np.float64)
        lm = np.asarray(landmarks, np.float64)
        n = len(xyz)
        w_all = np.zeros(n, np.float32)
        cent_all = np.zeros(n, np.float32)

        if region not in ("eyeliner", "lashes", "eyebrow"):
            raise ValueError(f"landmark_band_3d 不支持区域 {region}")
        if len(lm) < 400:
            return w_all, cent_all                  # 地标不完整：回退 UV
        need = list(EYE_UPPER["left"]) + list(BROW_PTS["right"])
        if np.linalg.norm(lm[[i for i in need if i < len(lm)]], axis=1).min() <= 0:
            return w_all, cent_all                  # 关键地标缺失：回退 UV
        dip = float(np.linalg.norm(lm[EYE_OUTER["left"]] - lm[EYE_OUTER["right"]]))
        if dip < 1e-6:
            return w_all, cent_all                  # 脸尺度不可用：回退 UV

        thickness = float(shape.get("thickness", 0.5))
        polys: list[tuple[np.ndarray, float, np.ndarray]] = []   # (折线, 半宽, 升起向量)
        if region in ("eyeliner", "lashes"):
            half = dip * (0.010 if region == "lashes" else 0.0065)
            half *= 0.7 + 0.6 * float(shape.get("thickness", 0.5))
            for side in ("left", "right"):
                P = lm[list(EYE_UPPER[side])]
                inner, outer = lm[EYE_INNER[side]], lm[EYE_OUTER[side]]
                # 睫毛：沿"眼窝中心→睑缘"方向外推少许（长在睑缘外侧）
                if region == "lashes":
                    lower_mid = lm[list(EYE_LOWER[side])].mean(0)
                    ups = P - lower_mid
                    ups /= np.linalg.norm(ups, axis=1, keepdims=True) + 1e-12
                    P = P + ups * (dip * 0.006)
                # 眼线拉长（wing）：外眼角沿 (外-内) 方向延伸
                wing = float(shape.get("wing", 0.0)) if region == "eyeliner" else 0.0
                if wing > 0.01:
                    d_out = outer - inner
                    d_out /= np.linalg.norm(d_out) + 1e-12
                    P = np.vstack([P, P[-1] + d_out * (wing * dip * 0.10)])
                polys.append((_catmull_rom(P, 24), half, np.zeros(3)))
        else:                                        # eyebrow
            half = dip * (0.008 + 0.026 * thickness)
            for side in ("left", "right"):
                polys.append((_catmull_rom(lm[list(BROW_PTS[side])], 24), half,
                              np.zeros(3)))

        for P, half, _lift in polys:
            best_d = np.full(n, np.inf)
            best_c = np.zeros(n)
            for si in range(len(P) - 1):
                d, t = _seg_dist_cent(xyz, P[si], P[si + 1])
                c = (si + t) / max(len(P) - 1, 1)
                take = d < best_d
                best_d[take] = d[take]
                best_c[take] = c[take]
            s = np.clip(1.0 - best_d / max(half, 1e-9), 0.0, 1.0)
            s = s * s * (3 - 2 * s)                  # smoothstep：带缘无锯齿
            take = s > w_all
            w_all[take] = s[take]
            cent_all[take] = best_c[take]
        return (w_all * float(np.clip(strength, 0.0, 1.4))).astype(np.float32), \
            cent_all.astype(np.float32)

    # ---------- 烘焙回 splat ----------

    def apply_to_cloud(self, cloud: dict[str, np.ndarray], maps: UvMakeupMaps,
                       uv: np.ndarray, valid: np.ndarray,
                       treatment: str = "lab",
                       lip3d: tuple[np.ndarray, np.ndarray] | None = None,
                       intensity: float = 0.8,
                       near: np.ndarray | None = None,
                       bands3d: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                       ) -> dict[str, np.ndarray]:
        """UV 目标场 → per-splat 颜色（Lab 部分迁移，保纹理）+ 材质通道。

        lip3d：(w (n,), cent (n,)) —— 由 `lip_band_3d` 用三角化真实地标计算的
        唇带权重（唇带与 canonical 模板有 ~2% 系统错位，唇妆必须 3D 锚定；
        缺省时回退 UV 唇带兜底通道）。intensity 与 bake 时同源。
        near：(n,) splat 到 canonical 表面的距离（bind_uv 产物）——边界软门控，
        UV 归属在鼻翼/眼角/轮廓处的误差按距离衰减妆权重，收掉边界晕。
        bands3d：{"eyeliner"/"lashes"/"eyebrow": (w, cent)} —— `landmark_band_3d`
        的观测锚定带，覆盖同区域 UV 模板权重（颜色仍取 UV 场，带只改"涂在哪"）。"""
        # —— _bilinear 内部按全局 TEX 钳制坐标，采样 2048 图前必须切全局尺寸 ——
        core = _load_core()
        prev_tex = core.TEX
        core.set_texture_size(maps.tex)
        try:
            out = self._apply_locked(cloud, maps, uv, valid, treatment, core,
                                     lip3d, near, bands3d)
        finally:
            core.set_texture_size(prev_tex)
        return out

    @staticmethod
    def _base_micro_hp(cloud: dict[str, np.ndarray], t: int,
                       tx: np.ndarray, ty: np.ndarray, valid: np.ndarray):
        """底模 albedo 的 UV 空间亮度高通（真实唇纹/毛孔/皮肤起伏的第一来源）。

        底模颜色 splat 进 UV 网格 → 加权模糊空洞填补 → 亮度减低频 → 归一化。
        返回逐 splat (n,) 的 ±1 高通值；覆盖不足时返回 None（退回程序噪声）。"""
        w = (np.asarray(cloud["rgba"], np.float64)[:, 3]
             * np.asarray(valid, np.float64)).astype(np.float32)
        if float(w.sum()) < 1.0:
            return None
        rgb = np.clip(np.asarray(cloud["rgba"], np.float64)[:, :3], 0, 1)
        ti, tyi = tx.astype(np.int32), ty.astype(np.int32)
        wmap = np.zeros((t, t), np.float32)
        amap = np.zeros((t, t, 3), np.float32)
        np.add.at(wmap, (tyi, ti), w)
        np.add.at(amap, (tyi, ti), (rgb * w[:, None]).astype(np.float32))
        wblur = cv2.GaussianBlur(wmap, (0, 0), 3)
        amblur = cv2.GaussianBlur(amap, (0, 0), 3)
        cov = wblur > 1e-4
        if cov.mean() < 0.02:
            return None
        alb = np.where(cov[..., None],
                       amblur / np.maximum(wblur, 1e-6)[..., None], 0.0)
        lum = alb @ np.array([0.299, 0.587, 0.114], np.float32)
        # 覆盖洞先扩散填补（洞缘是"0 对比邻域"的假高通，幅度可达真实纹理 10 倍）
        filled = np.where(cov, lum, 0.0).astype(np.float32)
        wf = cov.astype(np.float32)
        for _ in range(4):
            lb = cv2.GaussianBlur(filled, (0, 0), 3)
            wb = cv2.GaussianBlur(wf, (0, 0), 3)
            holes = (wf < 0.5) & (wb > 1e-3)
            if not holes.any():
                break
            filled[holes] = lb[holes] / wb[holes]
            wf[holes] = 0.5
        hp = filled - cv2.GaussianBlur(filled, (0, 0), 4)
        # 采样密度接近峰值的内部区才统计（稀疏边缘是部分采样伪影）
        core = wf > 0.5
        hp[~core] = 0.0
        if not core.any():
            return None
        scale = float(np.percentile(np.abs(hp[core]), 99))
        if scale < 1e-3:
            return None
        hp = np.clip(hp / scale, -1.0, 1.0)
        return hp[tyi, ti]

    def _assignment(self, cloud, maps, uv, valid, core, lip3d, near, bands3d) -> dict:
        """逐 splat 妆容指派：权重（全部门控后）+ 目标色 + 材质通道。

        原位重染色（_apply_locked）与独立壳层（build_makeup_layer）共用同一
        指派——"涂在哪 / 涂什么色 / 什么材质"两条路径永远一致。返回里
        *_raw 是 UV 通道的原始采样（原位路径按 w 混肤），rough/coat/sss/
        sheen 是全强度目标（壳层路径直接用，含 0 值回退皮肤基准）。"""
        t = maps.tex
        tx = np.clip(uv[:, 0] * (t - 1), 0, t - 1.001)
        ty = np.clip((1 - uv[:, 1]) * (t - 1), 0, t - 1.001)

        def smp(ch: np.ndarray) -> np.ndarray:
            return core._bilinear(ch[..., None], tx, ty)[..., 0]

        lip_zone = None
        w = smp(maps.w).astype(np.float32)
        w[~valid] = 0.0
        # near 软门控：对全部妆权重生效（含唇兜底与 3D 锚定带）——离 canonical
        # 表面远的 splat（UV 归属误差/头发/背景）不因任何路径漏涂
        gate = np.ones(len(w), np.float32)
        if near is not None:
            nv = np.asarray(near, np.float64)[valid]
            if len(nv) and float(np.median(nv)) > 1e-9:
                med = float(np.median(nv))
                gate = np.exp(-0.5 * np.maximum(
                    np.asarray(near, np.float64) - med, 0.0) ** 2
                    / (1.8 * med) ** 2).astype(np.float32)
                gate[~valid] = 0.0
                w = w * gate
        tgt = np.stack([core._bilinear(maps.albedo[..., k][..., None], tx, ty)[..., 0]
                        for k in range(3)], axis=1)
        kL = smp(maps.kL).astype(np.float64)
        chroma = smp(maps.chroma).astype(np.float64)

        # 底模真实微观纹理（唇纹/毛孔高通）：调制妆后 albedo 对比与粗糙度。
        # 覆盖不足时 hp=None，保持 bake 阶段的程序噪声调制不变。
        hp = self._base_micro_hp(cloud, t, tx, ty, valid)
        if hp is not None:
            w_micro = w[..., None]
            tgt = np.clip(tgt * (1.0 + hp[:, None] * 0.06) * w_micro
                          + tgt * (1.0 - w_micro), 0, 1)

        # ---- 眼线/睫毛/眉：3D 地标锚定带（带只改权重，颜色仍取 UV 场） ----
        if bands3d:
            for _region, (bw, _bc) in bands3d.items():
                bw = np.asarray(bw, np.float32) * gate
                bw[~valid] = 0.0
                w = np.maximum(w, bw)

        # ---- 唇妆：3D 锚定优先，UV 兜底 ----
        if maps.lip_w is not None:
            lip_uv = smp(maps.lip_w).astype(np.float32)
            lip_uv[~valid] = 0.0
            lip_kL, lip_ch = PHOTOREAL_LAB["lipstick"]
            rough_t, coat_t, sheen_t = FINISH_TARGET.get(maps.lip_finish,
                                                         FINISH_TARGET["gloss"])
            if lip3d is not None and float(np.max(lip3d[0])) > 0.02:
                w_lip = np.clip(lip3d[0] * 1.4 * maps.lip_opacity, 0, 1
                                ).astype(np.float32) * gate
                cent = np.clip(lip3d[1], 0, 1)
                core2 = _load_core()
                lip_tgt = np.clip(core2.sample_ramp(maps.lip_stops, cent), 0, 1)
            else:                                   # 兜底：UV 唇带（带真实目标色）
                w_lip = lip_uv * gate
                if maps.lip_albedo is not None:
                    if maps.lip_cent is not None:
                        cent = smp(maps.lip_cent).astype(np.float32)
                    else:
                        cent = np.zeros(len(w), np.float32)
                    lip_tgt = np.stack(
                        [core._bilinear(maps.lip_albedo[..., k][..., None], tx, ty)[..., 0]
                         for k in range(3)], axis=1)
                else:
                    cent = np.zeros(len(w), np.float32)
                    lip_tgt = np.zeros((len(w), 3), np.float32)
            zone = w_lip > 0.02
            w = np.where(zone, w_lip, w)
            if zone.any():
                tgt = np.where(zone[:, None], lip_tgt, tgt)
                kL = np.where(zone, lip_kL, kL)
                chroma = np.where(zone, lip_ch, chroma)
                lip_zone = zone            # 材质通道在唇区显式覆盖（见下）

        cur = np.clip(np.asarray(cloud["rgba"], np.float64)[:, :3], 0, 1)
        rough_raw = smp(maps.channels["rough"]).astype(np.float32)
        coat_raw = smp(maps.channels["coat"]).astype(np.float32)
        sss_raw = smp(maps.channels["sss"]).astype(np.float32)
        sheen_raw = smp(maps.channels["sheen"]).astype(np.float32)
        no_lip = ~(lip_zone if lip_zone is not None else np.zeros(len(w), bool))
        # 壳层的全强度材质目标：0 值（UV 通道未写入的边界 texel）回退皮肤
        # 基准，避免边界 splat 变成 rough=0 的超镜面
        rough_f = np.where(rough_raw > 0.02, rough_raw, SKIN_ROUGH)
        coat_f = np.where(coat_raw > 0.005, coat_raw, 0.06)
        if hp is not None:
            rough_f = np.where((w > 0.02) & no_lip,
                               np.clip(rough_f + hp * 0.10, 0.05, 0.95), rough_f)
        if lip_zone is not None and lip_zone.any():
            rough_t, coat_t, sheen_t = FINISH_TARGET.get(
                maps.lip_finish, FINISH_TARGET["gloss"])
            rough_f = np.where(lip_zone, rough_t, rough_f)
            coat_f = np.where(lip_zone, coat_t, coat_f)
            sss_f = np.where(lip_zone, 1.0, sss_raw)
            sheen_f = np.where(lip_zone, sheen_t, sheen_raw)
        else:
            sss_f, sheen_f = sss_raw, sheen_raw
        return {"tx": tx, "ty": ty, "w": w, "gate": gate, "tgt": tgt,
                "kL": kL, "chroma": chroma, "cur": cur, "hp": hp,
                "lip_zone": lip_zone,
                "rough_raw": rough_raw, "coat_raw": coat_raw,
                "sss_raw": sss_raw, "sheen_raw": sheen_raw,
                "rough": rough_f.astype(np.float32), "coat": coat_f.astype(np.float32),
                "sss": sss_f.astype(np.float32), "sheen": sheen_f.astype(np.float32)}

    @staticmethod
    def _lab_migrate(cur: np.ndarray, tgt: np.ndarray, kL: np.ndarray,
                     chroma: np.ndarray, weight: np.ndarray,
                     full: bool = False) -> np.ndarray:
        """float Lab + pigment-safe 极坐标迁移。

        full=False（原位重染色）：weight=w 的折叠迁移，与既有行为逐位一致；
        full=True（壳层颜色）：kL/chroma 不乘 w 的完整迁移——壳与皮肤的混合
        交给 alpha 合成，物理层完成"部分迁移"的语义。色相旋转权重封顶 1
        （向量插值系数>1 会穿过 a-b 零点把红唇翻成绿调）；chroma 幅值可 >1
        外推且保底不低于素颜（透色染料：只加色不褪色）。"""
        cur_lab = rgb2lab(cur)
        tgt_lab = rgb2lab(tgt)
        kL_w = kL if full else kL * weight
        adapt = lab_adapt(cur_lab[:, 0], tgt_lab[:, 0])
        new_L = cur_lab[:, 0] + (tgt_lab[:, 0] - cur_lab[:, 0]) * kL_w * adapt
        a_cur, b_cur = cur_lab[:, 1], cur_lab[:, 2]
        a_tgt, b_tgt = tgt_lab[:, 1], tgt_lab[:, 2]
        c_cur = np.hypot(a_cur, b_cur)
        c_tgt = np.hypot(a_tgt, b_tgt)
        h_cur = np.arctan2(b_cur, a_cur)
        h_tgt = np.arctan2(b_tgt, a_tgt)
        dh = np.mod(h_tgt - h_cur + np.pi, 2 * np.pi) - np.pi
        k_ch = np.clip(chroma if full else chroma * weight, 0.0, 1.0)
        h_new = h_cur + dh * k_ch
        c_new = c_cur + (c_tgt - c_cur) * (chroma if full else chroma * weight)
        c_new = np.maximum(c_new, c_cur)              # 色度保底
        new_ab = np.stack([c_new * np.cos(h_new), c_new * np.sin(h_new)], axis=1)
        return lab2rgb(np.stack(
            [new_L, new_ab[:, 0], new_ab[:, 1]], axis=1)).astype(np.float32)

    def _apply_locked(self, cloud, maps, uv, valid, treatment, core, lip3d,
                      near, bands3d) -> dict:
        a = self._assignment(cloud, maps, uv, valid, core, lip3d, near, bands3d)
        w, tgt, cur = a["w"], a["tgt"], a["cur"]
        out = {}
        for k, v in cloud.items():
            out[k] = v.copy() if isinstance(v, np.ndarray) else v

        if treatment == "lab":
            back = self._lab_migrate(cur, tgt, a["kL"], a["chroma"], w)
        else:                                   # alpha-over（合成脸校准）
            back = cur * (1 - w[..., None]) + tgt * w[..., None]
        out["rgba"][:, :3] = back

        from .pbr import Material
        lip_zone, hp = a["lip_zone"], a["hp"]
        rough = SKIN_ROUGH * (1 - w) + a["rough_raw"] * w
        coat = np.maximum(a["coat_raw"] * w, 0.06 * (1 - w))
        sss = a["sss_raw"] * w
        sheen = a["sheen_raw"] * w
        if hp is not None:
            # 真实唇纹/粉感 → 粗糙度微起伏（凸处更糙/凹处更光），唇区随后整体覆盖
            rough = np.where((w > 0.02) & ~(lip_zone if lip_zone is not None
                                           else np.zeros(len(w), bool)),
                             np.clip(rough + hp * 0.10 * w, 0.05, 0.95), rough)
        if lip_zone is not None and lip_zone.any():
            rough_t, coat_t, sheen_t = FINISH_TARGET.get(
                maps.lip_finish, FINISH_TARGET["gloss"])
            rough = np.where(lip_zone, rough_t, rough)
            coat = np.where(lip_zone, coat_t, coat)
            sss = np.where(lip_zone, 1.0, sss)
            sheen = np.where(lip_zone, sheen_t, sheen)
        out["material"] = Material(rough.astype(np.float32), coat.astype(np.float32),
                                   sss.astype(np.float32), sheen.astype(np.float32))
        # 旧渲染路径（fit_makeup.render_cloud / 旧 Unity 端）兼容字段
        from .pbr import rough_to_shin
        out["gloss"] = coat
        out["shin"] = rough_to_shin(rough).astype(np.float32)
        out["makeup_w"] = w
        return out

    def build_makeup_layer(self, cloud: dict[str, np.ndarray], maps: UvMakeupMaps,
                           uv: np.ndarray, valid: np.ndarray,
                           treatment: str = "lab",
                           lip3d: tuple[np.ndarray, np.ndarray] | None = None,
                           near: np.ndarray | None = None,
                           bands3d: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                           min_w: float = 0.05, scale_ratio: float = 1.0,
                           opacity_gain: float = 1.0,
                           normal_offset: float = 0.15,
                           hp_gain: float = 0.10,
                           ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """妆容壳层 = 真实高斯球：逐个对应底模 splat 生成一层重叠新高斯。

        与原位重染色（apply_to_cloud）的本质区别：妆在**几何上存在**——
        每个 w > min_w 的底模 splat 生成一个新高斯：
            xyz       底模 splat 沿外法线偏移 normal_offset×min(scale)——
                      薄层厚度（化妆品 ~10-40μm 的 splat 尺度对应物），同时
                      消除同深度 z-tie（完全共心的两片高斯深度排序不稳定，
                      强妆色差会在视频里闪）；法线 = min(scale) 薄轴 + 径向
                      定向 + kNN 平滑，偏移量在 splat 自身足迹内不外凸；
            rot/scale 与底模一致（同足迹，不改表面形状）；
            颜色      完整 Lab 迁移目标（kL/chroma 不乘 w——"部分迁移"由
                      alpha 合成完成，物理层即薄层化妆品）× 底模高通纹理
                      （hp_gain：训练出来的唇纹/毛孔透过妆层可见——UV 目标
                      场只是程序噪声，真实微观质感的第一来源是底模本身）；
            opacity   w × 底模 opacity × opacity_gain（壳层堆叠比例与底模
                      一致，妆强 ≈ w）；
            sh_rest   置零（妆色视角无关，不被底模素颜 SH 残差调制；视角
                      相关高光由渲染端 material AOV 合成补回）；
            material  finish 全强度（唇釉清漆/sss/珠光直接挂在壳层上）。
        返回 (layer_cloud, src_idx)；`merge_makeup_layer` 合并进底模导出。"""
        core = _load_core()
        prev_tex = core.TEX
        core.set_texture_size(maps.tex)
        try:
            a = self._assignment(cloud, maps, uv, valid, core, lip3d, near, bands3d)
        finally:
            core.set_texture_size(prev_tex)

        w = a["w"]
        idx = np.nonzero(w > float(min_w))[0]
        if treatment == "lab":
            col = self._lab_migrate(a["cur"], a["tgt"], a["kL"], a["chroma"],
                                    w, full=True)
        else:
            col = a["tgt"]
        if a["hp"] is not None and hp_gain > 0:
            # 真实微观质感透出：底模 albedo 高通（唇纹/毛孔/皮肤起伏）直接
            # 调制壳层颜色，妆下纹理不再是程序噪声的平涂
            col = np.clip(col * (1.0 + a["hp"][:, None] * float(hp_gain)), 0, 1)
        base_op = np.clip(np.asarray(cloud["rgba"], np.float64)[:, 3], 0.0, 0.98)
        op = np.clip(w[idx] * base_op[idx] * float(opacity_gain), 1e-4, 0.98)
        layer = {
            "xyz": np.asarray(cloud["xyz"], np.float32)[idx].copy(),
            "scale": (np.asarray(cloud["scale"], np.float32)[idx]
                      * float(scale_ratio)).astype(np.float32),
            "rot": np.asarray(cloud["rot"], np.float32)[idx].copy(),
            "rgba": np.concatenate(
                [np.clip(col[idx], 0, 1).astype(np.float32), op[:, None]], 1),
            "makeup_w": w[idx].copy(),
        }
        if float(normal_offset) > 0:
            # 薄层厚度：沿外法线偏移。法线符号用径向定向（薄轴符号逐 splat
            # 任意，直接用会有一半壳层陷进表面），kNN 平滑保证邻域连续
            from .normals import axis_normals, smooth_normals
            xyz_all = np.asarray(cloud["xyz"], np.float64)
            n_all = axis_normals(np.asarray(cloud["rot"], np.float64),
                                 np.asarray(cloud["scale"], np.float64))
            radial = xyz_all - np.median(xyz_all, axis=0)
            flip = (n_all * radial).sum(1) < 0
            n_all[flip] *= -1.0
            n_all = smooth_normals(xyz_all.astype(np.float32), n_all, k=12, iters=2)
            thin = np.asarray(cloud["scale"], np.float64).min(axis=1)
            off = n_all[idx] * (float(normal_offset) * thin[idx])[:, None]
            layer["xyz"] = (layer["xyz"] + off.astype(np.float32))
        from .pbr import Material, rough_to_shin
        layer["material"] = Material(a["rough"][idx].copy(), a["coat"][idx].copy(),
                                     a["sss"][idx].copy(), a["sheen"][idx].copy())
        layer["gloss"] = a["coat"][idx].copy()
        layer["shin"] = rough_to_shin(a["rough"][idx]).astype(np.float32)
        sh = cloud.get("sh_rest")
        if sh is not None:
            layer["sh_rest"] = np.zeros((len(idx), np.asarray(sh).shape[1], 3),
                                        np.float32)
        return layer, idx


def merge_makeup_layer(base: dict[str, np.ndarray], layer: dict[str, np.ndarray],
                       src_idx: np.ndarray) -> dict[str, np.ndarray]:
    """底模 + 妆容壳层 → 单一可导出点云（madeup.ply/.splat 的最终形态）。

    壳层 splat 按构造与底模对应 splat 同位重叠；逐 splat 属性（uv/near 等
    绑定字段）用 src_idx 对应行延伸，保证换妆重绑自洽；壳层独有的标量场
    （makeup_w/gloss/shin）底模侧补零；材质对象逐通道拼接（底模无材质时
    补皮肤基准）。"""
    from .pbr import Material

    n_b = len(base["xyz"])
    n_l = len(layer["xyz"])
    out: dict[str, np.ndarray] = {}
    for k, v in base.items():
        lv = layer.get(k)
        if (isinstance(v, np.ndarray) and v.shape[:1] == (n_b,)
                and not isinstance(lv, Material)):
            piece = (lv if isinstance(lv, np.ndarray) and lv.shape[:1] == (n_l,)
                     else v[src_idx])
            try:
                out[k] = np.concatenate([v, piece], axis=0)
                continue
            except ValueError:
                pass
        out[k] = v
    for k, lv in layer.items():                    # 壳层独有标量场：底模侧补零
        if k in out or k == "material" or not isinstance(lv, np.ndarray):
            continue
        fill = np.zeros((n_b,) + lv.shape[1:], lv.dtype)
        out[k] = np.concatenate([fill, lv], axis=0)
    mb, ml = base.get("material"), layer.get("material")
    if ml is not None:
        if mb is None:
            mb = Material.skin(n_b)
        out["material"] = Material(
            np.concatenate([mb.rough, ml.rough]).astype(np.float32),
            np.concatenate([mb.coat, ml.coat]).astype(np.float32),
            np.concatenate([mb.sss, ml.sss]).astype(np.float32),
            np.concatenate([mb.sheen, ml.sheen]).astype(np.float32))
    return out
