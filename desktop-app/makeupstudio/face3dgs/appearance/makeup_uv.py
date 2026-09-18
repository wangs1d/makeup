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
}
SKIN_ROUGH = 0.52
MAP_KEYS = ("rough", "coat", "sss", "sheen")

# 写实底模专用的 Lab 迁移系数（底模自带真实光影，系数整体比 fit_makeup 的
# 模板路径保守；底妆只"匀肤"，不提亮——全脸提亮=刷墙）。
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
            for layer in layers:
                if not layer.get("enabled", True):
                    continue
                region = layer["region"]
                opacity = float(layer.get("opacity", 0.7)) * float(intensity)
                if opacity <= 1e-3:
                    continue
                finish = layer.get("finish", "satin")
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

    # ---------- 烘焙回 splat ----------

    def apply_to_cloud(self, cloud: dict[str, np.ndarray], maps: UvMakeupMaps,
                       uv: np.ndarray, valid: np.ndarray,
                       treatment: str = "lab",
                       lip3d: tuple[np.ndarray, np.ndarray] | None = None,
                       intensity: float = 0.8) -> dict[str, np.ndarray]:
        """UV 目标场 → per-splat 颜色（Lab 部分迁移，保纹理）+ 材质通道。

        lip3d：(w (n,), cent (n,)) —— 由 `lip_band_3d` 用三角化真实地标计算的
        唇带权重（唇带与 canonical 模板有 ~2% 系统错位，唇妆必须 3D 锚定；
        缺省时回退 UV 唇带兜底通道）。intensity 与 bake 时同源。"""
        # —— _bilinear 内部按全局 TEX 钳制坐标，采样 2048 图前必须切全局尺寸 ——
        core = _load_core()
        prev_tex = core.TEX
        core.set_texture_size(maps.tex)
        try:
            out = self._apply_locked(cloud, maps, uv, valid, treatment, core, lip3d)
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
        return hp[tyi, tyi]

    def _apply_locked(self, cloud, maps, uv, valid, treatment, core, lip3d) -> dict:
        t = maps.tex
        tx = np.clip(uv[:, 0] * (t - 1), 0, t - 1.001)
        ty = np.clip((1 - uv[:, 1]) * (t - 1), 0, t - 1.001)

        def smp(ch: np.ndarray) -> np.ndarray:
            return core._bilinear(ch[..., None], tx, ty)[..., 0]

        lip_zone = None
        w = smp(maps.w).astype(np.float32)
        w[~valid] = 0.0
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

        # ---- 唇妆：3D 锚定优先，UV 兜底 ----
        if maps.lip_w is not None:
            lip_uv = smp(maps.lip_w).astype(np.float32)
            lip_uv[~valid] = 0.0
            lip_kL, lip_ch = PHOTOREAL_LAB["lipstick"]
            rough_t, coat_t, sheen_t = FINISH_TARGET.get(maps.lip_finish,
                                                         FINISH_TARGET["gloss"])
            if lip3d is not None and float(np.max(lip3d[0])) > 0.02:
                w_lip = np.clip(lip3d[0] * 1.4 * maps.lip_opacity, 0, 1).astype(np.float32)
                cent = np.clip(lip3d[1], 0, 1)
                core2 = _load_core()
                lip_tgt = np.clip(core2.sample_ramp(maps.lip_stops, cent), 0, 1)
            else:                                   # 兜底：UV 唇带
                w_lip = lip_uv
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
        out = {}
        for k, v in cloud.items():
            out[k] = v.copy() if isinstance(v, np.ndarray) else v

        kL_w = (kL * w)[..., None]
        a_ch = (chroma * w)[..., None]
        if treatment == "lab":
            cur_lab = cv2.cvtColor((cur * 255).astype(np.uint8)[:, None, :],
                                   cv2.COLOR_RGB2Lab).astype(np.float32)[:, 0, :]
            tgt_lab = cv2.cvtColor((tgt * 255).astype(np.uint8)[:, None, :],
                                   cv2.COLOR_RGB2Lab).astype(np.float32)[:, 0, :]
            new_lab = np.stack([
                cur_lab[:, 0] + (tgt_lab[:, 0] - cur_lab[:, 0]) * kL_w[:, 0],
                cur_lab[:, 1] + (tgt_lab[:, 1] - cur_lab[:, 1]) * a_ch[:, 0],
                cur_lab[:, 2] + (tgt_lab[:, 2] - cur_lab[:, 2]) * a_ch[:, 0],
            ], axis=1)
            back = cv2.cvtColor(np.clip(new_lab, 0, 255).astype(np.uint8)[:, None, :],
                                cv2.COLOR_Lab2RGB).astype(np.float32)[:, 0, :] / 255.0
        else:                                   # alpha-over（合成脸校准）
            back = cur * (1 - w[..., None]) + tgt * w[..., None]
        out["rgba"][:, :3] = back

        from .pbr import Material
        rough = SKIN_ROUGH * (1 - w) + smp(maps.channels["rough"]) * w
        coat = np.maximum(smp(maps.channels["coat"]) * w, 0.06 * (1 - w))
        sss = smp(maps.channels["sss"]) * w
        sheen = smp(maps.channels["sheen"]) * w
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
