#!/usr/bin/env python3
"""compositor — 摄像头帧的实时妆容合成（图像域蒙版路径）。

与 makeup-skill/scripts/preview_render.py 的 RegionMasks.bake 一一对应的区域几何
（同一 landmark-regions.json 分组、同一 color_stops 向心度取色、同一 FEATHER 基准），
只是把"UV 纹理空间"换到"摄像头帧像素空间"，蒙版直接锚定在 MediaPipe 关键点上。

实时性取舍：羽化用高斯模糊近似距离场衰减（省 distanceTransform），大面积区域
0.7 分辨率 + 细节区域全分辨率的混合管线，单帧约 <15ms @720p、<40ms @1080p。
"""
from __future__ import annotations

import importlib.util
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

_SKILL_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "makeup-skill" / "scripts"


def _load_core():
    """复用渲染内核的取色/羽化规则，保证与 3DGS 预览、参考图同一套妆容语义。"""
    if "preview_render_core" not in globals():
        spec = importlib.util.spec_from_file_location(
            "preview_render_core", _SKILL_SCRIPTS / "preview_render.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        globals()["preview_render_core"] = mod
    return globals()["preview_render_core"]


REFS = _SKILL_SCRIPTS.parent / "references"
FINISH = {"matte": 0.0, "satin": 0.25, "dewy": 0.55, "gloss": 0.9}
SKIN_BASE = np.array([0.925, 0.776, 0.678], np.float32)   # 与内核 bake_skin 底色一致

# 色彩迁移的明度迁移系数：越高 = 妆色越“实”，越低 = 越保留原皮肤明暗/质感。
# 对照内核参考渲染校准：唇/眉是深色妆品，明度必须大幅跟随（否则只剩粉感），
# 高光只轻微提亮（避免白色条带），底妆保留大部分原明度以保纹理。
LUM_SHIFT = {"foundation": 0.25, "concealer": 0.25, "contour": 0.40, "blush": 0.55,
             "eyeshadow": 0.50, "eyebrow": 0.55, "lipstick": 0.95, "highlight": 0.55}
# 彩度增益：Lab 的 a/b 迁移在低 alpha 下等效彩度低于内核的 alpha-over 合成，
# 彩色妆品（腮红/眼影/唇）需要额外增益才能达到参考渲染的饱和度。
# 底妆类相反：目标色是低饱和象牙色，全力迁移会把皮肤暖调洗灰，只轻推色相
CHROMA_GAIN = {"blush": 1.6, "eyeshadow": 1.5, "lipstick": 1.25,
               "contour": 1.2, "eyebrow": 1.2, "foundation": 0.45, "concealer": 0.6}
# 直接混色（不保明度）：眼线/睫毛本就是近黑硬边，色彩迁移反而发灰
DIRECT_BLEND = {"eyeliner", "lashes"}


class ImageRegionMasks:
    """RegionMasks 的图像域版本：_pts 返回关键点像素坐标，尺寸单位 = 脸宽比例。"""

    def __init__(self, regions_json: str | Path):
        data = json.loads(Path(regions_json).read_text(encoding="utf-8"))
        self.regions = data["regions"]

    def bind(self, px: np.ndarray, origin: tuple[float, float] = (0.0, 0.0)) -> None:
        """绑定当前帧关键点（全帧像素坐标）与绘制原点（ROI 左上角），并估计脸宽比例尺度 S。"""
        self.px = px
        self.origin = np.asarray(origin, np.float64)
        oval = self.regions["foundation_face_oval"]["indices"]
        pts = px[oval]
        self.S = float(max(pts[:, 0].max() - pts[:, 0].min(),
                           pts[:, 1].max() - pts[:, 1].min()))
        self.S = max(self.S, 32.0)

    def _pts(self, group: str) -> np.ndarray:
        return self.px[self.regions[group]["indices"]] - self.origin

    def _ext(self, group: str) -> np.ndarray:
        ext = self.regions[group].get("extended")
        return (self.px[ext] - self.origin) if ext else np.zeros((0, 2))

    def _anchor(self, group: str) -> np.ndarray:
        return self.px[self.regions[group]["center_anchor"]] - self.origin

    # ---------- 图元（尺寸 = 脸宽比例 × S，与内核 UV 比例语义一致） ----------

    def _fill(self, canvas, pts, value=1.0):
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], value)

    def _stroke(self, canvas, pts, width_frac, value=1.0):
        w = max(1, int(round(width_frac * self.S)))
        cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, value,
                      thickness=w, lineType=cv2.LINE_AA)

    def _ellipse(self, canvas, c, rx, ry, angle, value=1.0):
        cv2.ellipse(canvas, (int(round(c[0])), int(round(c[1]))),
                    (max(1, int(rx * self.S)), max(1, int(ry * self.S))),
                    angle, 0, 360, value, -1, lineType=cv2.LINE_AA)

    # ---------- 各部位（几何与内核 RegionMasks.bake 相同，center_uv 分支用锚点近似） ----------

    def _eyeshadow(self, canvas, side, shape):
        for one in ("left", "right"):
            if side != "both" and one != side:
                continue
            pts = np.vstack([self._pts(f"eyelid_{one}"), self._ext(f"eyelid_{one}")])
            # 局部画布：填充/距离场只覆盖眼睑包围盒（全画布距离变换太慢）
            lo = pts.min(axis=0) - 0.05 * self.S
            hi = pts.max(axis=0) + 0.05 * self.S
            bx0, by0 = int(max(0, lo[0])), int(max(0, lo[1]))
            bx1 = int(min(self.canvas_shape[1], hi[0] + 1))
            by1 = int(min(self.canvas_shape[0], hi[1] + 1))
            if bx1 <= bx0 or by1 <= by0:
                continue
            tmp = np.zeros((by1 - by0, bx1 - bx0), np.float32)
            cv2.fillPoly(tmp, [np.round(pts - (bx0, by0)).astype(np.int32)], 1.0)
            # 睫毛线处最实、向上渐淡（真实眼影的画法），而非整块均匀填充
            h = max(float(pts[:, 1].max() - pts[:, 1].min()), 1.0)
            lash = self._pts(f"eyeliner_{one}")
            line = np.zeros_like(tmp)
            cv2.polylines(line, [np.round(lash - (bx0, by0)).astype(np.int32)],
                          False, 1.0, max(1, int(0.002 * self.S)), cv2.LINE_AA)
            dist = cv2.distanceTransform((line <= 0.5).astype(np.uint8), cv2.DIST_L2, 3)
            tmp *= np.exp(-(dist / max(0.5 * h, 1.0)) ** 1.5)
            np.maximum(canvas[by0:by1, bx0:bx1], tmp, out=canvas[by0:by1, bx0:bx1])
            lower = float(shape.get("lower_lid", 0.0) or 0.0)
            if lower > 0.01:
                low = np.zeros(self.canvas_shape, np.float32)
                self._fill(low, self._pts(f"lower_lid_{one}"))
                np.maximum(canvas, low * lower, out=canvas)

    def _eyeliner(self, canvas, side, shape, lashes=False):
        width = (0.006 if lashes else 0.0) + float(shape.get("thickness", 0.25)) * 0.008
        wing = float(shape.get("wing", 0.3) or 0.0)
        for one in ("left", "right"):
            if side != "both" and one != side:
                continue
            pts = self._pts(f"eyeliner_{one}")
            self._stroke(canvas, pts, width)
            if not lashes and wing > 0.01:
                wa = self.regions[f"eyeliner_{one}"]["wing_anchor"]
                wing_px = self.px[wa] - self.origin
                if len(wing_px) >= 2:
                    tail = wing_px[-1]
                    d = wing_px[-1] - wing_px[0]
                    outward = 1.0 if d[0] >= 0 else -1.0
                    lift = np.array([outward * 0.65, 0.35])
                    lift /= np.linalg.norm(lift) + 1e-9
                    ext = tail + lift * (0.008 + 0.018 * wing) * self.S
                    self._stroke(canvas, np.vstack([tail, ext]), width * 0.8)

    def _blush(self, canvas, side, shape):
        radius = float(shape.get("radius", 0.12))
        angle = float(shape.get("angle_deg", 15))
        for one in ("left", "right"):
            if side != "both" and one != side:
                continue
            c = self._anchor(f"blush_{one}")
            sign = 1.0 if one == "left" else -1.0
            rx, ry = radius * self.S, radius * 1.25 * self.S
            # 径向渐变：中心最实、向外连续衰减到 0（真实腮红没有“硬芯”）
            x0, x1 = int(max(0, c[0] - rx - 2)), int(min(self.canvas_shape[1], c[0] + rx + 3))
            y0, y1 = int(max(0, c[1] - ry - 2)), int(min(self.canvas_shape[0], c[1] + ry + 3))
            if x1 <= x0 or y1 <= y0:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            dx, dy = xs - c[0], ys - c[1]
            a = np.deg2rad(sign * angle)
            u = dx * np.cos(a) + dy * np.sin(a)
            v = -dx * np.sin(a) + dy * np.cos(a)
            grad = np.clip(1.0 - np.sqrt((u / max(rx, 1e-3)) ** 2 + (v / max(ry, 1e-3)) ** 2),
                           0, 1) ** 1.5
            np.maximum(canvas[y0:y1, x0:x1], grad, out=canvas[y0:y1, x0:x1])

    def _highlight(self, canvas, shape):
        areas = set(shape.get("areas") or ["cheek", "nose"])
        if "cheek" in areas:
            for g, ang in (("highlight_cheek_left", 10), ("highlight_cheek_right", -10)):
                self._ellipse(canvas, self._anchor(g), 0.034, 0.024, ang)
        if "nose" in areas:
            self._stroke(canvas, self._pts("highlight_nose"), 0.012)
        if "cupid" in areas:
            self._stroke(canvas, self._pts("highlight_cupid"), 0.009)

    def bake(self, layer: dict) -> np.ndarray:
        region = layer["region"]
        side = layer.get("side", "both")
        shape = layer.get("shape") or {}
        cov = np.zeros(self.canvas_shape, np.float32)

        if region == "foundation":
            self._fill(cov, self._pts("foundation_face_oval"))
            # 腔洞抠减：眼睛/眉/唇不能被粉底平涂（“面具感”的主要来源）。
            # 眼位用上睑+下睑关键点的联合包围圈估计。
            # 注意 _ellipse 的半径语义是"占脸宽比例"，这里拿到的是像素半径，
            # 必须直接用 cv2.ellipse（否则画出盖住全脸的巨型空洞，粉底全没）
            for one in ("left", "right"):
                e = np.vstack([self._pts(f"eyelid_{one}"), self._pts(f"lower_lid_{one}")])
                c = e.mean(axis=0)
                r = float(np.ptp(e, axis=0).max()) / 2
                cv2.ellipse(cov, (int(round(c[0])), int(round(c[1]))),
                            (int(r * 1.25 + 0.004 * self.S), int(r * 1.1)),
                            0, 0, 360, 0.0, -1, lineType=cv2.LINE_AA)
            for one in ("left", "right"):
                self._stroke(cov, self._pts(f"eyebrow_{one}"), 0.018, 0.0)
            self._fill(cov, self._pts("lips_outer"), 0.0)
        elif region == "concealer":
            if shape.get("under_eye", True) is not False:
                size = min(max(float(shape.get("size", 1.0) or 1.0), 0.4), 2.0)
                for one in ("left", "right"):
                    pts = self._pts(f"lower_lid_{one}")
                    c = pts.mean(axis=0) + np.array([0.0, 0.012 * self.S])
                    self._ellipse(cov, c, 0.055 * size, 0.028 * size, 0)
        elif region == "contour":
            s = float(shape.get("strength", 0.5))
            f = 0.6 + 0.8 * s
            self._stroke(cov, self._pts("contour_forehead"), 0.035 * f)
            if shape.get("jaw", True):
                self._stroke(cov, self._pts("contour_jaw_left"), 0.03 * f)
                self._stroke(cov, self._pts("contour_jaw_right"), 0.03 * f)
            if shape.get("nose", True):
                self._stroke(cov, self._pts("contour_nose"), 0.012 * f)
        elif region == "eyebrow":
            w = 0.010 + float(shape.get("thickness", 0.4)) * 0.006
            for one in ("left", "right"):
                if side != "both" and one != side:
                    continue
                self._stroke(cov, self._pts(f"eyebrow_{one}"), w)
        elif region == "eyeshadow":
            self._eyeshadow(cov, side, shape)
        elif region == "eyeliner":
            self._eyeliner(cov, side, shape, lashes=False)
        elif region == "lashes":
            self._eyeliner(cov, side, shape, lashes=True)
        elif region == "blush":
            self._blush(cov, side, shape)
        elif region == "highlight":
            self._highlight(cov, shape)
        elif region == "lipstick":
            self._fill(cov, self._pts("lips_outer"))
            overline = min(max(float(shape.get("overline", 0.0) or 0.0), 0.0), 1.0)
            if overline > 0.01:
                self._stroke(cov, self._pts("lips_outer"), 0.004 + overline * 0.012)
        else:
            raise ValueError(f"未知 region: {region}")
        return cov


class WebcamCompositor:
    """把当前妆容（spec.layers）实时合成到摄像头帧上。

    process(frame_bgr, lm_px, intensity) → BGR 帧。set_look 在妆容变化时调用一次。

    双分辨率管线：大面积区域（底妆/腮红/修容…）在 0.7 分辨率 ROI 内处理，
    细节区域（眼影/眼线/睫毛/眉/唇）在全分辨率上处理——唇纹眼线的锐度
    是"像不像真妆"的关键，低分辨率上采样会全部糊掉。
    """

    # 细节区域：全分辨率处理（锐度敏感）。眼影虽在眼周，但本质是软渐变，
    # 0.7 分辨率足够；真正吃分辨率的只有线状/硬边的眼线、睫毛、眉与唇
    DETAIL_REGIONS = {"eyeliner", "lashes", "eyebrow", "lipstick"}
    # 均匀肤色的区域：a/b 向局部中值收敛（粉底/遮瑕的真实作用是均匀肤色）
    EVEN_REGIONS = {"foundation", "concealer"}

    def __init__(self, regions_json: str | Path = REFS / "landmark-regions.json"):
        self.core = _load_core()
        self.masks = ImageRegionMasks(regions_json)
        self._layers: list[dict] = []
        self._key = None
        self.env = "neutral"

    def set_env(self, env: str) -> None:
        """环境光预设（preview_render.ENVS 同名）。"""
        self.env = env if env in self.core.ENVS else "neutral"

    def set_look(self, layers: list[dict]):
        layers = [dict(l) for l in layers if l.get("enabled", True)]
        key = zlib.crc32(json.dumps(layers, sort_keys=True).encode())
        if key != self._key:
            self._key = key
            self._layers = layers
            # 色带 LUT：向心度 0..1 → 256 级 RGB，替代全图 sample_ramp
            ts = np.linspace(0.0, 1.0, 256).astype(np.float32)
            self._luts = [self.core.sample_ramp(l["color_stops"], ts) for l in layers]
            # 颗粒纹理（与内核 bake_makeup 的 grain 同源同种子）：粉质/唇纹的微噪声
            self._grains = []
            for l in layers:
                grain = float(l.get("texture_strength", 0.3) or 0)
                if grain > 0.01:
                    seed = zlib.crc32(l["id"].encode()) & 0xFFFF
                    g = np.random.default_rng(seed).normal(0.5, 0.18, (256, 256))
                    self._grains.append(g.astype(np.float32))
                else:
                    self._grains.append(None)

    def _feather_width(self, region, shape) -> float:
        falloff = float(shape.get("falloff", 0.65) or 0.65)
        base = self.core.FEATHER_PX.get(region, 8.0)
        if region == "lipstick":
            base = 3.0 + float(shape.get("blur", 0.15) or 0.0) * 30.0
        return base * (0.5 + min(max(falloff, 0.0), 1.0)) * (self.masks.S / 512.0)

    def _feather(self, cov, region, shape):
        fw = self._feather_width(region, shape)
        if fw < 0.8:
            return cov
        if fw >= 12:
            # 大羽化是低频信号：降采样 → 小核模糊 → 放大，视觉等价、
            # 成本从 O(bbox·fw²) 降约 (fw/6)² 倍（高分辨率下不掉帧的关键）
            r = int(min(24, max(2, round(fw / 6))))
            small = cv2.resize(cov, (max(1, cov.shape[1] // r), max(1, cov.shape[0] // r)),
                               interpolation=cv2.INTER_AREA)
            k = int(max(3, round(fw * 2 / r))) | 1
            small = cv2.GaussianBlur(small, (k, k), fw / (2.0 * r))
            return cv2.resize(small, (cov.shape[1], cov.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        k = int(max(3, round(fw * 2))) | 1
        return cv2.GaussianBlur(cov, (k, k), fw / 2.0)

    # ---- 单层合成（canvas 为 uint8 画布；float 运算只发生在本层包围盒内） ----

    def _apply_layer(self, canvas, layer, li, tint, intensity, glow, parse=None):
        region = layer["region"]
        cov = self.masks.bake(layer)
        # 解析蒙版可用时，唇/底妆用像素级分割替代 landmark 多边形：
        # 唇形严丝合缝；skin 天然排除眉/眼/唇，粉底不用再手抠腔洞。
        # 羽化用"外缘羽化"（mask×blur(mask)）：普通高斯羽化会把周围妆色
        # 回填进细长的眉/眼缝空洞里
        if parse is not None:
            pm = None
            if region == "foundation":
                pm = parse.get("skin")
            elif region == "lipstick":
                pm = parse.get("lips")
            if pm is not None:
                sigma = max(self._feather_width(region, layer.get("shape") or {}) / 2.0, 0.5)
                cov = pm * cv2.GaussianBlur(pm, (0, 0), sigma)
        # 先裁到包围盒（留羽化余量），羽化/距离场等重操作只在局部进行。
        # boundingRect 是 C 实现，比 np.nonzero 构建索引数组快一个量级；
        # bool→uint8 用零拷贝 view（bool 数组内存布局即 0/1 uint8）
        bin8 = (cov > 0.004).view(np.uint8)
        bx0, by0, bw, bh = cv2.boundingRect(bin8)
        if bw == 0 or bh == 0:
            return
        pad = int(self._feather_width(region, layer.get("shape") or {}) * 3) + 4
        by0 = max(0, by0 - pad)
        bx0 = max(0, bx0 - pad)
        by1 = min(cov.shape[0], by0 + bh + 2 * pad)
        bx1 = min(cov.shape[1], bx0 + bw + 2 * pad)
        sub = self._feather(cov[by0:by1, bx0:bx1], region, layer.get("shape") or {})
        if region in self.EVEN_REGIONS:
            # 底妆类色带渐变意义弱，跳过向心度距离场（大 bbox 上很贵）
            cent = sub
        else:
            # 向心度（渐变取色）：ROI 内距离场。取色是低频信号 → 1/4 分辨率
            # 上算距离场再放大，成本降 ~16 倍，渐变色带不变
            bh, bw = sub.shape
            small = cv2.resize(sub, (max(1, bw // 4), max(1, bh // 4)),
                               interpolation=cv2.INTER_AREA)
            dist_c = cv2.distanceTransform((small > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
            dmax = dist_c.max()
            cent_small = np.where(small > 0.01, 1.0 - dist_c / max(dmax, 1e-3), 0.0)
            cent = cv2.resize(cent_small.astype(np.float32), (bw, bh),
                              interpolation=cv2.INTER_LINEAR)
        idx = np.clip(cent * 255, 0, 255).astype(np.uint8)
        a = np.clip(sub * float(layer.get("opacity", 0.7)) * float(intensity), 0, 1)
        # 颗粒纹理：颜色与覆盖同时微扰（与内核 grain 语义一致），打破"色块感"
        g = self._grains[li]
        if g is not None and region != "foundation":
            gw = cv2.resize(g, (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_LINEAR)
            a = np.clip(a * (1 + (gw - 0.5) * 0.25), 0, 1)
        out_bb_u8 = canvas[by0:by1, bx0:bx1]
        out_bb = out_bb_u8.astype(np.float32) * (1.0 / 255.0)

        if region in DIRECT_BLEND:
            # 近黑硬边：直接 alpha 混色最干净
            col = np.clip(self._luts[li][idx], 0, 1) * tint
            if g is not None:
                col = np.clip(col * (1 + (gw[..., None] - 0.5) * 0.3), 0, 1)
            out_bb *= (1 - a[..., None])
            out_bb += col[..., ::-1] * a[..., None]
        else:
            # Lab 色彩迁移：只把色相/饱和度（a,b）推向妆色，明度只按 LUM_SHIFT
            # 部分迁移 → 皮肤纹理、光影和毛孔全保留，不再是"平涂色块"。
            # 目标色 Lab 用 256 级 LUT 转换后查表（避免对整个 bbox 做色彩空间转换）
            lut_rgb = np.clip(self._luts[li] * tint, 0, 1).astype(np.float32)[None, ...]
            lab_lut = cv2.cvtColor(lut_rgb, cv2.COLOR_RGB2Lab)[0]
            t_lab = lab_lut[idx]
            cur_lab = cv2.cvtColor(np.ascontiguousarray(out_bb), cv2.COLOR_BGR2Lab)
            kL = LUM_SHIFT.get(region, 0.3)
            tL = t_lab[..., 0] / 100.0
            tA = t_lab[..., 1] / 255.0 + 0.5
            tB = t_lab[..., 2] / 255.0 + 0.5
            if g is not None and region != "foundation":
                tL = np.clip(tL * (1 + (gw - 0.5) * 0.18), 0, 1)
            cL = cur_lab[..., 0] / 100.0
            cA = cur_lab[..., 1] / 255.0 + 0.5
            cB = cur_lab[..., 2] / 255.0 + 0.5
            if region in self.EVEN_REGIONS:
                # 均匀肤色：a/b 向局部中值收敛（去泛红/色斑），明度纹理不动——
                # 这是真实粉底的作用方式。色度是低频信号：1/4 分辨率上模糊
                # 再放大回贴，效果一致但模糊耗时降 ~90%
                sw, sh_ = max(1, cA.shape[1] // 4), max(1, cA.shape[0] // 4)
                k = int(max(3, round(self.masks.S * 0.015))) | 1
                sA = cv2.GaussianBlur(cv2.resize(cA, (sw, sh_), interpolation=cv2.INTER_AREA),
                                      (k, k), k / 5.0)
                sB = cv2.GaussianBlur(cv2.resize(cB, (sw, sh_), interpolation=cv2.INTER_AREA),
                                      (k, k), k / 5.0)
                cA += (cv2.resize(sA, (cA.shape[1], cA.shape[0]),
                                  interpolation=cv2.INTER_LINEAR) - cA) * (0.65 * a)
                cB += (cv2.resize(sB, (cB.shape[1], cB.shape[0]),
                                  interpolation=cv2.INTER_LINEAR) - cB) * (0.65 * a)
            a_ch = np.clip(a * CHROMA_GAIN.get(region, 1.0), 0, 1)
            new_lab = cv2.merge([
                (cL + (tL - cL) * (kL * a)) * 100.0,
                (cA + (tA - cA) * a_ch) * 255.0 - 128.0,
                (cB + (tB - cB) * a_ch) * 255.0 - 128.0])
            out_bb[...] = cv2.cvtColor(new_lab, cv2.COLOR_Lab2BGR)

        # 光泽 finish：清漆镜面。阈值取高（只提真正亮的部位）、强度克制，
        # 唇妆的"玻璃感"由下面基于几何的高光条负责，不走这个通用高光
        sh = FINISH.get(layer.get("finish", "satin"), 0.25) * glow
        if sh > 0.05 and region != "lipstick":
            lum = out_bb.mean(axis=2, keepdims=True)
            spec = np.clip(lum - 0.70, 0, 1) / 0.30
            out_bb += (spec * a[..., None] * sh * 0.35) * np.array([0.9, 0.95, 1.0], np.float32)

        if region == "lipstick" and sh > 0.05:
            # 下唇中央高光条：真实唇妆"玻璃感"的主要来源（只取距场峰值附近）
            lp = self.masks._pts("lips_outer")
            lower = lp[lp[:, 1] > lp[:, 1].mean()]
            if len(lower) >= 3:
                # 局部画布（下唇包围盒），避免全画布距离变换
                lo = lower.min(axis=0) - 3
                hi = lower.max(axis=0) + 4
                mx0, my0 = int(max(0, lo[0])), int(max(0, lo[1]))
                mx1 = int(min(cov.shape[1], hi[0] + 1))
                my1 = int(min(cov.shape[0], hi[1] + 1))
                if mx1 > mx0 and my1 > my0:
                    low_mask = np.zeros((my1 - my0, mx1 - mx0), np.float32)
                    cv2.fillPoly(low_mask,
                                 [np.round(lower - (mx0, my0)).astype(np.int32)], 1.0)
                    d_low = cv2.distanceTransform((low_mask > 0.5).astype(np.uint8),
                                                  cv2.DIST_L2, 3)
                    if d_low.max() > 1.0:
                        band = np.clip((d_low / d_low.max() - 0.6) * 2.5, 0, 1) ** 1.5
                        # 唇 bbox 由 landmark 推导、sub 窗口来自本层蒙版（解析分割
                        # 缺角/错位时边界不一致）：裁到包围盒交集，防止负索引回绕
                        # 产生空/错位切片使相机线程崩溃
                        gx0, gy0 = max(mx0, bx0), max(my0, by0)
                        gx1, gy1 = min(mx1, bx1), min(my1, by1)
                        if gx1 > gx0 and gy1 > gy0:
                            gb = band[gy0 - my0:gy1 - my0, gx0 - mx0:gx1 - mx0] \
                                * (sub[gy0 - by0:gy1 - by0, gx0 - bx0:gx1 - bx0] > 0.05)
                            out_bb[gy0 - by0:gy1 - by0, gx0 - bx0:gx1 - bx0] += \
                                (gb[..., None] * (0.18 * sh)) \
                                * np.array([0.85, 0.92, 1.0], np.float32)

        out_bb_u8[...] = np.clip(out_bb * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def set_parse(self, masks: dict[str, np.ndarray] | None, lm_px: np.ndarray | None):
        """登记最新一次人脸解析结果（masks 与产生它的那帧 landmark 成对）。
        解析低频跑；process 每帧把蒙版按 landmark 相似变换对齐到当前帧。"""
        self._parse = (masks, lm_px)

    def process(self, frame_bgr: np.ndarray, lm_px: np.ndarray, intensity: float = 1.0,
                glow: float = 1.0) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        # 只在人脸包围盒 ROI 内合成
        pad = int(0.35 * max(
            lm_px[:, 0].max() - lm_px[:, 0].min(), lm_px[:, 1].max() - lm_px[:, 1].min(), 64))
        x0 = max(0, int(lm_px[:, 0].min()) - pad)
        x1 = min(w, int(lm_px[:, 0].max()) + pad)
        y0 = max(0, int(lm_px[:, 1].min()) - pad)
        y1 = min(h, int(lm_px[:, 1].max()) + pad)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame_bgr
        origin = np.array([x0, y0], np.float64)
        roi_full = frame_bgr[y0:y1, x0:x1]

        # 解析蒙版对齐到当前帧（解析结果比当前帧旧时，人脸可能已移动）
        parse = None
        pm, plm = getattr(self, "_parse", (None, None))
        if pm and plm is not None and len(plm) == len(lm_px):
            from .parser import estimate_face_affine, warp_masks
            affine = estimate_face_affine(plm, lm_px)
            parse = warp_masks(pm, affine, (h, w))

        # 环境自适应：脸内皮肤中位色 → 色温系数（降采样取中位数，np.median 很慢）
        lm_roi = lm_px - origin
        oval = lm_roi[self.masks.regions["foundation_face_oval"]["indices"]].round().astype(np.int32)
        skin_mask = np.zeros(roi_full.shape[:2], np.uint8)
        cv2.fillPoly(skin_mask, [oval], 1)
        vals = roi_full[skin_mask > 0]
        if len(vals) > 16:
            if len(vals) > 8000:                    # 最多取 8k 个样本估计中位色
                vals = vals[::len(vals) // 8000]
            # roi_full 是 0-255 的 uint8，SKIN_BASE 是 0-1 基线：先归一
            skin = np.median(vals, axis=0)[::-1] / 255.0    # BGR→RGB
            tint = np.clip(skin / SKIN_BASE, 0.55, 1.45).astype(np.float32)
        else:
            tint = np.ones(3, np.float32)
        env_preset = self.core.ENVS[self.env]
        tint = np.clip(tint * np.array(env_preset["tint"], np.float32), 0.4, 1.8)

        # ---- Pass 1：大面积区域。处理比例自适应：人脸在处理空间保持 ~430px，
        # 相对清晰度与帧率在 720p~4K 下一致（相机帧越大比例越低，但都比旧固定
        # 0.42 在 720p 下的 ~240px 清晰一倍）；uint8 画布 ----
        big = [li for li, l in enumerate(self._layers)
               if l["region"] not in self.DETAIL_REGIONS]
        if big:
            oval_px = lm_px[self.masks.regions["foundation_face_oval"]["indices"]]
            face_px = float(max(np.ptp(oval_px[:, 0]), np.ptp(oval_px[:, 1]), 32.0))
            scale = min(0.72, max(0.42, 430.0 / face_px))
            w2 = max(8, int((x1 - x0) * scale))
            h2 = max(8, int((y1 - y0) * scale))
            self.masks.bind((lm_px - origin) * scale, origin=(0.0, 0.0))
            self.masks.canvas_shape = (h2, w2)
            out = cv2.resize(roi_full, (w2, h2), interpolation=cv2.INTER_AREA)
            parse_small = None
            if parse:
                parse_small = {k: cv2.resize(v[y0:y1, x0:x1], (w2, h2),
                                             interpolation=cv2.INTER_LINEAR)
                               for k, v in parse.items()}
            for li in big:
                self._apply_layer(out, self._layers[li], li, tint, intensity, glow,
                                  parse=parse_small)
            roi_full[:] = cv2.resize(out, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)

        # ---- Pass 2：细节区域，全分辨率（锐度），直接写在帧 ROI 上 ----
        detail = [li for li, l in enumerate(self._layers)
                  if l["region"] in self.DETAIL_REGIONS]
        if detail:
            self.masks.bind(lm_roi, origin=(0.0, 0.0))
            self.masks.canvas_shape = roi_full.shape[:2]
            parse_roi = None
            if parse:
                parse_roi = {k: v[y0:y1, x0:x1] for k, v in parse.items()}
            for li in detail:
                self._apply_layer(roi_full, self._layers[li], li, tint, intensity, glow,
                                  parse=parse_roi)

        # ---- 环境光分级：脸区软蒙版内整体色温/曝光迁移（dim 再整体压暗），
        # 让"暖光/冷光/暗光下的妆效观感"在实时镜面上可感；neutral 跳过 ----
        if self.env != "neutral":
            t_bgr = np.array(env_preset["tint"][::-1], np.float32)
            graded = roi_full.astype(np.float32) * t_bgr
            if self.env == "dim":
                graded *= 0.82
            m = np.clip(cv2.GaussianBlur(skin_mask, (0, 0), 10), 0, 1)[..., None]
            strength = 0.8
            roi_full[:] = np.clip(
                roi_full * (1 - m * strength) + graded * (m * strength),
                0, 255).astype(np.uint8)
        return frame_bgr
