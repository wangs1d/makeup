#!/usr/bin/env python3
"""preview_render — 试妆渲染内核（skill 自包含，Unity 渲染管线的 Python 等价实现）。

同一 canonical 468 点网格、同一 RegionMask 蒙版规则、同一 spec/溅射规则、同一光照模型：
    · 妆容不是无光照贴片：环境色温（env_tint）× wrap diffuse（次表面近似）×
      双层高光（sheen + dewy/gloss 清漆镜面）；
    · 脸缘羽化（face edge feather）消除网格边界"面具感"；
    · splat 层 = 锚定关键点的各向异性高斯溅射（唇釉/水光体积感）。

供三处使用：
    1) preview/render_demo_video.py — 演示视频离线复现
    2) makeup-skill/scripts/render_look.py — 目标妆参考图（live_coach 给 VLM 做视觉对比）
    3) makeup-skill/scripts/parse_look.py — 解析结果的妆效预览
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import cv2
import numpy as np

TEX = 512
FACE_HEIGHT_M = 0.22          # 真实脸高估计（米），溅射 sigma/offset 的米 → 世界单位换算
REFS = Path(__file__).resolve().parent.parent / "references"

# ---------------- 环境光预设 ----------------

ENVS = {
    "neutral": {"tint": (1.00, 1.00, 1.00), "light": (0.15, 0.45, 0.85)},
    "warm":    {"tint": (1.10, 0.98, 0.86), "light": (0.40, 0.35, 0.80)},
    "cool":    {"tint": (0.93, 0.98, 1.08), "light": (-0.30, 0.55, 0.75)},
    "dim":     {"tint": (1.00, 0.92, 0.85), "light": (0.05, 0.20, 0.90)},
}


# ---------------- 网格 ----------------

def load_uvs(obj_path: str | Path) -> np.ndarray:
    """顶点索引 → UV。canonical obj 的 vt 是图集置换索引（f 里 v/vt 一一对应但编号不同），
    必须从面片解析顶点→vt 置换后再取 UV。"""
    pair = {}
    uvs = []
    for line in Path(obj_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("vt "):
            _, a, b = line.split()[:3]
            uvs.append((float(a), float(b)))
        elif line.startswith("f "):
            for p in line.split()[1:4]:
                v, vt = p.split("/")[:2]
                pair[int(v) - 1] = int(vt) - 1
    vt = np.asarray(uvs, np.float64)
    return vt[[pair[i] for i in range(len(vt))]]


class FaceModel:
    def __init__(self, obj_path: str | Path):
        verts, tris = [], []
        for line in Path(obj_path).read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:4]]
                tris.append(idx)
        V = np.asarray(verts, np.float64)
        # 自动纠正朝向：canonical obj 原始 +y 向上（额头10 > 下巴152）、鼻尖朝 +z
        if V[10, 1] < V[152, 1]:
            V[:, 1] *= -1
        if V[1, 2] < V[152, 2]:
            V[:, 2] *= -1
        self.tris = np.asarray(tris, np.int32)
        # 绕向：以鼻梁突出方向为外法线基准统一翻转，保证顶点法线朝外、屏幕绕向可背面剔除
        p0, p1, p2 = V[self.tris[:, 0]], V[self.tris[:, 1]], V[self.tris[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)
        nose_dir = V[1] - (V[10] + V[152]) / 2
        if np.mean((fn * nose_dir).sum(1)) < 0:
            self.tris = self.tris[:, [0, 2, 1]].copy()
        self.uvs = load_uvs(obj_path)
        # 居中 + 归一化脸高 = 1 世界单位
        lo, hi = V.min(0), V.max(0)
        self.scale = 1.0 / (hi[1] - lo[1])
        self.base = (V - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])) * self.scale
        self.n_verts = len(V)

    def pose(self, t: float, mouth_k: float, smile_k: float) -> np.ndarray:
        """姿态动画：转头 + 张嘴 + 微笑。返回世界坐标顶点。t 固定即为静态姿势。"""
        yaw = 20.0 * np.sin(2 * np.pi * t / 9.0) + 6.0
        pitch = 5.0 * np.sin(2 * np.pi * t / 13.0 + 1.0) - 2.0
        roll = 2.5 * np.sin(2 * np.pi * t / 17.0)
        V = self.pose_explicit(yaw, pitch, roll, mouth_k, smile_k)
        V[:, 1] += 0.004 * np.sin(2 * np.pi * t / 4.2)     # 呼吸
        return V

    def pose_explicit(self, yaw_deg: float, pitch_deg: float, roll_deg: float,
                      mouth_k: float, smile_k: float) -> np.ndarray:
        """显式姿态（度）+ 表情参数。返回世界坐标顶点。"""
        V = self.base.copy()
        yaw, pitch, roll = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), np.deg2rad(roll_deg)
        cy, sy, cx, sx, cz, sz = (np.cos(yaw), np.sin(yaw), np.cos(pitch),
                                  np.sin(pitch), np.cos(roll), np.sin(roll))
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        V = V @ (Rz @ Rx @ Ry).T
        # 张嘴：lips_inner 上半上移、下半下移；微笑：嘴角上提外拉
        inner = self.inner_idx
        upper = V[inner, 1] < V[inner, 1].mean()
        V[inner[upper], 1] += mouth_k * 0.045
        V[inner[~upper], 1] -= mouth_k * 0.055
        for corner in self.mouth_corners:
            V[corner, 1] -= smile_k * 0.02
            V[corner, 0] += np.sign(V[corner, 0] + 1e-9) * smile_k * 0.008
        return V

    def vertex_normals(self, V: np.ndarray) -> np.ndarray:
        p0, p1, p2 = V[self.tris[:, 0]], V[self.tris[:, 1]], V[self.tris[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        N = np.zeros_like(V)
        for k in range(3):
            np.add.at(N, self.tris[:, k], fn)
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
        return N

    def prepare(self, regions_json, obj_path):
        """预计算拓扑相关量（一次性）。"""
        data = json.loads(Path(regions_json).read_text(encoding="utf-8"))
        self.inner_idx = np.array(data["regions"]["lips_inner"]["indices"])
        self.mouth_corners = [data["regions"]["lips_outer"]["indices"][0],
                              data["regions"]["lips_outer"]["indices"][10]]
        self.eye_centers = []
        for g in ("eyelid_left", "eyelid_right"):
            idx = data["regions"][g]["indices"][:9]
            self.eye_centers.append(idx)
        self.lips_outer_idx = np.array(data["regions"]["lips_outer"]["indices"])
        self.regions = data


# ---------------- 蒙版（与 unity-app RegionMaskBaker.cs 同逻辑） ----------------

class RegionMasks:
    def __init__(self, regions_json: str | Path, obj_path: str | Path):
        data = json.loads(Path(regions_json).read_text(encoding="utf-8"))
        self.regions = data["regions"]
        self.uvs = load_uvs(obj_path)

    def _pts(self, group: str) -> np.ndarray:
        idx = self.regions[group]["indices"]
        uv = self.uvs[idx]
        return np.stack([uv[:, 0] * TEX, (1.0 - uv[:, 1]) * TEX], axis=1)

    def _ext(self, group: str) -> np.ndarray:
        ext = self.regions[group].get("extended")
        if not ext:
            return np.zeros((0, 2))
        uv = self.uvs[ext]
        return np.stack([uv[:, 0] * TEX, (1.0 - uv[:, 1]) * TEX], axis=1)

    def _anchor(self, group: str) -> np.ndarray:
        a = self.regions[group]["center_anchor"]
        u, v = self.uvs[a]
        return np.array([u * TEX, (1.0 - v) * TEX])

    # ---------- 图元 ----------

    @staticmethod
    def _fill(canvas, pts, value=1.0):
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], value)

    @staticmethod
    def _stroke(canvas, pts, width_uv, value=1.0):
        w = max(1, int(round(width_uv * TEX)))
        cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, value,
                      thickness=w, lineType=cv2.LINE_AA)

    @staticmethod
    def _ellipse(canvas, c, rx, ry, angle, value=1.0):
        cv2.ellipse(canvas, (int(round(c[0])), int(round(c[1]))),
                    (max(1, int(rx * TEX)), max(1, int(ry * TEX))),
                    angle, 0, 360, value, -1, lineType=cv2.LINE_AA)

    # ---------- 各部位（与 RegionMaskBaker.cs 一一对应） ----------

    def _eyeshadow(self, canvas, side, shape):
        for one in ("left", "right"):
            if side != "both" and one != side:
                continue
            pts = np.vstack([self._pts(f"eyelid_{one}"), self._ext(f"eyelid_{one}")])
            self._fill(canvas, pts)
            lower = float(shape.get("lower_lid", 0.0) or 0.0)
            if lower > 0.01:
                low = np.zeros_like(canvas)
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
                wing_pts = self.uvs[self.regions[f"eyeliner_{one}"]["wing_anchor"]]
                wing_px = np.stack([wing_pts[:, 0] * TEX, (1 - wing_pts[:, 1]) * TEX], 1)
                if len(wing_px) >= 2:
                    # 眼尾向外上拉：水平方向取眼线走向（内眼角→外眼角）的外侧，左右对称
                    tail = wing_px[-1]
                    d = wing_px[-1] - wing_px[0]
                    outward = 1.0 if d[0] >= 0 else -1.0
                    lift = np.array([outward * 0.65, 0.35])
                    lift = lift / (np.linalg.norm(lift) + 1e-9)
                    ext = tail + lift * (0.008 + 0.018 * wing) * TEX
                    self._stroke(canvas, np.vstack([tail, ext]), width * 0.8)

    def _blush(self, canvas, side, shape):
        radius = float(shape.get("radius", 0.12))
        angle = float(shape.get("angle_deg", 15))
        cu = shape.get("center_uv")
        for one in ("left", "right"):
            if side != "both" and one != side:
                continue
            if cu:
                x, y = float(cu[0]), float(cu[1])
                c = np.array([x if one == "left" else 1 - x, y]) * TEX
            else:
                c = self._anchor(f"blush_{one}")
            sign = 1.0 if one == "left" else -1.0
            self._ellipse(canvas, c, radius, radius * 1.25, sign * angle)

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
        cov = np.zeros((TEX, TEX), np.float32)

        if region == "foundation":
            self._fill(cov, self._pts("foundation_face_oval"))
        elif region == "concealer":
            if not (shape.get("under_eye", True) is False):
                size = min(max(float(shape.get("size", 1.0) or 1.0), 0.4), 2.0)
                for one in ("left", "right"):
                    pts = self._pts(f"lower_lid_{one}")
                    c = pts.mean(axis=0) + np.array([0.0, 0.012 * TEX])
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

        # 羽化：按"到区域边缘的距离"软衰减（大面积腮红/眼影才会有自然的晕染过渡），
        # 再做小核模糊抗锯齿；羽化宽度见 FEATHER_PX（与 RegionMaskBaker.cs 同表）
        falloff = float(shape.get("falloff", 0.65) or 0.65)
        feather = feather_px(region, shape, falloff)
        if feather > 0.0:
            dist = cv2.distanceTransform((cov > 0.05).astype(np.uint8), cv2.DIST_L2, 5)
            cov = cov * np.clip(dist / feather, 0, 1).astype(np.float32)
        cov = cv2.GaussianBlur(cov, (5, 5), 0)
        # 向心度：区域中心 1 → 边缘 0（渐变色带取色）
        dist_c = cv2.distanceTransform((cov > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        dmax = dist_c.max()
        cent = np.where(cov > 0.01, 1.0 - dist_c / max(dmax, 1e-3), 0.0).astype(np.float32)
        return np.stack([cov, cent], axis=-1)


# 各部位羽化基准宽度（像素，512 UV 尺度）；实际 = 基准 × (0.5 + falloff)。
# 0 = 细线条（眉/眼线/睫毛）不做距离衰减，只抗锯齿，否则 1~3px 的线会被侵蚀。
FEATHER_PX = {
    "foundation": 10.0, "concealer": 20.0, "contour": 16.0, "eyebrow": 0.0,
    "eyeshadow": 24.0, "eyeliner": 0.0, "lashes": 0.0, "blush": 44.0,
    "highlight": 14.0, "lipstick": 4.0,
}


def feather_px(region: str, shape: dict, falloff: float) -> float:
    base = FEATHER_PX.get(region, 8.0)
    if region == "lipstick":
        base = 3.0 + float(shape.get("blur", 0.15) or 0.0) * 30.0     # 咬唇妆 blur 调大
    if base <= 0.0:
        return 0.0
    return base * (0.5 + min(max(falloff, 0.0), 1.0))


def sample_ramp(stops: list[dict], t: np.ndarray) -> np.ndarray:
    """向心度图 → RGB。t 取值 0..1（0=边缘色, 1=中心色）。返回 (...,3) float 0..1。"""
    pts = sorted(((float(s["at"]), _hex(s["hex"])) for s in stops), key=lambda p: p[0])
    ts = np.array([p[0] for p in pts])
    cols = np.array([p[1] for p in pts])
    out = np.zeros(t.shape + (3,), np.float32)
    ti = np.clip(t, ts[0], ts[-1])
    for (a0, c0), (a1, c1) in zip(zip(ts, cols), zip(ts[1:], cols[1:])):
        seg = (ti >= a0) & (ti <= a1)
        f = np.where(a1 > a0, (ti - a0) / max(a1 - a0, 1e-9), 0.0)
        out[seg] = c0 + (c1 - c0) * f[seg, None]
    out[ti >= ts[-1]] = cols[-1]
    return out


def _hex(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], np.float32) / 255.0


# ---------------- 纹理烘焙 ----------------

def bake_skin(regions: RegionMasks) -> np.ndarray:
    """程序化皮肤底色（UV 空间，烘焙一次）。返回 (TEX,TEX,3) float 0..1。"""
    yy, xx = np.mgrid[0:TEX, 0:TEX].astype(np.float32)
    u, v = xx / (TEX - 1), 1.0 - yy / (TEX - 1)

    def cov(group, blur=3):
        c = np.zeros((TEX, TEX), np.float32)
        pts = regions._pts(group)
        cv2.fillPoly(c, [np.round(pts).astype(np.int32)], 1.0)
        return cv2.GaussianBlur(c, (blur * 2 + 1, blur * 2 + 1), 0)

    base = np.zeros((TEX, TEX, 3), np.float32)
    base[:] = (0.925, 0.776, 0.678)                      # 暖肤底色
    base += ((v - 0.5) * 0.045)[..., None]               # 纵向明暗
    base += (0.06 * cov("foundation_face_oval", 8))[..., None] * np.array([0.05, 0.03, 0.02])
    lips = cov("lips_outer", 2)
    base = base * (1 - lips * 0.6)[..., None] + lips[..., None] * 0.6 * np.array([0.80, 0.52, 0.50])
    inner = np.zeros((TEX, TEX), np.float32)
    cv2.fillPoly(inner, [np.round(regions._pts("lips_inner")).astype(np.int32)], 1.0)
    base = base * (1 - inner * 0.35)[..., None] + inner[..., None] * 0.35 * np.array([0.62, 0.36, 0.36])
    for g in ("eyebrow_left", "eyebrow_right"):
        b = cov(g, 1)
        base = base * (1 - b * 0.8)[..., None] + b[..., None] * 0.8 * np.array([0.36, 0.27, 0.21])
    for g in ("eyelid_left", "eyelid_right"):
        s = cov(g, 4) * 0.12
        base *= (1 - s)[..., None] + s[..., None] * np.array([0.94, 0.92, 0.91])
    for g, ang in (("blush_left", 15), ("blush_right", -15)):
        c = regions._anchor(g)
        r = np.zeros((TEX, TEX), np.float32)
        cv2.ellipse(r, (int(c[0]), int(c[1])), (int(0.12 * TEX), int(0.15 * TEX)),
                    ang, 0, 360, 1.0, -1)
        r = cv2.GaussianBlur(r, (49, 49), 0) * 0.22
        base += r[..., None] * np.array([0.10, 0.02, 0.01])

    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1, (TEX, TEX)).astype(np.float32)
    base *= 1 + noise[..., None] * 0.012                                  # 细颗粒
    base *= 1 + cv2.GaussianBlur(noise, (0, 0), 6)[..., None] * 0.035     # 低频斑驳
    return np.clip(base, 0, 1)


def bake_makeup(layers: list[dict], regions: RegionMasks) -> tuple[np.ndarray, np.ndarray]:
    """spec layers → 妆容 RGBA 纹理 + sheen 强度图（一次性，spec 变化时重烘焙）。

    intensity 不烘进纹理（渲染时全局乘，便于滑杆动画）。
    """
    rgba = np.zeros((TEX, TEX, 4), np.float32)
    sheen = np.zeros((TEX, TEX), np.float32)
    FINISH = {"matte": 0.0, "satin": 0.25, "dewy": 0.55, "gloss": 0.9}
    for layer in layers:
        if not layer.get("enabled", True):
            continue
        mask = regions.bake(layer)
        cov, cent = mask[..., 0], mask[..., 1]
        if cov.max() < 1e-4:
            continue
        col = sample_ramp(layer["color_stops"], cent)
        grain = float(layer.get("texture_strength", 0.3) or 0)
        g = None
        if grain > 0.01:
            seed = zlib.crc32(layer["id"].encode()) & 0xFFFF
            g = np.random.default_rng(seed).normal(0.5, 0.18, (TEX, TEX)).astype(np.float32)
            col = col * (1 + (g[..., None] - 0.5) * grain * 0.6)
        a = cov * float(layer.get("opacity", 0.7))
        if g is not None:
            a = a * (1 + (g - 0.5) * grain * 0.25)
        # alpha-over 合成
        a = np.clip(a, 0, 1)
        out_a = rgba[..., 3]
        new_a = np.clip(a + out_a * (1 - a), 0, 1)
        w_new = np.where(new_a > 1e-6, a / np.maximum(new_a, 1e-6), 0.0)
        w_old = np.where(new_a > 1e-6, out_a * (1 - a) / np.maximum(new_a, 1e-6), 0.0)
        rgba[..., :3] = np.clip(col, 0, 1) * w_new[..., None] + rgba[..., :3] * w_old[..., None]
        rgba[..., 3] = new_a
        s = FINISH.get(layer.get("finish", "satin"), 0.25)
        sheen = np.maximum(sheen, np.where(cov > 0.02, s, 0.0))
    return np.clip(rgba, 0, 1), sheen


# ---------------- 采样 ----------------

def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """img (TEX,TEX,C)，x/y 像素坐标（已按 iy=(1-v)*TEX 换算）。"""
    x = np.clip(x, 0, TEX - 1.001)
    y = np.clip(y, 0, TEX - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    if img.ndim == 2:
        img = img[..., None]
    c = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return c


# ---------------- 渲染器 ----------------

class FaceRenderer:
    def __init__(self, w: int, h: int, obj_path=None, regions_json=None):
        obj_path = obj_path or (REFS / "canonical_face_model.obj")
        regions_json = regions_json or (REFS / "landmark-regions.json")
        self.model = FaceModel(obj_path)
        self.model.prepare(regions_json, obj_path)
        self.regions = RegionMasks(regions_json, obj_path)
        self.skin = bake_skin(self.regions)
        self.edge = self._bake_edge()            # 脸缘羽化（消除面具边界）
        self.w, self.h = w, h
        self.f = h * 1.85          # 焦距（像素）
        self.d = 2.3               # 相机距离（世界单位）
        self.env = "neutral"
        self._cache: dict = {}

    # ---------- 一次性纹理 ----------

    def _bake_edge(self) -> np.ndarray:
        oval = np.zeros((TEX, TEX), np.float32)
        cv2.fillPoly(oval, [np.round(self.regions._pts("foundation_face_oval")).astype(np.int32)], 1.0)
        dist = cv2.distanceTransform((oval > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        return np.clip(dist / 26.0, 0, 1).astype(np.float32)

    def makeup_for(self, key: str, layers: list[dict]):
        if key not in self._cache:
            self._cache[key] = bake_makeup(layers, self.regions)
        return self._cache[key]

    def set_env(self, name: str) -> None:
        self.env = name if name in ENVS else "neutral"

    # ---------- 渲染 ----------

    def render(self, t: float, makeup: tuple[np.ndarray, np.ndarray] | None,
               prev_makeup=None, fade_w: float = 1.0, intensity: float = 1.0,
               splat_layers: list[dict] | None = None, presence: float = 1.0) -> np.ndarray:
        """返回 (h, w, 3) BGR uint8 视口画面。presence<1 时妆容整体淡出。"""
        W, H = self.w, self.h
        mouth = max(0.0, float(np.sin(2 * np.pi * t / 6.0))) ** 2 * 0.55
        smile = 0.5 + 0.5 * np.sin(2 * np.pi * t / 12.0 + 2.0)
        V = self.model.pose(t, mouth, smile)
        N = self.model.vertex_normals(V)

        # 透视投影（相机 +z 看 -z）
        depth = self.d - V[:, 2]
        px = W / 2 + V[:, 0] * self.f / depth
        py = H / 2 - V[:, 1] * self.f / depth
        P2 = np.stack([px, py], axis=1)

        canvas = self._background(t)
        zbuf = np.full((H, W), 1e9, np.float32)
        cbuf = np.zeros((H, W, 3), np.float32)

        self._draw_cavities(canvas, P2, V, depth, mouth)

        # 三角形光栅化（背面剔除 + zbuffer）
        tris = self.model.tris
        p0, p1, p2 = P2[tris[:, 0]], P2[tris[:, 1]], P2[tris[:, 2]]
        fnz = np.cross(p1 - p0, p2 - p0)            # 屏幕绕向
        keep = fnz < 0                               # 正面
        for ti in np.nonzero(keep)[0]:
            self._raster(ti, tris, P2, V, N, depth, zbuf, cbuf,
                         makeup, prev_makeup, fade_w, intensity, presence)
        canvas = np.where(zbuf[..., None] < 1e8, cbuf, canvas)

        if splat_layers and presence > 0.01:
            self._draw_splats(canvas, V, N, depth, splat_layers, intensity * presence)
        # 内部统一 RGB，输出转 BGR（cv2/VideoWriter 约定）
        return np.clip(canvas * 255, 0, 255).astype(np.uint8)[..., ::-1]

    def render_still(self, layers: list[dict], intensity: float = 1.0,
                     yaw_deg: float = 0.0, smile: float = 0.12,
                     splat_layers: list[dict] | None = None) -> np.ndarray:
        """静态正脸（或指定偏航角）渲染——参考图/预览用。"""
        V = self.model.pose_explicit(yaw_deg, -2.0, 0.0, 0.03, smile)
        N = self.model.vertex_normals(V)
        W, H = self.w, self.h
        depth = self.d - V[:, 2]
        P2 = np.stack([W / 2 + V[:, 0] * self.f / depth,
                       H / 2 - V[:, 1] * self.f / depth], axis=1)
        canvas = self._background(2.6)
        zbuf = np.full((H, W), 1e9, np.float32)
        cbuf = np.zeros((H, W, 3), np.float32)
        self._draw_cavities(canvas, P2, V, depth, 0.03)
        mk = None
        if layers:
            key = "still:" + str(zlib.crc32(json.dumps(layers, sort_keys=True).encode()))
            mk = self.makeup_for(key, layers)
        tris = self.model.tris
        p0, p1, p2 = P2[tris[:, 0]], P2[tris[:, 1]], P2[tris[:, 2]]
        keep = np.cross(p1 - p0, p2 - p0) < 0
        for ti in np.nonzero(keep)[0]:
            self._raster(ti, tris, P2, V, N, depth, zbuf, cbuf, mk, None, 1.0, intensity, 1.0)
        canvas = np.where(zbuf[..., None] < 1e8, cbuf, canvas)
        if splat_layers:
            self._draw_splats(canvas, V, N, depth, splat_layers, intensity)
        return np.clip(canvas * 255, 0, 255).astype(np.uint8)[..., ::-1]

    # ---------- 内部 ----------

    def _background(self, t: float) -> np.ndarray:
        W, H = self.w, self.h
        yy = np.linspace(0, 1, H)[:, None]
        top = np.array([0.10, 0.085, 0.075])
        bot = np.array([0.16, 0.135, 0.115])
        bg = top * (1 - yy[..., None]) + bot * yy[..., None]
        bg = np.repeat(bg, W, axis=1)
        xx = np.linspace(-1, 1, W)[None, :]
        vig = 1 - 0.22 * (xx ** 2 + (yy * 2 - 1) ** 2 * 0.5)
        return bg * vig[..., None]

    def _draw_cavities(self, canvas, P2, V, depth, mouth):
        """眼睛虹膜与口腔暗部（画在网格后面，透过孔洞可见）。"""
        for idx in self.model.eye_centers:
            c = P2[idx].mean(axis=0)
            r = np.linalg.norm(P2[idx].max(0) - P2[idx].min(0)) / 2
            cv2.ellipse(canvas, (int(c[0]), int(c[1])),
                        (max(2, int(r * 0.9)), max(2, int(r * 0.55))), 0, 0, 360,
                        (0.16, 0.10, 0.10), -1)
            cv2.circle(canvas, (int(c[0]), int(c[1])), max(2, int(r * 0.32)),
                       (0.10, 0.07, 0.08), -1)
        lp = P2[self.model.inner_idx]
        c = lp.mean(axis=0)
        rx = (lp[:, 0].max() - lp[:, 0].min()) / 2 * 0.9
        ry = max(2.0, rx * (0.08 + 0.5 * mouth))
        cv2.ellipse(canvas, (int(c[0]), int(c[1])), (max(2, int(rx)), int(ry)),
                    0, 0, 360, (0.13, 0.07, 0.08), -1)

    def _raster(self, ti, tris, P2, V, N, depth, zbuf, cbuf,
                makeup, prev_makeup, fade_w, intensity, presence):
        i0, i1, i2 = tris[ti]
        a, b, c = P2[i0], P2[i1], P2[i2]
        xmin = max(0, int(min(a[0], b[0], c[0])))
        xmax = min(self.w - 1, int(max(a[0], b[0], c[0])) + 1)
        ymin = max(0, int(min(a[1], b[1], c[1])))
        ymax = min(self.h - 1, int(max(a[1], b[1], c[1])) + 1)
        if xmin >= xmax or ymin >= ymax:
            return
        xs = np.arange(xmin, xmax + 1) + 0.5
        ys = np.arange(ymin, ymax + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)

        d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(d) < 1e-9:
            return
        w0 = ((b[0] - gx) * (c[1] - gy) - (b[1] - gy) * (c[0] - gx)) / d
        w1 = ((c[0] - gx) * (a[1] - gy) - (c[1] - gy) * (a[0] - gx)) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            return

        z = 1.0 / (w0 / depth[i0] + w1 / depth[i1] + w2 / depth[i2])
        upd = inside & (z < zbuf[ymin:ymax + 1, xmin:xmax + 1])
        if not upd.any():
            return

        # UV（透视矫正）
        u = w0 * self.model.uvs[i0, 0] / depth[i0] + w1 * self.model.uvs[i1, 0] / depth[i1] \
            + w2 * self.model.uvs[i2, 0] / depth[i2]
        v = w0 * self.model.uvs[i0, 1] / depth[i0] + w1 * self.model.uvs[i1, 1] / depth[i1] \
            + w2 * self.model.uvs[i2, 1] / depth[i2]
        u = u / (w0 / depth[i0] + w1 / depth[i1] + w2 / depth[i2])
        v = v / (w0 / depth[i0] + w1 / depth[i1] + w2 / depth[i2])

        nx = w0 * N[i0, 0] + w1 * N[i1, 0] + w2 * N[i2, 0]
        ny = w0 * N[i0, 1] + w1 * N[i1, 1] + w2 * N[i2, 1]
        nz = w0 * N[i0, 2] + w1 * N[i1, 2] + w2 * N[i2, 2]
        nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
        nx, ny, nz = nx / nl, ny / nl, nz / nl
        ndl = nx * self.L[0] + ny * self.L[1] + nz * self.L[2]
        ndh = np.clip((nx * self.H[0] + ny * self.H[1] + nz * self.H[2]), 0, 1)

        yy, xx = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
        tex_x = u * (TEX - 1)
        tex_y = (1.0 - v) * (TEX - 1)
        skin = _bilinear(self.skin, tex_x, tex_y)

        # ---- 皮肤着色（wrap diffuse 次表面近似 + 阴影缘透红）----
        wrap = 0.25
        ndl_w = np.clip((ndl + wrap) / (1 + wrap), 0, 1)
        diff = 0.35 + 0.65 * ndl_w
        sss = (1.0 - ndl_w) * ndl_w
        col = skin * (diff + 0.35 * sss)[..., None] * self.tint
        col += (sss * 0.06)[..., None] * np.array([0.10, 0.02, 0.01])
        col += (ndh ** 42)[..., None] * 0.06 * self.tint     # 皮肤本身微高光

        if makeup is not None:
            rgba, sheen_map = makeup
            mk = _bilinear(rgba, tex_x, tex_y)
            sh = _bilinear(sheen_map, tex_x, tex_y)[..., 0]
            if prev_makeup is not None and fade_w < 1:
                mkp, shp = prev_makeup
                mk = mk * fade_w + _bilinear(mkp, tex_x, tex_y) * (1 - fade_w)
                sh = sh * fade_w + _bilinear(shp, tex_x, tex_y)[..., 0] * (1 - fade_w)
            alpha = np.clip(mk[..., 3] * intensity * presence, 0, 1)
            # 妆容光影合成（与 MakeupLayer.shader 同模型）：
            #   环境色温 × wrap diffuse × sheen/clearcoat 双层高光 × 脸缘羽化
            mwrap = 0.35
            mndl = np.clip((ndl + mwrap) / (1 + mwrap), 0, 1)
            mdiff = 0.55 + 0.45 * mndl
            mcol = mk[..., :3] * self.tint * mdiff[..., None]
            sheen = fres_term(nx, ny, nz, self.view_dir) * sh * (0.15 + sh * 0.5)
            finish_gloss = np.clip(sh - 0.5, 0, 1) * 2.0
            spec = (ndh ** (60 + 60 * sh)) * finish_gloss * 0.5
            mcol += (sheen + spec)[..., None] * self.tint
            edge = _bilinear(self.edge, tex_x, tex_y)[..., 0]
            alpha = alpha * np.clip(edge, 0, 1)
            col = col * (1 - alpha[..., None]) + mcol * alpha[..., None]

        yy = yy[upd]
        xx = xx[upd]
        zbuf[yy, xx] = z[upd]
        cbuf[yy, xx] = col[upd]

    def _draw_splats(self, canvas, V, N, depth, splat_layers, intensity):
        for layer in splat_layers:
            for a in layer.get("anchors", []):
                li = a["landmark"]
                pos = V[li]
                if "toward" in a:   # 外唇线等锚点内收（见 build_splats）
                    pos = pos + (V[a["toward"]] - pos) * float(a.get("inset", 0.45))
                pos = pos + N[li] * (a.get("offset", 0.0012) / FACE_HEIGHT_M)
                d = self.d - pos[2]
                px = self.w / 2 + pos[0] * self.f / d   # 屏幕 x
                py = self.h / 2 - pos[1] * self.f / d   # 屏幕 y
                sigma = np.array(a.get("sigma", [0.0015, 0.0012])) / FACE_HEIGHT_M
                r_px = float(sigma.max() * 3 * self.f / d)
                if r_px < 1.5:
                    continue
                col = np.array(_hex_rgb(a["color"]), np.float32) * self.tint
                peak = float(a.get("alpha", 0.7)) * intensity
                half = int(r_px) + 1
                x0, x1 = int(px) - half, int(px) + half
                y0, y1 = int(py) - half, int(py) + half
                if x1 < 0 or y1 < 0 or x0 >= self.w or y0 >= self.h:
                    continue
                rows, cols = np.mgrid[y0:y1 + 1, x0:x1 + 1]
                sx = (cols - px) / (r_px / 3)   # 单位 = σ
                sy = (rows - py) / (r_px / 3)
                g = np.exp(-0.5 * (sx * sx + sy * sy)) * peak
                sl = canvas[y0:y1 + 1, x0:x1 + 1]
                sl[:] = sl * (1 - g[..., None]) + col * g[..., None]

    @property
    def L(self) -> np.ndarray:
        l = np.array(ENVS[self.env]["light"], np.float32)
        return l / np.linalg.norm(l)

    @property
    def H(self) -> np.ndarray:
        h = self.L + self.view_dir
        return h / np.linalg.norm(h)

    @property
    def tint(self) -> np.ndarray:
        return np.array(ENVS[self.env]["tint"], np.float32)

    @property
    def view_dir(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0], np.float32)


def fres_term(nx, ny, nz, view) -> np.ndarray:
    """fresnel = (1 - n·v)^3（view 为单位向量）。"""
    ndv = np.clip(nx * view[0] + ny * view[1] + nz * view[2], 0, 1)
    return (1.0 - ndv) ** 3


def _hex_rgb(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


# ---------------- splat 配置（与 bake_assets.splat_config 同规则轻量版） ----------------

SPLAT_GROUP_HINTS = {
    "lipstick": ["lips_outer", "lips_inner"],
    "highlight": None,   # 按 shape.areas 过滤
    "contour": ["contour_forehead", "contour_jaw_left", "contour_jaw_right", "contour_nose"],
    "eyeshadow": ["eyelid_left", "eyelid_right"],
    "blush": ["blush_left", "blush_right"],
    "eyebrow": ["eyebrow_left", "eyebrow_right"],
    "lashes": ["eyeliner_left", "eyeliner_right"],
}


def build_splats(layers: list[dict], lm: dict) -> list[dict]:
    """按 spec layers 生成溅射配置（render.type == splat 的层）。lm 为 landmark-regions.json 的 regions。

    锚点字段：landmark（位置关键点）、group、t（组内参数位置，取色/长轴用）、color、sigma[2]（米）、
    offset（沿法线抬升，米）、alpha；可选 toward/inset：位置 = lerp(landmark, toward, inset)，
    用于把外唇线锚点内收到唇体上，避免溅射沿轮廓外凸成锯齿。
    Unity SplatLayerRenderer 与本模块 _draw_splats 同规则解释这些字段。
    """
    out = []
    for layer in layers:
        if layer.get("render", {}).get("type") != "splat":
            continue
        region = layer["region"]
        groups = SPLAT_GROUP_HINTS.get(region)
        if region == "highlight":
            areas = set((layer.get("shape") or {}).get("areas") or ["cheek", "nose"])
            groups = []
            if "cheek" in areas:
                groups += ["highlight_cheek_left", "highlight_cheek_right"]
            if "nose" in areas:
                groups += ["highlight_nose"]
            if "cupid" in areas:
                groups += ["highlight_cupid"]
        if groups is None:
            continue
        splat = layer.get("render", {}).get("splat") or {}
        th = float(splat.get("thickness", 0.0012))
        den = float(splat.get("density", 0.7))
        anchors = []
        for g in groups:
            info = lm.get(g)
            if not info:
                continue
            idxs = info.get("indices") or ([info["center_anchor"]] if info.get("center_anchor") else [])
            n = len(idxs)
            # 外唇线锚点向同序号内唇点内收 45%，尺寸/强度略减 → 溅射落在唇体上而非轮廓外
            toward = None
            sig_k, alpha_k = 1.0, 1.0
            if region == "lipstick" and g == "lips_outer":
                toward = (lm.get("lips_inner") or {}).get("indices") or None
                sig_k, alpha_k = 0.8, 0.85
            for k, i in enumerate(idxs):
                a = {
                    "landmark": int(i),
                    "group": g,
                    "t": k / max(n - 1, 1),
                    "color": _ramp_hex(layer["color_stops"], k / max(n - 1, 1)),
                    "sigma": [round(th * 1.6 * sig_k, 6), round(th * 1.2 * sig_k, 6)],
                    "offset": th,
                    "alpha": round(min(layer.get("opacity", 0.7) * den * alpha_k, 1.0), 3),
                }
                if toward and k < len(toward):
                    a["toward"] = int(toward[k])
                    a["inset"] = 0.45
                anchors.append(a)
        if anchors:
            out.append({"id": layer["id"], "region": region, "side": layer.get("side", "both"),
                        "finish": layer.get("finish", "satin"), "anchor_count": len(anchors),
                        "opacity": layer.get("opacity", 0.7), "anchors": anchors})
    return out


def _ramp_hex(stops, t):
    pts = sorted(((float(s["at"]), s["hex"]) for s in stops), key=lambda p: p[0])
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for (a0, c0), (a1, c1) in zip(pts, pts[1:]):
        if a0 <= t <= a1:
            f = 0 if a1 == a0 else (t - a0) / (a1 - a0)
            h0, h1 = c0.lstrip("#"), c1.lstrip("#")
            return "#{:02X}{:02X}{:02X}".format(
                *(round(int(h0[i:i + 2], 16) + (int(h1[i:i + 2], 16) - int(h0[i:i + 2], 16)) * f)
                  for i in (0, 2, 4)))
    return pts[-1][1]


# ---------------- 单例（供 CLI/脚本复用，避免重复烘焙） ----------------

_renderer_cache: dict = {}


def get_renderer(w: int = 640, h: int = 538) -> FaceRenderer:
    key = (w, h)
    if key not in _renderer_cache:
        _renderer_cache[key] = FaceRenderer(w, h)
    return _renderer_cache[key]


def render_reference(spec: dict, out_path: str | Path | None = None, size: int = 640,
                     env: str = "neutral", intensity: float | None = None,
                     yaw_deg: float = 0.0) -> np.ndarray:
    """渲染目标妆容参考图（正脸静态）。返回 BGR。live_coach 参考图 / parse_look 预览共用。"""
    r = get_renderer(size, int(size * 0.84))
    r.set_env(env)
    layers = [dict(l) for l in spec.get("layers", []) if l.get("enabled", True)]
    inten = spec.get("intensity", 0.8) if intensity is None else intensity
    lm = json.loads((REFS / "landmark-regions.json").read_text(encoding="utf-8"))["regions"]
    splats = build_splats(layers, lm)
    img = r.render_still(layers, intensity=float(inten), yaw_deg=yaw_deg, splat_layers=splats)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return img
