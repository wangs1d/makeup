"""offline_render — 后台高保真渲染交付物（产品形态：离线重渲染给用户看，无 Unity）。

渲染走 gsplat——与训练同一光栅化器（EWA + SH 视角色 + antialiased），取代
numpy 圆核预览（render_pbr 只作无 GPU 兜底，保真度不足以交付）。

关键教训（实测踩坑）：SH 高阶系数是"视角分布内"的拟合——合成轨道若离开
采集相机覆盖的方向，21 万 splat 各自的 SH 残差同时外推，渲染成彩虹碎裂。
因此环绕相机两条策略：
    有 SfM 位姿（默认，推荐）→ 在真实相机位置序列的方位角范围内插值环绕，
    视线严格落在训练分布内，SH 全开；
    无位姿 → landmark/PCA 推轴 + 窄幅环绕，且渲染退化为 DC-only（不带 SH），
    任意角度稳定（损失视角相关高光，不碎）。

    deliver(ply, out_dir, sfm_dir=...) → front/left/right.png + turntable.mp4
    + compare_*.png
"""
from __future__ import annotations

import json
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ---------------- 资产加载与轴向推断 ----------------

def load_prepared(cloud: dict[str, np.ndarray], use_sh: bool = True) -> dict:
    """cloud dict → gsplat 张量（rot xyzw→wxyz；sh_rest→(n,K,3) SH 系数）。

    use_sh=False 时丢弃高阶（DC-only 渲染：视角无关颜色，合成视角下稳定）。"""
    import torch

    sh = cloud.get("sh_rest") if use_sh else None
    if sh is not None:
        deg = int(round(np.sqrt(sh.shape[1] + 1))) - 1      # pc=8→deg2, pc=3→deg1
        colors = np.zeros((len(cloud["xyz"]), (deg + 1) ** 2, 3), np.float32)
        colors[:, 0, :] = (cloud["rgba"][:, :3] - 0.5) / 0.28209479112561376
        colors[:, 1:, :] = sh
    else:
        deg = None
        colors = cloud["rgba"][:, :3].copy()
    return {
        "xyz": torch.tensor(np.ascontiguousarray(cloud["xyz"], np.float32), device="cuda"),
        "rot": torch.tensor(np.ascontiguousarray(cloud["rot"][:, [3, 0, 1, 2]], np.float32),
                            device="cuda"),                  # 内部 xyzw → gsplat wxyz
        "scale": torch.tensor(np.ascontiguousarray(cloud["scale"], np.float32), device="cuda"),
        "opacity": torch.tensor(np.ascontiguousarray(cloud["rgba"][:, 3], np.float32), device="cuda"),
        "colors": torch.tensor(np.ascontiguousarray(colors, np.float32), device="cuda"),
        "deg": deg,
    }


def load_ply(path: str | Path) -> dict[str, np.ndarray]:
    from ..splat_io import read_ply
    return read_ply(path)


# ---------------- 妆感材质（rough/coat/sss/sheen 进交付渲染） ----------------
# material.bin/light.bin 的唯一在线消费端：Unity 退役后，妆感（水光/镜面唇/
# 珠光/唇部透光）必须由离线渲染自己表达，否则 finish 维度在交付物里整体消失。
# 做法：主色渲染之外再光栅化 3 个 AOV（法线/rough-coat-sss/sheen-深度），
# 在像素级按 pbr.shade_points 同式叠加薄层高光——与训练同款光栅化器天然对齐。

def read_light_bin(path: str | Path) -> tuple[np.ndarray, float, np.ndarray]:
    """MKLT1 → (dir 世界系指向光源, strength, tint)。头部 16B + f32×7。"""
    raw = Path(path).read_bytes()
    if raw[:4] != b"MKLT":
        raise ValueError(f"light.bin magic 错误：{raw[:4]!r}")
    (n,) = struct.unpack_from("<I", raw, 8)
    body = struct.unpack_from(f"<{n * 7}f", raw, 16)
    return (np.array(body[0:3]), float(body[3]), np.array(body[4:7]))


def read_material_bin(path: str | Path) -> dict[str, np.ndarray]:
    """MKMAT1 → {normal (n,3), rough/coat/sss/sheen (n,)}。头部 16B + f32×8。"""
    raw = Path(path).read_bytes()
    if raw[:4] != b"MKMA":
        raise ValueError(f"material.bin magic 错误：{raw[:4]!r}")
    (n,) = struct.unpack_from("<I", raw, 8)
    body = np.frombuffer(raw, np.dtype("<f4"), count=n * 8, offset=16).reshape(n, 8)
    return {"normal": body[:, 0:3].copy(), "rough": body[:, 3].copy(),
            "coat": body[:, 4].copy(), "sss": body[:, 5].copy(),
            "sheen": body[:, 6].copy()}


@dataclass
class ShadeContext:
    """逐 splat 妆感材质 + 主光。cloud 自带 material 时优先，否则读 sidecar。"""
    normal: np.ndarray            # (n,3) 世界系单位
    rough: np.ndarray             # (n,)
    coat: np.ndarray
    sss: np.ndarray
    sheen: np.ndarray
    light_dir: np.ndarray         # (3,) 世界系指向光源
    tint: np.ndarray              # (3,) 光色 0..1
    strength: float = 0.9         # 合成高光全局强度（烘焙光已存在，纯叠加）
    z_scale: float = 1.0          # 深度 AOV 的归一化尺度
    _gpu: dict = field(default_factory=dict, repr=False)


def build_shade(cloud: dict[str, np.ndarray],
                mat_dir: str | Path | None = None) -> ShadeContext | None:
    """从 cloud（in-memory material）或 mat_dir 下 material.bin/light.bin 构建。

    什么都拿不到（无材质无 light.bin）时仍用皮肤基准 + 默认光——高光极弱，
    与旧渲染视觉等价，调用端无需分支。"""
    n = len(cloud["xyz"])
    from .normals import axis_normals, smooth_normals
    from .pbr import Material

    mat = cloud.get("material")
    normal = rough = coat = sss = sheen = None
    if mat is None and mat_dir is not None:
        p = Path(mat_dir) / "material.bin"
        if p.exists():
            try:
                mb = read_material_bin(p)
                if len(mb["rough"]) == n:
                    normal = mb["normal"]
                    rough, coat = mb["rough"], mb["coat"]
                    sss, sheen = mb["sss"], mb["sheen"]
            except (ValueError, OSError):
                pass
    if normal is None:
        normal = smooth_normals(np.asarray(cloud["xyz"], np.float32),
                                axis_normals(np.asarray(cloud["rot"], np.float64),
                                             np.asarray(cloud["scale"], np.float64)),
                                k=12, iters=2) if n > 64 \
            else axis_normals(np.asarray(cloud["rot"], np.float64),
                              np.asarray(cloud["scale"], np.float64))
    if rough is None:
        m = mat if mat is not None else Material.skin(n)
        rough = np.asarray(m.rough, np.float32)
        coat = np.asarray(m.coat, np.float32)
        sss = np.asarray(m.sss, np.float32)
        sheen = np.asarray(m.sheen, np.float32)

    light_dir, strength, tint = np.array([0.30, 0.55, 0.80]), 1.0, np.ones(3)
    if mat_dir is not None:
        lp = Path(mat_dir) / "light.bin"
        if lp.exists():
            try:
                light_dir, _s, tint = read_light_bin(lp)
            except (ValueError, OSError):
                pass
    nd = np.linalg.norm(light_dir)
    if nd < 1e-9:
        light_dir, nd = np.array([0.30, 0.55, 0.80]), 1.0
    center = np.median(np.asarray(cloud["xyz"], np.float64), axis=0)
    z_scale = float(np.median(np.linalg.norm(
        np.asarray(cloud["xyz"], np.float64) - center, axis=1))) + 1e-9
    return ShadeContext(
        normal=np.asarray(normal, np.float32),
        rough=np.clip(rough, 0, 1).astype(np.float32),
        coat=np.clip(coat, 0, 1).astype(np.float32),
        sss=np.clip(sss, 0, 1).astype(np.float32),
        sheen=np.clip(sheen, 0, 1).astype(np.float32),
        light_dir=light_dir / nd, tint=np.clip(tint, 0, 1),
        z_scale=z_scale)


def composite_shade(img: np.ndarray, alpha: np.ndarray, normal: np.ndarray,
                    rough: np.ndarray, coat: np.ndarray, sss: np.ndarray,
                    sheen: np.ndarray, zmap: np.ndarray, w2c: np.ndarray,
                    K: np.ndarray, light_dir: np.ndarray, tint: np.ndarray,
                    strength: float = 0.9) -> np.ndarray:
    """像素级薄层妆感合成（pbr.shade_points 的逐像素版，relight=0 纯叠加）。

    img  主色渲染 (h,w,3) float 0..1（烘焙光照已在其中）
    normal (h,w,3) 单位法线；rough/coat/sss/sheen/zmap (h,w)；背景像素由
    alpha 门控。返回 img + (唇釉清漆高光 + 掠射绒光 + sss 背光红移)。"""
    from .pbr import WRAP, rough_to_shin

    h, w = img.shape[:2]
    R, t = w2c[:3, :3], np.asarray(w2c[:3, 3], np.float64)
    eye = -R.T @ t
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0],
                      (v - K[1, 2]) / K[1, 1],
                      np.ones_like(u, np.float64)], axis=-1)
    d_world = d_cam @ R.T                                    # 相机光线 → 世界系
    P = eye + d_world * zmap[..., None]
    V = eye - P
    V /= np.linalg.norm(V, axis=-1, keepdims=True) + 1e-12
    L = np.asarray(light_dir, np.float64)
    L = L / (np.linalg.norm(L) + 1e-12)
    N = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    ndl = np.clip(N @ L, -1, 1)
    ndv = np.clip((N * V).sum(-1), 0, 1)
    H = L[None, None, :] + V
    H /= np.linalg.norm(H, axis=-1, keepdims=True) + 1e-12
    ndh = np.clip((N * H).sum(-1), 0, 1)
    hdv = np.clip((H * V).sum(-1), 0, 1)

    fac = np.clip((ndl + WRAP) / (1 + WRAP), 0, 1)
    lift = (np.clip(sss, 0, 1) * 0.30 * (1 - fac))[..., None] * \
        np.array([0.90, 0.22, 0.45])
    shin = rough_to_shin(np.clip(rough, 0.03, 1.0))
    spec = np.power(ndh, shin) * (0.028 + 0.972 * (1 - hdv) ** 5) * \
        np.clip(coat, 0, 1) * strength
    sheen_t = np.clip(sheen, 0, 1) * (1 - ndv) ** 3 * 0.35
    add = (spec + sheen_t)[..., None] * np.asarray(tint, np.float64) \
        + img * lift
    return img + add * np.clip(alpha, 0, 1)[..., None]


def _aov_raster(prepared: dict, colors_np: np.ndarray, w2c: np.ndarray,
                K: np.ndarray, big: int):
    """DC-only AOV 光栅化（与主色同光栅化器/分辨率）。返回 (h,w,3) float。"""
    import torch
    from gsplat import rasterization

    col = torch.tensor(np.ascontiguousarray(colors_np, np.float32),
                       device="cuda")
    r, _a, _i = rasterization(
        prepared["xyz"], prepared["rot"], prepared["scale"], prepared["opacity"],
        col[None], torch.tensor(w2c, dtype=torch.float32, device="cuda")[None],
        torch.tensor(K, dtype=torch.float32, device="cuda")[None], big, big,
        packed=False, backgrounds=torch.zeros(1, 3, device="cuda"))
    return r[0].cpu().numpy()


def _shade_aovs(shade: "ShadeContext", prepared: dict, w2c: np.ndarray,
                K: np.ndarray, big: int, img_f: np.ndarray, alpha: np.ndarray,
                ) -> np.ndarray:
    """3 个 AOV + composite_shade，返回合成后的 float 图。AOV 颜色数组与
    prepared 无关，lazy 建一次复用（逐帧仅 3 次光栅化 + H2D 拷贝）。"""
    if "n" not in shade._gpu:
        n = len(shade.normal)
        shade._gpu["n"] = shade.normal * 0.5 + 0.5
        shade._gpu["m"] = np.stack([shade.rough, shade.coat, shade.sss], 1)
        shade._gpu["s"] = np.stack([shade.sheen, np.full(n, 0.5, np.float32),
                                    np.zeros(n, np.float32)], 1)
    nmap = _aov_raster(prepared, shade._gpu["n"], w2c, K, big)[..., :3] * 2 - 1
    mmap = _aov_raster(prepared, shade._gpu["m"], w2c, K, big)[..., :3]
    smap = _aov_raster(prepared, shade._gpu["s"], w2c, K, big)[..., :3]
    zmap = smap[..., 1] * shade.z_scale
    return composite_shade(img_f, alpha, nmap, mmap[..., 0], mmap[..., 1],
                           mmap[..., 2], smap[..., 0], zmap, w2c, K,
                           shade.light_dir, shade.tint, shade.strength)


def infer_axes(xyz: np.ndarray, landmarks: np.ndarray | None = None) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """(center, up, front) 单位轴。地标优先（L10 额顶-L152 颏底定 up，
    朝向由 PCA 最小方差轴补足并按地标质心定向）；PCA 全回退。"""
    center = np.median(xyz, axis=0)
    if landmarks is not None and len(landmarks) >= 153:
        up = np.asarray(landmarks[10], np.float64) - np.asarray(landmarks[152], np.float64)
    else:
        up = None
    q = xyz - center
    cov = q.T @ q / max(len(q), 1)
    w, v = np.linalg.eigh(cov)                               # 升序
    front = v[:, 0]                                          # 最小方差轴 = 正背向
    if up is None:
        up = v[:, 1]
    up = up / (np.linalg.norm(up) + 1e-12)
    front = front - up * (front @ up)
    # 地标可用时用质心定向 front（鼻区 z 明显偏离脸颊质心一侧）
    if landmarks is not None and len(landmarks) >= 2:
        nose = np.asarray(landmarks[1], np.float64) - center
        nose = nose - up * (nose @ up)
        if nose @ front < 0:
            front = -front
    front = front / (np.linalg.norm(front) + 1e-12)
    return center, up, front


def face_height(xyz: np.ndarray, up: np.ndarray, center: np.ndarray) -> float:
    proj = (xyz - center) @ up
    return max(float(np.percentile(proj, 99.5) - np.percentile(proj, 0.5)), 1e-6)


def _lookat(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV 约定 look-at（x 右 y 下 z 朝场景），y 与 up 反向，图像正立。"""
    zc = target - eye
    zc = zc / (np.linalg.norm(zc) + 1e-12)
    xc = np.cross(zc, up)                                    # x = z×up（y=-up ⇒ 右手）
    xc = xc / (np.linalg.norm(xc) + 1e-12)
    yc = np.cross(zc, xc)
    w2c = np.eye(4)
    w2c[:3, :3] = np.stack([xc, yc, zc], axis=0)
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    return w2c


# ---------------- 环绕相机 ----------------

def orbit_from_poses(w2cs: list[np.ndarray], Ks: list[np.ndarray],
                     center: np.ndarray, n: int = 36, size: int = 1024,
                     yaw_pad_deg: float = 3.0, face_h: float | None = None,
                     frame_fill: float = 0.0, radius_min: float = 0.0
                     ) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """在真实采集相机的方位角范围内插值环绕（SH 安全区，推荐路径）。

    半径/焦距取真实相机中位数；sweep 覆盖真实相机方位角 [min-pad, max+pad]；
    up 取各相机 up 中位数（图像正立）。face_h 给定时沿视线收放距离，让脸高
    占画面 frame_fill 比例（视线方向不变，SH 依赖方向不依赖距离，安全）。"""
    centers, ups = [], []
    for w in w2cs:
        ups.append(-w[:3, 1])                                # 相机 y=下 → 取反
        centers.append(-w[:3, :3].T @ w[:3, 3])
    up = np.median(np.stack(ups), axis=0)
    up = up / (np.linalg.norm(up) + 1e-12)
    V = np.stack(centers) - center
    radius = float(np.median(np.linalg.norm(V, axis=1)))
    fx = float(np.median([K[0, 0] for K in Ks]))
    if face_h is not None and frame_fill > 0:
        # 取景收放：投影脸高 px = fh·fx/d → d = fh·fx/(frame_fill·size)。
        # 下限 = max(真实相机距离的 0.85 倍, radius_min)：更近会放大头发/高光
        # 大 splat（SR 帧尤其明显），是资产级噪声，渲染端不硬扛。
        d_fit = face_h * fx / (frame_fill * size)
        lo_r = max(0.85 * radius, radius_min)
        radius = float(np.clip(d_fit, lo_r, 1.3 * radius))
    v0 = V[np.linalg.norm(V, axis=1).argmin()]
    # 参考正前 = 相机方位的中位向量：若参考系指到脑后，yaw 跨 ±180 边界，
    # min/max sweep 会变成整圈 360°（定妆照落到后脑勺）
    fwd0 = np.median(V, axis=0)
    fwd0 = fwd0 / (np.linalg.norm(fwd0) + 1e-12)
    side0 = np.cross(up, fwd0)
    side0 = side0 / (np.linalg.norm(side0) + 1e-12)
    fwd = np.cross(up, side0)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    yaws = np.degrees([np.arctan2(v @ side0, v @ fwd0) for v in V])
    if float(np.abs(yaws).max()) > 90:                       # 仍跨边界：参考系翻向
        fwd, side0 = -fwd, -side0
        yaws = np.degrees([np.arctan2(v @ side0, v @ fwd0) for v in V])
    yaws = [float(y) for y in yaws]
    lo = float(np.percentile(yaws, 2)) - yaw_pad_deg         # 分位防离群位姿
    hi = float(np.percentile(yaws, 98)) + yaw_pad_deg
    if hi - lo < 2.0:                                        # 仅防退化：不外扩出
        mid = (hi + lo) / 2                                  # 训练方向分布（SH 安全）
        lo, hi = mid - 1.0, mid + 1.0
    K = np.array([[fx, 0, size / 2], [0, fx, size / 2], [0, 0, 1]], np.float64)
    cams = []
    for yaw in np.linspace(lo, hi, n):
        a = np.radians(yaw)
        eye = center + (fwd * np.cos(a) + side0 * np.sin(a)) * radius
        cams.append((_lookat(eye, center, up), K, float(yaw)))
    return cams


def orbit_synthetic(center: np.ndarray, up: np.ndarray, front: np.ndarray,
                    fh: float, n: int = 36, yaw_range_deg: float = 24.0,
                    pitch_deg: float = 6.0, fov_deg: float = 38.0, size: int = 1024
                    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """无位姿回退：窄幅合成环绕（配 DC-only 渲染——SH 会碎，见模块 docstring）。"""
    f = (size / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    d = 1.3 * (fh / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    side = np.cross(up, front)
    side = side / (np.linalg.norm(side) + 1e-12)
    cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]], np.float64)
    cams = []
    for yaw in np.linspace(-yaw_range_deg, yaw_range_deg, n):
        a = np.radians(yaw)
        eye = center + (front * np.cos(a) + side * np.sin(a)) * d * cp + up * d * sp
        cams.append((_lookat(eye, center, up), K, float(yaw)))
    return cams


# ---------------- 渲染 ----------------

def render_pose(prepared: dict, w2c: np.ndarray, K: np.ndarray, size: int = 1024,
                ssaa: int = 2, bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
                denoise: bool = True,
                return_alpha: bool = False,
                shade: "ShadeContext | None" = None):
    """单视角渲染（SSAA 超采样抗锯齿后缩回）。返回 (size,size,3) uint8 RGB；
    return_alpha=True 时附返回 (size,size) alpha（主体覆盖，用于自动裁切）。

    denoise：边缘保持滤波压低低清源资产的泼溅颗粒（交付默认开）。
    shade：妆感材质上下文（build_shade）——主色渲染后叠加像素级薄层高光
    （清漆/绒光/sss），在 denoise 之前合成，保留高光锐度。"""
    import torch
    from gsplat import rasterization

    big = size * max(int(ssaa), 1)
    Kb = K.copy()
    Kb[:2] *= big / size
    w2c_t = torch.tensor(w2c, dtype=torch.float32, device="cuda")[None]
    Kt = torch.tensor(Kb, dtype=torch.float32, device="cuda")[None]
    r, a, _i = rasterization(
        prepared["xyz"], prepared["rot"], prepared["scale"], prepared["opacity"],
        prepared["colors"], w2c_t, Kt, big, big,
        sh_degree=prepared["deg"], rasterize_mode="antialiased", packed=False,
        backgrounds=torch.tensor(bg, dtype=torch.float32, device="cuda")[None])
    alpha = a[0][..., 0].cpu().numpy()
    if shade is not None:
        img = np.clip(_shade_aovs(shade, prepared, w2c, Kb, big,
                                  r[0].cpu().numpy().astype(np.float64), alpha),
                      0, 1)
        img = (img * 255).astype(np.uint8)
    else:
        img = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)
    if denoise:
        img = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.45)
    return (img, alpha) if return_alpha else img


# ---------------- 交付物 ----------------

def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (170, 34), (255, 255, 255), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return out


def _pick_views(cams: list) -> dict[str, int]:
    """front = 最接近扫掠中点（真实相机正前方位角）的帧；left/right = 中点
    ±span/4。位姿环绕的扫掠边缘带 pad，是 SH 外推伪影区，不能当定妆照。"""
    yaws = [yaw for _, _, yaw in cams]
    lo, hi = min(yaws), max(yaws)
    mid = (lo + hi) / 2

    def nearest(target: float) -> int:
        return min(range(len(yaws)), key=lambda i: abs(yaws[i] - target))

    return {"front": nearest(mid),
            "left": nearest(mid - (hi - lo) / 4),
            "right": nearest(mid + (hi - lo) / 4)}


def face_bbox(lm: np.ndarray | None, w2c: np.ndarray, K: np.ndarray,
             size: int, margin: float = 0.14) -> tuple[int, int, int, int] | None:
    """把 3D 地标投影到相机，返回脸部 bbox (x0,y0,x1,y1)，外扩 margin。"""
    if lm is None:
        return None
    valid = lm[np.linalg.norm(lm, axis=1) > 0]
    if len(valid) < 50:
        return None
    hom = np.concatenate([valid, np.ones((len(valid), 1))], 1)
    cam = (hom @ w2c.T)[:, :3]
    ok = cam[:, 2] > 1e-6
    if ok.sum() < 50:
        return None
    px = K[0, 0] * cam[ok, 0] / cam[ok, 2] + K[0, 2]
    py = K[1, 1] * cam[ok, 1] / cam[ok, 2] + K[1, 2]
    x0, x1 = float(np.percentile(px, 1)), float(np.percentile(px, 99))
    y0, y1 = float(np.percentile(py, 1)), float(np.percentile(py, 99))
    w, h = x1 - x0, y1 - y0
    x0, x1 = x0 - margin * w, x1 + margin * w
    y0, y1 = y0 - margin * h, y1 + margin * h
    # 拉成正方形（交付画布是方的）
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2
    xi0, xi1 = int(max(0, cx - half)), int(min(size - 1, cx + half))
    yi0, yi1 = int(max(0, cy - half)), int(min(size - 1, cy + half))
    if xi1 - xi0 < 32 or yi1 - yi0 < 32:
        return None
    return xi0, yi0, xi1, yi1


def render_stills(prepared: dict, cams: list, out_dir: Path, size: int = 1024,
                  ssaa: int = 2, denoise: bool = True, lm: np.ndarray | None = None,
                  closeup: bool = True, shade: "ShadeContext | None" = None) -> list[Path]:
    """前/左/右三张定妆照。

    closeup=True（默认）：在 SSAA 大图缓冲内按主体（alpha）bbox 裁脸再缩到
    size——特写走的是超采样原生分辨率，避免二次缩放糊掉；全帧另存为
    still_*_full.png。"""
    import torch
    from gsplat import rasterization

    big = size * max(int(ssaa), 1)
    outs = []
    for name, idx in _pick_views(cams).items():
        w2c, K, _yaw = cams[idx]
        Kb = K.copy()
        Kb[:2] *= big / size
        r, a, _i = rasterization(
            prepared["xyz"], prepared["rot"], prepared["scale"], prepared["opacity"],
            prepared["colors"], torch.tensor(w2c, dtype=torch.float32, device="cuda")[None],
            torch.tensor(Kb, dtype=torch.float32, device="cuda")[None], big, big,
            sh_degree=prepared["deg"], rasterize_mode="antialiased", packed=False,
            backgrounds=torch.ones(1, 3, device="cuda"))
        a_big = a[0][..., 0].cpu().numpy()
        if shade is not None:
            img_big = np.clip(
                _shade_aovs(shade, prepared, w2c, Kb, big,
                            r[0].cpu().numpy().astype(np.float64), a_big),
                0, 1)
            img_big = (img_big * 255).astype(np.uint8)
        else:
            img_big = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        full = cv2.resize(img_big, (size, size), interpolation=cv2.INTER_AREA)
        if denoise:
            full = cv2.edgePreservingFilter(full, flags=1, sigma_s=60, sigma_r=0.45)
        pf = out_dir / f"still_{name}_full.png"
        cv2.imwrite(str(pf), cv2.cvtColor(_label(full, name), cv2.COLOR_RGB2BGR))
        outs.append(pf)

        if closeup:
            ys, xs = np.nonzero(a_big > 0.3)
            if len(xs) >= 200:
                x0, x1 = np.percentile(xs, [2, 98]).astype(int)
                y0, y1 = np.percentile(ys, [2, 98]).astype(int)
                w_, h_ = x1 - x0, y1 - y0
                x0, x1 = max(0, x0 - w_ // 10), min(big, x1 + w_ // 10)
                y0, y1 = max(0, y0 - h_ // 10), min(big, y1 + h_ // 10)
                side = max(x1 - x0, y1 - y0)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                x0 = max(0, cx - side // 2); x1 = min(big, x0 + side)
                y0 = max(0, cy - side // 2); y1 = min(big, y0 + side)
                crop = cv2.resize(img_big[y0:y1, x0:x1], (size, size),
                                  interpolation=cv2.INTER_AREA)
                if denoise:
                    crop = cv2.edgePreservingFilter(crop, flags=1, sigma_s=100, sigma_r=0.5)
                    crop = cv2.GaussianBlur(crop, (3, 3), 0.8)
                pc = out_dir / f"still_{name}_closeup.png"
                cv2.imwrite(str(pc), cv2.cvtColor(_label(crop, name), cv2.COLOR_RGB2BGR))
                outs.append(pc)
    return outs


def render_turntable(prepared: dict, cams: list, out_dir: Path, size: int = 1024,
                     ssaa: int = 2, fps: int = 30, denoise: bool = True,
                     shade: "ShadeContext | None" = None,
                     progress: Callable[[float, str], None] | None = None) -> Path:
    """环绕视频：帧序列 + MP4（ping-pong 回放更自然，帧数翻倍）。"""
    frames: list[np.ndarray] = []
    cb = progress or (lambda f, m: None)
    for i, (w2c, K, _yaw) in enumerate(cams):
        frames.append(render_pose(prepared, w2c, K, size=size, ssaa=ssaa,
                                  denoise=denoise, shade=shade))
        cb((i + 1) / len(cams) / 2, f"turntable {i + 1}/{len(cams)}")
    pingpong = frames + frames[-2:0:-1]
    mp4 = out_dir / "turntable.mp4"
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (size, size))
    try:
        for j, fr in enumerate(pingpong):
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            if j % 12 == 0:
                cb(0.5 + j / len(pingpong) / 2, f"encode {j}/{len(pingpong)}")
    finally:
        vw.release()
    return mp4


def render_compare(base: dict, made: dict, cams: list, out_dir: Path,
                   size: int = 1024, ssaa: int = 2, use_sh: bool = True,
                   denoise: bool = True,
                   shade: "ShadeContext | None" = None,
                   shade_base: "ShadeContext | None" = None) -> list[Path]:
    """素颜|妆后 并排对比（同一相机，逐像素可比）。"""
    pb, pm = load_prepared(base, use_sh=use_sh), load_prepared(made, use_sh=use_sh)
    outs = []
    for name, idx in _pick_views(cams).items():
        w2c, K, _yaw = cams[idx]
        b = _label(render_pose(pb, w2c, K, size=size, ssaa=ssaa, denoise=denoise,
                               shade=shade_base), "bare")
        m = _label(render_pose(pm, w2c, K, size=size, ssaa=ssaa, denoise=denoise,
                               shade=shade), "madeup")
        p = out_dir / f"compare_{name}.png"
        cv2.imwrite(str(p), cv2.cvtColor(np.concatenate([b, m], axis=1), cv2.COLOR_RGB2BGR))
        outs.append(p)
    return outs


def deliver(ply: str | Path, out_dir: str | Path, base_ply: str | Path | None = None,
            landmarks_path: str | Path | None = None, sfm_dir: str | Path | None = None,
            n_frames: int = 36, size: int = 1024, ssaa: int = 3, fps: int = 30,
            frame_fill: float = 0.45, denoise: bool = True,
            progress: Callable[[float, str], None] | None = None) -> dict:
    """一键交付：已有资产 → 妆后定妆照 + 环绕视频（+ 素颜对比）。

    sfm_dir（COLMAP sparse）给定时走真实位姿环绕 + SH 全开（推荐）；否则
    landmark/PCA 推轴窄幅环绕 + DC-only 渲染（SH 视角外推会碎，见 docstring）。
    要求 CUDA + gsplat（训练同款光栅化器）。"""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("离线高保真渲染需要 CUDA（gsplat 光栅化器）")
    except ImportError as e:
        raise RuntimeError(f"渲染依赖缺失：{e}") from e

    cb = progress or (lambda f, m: None)
    ply = Path(ply)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lm = None
    if landmarks_path is None:
        cand = ply.parent / "landmarks.npy"
        landmarks_path = cand if cand.exists() else None
    if landmarks_path is not None:
        lm = np.load(landmarks_path)

    made = load_ply(ply)
    # 取景中心 = splat 中位数。三角化 3D 地标在本数据上整体不可靠（中位离云
    # 70+ 单位），不能当取景中心；landmark 仅用于 infer_axes 的 up 轴。
    center, up, front = infer_axes(made["xyz"], lm)
    use_sh = sfm_dir is not None
    if use_sh:
        # 真实位姿环绕：视线落在训练分布内，SH 视角相关外观安全
        from .. import colmap_io
        model = colmap_io.read_sparse(sfm_dir)
        w2cs, Ks = [], []
        for name in sorted(model.images):
            im = model.images[name]
            R = colmap_io.quat_to_rotmat(im["qvec"])
            t = np.asarray(im["tvec"], np.float64)
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t
            w2cs.append(w2c)
            Ks.append(np.array([[model.camera.params[0], 0, model.camera.params[1]],
                                [0, model.camera.params[0], model.camera.params[2]],
                                [0, 0, 1]], np.float64))
        q = made["xyz"] - center
        front_extent = float(np.percentile(q @ front, 99.5))   # 脸前表面伸出量
        fh_lm = face_height(made["xyz"], up, center)
        # 取景中心 = 相机光线最小二乘汇聚点：拍摄时相机对着脸，光线交点即脸
        # （splat 中位数会被头发拉偏；landmark 三角化在本数据上不可靠）
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for w in w2cs:
            d = w[2, :3]
            o = -w[:3, :3].T @ w[:3, 3]
            P = np.eye(3) - np.outer(d, d)
            A += P
            b += P @ o
        center = np.linalg.solve(A + 1e-9 * np.eye(3), b)
        cams = orbit_from_poses(w2cs, Ks, center, n=n_frames, size=size,
                                face_h=fh_lm, frame_fill=frame_fill,
                                radius_min=front_extent + 0.35 * fh_lm)
    else:
        cams = orbit_synthetic(center, up, front,
                               face_height(made["xyz"], up, center),
                               n=n_frames, size=size)
    prepared = load_prepared(made, use_sh=use_sh)
    # 妆感材质进交付：material.bin/light.bin 在 ply 旁时构建着色上下文
    shade = build_shade(made, mat_dir=ply.parent)

    cb(0.05, "定妆照…")
    stills = render_stills(prepared, cams, out_dir, size=size, ssaa=ssaa,
                           denoise=denoise, lm=lm, shade=shade)
    cb(0.35, "环绕视频…")
    mp4 = render_turntable(prepared, cams, out_dir, size=size, ssaa=ssaa, fps=fps,
                           denoise=denoise, shade=shade, progress=cb)
    outs = {"stills": [str(p) for p in stills], "turntable": str(mp4), "sh": use_sh,
            "shade": shade is not None}
    if base_ply is not None and Path(base_ply).exists():
        cb(0.9, "素颜对比…")
        base_cloud = load_ply(base_ply)
        cmps = render_compare(base_cloud, made, cams, out_dir,
                              size=size, ssaa=ssaa, use_sh=use_sh, denoise=denoise,
                              shade=shade, shade_base=build_shade(
                                  base_cloud, mat_dir=Path(base_ply).parent))
        outs["compare"] = [str(p) for p in cmps]
    (out_dir / "renders.json").write_text(
        json.dumps(outs, ensure_ascii=False, indent=1), encoding="utf-8")
    cb(1.0, f"完成 → {out_dir}")
    return outs
