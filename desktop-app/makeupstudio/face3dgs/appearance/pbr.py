"""pbr — 化妆品材质着色（Python 参考实现；WebGL/Unity 端按同式移植）。

化妆品的真实感来自"薄层材质变更"，不是 albedo 替换：
    rough   粉状↑（粉底/腮红/眼影哑光），釉状↓（唇釉/眼线）——高光宽度
    coat    清漆层（唇釉/珠光的"水光"），独立 Schlick Fresnel，随视角流动
    sss     唇部次表面：背光侧透光红移（真实嘴唇 vs 涂漆的分界线）
    sheen   掠射角绒光（珠光高光扫）
与旧版 blinn_phong_spec 的本质区别：高光强度/宽度由 rough×coat 两个物理量
决定并逐点携带，而不是一个全局 gloss 标量。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

F0_SKIN = 0.028          # 皮肤/清漆界面 Fresnel 基准
WRAP = 0.25              # 包裹漫反射（与内核一致）


@dataclass
class Material:
    """逐点材质通道 (n,) float32，全部 0..1。"""
    rough: np.ndarray            # 表面粗糙度（微面）
    coat: np.ndarray             # 清漆/釉层强度
    sss: np.ndarray              # 次表面散射（唇）
    sheen: np.ndarray            # 绒光（珠光）

    @staticmethod
    def skin(n: int) -> "Material":
        """素颜皮肤基准：中等粗糙、极弱清漆。"""
        return Material(np.full(n, 0.52, np.float32), np.full(n, 0.06, np.float32),
                        np.zeros(n, np.float32), np.zeros(n, np.float32))


def rough_to_shin(rough: np.ndarray) -> np.ndarray:
    """粗糙度 → Blinn 指数（energy 近似映射，rough↓ → 高光更窄更亮）。"""
    r = np.clip(np.asarray(rough, np.float64), 0.03, 1.0)
    return np.clip(2.0 / (r ** 4 + 1e-4), 8.0, 1200.0)


def shade_points(rgba: np.ndarray, normal: np.ndarray, view: np.ndarray,
                 light: np.ndarray, tint: np.ndarray, mat: Material,
                 spec_strength: float = 1.0, ambient_floor: float = 0.30,
                 relight: float = 0.0,
                 ) -> np.ndarray:
    """逐点 PBR 化妆品着色。全部输入 numpy (n,·)，返回 (n,3) 0..1。

    rgba    基础色（已含烘焙光照的 albedo；漫反射只做轻校正避免双重光照）
    normal  (n,3) 单位法线；view/light (3,) 世界系（light 指向光源）
    tint    (3,) 光色
    relight 合成漫反射重打光量 0..1。烘焙资产（光度训练产物）颜色里已含真实
    光照，默认 0 = 纯叠加（只加 spec/sheen/sss）；程序化模板资产无烘焙光照时
    可开到 1 走完整 wrap diffuse。
    """
    rgb = np.clip(np.asarray(rgba, np.float64)[:, :3], 0, 1)
    n = rgb.shape[0]
    Nn = np.asarray(normal, np.float64)
    Nn = Nn / (np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12)
    V = np.asarray(view, np.float64)
    V = V / (np.linalg.norm(V) + 1e-12)
    L = np.asarray(light, np.float64)
    L = L / (np.linalg.norm(L) + 1e-12)
    tint = np.asarray(tint, np.float64)
    rough = np.clip(np.asarray(mat.rough, np.float64), 0.03, 1.0)
    coat = np.clip(np.asarray(mat.coat, np.float64), 0.0, 1.0)
    sss = np.clip(np.asarray(mat.sss, np.float64), 0.0, 1.0)
    sheen = np.clip(np.asarray(mat.sheen, np.float64), 0.0, 1.0)

    ndl = np.clip(Nn @ L, -1, 1)
    ndv = np.clip(Nn @ V, 0, 1)
    H = L[None, :] + V[None, :]
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-12
    ndh = np.clip((Nn * H).sum(1), 0, 1)
    hdv = np.clip(H @ V, 0, 1)

    # 1) 漫反射：wrap 校正 + 唇部 SSS（背光侧透光，红移偏置）
    fac = np.clip((ndl + WRAP) / (1 + WRAP), 0, 1)
    wrap_mix = (1.0 - relight) + relight * (ambient_floor + (1 - ambient_floor) * fac)
    lift = (sss * 0.30 * (1 - fac))[..., None] * np.array([0.90, 0.22, 0.45])
    col = rgb * wrap_mix[..., None] * tint[None, :] + rgb * lift

    # 2) 清漆高光：Blinn(rough) × Schlick(hdv)，随视角流动
    shin = rough_to_shin(rough)
    spec_blinn = ndh ** shin                              # (n,) 逐点指数
    fres = F0_SKIN + (1 - F0_SKIN) * (1 - hdv) ** 5
    spec = (spec_blinn * fres * coat * spec_strength)[:, None]

    # 3) 掠射绒光（珠光）
    sheen_t = (sheen * (1 - ndv) ** 3)[:, None] * 0.35

    out = col + (spec + sheen_t) * tint[None, :]
    return np.clip(out, 0, 1)
