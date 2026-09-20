"""makeup_pack — 妆容 UV 图集（pack）：跨用户可移植的妆容资产 + 逐像素合成。

壳层路径（makeup_uv.build_makeup_layer）把妆容烘成逐 splat 常数色，有两个
结构性上限：①边缘锐度 = splat 足迹（2048² 目标场只在 splat 中心被采样一次）；
②线性 alpha 合成对浓妆透光估计偏高（真实颜料是 Beer-Lambert 指数吸收，
chroma 过冲补偿 hack 是同根因症状）。本模块把上妆从"splat 中心采样"推进到
"逐像素采样"，壳层高斯从"颜色载体"退化为导出形态的几何载体：

    compose_pack   bake 出的 UV 目标场 + 3D 锚定带（per-splat 权重光栅化回
                   UV）→ 单一 2048² 贴图包。"涂在哪"来自 3D 锚定（与壳层同一
                   _assignment 语义），"涂什么色"来自 UV 场（保 2048² 细节）。
    save/load      npz(f16)。preset 级 pack 不含用户绑定（跨用户复用，用户
                   差异由逐像素 lab_adapt 自动适应）；用户级 pack 附带
                   uv/valid 绑定，离线渲染直接消费。
    composite_makeup_pixel   渲染端逐像素合成：像素 UV 采样 pack → Lab 迁移
                   （迁移对象是渲染出的皮肤像素——纹理/光影自动全保留，不再
                   需要 hp 高通近似）→ Beer-Lambert 薄层吸收合成。线性 alpha
                   模式保留做 A/B。纯 numpy，无 torch/gsplat 依赖，可单测。

渲染端消费见 offline_render.render_pose_pixel（UV AOV = (u, v, valid) 三通道
DC 光栅化，与主色/材质 AOV 同一光栅化器）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PACK_VERSION = 1
PACK_NAME = "makeup_maps.npz"
FIELD_KEYS = ("kL", "chroma", "rough", "coat", "sss", "sheen")

# Beer-Lambert 吸收系数：T = exp(-σ·w)。σ=2 时 w=1 → T≈0.135（与旧壳层
# opacity≈0.85 的满涂透光几乎一致），w=0.3 → T≈0.55（多层薄涂语义：淡涂
# 明显更透，浓涂饱和更快——线性 alpha 在全浓度段的欠饱和即被修正）。
DEFAULT_SIGMA = 2.0
# 线性 A/B 模式的满涂不透明度（对齐旧壳层 opacity = w × 底模 ≈ 0.92）
LINEAR_OPACITY = 0.92

# 逐像素合成的门限：valid AOV 占比 / 主体覆盖 / 最小妆权重
VALID_THRESH = 0.5
MIN_ALPHA = 0.25
MIN_W = 0.004


# ---------------- 双线性采样（UV 场 → 任意点/像素） ----------------

def sample_uv(field: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(t,t[,C]) 场在 uv∈[0,1]² 处的双线性采样。行 = (1-v)·(t-1)，与
    _assignment/bind_uv 的 texel 行约定一致。u/v 形状任意。"""
    t = field.shape[0]
    tx = np.clip(np.asarray(u, np.float64) * (t - 1), 0.0, t - 1.001)
    ty = np.clip((1.0 - np.asarray(v, np.float64)) * (t - 1), 0.0, t - 1.001)
    x0 = tx.astype(np.int32)
    y0 = ty.astype(np.int32)
    fx = (tx - x0)[..., None]
    fy = (ty - y0)[..., None]
    f = field if field.ndim == 3 else field[..., None]
    x1 = np.minimum(x0 + 1, t - 1)
    y1 = np.minimum(y0 + 1, t - 1)
    a = f[y0, x0]
    b = f[y0, x1]
    c = f[y1, x0]
    d = f[y1, x1]
    out = (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
           + c * (1 - fx) * fy + d * fx * fy)
    return out[..., 0] if field.ndim == 2 else out


def splat_field(values: np.ndarray, tx: np.ndarray, ty: np.ndarray, tex: int,
                sigma: float | None = None, max_sigma: float = 8.0
                ) -> np.ndarray:
    """逐 splat 标量 → (tex,tex) 场：max 池化 + 密度归一化模糊。

    splat 散点是冲激序列且在 UV 空间成簇（实测 159k splats 只覆盖 47k
    texel），σ 必须显式盖过簇间空隙：缺省 σ=tex/342（2048 下 =6 texel
    ≈0.44mm，与壳层 splat 足迹同量级）。blur(v·1)/blur(1) 恢复局部均值
    语义；无覆盖 texel 保持 0。"""
    tyi = np.asarray(ty, np.float64).astype(np.int64)
    txi = np.asarray(tx, np.float64).astype(np.int64)
    grid = np.zeros((tex, tex), np.float32)
    cov = np.zeros((tex, tex), np.float32)
    np.maximum.at(grid, (tyi, txi), np.asarray(values, np.float32))
    np.add.at(cov, (tyi, txi), 1.0)
    sig = float(sigma if sigma is not None else max(1.2, tex / 342.0))
    sig = min(sig, max_sigma)
    vb = cv2.GaussianBlur(grid, (0, 0), sig)
    cb = cv2.GaussianBlur(cov, (0, 0), sig)
    field = np.where(cb > 0.02, vb / np.maximum(cb, 1e-6), 0.0).astype(np.float32)
    return cv2.GaussianBlur(field, (0, 0), sig * 0.5)


def avg_field(values: np.ndarray, weight: np.ndarray, tx: np.ndarray,
              ty: np.ndarray, tex: int, sigma: float | None = None
              ) -> np.ndarray:
    """加权平均散点场：Σblur(v·w)/Σblur(w)——门控/渐变坐标等"逐 splat 平滑
    量"的正确聚合（max 池化会破坏幅值语义）。无覆盖处返回 0。"""
    tyi = np.asarray(ty, np.float64).astype(np.int64)
    txi = np.asarray(tx, np.float64).astype(np.int64)
    sig = float(sigma if sigma is not None else max(1.2, tex / 342.0))
    num = np.zeros((tex, tex), np.float32)
    den = np.zeros((tex, tex), np.float32)
    np.add.at(num, (tyi, txi), np.asarray(values, np.float32)
              * np.asarray(weight, np.float32))
    np.add.at(den, (tyi, txi), np.asarray(weight, np.float32))
    nb = cv2.GaussianBlur(num, (0, 0), sig)
    db = cv2.GaussianBlur(den, (0, 0), sig)
    return np.where(db > 1e-3, nb / np.maximum(db, 1e-6), 0.0).astype(np.float32)


# ---------------- pack 合成（pipeline 用，带 3D 锚定） ----------------

def _build_slots(maps, ordered_layers: list[dict], t: int) -> tuple[list, list, list, list]:
    """按上妆顺序把活跃层填入 K 个槽位（每 texel 最多 MAX_SLOTS 层）。

    ordered_layers: [{w (t,t), tgt (t,t,3), kL float, chroma float}]（spec
    顺序，唇最外）。每 texel 的第 k 活跃层进槽 k——渲染端逐槽"迁移-合成"
        c = c·T_k + a·(1-T_k)·migrate(c, tgt_k, kL_k, chroma_k)
    即真实上妆的层链语义（腮红迁移自底妆修正后的肤色，而非素颜）。"""
    slot_w: list = []
    slot_albedo: list = []
    slot_kL: list = []
    slot_chroma: list = []
    occupied = [np.zeros((t, t), bool) for _ in range(MAX_SLOTS)]

    def _ensure(k: int) -> None:
        while len(slot_w) <= k:
            slot_w.append(np.zeros((t, t), np.float32))
            slot_albedo.append(np.zeros((t, t, 3), np.float32))
            slot_kL.append(np.zeros((t, t), np.float32))
            slot_chroma.append(np.zeros((t, t), np.float32))

    for layer in ordered_layers:
        remaining = np.asarray(layer["w"], np.float32) > 0.02
        if not remaining.any():
            continue
        for k in range(MAX_SLOTS):
            take = remaining & ~occupied[k]
            if not take.any():
                continue
            _ensure(k)
            slot_w[k][take] = layer["w"][take]
            slot_albedo[k][take] = layer["tgt"][take]
            slot_kL[k][take] = float(layer["kL"])
            slot_chroma[k][take] = float(layer["chroma"])
            occupied[k] |= take
            remaining = remaining & ~take
        # 超出 MAX_SLOTS 的深层丢弃（当前 spec 结构下实测 ≤3，防御性截断）
    return slot_w, slot_albedo, slot_kL, slot_chroma


def compose_pack(baker, cloud: dict[str, np.ndarray], maps, uv: np.ndarray,
                 valid: np.ndarray, lip3d=None, near=None, bands3d=None,
                 sigma: float = DEFAULT_SIGMA) -> "MakeupPack":
    """bake 目标场 + per-splat 指派 → 融合 pack。

    baker: UvMakeupBaker（复用其 _assignment——唇/眼线 3D 锚定、near 软门控、
    UV 兜底与壳层路径完全同源）。权重场（w/唇区）由 per-splat 指派 max 池化
    回 UV（"涂在哪"以 3D 锚定为准），颜色/材质场直接取 2048² UV 目标场
    （"涂什么色"保贴图分辨率；眼线带"带只改权重、颜色取 UV 场"同语义）。
    maps.tex 必须 ≥256：core.set_texture_size 有 max(256,·) 下限钳制，更小
    的请求会让 regions.bake 返回 256² 场与目标场错位。"""
    from ..fit_makeup import _load_core
    from .makeup_uv import FINISH_TARGET, PHOTOREAL_LAB, SKIN_ROUGH

    tex = maps.tex
    if tex < 256:
        raise ValueError(f"maps.tex={tex} 低于 core 纹理下限 256，"
                         "请用 tex≥256 的 UvMakeupBaker")
    core = _load_core()
    prev_tex = core.TEX
    core.set_texture_size(tex)
    try:
        a = baker._assignment(cloud, maps, np.asarray(uv, np.float64),
                              np.asarray(valid, bool), core, lip3d, near, bands3d)
    finally:
        core.set_texture_size(prev_tex)

    t = tex
    uv_c = np.clip(np.asarray(uv, np.float64), 0.0, 1.0)
    tx = (uv_c[:, 0] * (t - 1))
    ty = ((1.0 - uv_c[:, 1]) * (t - 1))
    sig = max(1.2, t / 342.0)          # 2048 下 6 texel ≈ splat 足迹量级
    # ---- 场构造原则：能用"本来就平滑的 UV 目标场"绝不重散点 ----
    # 全局区域（底妆/腮红/眼影）= maps.w × 门控场（门控是唯一 per-splat 3D
    # 信息，avg 散点平滑化）；3D 锚定带（唇/眼线/眉）权重才走 splat_field。
    # （点散点直接重构全场会产生椒盐/大理石伪影——splat 在 UV 空间成簇，
    # 159k splats 仅覆盖 47k texel，实测教训。）
    gate_f = avg_field(a["gate"], np.ones(len(a["gate"])), tx, ty, t, sig)
    w_global = np.clip(maps.w * np.clip(gate_f, 0.0, 1.0), 0.0, 1.0)
    w_3d = splat_field(a["w"], tx, ty, t, sigma=sig)
    if lip3d is not None and maps.lip_w is not None:
        # 3D 锚定唇带：权重散点 + 渐变坐标加权平均 → 色带在 UV 空间重建
        lip_mask = (a["lip_zone"].astype(np.float32) if a["lip_zone"] is not None
                    else np.zeros(len(a["w"]), np.float32))
        w_lip_f = splat_field(np.where(lip_mask > 0.5, a["w"], 0.0),
                              tx, ty, t, sigma=sig)
        cent_f = avg_field(a["cent"], lip_mask, tx, ty, t, sig)
        lip_tgt_f = np.clip(core.sample_ramp(maps.lip_stops,
                                             np.clip(cent_f, 0, 1)), 0, 1)
        lip_f = w_lip_f > 0.02
    elif maps.lip_w is not None:
        # UV 兜底唇带：本来就是平滑场，直接用
        w_lip_f = maps.lip_w
        lip_f = maps.lip_w > 0.02
        lip_tgt_f = maps.lip_albedo if maps.lip_albedo is not None else maps.albedo
    else:
        w_lip_f = np.zeros((t, t), np.float32)
        lip_f = np.zeros((t, t), bool)
        lip_tgt_f = maps.albedo
    w_f = np.max(np.stack([w_global, w_3d,
                           np.where(lip_f, w_lip_f, 0.0)]), axis=0)

    if maps.lip_albedo is not None or lip3d is not None:
        albedo = np.where(lip_f[..., None], lip_tgt_f, maps.albedo)
    else:
        albedo = maps.albedo
    lip_kL, lip_ch = PHOTOREAL_LAB["lipstick"]
    kL = np.where(lip_f, lip_kL, maps.kL).astype(np.float32)
    chroma = np.where(lip_f, lip_ch, maps.chroma).astype(np.float32)

    # ---- 序列合成槽位：spec 层序（bake.layer_fields）+ 唇（最外） ----
    ordered = list(getattr(maps, "layer_fields", []) or [])
    ordered.append({"region": "lipstick", "w": np.where(lip_f, w_lip_f, 0.0),
                    "tgt": lip_tgt_f, "kL": lip_kL, "chroma": lip_ch})
    slot_w, slot_albedo, slot_kL, slot_chroma = _build_slots(maps, ordered, t)

    # 唇区材质 = finish 全强度（与 _assignment 壳层分支同式）；其余区域保
    # UV 通道并做 0 值皮肤兜底（0 是合法低值，不能被兜底吞掉——只在原始
    # 通道未写入的 texel 兜底，语义同 _assignment）
    rough_t, coat_t, sheen_t = FINISH_TARGET.get(
        getattr(maps, "lip_finish", "gloss") or "gloss", FINISH_TARGET["gloss"])
    ch = maps.channels
    rough = np.where(lip_f, rough_t, np.where(ch["rough"] > 0.02, ch["rough"],
                                              SKIN_ROUGH)).astype(np.float32)
    coat = np.where(lip_f, coat_t, np.where(ch["coat"] > 0.005, ch["coat"],
                                            0.06)).astype(np.float32)
    sss = np.where(lip_f, 1.0, ch["sss"]).astype(np.float32)
    sheen = np.where(lip_f, sheen_t, ch["sheen"]).astype(np.float32)
    # 妆区外（w≈0）材质不参与合成（coverage 门控），但仍烘进 pack 保证
    # 自描述；w=0 区域的 albedo 同理。
    pack = MakeupPack(tex=tex, w=w_f, albedo=albedo.astype(np.float32),
                      kL=kL, chroma=chroma, rough=rough, coat=coat,
                      sss=sss, sheen=sheen, sigma=float(sigma),
                      slot_w=slot_w, slot_albedo=slot_albedo,
                      slot_kL=slot_kL, slot_chroma=slot_chroma)
    pack.lip_zone = lip_f
    return pack


# ---------------- pack 存取 ----------------

MAX_SLOTS = 3          # 每 texel 最大叠加层数（底妆+特征层+高光/唇，实测够）


@dataclass
class MakeupPack:
    """妆容 UV 图集：2048²（或指定 tex）的 (w, albedo, kL, chroma, 材质×4)。

    slot_* 为**序列合成槽位**（真实上妆顺序：底妆先改肤色、腮红叠加其上、
    高光最外）——每 texel 按 spec 顺序把活跃层依次填入槽位，渲染端逐槽
    迁移-合成（单层融合场无法表达"腮红迁移自底妆修正后的肤色"）。
    融合字段（w/albedo/kL/chroma）保留：诊断、跨用户壳层导出
    （apply_pack_to_cloud）与旧单层合成路径仍消费。
    uv/valid 为用户绑定（可空）：preset 级 pack 不带（跨用户复用），用户级
    pack 带上后 offline_render 像素路径可直接消费。"""
    tex: int
    w: np.ndarray
    albedo: np.ndarray
    kL: np.ndarray
    chroma: np.ndarray
    rough: np.ndarray
    coat: np.ndarray
    sss: np.ndarray
    sheen: np.ndarray
    sigma: float = DEFAULT_SIGMA
    uv: np.ndarray | None = None
    valid: np.ndarray | None = None
    lip_zone: np.ndarray | None = field(default=None, repr=False)
    slot_w: list = field(default_factory=list)          # [K](t,t)
    slot_albedo: list = field(default_factory=list)     # [K](t,t,3)
    slot_kL: list = field(default_factory=list)         # [K](t,t)
    slot_chroma: list = field(default_factory=list)     # [K](t,t)
    even_gain: float = 0.65              # 底妆匀肤强度（彩点/肤色斑驳抑制）

    @property
    def n_slots(self) -> int:
        return len(self.slot_w)

    def fields(self) -> dict[str, np.ndarray]:
        return {"w": self.w, "albedo": self.albedo, "kL": self.kL,
                "chroma": self.chroma, "rough": self.rough, "coat": self.coat,
                "sss": self.sss, "sheen": self.sheen}

    def with_binding(self, uv: np.ndarray, valid: np.ndarray) -> "MakeupPack":
        """preset pack + 用户绑定 → 用户 pack（跨用户复用的组装步）。"""
        uv = np.asarray(uv, np.float64)[:, :2]
        valid = np.asarray(valid, bool)
        if len(uv) != len(valid):
            raise ValueError(f"uv/valid 数量不一致：{len(uv)} vs {len(valid)}")
        import dataclasses
        return dataclasses.replace(self, uv=uv, valid=valid)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        payload: dict[str, np.ndarray] = {
            "w": self.w.astype(np.float16), "albedo": self.albedo.astype(np.float16),
            "kL": self.kL.astype(np.float16), "chroma": self.chroma.astype(np.float16),
            "rough": self.rough.astype(np.float16), "coat": self.coat.astype(np.float16),
            "sss": self.sss.astype(np.float16), "sheen": self.sheen.astype(np.float16),
        }
        for k, arr in enumerate(self.slot_w):
            payload[f"slot{k}_w"] = arr.astype(np.float16)
            payload[f"slot{k}_albedo"] = self.slot_albedo[k].astype(np.float16)
            payload[f"slot{k}_kL"] = self.slot_kL[k].astype(np.float16)
            payload[f"slot{k}_chroma"] = self.slot_chroma[k].astype(np.float16)
        if self.lip_zone is not None:
            payload["lip_zone"] = self.lip_zone
        if self.uv is not None:
            payload["uv"] = self.uv.astype(np.float32)
            payload["valid"] = self.valid.astype(bool)
        meta = np.array([PACK_VERSION, self.tex, float(self.sigma),
                         float(self.even_gain), self.n_slots])
        np.savez_compressed(path, _meta=meta, **payload)
        return path

    @staticmethod
    def load(path: str | Path) -> "MakeupPack":
        raw = np.load(Path(path), allow_pickle=False)
        meta = [float(x) for x in raw["_meta"]]
        version, tex, sigma = meta[0], int(meta[1]), meta[2]
        even_gain = meta[3] if len(meta) > 3 else 0.65
        pack = MakeupPack(
            tex=tex, sigma=sigma, even_gain=even_gain,
            w=raw["w"].astype(np.float32), albedo=raw["albedo"].astype(np.float32),
            kL=raw["kL"].astype(np.float32), chroma=raw["chroma"].astype(np.float32),
            rough=raw["rough"].astype(np.float32), coat=raw["coat"].astype(np.float32),
            sss=raw["sss"].astype(np.float32), sheen=raw["sheen"].astype(np.float32),
            uv=raw["uv"].astype(np.float64) if "uv" in raw.files else None,
            valid=raw["valid"].astype(bool) if "valid" in raw.files else None,
            lip_zone=raw["lip_zone"] if "lip_zone" in raw.files else None)
        k = 0
        while f"slot{k}_w" in raw.files:
            pack.slot_w.append(raw[f"slot{k}_w"].astype(np.float32))
            pack.slot_albedo.append(raw[f"slot{k}_albedo"].astype(np.float32))
            pack.slot_kL.append(raw[f"slot{k}_kL"].astype(np.float32))
            pack.slot_chroma.append(raw[f"slot{k}_chroma"].astype(np.float32))
            k += 1
        return pack


def bake_preset_pack(spec: dict, tex: int = 2048, intensity: float = 1.0,
                     sigma: float = DEFAULT_SIGMA) -> "MakeupPack":
    """spec → canonical pack（无用户绑定、无 3D 锚定——preset 级资产）。

    跨用户复用的正解：妆效烘焙一次成图集，换用户只需重绑 UV；用户肤色差异
    由逐像素 lab_adapt 在合成时自动适应（不再依赖逐用户重烘）。"""
    from .makeup_uv import FINISH_TARGET, PHOTOREAL_LAB, SKIN_ROUGH, UvMakeupBaker

    tex = max(256, int(tex))                     # core.set_texture_size 下限钳制
    baker = UvMakeupBaker(tex=tex)
    maps = baker.bake(spec.get("layers", []), intensity=intensity)
    lip = maps.lip_w > 0.02 if maps.lip_w is not None else np.zeros((tex, tex), bool)
    albedo = maps.albedo
    if maps.lip_w is not None and maps.lip_albedo is not None:
        albedo = np.where(lip[..., None], maps.lip_albedo, maps.albedo)
    w = maps.w if maps.lip_w is None else np.maximum(maps.w, maps.lip_w)
    lip_kL, lip_ch = PHOTOREAL_LAB["lipstick"]
    rough_t, coat_t, sheen_t = FINISH_TARGET.get(
        maps.lip_finish or "gloss", FINISH_TARGET["gloss"])
    ch = maps.channels
    pack = MakeupPack(
        tex=tex, sigma=float(sigma),
        w=w.astype(np.float32), albedo=albedo.astype(np.float32),
        kL=np.where(lip, lip_kL, maps.kL).astype(np.float32),
        chroma=np.where(lip, lip_ch, maps.chroma).astype(np.float32),
        rough=np.where(lip, rough_t,
                       np.where(ch["rough"] > 0.02, ch["rough"], SKIN_ROUGH)).astype(np.float32),
        coat=np.where(lip, coat_t,
                      np.where(ch["coat"] > 0.005, ch["coat"], 0.06)).astype(np.float32),
        sss=np.where(lip, 1.0, ch["sss"]).astype(np.float32),
        sheen=np.where(lip, sheen_t, ch["sheen"]).astype(np.float32),
        lip_zone=lip)
    # 序列合成槽位（跨用户 preset 同样携带层链）
    ordered = list(getattr(maps, "layer_fields", []) or [])
    if maps.lip_w is not None:
        ordered.append({"region": "lipstick", "w": maps.lip_w,
                        "tgt": maps.lip_albedo if maps.lip_albedo is not None
                        else maps.albedo,
                        "kL": lip_kL, "chroma": lip_ch})
    (pack.slot_w, pack.slot_albedo,
     pack.slot_kL, pack.slot_chroma) = _build_slots(maps, ordered, tex)
    return pack


def apply_pack_to_cloud(cloud: dict[str, np.ndarray], pack: MakeupPack,
                        uv: np.ndarray, valid: np.ndarray) -> tuple[dict, np.ndarray]:
    """用户点云 + pack → 壳层（导出形态）。跨用户路径的导出侧：pack 采样出
    逐 splat 指派（w/albedo/kL/chroma/材质），走与 spec 路径同一壳层构造
    （浓度已在 bake 时烘进 pack，不再有 intensity 折叠）。"""
    from .makeup_uv import UvMakeupMaps

    maps = UvMakeupMaps.empty(pack.tex)
    maps.w = pack.w
    maps.albedo = pack.albedo
    maps.kL = pack.kL
    maps.chroma = pack.chroma
    maps.channels = {"rough": pack.rough, "coat": pack.coat,
                     "sss": pack.sss, "sheen": pack.sheen}
    from .makeup_uv import UvMakeupBaker
    baker = UvMakeupBaker(tex=pack.tex)
    return baker.build_makeup_layer(cloud, maps, np.asarray(uv, np.float64),
                                    np.asarray(valid, bool))


# ---------------- 逐像素合成（offline_render 消费；纯 numpy） ----------------

def composite_makeup_pixel(img_black: np.ndarray, alpha: np.ndarray,
                           uvmap: np.ndarray, pack: MakeupPack,
                           mode: str = "beer", sigma: float | None = None,
                           ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """素颜渲染（premultiplied-over-black）+ 像素 UV AOV + pack → 妆后。

    img_black (h,w,3) float 素颜主色（premultiplied over black）
    alpha     (h,w) 主体覆盖（含头发；发区 valid=0 自然不涂）
    uvmap     (h,w,3) UV AOV：R=u, G=v, B=valid（脸区主导占比，须 opaque 光栅化）
    mode      "beer"：T=exp(-σ·w) 颜料吸收（默认）；"linear"：T=1-0.92w
              （对齐旧壳层，A/B 用）；"none"：跳过上妆（纯素颜，管线对齐用）
    返回 (妆后 img_black, 妆层覆盖 coverage (h,w), 妆层材质采样 {k: (h,w)})。

    语义（序列合成，pack.slot_* 存在时）：真实上妆是层链——底妆先修正肤色，
    腮红迁移自修正后的肤色，高光最外。逐槽执行
        c = c·T_k + a_f·(1-T_k)·migrate(c_un, tgt_k, kL_k, chroma_k)
    迁移对象始终是"当前运行合成的肤色像素"→ 皮肤纹理/光影自动全保留。
    另含**底妆匀肤**：slot0（底妆层）覆盖处对皮肤做边缘保持滤波（抑制采集
    资产的彩色泼溅点/肤色斑驳——粉底的真实作用就是匀肤色不匀纹理）。
    无槽位时回退单层融合场合成（旧 pack 兼容）。"""
    h, w = img_black.shape[:2]
    out = img_black.copy()
    coverage = np.zeros((h, w), np.float64)
    mat = {k: np.zeros((h, w), np.float64) for k in FIELD_KEYS if k not in ("kL", "chroma")}
    if mode == "none":
        return out, coverage, mat

    valid = uvmap[..., 2] > VALID_THRESH
    cand = valid & (alpha > MIN_ALPHA)
    if not cand.any():
        return out, coverage, mat

    ys, xs = np.nonzero(cand)
    u = uvmap[ys, xs, 0]
    v = uvmap[ys, xs, 1]
    a_f = alpha[ys, xs]
    c_pm = img_black[ys, xs].copy()
    T_all = np.ones(len(ys), np.float64)

    if pack.n_slots > 0:
        # ---- 底妆匀肤：slot0 覆盖处抑制色度噪声/彩点（纹理由迁移保留） ----
        if pack.even_gain > 0.01:
            skin_un = img_black / np.clip(alpha, MIN_ALPHA, 1.0)[..., None]
            # 主体外 unpm 无定义——归一化卷积（blur(v·m)/blur(m)）天然填洞；
            # σ=2px@4096（≈0.2mm）只压 1-3px 的彩点/色度噪声，唇纹/毛孔尺度
            # 不受影响，且迁移链保持 L（输出明度=原始皮肤明度，纹理保留）
            m_body = (alpha > MIN_ALPHA).astype(np.float32)
            sig_px = max(1.5, img_black.shape[0] / 2048.0)
            mb = cv2.GaussianBlur(m_body, (0, 0), sig_px)
            for c in range(3):
                vb = cv2.GaussianBlur(
                    np.where(m_body > 0, skin_un[..., c], 0.0).astype(np.float32),
                    (0, 0), sig_px)
                skin_un[..., c] = np.where(mb > 1e-3,
                                           vb / np.maximum(mb, 1e-6), skin_un[..., c])
            skin_lp = skin_un
            w0 = sample_uv(pack.slot_w[0], u, v)
            g = np.clip(float(pack.even_gain) * np.clip(w0 / 0.55, 0.0, 1.0),
                        0.0, 1.0)[:, None]
            skin_base = skin_un[ys, xs] * (1 - g) + skin_lp[ys, xs] * g
        else:
            skin_base = img_black[ys, xs] / np.clip(a_f, MIN_ALPHA, 1.0)[:, None]
        c_un = skin_base
        sig = float(sigma if sigma is not None else pack.sigma)
        for k in range(pack.n_slots):
            wv = sample_uv(pack.slot_w[k], u, v)
            act = wv > MIN_W
            if not act.any():
                continue
            tgt = sample_uv(pack.slot_albedo[k], u[act], v[act])
            kLs = sample_uv(pack.slot_kL[k], u[act], v[act])
            chs = sample_uv(pack.slot_chroma[k], u[act], v[act])
            from .makeup_uv import UvMakeupBaker
            pig = np.asarray(UvMakeupBaker._lab_migrate(
                c_un[act], tgt, kLs, chs, np.ones(int(act.sum())), full=True),
                np.float64)
            if mode == "beer":
                T = np.exp(-sig * wv[act])
            else:
                T = 1.0 - LINEAR_OPACITY * wv[act]
            T = np.clip(T, 0.0, 1.0)
            c_pm[act] = c_pm[act] * T[:, None] \
                + (a_f[act] * (1 - T))[:, None] * pig
            T_all[act] *= T
            # 运行合成刷新（下一层迁移自上一层结果）
            c_un = c_pm / np.clip(a_f, MIN_ALPHA, 1.0)[:, None]
        cover = (1.0 - T_all) * a_f
        out[ys, xs] = c_pm
        coverage[ys, xs] = cover
    else:
        # 旧单层融合场（兼容无槽位 pack）
        wv = sample_uv(pack.w, u, v)
        m = wv > MIN_W
        if not m.any():
            return out, coverage, mat
        ys2, xs2, u2, v2, wv2 = ys[m], xs[m], u[m], v[m], wv[m]
        sig = float(sigma if sigma is not None else pack.sigma)
        if mode == "beer":
            T = np.exp(-sig * wv2)
        else:
            T = 1.0 - LINEAR_OPACITY * wv2
        T = np.clip(T, 0.0, 1.0)
        cover = (1.0 - T) * alpha[ys2, xs2]
        skin = img_black[ys2, xs2] / np.clip(alpha[ys2, xs2], MIN_ALPHA, 1.0)[:, None]
        tgt = sample_uv(pack.albedo, u2, v2)
        kL = sample_uv(pack.kL, u2, v2)
        chroma = sample_uv(pack.chroma, u2, v2)
        from .makeup_uv import UvMakeupBaker
        pig = np.asarray(UvMakeupBaker._lab_migrate(
            skin, tgt, kL, chroma, np.ones(len(ys2)), full=True), np.float64)
        out[ys2, xs2] = img_black[ys2, xs2] * T[:, None] + cover[:, None] * pig
        coverage[ys2, xs2] = cover
    for k in ("rough", "coat", "sss", "sheen"):
        mat[k][ys, xs] = sample_uv(getattr(pack, k), u, v)
    return out, coverage, mat


def mix_material(skin_maps: dict[str, np.ndarray], coverage: np.ndarray,
                 makeup_maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """像素级材质混合：妆层覆盖处用妆层材质（清漆/sss/珠光挂在妆上），
    素材透出处回皮肤材质。skin/makeup 各 (h,w)，coverage (h,w) 0..1。"""
    c = np.clip(coverage, 0.0, 1.0)
    return {k: skin_maps[k] * (1 - c) + makeup_maps[k] * c
            for k in skin_maps}
