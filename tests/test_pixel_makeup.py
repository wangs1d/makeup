"""test_pixel_makeup — 妆容 UV 图集（pack）+ 逐像素合成 + 壳层加密单测。

覆盖：pack 合成（3D 锚定权重光栅化 + UV 场融合，makeup_pack.compose_pack）、
save/load 往返（f16）、逐像素 Beer-Lambert/线性合成数学（composite_makeup_pixel，
premultiplied 代数 + valid 门控 + 2048² 细节透出）、preset pack 烘焙与跨用户
应用（bake_preset_pack/with_binding/apply_pack_to_cloud）、壳层 2×2 分裂加密
（makeup_uv.subdivide_makeup_layer：梯度区才分裂/子 splat UV 重采样/足迹内）、
guidance 自动升级判定（pipeline.wants_guidance）。全部 CPU/numpy，不依赖
gsplat/CUDA（fit_makeup 的 canonical 模板为纯 numpy 依赖）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parent.parent / "desktop-app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


# ---------------- 测试用合成数据 ----------------

def _synthetic_maps(tex: int = 256):
    """带妆语义缩影的 UvMakeupMaps：中部 foundation、下方唇带。

    tex 必须 ≥256（core.set_texture_size 有 max(256,·) 下限钳制）。"""
    from makeupstudio.face3dgs.appearance.makeup_uv import UvMakeupMaps

    maps = UvMakeupMaps.empty(tex)
    yy, xx = np.mgrid[0:tex, 0:tex] / (tex - 1)
    face = (xx > 0.2) & (xx < 0.8) & (yy > 0.2) & (yy < 0.8)
    maps.w[face] = 0.6
    maps.albedo[face] = np.array([0.85, 0.65, 0.60], np.float32)
    maps.kL[face] = 0.08
    maps.chroma[face] = 0.35
    maps.channels["rough"][face] = 0.62
    maps.channels["coat"][face] = 0.10
    # 唇带：独立通道 + 真实目标色（红）
    lip = (np.hypot(xx - 0.5, (yy - 0.62) * 1.6) < 0.06)
    maps.lip_w = np.where(lip, 0.9, 0.0).astype(np.float32)
    maps.lip_albedo = np.where(lip[..., None],
                               np.array([0.70, 0.10, 0.14], np.float32), 0.0)
    maps.lip_cent = np.clip((yy - 0.56) / 0.12, 0, 1).astype(np.float32)
    maps.lip_stops = [{"at": 0.0, "hex": "#8F1D2C"}, {"at": 1.0, "hex": "#B23040"}]
    maps.lip_opacity = 0.9
    maps.lip_finish = "gloss"
    return maps


def _synthetic_cloud(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    xyz = rng.random((n, 3)).astype(np.float32)
    return {
        "xyz": xyz,
        "scale": np.full((n, 3), 0.03, np.float32),
        "rot": np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], np.float32), (n, 1)),
        "rgba": np.concatenate(
            [np.full((n, 3), 0.75, np.float32),
             np.full((n, 1), 0.95, np.float32)], 1),
        "sh_rest": np.zeros((n, 8, 3), np.float32),
    }


# ---------------- pack 合成 ----------------

def test_compose_pack_fuses_band_weights_and_uv_colors():
    """w 场来自 3D 锚定指派（唇区覆写），颜色场保 UV 细节与唇目标色。"""
    from makeupstudio.face3dgs.appearance.makeup_pack import compose_pack
    from makeupstudio.face3dgs.appearance.makeup_uv import UvMakeupBaker

    tex = 256
    maps = _synthetic_maps(tex)
    cloud = _synthetic_cloud()
    # uv 均匀撒在脸上（valid），uv→像素坐标含唇区
    uv = np.stack([np.linspace(0.2, 0.8, 20).repeat(20),
                   np.tile(np.linspace(0.2, 0.8, 20), 20)], 1)
    valid = np.ones(len(uv), bool)
    baker = UvMakeupBaker(tex=tex)

    pack = compose_pack(baker, cloud, maps, uv, valid, lip3d=None, near=None)
    assert pack.tex == tex
    # 唇区 w 来自 UV 唇带兜底（无 3D 锚定），全局区来自 foundation 场
    assert float(pack.w.max()) > 0.3
    # 唇区目标色 = 红系（r 明显大于 g/b）；脸区 = foundation 色系
    lip_px = np.unravel_index(np.argmax(
        pack.w * ((np.mgrid[0:tex, 0:tex][1] / (tex - 1) - 0.62) ** 2 < 1e-4)), pack.w.shape)
    # 唇带附近 texel 的 albedo 应为唇色
    ty, tx_ = np.where(pack.w > 0.5)
    r = pack.albedo[ty, tx_, 0]
    g = pack.albedo[ty, tx_, 1]
    assert float((r - g).max()) > 0.1        # 存在明显红于绿的妆区（唇）
    # 材质已做皮肤兜底：rough 全场在合法区间（唇区 gloss≈0.16，皮肤基准 0.52）
    assert float(pack.rough.min()) >= 0.05 and float(pack.rough.max()) <= 0.95


def test_pack_save_load_roundtrip(tmp_path):
    from makeupstudio.face3dgs.appearance.makeup_pack import compose_pack
    from makeupstudio.face3dgs.appearance.makeup_uv import UvMakeupBaker

    maps = _synthetic_maps()
    cloud = _synthetic_cloud()
    uv = np.random.default_rng(1).random((len(cloud["xyz"]), 2))
    pack = compose_pack(UvMakeupBaker(tex=256), cloud, maps,
                        uv, np.ones(len(uv), bool))
    pack.uv, pack.valid = uv, np.ones(len(uv), bool)
    p = pack.save(tmp_path / "m.npz")
    back = type(pack).load(p)
    assert back.tex == pack.tex and abs(back.sigma - pack.sigma) < 1e-6
    for k in ("w", "kL", "chroma", "rough"):
        a, b = getattr(pack, k), getattr(back, k)
        assert a.shape == b.shape
        # f16 量化误差应远小于 1/255（不引入可见色带）
        assert float(np.abs(a - b).max()) < 2e-3
    assert back.uv.shape == (len(uv), 2) and back.valid.all()


# ---------------- 逐像素合成 ----------------

def _render_fixture(tex: int = 64):
    """8×8 渲染 fixture：2×2 脸区（valid=1，alpha=1），外部背景。"""
    h = w = 8
    img_black = np.full((h, w, 3), 0.5)          # 素颜灰
    alpha = np.zeros((h, w))
    uvmap = np.zeros((h, w, 3))
    ys, xs = np.mgrid[0:h, 0:w]
    face = (ys >= 2) & (ys < 6) & (xs >= 2) & (xs < 6)
    alpha[face] = 1.0
    # uv = 归一化像素坐标（v 图像行向下 → pack v=1-y 惯例）
    u = (xs + 0.5) / w
    v = 1.0 - (ys + 0.5) / h
    uvmap[..., 0] = np.where(face, u, 0.0)
    uvmap[..., 1] = np.where(face, v, 0.0)
    uvmap[..., 2] = face.astype(np.float64)
    return img_black, alpha, uvmap


def test_pixel_composite_beer_vs_linear_and_gating():
    """Beer-Lambert 满涂 ≈ 旧壳层满涂；淡涂更透；invalid 区不涂。"""
    from makeupstudio.face3dgs.appearance.makeup_pack import (
        MakeupPack, composite_makeup_pixel)

    tex = 64
    pack = MakeupPack(
        tex=tex,
        w=np.full((tex, tex), 1.0, np.float32),
        albedo=np.zeros((tex, tex, 3), np.float32) + np.array([0.9, 0.2, 0.2]),
        kL=np.full((tex, tex), 1.0, np.float32),       # L 全量跟随
        chroma=np.full((tex, tex), 1.0, np.float32),
        rough=np.full((tex, tex), 0.2, np.float32),
        coat=np.full((tex, tex), 0.5, np.float32),
        sss=np.zeros((tex, tex), np.float32),
        sheen=np.zeros((tex, tex), np.float32),
        sigma=2.0)
    img_black, alpha, uvmap = _render_fixture()

    out_b, cov_b, mat_b = composite_makeup_pixel(img_black.copy(), alpha, uvmap,
                                                 pack, mode="beer")
    out_l, cov_l, _ = composite_makeup_pixel(img_black.copy(), alpha, uvmap,
                                             pack, mode="linear")
    face = alpha > 0.5
    bg = ~(uvmap[..., 2] > 0.5)
    # 满涂：两模式妆区都接近 pigment 色且远离素颜灰
    assert np.abs(out_b[face] - out_l[face]).max() < 0.15
    assert np.abs(out_b[face] - np.array([0.9, 0.2, 0.2])).max() < 0.2
    # 淡涂（w=0.2）：Beer-Lambert 几何积累——σ=2 的低强度吸收斜率（2.0）
    # 大于线性模式的 0.92，同样 w 下妆效更强（浓淡分离度被放大）
    pack.w[:] = 0.2
    out_b2, cov_b2, _ = composite_makeup_pixel(img_black.copy(), alpha, uvmap,
                                               pack, mode="beer")
    out_l2, cov_l2, _ = composite_makeup_pixel(img_black.copy(), alpha, uvmap,
                                               pack, mode="linear")
    assert np.abs(out_b2[face] - 0.5).max() > np.abs(out_l2[face] - 0.5).max()
    assert cov_b2[face].mean() > cov_l2[face].mean()
    # 背景区不动、coverage=0、材质为 0
    assert np.allclose(out_b[bg], img_black[bg])
    assert cov_b[bg].max() == 0.0 and mat_b["rough"][bg].max() == 0.0
    # coverage ∈ [0, alpha]
    assert cov_b2[face].max() <= alpha[face].max() + 1e-9


def test_pixel_composite_passes_2048_texture_detail():
    """贴图里的深/浅 texel 行在逐像素输出中分明（splat 常数色做不到的事）。"""
    from makeupstudio.face3dgs.appearance.makeup_pack import (
        MakeupPack, composite_makeup_pixel)

    tex = 64
    w = np.zeros((tex, tex), np.float32)
    albedo = np.full((tex, tex, 3), 0.8, np.float32)
    w[26:40, :] = 1.0                                # 横向妆带（v∈[0.365,0.587]）
    albedo[26:32, :] = 0.1                           # 带内深色半段
    # fixture 像素行 3 采样 v=0.5625 → texel 行≈27.6（深），行 4 采样 v=0.4375
    # → texel 行≈35.4（浅）——两行都在妆带内、分属贴图不同 texel 行
    pack = MakeupPack(tex=tex, w=w, albedo=albedo,
                      kL=np.full((tex, tex), 1.0, np.float32),
                      chroma=np.full((tex, tex), 1.0, np.float32),
                      rough=np.full((tex, tex), 0.5, np.float32),
                      coat=np.zeros((tex, tex), np.float32),
                      sss=np.zeros((tex, tex), np.float32),
                      sheen=np.zeros((tex, tex), np.float32))
    img_black, alpha, uvmap = _render_fixture()
    out, _cov, _mat = composite_makeup_pixel(img_black.copy(), alpha, uvmap,
                                             pack, mode="beer")
    row_deep = out[3, 2:6]
    row_light = out[4, 2:6]
    # 两行都上了妆（偏离素颜灰），且贴图深浅差异在输出中保留 ≥0.1
    assert abs(row_light.mean() - 0.5) > 0.05
    assert abs(row_deep.mean() - 0.5) > 0.05
    assert row_light.mean() > row_deep.mean() + 0.1


def test_pixel_composite_none_mode_is_identity():
    from makeupstudio.face3dgs.appearance.makeup_pack import (
        MakeupPack, composite_makeup_pixel)

    pack = MakeupPack(tex=8, w=np.ones((8, 8), np.float32),
                      albedo=np.ones((8, 8, 3), np.float32),
                      kL=np.ones((8, 8), np.float32), chroma=np.ones((8, 8), np.float32),
                      rough=np.ones((8, 8), np.float32), coat=np.ones((8, 8), np.float32),
                      sss=np.ones((8, 8), np.float32), sheen=np.ones((8, 8), np.float32))
    img_black, alpha, uvmap = _render_fixture()
    out, cov, mat = composite_makeup_pixel(img_black.copy(), alpha, uvmap,
                                           pack, mode="none")
    assert np.array_equal(out, img_black) and cov.max() == 0.0


# ---------------- preset pack：跨用户复用（④） ----------------

def test_bake_preset_pack_and_cross_user_apply(tmp_path):
    """preset pack 无绑定可烘焙；两个"用户"（不同肤色点云）绑定后各自出壳层。"""
    from makeupstudio.face3dgs.appearance.makeup_pack import (
        apply_pack_to_cloud, bake_preset_pack)
    from makeupstudio.face3dgs.appearance.makeup_uv import UvMakeupBaker

    spec = {"layers": [
        {"region": "foundation", "opacity": 0.7, "finish": "satin",
         "color_stops": [{"at": 0.0, "hex": "#EFD3BE"}, {"at": 1.0, "hex": "#EACBB4"}]},
        {"region": "lipstick", "opacity": 0.9, "finish": "gloss",
         "color_stops": [{"at": 0.0, "hex": "#8F1D2C"}, {"at": 1.0, "hex": "#B23040"}]},
    ]}
    pack = bake_preset_pack(spec, tex=256, intensity=0.8)
    assert pack.uv is None and pack.valid is None
    p = pack.save(tmp_path / "preset.pack.npz")
    pack2 = type(pack).load(p)
    assert pack2.uv is None

    # 两个"用户"：不同肤色的点云，均确定性覆盖唇区中心（pack.lip_zone 取样）
    assert pack2.lip_zone is not None and pack2.lip_zone.any()
    lz = np.argwhere(pack2.lip_zone)
    ly, lx = lz[len(lz) // 2]
    t = pack2.tex
    u_lip, v_lip = lx / (t - 1), 1 - ly / (t - 1)
    shell_colors = []
    for skin in (0.85, 0.55):
        rng = np.random.default_rng(7)
        n = 300
        uv = np.stack([np.clip(u_lip + rng.normal(0, 0.004, n), 0.01, 0.99),
                       np.clip(v_lip + rng.normal(0, 0.004, n), 0.01, 0.99)], 1)
        cloud = _synthetic_cloud(n, seed=int(skin * 100))
        cloud["rgba"][:, :3] = skin
        user_pack = pack2.with_binding(uv, np.ones(n, bool))
        layer, idx = apply_pack_to_cloud(cloud, user_pack, uv, np.ones(n, bool))
        assert len(idx) > 0
        shell_colors.append(layer["rgba"][:, :3])
    # 两个用户都涂上了唇色（红明显强于绿），且深肤用户的壳层颜色经 lab_adapt
    # 后与浅肤用户不同（跨用户自适应而非同一常数色）
    for cols in shell_colors:
        assert float((cols[:, 0] - cols[:, 1]).max()) > 0.1
    assert float(np.abs(shell_colors[0] - shell_colors[1]).mean()) > 0.005


# ---------------- 壳层 2×2 分裂加密（②） ----------------

def test_subdivide_splits_only_gradient_region():
    """妆权重有梯度的壳层 splat 分裂为 4 子；平缓区不分裂；子 splat 属性自洽。"""
    from makeupstudio.face3dgs.appearance.makeup_pack import MakeupPack
    from makeupstudio.face3dgs.appearance.makeup_uv import subdivide_makeup_layer

    n = 64
    tex = 256
    cloud = _synthetic_cloud(n)
    # uv.u = xyz.x（雅可比可辨识）：子 splat 沿 x 的偏移会真实改变 uv
    u = np.linspace(0.1, 0.9, n)
    cloud["xyz"][:, 0] = u
    uv = np.stack([u, np.full(n, 0.5)], 1)
    # w 沿 u 全平面线性梯度（0→1）：中部 splat 的子足迹必有 |Δw|
    w = np.tile(np.linspace(0.0, 1.0, tex), (tex, 1)).astype(np.float32)
    pack = MakeupPack(tex=tex, w=w,
                      albedo=np.full((tex, tex, 3), 0.8, np.float32),
                      kL=np.full((tex, tex), 0.6, np.float32),
                      chroma=np.full((tex, tex), 1.0, np.float32),
                      rough=np.full((tex, tex), 0.3, np.float32),
                      coat=np.zeros((tex, tex), np.float32),
                      sss=np.zeros((tex, tex), np.float32),
                      sheen=np.zeros((tex, tex), np.float32))
    layer = {
        "xyz": cloud["xyz"].copy(),
        "scale": np.full((n, 3), 0.05, np.float32),
        "rot": cloud["rot"].copy(),
        "rgba": np.concatenate([np.full((n, 3), 0.7, np.float32),
                                np.full((n, 1), 0.9, np.float32)], 1),
        "makeup_w": u.astype(np.float32),
        "sh_rest": np.zeros((n, 8, 3), np.float32),
    }
    from makeupstudio.face3dgs.appearance.pbr import Material
    layer["material"] = Material.skin(n)
    src_idx = np.arange(n)
    out, src2 = subdivide_makeup_layer(cloud, layer, src_idx, pack, uv,
                                       np.ones(n, bool), min_grad=0.02)
    assert len(src2) > n                               # 梯度区发生了分裂
    assert len(out["xyz"]) == len(src2) == len(out["rgba"]) == len(out["material"].rough)
    # 子 splat scale 主轴收缩、颜色自 Lab 迁移自重采样的目标色
    assert float(out["scale"].max()) <= 0.05 + 1e-6
    # 平缓区（w 恒 0.5 的 pack）不分裂
    pack2 = MakeupPack(tex=tex, w=np.full((tex, tex), 0.5, np.float32),
                       albedo=pack.albedo, kL=pack.kL, chroma=pack.chroma,
                       rough=pack.rough, coat=pack.coat, sss=pack.sss, sheen=pack.sheen)
    out2, src3 = subdivide_makeup_layer(cloud, layer, src_idx, pack2, uv,
                                        np.ones(n, bool), min_grad=0.02)
    assert len(src3) == n and np.allclose(np.sort(src3), np.sort(src_idx))


# ---------------- guidance 自动升级判定（③） ----------------

def test_wants_guidance_shape_heavy_vs_plain():
    from makeupstudio.face3dgs.appearance.pipeline import wants_guidance

    preset_dir = (Path(__file__).resolve().parent.parent
                  / "makeup-skill" / "presets")
    smoky = json.loads((preset_dir / "smoky-night.json").read_text(encoding="utf-8"))
    assert wants_guidance(smoky)                       # wing 0.65 → 形状级
    plain = {"layers": [{"region": "lipstick", "enabled": True, "opacity": 0.8,
                         "shape": {}}]}
    assert not wants_guidance(plain)
    below = {"layers": [{"region": "eyeliner", "enabled": True,
                         "shape": {"wing": 0.1}}]}
    assert not wants_guidance(below)
    disabled = {"layers": [{"region": "eyeliner", "enabled": False,
                            "shape": {"wing": 0.9}}]}
    assert not wants_guidance(disabled)


# ---------------- 序列合成槽位（层链语义） ----------------

def _slot_pack(tex: int = 64):
    from makeupstudio.face3dgs.appearance.makeup_pack import MakeupPack

    def layer(wv, rgb):
        return {"w": np.full((tex, tex), wv, np.float32),
                "tgt": np.zeros((tex, tex, 3), np.float32) + rgb,
                "kL": 1.0, "chroma": 1.0}
    pack = MakeupPack(
        tex=tex, w=np.zeros((tex, tex), np.float32),
        albedo=np.zeros((tex, tex, 3), np.float32),
        kL=np.zeros((tex, tex), np.float32), chroma=np.zeros((tex, tex), np.float32),
        rough=np.full((tex, tex), 0.5, np.float32), coat=np.zeros((tex, tex), np.float32),
        sss=np.zeros((tex, tex), np.float32), sheen=np.zeros((tex, tex), np.float32))
    pack.slot_w = [layer(0.6, (0.9, 0.8, 0.7))["w"], layer(0.7, (0.8, 0.2, 0.2))["w"]]
    pack.slot_albedo = [layer(0, (0.9, 0.8, 0.7))["tgt"], layer(0, (0.8, 0.2, 0.2))["tgt"]]
    pack.slot_kL = [np.full((tex, tex), 1.0, np.float32),
                    np.full((tex, tex), 1.0, np.float32)]
    pack.slot_chroma = [np.full((tex, tex), 1.0, np.float32),
                        np.full((tex, tex), 1.0, np.float32)]
    return pack


def test_sequential_slots_apply_layer_chain():
    """两层槽位按序迁移-合成：slot1（腮红）迁移自 slot0（底妆）修正后的肤色。"""
    from makeupstudio.face3dgs.appearance.makeup_pack import (
        composite_makeup_pixel, sample_uv)
    pack = _slot_pack()
    img_black, alpha, uvmap = _render_fixture()
    out, cov, _mat = composite_makeup_pixel(img_black.copy(), alpha, uvmap,
                                            pack, mode="beer")
    face = alpha > 0.5
    # T0=exp(-2·0.6)=0.30, T1=exp(-2·0.7)=0.25 → 输出被 slot1 红色主导
    assert out[face][:, 0].mean() > 0.6
    assert out[face][:, 0].mean() > out[face][:, 1].mean()
    # 层链语义：slot1 迁移自 slot0 修正后的肤色——若换成倒序，结果应不同
    pack_r = _slot_pack()
    pack_r.slot_w = pack.slot_w[::-1]
    pack_r.slot_albedo = pack.slot_albedo[::-1]
    out_r, _c, _m = composite_makeup_pixel(img_black.copy(), alpha, uvmap,
                                           pack_r, mode="beer")
    assert np.abs(out - out_r)[face].max() > 0.02


def test_sequential_slots_roundtrip(tmp_path):
    from makeupstudio.face3dgs.appearance.makeup_pack import (
        MakeupPack, composite_makeup_pixel)
    pack = _slot_pack()
    pack.even_gain = 0.5
    p = pack.save(tmp_path / "slots.npz")
    back = MakeupPack.load(p)
    assert back.n_slots == 2 and back.even_gain == 0.5
    img_black, alpha, uvmap = _render_fixture()
    o1, _c1, _m1 = composite_makeup_pixel(img_black.copy(), alpha, uvmap, pack)
    o2, _c2, _m2 = composite_makeup_pixel(img_black.copy(), alpha, uvmap, back)
    assert np.abs(o1 - o2).max() < 2e-3


def test_foundation_even_skin_suppresses_chroma_noise():
    """底妆匀肤：slot0 覆盖处单像素彩点（资产泼溅噪声）被抑制，机制生效。"""
    from makeupstudio.face3dgs.appearance.makeup_pack import (
        MakeupPack, composite_makeup_pixel)
    tex = 64
    rng = np.random.default_rng(3)
    pack = MakeupPack(
        tex=tex, w=np.zeros((tex, tex), np.float32),
        albedo=np.zeros((tex, tex, 3), np.float32),
        kL=np.zeros((tex, tex), np.float32), chroma=np.zeros((tex, tex), np.float32),
        rough=np.full((tex, tex), 0.5, np.float32), coat=np.zeros((tex, tex), np.float32),
        sss=np.zeros((tex, tex), np.float32), sheen=np.zeros((tex, tex), np.float32),
        even_gain=0.85)
    pack.slot_w = [np.full((tex, tex), 0.8, np.float32)]
    pack.slot_albedo = [np.zeros((tex, tex, 3), np.float32) + 0.5]
    pack.slot_kL = [np.zeros((tex, tex), np.float32)]
    pack.slot_chroma = [np.zeros((tex, tex), np.float32)]

    # 32×32 fixture：素颜灰 + 妆区内散布单像素彩点（±0.2）
    n = 32
    img = np.full((n, n, 3), 0.5)
    alpha = np.zeros((n, n))
    uvmap = np.zeros((n, n, 3))
    yy, xx = np.mgrid[0:n, 0:n]
    face = (yy >= 4) & (yy < n - 4) & (xx >= 4) & (xx < n - 4)
    alpha[face] = 1.0
    u = (xx + 0.5) / n
    v = 1.0 - (yy + 0.5) / n
    uvmap[..., 0] = np.where(face, u, 0.0)
    uvmap[..., 1] = np.where(face, v, 0.0)
    uvmap[..., 2] = face.astype(np.float64)
    dots = rng.random((n, n)) < 0.08
    dots &= face
    for c, amp in zip((0, 1, 2), (0.22, -0.18, 0.15)):
        img[..., c] = np.clip(img[..., c] + dots * amp, 0, 1)
    img_black = img * alpha[..., None]                # premultiplied over black

    out, _cov, _mat = composite_makeup_pixel(img_black.copy(), alpha, uvmap, pack)
    # 输入彩点强度（妆区 RGB 方向偏差）
    in_dev = np.abs(img_black - 0.5)[face].max()
    out_dev = np.abs(out - 0.5)[face].max()
    # 底妆 w=0.8：T=exp(-2·0.8)=0.20 → 80% pigment（匀肤后≈0.5 灰）
    assert out_dev < 0.5 * in_dev                     # 匀肤+覆盖显著压平彩点
    # 机制对照：even_gain=0 时彩点只被颜料覆盖部分压低，压不平
    pack0 = MakeupPack(**{**{k: getattr(pack, k) for k in
                            ("tex", "w", "albedo", "kL", "chroma", "rough", "coat",
                             "sss", "sheen")}, "even_gain": 0.0,
                         "slot_w": pack.slot_w, "slot_albedo": pack.slot_albedo,
                         "slot_kL": pack.slot_kL, "slot_chroma": pack.slot_chroma})
    out0, _c, _m = composite_makeup_pixel(img_black.copy(), alpha, uvmap, pack0)
    assert np.abs(out0 - out)[face].max() > 0.02      # 匀肤通道真实生效