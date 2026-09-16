#!/usr/bin/env python3
"""compare_bare_vs_makeup — 素颜 vs 妆容 对比图生成（P0 升级后管线的验收图）。

两条 3DGS 管线各出一张对比图（写入 out/compare/）：
    1. compare_synth_3dgs.png — 合成 canonical 头像（splat3d 路径）：
       正面 + 3/4 侧视角，验证高分辨率蒙版烘焙、超细细节层、延迟视角相关高光；
    2. compare_fit_user.png — fit 路径（真实用户点云的合成替身）：
       canonical 表面采样 + 相似变换 + 头发壳/背景离群 splat，
       验证 kNN UV 归属、离群剔除（头发不被涂妆）、密度增强、Blinn-Phong 唇釉镜面。

用法：python preview/compare_bare_vs_makeup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "desktop-app"))
sys.path.insert(0, str(ROOT / "makeup-skill" / "scripts"))

from makeupstudio.face3dgs.fit_makeup import FaceMakeupFitter, render_cloud  # noqa: E402
from makeupstudio.splat3d import (SplatCloudBuilder, fidelity_psnr,  # noqa: E402
                                  render_splats_python)
import preview_render as prc  # noqa: E402

OUT = ROOT / "out" / "compare"
PRESET = json.loads((ROOT / "makeup-skill" / "presets" / "date-rose.json")
                    .read_text(encoding="utf-8"))
W, H = 640, 537                     # 与 render_reference(size=640) 的 h=int(640*0.84) 一致


def label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((34, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (235, 230, 225), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def hstack_border(ims: list[np.ndarray], gap: int = 6) -> np.ndarray:
    h = max(im.shape[0] for im in ims)
    w = sum(im.shape[1] for im in ims) + gap * (len(ims) - 1)
    canvas = np.full((h, w, 3), 24, np.uint8)
    x = 0
    for im in ims:
        canvas[:im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    return canvas


def crop_zoom(img, cx, cy, half=70, zoom=3):
    c = img[cy - half:cy + half, cx - half:cx + half]
    return cv2.resize(c, (half * 2 * zoom, half * 2 * zoom),
                      interpolation=cv2.INTER_NEAREST)


def demo_synth() -> None:
    """合成 canonical 3DGS：素颜 / 妆容 × 正面 / 3/4 侧。"""
    print("[1/2] 合成 canonical 3DGS（tex=1024，含超细细节层 + 延迟高光）…")
    b = SplatCloudBuilder(tex=1024)
    bare = b.build([dict(l, enabled=False) for l in PRESET["layers"]],
                   intensity=0.0, n_base=70000, n_makeup=0)
    made = b.build(PRESET["layers"], intensity=0.85, n_base=70000, n_makeup=110000)

    ref = prc.render_reference(PRESET, intensity=0.85, size=W)
    got = render_splats_python(made, W, H, 0.0, max_splats=200000)
    print(f"    3DGS 还原度 vs 内核参考渲染：PSNR = {fidelity_psnr(ref, got):.2f} dB")

    rows = []
    for yaw, tag in ((0.0, "front"), (30.0, "3/4 view, spec flows with view")):
        im_bare = label(render_splats_python(bare, W, H, yaw, max_splats=200000),
                        f"BARE  ({tag})")
        im_made = label(render_splats_python(made, W, H, yaw, max_splats=200000),
                        f"MAKEUP  date-rose  ({tag})")
        rows.append(hstack_border([im_bare, im_made]))
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "compare_synth_3dgs.png"), np.vstack(rows))
    print(f"    -> {OUT / 'compare_synth_3dgs.png'}")


def _quat_to(n: np.ndarray) -> np.ndarray:
    """(0,0,1)→n 的四元数 (w,x,y,z)，n 单位向量。"""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n)
    s = np.linalg.norm(v)
    a = np.arctan2(s, n[2])
    axis = v / (s + 1e-12)
    half = a / 2
    return np.array([np.cos(half), axis[0] * np.sin(half),
                     axis[1] * np.sin(half), axis[2] * np.sin(half)], np.float32)


def _user_cloud(builder: SplatCloudBuilder, fitter: FaceMakeupFitter):
    """P2 场景的"用户点云"：FlashAvatar 类 FLAME 底座导出的 canonical 点云。

    脸 = 素颜稠密脸壳（SplatCloudBuilder，canonical 姿态、真实法线），
    再拼一圈头发壳（距 canonical 表面很远）验证离群剔除。返回 (bare, landmarks)。
    """
    bare = builder.build([dict(l, enabled=False) for l in PRESET["layers"]],
                         intensity=0.0, n_base=70000, n_makeup=0)
    rng = np.random.default_rng(5)
    n_h = 9000
    th = rng.uniform(0, 2 * np.pi, n_h)
    ph = rng.uniform(0.05, 1.35, n_h)
    r = 0.40 + rng.normal(0, 0.03, n_h)
    shell = np.stack([r * np.sin(ph) * np.cos(th), r * np.cos(ph) * 1.02,
                      -r * np.sin(ph) * 0.55], axis=1)          # canonical 帧：-z = 脑后
    cloud = {
        "xyz": np.concatenate([bare["xyz"], shell]).astype(np.float32),
        "scale": np.concatenate([bare["scale"], np.full((n_h, 3), 0.018)]).astype(np.float32),
        "rot": np.concatenate([bare["rot"],
                               np.tile([0.0, 0.0, 0.0, 1.0], (n_h, 1))]).astype(np.float32),
        "rgba": np.concatenate([
            bare["rgba"],
            np.column_stack([np.tile([0.16, 0.11, 0.09], (n_h, 1)),
                             np.full((n_h, 1), 0.95)])]).astype(np.float32),
    }
    return cloud, fitter.model.base[:468]


def demo_fit() -> None:
    """fit_canonical 路径：FLAME 底座导出的 canonical 点云 → 上妆（P2 入口，
    跳过视频三角化）；头发壳验证离群剔除；densify + Blinn-Phong 唇釉镜面。"""
    print("[2/2] fit_canonical 路径（canonical 点云 + 头发壳，densify=True）…")
    b = SplatCloudBuilder(tex=1024)
    fitter = FaceMakeupFitter()
    cloud, landmarks = _user_cloud(b, fitter)
    made = fitter.fit_canonical(cloud, PRESET, OUT.parent / "fitted_canonical",
                                intensity=0.85, densify=True)
    print(f"    splats：{len(cloud['xyz'])} -> {len(made['xyz'])}"
          f"（密度增强 +{len(made['xyz']) - len(cloud['xyz'])}）")

    # canonical 帧（鼻尖 +z）→ render_cloud 的 COLMAP 帧（相机看 +z、y 向下）：
    # 点云 z 平移 -2 后，R = Rx(π)（x, -y, -z）：脸正面朝相机、图像正立不镜像
    class _Cam:
        cam_id, width, height = 1, 640, 480
        params = np.array([600.0, 320.0, 240.0])

    R = np.diag([1.0, -1.0, -1.0])
    t = np.zeros(3)
    shift = np.array([0.0, 0.0, -2.0])
    view = {k: v + shift if k == "xyz" else v for k, v in cloud.items()}
    made_v = {k: v + shift if k == "xyz" else v for k, v in made.items()}
    # 光源偏右上（世界系）：高光只落在唇峰/颧骨等朝向光源的面上
    headlight = np.array([0.45, 0.40, 0.80])
    im_bare = label(render_cloud(view, R, t, _Cam, w=640, h=480), "BARE (canonical cloud)")
    im_made = label(render_cloud(made_v, R, t, _Cam, w=640, h=480,
                                 light_dir=headlight, spec_strength=0.45),
                    "MAKEUP  date-rose + densify + gloss (fit_canonical)")

    # py = 240 - 300·y_canon（f=600, z≈2）：唇 y≈-0.30 → 330；眼 y≈+0.16 → 192
    face = im_made[34:]
    lip = label(crop_zoom(face, 320, 330, 60, 3), "LIP zoom (gloss spec)")
    eye = label(crop_zoom(face, 371, 192, 60, 3), "EYE zoom (1024 mask)")
    top = hstack_border([im_bare, im_made])
    bottom = hstack_border([lip, eye])
    if bottom.shape[1] < top.shape[1]:      # 放大条窄于上排 → 居中垫到同宽
        pad = top.shape[1] - bottom.shape[1]
        bottom = np.pad(bottom, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)),
                        constant_values=24)
    grid = np.vstack([top, bottom])
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "compare_fit_user.png"), grid)
    print(f"    -> {OUT / 'compare_fit_user.png'}")


if __name__ == "__main__":
    demo_synth()
    demo_fit()
    print("完成。")
