#!/usr/bin/env python3
"""sculpt_orbit_video — 合成"满足采集规范"的环绕视频（画质门禁/上限验证基准）。

把 sculpt_face_avatar 的素颜头模渲染成环绕视频：1920×1080（短边 ≥1000，
质量门禁 A 档）、±55° yaw 均匀环绕、固定表情、Lambert+Blinn 着色（法线取
雕刻高斯薄轴，光固定于世界系——视角相关高光留给 SH 学）、SfM 友好的
点阵球壳背景。产物：<project>/images/frame_*.png（无损直喂管线，不经
视频压缩）+ preview.mp4（人工核效用）。

用法：
    python sculpt_orbit_video.py --project out/scan_hd [--frames 360] [--yaw 55]

验证用途：同一管线在 640×536（29 帧可用）vs 1080p（~120 帧可用）上的
资产质量对照——分辨率/视角数是逼真度的第一决定因素。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "desktop-app"))


def lookat(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z = target - eye
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.cross(z, up)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    w2c = np.eye(4)
    w2c[:3, :3] = np.stack([x, y, z])
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    return w2c


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → (qw,qx,qy,qz)（COLMAP images.bin 约定）。"""
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
             (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
             (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def write_sparse_gt(names: list[str], w2cs: list[np.ndarray], f: float,
                    size: tuple[int, int], out_dir: Path) -> None:
    """渲染位姿 → 标准COLMAP 二进制稀疏模型（PINHOLE，GT 无需 SfM）。

    合成视频的相机位姿精确已知——SfM 在重复点阵纹理上不可靠（实测初始
    像对两视图几何全失败），基准实验直接注入 GT 位姿。colmap_io 消费
    cameras.bin + images.bin 即可。"""
    import struct

    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = size
    cam_bytes = struct.pack("<Q", 1) + struct.pack("<i", 1) + \
        struct.pack("<i", 1) + struct.pack("<Q", W) + struct.pack("<Q", H) + \
        struct.pack("<4d", f, f, W / 2, H / 2)
    (out_dir / "cameras.bin").write_bytes(cam_bytes)

    parts = [struct.pack("<Q", len(names))]
    for i, (name, w2c) in enumerate(zip(names, w2cs), start=1):
        q = rotmat_to_quat_wxyz(w2c[:3, :3])
        parts.append(struct.pack("<i", i))
        parts.append(struct.pack("<4d", *q))
        parts.append(struct.pack("<3d", *w2c[:3, 3]))
        parts.append(struct.pack("<i", 1))
        parts.append(name.encode("utf-8") + b"\x00")
        parts.append(struct.pack("<Q", 0))           # 2D 点数 0（管线不消费）
    (out_dir / "images.bin").write_bytes(b"".join(parts))
    print(f"[gt] GT 稀疏位姿 {len(names)} 帧 → {out_dir}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True, help="工程目录（images/ 写入帧）")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--yaw", type=float, default=55.0, help="单侧最大偏航角")
    ap.add_argument("--pitch", type=float, default=6.0)
    ap.add_argument("--fill", type=float, default=0.66, help="脸高占画面高度比例")
    ap.add_argument("--points", type=int, default=240_000, help="头模高斯数")
    ap.add_argument("--mp4", action="store_true", help="顺带编码 preview.mp4")
    args = ap.parse_args()

    W, H = (int(v) for v in args.size.lower().split("x"))

    from sculpt_face_avatar import build
    print(f"[sculpt] 生成头模 {args.points} 高斯…", flush=True)
    av, _anchors = build(args.points)

    from makeupstudio.face3dgs.appearance.normals import axis_normals
    from makeupstudio.face3dgs.appearance.offline_render import load_prepared, render_pose

    # 着色输入：薄轴法线（雕刻时法线即薄轴）+ 反照率。
    # av.quats 是 (w,x,y,z)；内部约定 xyzw → 轴序变换
    quat_xyzw = np.asarray(av.quats, np.float64)[:, [1, 2, 3, 0]]
    nrm = axis_normals(quat_xyzw, np.asarray(av.scales, np.float64))
    albedo = np.clip(np.asarray(av.colors, np.float64), 0, 1)
    L = np.array([-0.35, 0.55, 0.76])
    L = L / np.linalg.norm(L)

    # SfM 友好背景：远处球壳点阵（特征丰富，训练时被 oval 蒙版排除）。
    # 点必须小而亮（SIFT 要角点/高对比）：暗糊大点每帧只提得到 ~230 特征，
    # SfM 初始化直接失败（实测）
    rng = np.random.default_rng(11)
    n_bg = 16000
    d_bg = rng.normal(0, 1, (n_bg, 3))
    d_bg /= np.linalg.norm(d_bg, axis=1, keepdims=True)
    r_bg = rng.uniform(6.0, 10.0, (n_bg, 1))
    bg_pos = (d_bg * r_bg).astype(np.float32)
    bg_col = (np.array([0.60, 0.66, 0.78])[None]
              * (0.35 + 0.65 * rng.random((n_bg, 1)))).astype(np.float32)
    bg_q = np.tile(np.array([[0.0, 0, 0, 1]], np.float32), (n_bg, 1))  # xyzw 单位
    bg_s = rng.uniform(0.005, 0.016, (n_bg, 3)).astype(np.float32)

    xyz = np.concatenate([av.means, bg_pos]).astype(np.float32)
    rot = np.concatenate([quat_xyzw, bg_q]).astype(np.float32)
    scale = np.concatenate([av.scales, bg_s]).astype(np.float32)
    opacity = np.concatenate([av.opacities, np.ones(n_bg, np.float32)]).astype(np.float32)
    albedo_all = np.concatenate([albedo, bg_col]).astype(np.float32)
    n_all = np.concatenate([nrm, d_bg])
    wrap = 0.3
    diff = np.clip((n_all @ L + wrap) / (1 + wrap), 0, 1)

    prepared = load_prepared({"xyz": xyz, "rot": rot, "scale": scale,
                              "rgba": np.concatenate(
                                  [albedo_all, np.ones((len(xyz), 1), np.float32)], 1)},
                             use_sh=False)
    import torch
    bg_slice = slice(len(albedo), len(albedo) + n_bg)

    # 相机：脸高=1 归一空间。render_pose 是方画布 → 渲 max(W,H)=W 方图，
    # 竖向居中裁到 H。f 按脸高占 H 的 fill 比例反推距离。
    sq = max(W, H)
    dist = 3.4
    f = args.fill * H * dist
    K = np.array([[f, 0, sq / 2], [0, f, sq / 2], [0, 0, 1]], np.float64)
    target = np.array([0.0, 0.01, 0.0])

    project = Path(args.project)
    images = project / "images"
    images.mkdir(parents=True, exist_ok=True)
    vw = None
    if args.mp4:
        vw = cv2.VideoWriter(str(project / "preview.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))

    t0 = cv2.getTickCount()
    w2cs: list[np.ndarray] = []
    names: list[str] = []
    for i in range(args.frames):
        yaw = np.radians(-args.yaw + 2 * args.yaw * i / max(args.frames - 1, 1))
        pitch = np.radians(args.pitch)
        eye = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch),
                        np.cos(yaw) * np.cos(pitch)]) * dist
        w2c = lookat(eye, target, np.array([0.0, 1.0, 0.0]))
        w2cs.append(w2c)
        names.append(f"frame_{i:05d}.png")
        # 逐帧着色：Lambert wrap + Blinn 高光（V 随视角变化 → SH 有东西可学）
        V = (target - eye)
        V = V / (np.linalg.norm(V) + 1e-12)
        Hv = L + V
        Hv = Hv / (np.linalg.norm(Hv) + 1e-12)
        ndh = np.clip(n_all @ Hv, 0, 1)
        spec = (0.10 * ndh ** 40)[:, None] * np.array([0.95, 0.97, 1.0])
        col = albedo_all * (0.38 + 0.72 * diff)[:, None] + spec
        col[bg_slice] = bg_col * 0.9                     # 背景点平光
        col = np.clip(col, 0, 1).astype(np.float32)
        # use_sh=False → gsplat 直接把 colors 当原始 RGB（不做 SH 解码）
        prepared["colors"][:, :] = torch.tensor(col, device="cuda")
        img = render_pose(prepared, w2c, K, size=sq, ssaa=1,
                          bg=(0.045, 0.05, 0.06), denoise=False)
        # 方画布 → 竖向居中裁到 H
        y0 = (img.shape[0] - H) // 2
        img = img[y0:y0 + H]
        if img.shape[1] > W:
            x0 = (img.shape[1] - W) // 2
            img = img[:, x0:x0 + W]
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(images / f"frame_{i:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if vw is not None:
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if (i + 1) % 30 == 0:
            el = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
            print(f"[orbit] {i + 1}/{args.frames}  {el:.0f}s", flush=True)
    if vw is not None:
        vw.release()
    write_sparse_gt(names, w2cs, f, (W, H), project / "colmap" / "sparse_gt")
    print(f"[orbit] {args.frames} 帧 → {images}（{W}x{H}, ±{args.yaw}°）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
