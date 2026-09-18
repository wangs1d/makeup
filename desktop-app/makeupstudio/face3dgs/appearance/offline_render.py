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
from collections.abc import Callable
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
                     frame_fill: float = 0.72
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
    if face_h is not None:
        # 取景收放：投影脸高 px = fh·fx/d → d = fh·fx/(frame_fill·size)，
        # 并夹在原距离的 0.3~1.2 倍内（防止极端fh把相机推进/拉飞）
        d_fit = face_h * fx / (frame_fill * size)
        radius = float(np.clip(d_fit, 0.3 * radius, 1.2 * radius))
    v0 = V[np.linalg.norm(V, axis=1).argmin()]
    side = np.cross(up, v0)
    side = side / (np.linalg.norm(side) + 1e-12)
    fwd = np.cross(up, side)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    yaws = [float(np.degrees(np.arctan2(v @ side, v @ fwd))) for v in V]
    lo, hi = min(yaws) - yaw_pad_deg, max(yaws) + yaw_pad_deg
    if hi - lo < 10.0:                                       # 位姿过近：给最小环绕幅
        mid = (hi + lo) / 2
        lo, hi = mid - 5.0, mid + 5.0
    K = np.array([[fx, 0, size / 2], [0, fx, size / 2], [0, 0, 1]], np.float64)
    cams = []
    for yaw in np.linspace(lo, hi, n):
        a = np.radians(yaw)
        eye = center + (fwd * np.cos(a) + side * np.sin(a)) * radius
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
                ssaa: int = 2, bg: tuple[float, float, float] = (1.0, 1.0, 1.0)
                ) -> np.ndarray:
    """单视角渲染（SSAA 超采样抗锯齿后缩回）。返回 (size,size,3) uint8 RGB。"""
    import torch
    from gsplat import rasterization

    big = size * max(int(ssaa), 1)
    Kb = K.copy()
    Kb[:2] *= big / size
    w2c_t = torch.tensor(w2c, dtype=torch.float32, device="cuda")[None]
    Kt = torch.tensor(Kb, dtype=torch.float32, device="cuda")[None]
    r, _a, _i = rasterization(
        prepared["xyz"], prepared["rot"], prepared["scale"], prepared["opacity"],
        prepared["colors"], w2c_t, Kt, big, big,
        sh_degree=prepared["deg"], rasterize_mode="antialiased", packed=False,
        backgrounds=torch.tensor(bg, dtype=torch.float32, device="cuda")[None])
    img = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ---------------- 交付物 ----------------

def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (170, 34), (255, 255, 255), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return out


def render_stills(prepared: dict, cams: list, out_dir: Path, size: int = 1024,
                  ssaa: int = 2) -> list[Path]:
    """前/左/右三张定妆照。"""
    picks = {0: "front", len(cams) // 4: "left", 3 * len(cams) // 4: "right"}
    outs = []
    for idx, name in picks.items():
        w2c, K, _yaw = cams[idx]
        img = render_pose(prepared, w2c, K, size=size, ssaa=ssaa)
        p = out_dir / f"still_{name}.png"
        cv2.imwrite(str(p), cv2.cvtColor(_label(img, name), cv2.COLOR_RGB2BGR))
        outs.append(p)
    return outs


def render_turntable(prepared: dict, cams: list, out_dir: Path, size: int = 1024,
                     ssaa: int = 2, fps: int = 30,
                     progress: Callable[[float, str], None] | None = None) -> Path:
    """环绕视频：帧序列 + MP4（ping-pong 回放更自然，帧数翻倍）。"""
    frames: list[np.ndarray] = []
    cb = progress or (lambda f, m: None)
    for i, (w2c, K, _yaw) in enumerate(cams):
        frames.append(render_pose(prepared, w2c, K, size=size, ssaa=ssaa))
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
                   size: int = 1024, ssaa: int = 2, use_sh: bool = True) -> list[Path]:
    """素颜|妆后 并排对比（同一相机，逐像素可比）。"""
    pb, pm = load_prepared(base, use_sh=use_sh), load_prepared(made, use_sh=use_sh)
    picks = {0: "front", len(cams) // 4: "left", 3 * len(cams) // 4: "right"}
    outs = []
    for idx, name in picks.items():
        w2c, K, _yaw = cams[idx]
        b = _label(render_pose(pb, w2c, K, size=size, ssaa=ssaa), "bare")
        m = _label(render_pose(pm, w2c, K, size=size, ssaa=ssaa), "madeup")
        p = out_dir / f"compare_{name}.png"
        cv2.imwrite(str(p), cv2.cvtColor(np.concatenate([b, m], axis=1), cv2.COLOR_RGB2BGR))
        outs.append(p)
    return outs


def deliver(ply: str | Path, out_dir: str | Path, base_ply: str | Path | None = None,
            landmarks_path: str | Path | None = None, sfm_dir: str | Path | None = None,
            n_frames: int = 36, size: int = 1024, ssaa: int = 2, fps: int = 30,
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
        cams = orbit_from_poses(w2cs, Ks, center, n=n_frames, size=size,
                                face_h=face_height(made["xyz"], up, center))
    else:
        cams = orbit_synthetic(center, up, front,
                               face_height(made["xyz"], up, center),
                               n=n_frames, size=size)
    prepared = load_prepared(made, use_sh=use_sh)

    cb(0.05, "定妆照…")
    stills = render_stills(prepared, cams, out_dir, size=size, ssaa=ssaa)
    cb(0.35, "环绕视频…")
    mp4 = render_turntable(prepared, cams, out_dir, size=size, ssaa=ssaa, fps=fps,
                           progress=cb)
    outs = {"stills": [str(p) for p in stills], "turntable": str(mp4), "sh": use_sh}
    if base_ply is not None and Path(base_ply).exists():
        cb(0.9, "素颜对比…")
        cmps = render_compare(load_ply(base_ply), made, cams, out_dir,
                              size=size, ssaa=ssaa, use_sh=use_sh)
        outs["compare"] = [str(p) for p in cmps]
    (out_dir / "renders.json").write_text(
        json.dumps(outs, ensure_ascii=False, indent=1), encoding="utf-8")
    cb(1.0, f"完成 → {out_dir}")
    return outs
