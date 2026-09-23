"""lam_adapter — LAM（aigc3d，SIGGRAPH 2025）单图 → 3DGS 资产的门控适配器（P4）。

产品链路的"零门槛入口"：视频链路的资产化要先环绕拍摄（COLMAP 稀疏重建 +
gsplat 光度训练）；LAM 用**一张正面照**直接回归 canonical 高斯，把"想试一下"
的用户挡在门外的采集门槛降到一张照片。与 flame_avatar / guidance 同一套门控
约定：环境独立（LAM repo + 权重 + conda env lam），缺失时给可执行提示、
**主链路不阻断**。

诚实边界（产品文案不得含糊）：
    · LAM 底模是回归出来的先验人脸，个体相似度与细节（毛孔/痣/发丝/牙齿）低于
      视频链路的光度重建——单图入口用于快速试妆/预览，交付级仍走视频链路；
    · 单图只有一帧观测，语义妆区只能到 single_seg 级（无多视角投票），
      report.json 的 makeup_zones.level 会如实标出；
    · canonical 高斯是中性表情；照片里的张嘴/侧脸只影响语义蒙版的投影精度，
      不改变资产本身。

数据契约（能否复用整条妆容链路的关键）：LAM 输出的是 canonical 帧高斯云，而
下游（offline_render.infer_axes / uvbind.bind_uv / makeup_uv）一律要求"**地标与
点云同帧**"。本模块用 canonical 人脸模板（MediaPipe FaceMesh 468 序）在 LAM
点云上做 PCA 轴初值 + 鲁棒 ICP 相似配准，得到同帧的 468 地标随资产保存
（landmarks.npy）——于是 bind_uv 的 register() 残差 ≈ 0，UV 绑定/妆容目标场/
壳层/离线渲染全部原样复用，单图入口不需要在管线里开分支。

照片 → 高斯的语义提升：单图没有相机内参/位姿，用"canonical 模板 → 照片 2D
地标"的弱透视（正交 + 尺度）拟合拿到模板帧下的视角，再与"模板帧 → 点云帧"
复合，得到点云帧下与照片同视角的针孔相机，把 face-parsing 蒙版沿它投到每个
高斯（semantics.vote_regions / build_zones，单视角自然落在 single_seg 级）。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..fit_makeup import kabsch_similarity

REPO_DIR = Path(__file__).resolve().parents[3] / "research" / "LAM"
WEIGHTS_DIR = REPO_DIR / "pretrained"          # 按其 README 约定的权重目录
CONDA_EXE = Path(r"D:\miniconda3\Scripts\conda.exe")
CONDA_ENV = "lam"                              # 约定的独立环境名
DEFAULT_ENTRY = REPO_DIR / "inference.py"
LAM_PLY_NAME = "gaussian.ply"
# canonical 地标模板（LAM 帧）：存在时直接读，否则由本项目 canonical 模板配准而来
TEMPLATE_PATHS = (REPO_DIR / "assets" / "canonical_landmarks.npy",
                  REPO_DIR / "assets" / "canonical_landmarks.npz")

N_LM = 468                 # MediaPipe FaceMesh / canonical 模板一一对应
ICP_ITERS = 4              # 粗配准迭代（PCA 初值 → 最近点 → 相似变换）
ICP_FULL_ITERS = 2         # 抽稀点云上选优后，回全量点云的精化迭代
ICP_MAX_POINTS = 4000      # 粗配准用的抽稀点数（468×N 距离矩阵是开销大头）
ICP_TRIM = 0.25            # 每轮裁掉残差最大的 25%（头发/背景/口腔面是离群）
ICP_BOX_TRIMS = (1.0, 0.7)  # 尺度初值候选：全量包围盒比 + 半径裁剪后的下界估计
WP_ITERS = 3               # 弱透视拟合迭代（表情差异是离群，靠裁剪压住）
WP_TRIM = 0.20
DEPTH_SPAN = 10.0          # 弱透视近似深度 = 10×地标尺度（深度变化 <5%）


# ---------------- 环境门控 ----------------

@dataclass
class LamStatus:
    repo: bool
    weights: bool
    conda_env: bool
    missing_hint: str = ""

    @property
    def ok(self) -> bool:
        return self.repo and self.weights and self.conda_env


def status() -> LamStatus:
    repo = REPO_DIR.is_dir()
    weights = WEIGHTS_DIR.is_dir() and any(
        p.suffix in (".pth", ".ckpt", ".safetensors")
        for p in WEIGHTS_DIR.rglob("*") if p.is_file())
    conda = CONDA_EXE.is_file() and _env_exists()
    hints = []
    if not repo:
        hints.append(f"git clone https://github.com/aigc3d/LAM.git 到 {REPO_DIR}")
    if repo and not weights:
        hints.append(f"按该仓库 README 下载 LAM 权重到 {WEIGHTS_DIR}/")
    if not conda:
        hints.append(f"conda create -n {CONDA_ENV} python=3.10 并按其 requirements.txt"
                     " 装依赖")
    return LamStatus(repo, weights, conda, "；".join(hints))


def _env_exists() -> bool:
    try:
        r = subprocess.run([str(CONDA_EXE), "env", "list"], capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return CONDA_ENV in (r.stdout or "")


# ---------------- 推理（独立环境） ----------------

def predict(image: str | Path, out_dir: str | Path, timeout: int = 1800) -> Path:
    """单张照片 → canonical 高斯 ply（在独立 conda 环境里跑 LAM 推理）。

    环境不完整时抛 RuntimeError（消息本身即操作指引）——调用方据此优雅降级，
    视频主链路不受影响。"""
    st = status()
    if not st.ok:
        raise RuntimeError(f"LAM 环境不完整：{st.missing_hint}")
    image, out_dir = Path(image), Path(out_dir)
    if not image.is_file():
        raise RuntimeError(f"照片不存在：{image}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(CONDA_EXE), "run", "-n", CONDA_ENV, "--no-capture-output", "python",
           str(DEFAULT_ENTRY), "--image", str(image), "--out", str(out_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=str(REPO_DIR))
    ply = _find_ply(out_dir)
    if r.returncode != 0 or ply is None:
        raise RuntimeError(f"LAM 推理失败（exit={r.returncode}）："
                           f"{(r.stderr or r.stdout or '')[-800:]}")
    return ply


def _find_ply(out_dir: Path) -> Path | None:
    """LAM 产物定位：约定名优先，否则取输出目录里最新的 ply。"""
    direct = out_dir / LAM_PLY_NAME
    if direct.is_file():
        return direct
    plys = sorted(out_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime)
    return plys[-1] if plys else None


# ---------------- canonical 模板 / 相似配准 ----------------

def canonical_template(path: str | Path | None = None) -> np.ndarray:
    """LAM 帧下的 canonical 468 地标模板 (468,3)。

    path / TEMPLATE_PATHS 命中即用（LAM 仓库自带的地标导出）；否则回退到本项目
    canonical 人脸模板顶点——后者只在两套 canonical 帧方向一致时才可直用，
    不一致时由 fit_similarity 的相似配准把它整体搬到 LAM 帧。"""
    for p in ([Path(path)] if path else list(TEMPLATE_PATHS)):
        if p.is_file():
            data = np.load(p)
            arr = data["landmarks"] if getattr(data, "files", None) else data
            arr = np.asarray(arr, np.float64)
            if arr.shape[0] >= N_LM and arr.shape[1] >= 3:
                return arr[:N_LM, :3]
    from .uvbind import canonical_landmarks         # 延迟导入（重依赖在 fit_makeup）
    return np.asarray(canonical_landmarks(), np.float64)[:N_LM, :3]


def nearest_points(cloud_xyz: np.ndarray, query: np.ndarray) -> tuple[np.ndarray,
                                                                    np.ndarray]:
    """query (m,3) → 点云最近点（索引 + 距离）。分块 numpy，避免 m×n 距离矩阵。"""
    P = np.asarray(cloud_xyz, np.float64)
    q = np.asarray(query, np.float64)
    idx = np.empty(len(q), np.int64)
    dist = np.empty(len(q), np.float64)
    step = max(1, int(4e6 // max(len(P), 1)))
    for i in range(0, len(q), step):
        d = np.linalg.norm(P[None, :, :] - q[i:i + step, None, :], axis=2)
        j = d.argmin(1)
        idx[i:i + step] = j
        dist[i:i + step] = d[np.arange(len(j)), j]
    return idx, dist


def _basis(up: np.ndarray, front: np.ndarray) -> np.ndarray:
    """(up, front) → 右手正交基 [side, up, front]（列 = 世界系下的局部轴）。"""
    up = np.asarray(up, np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)
    front = np.asarray(front, np.float64)
    front = front - up * (front @ up)
    front = front / (np.linalg.norm(front) + 1e-12)
    side = np.cross(up, front)
    side = side / (np.linalg.norm(side) + 1e-12)
    return np.stack([side, up, front], axis=1)


def _radial_trim(points: np.ndarray, q: float) -> np.ndarray:
    """按"到中位点的半径"保留内层 q 比例（离群点半径远大于脸 → 被裁掉）。"""
    if q >= 1.0:
        return points
    d = np.linalg.norm(points - np.median(points, 0), axis=1)
    return points[d <= float(np.quantile(d, q))]


def _box_scale(cloud: np.ndarray, template: np.ndarray, q: float = 1.0) -> float:
    """包围盒对角线比 = 尺度初值（对点密度不敏感）。

    顶点集与面片均匀采样覆盖的是**同一张曲面**，极值（包围盒）因此一致，尺度比
    无偏；而中位径向距离比会因两套点的密度分布不同偏 40%+（实测 3.89 vs 真值
    2.5），错尺度会让 ICP 掉进错误局部极小。q<1 时先做半径裁剪——头发/脖颈离群
    只能把包围盒**撑大**，故裁剪版给出下界估计，与全量版一起作为候选。"""
    a = float(np.linalg.norm(np.ptp(_radial_trim(cloud, q), axis=0)))
    b = float(np.linalg.norm(np.ptp(_radial_trim(template, q), axis=0)))
    return a / max(b, 1e-9)


def _icp_refine(cloud: np.ndarray, template: np.ndarray, s: float, R: np.ndarray,
                t: np.ndarray, iters: int, trim: float
                ) -> tuple[float, np.ndarray, np.ndarray]:
    """最近点 → 裁剪 → 相似变换（kabsch）迭代精化。"""
    for _ in range(max(1, int(iters))):
        idx, d = nearest_points(cloud, template @ (s * R).T + t)
        keep = d <= float(np.quantile(d, 1.0 - trim))
        if int(keep.sum()) < 8:
            break
        s, R, t = kabsch_similarity(template[keep], cloud[idx[keep]])
    return s, R, t


def _trimmed_rmse(d: np.ndarray, trim: float) -> float:
    """残差的内层 (1-trim) 均方根——离群点不参与（评分用）。"""
    k = max(8, int(len(d) * (1.0 - trim)))
    return float(np.sqrt(np.mean(np.sort(d)[:min(k, len(d))] ** 2)))


def fit_similarity(cloud_xyz: np.ndarray, template: np.ndarray,
                   iters: int = ICP_ITERS, trim: float = ICP_TRIM
                   ) -> tuple[float, np.ndarray, np.ndarray, float]:
    """canonical 模板 → 点云帧的相似变换 (s,R,t)：PCA 轴初值 + 鲁棒 ICP 精化。

    满足 template @ (sR).T + t ≈ 点云表面。点云只有表面点、没有地标，PCA 给不出
    上/前的**符号**（脸不是整头，PCA 轴还偏 ~20°），故 4 种符号组合各做一次 ICP。
    尺度初值取包围盒比（点密度无关）；头发/脖颈离群会撑大包围盒，故再叠加一份
    半径裁剪后的下界估计。选优按**尺度归一化**的裁剪残差——否则"整体缩小"的
    错误局部极小会因为残差绝对值小而胜出。粗配准在抽稀点云上做（468×N 距离矩阵
    是唯一开销大头），胜者回全量点云精化。返回 (s, R, t, 全量点云 rmse)。"""
    P = np.asarray(cloud_xyz, np.float64)
    X = np.asarray(template, np.float64)
    cp, cx = np.median(P, 0), np.median(X, 0)
    BX = _basis(X[10] - X[152], X[1] - cx)
    from .offline_render import infer_axes          # PCA 轴（无地标也可用）
    _, up_p, front_p = infer_axes(P)
    step = max(1, len(P) // ICP_MAX_POINTS)
    Ps = P[::step] if step > 1 else P

    best = None
    for q in ICP_BOX_TRIMS:
        s0 = _box_scale(P, X, q)
        for su in (1.0, -1.0):
            for sf in (1.0, -1.0):
                R = _basis(su * up_p, sf * front_p) @ BX.T
                t = cp - s0 * (R @ cx)
                s, R, t = _icp_refine(Ps, X, s0, R, t, iters, trim)
                _idx, d = nearest_points(Ps, X @ (s * R).T + t)
                score = _trimmed_rmse(d, trim) / max(abs(s), 1e-12)
                if best is None or score < best[0]:
                    best = (score, s, R, t)
    _, s, R, t = best
    if step > 1:                                    # 回全量点云精化
        s, R, t = _icp_refine(P, X, s, R, t, ICP_FULL_ITERS, trim)
    _idx, d = nearest_points(P, X @ (s * R).T + t)
    return s, R, t, float(np.sqrt((d ** 2).mean()))


def landmarks_in_frame(cloud_xyz: np.ndarray, template: np.ndarray | None = None,
                       path: str | Path | None = None
                       ) -> tuple[np.ndarray, tuple[float, np.ndarray, np.ndarray],
                                  float]:
    """canonical 468 地标 → 点云帧。下游一律要求"地标与点云同帧"。

    返回 (landmarks (468,3), (s, R, t), rmse)。"""
    X = canonical_template(path) if template is None else np.asarray(template, np.float64)
    s, R, t, rmse = fit_similarity(cloud_xyz, X)
    return X @ (s * R).T + t, (s, R, t), rmse


def load_canonical(ply: str | Path, landmarks_out: str | Path | None = None,
                   template: np.ndarray | None = None, path: str | Path | None = None
                   ) -> tuple[dict, np.ndarray, dict]:
    """LAM 产物 ply → (cloud, 点云帧 468 地标, 配准信息)。

    landmarks_out 给定时落 landmarks.npy（随资产保存，与视频链路同名同格式）。"""
    from ..splat_io import read_ply
    cloud = read_ply(ply)
    lm, (s, _R, _t), rmse = landmarks_in_frame(cloud["xyz"], template, path)
    if landmarks_out is not None:
        Path(landmarks_out).parent.mkdir(parents=True, exist_ok=True)
        np.save(landmarks_out, lm)
    return cloud, lm, {"scale": float(s), "landmarks_rmse": float(rmse),
                       "landmarks": int(len(lm))}


# ---------------- 照片视角（弱透视拟合 → 点云帧针孔相机） ----------------

def weak_perspective_fit(template: np.ndarray, px: np.ndarray,
                         iters: int = WP_ITERS, trim: float = WP_TRIM
                         ) -> tuple[float, np.ndarray, np.ndarray]:
    """canonical 3D 模板 → 照片 2D 地标的弱透视拟合：px ≈ s·(R @ X)[:, :2] + t。

    单图无内参/位姿，弱透视是标准可解形式：中心化线性最小二乘 → 行向量正交
    化 → 残差裁剪迭代（表情差异在唇/眼索引上是离群）。返回 (s, R, t)。"""
    X = np.asarray(template, np.float64)
    p = np.asarray(px, np.float64)[:len(X), :2]
    keep = np.ones(len(X), bool)
    s, R, t = 1.0, np.eye(3), np.zeros(2)
    for _ in range(max(1, int(iters))):
        Xs, ps = X[keep], p[keep]
        if len(Xs) < 8:
            break
        cx, cp = Xs.mean(0), ps.mean(0)
        A, b = Xs - cx, ps - cp
        M = b.T @ A @ np.linalg.inv(A.T @ A + 1e-9 * np.eye(3))
        s = float((np.linalg.norm(M[0]) + np.linalg.norm(M[1])) / 2.0)
        if s < 1e-12:
            return 1.0, np.eye(3), np.zeros(2)
        Rr = M / s
        R3 = np.stack([Rr[0], Rr[1], np.cross(Rr[0], Rr[1])])
        u, _sv, vt = np.linalg.svd(R3)
        R = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
        t = cp - s * (R @ cx)[:2]
        res = np.linalg.norm(s * (X @ R.T)[:, :2] + t - p, axis=1)
        keep = res <= float(np.quantile(res, 1.0 - trim))
    return s, R, t


def photo_camera(landmarks: np.ndarray, px: np.ndarray,
                 srt: tuple[float, np.ndarray, np.ndarray],
                 template: np.ndarray | None = None) -> tuple[np.ndarray,
                                                              np.ndarray,
                                                              np.ndarray]:
    """照片视角在**点云帧**里的相机 (R, t, K)。

    照片 = "canonical 模板的某个视角"：弱透视给出模板帧下 (s2,R2,t2)，与模板帧
    → 点云帧的 (s1,R1,t1) 复合即得点云帧下的正交相机；投影/采样走的是针孔模型，
    故再等价换算成"远处小视场"（depth = 10×地标尺度，深度变化 <5%）的透视相机。
    这样 semantics.project_points/vote_regions 可以直接消费。"""
    X = canonical_template() if template is None else np.asarray(template, np.float64)
    lm = np.asarray(landmarks, np.float64)
    s2, R2, t2 = weak_perspective_fit(X, px)
    s1, R1, t1 = srt
    R = R2 @ np.asarray(R1, np.float64).T
    so = s2 / max(float(s1), 1e-12)
    span = float(np.linalg.norm(np.ptp(lm, axis=0)))
    d = DEPTH_SPAN * max(span, 1e-6)
    center = lm.mean(0)
    C = center - d * R[2]                     # 光心：让 center 落在深度 d 上
    t = -R @ C
    f = so * d
    cxy = np.asarray(t2, np.float64) + so * (R @ (C - np.asarray(t1, np.float64)))[:2]
    K = np.array([[f, 0.0, cxy[0]], [0.0, f, cxy[1]], [0.0, 0.0, 1.0]], np.float64)
    return R, t, K


def detect_landmarks(img_bgr: np.ndarray) -> np.ndarray | None:
    """照片 → (478,2) 像素地标。MediaPipe 缺失/未检出返回 None（调用方降级）。"""
    try:
        from ...tracker import FaceTracker
        tracker = FaceTracker(smooth=False)
    except Exception:                                # 依赖缺失/模型文件缺失
        return None
    try:
        det = tracker.detect(img_bgr, 0.0)
    except Exception:
        return None
    finally:
        tracker.close()
    return None if det is None else np.asarray(det["px"], np.float64)


# ---------------- 语义观测妆区（单视角 → single_seg） ----------------

def photo_zones(cloud: dict, landmarks: np.ndarray, image: str | Path,
                regions: tuple[str, ...], px: np.ndarray | None = None,
                parser=None, srt: tuple | None = None,
                template: np.ndarray | None = None,
                region_scale: dict[str, float] | None = None,
                feather_px: dict[str, float] | None = None,
                normals: np.ndarray | None = None,
                progress=None):
    """单张照片的语义观测妆区：face-parsing 蒙版沿照片视角投到每个高斯。

    只有一帧观测 → 逐 splat 可见视角数为 1 → build_zones 自动落到 single_seg
    级（report 如实留痕，不冒充多视角投票）。门控：照片读不到 / 未检出人脸 /
    face-parsing 不可用 → 返回 None，调用方保持几何兜底（landmark_band /
    uv_template），主链路不阻断。px / parser / normals 可注入（离线单测不依赖
    mediapipe / torch）。"""
    from .semantics import build_zones, image_masks

    cb = progress or (lambda *a: None)
    img = cv2.imread(str(image))
    if img is None:
        cb("lam", 1.0, f"照片读取失败：{image}")
        return None
    if px is None:
        px = detect_landmarks(img)
    if px is None:
        cb("lam", 1.0, "照片未检出人脸（MediaPipe），保持几何兜底")
        return None
    xyz = np.asarray(cloud["xyz"], np.float64)
    if srt is None:
        srt = landmarks_in_frame(xyz, template)[1]
    R, t, K = photo_camera(landmarks, np.asarray(px, np.float64), srt, template)
    if parser is None:
        from ...parser import FaceParser
        parser = FaceParser()
    if not parser.available():
        hint = getattr(parser, "load_error", None)
        cb("lam", 1.0, f"face-parsing 不可用（保持几何兜底）{'：' + hint if hint else ''}")
        return None
    parsed = parser.parse(img)
    if not parsed:
        cb("lam", 1.0, "照片语义解析失败（保持几何兜底）")
        return None
    masks = image_masks(img, parsed, tuple(regions), feather_px)
    if not masks:
        cb("lam", 1.0, "照片无可用妆区蒙版（保持几何兜底）")
        return None
    h, w = img.shape[:2]
    view = {"name": Path(image).name, "R": R, "t": t, "K": K,
            "masks": masks, "size": (w, h)}
    if normals is None:
        try:                            # 背面剔除：后脑勺高斯会投进脸部轮廓内
            from .normals import axis_normals
            normals = axis_normals(np.asarray(cloud["rot"], np.float64),
                                   np.asarray(cloud["scale"], np.float64))
        except Exception:
            pass
    cb("lam", 1.0, f"照片语义观测（{len(masks)} 区域，单视角 → single_seg）")
    return build_zones(xyz, [view], tuple(regions),
                       region_scale=region_scale, normals=normals,
                       progress=cb)