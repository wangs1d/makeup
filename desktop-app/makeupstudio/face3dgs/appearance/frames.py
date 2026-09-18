"""frames — 表情一致性选帧（真·3DGS 训练的前提）。

根因：单目自拍视频里主体表情剧烈变化（吐舌/咧嘴/皱鼻）。3DGS 是静态场景模型，
把跨表情的帧混在一起训练，同一表面点要同时解释多种外观 → 梯度互相抵消，
densify 崩塌（历史产物 final.ply 只有 1918 个高斯，这是底模糊的第一根因）。

修法（GaussianAvatars/AvatarMakeup"canonical 表情"思想的静态版）：写实静态
头像不需要 4D——按 MediaPipe 表情特征聚类，只取"主表情簇"的帧做光度训练，
得到清晰的 canonical 头像；其余帧弃用（表情驱动走 FLAME rig，不归本链路）。

特征（全部用眼距归一，对镜头距离鲁棒）：
    mouth_open  上下内唇开合          mouth_wide  嘴角宽度（咧嘴/嘟嘴）
    brow_gap    眉-眼垂直距（挑眉）   eye_open    睁眼度（眯眼笑/瞪眼）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# MediaPipe FaceMesh 468 拓扑的稳定索引
_P_L_EYE_OUT, _P_R_EYE_OUT = 33, 263          # 眼角（尺度基准）
_P_LIP_UP, _P_LIP_DN = 13, 14                 # 上下内唇
_P_MOUTH_L, _P_MOUTH_R = 61, 291              # 嘴角
_P_L_BROW, _P_L_LID = 105, 159                # 左眉 / 左上眼睑
_P_R_BROW, _P_R_LID = 334, 386                # 右眉 / 右上眼睑
FEATURE_DIMS = ("mouth_open", "mouth_wide", "brow_gap", "eye_open")


@dataclass
class FrameSelection:
    names: list[str]                  # 入选帧（COLMAP 已配准 ∩ 磁盘存在）
    features: dict[str, np.ndarray]   # name → (4,) 特征
    px: dict[str, np.ndarray]         # name → (478,2) 像素地标（训练蒙版用）
    ref: str = ""                     # 参考帧（最接近主表情）
    rejected: dict[str, float] = field(default_factory=dict)  # 落选帧 → 距离

    def __len__(self) -> int:
        return len(self.names)


def expression_features(px: np.ndarray) -> np.ndarray:
    """(478,2) 像素地标 → (4,) 表情特征（眼距归一）。"""
    px = np.asarray(px, np.float64)
    io = float(np.linalg.norm(px[_P_L_EYE_OUT] - px[_P_R_EYE_OUT]))
    io = max(io, 1e-6)
    mouth_open = np.linalg.norm(px[_P_LIP_UP] - px[_P_LIP_DN]) / io
    mouth_wide = np.linalg.norm(px[_P_MOUTH_L] - px[_P_MOUTH_R]) / io
    brow_gap = 0.5 * ((px[_P_L_LID, 1] - px[_P_L_BROW, 1])
                      + (px[_P_R_LID, 1] - px[_P_R_BROW, 1])) / io
    eye_open = 0.5 * (np.linalg.norm(px[159] - px[145])
                      + np.linalg.norm(px[386] - px[374])) / io
    return np.array([mouth_open, mouth_wide, brow_gap, eye_open])


def select_frames(images_dir: str | Path, registered: list[str],
                  detect, min_frames: int = 24, max_frames: int = 48,
                  tau: float = 1.0, max_yaw: float = 45.0,
                  mouth_open_cap: float = 0.15,
                  progress=None) -> FrameSelection:
    """对已配准帧跑人脸检测 → 主表情簇选择。

    detect(frame_bgr, t_ms) -> {"px": (478,2), "pose": [yaw,...]} | None
        （即 makeupstudio.tracker.FaceTracker.detect 的契约）
    tau：鲁棒距离阈（MAD 单位）。簇过小自动放宽（×1.5，至多两轮）。
    max_yaw：大偏航角下 MediaPipe 会对遮挡侧给幻觉地标（fit_makeup 同款教训），
    选帧阶段即排除。
    mouth_open_cap：嘴开度绝对上限（眼距归一）。妆容主战场是唇，张嘴帧
    混入训练会把唇区撕成多模态糊——这是 3DGS 训练后唇部发糊的第一原因。
    """
    images_dir = Path(images_dir)
    feats: dict[str, np.ndarray] = {}
    feats_raw: dict[str, np.ndarray] = {}
    pxs: dict[str, np.ndarray] = {}
    t_ms = 0.0
    for i, name in enumerate(sorted(set(registered))):
        img_path = images_dir / name
        if not img_path.exists():
            continue
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        t_ms += 1000.0 / 25.0
        det = detect(img, t_ms)
        if det is None:
            continue
        if abs(float(det["pose"][0])) > max_yaw:
            continue
        feats_raw[name] = expression_features(det["px"][:468])
        pxs[name] = np.asarray(det["px"], np.float64)
    # 嘴开度窄带：贴中位数取（一致的部分张开状态可训练；闭嘴与张嘴混训
    # 才是唇区多模态的元凶）。带宽 ±0.025，下限 0.08 保底帧数。
    med_open = float(np.median([f[0] for f in feats_raw.values()]))
    cap = float(np.clip(med_open + 0.025, 0.08, mouth_open_cap))
    for k, (name, f) in enumerate(feats_raw.items()):
        if f[0] > cap:                      # 窄带外（含夸张张嘴）：唇区多模态元凶
            continue
        feats[name] = f
        if progress:
            progress((k + 1) / max(len(feats_raw), 1),
                     f"表情检测 {len(feats)}/{k + 1}")
    if len(feats) < min_frames:
        raise RuntimeError(
            f"仅 {len(feats)} 帧成功检测人脸（需 ≥{min_frames}）；检查视频质量/光线")

    F = np.stack(list(feats.values()))
    med = np.median(F, axis=0)
    mad = 1.4826 * np.median(np.abs(F - med), axis=0)
    scale = np.maximum(mad, 1e-4)
    names_all = list(feats)

    chosen: list[str]
    for _ in range(3):
        dist = {n: float(np.max(np.abs(feats[n] - med) / scale)) for n in names_all}
        chosen = [n for n in names_all if dist[n] <= tau]
        if len(chosen) >= min_frames:
            break
        tau *= 1.5
    if len(chosen) > max_frames:                 # 簇过大：按时间均匀降采样
        idx = np.linspace(0, len(chosen) - 1, max_frames).astype(int)
        chosen = [chosen[i] for i in sorted(idx)]
    dist = {n: float(np.max(np.abs(feats[n] - med) / scale)) for n in names_all}
    ref = min(chosen, key=lambda n: dist[n])
    return FrameSelection(names=sorted(chosen),
                          features={n: feats[n] for n in chosen},
                          px={n: pxs[n] for n in chosen},
                          ref=ref,
                          rejected={n: d for n, d in dist.items() if d > tau})
