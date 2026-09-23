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
    gaze_x/y    虹膜中心相对眼角的偏移（视线方向；478 点含虹膜时才启用）——
                眼球朝向跨帧漂移是眼区多模态（碎斑/重影）的第二根因，嘴已由
                mouth_open 窄带处理，视线由本特征处理
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
_P_L_EYE_INNER, _P_R_EYE_INNER = 133, 362     # 内眼角
_P_L_IRIS, _P_R_IRIS = 468, 473               # 虹膜中心（refine_landmarks 478 点）
FEATURE_DIMS = ("mouth_open", "mouth_wide", "brow_gap", "eye_open",
                "gaze_x", "gaze_y")


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
    """(478,2) 像素地标 → (6,) 表情特征（眼距归一）。

    px 不足 478（无虹膜）时 gaze 两维填 0——所有帧同样缺失时不影响聚类
    （MAD=0 的维度距离恒 0）；部分帧缺失说明检测质量差，由调用方过滤。"""
    px = np.asarray(px, np.float64)
    io = float(np.linalg.norm(px[_P_L_EYE_OUT] - px[_P_R_EYE_OUT]))
    io = max(io, 1e-6)
    mouth_open = np.linalg.norm(px[_P_LIP_UP] - px[_P_LIP_DN]) / io
    mouth_wide = np.linalg.norm(px[_P_MOUTH_L] - px[_P_MOUTH_R]) / io
    brow_gap = 0.5 * ((px[_P_L_LID, 1] - px[_P_L_BROW, 1])
                      + (px[_P_R_LID, 1] - px[_P_R_BROW, 1])) / io
    eye_open = 0.5 * (np.linalg.norm(px[159] - px[145])
                      + np.linalg.norm(px[386] - px[374])) / io
    if (len(px) >= _P_R_IRIS + 1
            and np.linalg.norm(px[_P_L_IRIS]) > 0
            and np.linalg.norm(px[_P_R_IRIS]) > 0):
        # 虹膜中心相对眼角中点的偏移（双眼平均；不试图从 2D 分离头偏航——
        # 聚类只需要"眼区外观跨帧一致"，偏航相关的分量由 MAD 聚类自然收紧，
        # 簇变小由 select_frames 的 tau 放宽兜底）
        l_mid = 0.5 * (px[_P_L_EYE_OUT] + px[_P_L_EYE_INNER])
        r_mid = 0.5 * (px[_P_R_EYE_OUT] + px[_P_R_EYE_INNER])
        gaze_x = float(0.5 * ((px[_P_L_IRIS, 0] - l_mid[0])
                              + (px[_P_R_IRIS, 0] - r_mid[0])) / io)
        gaze_y = float(0.5 * ((px[_P_L_IRIS, 1] - l_mid[1])
                              + (px[_P_R_IRIS, 1] - r_mid[1])) / io)
    else:
        gaze_x = gaze_y = 0.0
    return np.array([mouth_open, mouth_wide, brow_gap, eye_open, gaze_x, gaze_y])


def select_frames(images_dir: str | Path, registered: list[str],
                  detect, min_frames: int = 24, max_frames: int = 48,
                  tau: float = 1.0, max_yaw: float = 45.0,
                  mouth_open_cap: float = 0.15,
                  target_frames: int = 24, tau_cap: float = 2.0,
                  closed_lips: bool = True, gaze_weight: float = 2.0,
                  progress=None) -> FrameSelection:
    """对已配准帧跑人脸检测 → 主表情簇选择。

    detect(frame_bgr, t_ms) -> {"px": (478,2), "pose": [yaw,...]} | None
        （即 makeupstudio.tracker.FaceTracker.detect 的契约）
    tau：鲁棒距离阈（MAD 单位）。簇过小自动放宽（×1.5）。
    max_yaw：大偏航角下 MediaPipe 会对遮挡侧给幻觉地标（fit_makeup 同款教训），
    选帧阶段即排除。
    closed_lips：**闭嘴 canonical 策略**。唇红面是妆容主战场，闭嘴时是干净的
    单模态表面（唇纹清晰、无口腔/牙齿）；露齿帧混入后唇区被撕成多模态糊
    （唇糊/牙齿拖影的第一根因）。开启时嘴开度带取"最闭的 1/4 分位"窄带，
    帧数不足 min_frames 自动回退中位数带（旧行为）。canonical 头像因此是
    闭唇形态——涂口红正需要的姿态。
    gaze_weight：gaze 维在 MAD 距离里的加权（视线漂移是眼区多模态的第二
    根因，聚类对它收紧一档，眼球区保持全监督而不是挖洞）。
    target_frames：**训练视图数是 3DGS 质量的第一杠杆**。min_frames 达标后
    继续按 ×1.5 放宽 tau（至 tau_cap）直到簇 ≥ target_frames。
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
    if not feats_raw:
        raise RuntimeError("没有任何帧检测到人脸；检查视频质量/光线")

    opens = np.array([f[0] for f in feats_raw.values()])
    med_open = float(np.median(opens))
    # 嘴开度窄带，按优先级依次尝试。闭嘴 canonical：先取"最闭的 1/4 分位"
    # 窄带（唇红单模态、无牙齿），帧数不足回退中位数带（一致的部分张开状态
    # 也可训练；闭嘴与张嘴混训才是唇区多模态的元凶）。closed_lips=False 时
    # 保持旧的中位数带行为。
    caps = [float(np.clip(np.quantile(opens, 0.25) + 0.012, 0.035, 0.09))]
    if not closed_lips:
        caps = [float(np.clip(med_open + 0.025, 0.08, mouth_open_cap))]
    else:
        caps.append(float(np.clip(med_open + 0.025, 0.08, mouth_open_cap)))
    feats = {}
    for cap in caps:
        feats = {n: f for n, f in feats_raw.items() if f[0] <= cap}
        if progress:
            progress(1.0, f"嘴开度带 cap={cap:.3f} → {len(feats)} 帧")
        if len(feats) >= min_frames:
            break
    if len(feats) < min_frames:
        raise RuntimeError(
            f"仅 {len(feats)} 帧成功检测人脸（需 ≥{min_frames}）；检查视频质量/光线")

    F = np.stack(list(feats.values()))
    med = np.median(F, axis=0)
    mad = 1.4826 * np.median(np.abs(F - med), axis=0)
    scale = np.maximum(mad, 1e-4)
    dim_w = np.ones(F.shape[1])
    dim_w[-2:] = float(gaze_weight)             # gaze 两维：聚类收紧一档
    names_all = list(feats)

    chosen: list[str]
    for _ in range(4):
        dist = {n: float(np.max(np.abs(feats[n] - med) * dim_w / scale))
                for n in names_all}
        chosen = [n for n in names_all if dist[n] <= tau]
        if len(chosen) >= min_frames and (len(chosen) >= target_frames or tau >= tau_cap):
            break
        if len(chosen) < min_frames and _ >= 3:
            break
        tau = min(tau * 1.5, tau_cap)
    if len(chosen) < min_frames:
        # MAD 聚类收紧过头（gaze 加权 + 小 MAD 分母会把簇砍穿）：嘴开度带
        # 才是语义门，簇内全部帧兜底——绝不能把训练视图饿到个位数
        chosen = list(names_all)
    if len(chosen) > max_frames:                 # 簇过大：按距离择优再回时间序
        # 择优 = 表情+视线最一致（时间均匀抽会放回视线漂移帧，眼区多模态
        # 在密集化后以碎斑形式显形）；pose 覆盖由 max_yaw 门 + 带本身保证
        chosen = sorted(sorted(chosen, key=lambda n: dist[n])[:max_frames])
    dist = {n: float(np.max(np.abs(feats[n] - med) * dim_w / scale))
            for n in names_all}
    ref = min(chosen, key=lambda n: dist[n])
    return FrameSelection(names=sorted(chosen),
                          features={n: feats[n] for n in chosen},
                          px={n: pxs[n] for n in chosen},
                          ref=ref,
                          rejected={n: d for n, d in dist.items() if d > tau})
