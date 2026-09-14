"""parser — 像素级人脸解析（HuggingFace jonathandinu/face-parsing，SegFormer-B0）。

实时妆容的"假"有一半来自蒙版边缘：468 个 landmark 连多边形，唇形/眼形永远
差几个像素。人脸解析是逐像素分割（19 类：皮肤/上下唇/眼/眉/鼻/头发…），
唇蒙版严丝合缝，还天然排除眉毛/眼睛（粉底不用再手抠腔洞）。

设计约束：
    · 模型可缺省 —— torch/transformers/权重任一缺失时 FaceParser.available()
      为 False，调用方回退 landmark 蒙版路径（功能不劣化，只是边缘不贴）；
    · GPU 优先 —— CUDA 可用时整帧解析 ~10-20ms；CPU ~200ms，调用方应降频
      （解析低频跑 + 蒙版随 landmark 相似变换对齐当前帧，见 warp_masks）。
"""
from __future__ import annotations

import os

import numpy as np

# 国内网络：默认走 hf-mirror；权重已缓存时离线加载（避免启动时联网检查卡死）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

HF_MODEL_ID = "jonathandinu/face-parsing"
INPUT_SIZE = 512

# CelebAMask-HQ 类别名 → 我们关心的语义组（名字小写包含匹配，跨版本稳）
_GROUP_OF = {
    "lips": ("u_lip", "l_lip"),
    "mouth": ("mouth",),
    "skin": ("skin", "nose"),          # 鼻梁也是要上粉底的皮肤
    "eye_l": ("l_eye",),
    "eye_r": ("r_eye",),
    "brow_l": ("l_brow",),
    "brow_r": ("r_brow",),
    "hair": ("hair",),
    "cloth": ("cloth", "neck_l"),
    "neck": ("neck",),
    "ear": ("l_ear", "r_ear", "ear_r"),
}


def _group_labels(id2label: dict[int, str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for lid, name in id2label.items():
        low = str(name).lower()
        for group, keys in _GROUP_OF.items():
            if any(k in low for k in keys):
                groups.setdefault(group, []).append(int(lid))
    return groups


class FaceParser:
    """SegFormer 人脸解析。load() 幂等；失败后 available() 为 False。"""

    def __init__(self, device: str | None = None):
        self._model = None
        self._proc = None
        self._groups: dict[str, list[int]] = {}
        self._device = device or "cuda"
        self._load_error: str | None = None

    def available(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            self.load()
        except Exception as e:                     # 权重下载失败/无网络/依赖缺失
            self._load_error = f"{type(e).__name__}: {e}"
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        # 先试离线（缓存命中零延迟），失败再联网下载
        try:
            self._proc = SegformerImageProcessor.from_pretrained(HF_MODEL_ID, local_files_only=True)
            model = SegformerForSemanticSegmentation.from_pretrained(HF_MODEL_ID,
                                                                     local_files_only=True)
        except Exception:
            self._proc = SegformerImageProcessor.from_pretrained(HF_MODEL_ID)
            model = SegformerForSemanticSegmentation.from_pretrained(HF_MODEL_ID)
        if self._device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"
        self._model = model.to(self._device).eval()
        self._torch = torch
        id2label = getattr(model.config, "id2label", {}) or {}
        self._groups = _group_labels({int(k): v for k, v in id2label.items()})

    def parse(self, frame_bgr: np.ndarray) -> dict[str, np.ndarray] | None:
        """BGR 帧 → 语义组蒙版（原图分辨率，float32 0/1）。不可用时返回 None。"""
        if not self.available():
            return None
        import cv2
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # 256 输入：蒙版会上采样后做软混合，512 的分辨率收益不值 4 倍耗时
        # （CPU 512 ≈ 960ms → 256 ≈ 240ms；GPU 上两者都 <50ms）
        inputs = self._proc(images=rgb, return_tensors="pt",
                            size={"height": INPUT_SIZE // 2, "width": INPUT_SIZE // 2})
        with self._torch.no_grad():
            logits = self._model(pixel_values=inputs["pixel_values"].to(self._device)).logits
        seg = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)   # (H/4, W/4)
        out: dict[str, np.ndarray] = {}
        for group, lids in self._groups.items():
            if not lids:
                continue
            m = np.isin(seg, lids).astype(np.float32)
            if m.sum() < 4:
                continue
            out[group] = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return out or None


def estimate_face_affine(src_lm: np.ndarray, dst_lm: np.ndarray) -> np.ndarray:
    """两帧 landmark → 相似变换（部分仿射）2×3 矩阵。用于把缓存的解析蒙版
    对齐到当前帧（解析低频跑时，人脸在帧间移动/转头，蒙版要跟着走）。"""
    import cv2
    n = min(len(src_lm), len(dst_lm))
    # 取轮廓+五官的关键子集即可稳定估计相似变换，RANSAC 抗个别地标抖动
    idx = np.linspace(0, n - 1, min(n, 100)).astype(int)
    m, _ = cv2.estimateAffinePartial2D(src_lm[idx].astype(np.float32),
                                       dst_lm[idx].astype(np.float32),
                                       method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if m is None:
        m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return m


def warp_masks(masks: dict[str, np.ndarray], affine: np.ndarray,
               size: tuple[int, int]) -> dict[str, np.ndarray]:
    """把解析蒙版从"解析时的帧"变换到"当前帧"坐标系。"""
    import cv2
    out = {}
    for k, m in masks.items():
        out[k] = cv2.warpAffine(m, affine, (size[1], size[0]),
                                flags=cv2.INTER_LINEAR, borderValue=0.0)
    return out
