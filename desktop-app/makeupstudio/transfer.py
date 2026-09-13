"""transfer — 妆容迁移（EleGANt）门控适配器。

EleGANt（ECCV22）把一张真实妆品参考图迁移到用户照片上，是"妆效与实际
妆品效果对齐"的天花板方案。逐帧跑不实时——定位是离线妆效预览与实时端
取色参数的标定来源，不进摄像头链路。

对外能力：
    · transfer(source, reference)          → 妆后 BGR 图；
    · compare_image(src, ref, result)      → 素颜|参考|迁移后 三联对比图；
    · transfer_and_save(src, ref, out_dir) → 迁移 + 对比图/结果图落盘；
    · imread_unicode / imwrite_unicode     → 中文路径安全读写。

入口：CLI `desktop-app/tools/makeup_transfer.py`；GUI 在主应用「妆效预览」
页的「参考图迁移」面板。

依赖（见 research/INTEGRATION.md，全部需手动准备）：
    · research/EleGANt 仓库（gitclone.com 镜像可加速）；
    · checkpoint：README 里 GDrive 的 sow_pyramid_a5_e3d2_remapped.pth
      → research/EleGANt/ckpts/；
    · 辅助权重：dlib 68 点 + BiSeNet resnet.pth → research/EleGANt/faceutils/；
    · torch（与 face-parsing 共用）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2] / "research" / "EleGANt"
CKPT = REPO_DIR / "ckpts" / "sow_pyramid_a5_e3d2_remapped.pth"
DLIB_DAT = REPO_DIR / "faceutils" / "dlibutils" / "shape_predictor_68_face_landmarks.dat"
BISENET_PTH = REPO_DIR / "faceutils" / "mask" / "resnet.pth"

_inference = None                                  # 进程内缓存（模型加载 ~数秒）


def find_checkpoint() -> Path | None:
    """checkpoint 查找顺序：环境变量 ELEGANT_CKPT → 官方默认文件名 →
    ckpts/ 下任意 .pth（从 GDrive 下载后原样丢进 ckpts/ 即可）。"""
    import os
    env = os.environ.get("ELEGANT_CKPT")
    if env and Path(env).is_file():
        return Path(env)
    if CKPT.is_file():
        return CKPT
    if (REPO_DIR / "ckpts").is_dir():
        pths = sorted((REPO_DIR / "ckpts").glob("*.pth"))
        if pths:
            return pths[0]
    return None


def missing_assets() -> list[str]:
    """资产文件级缺失清单（不导入重依赖，秒回）。全部就绪时为空。"""
    if not REPO_DIR.is_dir():
        return [f"git clone https://github.com/Chenyu-Yang-2000/EleGANt.git 到 {REPO_DIR}"]
    missing = []
    if find_checkpoint() is None:
        missing.append(f"下载 EleGANt README 的模型权重 → {CKPT}（或设环境变量 ELEGANT_CKPT 指向 .pth）")
    if not DLIB_DAT.exists():
        missing.append(f"dlib 68 点权重（dlib.net/files）→ {DLIB_DAT}")
    if not BISENET_PTH.exists():
        missing.append(f"BiSeNet 解析权重（GDrive 154JgKpz）→ {BISENET_PTH}")
    return missing


def status() -> tuple[bool, str]:
    """资产与运行时依赖就绪检查。返回 (ok, 缺失提示)。"""
    missing = missing_assets()
    if not missing:
        try:
            import torch                                  # noqa: F401
            sys.path.insert(0, str(REPO_DIR))
            try:
                from training.inference import Inference  # noqa: F401
            finally:
                sys.path.remove(str(REPO_DIR))
        except ImportError as e:
            missing.append(f"依赖缺失：{e}")
    return (not missing), "；".join(missing)


def transfer(source_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
    """source（素颜/淡妆用户照）+ reference（妆效参考图）→ 妆后 BGR 图。"""
    global _inference
    ok, hint = status()
    if not ok:
        raise RuntimeError(f"EleGANt 未就绪：{hint}")
    if _inference is None:
        sys.path.insert(0, str(REPO_DIR))
        from training.config import get_config
        from training.inference import Inference
        import torch

        ckpt = str(find_checkpoint())

        class _Args:                                   # demo.py 的最小参数面
            name = "makeupstudio"
            gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
            device = torch.device(gpu)
            load_path = ckpt
            save_folder = str(REPO_DIR / "result")

        _inference = Inference(get_config(), _Args, ckpt)

    from PIL import Image
    src = Image.fromarray(cvtColor(source_bgr))
    ref = Image.fromarray(cvtColor(reference_bgr))
    result = _inference.transfer(src, ref, postprocess=True)
    if result is None:
        raise RuntimeError(
            "迁移失败：EleGANt 未在输入中检测到人脸（素颜照与参考图都必须是清晰正面人脸）")
    out = np.array(result.resize((source_bgr.shape[1], source_bgr.shape[0])))
    return cvtColor(out, to_bgr=True)


def cvtColor(bgr: np.ndarray, to_bgr: bool = False) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB if not to_bgr else cv2.COLOR_RGB2BGR)


# cv2 的 Win32 读取走 ANSI 路径，中文/非 ASCII 路径会静默返回 None，须走内存编解码
def imread_unicode(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读取图片：{path}")
    return img


def imwrite_unicode(path: str | Path, img: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    buf.tofile(str(p))


# ---------------- 迁移对比图 ----------------

_BANNER = 44                                        # 每栏顶部标签条高度（px）
_GAP = 4                                            # 栏间白色分隔（px）

_ASCII_FALLBACK = {"素颜": "bare", "参考妆效": "reference", "迁移后": "transfer"}


def _label_font(px: int):
    """中文标签字体（Windows 优先微软雅黑；无 CJK 字体时返回 None 走 ASCII）。"""
    from PIL import ImageFont
    for candidate in (
        r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, px)
        except OSError:
            continue
    return None


def compare_image(source_bgr: np.ndarray, reference_bgr: np.ndarray,
                  result_bgr: np.ndarray,
                  labels: tuple[str, str, str] = ("素颜", "参考妆效", "迁移后"),
                  height: int = 560) -> np.ndarray:
    """三联对比图：素颜 | 参考妆效 | 迁移后（各栏等高、保持纵横比，带标签条）。"""
    panels = []
    font = _label_font(22)
    for img, text in zip((source_bgr, reference_bgr, result_bgr), labels):
        h, w = img.shape[:2]
        scale = height / h
        panel = cv2.resize(img, (max(1, round(w * scale)), height),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)
        strip = np.full((_BANNER, panel.shape[1], 3), 255, np.uint8)
        if font is not None:
            from PIL import Image, ImageDraw
            pil = Image.fromarray(strip)
            draw = ImageDraw.Draw(pil)
            box = draw.textbbox((0, 0), text, font=font)
            draw.text(((panel.shape[1] - (box[2] - box[0])) / 2 - box[0],
                       (_BANNER - (box[3] - box[1])) / 2 - box[1]),
                      text, font=font, fill=(29, 29, 31))
            strip = np.array(pil)
        else:                                       # 无 CJK 字体的环境退回 ASCII
            cv2.putText(strip, _ASCII_FALLBACK.get(text, text),
                        (10, _BANNER - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (29, 29, 31), 2, cv2.LINE_AA)
        panels.append(np.vstack([strip, panel]))

    width = sum(p.shape[1] for p in panels) + _GAP * (len(panels) - 1)
    canvas = np.full((height + _BANNER, width, 3), 255, np.uint8)
    x = 0
    for p in panels:
        canvas[:, x:x + p.shape[1]] = p
        x += p.shape[1] + _GAP
    return canvas


def transfer_and_save(source_bgr: np.ndarray, reference_bgr: np.ndarray,
                      out_dir: str | Path, height: int = 560,
                      stem: str | None = None) -> tuple[Path, Path]:
    """迁移并落盘：返回（三联对比图路径, 仅妆后结果路径）。文件名带时间戳。"""
    out = Path(out_dir)
    result = transfer(source_bgr, reference_bgr)
    name = stem or time.strftime("%Y%m%d_%H%M%S")
    cmp_path = out / f"transfer_{name}_compare.png"
    res_path = out / f"transfer_{name}_result.png"
    imwrite_unicode(cmp_path, compare_image(source_bgr, reference_bgr, result, height=height))
    imwrite_unicode(res_path, result)
    return cmp_path, res_path
