"""guidance — Stable-Makeup 妆容 guidance 生成器门控适配器（P1）。

参数化蒙版（fit_makeup.apply_makeup）表达力有上限：真实化妆品的珠光、闪片、
咬唇渐变烘不进多边形 + 色带。P1 路线把"妆容外观"交给 2D SOTA 妆容迁移模型
（Stable-Makeup，SIGGRAPH 2025，diffusion）：参考妆照 + 素颜渲染图 → guidance 图，
再由 fit_makeup.apply_guidance 多视角投影采样回写 splat 颜色。

与 flame_avatar.py 同一套门控约定：训练/推理环境独立（torch + diffusion 权重），
主应用不装这些重依赖——本模块只做：
    1. 仓库/权重/CUDA 存在性检查（缺什么、去哪拿，给可执行的提示）；
    2. 调 Stable-Makeup 生成 guidance 图（素颜驱动图 × 妆容参考图 → 带妆驱动图）；
    3. 不可用时给出参数化兜底（直接用 RegionMasks 烘焙，行为与现状一致）。

启用步骤：git clone https://github.com/Xiaojiu-z/Stable-Makeup 到
research/Stable-Makeup，按其 README 下载 SD 基座与 checkpoint 到
research/Stable-Makeup/checkpoints/，并准备激活的 conda 环境（约定名 stable-makeup）。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[3] / "research" / "Stable-Makeup"
CKPT_DIR = REPO_DIR / "checkpoints"
CONDA_EXE = Path(r"D:\miniconda3\Scripts\conda.exe")
CONDA_ENV = "stable-makeup"         # 约定的独立环境名
DEFAULT_ENTRY = REPO_DIR / "scripts" / "inference.py"


@dataclass
class GuidanceStatus:
    repo: bool
    weights: bool
    conda_env: bool
    missing_hint: str = ""

    @property
    def ok(self) -> bool:
        return self.repo and self.weights and self.conda_env


def status() -> GuidanceStatus:
    repo = REPO_DIR.is_dir()
    weights = CKPT_DIR.is_dir() and any(CKPT_DIR.glob("*.pth")) or any(CKPT_DIR.glob("*.safetensors"))
    conda = CONDA_EXE.exists() and _env_exists()
    hint = []
    if not repo:
        hint.append(f"git clone https://github.com/Xiaojiu-z/Stable-Makeup {REPO_DIR}")
    if not weights:
        hint.append(f"按该仓库 README 下载权重到 {CKPT_DIR}/")
    if not conda:
        hint.append(f"conda create -n {CONDA_ENV} python=3.10 并安装其 requirements.txt")
    return GuidanceStatus(repo, weights, conda, "；".join(hint))


def _env_exists() -> bool:
    if not CONDA_EXE.exists():
        return False
    r = subprocess.run([str(CONDA_EXE), "env", "list"], capture_output=True, text=True,
                       timeout=60)
    return f"{CONDA_ENV}" in (r.stdout or "")


def generate(drive_img: str | Path, ref_img: str | Path, out_img: str | Path) -> Path:
    """素颜驱动图 + 妆容参考图 → 带妆 guidance 图。环境不可用时抛可执行提示。"""
    st = status()
    if not st.ok:
        raise RuntimeError(f"Stable-Makeup 环境不完整：{st.missing_hint}")
    out_img = Path(out_img)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(CONDA_EXE), "run", "-n", CONDA_ENV, "--no-capture-output", "python",
           str(DEFAULT_ENTRY), "--drive", str(drive_img), "--ref", str(ref_img),
           "--out", str(out_img)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(REPO_DIR))
    if r.returncode != 0 or not out_img.exists():
        raise RuntimeError(f"Stable-Makeup 推理失败（exit={r.returncode}）：{r.stderr[-800:]}")
    return out_img
