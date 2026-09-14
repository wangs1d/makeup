"""flame_avatar — FlashAvatar（FLAME + 3DGS）门控适配器。

FlashAvatar 用参数化脸模型兜底几何，单目视频几分钟重建 3DGS 头像，
从根上绕开我们 COLMAP+Brush 管线"稀疏点太少/几何跑飞"的痛点。
但它的训练环境是独立的（py3.7 + torch1.12 + CUDA 子模块 + FLAME 资产），
主应用不装这些重依赖——本模块只做：
    1. 资产/环境存在性检查（缺什么、去哪拿，给可执行的提示）；
    2. 读取训练产物 ckpt → 导出主应用可消费的规范点云（.ply canonical 空间）。

启用步骤见 research/INTEGRATION.md。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[3] / "research" / "FlashAvatar-code"
FLAME_PKL = REPO_DIR / "flame" / "generic_model.pkl"
FLAME_ASSETS = REPO_DIR / "flame" / "assets"
CONDA_EXE = Path(r"D:\miniconda3\Scripts\conda.exe")
CONDA_ENV = "flashavatar"           # 约定的独立环境名


@dataclass
class FlashAvatarStatus:
    repo: bool
    flame_assets: bool
    conda_env: bool
    missing_hint: str

    @property
    def ok(self) -> bool:
        return self.repo and self.flame_assets and self.conda_env


def status() -> FlashAvatarStatus:
    repo = REPO_DIR.is_dir()
    assets = FLAME_PKL.is_file() or (FLAME_ASSETS.is_dir()
                                     and (any(FLAME_ASSETS.glob("*.pkl"))
                                          or any(FLAME_ASSETS.glob("*.pth"))))
    env_ok = False
    try:
        conda = str(CONDA_EXE) if CONDA_EXE.is_file() else "conda"
        r = subprocess.run([conda, "env", "list"], capture_output=True, text=True,
                           timeout=30)
        env_ok = CONDA_ENV in r.stdout
    except (OSError, subprocess.TimeoutExpired):
        pass
    hints = []
    if not repo:
        hints.append(f"git clone https://github.com/USTC3DV/FlashAvatar-code.git 到 {REPO_DIR}")
    if repo and not assets:
        hints.append("到 https://flame.is.tue.mpg.de 注册下载 FLAME 2020 包，"
                     f"把 generic_model.pkl 放到 {FLAME_PKL}")
    if not env_ok:
        hints.append(f"按 research/FlashAvatar-code/environment.yml 创建 conda 环境 "
                     f"（conda create -n {CONDA_ENV} -f environment.yml），"
                     "并编译 submodules/diff-gaussian-rasterization 与 simple-knn")
    return FlashAvatarStatus(repo, assets, env_ok,
                             "；".join(hints) or "")


def train(video: Path, id_name: str, on_line=None) -> Path:
    """在独立 conda 环境里跑 FlashAvatar 训练，返回 ckpt 路径。"""
    st = status()
    if not st.ok:
        raise RuntimeError(f"FlashAvatar 资产缺失：{st.missing_hint}")
    cmd = ["conda", "run", "-n", CONDA_ENV, "python", "train.py", "--idname", id_name]
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    ckpt = REPO_DIR / "dataset" / id_name / "log" / "ckpt" / "chkpnt.pth"
    for line in proc.stdout:                        # type: ignore[union-attr]
        if on_line:
            on_line(line.rstrip())
    proc.wait()
    if proc.returncode != 0 or not ckpt.exists():
        raise RuntimeError(f"FlashAvatar 训练失败 rc={proc.returncode}")
    return ckpt
