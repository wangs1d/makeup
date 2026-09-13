#!/usr/bin/env python3
"""coach — 本地步骤陪练（MVP）。

从当前妆容 layers 推导化妆步骤序列，定时推进：步骤文字 + 进度 + Windows SAPI TTS
语音提醒。VLM 视觉对比（makeup-skill live_coach）需要 API Key，作为后续增强接入点。
"""
from __future__ import annotations

import subprocess
import sys

REGION_STEPS = [
    ("foundation", "底妆：均匀涂抹粉底，由内向外推开"),
    ("concealer", "遮瑕：点涂眼下与瑕疵处，轻轻拍开"),
    ("contour", "修容：沿发际线、下颌缘与鼻侧轻扫"),
    ("eyebrow", "眉毛：顺毛流描画，尾端微微加深"),
    ("eyeshadow", "眼影：浅色打底，深色晕染眼尾"),
    ("eyeliner", "眼线：贴睫毛根部描画，眼尾平拉"),
    ("lashes", "睫毛：夹翘后Z字形涂刷睫毛膏"),
    ("blush", "腮红：颧骨斜向轻扫，少量多次"),
    ("highlight", "高光：颧骨、鼻梁与唇峰提亮"),
    ("lipstick", "唇妆：先描唇线，再填充唇体"),
]
REGION_NAMES = {"foundation": "底妆", "concealer": "遮瑕", "contour": "修容", "eyebrow": "眉毛",
                "eyeshadow": "眼影", "eyeliner": "眼线", "lashes": "睫毛", "blush": "腮红",
                "highlight": "高光", "lipstick": "唇妆"}


def steps_from_layers(layers: list[dict]) -> list[str]:
    enabled = {l["region"] for l in layers if l.get("enabled", True)}
    return [text for region, text in REGION_STEPS if region in enabled]


def _speak_win(text: str):
    """Windows SAPI TTS（免依赖，走系统 PowerShell）。"""
    ps = (f"Add-Type -AssemblyName System.Speech;"
          f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
          f"$s.Speak('{text}')")
    try:
        subprocess.Popen(["powershell", "-NoProfile", "-Command", ps],
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return True
    except OSError:
        return False


def speak(text: str) -> bool:
    if sys.platform == "win32":
        return _speak_win(text)
    return False
