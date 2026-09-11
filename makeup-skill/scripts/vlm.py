#!/usr/bin/env python3
"""VLM 访问层 — OpenAI 兼容接口的 provider 抽象。

配置（环境变量）：
    MAKEUP_VLM_BASE_URL  默认 https://open.bigmodel.cn/api/paas/v4（智谱 GLM）
    MAKEUP_VLM_API_KEY   必填（能力二/三需要）
    MAKEUP_VLM_MODEL     默认 glm-4v-flash

任何 OpenAI 兼容服务商都可替换（OpenAI: https://api.openai.com/v1 + gpt-4o；
Moonshot、Qwen-VL 等）。parse_look / live_coach 只通过本模块访问模型。
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path

VLM_BASE_URL = os.environ.get("MAKEUP_VLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
VLM_API_KEY = os.environ.get("MAKEUP_VLM_API_KEY", "")
VLM_MODEL = os.environ.get("MAKEUP_VLM_MODEL", "glm-4v-flash")

_client = None


def is_configured() -> bool:
    return bool(VLM_API_KEY)


def config_report() -> dict:
    return {"base_url": VLM_BASE_URL, "model": VLM_MODEL, "api_key_set": bool(VLM_API_KEY)}


def _get_client():
    global _client
    if _client is None:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("缺少依赖：pip install openai") from None
        if not VLM_API_KEY:
            raise RuntimeError("VLM 未配置：请设置环境变量 MAKEUP_VLM_API_KEY"
                               "（BASE_URL/MODEL 可选，当前 model=%s）" % VLM_MODEL)
        _client = OpenAI(api_key=VLM_API_KEY, base_url=VLM_BASE_URL)
    return _client


def encode_image_b64(image_path: str | Path, max_side: int = 1280, quality: int = 85) -> str:
    """读图片并压缩到合适尺寸，返回 base64（不含 data: 前缀）。"""
    try:
        import cv2
        img = cv2.imdecode(__import__("numpy").fromfile(str(image_path)), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法读取图片：{image_path}")
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError(f"JPEG 编码失败：{image_path}")
        return base64.b64encode(buf.tobytes()).decode()
    except ImportError:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = _pil_to_bytes(img, quality)
        return base64.b64encode(buf).decode()


def _pil_to_bytes(img, quality: int) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def chat(messages: list, temperature: float = 0.2, max_tokens: int = 4096,
         retries: int = 2) -> str:
    """普通对话。messages 为 OpenAI 格式；图片用 data URL 嵌入。"""
    client = _get_client()
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=VLM_MODEL, messages=messages,
                temperature=temperature, max_tokens=max_tokens)
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 — 网络类错误统一重试
            last_err = e
            if attempt < retries:
                import time
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"VLM 调用失败（{VLM_MODEL} @ {VLM_BASE_URL}）：{last_err}")


def chat_with_image(prompt: str, image_b64: str, temperature: float = 0.2,
                    max_tokens: int = 4096) -> str:
    """单图 + 文本 prompt。"""
    return chat([{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ],
    }], temperature=temperature, max_tokens=max_tokens)


def extract_json(text: str) -> dict | list:
    """从模型输出中提取 JSON（容忍 markdown 代码块与前后杂文本）。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"模型输出中未找到合法 JSON：{text[:200]}…")


if __name__ == "__main__":
    print(json.dumps(config_report(), ensure_ascii=False, indent=2))
    if not is_configured():
        print("状态：未配置（设置 MAKEUP_VLM_API_KEY 后启用能力二/三）", file=sys.stderr)
    else:
        print("状态：已配置 ✓")
