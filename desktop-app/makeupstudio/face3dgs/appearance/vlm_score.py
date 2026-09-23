"""vlm_score — 可选的 VLM 主观妆感评分（评测维度补全）。

PSNR/LPIPS/ΔE 都回答不了"这个妆看起来自不自然"——主观质感只能问多模态
模型。完全可选：无 `MAKEUP_VLM_API_KEY` 时返回 None，评测照常出报告。
环境变量与 makeup-skill 同约定（BASE_URL/ MODEL 可覆盖，默认 GLM-4V 兼容
的 OpenAI chat completions 形态）。评分只用于回归趋势对比，不进产品验收
硬门（VLM 打分不稳定）。
"""
from __future__ import annotations

import base64
import json
import os


def _config() -> tuple[str, str, str] | None:
    key = os.environ.get("MAKEUP_VLM_API_KEY", "").strip()
    if not key:
        return None
    base = os.environ.get("MAKEUP_VLM_BASE_URL",
                          "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
    model = os.environ.get("MAKEUP_VLM_MODEL", "glm-4v-flash")
    return key, base, model


PROMPT = (
    "你是化妆师与图像质量评审。对比这两组图：第一组是同一人素颜的3DGS渲染，"
    "第二组是渲染了妆容（{spec}）的效果。请只从「妆感自然度」角度评分："
    "妆是否像真实画上去的（颜色过渡、边缘、材质、与皮肤光影的融合），"
    "忽略底模本身的重建瑕疵（模糊/碎片），那不是妆容的锅。"
    '只输出 JSON：{{"score": 0-100 整数, "reason": "一句话"}}'
)


def score_makeup(bare_rgb: list[np.ndarray], made_rgb: list[np.ndarray],
                 spec_name: str = "") -> dict | None:
    """素颜|妆后渲染图对（可多视角）→ {"score": int, "reason": str} | None。

    任一环境/网络/解析失败都返回 None（评测链路永不因此中断）。"""
    try:
        import cv2
    except ImportError:
        return None
    cfg = _config()
    if cfg is None or not made_rgb:
        return None
    key, base, model = cfg

    def bgr_jpeg(img) -> str:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise ValueError("jpeg encode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    content: list[dict] = [{"type": "text", "text": PROMPT.format(spec=spec_name)}]
    for tag, imgs in (("素颜", bare_rgb), ("妆后", made_rgb)):
        for i, img in enumerate(imgs):
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{bgr_jpeg(img)}"}})
            content.append({"type": "text", "text": f"↑这是{tag}渲染图（视角{i + 1}）"})
    import urllib.request

    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps({"model": model, "messages": [
            {"role": "user", "content": content}], "temperature": 0.2}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            text = json.loads(resp.read().decode("utf-8"))["choices"][0] \
                ["message"]["content"]
        lo = text.find("{")
        hi = text.rfind("}") + 1
        parsed = json.loads(text[lo:hi])
        score = int(parsed["score"])
        return {"score": max(0, min(100, score)),
                "reason": str(parsed.get("reason", ""))[:300]}
    except Exception:
        return None
