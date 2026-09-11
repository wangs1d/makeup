"""live_coach 状态机 / 剧本 / 步骤构建 的单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

import live_coach as lc


class Args:
    test = False
    no_reference = False
    env = "neutral"
    session_dir = None


SPEC = {
    "spec_version": "1.0", "name": "t", "description": "测试妆", "intensity": 0.8,
    "layers": [
        {"id": "foundation-both", "region": "foundation", "side": "both", "enabled": True,
         "opacity": 0.7, "finish": "satin", "color_stops": [{"at": 0, "hex": "#AAAAAA"}]},
        {"id": "lipstick-both", "region": "lipstick", "side": "both", "enabled": True,
         "opacity": 0.8, "finish": "gloss", "color_stops": [{"at": 0, "hex": "#BB5555"}]},
        {"id": "blush-left", "region": "blush", "side": "left", "enabled": False,
         "opacity": 0.8, "finish": "matte", "color_stops": [{"at": 0, "hex": "#CC8877"}]},
    ],
    "steps": [
        {"order": 1, "area": "base", "regions": ["foundation"], "instruction": "拍开"},
        {"order": 2, "area": "lip", "regions": ["lipstick"], "instruction": "晕开"},
    ],
}


def test_build_steps_and_filter():
    steps = lc.build_steps(SPEC, None)
    assert [s["area"] for s in steps] == ["base", "lip"]
    only_lip = lc.build_steps(SPEC, ["lip"])
    assert [s["area"] for s in only_lip] == ["lip"]
    # 无 steps 的 spec：按 layers 生成占位（禁用层跳过）
    spec2 = dict(SPEC, steps=None)
    steps2 = lc.build_steps(spec2, None)
    assert [s["area"][0] for s in steps2] == ["f", "l"]


def test_spec_digest_contains_colors_and_notes():
    spec = dict(SPEC, layers=[dict(SPEC["layers"][1], notes="咬唇")])
    digest = lc.spec_digest(spec)
    assert "lipstick" in digest and "#BB5555" in digest and "咬唇" in digest


def test_vote_and_advance_requires_two_done():
    coach = lc.Coach(SPEC, 2.0, Args())
    coach.vote_and_advance("in_progress")
    coach.vote_and_advance("done")
    assert coach.step_idx == 0                    # 单次 done 不推进
    coach.vote_and_advance("done")
    assert coach.step_idx == 1                    # 窗口内 ≥2 done 推进
    assert coach.status_window == []              # 推进后清窗
    coach.vote_and_advance("in_progress")
    coach.vote_and_advance("done")
    assert coach.step_idx == 1                    # 最后一步不再前进


def test_vote_window_cannot_regress():
    coach = lc.Coach(SPEC, 2.0, Args())
    coach.vote_and_advance("done"); coach.vote_and_advance("done")
    coach.vote_and_advance("not_started")
    assert coach.step_idx == 1                    # 没有"回退"路径


def test_canned_verdicts_walkthrough():
    steps = lc.build_steps(SPEC, None)
    canned = lc.CannedVerdicts(steps)
    statuses = []
    for _ in range(8):
        v = canned.next()
        statuses.append(v["step_status"])
        assert v["face_visible"]
    # 每步 4 轮：ip ip done done → ip ip done done
    assert statuses == ["in_progress", "in_progress", "done", "done"] * 2
    # 第 2/4 个 done 带 next 提示 / 最后带评分
    v4 = None
    canned2 = lc.CannedVerdicts(steps)
    for _ in range(8):
        v4 = canned2.next()
    assert isinstance(v4.get("scores"), list) and v4["scores"]
    scores = v4["scores"]
    assert all(0 <= s["score"] <= 100 for s in scores)


def test_assess_lighting_dark_and_bright():
    dark = b"\xff\xd8" + np.zeros((64, 64), np.uint8).tobytes()
    dark_jpg = _jpg(np.zeros((64, 64), np.uint8))
    assert lc.assess_lighting(dark_jpg) is not None
    bright_jpg = _jpg(np.full((64, 64), 250, np.uint8))
    assert lc.assess_lighting(bright_jpg) is not None
    ok_jpg = _jpg(np.full((64, 64), 128, np.uint8))
    assert lc.assess_lighting(ok_jpg) is None


def _jpg(gray: np.ndarray) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", gray)
    assert ok
    return buf.tobytes()


def test_progress_push_fields():
    """coaching 消息带 step/steps/progress/step_name（App 进度条依赖）。"""
    import asyncio

    async def t():
        coach = lc.Coach(SPEC, 2.0, Args())
        coach.vote_and_advance("done")
        coach.vote_and_advance("done")
        coach.progress = 0.65
        sent = []

        class Conn:
            async def send(self, msg):
                sent.append(msg)
        await coach.push(Conn(), "测试提醒", "eye", "info")
        m = sent[0]
        assert m["step"] == 2 and m["steps"] == 2
        assert m["progress"] == 0.65
        assert m["step_name"] == "唇" and m["speak"] is True

    asyncio.run(t())
