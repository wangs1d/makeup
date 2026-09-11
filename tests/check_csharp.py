"""Unity C#/shader 静态检查（无法在此环境编译 Unity，做结构化静态断言）。

覆盖 P0 回归（右眉、协程失效、假 fps）与本次优化的关键接线（姿态解耦、环境光全局量、
实例化溅射、TTS、HTTP 资产、滑杆上报）。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "unity-app" / "Assets" / "Scripts"
SHADERS = ROOT / "unity-app" / "Assets" / "Shaders"
TOOLS = ROOT / "unity-app" / "tools"

CS = sorted(SCRIPTS.glob("*.cs"))
SH = sorted(SHADERS.glob("*.shader"))


def text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def strip_comments_and_strings(src: str) -> str:
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r'"(\\.|[^"\\])*"', '""', src)
    return src


def test_files_present():
    names = {p.name for p in CS}
    expected = {
        "MakeupAppMain.cs", "BridgeClient.cs", "UdpLandmarkReceiver.cs", "FaceMeshDeformer.cs",
        "CanonicalFaceModel.cs", "RegionMaskBaker.cs", "MakeupLayerRenderer.cs",
        "SplatLayerRenderer.cs", "WebcamDisplay.cs", "CoachingDisplay.cs", "OneEuroFilter.cs",
    }
    missing = expected - names
    assert not missing, f"缺少脚本：{missing}"
    assert len(SH) == 3


def test_braces_and_parens_balanced():
    for p in CS + SH:
        src = strip_comments_and_strings(text(p))
        for open_ch, close_ch in [("{", "}"), ("(", ")"), ("[", "]")]:
            assert src.count(open_ch) == src.count(close_ch), \
                f"{p.name}: {open_ch}{close_ch} 不平衡 {src.count(open_ch)} vs {src.count(close_ch)}"


def test_no_known_stale_apis():
    for p in CS:
        src = text(p)
        # P0：StopCoroutine 传新实例无法停止旧协程
        assert "StopCoroutine(RunLoop())" not in src, p.name
        assert "Application.titleValue" not in src, p.name
        assert "fps = 30" not in src and "landmarks = 468" not in src, f"{p.name}: 硬编码 fps/点数"
        assert "SendWithAck" not in src, f"{p.name}: 幽灵 API 注释"


def test_region_mask_baker_eyebrow_both_sides():
    src = text(SCRIPTS / "RegionMaskBaker.cs")
    # P0 回归：眉毛必须遍历两侧（Sides 迭代器），不允许恒返回 left 的 Side()
    assert "foreach (string one in Sides(side))" in src
    assert "private string Side(string side)" not in src
    # concealer 消费 shape
    assert "under_eye" in src and "PaintConcealer" in src
    # 羽化表与 Python 同源
    assert '"blush", 44f' in src and "FeatherPx" in src


def test_pose_decoupling_wiring():
    recv = text(SCRIPTS / "UdpLandmarkReceiver.cs")
    assert "MKT2" in recv and "GlToUnity" in recv and "maxFrameAge" in recv
    deform = text(SCRIPTS / "FaceMeshDeformer.cs")
    assert "ApplyPose" in deform and "ApplyLegacy" in deform
    assert "OneEuroCloud" in deform and "PoseSmoother" in deform
    assert "FacePresence" in deform
    # 零 GC：不每帧取网格数组
    assert "mesh.vertices" not in deform and "TransformPoints" not in deform
    assert "_mesh.SetVertices(_verts)" in deform


def test_protocol_matches_python_side():
    py = text(TOOLS / "tracking_protocol.py")
    cs = text(SCRIPTS / "UdpLandmarkReceiver.cs")
    assert b'"MKT2"'.decode() in py and "MKT2" in cs
    assert 'HEADER_FMT = "<4sHIdHHfHH"' in py
    assert cs.count("0.01f") >= 3       # 厘米→米


def test_ambient_and_layer_shader_globals():
    shader = text(SHADERS / "MakeupLayer.shader")
    for name in ("_EnvTint", "_EnvLumTex", "_LightDirWorld", "_EnvStrength",
                 "_MakeupIntensity", "_FacePresence", "_FaceEdgeTex"):
        assert name in shader, f"MakeupLayer 缺少 {name}"
    main = text(SCRIPTS / "MakeupAppMain.cs")
    webcam = text(SCRIPTS / "WebcamDisplay.cs")
    assert "PublishAmbientGlobals" in webcam and main.count("PublishAmbientGlobals") == 1
    assert "GL.invertCulling" in main and "Matrix4x4.Scale(new Vector3(-1f, 1f, 1f))" in main


def test_splat_instanced_sorted_backend():
    src = text(SCRIPTS / "SplatLayerRenderer.cs")
    assert "Graphics.RenderMeshInstanced" in src
    assert "Array.Sort(_depthKeys, _order)" in src          # 视深排序
    assert "UNITY_DEFINE_INSTANCED_PROP" in text(SHADERS / "GaussianSplat.shader")
    assert "GetComponent<MeshRenderer>()" not in src        # 每帧 GetComponent 移除
    assert "toward" in src and "inset" in src               # 唇锚点内收


def test_coaching_queue_and_tts():
    src = text(SCRIPTS / "CoachingDisplay.cs")
    assert "SetProgress" in src and "fillAmount" in src
    assert "System.Speech" in src and "SpeakViaPowerShell" in src
    assert "dedupeSeconds" in src


def test_app_main_features():
    src = text(SCRIPTS / "MakeupAppMain.cs")
    assert "assets_url" in src and "UnityWebRequest" in src
    assert "intensity_changed" in src
    assert "CaptureJpegAsync" in src
    assert '["pose"] = frame.hasPose' in src
    assert 'MeasuredFps' in src


def test_face_tracker_modes():
    src = text(TOOLS / "face_tracker.py")
    for token in ("--relay", "--synthetic", "--legacy-json", "LandmarkModel", "SyntheticModel"):
        assert token in src


def test_webcam_background_keeps_simple():
    src = text(SHADERS / "WebcamBackground.shader")
    assert "_Mirror" in src
