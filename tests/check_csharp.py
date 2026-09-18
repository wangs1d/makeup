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
    """单遍扫描：字符串/字符字面量内的 // 与 /* 不是注释；注释内的引号也不是字符串。
    （此前"先删注释后删字符串"会把 ws:// 这类字符串内容截断，留下悬空引号吞掉整个文件。）"""
    out = []
    i, n = 0, len(src)
    state = "code"
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == '"':
                state = "str"
                out.append('"')
            elif c == "'":
                state = "char"
                out.append("'")
            elif c == "/" and nxt == "/":
                state = "line"
                i += 1
            elif c == "/" and nxt == "*":
                state = "block"
                i += 1
            else:
                out.append(c)
        elif state == "str":
            if c == "\\":
                i += 1                     # 跳过转义字符
            elif c == '"':
                state = "code"
                out.append('"')
            elif c == "\n":                # 未闭合字符串（异常源码）兜底
                state = "code"
                out.append("\n")
        elif state == "char":
            if c == "\\":
                i += 1
            elif c == "'":
                state = "code"
                out.append("'")
        elif state == "line":
            if c == "\n":
                state = "code"
                out.append("\n")
        else:  # block
            if c == "*" and nxt == "/":
                state = "code"
                i += 1
            elif c == "\n":
                out.append("\n")           # 保留换行，避免相邻 token 粘连
        i += 1
    return "".join(out)


def test_files_present():
    names = {p.name for p in CS}
    expected = {
        "MakeupAppMain.cs", "BridgeClient.cs", "UdpLandmarkReceiver.cs", "FaceMeshDeformer.cs",
        "CanonicalFaceModel.cs", "RegionMaskBaker.cs", "MakeupLayerRenderer.cs",
        "SplatLayerRenderer.cs", "WebcamDisplay.cs", "CoachingDisplay.cs", "OneEuroFilter.cs",
        # P5 画像妆容台
        "GaussianAvatarParser.cs", "GaussianAvatarRenderer.cs",
        "AvatarSplatRenderer.cs", "AvatarStationFlow.cs",
    }
    missing = expected - names
    assert not missing, f"缺少脚本：{missing}"
    assert len(SH) == 4, f"应含 4 个 shader（含 GaussianAvatarSplat）：{[p.name for p in SH]}"


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


# ---------------- P5 画像妆容台 ----------------

def test_avatar_parser_and_renderer():
    parser = text(SCRIPTS / "GaussianAvatarParser.cs")
    # 标准 3DGS PLY（SH DC→sRGB、sigmoid、exp、quat 归一）与 MKMKP1 tint sidecar
    for token in ("binary_little_endian", "f_dc_0", "opacity", "rot_0",
                  "2.2f", "Mathf.Exp", "MKMK", "LoadTint", "LoadPly"):
        assert token in parser, f"parser 缺少 {token}"
    # R+ 升级：SH 高阶解析 + MKLT1 主光 sidecar
    for token in ("f_rest_", "ShRest", "MKLT", "LoadLight"):
        assert token in parser, f"parser 缺少 {token}"
    renderer = text(SCRIPTS / "GaussianAvatarRenderer.cs")
    for token in ("ComputeBuffer", "ApplyTint", "ApplyMaterial", "ApplyLight",
                  "Array.Sort(_depthKeys, _orderCpu)", "DrawProceduralNow",
                  "_RootMatrix", "ClearTint", "resortIntervalFrames",
                  "GpuResort", "_shRest", "_Relight"):
        assert token in renderer, f"renderer 缺少 {token}"
    shader = text(SHADERS / "GaussianAvatarSplat.shader")
    for token in ("StructuredBuffer<float4> _Tints", "StructuredBuffer<uint>   _Order",
                  "StructuredBuffer<float3> _ShRest",
                  "_MakeupIntensity", "_FocalPx", "Blend One OneMinusSrcAlpha",
                  "premultiplied", "exp(-0.5",
                  # 片元级 PBR：全精度插值器（COLOR0 会被钳制）、SH 求值、2.5σ、relight 门控
                  "ShEvalDeg2", "QuadSigma = 2.5", "_Relight",
                  "col : TEXCOORD1"):
        assert token in shader, f"avatar shader 缺少 {token}"
    assert "COLOR0" not in shader, "高光 HDR 输出不得走 COLOR0（会被钳制）"
    # GPU bitonic 排序 compute：双核 + ping-pong
    comp = text(SCRIPTS / "GaussianAvatarSort.compute")
    for token in ("#pragma kernel DepthKeys", "#pragma kernel Bitonic",
                  "_KeysIn", "_KeysOut", "_IdxIn", "_IdxOut", "_CamFwd"):
        assert token in comp, f"sort compute 缺少 {token}"


def test_avatar_station_flow_state_machine():
    flow = text(SCRIPTS / "AvatarStationFlow.cs")
    # 状态机：idle → registered → preview → confirmed → station
    for token in ("Registered", "Preview", "Confirmed", "Station", "StationState"):
        assert token in flow, f"flow 缺少状态 {token}"
    # Bridge v1.2 五类消息 + station_state 广播
    for token in ("avatar_register", "avatar_preview", "avatar_confirm",
                  "enter_station", "leave_station", "station_state"):
        assert token in flow, f"flow 缺少消息 {token}"
    # 注册时拉取 PBR 材质与主光 sidecar（否则 Unity 端材质永远不生效）
    for token in ("material.bin", "light.bin", "ApplyMaterial", "ApplyLight"):
        assert token in flow, f"flow 缺少 sidecar 接线 {token}"
    # 妆容台布局：画像侧栏参照
    assert "stationPosition" in flow and "previewPosition" in flow
    splats = text(SCRIPTS / "AvatarSplatRenderer.cs")
    assert "RenderMeshInstanced" in splats and "Array.Sort(_depthKeys, _order)" in splats
    assert "add_splats.json" in splats or "splats" in splats


def test_app_main_avatar_mode_default_no_face_makeup():
    src = text(SCRIPTS / "MakeupAppMain.cs")
    # 默认关闭真脸附妆（妆容只渲染在画像上）
    assert "public bool legacyFaceMakeup = false;" in src
    assert "legacyFaceMakeup" in src
    # apply_spec 在画像模式下显式拒绝并指引 avatar_session
    assert "avatar_mode_active" in src and "avatar_session" in src
    # 画像/妆容台接线 + v1.2 消息分发
    assert "avatarRenderer" in src and "avatarSplats" in src and "station.Handles" in src
    # 既有能力保留（摄像头/指导/上报）
    assert "PublishAmbientGlobals" in src
    assert "assets_url" in src and "UnityWebRequest" in src
    assert "CaptureJpegAsync" in src and "intensity_changed" in src
