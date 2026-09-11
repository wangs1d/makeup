# MakeupMirror 装配说明（Unity 2022.3 LTS / Windows）

App 由 9 个脚本 + 3 个 shader 组成（`Assets/Scripts/`、`Assets/Shaders/`），
主入口 `MakeupAppMain` 会**程序化创建**背景 quad、人脸网格物体和 UI，
所以场景装配只有少量拖拽与包依赖。

## 1. 依赖包

Package Manager（Window → Package Manager → + → Add by git URL / by name）：

| 包 | 用途 |
|---|---|
| `com.unity.nuget.newtonsoft-json` | JSON 解析（BridgeClient/RegionMaskBaker/MakeupLayerRenderer） |
| （可选）`https://github.com/aras-p/UnityGaussianSplatting.git` | 真·3DGS 渲染后端（见第 6 节） |

不需要 MediaPipe Unity 插件——人脸关键点由 Python sidecar 提供（第 5 节）。

## 2. 场景搭建（10 分钟）

1. 新建场景 `Mirror.unity`，只建一个空物体 `MakeupApp`，挂 `MakeupAppMain`。
2. 其余全部留空 —— `Awake()` 自动创建：
   - `Main Camera`（若场景没有）
   - `WebcamBackground` quad + `WebcamMirror/WebcamBackground` 材质 + `WebcamDisplay`
   - `FaceMesh`（MeshFilter/MeshRenderer + `FaceMeshDeformer`）
   - `MakeupLayerRenderer`、`SplatLayerRenderer`、`CoachingDisplay`、UI Canvas、`BridgeClient`
3. Player Settings：允许 `Windows Camera` 能力（旧版本勾选 Webcam）；分辨率 1280×720 起步。
4. Build：Windows x86_64。产物运行前确认第 5 节 sidecar 在跑。

需要手动微调时（可选）：
- `FaceMeshDeformer.faceDistance/scaleX/scaleY`：脸在画面里的距离与大小（默认 0.55m / 0.42）。
- `WebcamDisplay.mirror`：镜像显示；sidecar 用了 `--flip` 时把它关掉，二者只开其一。

## 3. StreamingAssets（已随仓库附带）

| 文件 | 来源 | 用途 |
|---|---|---|
| `canonical_face_model.obj` | MediaPipe 官方 | 468 顶点拓扑 + UV（蒙版光栅化基准） |
| `landmark-regions.json` | makeup-skill/references/ | 妆容区域 → 关键点索引映射 |

更新 skill 的 landmark-regions.json 后，重新复制覆盖并重编译。

## 4. 运行时数据流

```
face_tracker.py ──UDP 468 点──▶ UdpLandmarkReceiver ──▶ FaceMeshDeformer（网格变形）
makeup bridge_server ──WebSocket──▶ BridgeClient ──▶ MakeupAppMain
     ├─ apply_spec(+assets) → RegionMaskBaker（蒙版）→ MakeupLayerRenderer（mesh 层）
     │                                        └─▶ SplatLayerRenderer（高斯溅射层）
     ├─ coaching → CoachingDisplay（文字+TTS 钩子）
     ├─ request_frame → WebcamDisplay.CaptureJpeg → frame 回传
     └─ set_intensity / clear_makeup → 渲染器
```

## 5. 人脸追踪 sidecar（开发机）

```bash
pip install mediapipe opencv-python numpy
python tools/face_tracker.py          # 先启动，再启动 App
```

## 6. 3DGS 升级路径（当前 billboard 后端 → 真 splat 管线）

`SplatLayerRenderer` 目前是"锚点 + 各向异性高斯面片"后端（billboard 型），
不依赖插件即可给出唇部/高光的体积感。要换成 UnityGaussianSplatting：

1. 安装包后，把 `SplatGaussianAsset`（srcData = 运行时 anchors）接到 `ISplatBackend`；
   anchors 数据已由 bake_assets.py 的 `splat_layers.json` 给出（position=关键点、
   scale=sigma、opacity=alpha、color=色带取色），只需写一个运行时 converter。
2. 渲染顺序：splat 层在 mesh 层之后（renderQueue Transparent+100 已留好）。

## 7. TTS 钩子

`CoachingDisplay.Show(..., speak=true)` 目前打日志。接入 Windows 语音最短路径：
`UnityEngine.Windows.Speech` 不支持合成，用 `System.Speech.Synthesis.SpeechSynthesizer`
（仅 Editor/Standalone，需 mscorlib 兼容设置）在 `Speak` 钩子里 `SpeakAsync(message)`。

## 8. 故障排查

| 现象 | 原因/处理 |
|---|---|
| 标题栏 `tracking: waiting` 一直不变 | sidecar 没启动 / 端口不对（8766）/ 摄像头被占用 |
| 标题栏 `bridge: reconnecting` | bridge_server 没跑（`python bridge_server.py`）|
| 下发妆容无反应 | 看控制台日志：找 `[makeup]`/`[masks]` 错误；多为 landmark-regions.json 缺失 |
| 妆容位置整体偏移 | 用户离摄像头过近/过远，或 mirror 设置与 sidecar `--flip` 重复 |
| 溅射层不动 | `deformer.HasFace` 为 false 时自动隐藏；对准脸即可 |
