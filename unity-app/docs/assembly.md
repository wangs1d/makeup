# MakeupMirror 装配说明（Unity 2022.3 LTS / Windows）

App 由 11 个脚本 + 3 个 shader 组成（`Assets/Scripts/`、`Assets/Shaders/`），
主入口 `MakeupAppMain` 会**程序化创建**背景 quad、人脸网格物体和 UI（含强度滑杆、
步骤进度条、状态栏），场景装配只有少量拖拽与包依赖。

## 1. 依赖包

Package Manager（Window → Package Manager → + → Add by git URL / by name）：

| 包 | 用途 |
|---|---|
| `com.unity.nuget.newtonsoft-json` | JSON 解析（BridgeClient/RegionMaskBaker/MakeupLayerRenderer） |
| （可选）`https://github.com/aras-p/UnityGaussianSplatting.git` | 真·3DGS 库后端（见第 6 节；内置的排序实例化后端已可用，此包为可选增强） |

不需要 MediaPipe Unity 插件——人脸关键点与 6DoF 姿态由 Python sidecar 提供（第 5 节）。

## 2. 场景搭建（10 分钟）

1. 新建场景 `Mirror.unity`，只建一个空物体 `MakeupApp`，挂 `MakeupAppMain`。
2. 其余全部留空 —— `Awake()` 自动创建：
   - `Main Camera`（固定在原点、朝 +Z、旋转恒等——世界空间即相机空间，勿移动）
   - `WebcamBackground` quad + `MakeupMirror/WebcamBackground` 材质 + `WebcamDisplay`
   - `AvatarRoot`（`GaussianAvatarRenderer` + `AvatarSplatRenderer`，3DGS 画像渲染）
   - `AvatarStationFlow`（妆容台状态机：registered→preview→confirmed→station 布局切换）
   - `CoachingDisplay`、UI Canvas（滑杆/进度条）、`BridgeClient`
   - （legacy，默认不创建）`FaceMesh` + `MakeupLayerRenderer` + `SplatLayerRenderer`——
     仅当 `MakeupAppMain.legacyFaceMakeup = true` 时装配（P1 真脸附妆链路）
3. Player Settings：允许 `Windows Camera` 能力；分辨率 1280×720 起步。
4. Build：Windows x86_64。画像流程不需要 sidecar；legacy 真脸链路运行前确认第 5 节 sidecar 在跑。

可调参数（可选）：
- `MakeupAppMain.legacyFaceMakeup`：真脸附妆开关（**默认 false**——妆容只渲染在画像上）。
- `MakeupAppMain.mirror`：镜像显示。通过**投影矩阵 x 翻转 + GL.invertCulling** 一处实现，
  视频与 3D 同步镜像（`WebcamDisplay.mirror` 保持 false，不要两边都开）。
- `MakeupAppMain.backgroundDistance`：背景 quad 距离（需大于画像距离，默认 2.5m）。
- `AvatarStationFlow.previewPosition / stationPosition / faceHeightMeters`：
  画像预览（居中正对）与妆容台（右侧栏参照）布局、画像显示脸高（默认 0.22m）。
- `GaussianAvatarRenderer.makeupIntensity / resortIntervalFrames`：妆容浓度（滑杆联动）/
  视深排序节流（相机静止时每 6 帧重排一次）。
- `AvatarSplatRenderer.sizeScale / viewBlend / intensity`：附加溅射尺寸/侧向倾角/浓度。
- （legacy）`FaceMeshDeformer.responsiveness`、`MakeupLayerRenderer.envStrength`、
  `SplatLayerRenderer.sizeScale / viewBlend`。

## 3. StreamingAssets（已随仓库附带）

| 文件 | 来源 | 用途 |
|---|---|---|
| `canonical_face_model.obj` | MediaPipe 官方 | 468 顶点拓扑 + UV（蒙版光栅化基准；载入时转为米、(x,y,-z) Unity 约定） |
| `landmark-regions.json` | makeup-skill/references/ | 妆容区域 → 关键点索引映射 |

更新 skill 的 landmark-regions.json 后，重新复制覆盖并重编译。

## 4. 运行时数据流

```
WebcamDisplay ──UDP 分片(MKF1 :8767)──▶ face_tracker --relay（同一颗摄像头，帧与关键点同源）
face_tracker ──UDP v2 二进制(MKT2 :8766)──▶ UdpLandmarkReceiver ──▶ FaceMeshDeformer
                                                │   根节点=4×4 姿态（位置/朝向/距离）
                                                │   顶点=头部局部点（表情层，1€ 滤波）
makeup bridge_server ──WebSocket──▶ BridgeClient ──▶ MakeupAppMain
     ├─ apply_spec(assets 内嵌或 assets_url HTTP 下载) → RegionMaskBaker(后台线程蒙版)
     │        → MakeupLayerRenderer（环境光合成/脸缘羽化/丢脸淡出）
     │        → SplatLayerRenderer（实例化+视深排序高斯溅射）
     ├─ coaching(含 progress) → CoachingDisplay（队列+TTS+步骤进度条）
     ├─ request_frame → WebcamDisplay.CaptureJpegAsync（异步，不卡帧）
     └─ set_intensity / clear_makeup → 渲染器 + UI 滑杆同步
WebcamDisplay.Ambient ──每 0.4s──▶ 全局量 _EnvTint/_EnvLumTex/_LightDirWorld（妆容光影合成）
```

## 5. 人脸追踪 sidecar（开发机）

```bash
pip install mediapipe opencv-python numpy
python tools/face_tracker.py --relay     # 推荐：接收 App 中继帧（单摄像头闭环）
# 或：python tools/face_tracker.py --camera 0   （sidecar 自己开摄像头，旧模式）
# 无摄像头联调：python tools/face_tracker.py --synthetic --show
```

先启动 sidecar 再启动 App；标题状态栏显示 `tracking: ok · N fps · 468 pts · pose ✓` 即正常。
sidecar 完全本地运行，不上传任何画面。

协议：v2 二进制（默认）——header + 4×4 姿态（厘米，OpenGL 相机系）+ 头部局部关键点 + 图像
坐标，见 `tools/tracking_protocol.py`；App 自动兼容 v1 JSON（`--legacy-json`，退回固定平面
映射、无姿态对齐）。App→sidecar 帧中继为 `MKF1` 分片（≤16KB/片）。

## 6. 溅射与画像渲染后端

**画像主渲染（P5，默认）**：`GaussianAvatarRenderer` 把注册画像的 PLY 解析为
ComputeBuffer（位置/3D 协方差/颜色/tint），CPU 视深排序（相机动过阈值或每 6 帧重排）+
`Graphics.DrawProceduralNow` 六顶点展开，`MakeupMirror/GaussianAvatarSplat` 在顶点着色器做
标准 EWA 投影（Σworld→视空间→焦距雅可比→2D 协方差特征轴），premultiplied 混合。
妆容 = `tint.bin`（per-Gaussian rgba，`avatar_session preview` 下发）×`_MakeupIntensity`
插值，换妆不重建位置 buffer。附加溅射（唇釉/闪片）由 `AvatarSplatRenderer` 在画像局部空间
实例化绘制（复用 `GaussianSplat` shader，queue 3150）。

**legacy 真脸溅射**：`SplatLayerRenderer`（锚点→切平面面片、外唇锚点内收、视深排序实例化）
仅 `legacyFaceMakeup=true` 时装配。要换 aras-p/UnityGaussianSplatting：`splat_layers.json`
锚点数据已是该库的 splat 语义，写一个 converter 填 `SplatGaussianAsset` 即可。

## 7. TTS（指导语音播报）

`CoachingDisplay` 在 Windows 上两级实现：
1. 反射加载 `System.Speech`（进程内、低延迟）——需要把 .NET Framework 的
   `System.Speech.dll` 放入 `Assets/Plugins/`（且 Api Compatibility Level 选 .NET Framework）；
2. 加载失败自动退回 `powershell -c "Add-Type -AssemblyName System.Speech; …Speak()"` 子进程
   （零配置，延迟略高）。

播报策略：`speak=true` 且 priority=warn 或 note=done 才读；同句 60s 去重。其它平台仅日志。

## 8. 故障排查

| 现象 | 原因/处理 |
|---|---|
| 状态栏 `tracking: waiting` 不变 | sidecar 没启动 / 端口不对（8766）/ `--relay` 模式下 App 发帧被防火墙拦（本机回环一般无碍） |
| `tracking: no-face` 频繁 | 距离 30~40cm、光线充足；`--synthetic` 可验证 App 侧链路 |
| 妆容整体偏移/缩放不对 | 状态栏看 `pose ✓` 是否出现：无 pose 说明 sidecar 老版本（v1 JSON → 固定平面映射）；focal 估计与真实摄像头视场差太多时调 `--focal-ratio` |
| 妆容有"面具边缘" | 确认 `MakeupLayerRenderer.envStrength > 0`（脸缘羽化全局纹理已启用）且 StreamingAssets 的 landmark-regions.json 是新版 |
| 溅射层不动/消失 | `deformer.FacePresence` 为 0（丢脸淡出）；对准脸即可 |
| 标题栏 `bridge: reconnecting` | bridge_server 没跑（`python bridge_server.py`）；端口被其他服务占用时换 `--port` 并改 App 的 bridgeUrl |
| 大妆容下发超时 | 走资产侧车（bridge 默认 8868/8768 开启），App caps 需含 `assets_url` |
