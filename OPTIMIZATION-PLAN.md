# 优化创新方案（Optimization Plan）

> **实施进度（2026-09-11）**：P0–P4 已全部落地并通过测试（`pytest tests/` 59 项全过 + 端到端
> Bridge 会话 + 重录 demo 视频）。长线项（FLAME/移动端）保持路线图。逐项状态见文中 ✅ 标注。

> 基于 2026-09-10 对全仓代码的技术调研（Unity 10 个 C# 脚本、3 个 shader、skill 全部 Python 脚本、
> preview 渲染管线、bridge 协议与文档），对照既定目标（README 路线图 + `.zcode/plans/` 原实施方案）
> 给出的分阶段优化与创新计划。每项含动机、技术路径（落到文件）、验收标准。

## 零、现状基线与核心差距

已达成：三大能力闭环（选妆试妆 / 素材解析 / 实时指导）、Bridge 协议端到端可用、
资产烘焙管线（蒙版+渐变+溅射配置）、Python 免 Unity 预览与演示视频。

对照"逼真实时试妆 + AI 化妆助手"的目标，四个结构性差距：

| # | 差距 | 现状证据 |
|---|---|---|
| G1 | **妆容没有画在用户脸上**：App 是"背景摄像头画面 + 前景悬浮面具"，mesh 用固定平面假设（0.55m/0.42 缩放）摆放，不随用户头部透视对齐 | `FaceMeshDeformer.cs:69-79`；sidecar 与 App 各开一路摄像头，关键点与画面两路不同步（`face_tracker.py:49` + `WebcamDisplay.cs`） |
| G2 | **渲染离"逼真"有距离**：妆容层是无光照 unlit 叠加，不感知环境光、无 SSS/clearcoat；溅射层是 billboard 高斯近似而非真 3DGS（无深度排序/各向异性核） | `MakeupLayer.shader:16-19`；`SplatLayerRenderer.cs:5-7,98-110`；`assembly.md:61-69` 承认 converter 未写 |
| G3 | **指导智能靠单帧+纯文字**：VLM 只拿到文字版 spec digest + 当前帧，从未见过目标妆的渲染效果；step 推进靠单帧判断易来回跳；TTS 只有日志钩子 | `live_coach.py:141-152`；`CoachingDisplay.cs:27` |
| G4 | **工程质量债**：已证实 bug（眉 side=both 只画左眉、StopCoroutine 失效）、追踪无平滑、UDP 走 JSON 文本、每帧全量 GC、request_frame 卡主线程 | `RegionMaskBaker.cs:75,107`；`BridgeClient.cs:31`；`FaceMeshDeformer.cs:80-83`；`UdpLandmarkReceiver.cs:80-98`；`WebcamDisplay.cs:45-65` |

优先级逻辑：**P0 消坑（小成本）→ P1 补产品根本（真 AR 贴脸）→ P2 补目标根本（逼真度）→
P3 做 AI 差异化创新 → P4 工程加固**。P0+P1 是下一迭代主战场。

---

## P0 正确性修复 ✅（1~2 人日 → 已完成：右眉/协程/concealer/实测 fps/文档全部修复，见 tests/check_csharp.py 静态回归）

已证实的 bug 与协议不一致，先清零，避免被上层优化掩盖：

1. **眉 side=both 只画左眉**：`RegionMaskBaker.cs:75` + `:107` 的 `Side()` 恒返回 left。
   改为与 Python 侧 `masks_py._eyeshadow` 相同的两侧循环（Python/Unity 已分叉，以 Python 为准）。
2. **BridgeClient 停不掉旧协程**：`BridgeClient.cs:31` `StopCoroutine(RunLoop())` 传的是新
   IEnumerator 实例。缓存协程句柄再停，否则 OnDisable 后残留旧连接循环（重复 hello、消息错乱）。
3. **concealer/wing 形状参数**：concealer 忽略 shape（`RegionMaskBaker.cs:210-218`）、
   eyeliner wing 方向不区分左右（`:168`）。对齐 schema.md 的 shape 语义。
4. **tracking_state 造假**：fps=30、landmarks=468 是硬编码（`MakeupAppMain.cs:139-145`）。
   上报实测 fps（滑动平均）与实际点数；`intensity_changed` 协议已定义但 App 从不发送——
   在 P1 的滑杆 UI 里一并补上。
5. **文档对齐**：bridge-protocol.md 的 frame.w/h、上报时机描述与实现对齐；
   assembly.md "9 个脚本"改 10；删除 `BridgeClient.cs` 头注释里不存在的 SendWithAck。

**验收**：`--test` 冒烟 + preview 双侧渲染同一 spec 逐像素抽查；双侧眉毛、双眼皮线对称出现。

---

## P1 真·AR 贴脸试妆 ✅（已完成：单摄像头中继 + MKT2 二进制协议 + 4×4 姿态解耦 + 投影翻转镜像 + 淡出；姿态求解误差 ≤2.5°）

目标：妆容直接渲染在摄像头画面里的用户脸上，转头/低头/微笑时稳定贴合，消除"面具感"。

### 1.1 单摄像头闭环（消灭双路错位）

现状 sidecar 自己开 640×480 摄像头，App 另开 1280×720——关键点来自 A 画面、背景显示 B 画面。

- `face_tracker.py` 增加 `--relay` 模式：不再本地开摄像头，监听 `127.0.0.1:8767` 接收
  App 发来的 JPEG 帧（每 2 帧一采样，质量 70，约 3MB/s）做追踪。
- `WebcamDisplay.cs`：取帧循环里把降采样帧异步发出（独立 UDP socket，发送不阻塞主线程）。
- 帧与关键点同源同路 → 姿态对齐才有意义；同时摄像头占用冲突消失。

### 1.2 每帧 6DoF 姿态对齐（核心）

- MediaPipe FaceMesh 开 `output_facial_transformation_matrixes=True`，sidecar 直接获得
  4×4 头部刚体变换矩阵，随关键点包一起发（多 64 字节）。
- `FaceMeshDeformer.cs`：删除固定 0.55m/0.42 假设，改为
  `mesh 根节点变换 = 相机内参逆 × facialTransformation`（近似焦距=帧宽、主点=帧心，
  与 relay 帧分辨率绑定标定一次）；顶点仍由 468 点局部驱动（表情层），
  根节点矩阵负责全局姿态（位置层）——表情与姿态解耦，贴合且稳。
- 相机 FOV 对齐：Unity 相机焦距参数化，与 WebCamTexture 实际分辨率联动，保证投影重合。

### 1.3 贴合体验细节

- 丢脸时妆容整体淡出（现有 fade 通道复用），替代当前"冻结 0.6s"（`FaceMeshDeformer.cs:60-67`）。
- mesh 边缘 alpha 羽化带（UV 边界 5% 渐隐，MakeupLayer shader 加边界衰减），
  眼窝/嘴洞透出真实五官而非暗洞（`MakeupAppMain` 的 cavity 绘制改为半透明加深）。
- 追踪置信度低（检测分数 < 阈值）时 UI 提示"靠近一点"而非静默漂移。

**验收**：用户 30cm 处左右转头 ±30°、点头、微笑，妆容边缘始终落在真实唇线/眼睑上，
无面具悬浮边缘；端到端延迟 < 120ms（帧发出→渲染显示）。

---

## P2 渲染逼真度 ✅（已完成：环境色温/屏幕亮度/wrap diffuse/清漆双层高光/脸缘羽化，Unity 与 Python 同模型；溅射=切平面朝向+视深排序实例化，唇锚点内收消锯齿，锚点扩到 7 个 region）

### 2.1 环境光合成妆容（视觉真实感最大杠杆）

现状妆容 unlit 叠加，与环境光影脱节。方案：

- App 端每 500ms 对当前帧降采样估环境光（平均亮度 + R/B 比值 → 色温 + 主光方向近似），
  作为全局参数传入 `MakeupLayer.shader`。
- shader 改造：妆容 albedo 乘环境光色调；新增屏幕空间亮度调制——把背景帧做 1/16 低通
  亮度图传入，妆容采样同屏幕位置的亮度调制自身明暗，妆容"长进"脸的光影里。
- `face_render.py` 同步实现同一模型（它本来就是管线的 Python 等价物），预览与 App 观感一致。

### 2.2 皮肤材质升级（shader + Python 双侧）

- wrap diffuse 次表面近似（唇/颊），`finish: matte→gloss` 映射双层高光：
  底层宽 sheen + 清漆层窄高光（gloss/dewy 唇釉才有 clearcoat）。
- 细噪声微表面（现有 grain 从 UV 空间改为切线空间三平面采样，转头不"游走"）。

### 2.3 真 3DGS 溅射后端（兑现路线图）

按 `assembly.md:61-69` 既定路径落地：

- 写运行时 converter：`splat_layers.json` → 实例属性（pos=关键点+法线偏移、
  scale=σ、rot=切平面朝向、color/alpha）→ `Graphics.DrawMeshInstanced` +
  每帧视深 CPU 排序（splat 数量 ≤ 数百，CPU 排序足够）写 instance buffer。
- 保留现有 billboard 后端作为 fallback（`ISplatBackend` 二选一，caps 协商）。
- 顺带修 `SplatLayerRenderer.cs:107-109` 每帧 `GetComponent`+SetColor 的开销与
  renderQueue 3000+n 层数上限（改分层 offset 计算，取消 100 层硬上限）。

### 2.4 溅射锚点扩容

`bake_assets.py:34-39` SPLAT_ANCHORS 只覆盖 lipstick/highlight/contour，
eyeshadow 的闪片妆、nose highlight 的水光等 splat 请求被静默丢弃。扩锚点表 + 未支持时
显式 warning（不再静默）。

**验收**：同一 spec 在"暖黄台灯/冷白顶灯"两种环境下 A/B 截图，妆容明暗与色调跟随环境；
date-rose 的唇釉呈体积光斑而非平面色块；演示视频重录对比。

---

## P3 AI 指导升级 ✅（已完成：目标妆参考图随帧发给 VLM、3 帧滑窗投票、进度条推送、System.Speech/PowerShell 两级 TTS、结束评分报告、--test 离线剧本）

### 3.1 目标妆参考图对比（低成本高收益）

`live_coach.py` 启动时用 preview 渲染管线（`face_render.py` 正脸静态帧）把目标 spec
渲染成"目标效果图"，与当前帧**两张图一起**发给 VLM：视觉对比替代纯文字 digest。
指导准确性上限直接抬升（VLM 看得见"目标眼影晕染范围"长什么样）。
`parse_look.py` 解析完也先渲染预览图，让用户确认"同款"再试妆。

### 3.2 时序状态机

- step_status 用最近 3 帧滑动窗口投票（2/3 多数才推进/回退），消除单帧误判来回跳；
- 推进/回退在 App 顶部以步骤进度条可视化（coaching 消息已带 progress 字段，补 UI）。

### 3.3 TTS 真实现（兑现路线图）

`CoachingDisplay.cs` 接入 `System.Speech.Synthesis`（Windows 自带，零依赖）：
warn/urgency 高的提醒与步骤完成播报，info 不播；设置开关与音量；
同一提醒 60s 内不重复播（配合现有 last_actions 去重）。

### 3.4 妆面完成度评分（会话总结创新）

结束轮由 VLM 输出各部位 0-100 分 + 一句话建议，生成 `session_report.md`
（对齐 parse_look 的 analysis.md 风格），App 显示总分卡，agent 口头总结。

**验收**：陪练过程中同一部位不再出现互相矛盾的先后提醒；目标妆参考图进入提示词
（可在 bridge 消息日志中证实）；TTS 播报可闻；结束产出评分报告。

---

## P4 工程加固 ✅（已完成：MKT2 二进制追踪+帧龄检查、1€ 滤波（米尺度参数）、零 GC 网格更新、AsyncGPUReadback 取帧、bridge 1.1 精确路由/重连/资产 HTTP 侧车、setup_check 端口占用识别）

1. **追踪链路**：
   - UDP 二进制格式：`header(seq,t,ok,w,h) + 468×3 float32` ≈ 5.6KB/帧，替代 ~80KB JSON
     文本（`face_tracker.py:84-91`、`UdpLandmarkReceiver.cs:80-98`），JSON 保留兼容开关；
   - One Euro Filter 平滑（C# 端，`FaceMeshDeformer` 消费前），强度参数暴露
     （"稳定/敏捷"两档），抖动可见消除；
   - 帧龄检查：>100ms 的追踪包丢弃用旧姿态（防乱序回跳）。
2. **零 GC 渲染**：`FaceMeshDeformer.cs:80-83` 预分配数组 + `Mesh.SetVertices` 直接写，
   法线改预计算拓扑累加表，消除每帧 GC 尖峰。
3. **request_frame 异步化**：`WebcamDisplay.cs:45-65` 改 `AsyncGPUReadback` + 编码线程化，
   消除指导取帧时的主线程卡顿。
4. **Bridge 寻址与重连**：消息加 `to` 字段精确路由（替代"最近 agent"启发式，
   `bridge_server.py:102-121`）；`bridge_common.py` 加指数退避自动重连；
   大资产改 HTTP 侧车下载（`http://127.0.0.1:8766/assets/<hash>`）替代 24MB base64 内嵌，
   解除 `max_size 32MB` 悬顶之石（`bridge_server.py:184`）。

---

## P5 真实用户脸部的 3DGS 化与妆容贴合 ✅（2026-09-12：face3dgs 包落地，管线与 ooosplat 对齐，81 项测试全过）

取代 P2.3"合成网格溅射"的用户价值版本：不再对 canonical 假脸做还原度打磨，
而是把用户本人的脸重建为 3DGS，再把妆容贴合到用户自己的点云上。技术路线与
[ooolabdev/ooosplat](https://github.com/ooolabdev/ooosplat)（Apache-2.0）一致：
FFmpeg 抽帧 → COLMAP 相机重建 → Brush 训练 → final.ply。

### 5.1 采集引导（`face3dgs/capture.py`）
单摄像头环绕采集：左/正/右三个偏航区间各 ≥2s 有效覆盖，MediaPipe 姿态 +
光照/人脸占比质检实时反馈，产出 MP4 + 质量报告。纯逻辑类（帧进→状态出），
Qt Tab 与 CLI 共用；将来搬浏览器端（getUserMedia）判定规则不变。

### 5.2 重建编排（`face3dgs/reconstruct.py`）
`ReconstructionBackend` 抽象：当前 `LocalEngineBackend` 编排本机引擎（发现规则与
ooosplat 环境变量约定兼容）；**服务器部署时实现 `RemoteBackend` 即可整体上云**，
采集/隔离/贴合零改动。命令模板可用 `OOOSPLAT_COLMAP_MAPPER_ARGS`/`OOOSPLAT_BRUSH_TRAIN_ARGS`
覆盖，兼容引擎版本漂移。

### 5.3 脸部隔离（`face3dgs/isolate.py` + `colmap_io.py`）
全场景点云 → 脸部点云：各帧人脸框经 COLMAP 相机重投影投票（≥30% 含脸视角且 ≥2 票），
叠加尺度离群"漂浮高斯"剔除。COLMAP sparse bin 最小读写自实现（零新依赖）。

### 5.4 妆容贴合（`face3dgs/fit_makeup.py`）
无 GPU 拟合的纯几何路线：多帧 2D 地标 + 相机位姿 DLT 三角化 → 468 个用户 3D 地标；
canonical 脸模型（顶点与地标一一对应）带尺度 Procrustes 配准进 splat 世界系；
splat 最近邻查 canonical UV → 复用 `RegionMasks.bake` 区域蒙版（形状参数与 App 完全一致）
→ 非破坏式颜色混合（wrap-diffuse 光照，法线取自 splat 旋转）。产出 madeup.ply/.splat +
前后对比预览。

### 5.5 UI 与 CLI
主窗口新增"③ 我的 3D 脸"标签页（采集→重建→贴合三步向导，引擎缺失给安装指引）；
CLI：`py -m makeupstudio.face3dgs status|capture|rebuild|isolate|fit`。

**验收**：`pytest tests/` 82 项全过（含 face3dgs 12 项：Mock 引擎跑通完整编排、合成 COLMAP
数据验证投影/隔离/贴合几何）。**真机端到端实测**（2026-09-12，合成环绕视频 orbit2.mp4，
640×536/120 帧/±55° 点阵背景）：
- 引擎下载（gh-proxy 镜像分块）→ 引擎发现 → cv2 抽帧 → COLMAP → Brush 导出链路全部打通；
- 脸部隔离：MediaPipe 检出 40/40 视角，13000 → 7618 个脸部 splat（背景点阵被正确投票剔除）；
- 妆容贴合：40 帧地标三角化 + 鲁棒过滤（侧脸幻觉观测剔除，配准 RMSE 0.024 脸高），
  date-rose 全妆（底妆/眉/眼影/眼线/腮红/唇釉）正确落到对应五官区域，前后对比预览生成
  （`out/face3dgs/fitted/compare_*.png`）；
- 本机遗留问题（2026-09-12 已解决）：换装 OOOSplat 0.4.0 自带引擎（`.engines/ooosplat/engines`，
  `OOOSPLAT_ENGINE_DIR` 已 setx）后，COLMAP 换为官方 **4.0.4 CUDA 构建**（4.1.0.dev0）——
  nocuda 几何验证全零不再复现（原生 mapper 正常，OpenCV 回退未触发）；Brush v0.3 训练退化
  （46 splat）根因为 nocuda COLMAP 产出的稀疏输入质量，同一 Brush 二进制（sha256 与
  OOOSplat 打包一致）+ CUDA COLMAP 稀疏后正常产出 107,251 splat（240 帧全链路 389s）。
  顺带修复 `reconstruct.py` 特征提取旗标未跟随 COLMAP 4.x 改名的问题
  （`--SiftExtraction.use_gpu` → `--FeatureExtraction.use_gpu`，先新后旧自适应，82 项测试全过）。
另：顺带修复溅射合成预览三处还原度 bug（截断丢最近 splat、妆容层视差偏移、缺失锚点
溅射层与轮廓软边），PSNR 13.6→22.2 dB；修复 tracker._pose 在 OpenCV 5 下的崩溃
（solvePnP objectPoints 须为 (N,3)）——该 bug 影响实时化妆台的姿态估计。

## 长线（保持 README 路线图，依赖 P1/P2 铺垫）

- **FLAME 3DMM 拟合**：P1 完成姿态/表情解耦后，把表情层从 468 点直驱升级为 FLAME
  系数驱动（张口/皱眉时唇色眼影形变更真实），接口已留（`CanonicalFaceModel` 载入侧
  换 mesh + 系数偏移表）。
- **移动端**：ARKit `ARFaceAnchor` 直接给 blendshapes + UV 布局，RegionMaskBaker
  逻辑可整体复用；sidecar 换原生追踪（P4 的二进制协议即为此设计）。
- **自然语言改妆**：agent 直接改 spec 单层下发（`--only lipstick` + 色号替换），
  "口红换番茄色"一句话完成——skill 侧已具备，补一条 SKILL.md 工作流示例即可。

## 实施顺序建议

```
P0（1~2d）→ P1.1 单摄像头（1.5d）→ P1.2 姿态对齐（2d）→ P1.3 体验（1d）
→ P2.1 环境光合成（2d）→ P2.2 皮肤材质（1.5d）→ P3.1 参考图对比（1d）
→ P2.3 真 3DGS（2d）→ P3.2~3.4（2d）→ P4 穿插 → 重录 demo 视频
```

每个阶段完成即更新 README「已验证」一节并重跑 preview 冒烟，保证仓库随时可交付。
