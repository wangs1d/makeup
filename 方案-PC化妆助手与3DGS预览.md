# 方案：去 Unity 的 PC 摄像头化妆助手 + 3D 高斯泼溅效果预览

> 2026-09-12 · 目标：① 不依赖 Unity，在 PC 上打开摄像头辅助用户化妆；② 设计一个以 3D 高斯泼溅（3DGS）方式展示所选妆容效果的界面。

## 0. 现状结论（为什么去 Unity 成本低）

对仓库的调研结论：**Unity 只是渲染/展示外壳，算法核心已全在 Python 侧**。

| 能力 | 现状 | 位置 |
|---|---|---|
| 人脸追踪（MediaPipe FaceMesh 468 点 + 6DoF 姿态） | ✅ 已实现，`--camera` 模式现成 | `unity-app/tools/face_tracker.py` |
| 渲染内核（UV 蒙版烘焙、羽化、光照合成、splat 绘制） | ✅ 已实现，与 Unity 管线逐位等价（836 行） | `makeup-skill/scripts/preview_render.py` |
| 高斯 splat 生成（按妆容 layers 生成各向异性高斯锚点） | ✅ 已实现 | `preview_render.py::build_splats` |
| 妆容解析（VLM → makeup_spec）、陪练（live_coach）、预设 | ✅ 已实现 | `makeup-skill/scripts/*`、`presets/` |
| 摄像头取流 + 画面展示 + UI | ❌ 只在 Unity 里（`WebcamDisplay.cs` 等） | 需替换 |
| 真 3DGS（多视角训练重建） | ❌ 无；现有 splat 是“锚定关键点的 billboard 高斯近似” | 见 §3 |

因此：**去掉 Unity = 把“取流 + 渲染展示 + UI”搬进一个 Python 桌面壳，算法零重写。**

## 1. 总体架构

一个 PyQt6 桌面应用 **MakeupStudio**（新增 `desktop-app/` 目录），同一窗口承载两个目标界面：

```
┌────────────────────────────────────────────────────────────┐
│  MakeupStudio (PyQt6)                                      │
│  ┌──────────────────────────┬───────────────────────────┐  │
│  │  ① 实时化妆台             │  妆容库（3预设+VLM解析）    │  │
│  │  摄像头画面 + 实时妆效叠加  │  单品强度滑条(口红/眼影/…)  │  │
│  │  (目标一)                 ├───────────────────────────┤  │
│  │                          │  ② 3D 效果预览 (3DGS)      │  │
│  │                          │  可旋转/缩放 splat 头像     │  │
│  │                          │  素颜/上妆对比 · 快照导出    │  │
│  ├──────────────────────────┴───────────────────────────┤  │
│  │  live_coach 陪练条：当前步骤 · 进度 · VLM 建议 · TTS    │  │
│  └────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
   相机线程(OpenCV) → 追踪(MediaPipe, 进程内) → 渲染(preview_render)
   ↕ 可选保留 bridge_server.py (ws://8765) —— Agent skill 协议不变
```

进程/线程结构：
- **主进程**：Qt UI。
- **相机线程**：OpenCV `VideoCapture`（`face_tracker.py` 已验证可行）→ 帧队列。
- **追踪**：`face_tracker.py` 的 `LandmarkModel` 改为进程内 import（不再走 UDP sidecar；sidecar 模式保留用于联调）。
- **渲染**：复用 `preview_render.py`（见 §2 改造点）。
- **Bridge**：`bridge_server.py` 协议原样保留——AI Agent skill 仍可远程下发妆容；应用内操作与 bridge 写同一状态。

## 2. 模块一：实时化妆辅助界面（替代 Unity）

### 界面布局
- **左侧主画面**：摄像头实时画面 + 妆效叠加（含环境光自适应——把 `WebcamDisplay.cs` 的取帧/平均亮度逻辑搬过来，喂给 `FaceRenderer.set_env`）。
- **右侧面板**：妆容库（daily-natural / date-rose / office-polish + 用户上传照片经 `parse_look.py` VLM 解析的新妆）；单品强度滑条与开关（lipstick / eyeshadow / eyeliner / blush / contour / highlight，直接映射 `makeup_spec` 字段，等价于 `apply_spec.py` 的能力）。
- **底部陪练条**：`live_coach.py` 状态机直连——周期采帧 → VLM 对比目标参考图与当前妆面 → 步骤进度 + 文字建议 + TTS 语音提醒。

### 关键改造点（本方案唯一需要新写的核心代码）
`preview_render.py` 当前 `render()` 是“程序化背景 + canonical 网格变形”，没有真实摄像头帧输入路径。新增一条**图像域合成路径**：

1. `FaceRenderer.render_webcam(frame_bgr, landmarks, makeup)`：
   - 用 MediaPipe 468 点 + 现有 `landmark-regions.json` 映射，在**真实图像域**生成各妆容区域蒙版（复用 `RegionMasks.bake` 的形状/羽化逻辑，只是锚点从 UV 网格换到图像关键点）；
   - 复用 `bake_makeup` 的颜色/渐变/强度模型，按蒙版 alpha 与光照自适应系数叠到摄像头帧上。
   - 首期选图像域蒙版而非 mesh warp：实现快、30fps 无压力；若侧脸姿态跟随不足，再把现有网格变形路径接上（代码已在，作 B 方案）。
2. 关键点平滑：`face_tracker.py --smooth` 已有一德滤波思路，进程内直接复用。

## 3. 模块二：3D 高斯泼溅效果图展示界面

先说清一个事实约束：**真 3DGS（从照片多视角训练重建）无法用单摄像头实时完成**（需多视角采集 + 训练）。因此分两层设计：

### 3.1 本期交付：3D Splat 预览视图（可旋转真 3D 渲染）
- **数据源**：`canonical_face_model.obj` + `build_splats(layers)`——已经能按所选妆容生成各向异性高斯（位置 / 颜色 / 协方差 / 强度），妆容切换即重新生成。
- **新增导出器** `splat_exporter.py`：把 `build_splats` 输出转标准 `.ply`（3DGS 格式）或 `.splat`（antimatter15 格式），字段一一对应（xyz / scale / rot quaternion / opacity / RGB）。
- **渲染器选型**：
  - **A（推荐）**：`QWebEngineView` 内嵌 [gsplat.js](https://github.com/dylanebert/gsplat.js) 或 antimatter15 splat viewer——深度排序 + 预乘 alpha 高斯渲染开箱即用，代码量最小，视觉效果最佳；
  - **B（备选）**：纯 Python `moderngl` 点精灵/instanced quad + 自写高斯 shader——无浏览器内核、打包体积小，工作量略高。
- **交互设计**：
  - 鼠标轨道旋转（正视 / 3/4 侧 / 侧视）+ 滚轮缩放，相机自动回位动画；
  - 妆容切换 → 实时重烘焙 splat 颜色（与实时化妆台同一数据源，**预览即所得**）；
  - 素颜/上妆左右分屏对比模式；
  - 四联图快照导出（正/3/4/侧 PNG）。
- **展示形态**：效果图面板 + 一键“应用此妆到实时化妆台”。

### 3.2 进阶（长期项 P2，可选）：真 3DGS 头像
单张照片 → FLAME 参数化拟合 → 预训练高斯头管线（GaussianAvatars / FlashAvatar 类）→ 把 `bake_makeup` 的 UV 蒙版烘焙进高斯颜色属性，得到“用户本人脸型的真 3DGS 上妆预览”。需要 GPU 与模型权重，与 README 路线图中 FLAME 项合并推进；本期不做，但 §3.1 的导出格式与 UI 已为其预留（同一预览面板换数据源即可）。

## 4. 里程碑

| 里程碑 | 内容 | 验收标准 |
|---|---|---|
| **M1** 实时试妆最小闭环 | `desktop-app/`：Qt 壳 + 相机线程 + MediaPipe 追踪 + `render_webcam` 图像域合成 + 3 预设妆 | 摄像头打开，转头/张嘴妆效跟随，≥30fps |
| **M2** 完整化妆辅助 | 妆容库面板、单品强度滑条、`live_coach` 陪练条 + TTS、VLM 上传解析妆 | 滑条实时生效；陪练按妆推进步骤并语音提醒 |
| **M3** 3DGS 效果预览 | `splat_exporter.py` + 3D 预览面板（gsplat.js）、旋转/缩放/对比/快照 | 切妆后 3D 预览 3 秒内更新；与实时妆效一致 |
| **M4** 打包分发 | PyInstaller 打包 exe（注意 mediapipe / QtWebEngine 的 hidden imports 与资源） | 双击 exe 即用，无需 Python 环境 |

依赖变化：`requirements` 新增 `PyQt6`、`mediapipe`（现为注释项）、`gsplat.js` 走内嵌静态资源（无需 node）。

## 5. 备选与决策点

- **为什么 Python 桌面而非浏览器/Electron**：渲染内核、追踪、陪练、VLM 全是 Python，浏览器方案要重写渲染（WebGL shader）+ MediaPipe JS，工作量约翻倍且丢掉现有协议。若未来要 Web 分发，`splat_exporter` 与 makeup_spec 可直接复用到 Web 前端（MediaPipe Tasks Vision + three.js）。
- **QWebEngineView 打包体积大**（~100MB+）：可接受则选 A；在意体积则 M3 落 B（moderngl）。
- **现有 `preview/` 演示链路与 `tests/` 不依赖 Unity**，去 Unity 后继续可用；`unity-app/` 保留归档即可，不做删除。
