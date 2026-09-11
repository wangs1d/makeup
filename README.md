# Makeup Assistant —— 3D 实时试妆 + AI 化妆助手

协助女生化妆的应用原型，两部分组成：

| 部分 | 说明 |
|---|---|
| **[makeup-skill/](makeup-skill/)** | ★ Agent Skill 能力包（`SKILL.md` + Python 工具 + 妆容预设 + 协议文档）。整个目录即为分发形态：agent 下载后放入自己的 skills 目录即可识别调用 |
| **[unity-app/](unity-app/)** | Windows 桌面试妆 App（Unity 2022 LTS）。摄像头 + 人脸关键点驱动 3D 网格，妆容以蒙版材质层 + 3D 高斯溅射层逼真附着 |
| **[preview/](preview/)** | 演示与联调工具：真实 Bridge 会话录制器 + Python 预览渲染器（免 Unity 预览效果 / 生成演示视频） |

▶ **效果演示视频**：[out/demo/makeup-demo.mp4](out/demo/makeup-demo.mp4)（46s，真实 Bridge 协议会话的离线复现）

## 三大能力

1. **选妆试妆**：用户选妆容（内置预设或 AI 解析生成），App 实时渲染"化完妆"的样子——转头时妆容跟随脸部，支持 0–100% 浓度滑杆、单品试妆（只换口红）、一键卸妆。
2. **妆容素材解析**：上传妆容照片/视频 → VLM（OpenAI 兼容接口，默认 GLM-4V，可换 GPT-4o 等）逐帧解析 → 结构化 [`makeup_spec.json`](makeup-skill/references/schema.md) → 可直接试妆的"同款妆容"。
3. **实时化妆协助**：化妆过程中周期采样摄像头帧，VLM 对比目标妆容与当前进度，分步提醒推送到 App（文字 + TTS 钩子），附带环境光检测。

## 架构

```
用户/Agent(AI)                      试妆 App(Unity)                 本地摄像头
   │  apply_spec/coaching                ▲   │ frame/tracking
   ▼  (makeup-skill/scripts/*)           │   ▼
[bridge_server.py ws://127.0.0.1:8765] ──┴──[BridgeClient.cs]
                                             │
   parse_look.py → makeup_spec.json ─▶ RegionMaskBaker（关键点→UV 蒙版）
   bake_assets.py → 纹理+splat配置 ─▶ MakeupLayerRenderer（mesh 妆容层）
                                     └▶ SplatLayerRenderer（高斯溅射体积层）
   face_tracker.py ──UDP 468点──▶ FaceMeshDeformer（canonical mesh 变形）
```

- Skill 包轻量自包含（脚本+文档+预设）；App 按公开 [Bridge 协议](makeup-skill/references/bridge-protocol.md) 对接，可替换为任何客户端实现（preview/ 里就有一个 Python 等价实现）。
- 摄像头帧默认全程本地；仅实时指导开启时采样帧才发给 VLM 服务商。

## 快速开始

```bash
# 1) Agent 侧：把 makeup-skill/ 放入 skills 目录后
pip install -r makeup-skill/requirements.txt
python makeup-skill/scripts/setup_check.py     # 环境自检
python makeup-skill/scripts/bridge_server.py   # 启动 Bridge（后台）

# 2) App 侧：人脸追踪 sidecar（先于 App 启动）
pip install mediapipe opencv-python numpy
python unity-app/tools/face_tracker.py

# 3) App：按 unity-app/docs/assembly.md 用 Unity 2022 LTS 构建并运行 MakeupMirror.exe

# 4) 试妆一条龙（agent 执行或手动）
python makeup-skill/scripts/apply_spec.py --spec makeup-skill/presets/daily-natural.json
python makeup-skill/scripts/parse_look.py --input 想要的妆容.jpg --out out/parsed   # 需 MAKEUP_VLM_API_KEY
python makeup-skill/scripts/live_coach.py --spec makeup-skill/presets/date-rose.json  # 需 App + VLM
```

VLM 配置：环境变量 `MAKEUP_VLM_API_KEY`（必填）/ `MAKEUP_VLM_BASE_URL` / `MAKEUP_VLM_MODEL`，见 [SKILL.md](makeup-skill/SKILL.md)。

## 免 Unity 预览 / 演示视频

```bash
python preview/run_demo_session.py    # 真实 bridge 会话：apply_spec/单品/指导/卸妆 → events.json
python preview/render_demo_video.py   # Python 预览渲染器复现会话 → out/demo/makeup-demo.mp4
```

`preview/face_render.py` 是 Unity 渲染管线的 Python 等价实现（同一 canonical 网格、同一
`RegionMaskBaker` 逻辑、同一 spec/溅射规则），可在任何机器上验证妆容效果，无需构建 Unity 工程。

## 目录

```
makeup-skill/          SKILL.md｜scripts/（bridge、apply_spec、parse_look、bake_assets、
                       live_coach、setup_check、vlm 抽象、dev/mock_app）｜presets/（3 妆容）
                       references/（schema、bridge 协议、app-setup、关键点映射、VLM 提示词模板）
unity-app/             Assets/Scripts（9 个 C#）、Assets/Shaders（3 个 shader）、
                       StreamingAssets（canonical 人脸网格 + 关键点映射）、tools/face_tracker.py、
                       docs/assembly.md（装配/升级/排查）
preview/               recorder_app.py、run_demo_session.py、face_render.py、
                       masks_py.py、render_demo_video.py
out/demo/              makeup-demo.mp4 演示视频
```

## 已验证

- 全部 Python 脚本编译通过；Unity 文件静态检查通过
- Bridge 端到端：带资产下发完整妆容（19 纹理 446KB）、单品试妆、卸妆、摄像头取帧回传，均实测通过
- 演示会话 12 条事件（4 次妆容下发含资产、6 条实时指导、卸妆）全部 ack 成功
- 关键修复：canonical obj 的 v/vt 索引是图集置换（Unity 与 Python 两侧同步修复为顶点置换映射）

## 路线图

- 真·3DGS 排序渲染管线（UnityGaussianSplatting 后端接入，见 assembly.md 第 6 节）
- FLAME 3DMM 精确拟合（当前为 MediaPipe canonical 网格直驱，升级路径已留接口）
- iOS/Android 移植（ARKit/ARCore 前置摄像头 + 原生人脸追踪替代 sidecar）
- 指导语音化（Windows TTS 钩子已留）
