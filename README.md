# Makeup Assistant —— 3D 实时试妆 + AI 化妆助手

协助女生化妆的应用原型，两部分组成：

| 部分 | 说明 |
|---|---|
| **[makeup-skill/](makeup-skill/)** | ★ Agent Skill 能力包（`SKILL.md` + Python 工具 + 妆容预设 + 协议文档）。整个目录即为分发形态：agent 下载后放入自己的 skills 目录即可识别调用 |
| **[unity-app/](unity-app/)** | Windows 桌面试妆 App（Unity 2022 LTS）。单摄像头闭环：App 取流 → sidecar 追踪（468 点 + 6DoF 姿态）→ 姿态/表情解耦驱动 3D 网格，妆容以蒙版材质层 + 排序实例化高斯溅射层**贴在用户脸上** |
| **[preview/](preview/)** | 演示与联调工具：真实 Bridge 会话录制器 + Python 预览渲染器（免 Unity 预览效果 / 生成演示视频） |

▶ **效果演示视频**：[out/demo/makeup-demo.mp4](out/demo/makeup-demo.mp4)（54s，真实 Bridge 协议会话的离线复现，含 live_coach 离线陪练章节）

## 三大能力

1. **选妆试妆**：用户选妆容（内置预设或 AI 解析生成），App 实时渲染"化完妆"的样子——转头时妆容跟随脸部（6DoF 姿态对齐），支持 0–100% 浓度滑杆（双向同步）、单品试妆（只换口红）、一键卸妆（淡出过渡）。妆容随环境光合成光影，脸缘羽化消除"面具感"。
2. **妆容素材解析**：上传妆容照片/视频 → VLM（OpenAI 兼容接口，默认 GLM-4V，可换 GPT-4o 等）逐帧解析 → 结构化 [`makeup_spec.json`](makeup-skill/references/schema.md) → 自动渲染"同款妆效果图"预览 → 自动烘焙资产 → 可直接试妆。
3. **实时化妆协助**：启动即生成**目标妆参考图**；化妆过程中周期采样摄像头帧，VLM 对比参考图与当前进度，滑动窗投票推进步骤（防单帧误判），分步提醒推送到 App（文字 + 步骤进度条 + TTS 语音），附带环境光检测；结束产出带部位评分的会话报告。

## 架构

```
用户/Agent(AI)                      试妆 App(Unity)                    本地摄像头
   │  apply_spec/coaching                ▲   │ 帧中继(UDP :8767)         │
   ▼  (makeup-skill/scripts/*)           │   ▼                           │
[bridge_server 1.1 ws://127.0.0.1:8765]──┴──[BridgeClient]        [WebcamDisplay]
   │  精确路由(to/ref) · 资产HTTP侧车(:8768)      │
   parse_look.py → makeup_spec.json ─▶ RegionMaskBaker（后台线程 UV 蒙版）
   bake_assets.py → 纹理+splat配置 ─▶ MakeupLayerRenderer（环境光合成/羽化/淡出）
                                    └▶ SplatLayerRenderer（实例化+视深排序高斯溅射）
   face_tracker.py ──UDP v2 二进制(:8766)──▶ FaceMeshDeformer
   （4×4 姿态=根节点 · 头部局部点=表情层 · 1€ 滤波 · MKT2/MKF1 协议见 tracking_protocol.py）
```

- Skill 包轻量自包含（脚本+文档+预设+渲染内核）；App 按公开 [Bridge 协议](makeup-skill/references/bridge-protocol.md)（v1.1）对接，可替换为任何客户端实现（preview/ 里有 Python 等价实现）。
- 摄像头帧默认全程本地；仅实时指导开启时采样帧才发给 VLM 服务商。

## 快速开始

```bash
# 1) Agent 侧：把 makeup-skill/ 放入 skills 目录后
pip install -r makeup-skill/requirements.txt
python makeup-skill/scripts/setup_check.py     # 环境自检（端口被占会给出换端口指引）
python makeup-skill/scripts/bridge_server.py   # 启动 Bridge（含资产 HTTP 侧车）

# 2) App 侧：人脸追踪 sidecar（先于 App 启动；单摄像头闭环）
pip install mediapipe opencv-python numpy
python unity-app/tools/face_tracker.py --relay
#   无摄像头联调：python unity-app/tools/face_tracker.py --synthetic

# 3) App：按 unity-app/docs/assembly.md 用 Unity 2022 LTS 构建并运行 MakeupMirror.exe

# 4) 试妆一条龙（agent 执行或手动）
python makeup-skill/scripts/apply_spec.py --spec makeup-skill/presets/daily-natural.json
python makeup-skill/scripts/parse_look.py --input 想要的妆容.jpg --out out/parsed   # 需 MAKEUP_VLM_API_KEY
python makeup-skill/scripts/live_coach.py --spec makeup-skill/presets/date-rose.json  # 需 App + VLM
python makeup-skill/scripts/live_coach.py --spec ... --test                          # 离线剧本联调
```

VLM 配置：环境变量 `MAKEUP_VLM_API_KEY`（必填）/ `MAKEUP_VLM_BASE_URL` / `MAKEUP_VLM_MODEL`，见 [SKILL.md](makeup-skill/SKILL.md)。

## 免 Unity 预览 / 演示视频 / 效果图

```bash
python preview/run_demo_session.py    # 真实 bridge 会话：apply_spec/单品/陪练/解析/卸妆 → events.json
python preview/render_demo_video.py   # Python 预览渲染器复现会话 → out/demo/makeup-demo.mp4
python makeup-skill/scripts/render_look.py --spec makeup-skill/presets/date-rose.json \
    --out look.jpg --env warm --compare   # 静态效果图（无妆|有妆 并排）
```

`makeup-skill/scripts/preview_render.py` 是 Unity 渲染管线的 Python 等价实现（同一 canonical 网格、
同一 RegionMask 蒙版与羽化规则、同一 spec/溅射/光照模型），可在任何机器上验证妆效。

## 测试

```bash
python -m pytest tests/ -q     # 59 项：追踪协议/姿态求解/蒙版规则/渲染/bridge 路由/资产侧车/教练状态机/C# 静态检查
```

## 目录

```
makeup-skill/          SKILL.md｜scripts/（bridge、apply_spec、parse_look、bake_assets、live_coach、
                       render_look、preview_render、setup_check、vlm 抽象、dev/mock_app）｜
                       presets/（3 妆容）｜references/（schema、bridge 协议、app-setup、关键点映射、
                       canonical 网格、VLM 提示词模板）
unity-app/             Assets/Scripts（11 个 C#：主入口、Bridge、UDP 追踪接收、姿态变形、1€ 滤波、
                       蒙版烘焙、mesh 妆容层、溅射层、摄像头/中继/环境光、指导显示+TTS）、
                       Assets/Shaders（3 个 shader）、StreamingAssets（canonical 网格 + 关键点映射）、
                       tools/（face_tracker.py 三模式 + tracking_protocol.py）、docs/assembly.md
preview/               recorder_app.py、run_demo_session.py、face_render.py(薄封装)、render_demo_video.py
tests/                 pytest 套件（59 项）
out/demo/              makeup-demo.mp4 演示视频
```

## 已验证

- `pytest tests/` 59 项全过（姿态求解误差 ≤2.5°、蒙版左右对称/软羽化、bridge 精确路由/资产侧车、
  教练滑窗投票、C# 静态检查）
- Bridge v1.1 端到端：内嵌与 HTTP 侧车两种资产形态下发、按 ref 回帧、`to` 定向、断线重连，实测通过
- live_coach `--test` 真实闭环：取帧 → 离线剧本 → 进度推送 → 光线提醒 → 评分报告（session_report.md）
- 演示会话 32 条事件（5 次妆容下发含资产、16 条指导、9 次取帧、卸妆）全部 ack 成功
- 关键修复：右眉不渲染、StopCoroutine 协程失效、假 fps 上报、唇溅射轮廓锯齿（锚点内收）、
  1€ 滤波米尺度参数

## 路线图

- ~~真·3DGS 排序渲染管线~~ ✓（内置排序实例化后端；UnityGaussianSplatting 库接入为可选增强）
- ~~指导语音化~~ ✓（System.Speech / PowerShell SAPI 两级实现）
- FLAME 3DMM 精确拟合（姿态/表情已解耦，表情层可平滑升级为 FLAME 系数驱动）
- iOS/Android 移植（ARKit/ARCore 前置摄像头 + 原生人脸追踪替代 sidecar）
