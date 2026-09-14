# Makeup Assistant —— 3DGS 画像试妆 + AI 化妆助手

协助女生化妆的应用原型，两部分组成：

| 部分 | 说明 |
|---|---|
| **[makeup-skill/](makeup-skill/)** | ★ Agent Skill 能力包（`SKILL.md` + Python 工具 + 妆容预设 + 协议文档）。整个目录即为分发形态：agent 下载后放入自己的 skills 目录即可识别调用 |
| **[unity-app/](unity-app/)** | Windows 桌面试妆 App（Unity 2022 LTS）。**画像妆容台流程**（P5）：用户上传 3DGS 画像 → 选妆在画像上预览（EWA 高斯光栅 + 妆容 tint）→ 确认 → 进妆容台辅助化妆（画像侧栏参照 + 摄像头画面供 VLM 指导，真脸零渲染） |
| **[preview/](preview/)** | 演示与联调工具：真实 Bridge 会话录制器 + Python 预览渲染器（免 Unity 预览效果 / 生成演示视频） |

▶ **效果演示视频**：[out/demo/makeup-demo.mp4](out/demo/makeup-demo.mp4)（54s，真实 Bridge 协议会话的离线复现，含 live_coach 离线陪练章节）

## 三大能力

1. **画像试妆（默认）**：用户上传 3DGS 画像（单照片经 LAM 类模型生成，或扫描拟合导出），妆容编译为**每-Gaussian tint + 表面附加溅射**在画像上实时渲染；裸妆|妆后并排预览确认后进入妆容台，画像作目标参照。原"真脸 AR 附妆"保留为 legacy 开关。
2. **妆容素材解析**：上传妆容照片/视频 → VLM（OpenAI 兼容接口，默认 GLM-4V，可换 GPT-4o 等）逐帧解析 → 结构化 [`makeup_spec.json`](makeup-skill/references/schema.md) → 自动渲染"同款妆效果图"预览 → 自动烘焙资产 → 可直接试妆。
3. **实时化妆协助**：生成**目标妆参考图**（画像流程用画像渲染效果）；化妆过程中周期采样摄像头帧，VLM 对比参考图与当前进度，滑动窗投票推进步骤（防单帧误判），分步提醒推送到 App（文字 + 步骤进度条 + TTS 语音），附带环境光检测；结束产出带部位评分的会话报告。

## 架构

```
用户/Agent(AI)                      试妆 App(Unity)                    本地摄像头
   │  avatar_register/preview/station     ▲   │ 帧中继(UDP :8767，可选)    │
   ▼  apply_spec(legacy)/coaching         │   ▼                           │
[bridge_server 1.2 ws://127.0.0.1:8765]──┴──[BridgeClient]        [WebcamDisplay]
   │  精确路由(to/ref) · 资产HTTP侧车(:8768)      │
   avatar_session.py ─▶ 画像注册/妆容编译下发 ─▶ GaussianAvatarParser → GaussianAvatarRenderer
   │  （PLY→ComputeBuffer，CPU视深排序+DrawProcedural EWA光栅，tint per-Gaussian 混妆）
   │                                        └▶ AvatarStationFlow（idle→registered→preview→confirmed→station）
   parse_look.py → makeup_spec.json ─▶ (legacy) RegionMaskBaker/MakeupLayerRenderer/SplatLayerRenderer
   bake_assets.py → 纹理+splat配置      （真脸附妆链路，默认关闭）
   face_tracker.py ──UDP v2 二进制──▶ (legacy) FaceMeshDeformer（姿态/表情解耦 + 1€ 滤波）
```

- Skill 包轻量自包含（脚本+文档+预设+渲染内核）；App 按公开 [Bridge 协议](makeup-skill/references/bridge-protocol.md)（v1.2）对接，可替换为任何客户端实现（preview/ 里有 Python 等价实现）。
- 画像 ply 与编译产物全程本地（HTTP 侧车在本机 8768）；摄像头帧默认全程本地；仅实时指导开启时采样帧才发给 VLM 服务商。画像属敏感生物特征，采集前须用户知情。

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

# 4) 画像试妆一条龙（默认流程；画像获取见 makeup-skill/references/avatar-setup.md）
python makeup-skill/scripts/avatar_session.py register --ply 我的画像.ply --name 我的画像
python makeup-skill/scripts/avatar_session.py preview  --avatar 我的画像 --spec makeup-skill/presets/date-rose.json
python makeup-skill/scripts/avatar_session.py confirm  --avatar 我的画像
python makeup-skill/scripts/avatar_session.py station  --avatar 我的画像
python makeup-skill/scripts/live_coach.py --spec makeup-skill/presets/date-rose.json \
    --station --avatar-look "out/avatars/我的画像/compiled/date-rose"

# 5) 其他能力
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
python -m pytest tests/ -q     # 84 项：画像管线（PLY/语义/编译器/光栅）/追踪协议/蒙版规则/渲染/
                               # bridge 路由/资产侧车/教练状态机/C#·shader 静态检查
```

## 目录

```
makeup-skill/          SKILL.md｜scripts/（bridge、avatar_session、avatar_io/语义/编译器/画像渲染、
                       apply_spec、parse_look、bake_assets、live_coach、render_look、preview_render、
                       setup_check、vlm 抽象、dev/mock_app）｜presets/（3 妆容）｜
                       references/（schema、bridge 协议、avatar-setup、app-setup、关键点映射、
                       canonical 网格、VLM 提示词模板）
unity-app/             Assets/Scripts（15 个 C#：主入口、Bridge、画像解析/渲染、附加溅射、妆容台
                       状态机、UDP 追踪接收、姿态变形、1€ 滤波、蒙版烘焙、mesh 妆容层、溅射层、
                       摄像头/中继/环境光、指导显示+TTS）、Assets/Shaders（4 个 shader，含
                       GaussianAvatarSplat EWA 光栅）、StreamingAssets、tools/（face_tracker +
                       tracking_protocol）、docs/assembly.md
preview/               recorder_app.py、run_demo_session.py、face_render.py(薄封装)、render_demo_video.py
tests/                 pytest 套件（84 项，含 check_csharp.py 静态回归）
out/demo/              makeup-demo.mp4 演示视频
```

## 已验证

- `pytest tests/` **84 项**全过（新增画像管线离线单测：PLY 往返、语义归属、编译器 region
  隔离/--only、软件光栅裸妆/妆后差异；C#·shader 静态检查现随默认收集运行）
- **画像妆容台链路（P5）**：`avatar_session` register→preview→confirm→station 消息流、
  MKMKP1 tint / MKSEM1 语义二进制两侧一致、真脸附妆默认关闭（`legacyFaceMakeup=false`，
  apply_spec 在画像模式回明确指引）
- Bridge v1.1→v1.2 端到端：内嵌与 HTTP 侧车两种资产形态下发、按 ref 回帧、`to` 定向、断线重连，实测通过
- live_coach `--test` 真实闭环：取帧 → 离线剧本 → 进度推送 → 光线提醒 → 评分报告（session_report.md）；
  新增 `--station`/`--avatar-look`（妆容台联动 + 画像渲染参考图）
- 演示会话 32 条事件（5 次妆容下发含资产、16 条指导、9 次取帧、卸妆）全部 ack 成功
- 关键修复：右眉不渲染、StopCoroutine 协程失效、假 fps 上报、唇溅射轮廓锯齿（锚点内收）、
  1€ 滤波米尺度参数、check_csharp 字符串剥离与默认收集缺失

## 路线图

- ~~真·3DGS 排序渲染管线~~ ✓（画像：CPU 视深排序 + DrawProcedural EWA；>30 万点再上 GPU bitonic）
- ~~指导语音化~~ ✓（System.Speech / PowerShell SAPI 两级实现）
- 画像表情驱动（P6）：FLAME 系数 rig → 画像 Gaussian 蒙皮（静态画像渲染已就绪，rig 接口已留）
- ~~FLAME 3DMM 精确拟合~~（并入画像架构：LAM/扫描产物即 FLAME-rig 画像）
- iOS/Android 移植（ARKit/ARCore 前置摄像头 + 原生人脸追踪替代 sidecar）
