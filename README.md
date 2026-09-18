# Makeup Assistant —— 3DGS 数字资产试妆 + AI 化妆助手

协助女生化妆的应用原型。**产品主链路**（R 写实化重构后）：

```
上传视频 / 摄像头环绕扫描 ─→ 抽帧 + SfM(pycolmap) ─→ gsplat 光度训练（表情一致帧）
                          ─→ 3DGS 数字资产（base.ply，20万+ 高斯，~4 分钟）
换妆 ─→ UV 目标场合成 + 3D 唇锚定 + PBR 材质 ─→ madeup.ply/.splat + material.bin（秒级）
```

| 部分 | 说明 |
|---|---|
| **[desktop-app/makeupstudio/](desktop-app/)** | ★ 核心：写实试妆管线（`face3dgs/appearance/`：选帧/训练/UV 妆容/PBR/评测）+ 桌面 App（PySide6，采集→建模→贴妆三步走）+ CLI（`python -m makeupstudio.face3dgs asset|makeup`） |
| **[makeup-skill/](makeup-skill/)** | Agent Skill 能力包（`SKILL.md` + 妆容预设 + spec schema + Bridge 协议）：妆容解析（VLM）与实时化妆协助 |
| **[unity-app/](unity-app/)** | Windows 桌面试妆 App（Unity 2022 LTS）：3DGS 画像 EWA 高斯光栅渲染，`GaussianAvatarParser.LoadMaterial` 消费 MKMA 材质 sidecar 做 PBR |
| **[preview/](preview/)** | CLI：`run_photoreal.py`（上传视频全自动 / 已有工程分步）+ 演示工具 |

▶ 效果对比图：[out/compare/photoreal_3views.png](out/compare/photoreal_3views.png)（真实帧|素颜|妆后 三视角）· [out/compare/photoreal_lip_zoom.png](out/compare/photoreal_lip_zoom.png)（唇部特写）

## 快速开始（产品主链路）

```bash
# 0) 环境：CUDA GPU + pip install torch gsplat pycolmap（详见 docs/photoreal-pipeline.md）
cd desktop-app && python -m makeupstudio.face3dgs status   # 环境自检

# 1) 资产化（二选一）
#    a. 摄像头环绕扫描（桌面 App 内点击，或 CLI 引导式采集）
cd desktop-app && python -m makeupstudio.face3dgs capture -o ../out/face3dgs/me.mp4
#    b. 上传视频全自动：抽帧 → SfM → gsplat 训练 → base.ply
python preview/run_photoreal.py --video 我的视频.mp4 \
    --project out/scan --spec makeup-skill/presets/date-rose.json --out out/photoreal/me

# 2) 换妆（秒级，复用资产）
cd desktop-app && python -m makeupstudio.face3dgs makeup -p ../out/scan \
    --spec ../makeup-skill/presets/date-rose.json
```

## 旧链路退役说明

R 写实化重构（2026-09，详见 [docs/photoreal-pipeline.md](docs/photoreal-pipeline.md)）退役了以下路径，代码已删除或归入语义工具层：
- `preview/run_real_fit.py` / `preview/compare_bare_vs_makeup.py`（模板面具链路 CLI）
- `fit_makeup.py` 的模板染色（apply_makeup）、程序化壳层（build_makeup_shell）、fit/fit_canonical/apply_guidance 旧入口、render_cloud 旧渲染器——**保留**三角化/配准/唇拓扑/唇带权重/UV 归属（新链路的语义基础设施）
- `reconstruct.py` 的 Brush 重建链路（FFmpeg+COLMAP+Brush 引擎依赖）→ 被 pycolmap SfM + gsplat 训练取代（`reconstruct.py`/`isolate.py` 保留为参考实现，主链路不再依赖）
- desktop-app "我的 3D 脸"标签页：ReconWorker/FitWorker → ModelWorker（扫描建模）/MakeupWorker（贴妆）

三大能力中的 **妆容素材解析**（VLM → makeup_spec.json）与 **实时化妆协助**（进度跟踪/TTS/报告）流程不变，产物 spec 直接喂给新链路的妆容合成。

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
preview/               recorder_app.py、run_demo_session.py、face_render.py(薄封装)、render_demo_video.py、
                       sculpt_face_avatar.py(类真人雕刻头像：离线联调/渲染基准，产出锚点 JSON)
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

## 3DGS 妆容管线写实化重构（R0-R3，2026-09）

对照旧链路效果不逼真的四个根因（模板面具底模 / 退化重建 / 逐 splat 涂色 /
玩具级光照）做的架构级重构，详见 [docs/photoreal-pipeline.md](docs/photoreal-pipeline.md)：

- **R0 表情一致选帧 + gsplat 光度训练**（`appearance/frames.py`、`train_base.py`）
  - 单目说话视频中表情剧烈变化是 3DGS densify 崩塌的第一根因（历史产物
    仅 1918 高斯）→ MediaPipe 嘴开度窄带（贴中位数 ±0.025）+ MAD 表情聚类，
    只训主表情簇；
  - gsplat 30k iter 全参数训练（densify/split/opacity-reset），脸区凸包蒙版
    聚焦，产出 20 万+ 高斯的真 3DGS 底模；后处理剪枝（蒙版外漂浮物/巨块/低
    透明度）。8GB 显存 ~4 分钟。
- **R1 UV 空间妆容**（`appearance/uvbind.py`、`makeup_uv.py`）
  - canonical UV/区域覆盖绑定为点云一等属性，妆容在 2048² UV 目标场合成
    （albedo/rough/coat/sss/sheen/kL/chroma），Lab 部分迁移烘焙——皮肤纹理
    与原生光影保留；系数逐区域（底妆只匀肤，唇/腮红强色度）；
  - **唇妆 3D 锚定**：canonical 模板唇带与真实唇有 ~2% 错位（唇是高频小区域，
    UV 路径不可接受）→ 复用 fit_makeup L0 观测唇域锚定（三角化真实地标 +
    颜色门控），UV 唇带降级为兜底；
  - 微观纹理（唇纹/粉感）第一次有来源；`optimize.py` 预留 Stable-Makeup
    guidance 的图像空间联合求解回路（几何冻结，identity 锁）。
- **R2 PBR 化妆品材质**（`appearance/pbr.py`、Unity `GaussianAvatarSplat.shader`）
  - 逐 splat 材质通道：rough（粉↑釉↓）/ coat（唇釉清漆 + Schlick Fresnel）/
    sss（唇部背光透光红移）/ sheen（珠光绒光），Python/Unity 同式实现；
  - 导出 `material.bin`（MKMA v1：法线+材质，`GaussianAvatarParser.LoadMaterial`）。
- **R3 客观指标**（`appearance/bench.py`）：留出帧 PSNR / 唇区 ΔE / 非妆区
  ΔE（identity 锁验证），杜绝玄学迭代。
- 环境：CUDA torch 2.9.1+cu130 + gsplat 1.5.3（源码编译，v1.5.3 上游存在
  2DGS-bwd/from-world 内核签名不一致，已在安装中打桩绕过，主路径不受影响）。
- 一键跑法：
  `python preview/run_photoreal.py --project out/real --sfm out/real/sfm/sparse/3 --init out/real/project/face_dense.ply --spec makeup-skill/presets/date-rose.json --out out/photoreal/d6 --iters 30000`

## 3DGS 妆容管线升级（P0/P1/P2，2026-09）〔已退役，被 R 写实化重构取代〕

> 本节描述的模板染色/壳层路径已随 R 重构退役（保留的思想被继承）：
> kNN UV 归属 + 跨岛保护 + 离群剔除 → `appearance/uvbind.py`；唇红带 3D 拓扑 +
> 观测锚定 + 颜色门控 → `fit_makeup.FaceMakeupFitter`（appearance 消费）；
> guidance 多视角聚合思想 → `appearance/optimize.py`（图像空间联合求解）；
> DLT Hartley 归一化/cheirality、colmap_io standard 方言、精确 UV 优先等
> 真实场景修复全部保留在新链路。原 P0/P1/P2 细节见 git 历史。

- **P0 烘焙质量**（`face3dgs/fit_makeup.py`、`splat3d.py`）
  - kNN 距离加权 UV 归属 + 跨 UV 岛保护 + 离群剔除（头发/背景永不涂妆）；
  - `densify=True` 妆区克隆小 σ 子高斯（≤35%），唇线/眼线锐化；
  - 蒙版烘焙提到 1024（`FIT_TEX`）；`splat3d` 新增 alpha² 加权超细细节层；
  - finish 物理化：逐点 gloss/shininess 烘进点云，三条渲染路径（Python 预览 /
    WebGL viewer / Unity `GaussianSplat` shader）按真实视角加 Blinn-Phong +
    fresnel sheen——唇釉/珠光高光随视角流动，不再是 opacity 增益；
  - 修复 `splat3d.build` 烘焙缓存 key 撞车（同层数不同 spec → 妆容整体消失）。
- **P1 外观与参数解耦**（`face3dgs/fit_makeup.apply_guidance` + `face3dgs/guidance.py`）
  - 妆容外观可改由 guidance 图（Stable-Makeup 等 2D 迁移模型输出）多视角投影
    采样（逐通道中位数聚合），参数化 spec 降级为区域门控 + 强度滑杆；
  - Stable-Makeup 走与 FlashAvatar 相同的门控适配约定（`status()` 可执行提示 +
    conda 独立环境），环境不齐时自动回退参数化路径。
- **P2 FLAME 底座接缝**（`face3dgs/fit_makeup.fit_canonical`）
  - canonical 姿态点云（FlashAvatar 适配器导出）零配准直接上妆，
    `fit_report.mode="canonical"`；表情/动画一致性的底座已就位。
