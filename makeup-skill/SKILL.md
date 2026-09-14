---
name: makeup-assistant
description: >
  协助女生化妆的试妆与化妆指导能力包。当用户想试妆（选一个妆容在摄像头里实时看到自己化完妆的样子）、
  上传妆容照片/视频想要同款妆容、或者化妆过程中需要实时的分步指导与提醒时使用。
  支持基于 3D 人脸网格 + 高斯溅射(3DGS)的逼真试妆渲染、VLM 妆容素材解析、目标妆容对比的实时化妆协助。
---

# Makeup Assistant（化妆助手）

为用户提供三类化妆协助能力。所有能力通过本包 `scripts/` 下的 Python 工具完成，
工具之间通过本地 WebSocket Bridge（默认 `ws://127.0.0.1:8765`）与试妆 App 通信。

## 三大能力

1. **画像试妆（默认流程）** — 用户上传 3DGS 画像，选妆容在画像上实时预览（真高斯溅射渲染），确认后进入**妆容台**辅助化妆。真脸上不渲染任何妆容。
2. **解析妆容素材** — 用户上传目标妆容的照片/视频（博主视频、截图、自拍），解析成结构化妆容规格 `makeup_spec.json`，可直接用于画像试妆。
3. **实时化妆协助** — 化妆过程中，周期性采样摄像头帧，对比目标妆容与用户当前妆面，给出下一步动作提醒（如"左眼影向外晕染""口红内侧补色"），提示实时显示在 App 上。（legacy：真脸 AR 附妆试妆保留在 App 的 `legacyFaceMakeup` 开关后）

## 前置检查（每次会话开始时先执行）

```bash
python <skill_dir>/scripts/setup_check.py
```

它会报告：Python 依赖、VLM 配置、Bridge/App 连接状态。根据缺失项引导用户：

- 依赖未装 → `pip install -r <skill_dir>/requirements.txt`
- VLM 未配置 → 需要设置环境变量（见下文「VLM 配置」），能力二/三必须，能力一不需要
- Bridge 未运行 → `python <skill_dir>/scripts/bridge_server.py &`（后台启动，默认端口 8765；
  若提示"端口被其他服务占用"，换 `--port 8865` 并 `set MAKEUP_BRIDGE_URL=ws://127.0.0.1:8865`）
- App 未连接 → 按 `references/app-setup.md` 引导用户启动试妆 App（能力一/三必须）

## VLM 配置

所有视觉理解通过 OpenAI 兼容接口，环境变量配置（放用户的 shell 配置或 `.env`）：

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `MAKEUP_VLM_BASE_URL` | OpenAI 兼容 API 地址 | `https://open.bigmodel.cn/api/paas/v4`（智谱 GLM） |
| `MAKEUP_VLM_API_KEY` | API Key（必填才能用能力二/三） | — |
| `MAKEUP_VLM_MODEL` | 视觉模型名 | `glm-4v-flash` |

可换成任何 OpenAI 兼容服务商（如 `https://api.openai.com/v1` + `gpt-4o`）。

## 工作流〇：画像试妆 + 妆容台（默认流程）

前置：Bridge 运行中（必须含资产 HTTP 侧车，默认开启）+ App 已连接。试妆预览不需要 VLM。

流程：**用户上传 3DGS 画像 → 选妆在画像上预览 → 用户确认 → 进妆容台辅助化妆**。
妆容只渲染在画像上（真 3DGS 高斯溅射），摄像头里的真脸零附着；画像获取与语义锚点见
`references/avatar-setup.md`。

1. **注册画像**（用户提供 ply 文件后）：

```bash
python <skill_dir>/scripts/avatar_session.py register --ply <画像.ply> --name 我的画像
```

   自动归一化（居中+脸高=1）并抽稀到 12 万高斯，经 HTTP 侧车上传给 App 渲染。

2. **选妆预览**（把 `compiled/<look>/preview.jpg` 并排图给用户看，附各部位色号说明）：

```bash
python <skill_dir>/scripts/avatar_session.py preview --avatar 我的画像 --spec <skill_dir>/presets/date-rose.json
```

   - 单品改妆（用户说"只换口红"）：加 `--only lipstick`，其余部位清零重编译；
   - 换色号：改 spec 里对应 layer 的 `color_stops` 后重跑 preview（tint 按 region 索引，秒级）；
   - 语义标注首次自动（正脸渲染+MediaPipe 反投影），结果缓存；失败时按
     avatar-setup.md 给手动锚点 `--anchors anchors.json`。

3. **确认 + 进妆容台**（用户满意后）：

```bash
python <skill_dir>/scripts/avatar_session.py confirm --avatar 我的画像
python <skill_dir>/scripts/avatar_session.py station --avatar 我的画像
```

   App 进入妆容台布局：画像移到侧栏作目标参照，摄像头画面保留供取帧，指导字幕/进度条/语音全开。

4. **辅助化妆**：工作流三照常，但加 `--station`（开始/结束联动妆容台）与
   `--avatar-look <compiled目录>`（VLM 参考图改用画像渲染效果）：

```bash
python <skill_dir>/scripts/live_coach.py --spec <妆容spec> --station \
    --avatar-look out/avatars/我的画像/compiled/date-rose
```

## 工作流一：选妆试妆（legacy 真脸链路）

前置：Bridge 运行中 + App 已连接。不需要 VLM。（legacy：需要 App 开 `legacyFaceMakeup`；
默认画像流程走工作流〇）

1. 列出可选妆容：读取 `<skill_dir>/presets/*.json` 的 `name` 和 `description` 字段向用户展示；若之前解析过用户的妆容素材，也一并列出（见工作流四的产出目录）。
2. 用户选定后下发：

```bash
python <skill_dir>/scripts/apply_spec.py --spec <skill_dir>/presets/daily-natural.json
```

3. 可调参数（用户口头提出时）：
   - 整体妆感浓淡：`python <skill_dir>/scripts/apply_spec.py --spec <...> --intensity 0.6`（0~1，覆盖 spec 里的 intensity）
   - 只试单件（如只换口红不卸其他）：`--only lipstick`（region 名，多个逗号分隔）
   - 解析出的新妆容免手动烘焙：加 `--bake`（自动烘资产再下发，大资产自动走 HTTP 侧车）
   - 多个试妆 App 在线时定向下发：`--to <client_id>`（client_id 见 status）
   - 卸妆还原：`python <skill_dir>/scripts/apply_spec.py --clear`
4. 完成后告知用户 App 里可拖动强度滑杆微调；确认用户满意即结束。

### 自然语言改妆（工作流一变体）

用户已上妆后说"口红换成番茄色""眼影淡一点"这类一句话改妆需求时，不必重选整个妆容：
agent 直接改 spec 中对应 layer 的字段，再配合 `--only` 单层下发（其余部位保持不变）。

以"口红换番茄色"为例（把当前 spec 复制到工作目录后，将 lipstick layer 的
`color_stops` 两个 hex 改为番茄色系，如 `#D93A2B` / `#B32B1F`，`finish` 可改 `gloss`）：

```bash
# 1. 复制当前 spec 到工作目录并编辑 lipstick layer 的颜色字段
cp <skill_dir>/presets/daily-natural.json <工作目录>/makeup_spec.json
#    （编辑 spec：layers 里 region == "lipstick" 的层，改 color_stops 的 hex）
# 2. 只下发改过的口红层，脸上其余妆容不动
python <skill_dir>/scripts/apply_spec.py --spec <工作目录>/makeup_spec.json --only lipstick --bake
```

常见改妆映射（字段见 `references/schema.md`）：

| 用户说 | 改哪个字段 |
|---|---|
| 换色号 / 换颜色 | 对应 layer 的 `color_stops[*].hex`（常改动两侧渐变两档） |
| 浓一点 / 淡一点 | layer 的 `opacity`，或整体用 `--intensity` |
| 哑光 / 水光 | layer 的 `finish`（`matte` / `satin` / `gloss` / `dewy`） |
| 眼影范围大一点 | `shape.spread` / `shape.height` |
| 眼线加个小尾巴 | `shape.wing`（0~1） |

注意：左右各一个 layer 的 region（eyeshadow/blush）要两侧同时改，保持对称。

## 工作流二：解析妆容素材（照片/视频 → 同款妆容）

前置：Bridge 可选（解析不需要，试妆需要）；VLM 必须配置。

1. 拿到用户的素材文件路径（图片或视频）。
2. 运行解析：

```bash
python <skill_dir>/scripts/parse_look.py --input <素材路径> --out <工作目录>/parsed
```

   产出：`makeup_spec.json`（可直接试妆）+ `analysis.md`（人话版妆面报告）
   + `preview.jpg`（无妆 | 同款妆 并排效果图，与 App 同一套渲染规则）。
3. 把 `preview.jpg` 给用户看，并用 `analysis.md` 复述解析结果（妆面风格、各部位色号与画法），
   确认符合预期（也可以随时用 `render_look.py --spec <...> --out <...> --env warm` 重出效果图）。
4. 用户认可后按工作流一试妆：`apply_spec.py --spec <工作目录>/parsed/makeup_spec.json`。

注意事项：
- 视频会均匀抽帧、自动选最清晰的几帧分析；解析是多次 VLM 调用，约需 30~60 秒，提前告知用户。
- 若用户指定"只要眼妆"等，加 `--focus eyeshadow,eyeliner`（region 名见 `references/schema.md`）。

## 工作流三：实时化妆协助

前置：Bridge 运行中 + App 已连接（App 会回传摄像头帧）+ VLM 已配置。
画像流程（推荐）：`--station` 进妆容台 + `--avatar-look` 用画像渲染作参考图。

```bash
python <skill_dir>/scripts/live_coach.py --spec <目标妆容spec> --interval 5
```

- `--interval`：采样间隔秒数，默认 5。
- 启动时自动把目标妆容渲染成**参考图**（`preview_render`，与 App 同管线），随每轮与摄像头帧
  一起发给 VLM 做视觉对比（`--no-reference` 关闭）。
- 它会持续运行（Ctrl+C 或 `--duration 600` 限时结束）：每轮取帧 → VLM 对比 → 生成下一步指导 →
  推送到 App（文字+步骤进度条+语音提示），同时在控制台输出。
- 指导按 spec 的 `steps` 顺序推进；步骤判定用最近 3 轮滑动窗投票（≥2 个 done 才推进，单帧误判
  不跳步）；环境光过暗/过曝会顺带提醒。
- 结束时输出 `session_report.md`（各步骤完成情况 + VLM 按部位评分与建议），并向用户口头总结。
- 无 VLM key 的联调/演示：加 `--test`（离线剧本，走真实 Bridge 取帧与推送，不调模型）。

隐私：只有开启实时指导时才会把采样帧发给 VLM 服务商；试妆渲染全程本地。主动告知用户这一点。

## Bridge 通信协议

Agent 脚本与 App 都作为 WebSocket 客户端连到 bridge_server，消息格式与角色定义见
`references/bridge-protocol.md`。一般不需要手写消息——上面的脚本已封装；
仅当需要扩展 App 端功能（如自定义 UI 指令）时才阅读协议文档。

## 妆容规格 makeup_spec.json

格式定义见 `references/schema.md`。要点：`layers[]` 每层一个部位（region）、
颜色用 `color_stops` 渐变、`finish` 控制哑光/缎光/水光、`render.type` 决定
mesh 蒙版渲染还是 3DGS 溅射渲染（唇部体积感、高光用 splat）。

## 故障排查

| 现象 | 处理 |
|---|---|
| `apply_spec` 报 "no app connected" | App 未启动或端口不对，见 `references/app-setup.md`；确认 bridge_server 在运行 |
| parse_look 卡在 API 调用报 401/403 | `MAKEUP_VLM_API_KEY` 未设或无效 |
| App 渲染的妆容位置偏移 | 人脸追踪未就绪（App 首帧对齐需 1~2 秒），或用户离摄像头太远/太暗；重发一次 spec |
| live_coach 提醒千篇一律 | 检查 spec 的 `steps` 是否描述具体；间隔太短导致帧间无变化，调大 `--interval` |

## 文件索引

- `scripts/` 可执行工具（apply_spec.py 封装 bridge 下发；preview_render.py 为渲染内核、
  render_look.py 为效果图 CLI；其余见上文）
- `presets/` 内置妆容
- `references/schema.md` 妆容规格格式
- `references/bridge-protocol.md` WebSocket 协议
- `references/app-setup.md` 试妆 App 安装启动
- `references/landmark-regions.json` 面部区域 ↔ 人脸关键点索引映射
- `references/prompt-templates/` VLM 提示词模板（解析/指导）
