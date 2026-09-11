# makeup_spec.json 格式定义（spec_version 1.0）

妆容规格是整个系统的核心数据契约：预设、解析产出、App 渲染、实时指导都用它。
所有颜色用 `#RRGGBB` 十六进制；UV 坐标/尺寸均为 0~1 归一化。

## 顶层字段

```json
{
  "spec_version": "1.0",
  "name": "daily-natural",
  "description": "清透日常妆",
  "intensity": 0.8,
  "layers": [ ... ],
  "steps": [ ... ],
  "source": { "type": "preset", "origin": "built-in", "created_at": "2026-09-06" }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `spec_version` | string | 固定 `"1.0"` |
| `name` | string | 妆容唯一短名（文件名同名） |
| `description` | string | 一句话人话描述，用于向用户展示 |
| `intensity` | 0.0~1.0 | 整体妆感浓度，App 渲染与指导判定都受它缩放 |
| `layers` | array | 妆容图层，每层一个部位 |
| `steps` | array | 化妆步骤（实时指导的推进顺序），可缺省（缺省时按 layers 默认顺序） |
| `source` | object | 来源元信息：`type` = `preset` \| `parsed` \| `manual` |

## layers[] 字段

```json
{
  "id": "eyeshadow-left",
  "region": "eyeshadow",
  "side": "left",
  "enabled": true,
  "opacity": 0.85,
  "finish": "satin",
  "color_stops": [
    { "at": 0.0, "hex": "#E7B9A0" },
    { "at": 1.0, "hex": "#9C6B52" }
  ],
  "shape": { "spread": 0.55, "height": 0.4, "angle_deg": 15 },
  "texture_strength": 0.3,
  "render": { "type": "mesh" }
}
```

| 字段 | 说明 |
|---|---|
| `id` | 图层唯一 id（region+side 组合） |
| `region` | 部位名，见下表 |
| `side` | `both` \| `left` \| `right`（**用户自身视角**的左右） |
| `enabled` | false 时保留但不渲染（单品试妆用） |
| `opacity` | 该层基础不透明度 0~1，实际 = opacity × intensity |
| `finish` | `matte`（哑光）\| `satin`（缎光）\| `dewy`（水光）\| `gloss`（镜面唇釉） |
| `color_stops` | 渐变色标。`at`=0 是靠近边缘/起始色，`at`=1 是中心/加深色；单色也写成一段（at 0 与 1 同色） |
| `shape` | 区域形态参数（含义随 region 不同，见下表），可缺省用默认 |
| `texture_strength` | 粉感/肌理强度：0 纯色平滑，1 强颗粒肌理（哑光粉质建议 0.3~0.6，唇釉 0~0.1） |
| `render.type` | `mesh`：FLAME/MediaPipe 网格上的蒙版材质层；`splat`：高斯溅射体积层（唇釉、高光、卧蚕等需要"凸起来"的效果） |
| `render.splat` | splat 层参数：`thickness`（凸起厚度，米，典型 0.001~0.002）、`density`（0~1 高斯密度） |

## region 一览与 shape 语义

| region | 用途 | shape 键（默认值） |
|---|---|---|
| `foundation` | 粉底/底妆 | 无（全脸），`coverage`: `light`/`medium`/`full` 放 shape 里 |
| `concealer` | 遮瑕 | `under_eye`(默认 true) |
| `blush` | 腮红 | `center_uv`[x,y]（颊面内位置，默认 0.58,0.62）、`radius`(0.12)、`angle_deg`(15，斜向上)、`falloff`(0.65，越大边缘越柔) |
| `eyeshadow` | 眼影 | `spread`(0.5，眼窝晕染范围)、`height`(0.35，上睑高度占比)、`angle_deg`(0，上扬/下垂)、`lower_lid`(0.0，下睑延伸 0~1) |
| `eyeliner` | 眼线 | `thickness`(0.25)、`wing`(0.3，眼尾拉长上扬 0~1)、`lower`(0.0) |
| `eyebrow` | 眉 | `thickness`(0.4)、`arch`(0.5，眉峰强度)、`tint`(0.8) |
| `lipstick` | 唇 | `overline`(0.0，唇峰上扩)、`blur`(0.15，边缘晕染，咬唇妆调大)、`gradation`(0.0，1=咬唇渐变) |
| `highlight` | 高光 | `areas`: `["cheek","nose","cupid"]` 子集 |
| `contour` | 修容 | `strength`(0.5)、`jaw`(true)、`nose`(false) |
| `lashes` | 睫毛 | `volume`(0.5)、`length`(0.5)（mesh 渲染为阴影增强；splat 渲染为根部溅射） |

## steps[] 字段（实时指导顺序）

```json
{ "order": 1, "area": "base", "regions": ["foundation","concealer"], "instruction": "先上粉底，由面中向外拍开，遮瑕最后点涂" }
```

`area` 是人话分组名（base/eyebrow/eye/blush/lip），指导按 `order` 推进，
当前步骤涉及的所有 region 都达到目标后进入下一步。

## 解析产出的补充字段（工作流二）

parse_look.py 输出的 spec 额外带：
- `source.confidence`：0~1，VLM 对整体解析的置信度
- `source.frame_count`：参与分析的帧数
- 层内 `notes`：VLM 给的画法描述（如"眼影沿双眼皮褶向上晕染 5mm"），live_coach 会引用
