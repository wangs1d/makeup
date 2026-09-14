# P5 画像架构方案（3DGS Avatar Try-on & Station）

> 2026-09-14 立项。产品流程转向：**用户上传 3DGS 画像 → 选妆在画像上预览 → 确认后进入
> 妆容台（辅助化妆）**。妆容不再附着在摄像头里的真脸上（原 P1 贴脸渲染降级为 legacy 开关），
> 真实感由"画像 + 真实高斯溅射"承载，指导链路（VLM 对比真脸帧）不变。

## 一、新流程与组件

```
用户 --上传 3DGS ply--> avatar_session register --HTTP侧车--> App(GPU加载渲染裸画像)
用户 --选妆容--> avatar_session preview
    ├─ Python: 画像语义标注(一次性缓存) + 妆容编译器 → tint.bin + add_splats.json
    ├─ Python: avatar_render 软件光栅化 → preview.jpg（裸妆|目标妆 并排，用户确认用）
    └─ Bridge: avatar_preview → App 画像上实时预览
用户 --确认--> avatar_session confirm → App 锁定妆效
     --进妆容台--> avatar_session station → enter_station
        App：摄像头画面保留（供 VLM 指导取帧），画像+妆容移到侧栏作目标参照，
             coaching 字幕/进度条/TTS 全开；真脸零附着渲染
```

## 二、模块与数据格式

### Python（makeup-skill/scripts/）

| 模块 | 职责 |
|---|---|
| `avatar_io.py` | 标准 3DGS PLY 解析/写出（x y z f_dc opacity scale rot；SH 高阶忽略），颜色 SH DC→sRGB，透明度 sigmoid，尺度 exp，四元数归一；画像归一化（居中、脸高=1 世界单位） |
| `avatar_semantics.py` | 给每个 Gaussian 标 region-ID（uint8）。三种模式：`landmarks`（正脸渲染→MediaPipe 468 点→z-buffer 反投影出 3D 锚点）、`manual`（锚点 JSON，测试与人工校正）、`cache`（复用已标注结果）。区域半径剖面表 REGION_PROFILES 控制扩散（眼影向上扩、腮红大半径等） |
| `makeup_compiler.py` | spec layers + region-ID → **tint.bin**（每 Gaussian rgba：rgb=目标色，a=覆盖权重，浓度运行时乘）+ **add_splats.json**（溅射效果层：唇釉/闪片=贴表面的附加高斯，位置/法线/切轴/σ/色）+ manifest。`--only` 只重编译指定 region |
| `avatar_render.py` | 软件 EWA 光栅化（3D 协方差→屏幕 2D 协方差→深度排序 alpha 混合），环境色温/背景与 preview_render 同预设；裸妆/目标妆并排 preview.jpg、转台序列 |
| `avatar_session.py` | CLI：`register/preview/confirm/station/leave/status`。资产走 bridge HTTP 侧车；语义与编译产物缓存于 `out/avatars/<id>/` |

### 二进制格式（Python 写、C# 读，不引 NPZ 依赖）

- `MKSEM1` semantics.bin：`magic(4) ver(u16) rsv(u16) N(u32) regionId(u8×N)` — 仅 Python 侧编译用（App 不需要语义）。
- `MKMKP1` tint.bin：`magic(4) ver(u16) flags(u16) N(u32) rgba(f32×4×N)` — App 混色用（a=0 表示该点无妆容）。
- `add_splats.json`：`[{pos,normal,axis,sigma[2],color,alpha}]`（画像局部空间，米，脸高=1）。

### Bridge v1.2（新消息，复用既有路由/ack/侧车，server 无需改代码）

| type | 方向 | 载荷要点 |
|---|---|---|
| `avatar_register` | agent→app | `avatar_id, assets_url{avatar.ply, meta.json}` |
| `avatar_preview` | agent→app | `avatar_id, look{name,intensity}, assets_url{tint.bin, add_splats.json}` |
| `avatar_confirm` | agent→app | `avatar_id` |
| `enter_station` | agent→app | `avatar_id, look` — App 进入妆容台布局 |
| `leave_station` | agent→app | `{}` |
| `station_state` | app→agent | `{state: registered/preview/confirmed/station}` |

新 caps：`avatar`（可渲染画像）、`station`（支持妆容台布局）。

### Unity（Assets/）

| 文件 | 职责 |
|---|---|
| `GaussianAvatarParser.cs` | PLY 二进制解析 + MKMKP1/MKSEM1 解析（结构化偏移读取） |
| `GaussianAvatarRenderer.cs` | ComputeBuffer 化位置/协方差/颜色/妆容 tint；CPU 视深排序（相机动超阈值或每 N 帧重排，100k 级可 30fps）；`Graphics.DrawProceduralNow` 六顶点展开；shader 内 EWA 2D 协方差 |
| `GaussianAvatarSplat.shader` | 3DGS 高斯原语渲染：2D 协方差特征轴展开 quad、premultiplied 混合、环境色温、tint 混妆（`_MakeupIntensity`） |
| `AvatarSplatRenderer.cs` | add_splats 静态附加溅射（画像局部空间、视深排序、复用 GaussianSplat.shader 实例化） |
| `AvatarStationFlow.cs` | 状态机 `idle→registered→preview→confirmed→station`；消息处理、侧栏布局、coaching 开关 |
| `MakeupAppMain.cs`（改） | 默认 **不** 再创建 FaceMeshDeformer/MakeupLayerRenderer/SplatLayerRenderer 真脸附妆链路（`legacyFaceMakeup=false`）；接线画像渲染与妆容台流 |

## 三、画像获取路径（ references/avatar-setup.md 详述）

1. **单照片（推荐入口）**：LAM 类大模型（SIGGRAPH 2025）一次前向 → FLAME 绑定 Gaussian 头像，导出标准 ply 即可上传。
2. **引导扫描（高质量）**：用户正脸+±45°+俯仰录 15~30s → GaussianAvatars/MonoGaussianAvatar 类离线拟合 → 导出 ply。
3. **要求**：标准 3DGS 属性集；建议先裁剪到头肩、去背景飞点；点数 ≤ 30 万（App 端自动抽稀旋钮）。

## 四、验收标准

- `pytest tests/` 全绿（新增画像管线离线单测：synthetic 画像 + manual 锚点，不依赖 mediapipe 模型下载）。
- `python avatar_session.py preview` 能对合成画像产出 preview.jpg（裸妆|妆后并排），`tint.bin` 仅覆盖对应 region 的 Gaussian。
- check_csharp 新增断言：画像渲染/解析/状态机接线存在、真脸附妆默认关闭、新 shader 存在。
- 文档同步：SKILL.md 新工作流〇、bridge-protocol v1.2、README/assembly/OPTIMIZATION-PLAN 更新。

## 五、边界与后续（不在本轮）

- 表情驱动（FLAME 系数 rig → 画像 Gaussian 蒙皮）为 P6；本轮画像为静态渲染 + 侧栏参照。
- 移动端（ARKit blendshapes 驱动）延后；GPU 排序（bitonic compute）在点数 >30 万时再引入。
- live_coach 参考图默认接画像编译产物 preview（`--avatar-look`），无画像时回退 face_render。
