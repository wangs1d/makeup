# 3DGS 画像获取与语义标注（avatar-setup.md）

`avatar_session` 消费标准 3D Gaussian Splatting 画像（单文件 `.ply`）。本文说明
怎么得到合格画像、格式要求、以及语义标注（告诉编译器"哪里是嘴唇/眼睑"）的三种方式。

## 一、画像获取

任选其一，产物都是标准 3DGS ply：

| 路径 | 成本 | 说明 |
|---|---|---|
| **单照片生成（推荐入口）** | 一张正脸照，秒~分钟级 | LAM 类大模型（SIGGRAPH 2025）一次前向输出 FLAME 绑定的可动画 Gaussian 头像；导出 ply 直接用 |
| **引导扫描（高质量）** | 15~30s 头部扫描视频 + 离线拟合几分钟 | GaussianAvatars / MonoGaussianAvatar 类管线：视频 → FLAME 拟合 → 逐帧高斯优化；正脸+左右 ±45°+俯仰各一段 |
| 现成商业扫描导出 | — | Postshot / Luma / Scanverse 等 3DGS 导出亦可，需含下表属性 |

**格式要求**（`avatar_io.load_avatar` 校验）：

- `binary_little_endian` PLY，vertex 元素含：`x y z`、`f_dc_0..2`（SH 直流）、
  `opacity`、`scale_0..2`、`rot_0..3`（SH 高阶 `f_rest_*` 与法线会被忽略）；
- 建议先裁头肩、去背景飞点；点数上限 30 万（`register --max-gaussians` 默认抽稀到 12 万，
  该数值决定 `tint.bin` 行数，App 端与编译端必须一致，不要在 App 侧再抽稀）。

**隐私**：画像 ply 与编译产物全程本地（bridge 资产侧车在本机 8768 端口）；画像属于敏感
生物特征数据，请用户知情后再采集，不要上传到第三方。

## 二、语义标注（region 锚点）

画像是无语义的高斯点云，编译器需要知道哪些高斯属于哪个妆容部位。三种模式（`avatar_session
preview` 自动按此优先级处理）：

### 1. 缓存复用（默认）

首次标注成功后存 `<work-dir>/semantics.bin`（MKSEM1：per-Gaussian regionId u8 + 置信度），
之后自动复用。换妆容不需要重标。

### 2. landmarks 自动模式（默认首次）

正脸渲染一帧 → MediaPipe FaceMesh 检 468 点 → 沿渲染相机视线用深度图反投影回画像 3D 锚点
→ 按 `REGION_PROFILES` 半径剖面（与 RegionMaskBaker 的 UV 扩散量同源标定）做相对距离归属。
需要 `mediapipe` 已安装且画像正脸清晰可检。失败时 CLI 会明确报错并提示改用手动锚点。

### 3. manual 手动锚点（兜底/精修）

`--anchors anchors.json`，归一化空间（脸高=1、xy 居中、脸朝 +Z 凸出）：

```json
{
  "lipstick":  [[0.00, -0.30, 0.20], [0.06, -0.31, 0.20], [-0.06, -0.31, 0.20]],
  "eyeshadow": [[0.16, 0.16, 0.17], [-0.16, 0.16, 0.17]],
  "eyebrow":   [[0.16, 0.24, 0.15], [-0.16, 0.24, 0.15]],
  "blush":     [[0.24, -0.06, 0.18], [-0.24, -0.06, 0.18]],
  "foundation":[[0.00, 0.00, 0.25], [0.20, 0.30, 0.16], [-0.20, 0.30, 0.16]],
  "highlight": [[0.00, 0.05, 0.247], [0.00, -0.22, 0.19]],
  "contour":   [[0.00, 0.42, 0.09], [0.30, -0.34, 0.10], [-0.30, -0.34, 0.10]]
}
```

支持的 region 与半径剖面（脸高单位，可在 `avatar_semantics.REGION_PROFILES` 调）：
foundation 0.30 / concealer 0.075 / contour 0.085 / eyebrow 0.045 / eyeshadow 0.062 /
eyeliner 0.022 / blush 0.16 / highlight 0.055 / lipstick 0.052。

## 三、常用命令速查

```bash
python <skill_dir>/scripts/avatar_session.py register --ply 我的画像.ply --name 我的画像
python <skill_dir>/scripts/avatar_session.py preview  --avatar 我的画像 --spec presets/date-rose.json
python <skill_dir>/scripts/avatar_session.py preview  --avatar 我的画像 --spec presets/daily-natural.json --only lipstick   # 单品改妆
python <skill_dir>/scripts/avatar_session.py confirm  --avatar 我的画像
python <skill_dir>/scripts/avatar_session.py station  --avatar 我的画像
python <skill_dir>/scripts/live_coach.py --spec presets/date-rose.json \
    --avatar-look out/avatars/我的画像/compiled/date-rose --station   # 妆容台陪练
```

预览产物（`out/avatars/<id>/compiled/<look>/`）：`preview.jpg`（裸妆|妆后并排，给用户确认）、
`reference.jpg`（妆后单帧，live_coach 参考图）、`turntable/`（±24° 转台）、`tint.bin` +
`add_splats.json`（下发 App 的妆容 buffer）。
