# 写实化管线（R 升级）——真·3DGS 底模 + UV 妆容 + PBR 材质

> 2026-09 彻底重构。目标：妆容试戴达到照片级真实。本文记录根因、架构与用法。
>
> **交付形态变更（2026-09-18）：离线渲染，无 Unity**。产品流程收敛为
> "后台把用户画像资产上妆 → gsplat 高保真重渲染 → 出图/视频给用户"：
> `py -m makeupstudio.face3dgs render -p <工程> [--spec 妆容.json] [--sfm sparse目录]`
> → renders/（still_front|left|right.png + turntable.mp4 + compare_*.png）。
> 渲染器 `appearance/offline_render.py`（与训练同款 gsplat 光栅化器，SSAA 超采样、
> 白底、ping-pong 环绕；取景中心 = 采集相机光线最小二乘汇聚点）。桌面端 preview.png
> 与 pipeline 对比图同步换用该渲染器（无 CUDA 时回退 numpy 诊断渲染）。
>
> **低清源的超分重训路径（2026-09-18，`preview/run_sr1080.py` + `tools/upscale_frames.py`）**：
> 源视频只有 480² 且无重录条件时，Real-ESRGAN x4（自实现 RRDBNet 推理，零 basicsr
> 依赖；权重走 hf-mirror）把 COLMAP 已配准帧放大到 1920²，**复用现有 SfM 位姿**
> （几何不变，K 按图像尺寸自动缩放；训练羽化随分辨率 ×4），重训后再以 1080 离线渲染。
> 实测结论（诚实）：SR 资产结构更完整、细节更多，但 **SR 同时放大高光/噪声并烘进
> splat**（额头/眼部白色大块），近景取景下不可商用——它是"等重录期间的过渡方案"，
> 不是 1080p 的正解；真正的画质台阶必须 ≥1080p 重录（采集规范见上文）。
> 离线渲染取景三坑（已修，见 `offline_render.orbit_from_poses`）：yaw 跨 ±180 边界
> 导致扫掠整圈（定妆照拍到后脑勺）；splat 中位数被头发拉偏（取景中心用**相机光线
> 最小二乘汇聚点**）；三角化 3D 地标整体不可靠（不能当取景中心）。
>
> **离线渲染的关键教训：SH 视角分布**。SH 高阶系数只在采集相机覆盖的视线方向上
> 被训练归零——合成轨道一旦离开该分布，21 万 splat 的 SH 残差同时外推，渲染成
> 彩虹碎裂。因此环绕相机两条策略：有 SfM 位姿 → 在真实相机方位角范围内插值
> （`orbit_from_poses`，SH 全开，半径/焦距取中位数、按脸高占画面 72% 收放距离，
> 视线方向不变所以 SH 安全）；无位姿 → 窄幅环绕 + DC-only 渲染（`use_sh=False`，
> 视角无关颜色任意角度稳定）。Unity 端不受此限（实时视角都在用户可控范围内）。
>
> **R+ 渲染质量升级（2026-09-18）**：在 R 基础上补齐真实感最后一块——
> ① 训练端 SH degree 2（视角相关外观：油光/高光随视角流动，不再是"贴纸"）
>    + `rasterize_mode="antialiased"`（Mip-Splatting 式 2D 滤波，拉近拉远不呼吸）；
> ② 导出端法线修正：min(scale) 薄轴（`appearance/normals.axis_normals`，取代
>    "薄轴恒为 Z"的错误假设）+ kNN 邻域平滑（`smooth_normals`，高光连续）；
> ③ 主光方向估计（`train_base.estimate_light_dir`）→ `light.bin`(MKLT1) sidecar，
>    Unity 高光与烘焙光照同向；④ PBR 语义改为**纯叠加**（relight=0 默认）：
>    烘焙颜色已含真实光照，合成 wrap diffuse 重打光（旧 `0.30+0.70*fac`）
>    是双重光照 fake 感主因，现由 `_Relight` 滑杆显式开启（程序化模板资产用）；
> ⑤ Unity shader 重构：PBR 从顶点级挪到**片元级**（高光不再是 per-splat 色盘）、
>    线性空间数学后转显示空间混合、颜色输出走 TEXCOORD1（COLOR0 语义钳制 HDR
>    高光）、quad 2σ→2.5σ（消截断环）、SH degree2 逐 splat 求值；⑥ GPU bitonic
>    深度排序（`GaussianAvatarSort.compute`，每帧重排无 pop，CPU 排序保留兜底）；
> ⑦ 妆容微观纹理改用底模 albedo 的 UV 高通（真实唇纹/毛孔，随机噪声仅兜底）；
> ⑧ 会话接通：`avatar_session.register` 上传 material.bin/light.bin，
>    `AvatarStationFlow` 注册时拉取（此前 material.bin 从未到过 Unity）；
>    AvatarData 携带 sh_rest 贯穿归一化/抽稀/保存。SH 高阶在妆容烘焙时保持
>    底模值（Lab 迁移只改 DC），唇区视角相关残差可忽略。
>
> **F 升级（2026-09-19）——妆容还原度全链路**。还原度的四个缺口一次补齐：
> ① **妆感材质进交付渲染**：Unity 退役后 `material.bin` 一直无消费端，
>    rough/coat/sss/sheen 在交付的 stills/turntable 里整体不可见。现在
>    `offline_render` 在主色渲染外光栅化 3 个 AOV（法线/rough-coat-sss/
>    sheen-深度，与训练同款光栅化器），像素级按 `pbr.shade_points` 同式
>    叠加薄层高光（清漆 Fresnel + 掠射绒光 + sss 背光红移），`build_shade`
>    优先吃 in-memory material，否则读 sidecar（MKMAT1/MKLT1 读取器补齐，
>    头部 16B）；denoise 之前合成，高光不被滤波吃掉。
> ② **颜色端闭环（`appearance/calibrate.py`）**：参考妆照 → MediaPipe 地标
>    → 图像空间区域蒙版（多边形/椭圆/折线直出，不依赖 3D 配准）→ 逐区域
>    Lab 统计 → spec 色带/浓度替换（`calibrate_spec`，CLI `--reference`）；
>    渲染帧妆区 vs spec 目标色的逐区域 **CIEDE2000**（Sharma 2005 基准值
>    回归测试）进 `report.json` 的 `makeup_delta_e`——还原度第一次可度量。
> ③ **形状级还原**：`optimize.py`（Stable-Makeup guidance 图像空间联合求解，
>    修复 render_now 用错 c2w 键的存量 bug）与 `bake_guidance`（多视角
>    guidance → UV albedo 聚合）正式接进 `run_photoreal`：spec 带
>    `guidance.reference` 且环境就绪时自动走"素颜多视角渲染 → guidance →
>    聚合 + 精修"，环境缺失优雅跳过；眼线/睫毛/眉新增 `landmark_band_3d`
>    观测锚定（三角化真实地标的 Catmull-Rom 折线带 + wing/thickness 参数，
>    canonical 模板 ~2% 错位在这些 2-5mm 区域同样不可接受），带只改权重、
>    颜色仍取 UV 场，失败回退 UV 模板。
> ④ **烘焙质量**：Lab 迁移改 float 全程（uint8 往返的色带消除）+ 极坐标
>    pigment-safe 色度迁移——色相旋转权重封顶 1（旧向量插值系数>1 会穿过
>    a-b 零点把红唇翻成绿调，测试揪出的真 bug）、chroma 保底不低于素颜
>    （透色染料语义）、kL 按底色/妆色明度差自适应收缩（跨肤色还原一致，
>    `lab_adapt`）；`bind_uv.near` 首次被消费——对全部妆权重（含唇兜底与
>    3D 锚定带）做高斯软门控，收掉鼻翼/眼角/轮廓的 UV 归属误差晕；唇妆
>    UV 兜底路径补上真实目标场（此前 lip3d 缺席时目标是全零 = 涂黑）；
>    珠光 sheen 从均匀标量改为高频阈值噪声闪点图 + 新增 glitter finish；
>    验收对比图 460→760 并带妆感合成（460 看不见唇纹/粉感）。
> ⑤ **妆容壳层 = 真实高斯球（2026-09-19，`build_makeup_layer` +
>    `merge_makeup_layer`）**：妆不再只是"底模 splat 原地改色"，而是
>    **几何上存在的一层高斯**——每个 w>门限的底模 splat 生成一个重叠新高斯：
>    xyz/rot 与底模完全一致（重叠在对应位置，合并后有显式零误差自检）、
>    scale 同足迹、颜色 = 完整 Lab 迁移目标（kL/chroma 不乘 w，"部分迁移"
>    由 alpha 合成完成——物理层即薄层化妆品）、opacity = w×底模 opacity
>    （壳层堆叠比例与底模一致，妆强≈w）、sh_rest 置零（妆色视角无关，不被
>    底模素颜 SH 残差调制）、材质 = finish 全强度（唇釉清漆/sss/珠光直接
>    挂壳层）。底模保持素颜 → 对比图/identity 锁/guidance 素颜参考全部用
>    同一底模；madeup.ply = 底模+壳层合并导出。`_assignment` 统一指派
>    （权重/目标色/材质），原位重染色（as_layer=False，诊断用）与壳层两条
>    路径"涂在哪/涂什么色"永远一致；optimize 的 identity 锁改以素颜底模为
>    参考（bare_cloud），颜色残差直接修在壳层 splat 上。
>
> **Q 升级（2026-09-21）——底模质量 + 还原度闭环**。对照交付图的四大缺陷
> （嘴部牙齿烤进底模 / 眼区彩色碎斑 / 训练视图饥饿 / 还原度只有代码没有数字）：
> ① **器官一致性损失屏蔽**（`train_base.build_views`）：嘴内（内唇环多边形，
>   权重 0.12）与眼球（眼开多边形收缩 0.55，权重 0.35）从全权重损失里降权——
>   露齿帧的牙齿不再被光度优化烤进 canonical 脸（唇区最大视觉毁容的根因），
>   视线漂移的虹膜收敛到均值外观且梯度不足以触发 densify 异常生长；
>   退化多边形（闭嘴零面积）自动不屏蔽。
> ② **gaze 表情特征**（`frames.expression_features`，4 维→6 维）：虹膜中心
>   相对眼角中点的偏移（双眼平均，眼距归一）进 MAD 聚类——眼区外观跨帧一致
>   才有干净的眼部 splat；虹膜缺失时特征回 0（该维度 MAD≈0 不影响聚类）。
> ③ **训练视图扩容**：`eval_holdout` 6→4（7 个训练视角养不活 densify）；
>   `select_frames` 新增 `target_frames=24`——min_frames 达标后继续按 ×1.5
>   放宽 tau（至 tau_cap=2.0 MAD）直到簇达标，表情语义由 6 维特征把守。
> ④ **SH 高阶 L2 正则**（`TrainConfig.sh_weight=1e-4`）：≤15 个训练视角时
>   21 个高阶通道必然在无监督视线方向过拟合，合成视角一外推就碎成彩色斑。
> ⑤ **底妆压油光**（`makeup_uv.powder_w` + `POWDER_SH_GAIN=0.6`）：粉类
>   （foundation/concealer）覆盖处的底模 SH 残差按 (1−0.6·powder) 衰减——
>   粉把反光压哑（原位路径直接衰减；壳层路径经 `_sh_att` 在 merge 时衰减
>   底模侧）。
> ⑥ **壳层边缘补偿**：低权重边缘 splat 沿两个面内轴扩张
>   （≤ +40%，薄轴不动）——低 w 壳层 opacity 低、稀疏采样区斑驳露底的问题。
> ⑦ **形状级还原零依赖替代**（`appearance/refshape.py`）：参考妆照与用户
>   参考帧的 468 地标做 RANSAC 相似变换 → 参考图区域蒙版搬到用户帧 →
>   相机投影采样成逐 splat 形状权重（`shape3d`）。参考有妆处取并集、参考
>   明确无妆处模板权重压制到 0.35——眼影晕染范围/眼线翼形第一次可以跟
>   参考妆走，不再依赖 Stable-Makeup 全家桶。spec 带
>   `calibration.reference`/`guidance.reference` 时自动启用。
> ⑧ **还原度闭环真正接通**：`_delta_e_report` 富化为逐区域 {ΔE00, 渲染
>   Lab, 目标 Lab}；超阈值（均值>10 或单区>12）自动按缺妆/过妆方向调整
>   spec opacity 重烘（≤2 轮取最优，`_adjust_spec_from_delta`：彩妆看
>   chroma 比例、底妆看明度方向）；report.json 新增 `makeup_delta_e`
>   （富化版）/ `makeup_bench`（lip_delta_e + identity_shift + 唇线锐度
>   made/bare 比）/ `vlm_makeup_score`（可选，`MAKEUP_VLM_API_KEY` 存在时
>   问 VLM "妆感自然度"）/ `auto_calibrate` 迭代日志 / `spec_final_opacity`。
>   训练端 report 新增可选 `lpips_masked`（装了 lpips 包才算，蒙版 bbox
>   裁剪去背景差异）。
> ⑨ **2D/3D 一致性交叉检查**：`run_photoreal --look2d <2D人台渲染图>` 把
>   同一 spec 在 2D 预览渲染上的逐区域 ΔE00 进 report；`--preview-2d` 在
>   EleGANt 就绪时产出参考帧的 2D 迁移预览（预期管理用，门控增强）。
> ⑩ **评测补位**：`lpips` 缺包不阻断（返回 null）；VLM 打分不稳定，只用于
>   回归趋势，不进验收硬门。
>
> **Q2 高还原度迭代（2026-09-22）——"删除式修复"纠偏为"重建式修复"**。
> Q 版把嘴内/眼球从损失里降权，结果是牙呲没了但**真牙齿和眼睛的细节也被
> 一起饿死**（用户可见回退：模糊替代了错误）。Q2 的结论：器官问题要靠
> **让 canonical 表情干净**解决，不是靠蒙版挖洞：
> ① **闭嘴 canonical**（`frames.closed_lips`）：嘴开度带取"最闭的 1/4 分位"
>    （不足回退中位数带）——闭唇是单模态表面，唇纹清晰、天然无牙齿；
>    涂口红正需要的姿态。实测参考帧从露齿 grimace 换成近闭唇帧。
> ② **眼球恢复全监督**（`eye_weight=1.0`），视线一致性交给 gaze 特征
>    （`gaze_weight=2` 距离加权）；簇超员时**按表情+视线距离择优**取代
>    时间均匀抽帧。
> ③ **密度解锁**：`max_gs` 400k→900k（旧上限下 30k iter 在 15k 就触顶，
>    densify 后半程饿死）+ `grow_grad2d` 2e-4→1.3e-4（高频区更早分裂）。
>    实测 79 万 splats，唇面/鼻翼结构明显改善。
> ④ **逐视图曝光补偿**（`exposure_comp`）：每视图 3 维 log-gain 联合优化，
>    自拍视频自动曝光漂移从几何层问题挪到光照层（实测增益幅度 ~1.6dB）。
> ⑤ **少一点平滑**：`mip3d_gamma` 0.3→0.25、denoise sigma 60/0.45→40/0.28
>    ——花斑靠曝光补偿和密度压，锐度还给细节。
> ⑥ 留出帧数自适应（小视图池时验证 ≤20%）+ 聚类兜底（嘴开度带是语义门，
>    MAD 聚类绝不把训练视图饿到个位数——d8 首跑曾因此只剩 3 个训练视图）。
> **诚实现状**：480p 单目 + ~12 视图下，唇/鼻/轮廓达到可用，**眼区碎斑
> 仍在**（虹膜/眼睑高频 + 视角覆盖不足，是数据墙不是算法墙）——真正的
> 台阶是 ≥1080p 重录（采集规范见上）或 FLAME/LAM 先验头像（路线图 P6/L3）。
>
> **F+ 质感补强（2026-09-19）**：① **薄层厚度**——壳层沿外法线偏移
>    0.15×min(scale)（法线=薄轴+径向定向+kNN 平滑），彻底消除共心双片的
>    深度排序 z-tie（强妆色差在 turntable 里会闪），妆成为有物理厚度的
>    层而非贴面染色；② **真实纹理透出**——底模 albedo 高通直接调制壳层
>    颜色（hp_gain=0.10），训练出来的唇纹/毛孔透过妆层可见。顺带修掉
>    `_base_micro_hp` 的存量 bug：逐 splat 高通采样误写 `hp[tyi,tyi]`
>    （对角线索引），真实纹理一直是错值——修复后纹理透出强度 3.1×。
>
> **CUDA 实测结论（2026-09-22，RTX 5060 Ti，d9 壳层资产 f2_shell）**：
>    管线全链路 GPU 跑通——壳层 79.4 万 splats（薄层偏移自检 0.15×min_scale
>    ✓）、3D 锚定（唇 3340/眉 344/眼线 479）、ΔE00=19.07 + 自动重标定闭环、
>    材质 AOV 合成全程生效、壳层单独渲染结构完整位置正确。**交付级渲染被
>    资产质量挡住**：d9/d6 的 480p/SR 噪声 splat 毯式覆盖表面，精确训练视角
>    时恰好对齐隐藏；任何重投影（orbit ±2-6°/距离变化）都让噪声 splat 翻上
>    来——彩色噪斑 + 白洞，DC-only 也一样（d6 额外受损：28% splats opacity
>    <0.3）。SH 二阶残差实测是纯噪声（同机位 SH 比 DC 多一整层彩虹碎裂）。
>    据此：① `deliver` 新增 `sh_mode="auto"` 门控（探测帧 SH/DC 双向渲染按
>    "白蚀占比+高频能量"评分，SH 须显著更优 <0.85× 才启用；视角相关高光
>    由 shade AOV 补）；② 交付级画质必须 ≥1080p 重录重训（上文结论再次
>    被实测确认），壳层/材质/ΔE 架构在干净资产上的最终验收随之进行。
>
> **重建式降噪 A/B（2026-09-22 续）**：SH 降阶重训（同 480p 数据、同 30k
>    iters，仅 sh_degree 2→1）——deg1 全面胜出：留出 PSNR 23.96 vs 23.69、
>    orbit 渲染质量分 0.40 vs 0.44/0.47（d9）、一阶残差温和到可安全参与
>    渲染（deg2 残差即灾难）。**TrainConfig 默认 sh_degree 2→1**（≥1080p
>    多视角重录后可回 2 换视角相关油光）。SR 重训资产（d6-1080，PSNR
>    24.44）同期公平复测：结构最完整但白色大块噪声 splat 毯（SR 放大高光
>    烘进 DC），质量分与 d9 同级——SR/480p 两条路全部撞同一堵数据墙，
>    剩余提升空间只在重录。
>
> **R+ 验证循环揪出的三个资产级存量 bug（均已修复+回归测试）**：
> ① `train_base` 导出把 gsplat 的 **wxyz 四元数**直接当内部 xyzw 写入，
>    `export_ply` 再"转换"一次 → 循环错位，导出资产所有 splat 朝向错误
>    （连带 material.bin 法线、Unity 渲染、.splat 全错）；
> ② `SplatCloudBuilder.export_ply` opacity 用 `log(a/√(1-a))` 编码，读取端按
>    标准 logit 解码 → 高不透明度 splat 往返一次系统性变透明（0.95→0.81），
>    脸面渗底色；现统一标准 logit（`tools/fix_*.py` 可修旧产物）；
> ③ `_init_params` 单位四元数初始化只有 3 元素 → 任何全新训练在 gsplat 断言
>    崩溃（当前工作区无法复现训练，旧资产是更早版本产物）。
> **花斑主修复**：`mip3d_filter`（Mip-Splatting §3.1 的 3D 平滑滤波）——480² 帧
> 34 万 splats 时脸部 ~10 splats/px，亚像素 splat 渲染成点采样花斑/拉丝；
> 尺度下限 = γ×3近邻均距（γ=0.3 实测去斑保特征，0.5 起糊嘴）。这是对比图里
> "资产花"的第一根因，比任何光照/材质调整都靠前。

## 一、旧链路为什么不可能逼真（根因，不是参数问题）

| # | 根因 | 证据 | 后果 |
|---|------|------|------|
| 1 | **"底模"不是训练出来的 3DGS**：`run_real_fit` 路径 = MediaPipe 468 地标模板网格 + Kabsch 变换 + 视频投影取色 | `out/compare/compare_real_fit.png` 面具感 | 一切真实感无从谈起 |
| 2 | **真 3DGS 重建是退化的**：Brush 在表情剧烈变化的单目视频上直接训练，densify 崩塌 | `out/real/project/final.ply` 仅 **1918** 个高斯（一张逼真人脸需 20 万+） | 就算用真重建产物也一样糊 |
| 3 | **妆容 = 逐 splat 染色 + 程序化壳层贴片**：`fit_makeup.apply_makeup/build_makeup_shell` | `_diag_shell.png` 粉色模糊块 | 油漆感/贴纸感，无材质 |
| 4 | 光照是玩具级：全局 gloss 标量 × `ndh^90`，无 roughness/clearcoat/SSS | `AvatarSplatRenderer.cs:159` | 唇釉和粉饼看起来一样 |

## 二、新架构（appearance/）

```
视频帧 ─→ frames.select_frames      表情一致簇（嘴开度绝对上限 + MAD 聚类）
       ─→ train_base.train_base    gsplat 光度训练（densify 全开，30 万+ 高斯）
       │                            + 后处理剪枝（蒙版外漂浮物/巨块/低透明度）
       ─→ uvbind.bind_uv           canonical UV + 区域覆盖 → 点云一等属性
       ─→ makeup_uv.bake/apply     2048² UV 目标场（albedo/rough/coat/sss/sheen/kL/chroma）
       │                            唇红带 = 3D 拓扑光栅化进 UV；微观纹理（唇纹/粉感）
       ─→ optimize（可选）          Stable-Makeup guidance 存在时图像空间联合求解（几何冻结）
       ─→ 导出                      madeup.ply/.splat + material.bin(MKMAT1) + 对比图 + report
```

关键设计决策：

1. **静态写真人像不需要 4D**。单目视频训练崩塌的根源是跨表情帧混训（同一表面点
   要解释多种外观）。按表情聚类只训"主表情簇"，得到清晰的 canonical 头像；
   表情驱动走 FLAME rig（独立链路）。
2. **妆容在 UV 空间合成，不逐 splat 涂色**。锐度由贴图分辨率决定（2048²）；
   `apply_to_cloud` 用 Lab 部分迁移把目标场烘进 splat 颜色——皮肤纹理与原生
   光影保留，只有颜色走向妆色。系数逐区域写入 UV（`PHOTOREAL_LAB`）：
   底妆只匀肤不提亮，唇/腮红/眼影强色度。
3. **化妆品 = 薄层材质**。rough（粉↑釉↓）/ coat（唇釉清漆+Schlick Fresnel）/
   sss（唇部背光透光红移）/ sheen（珠光绒光）逐 splat 携带，三端同式实现：
   `appearance/pbr.py`（Python 参考）↔ `GaussianAvatarSplat.shader`（Unity）↔
   WebGL viewer（待接入 per-splat 材质，当前全局近似）。
4. **geometry 冻结的外观优化回路**（AvatarMakeup 本地化）：`optimize.py` 在
   有外部 2D guidance（Stable-Makeup 对多视角素颜渲染的输出）时，图像空间
   联合求解逐 splat 颜色残差；identity 锁保护非妆区（牙齿/眼白）。
   纯参数化路径不需要它（UV 烘焙即最优）。

## 三、用法

```bash
# 全流程（表情选帧 → 30k 光度训练 → UV 绑定 → 妆容 → 导出/对比图）
python preview/run_photoreal.py --project out/real --sfm out/real/sfm/sparse/3 \
    --init out/real/project/face_dense.ply \
    --spec makeup-skill/presets/date-rose.json --out out/photoreal/d6 --iters 30000

# 只改妆容重跑（--reuse-base 默认开启，底模秒级复用）
python preview/run_photoreal.py ... --iters 0 --skip-train
```

产物（out/photoreal/<name>/）：`base.ply/.splat`（素颜底模）、`madeup.ply/.splat`
（妆后）、`material.bin`（MKMAT1：法线+rough+coat+sss+sheen，Unity
`GaussianAvatarParser.LoadMaterial` 消费）、`makeup_uv_debug.png`（UV 妆容图）、
`compare_*.png`（真实帧|素颜|妆后）、`train_report.json`/`report.json`。

## 四、环境依赖（本机已就绪）

- CUDA torch 2.9.1+cu130 + **gsplat 1.5.3**（源码编译，见下）+ torchvision/lpips 可选
- gsplat Windows 编译三件套：VS Build Tools (C++ 工作负载) + CUDA Toolkit 13.4 + 
  `NVCC_FLAGS="-Xcompiler /Zc:preprocessor"`（CUDA 13 CCCL 强制要求标准预处理器）
- **上游 v1.5.3 的已知缺陷（本仓库已打桩绕过，标准 rasterization() 路径不受影响）**：
  `Rasterization.cpp` 调用的 `launch_rasterize_to_pixels_2dgs_bwd_kernel` 与
  `launch_rasterize_to_pixels_from_world_3dgs_{fwd,bwd}_kernel` 三族符号，
  其 .cu 定义与头文件声明签名不一致（前者 .cu 多一个 uint32_t 参数），
  Windows 链接失败；这三个内核不在 `rasterization()` 主路径上。
  安装脚本：`out/probe/install_gsplat.bat`（glm 子模块需手动放入 third_party）。

## 五、验收数据（out/real/d6，RTX 5060 Ti）

| 指标 | 旧链路 | R 升级（30k iters） |
|------|--------|--------------------|
| 脸部高斯数 | 1,918（全场景） | ~230,000（剪枝后） |
| 留出帧 PSNR（脸区） | — | ~25 dB |
| 训练耗时 | — | ~4 min |
| 唇部细节来源 | 无（色带×wrap） | 训练光度细节 + UV 微观纹理 |
| 视角一致性 | 投影取中位数（花斑） | 多视角光度优化（天然一致） |

## 六、旧链路退役清单（2026-09 清理）

产品定位收敛为：**上传/扫描 → 3DGS 数字资产 → 资产上渲染妆容**。据此清理：

| 项 | 处置 | 原因 |
|----|------|------|
| `preview/run_real_fit.py`、`preview/compare_bare_vs_makeup.py` | **删除** | 模板面具链路 CLI，效果不可接受 |
| `fit_makeup.py` 的 apply_makeup/build_makeup_shell/fit/fit_canonical/apply_guidance/render_cloud | **删除** | 逐 splat 染色 + 程序化壳层 = 油漆感根因；同文件保留三角化/配准/唇拓扑/唇带权重/UV 归属（新链路语义基础设施） |
| `reconstruct.py` Brush 链路（FFmpeg+COLMAP+Brush） | **主链路不再依赖** | 被 `appearance/sfm.py`(pycolmap) + `appearance/train_base.py`(gsplat) 取代；文件保留（ooosplat 互操作参考） |
| desktop-app `face3dgs_tab.py` ReconWorker/FitWorker | **替换** | ModelWorker（SfM+训练→资产）/ MakeupWorker（资产上妆，秒级） |
| CLI `python -m makeupstudio.face3dgs` | **重排** | `status/capture/asset/makeup`（rebuild/isolate/fit 移除） |
| 测试 `test_makeup_upgrade.py` / `test_face3dgs.py` 旧路径用例 | **重写/删除** | 语义层用例保留（拓扑/唇带/门控），渲染与上妆旧用例移除 |

## 七、后续路线

1. WebGL viewer per-splat 材质（material.json f16 base64 已产出；离线渲染
   已先行消费同一套语义，见 F 升级 ①）
2. ~~Stable-Makeup guidance 接入 `optimize.py`~~（F 升级 ③ 已接通；剩余是把
   Stable-Makeup 换成更强/multi-shot 的迁移模型时只需替换 `guidance.generate`。
   零依赖的形状级还原已由 Q 升级 ⑦ `refshape` 覆盖——guidance 剩余价值在
   妆效质感/渐变的像素级迁移）
3. LAM 单图入口（`face3dgs/lam_adapter.py`，门控）：无视频用户秒级底模
4. FLAME 表情驱动（P6）：canonical 头像 + LBS，妆随表情走
5. ~~还原度进产品验收门：`makeup_delta_e` 超阈值自动降级/重标定~~
   （Q 升级 ⑧ 已接通：超阈值自动调整 opacity 重烘 ≤2 轮取最优）
6. 底模画质台阶：≥1080p 重录采集规范落地（SR 过渡方案已证实不可商用）；
   器官屏蔽/gaze 聚类/SH 正则（Q ①-④）解决结构瑕疵，分辨率解决细节
