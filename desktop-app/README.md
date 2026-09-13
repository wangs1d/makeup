# MakeupStudio — PC 化妆助手（去 Unity 桌面应用）

PySide6 桌面应用，替代原 Unity App：**① 实时化妆台**（摄像头 + MediaPipe 478 点 + 实时妆容叠加 + 步骤陪练）、**② 3DGS 效果预览**（高密度 3D 高斯泼溅渲染所选妆容的上妆效果，可旋转/缩放/素颜对比/快照导出）与 **③ 我的 3D 脸**（摄像头环绕采集 → 真·3DGS 重建用户脸部 → 妆容贴合到用户自己的点云上）。

## 运行

```bash
pip install -r requirements.txt   # PySide6 / opencv-python / mediapipe / numpy / pillow / websockets
python main.py --camera 0
```

模型文件 `models/face_landmarker.task` 首次需联网下载（已随仓库提供）。

## 架构

```
desktop-app/
├── main.py                  # 入口
├── models/face_landmarker.task
├── makeupstudio/
│   ├── tracker.py           # MediaPipe Tasks FaceLandmarker + 1€ 滤波 + solvePnP 姿态
│   ├── compositor.py        # 摄像头帧妆容合成（图像域蒙版，几何规则与渲染内核一致）
│   ├── splat3d.py           # 3DGS 生成器：烘焙纹理→网格表面稠密采样→各向异性高斯
│   │                        #   + .splat / .ply 导出 + 还原度指标（前向泼溅 PSNR）
│   ├── face3dgs/            # ★ 真实用户脸部的 3DGS 重建与妆容贴合（管线与 ooosplat 对齐）
│   │   ├── engines.py       #   引擎发现：OOOSPLAT_FFMPEG/COLMAP/BRUSH → ENGINE_DIR → PATH
│   │   ├── capture.py       #   环绕采集引导（左/正/右偏航覆盖 + 光照/距离质检）
│   │   ├── reconstruct.py   #   FFmpeg 抽帧 → COLMAP → Brush → final.ply（后端可换远程服务）
│   │   ├── colmap_io.py     #   COLMAP sparse 模型最小读写 + 投影
│   │   ├── isolate.py       #   人脸框重投影投票 → 裁出脸部点云 face.ply
│   │   ├── fit_makeup.py    #   地标三角化 + canonical 配准 + 区域蒙版上色 → madeup.ply
│   │   └── __main__.py      #   CLI: status / capture / rebuild / isolate / fit
│   ├── face3dgs_tab.py      #   主窗口"③ 我的 3D 脸"标签页（采集/重建/贴合工作流）
│   ├── server.py            # 查看器静态服务 (127.0.0.1:8791)
│   ├── coach.py             # 步骤陪练（本地状态机 + Windows SAPI TTS）
│   └── app.py               # PySide6 主窗口（化妆台 / 3DGS 预览 / 3D 脸 / 控制面板 / 陪练条）
├── viewer/index.html        # 自研 WebGL2 高斯泼溅查看器（零依赖：实例化渲染 +
│                            #   CPU 计数排序 + 各向异性协方差投影 + 双点云分屏对比）
└── out/splat/               # 生成的 makeup.splat / bare.splat / version.json（热更新）
```

## 我的 3D 脸（真实脸部 3DGS 化）

链路与 [ooosplat](https://github.com/ooolabdev/ooosplat) 一致（Apache-2.0）：
**采集引导 → FFmpeg 抽帧 → COLMAP 相机重建 → Brush 训练 → 脸部隔离 → 妆容贴合**。

```bash
py -m makeupstudio.face3dgs status    # 检查 FFmpeg / COLMAP / Brush
py -m makeupstudio.face3dgs capture  -o out/face3dgs/me.mp4
py -m makeupstudio.face3dgs rebuild  -v out/face3dgs/me.mp4 -p out/face3dgs/proj --quality standard
py -m makeupstudio.face3dgs isolate  -p out/face3dgs/proj -o out/face3dgs/face.ply
py -m makeupstudio.face3dgs fit      -p out/face3dgs/proj -f out/face3dgs/face.ply \
                                     --spec makeup-skill/presets/date-rose.json -o out/face3dgs/fitted
```

- **引擎获取**：安装一次 [OOOSplat 桌面版](https://github.com/ooolabdev/ooosplat)（自带版本锁定的
  FFmpeg/COLMAP/Brush），设置 `OOOSPLAT_ENGINE_DIR` 指向其引擎目录即可；也接受系统 PATH 或
  `OOOSPLAT_FFMPEG`/`OOOSPLAT_COLMAP`/`OOOSPLAT_BRUSH` 显式路径。重建需 CUDA GPU（建议 ≥8GB 显存）。
- **版本与网络实测**（2026-09-12，Windows）：
  - COLMAP 建议 **3.11.1**（`COLMAP.bat` 启动，含正确 DLL 路径）；4.2.0 无 CUDA 版在本机
    存在随机崩溃（0xC0000409），且需要 `QT_QPA_PLATFORM=offscreen`。管线对 3.x/4.x 的
    匹配旗标差异（`--SiftMatching/--FeatureMatching.use_gpu`）自动兼容；
  - 无 CUDA 构建必须显式 `--SiftExtraction.use_gpu 0`（默认会尝试 OpenGL）；
  - GitHub 直连慢时用镜像分块下载（`.engines/pget.py`，走 gh-proxy.com，14 连接 ≈ 1MB/s）；
  - ffmpeg 可选：缺失时自动用 OpenCV 内置解码抽帧；
  - **COLMAP 几何验证回退**：本机实测 3.11/4.2 nocuda 的特征与匹配正常但几何验证全零
    （mapper 无法初始化），管线检测到该情况自动用 OpenCV RANSAC 重写验证（`verify_cv2.py`）；
    若 mapper 仍失败，可用真 CUDA 构建，或对已知位姿的场景用
    `desktop-app/tools/write_gt_txt.py` + `model_converter` 直接生成 sparse 模型；
  - Brush CLI（v0.3.0）：`brush_app <数据集根> --total-steps N --export-path D --export-name F`。
    注意其训练日志不进 stdout（GUI 子系统），质量验收以导出 ply 的 splat 数/位置为准。
- **服务器部署预留**：重建后端抽象为 `ReconstructionBackend`（`reconstruct.py`）。本地实现是
  `LocalEngineBackend`；上服务器时实现 `RemoteBackend`（提交视频→轮询→取回 final.ply + 相机），
  采集/隔离/贴合代码零改动——所有重活在后端机跑，客户端只做采集与展示。

## 与渲染内核的一致性

3DGS 预览与实时叠加共用 `makeup-skill/scripts/preview_render.py` 的同一套妆容语义
（`landmark-regions.json` 区域几何 / `color_stops` 向心度取色 / ENVS 环境光 / 羽化表）。
还原度以量化指标验收：同相机前向泼溅 vs 内核参考图 **PSNR ≈ 22 dB**
（`tests/test_desktop_app.py::test_3dgs_fidelity_to_reference_render`，锚点溅射层已烘入点云）。

## 已知限制（MVP 边界）

- **我的 3D 脸**依赖外部引擎（见上）；未安装时 Tab 内显示安装指引，其余功能不受影响。
- 妆容贴合 v1 为"区域蒙版 + wrap-diffuse 上色"的 splat 颜色混合；光泽高光随视角变化、
  妆容的细节纹理（闪片等）尚不逐视角建模。表情驱动（张嘴/眨眼时的妆容形变）待 FLAME 路线。
- 单摄像头环绕采集约覆盖 ±60° 半环，脑后无重建（妆容场景不需要）。
- 实时叠加为图像域蒙版；极端侧脸（>60°）跟随精度下降，可切换 mesh warp 路径（待接）。
- 陪练为本地步骤计时 + TTS；VLM 视觉对比需 API Key（makeup-skill `live_coach` 接入点已留）。
- 查看器需 WebGL2（QtWebEngine / Chrome / Edge 均满足）。
