"""face3dgs — 真实用户脸部的 3DGS 重建与妆容贴合。

链路（与 ooosplat 管线对齐）：
    1. capture      摄像头环绕采集引导 → MP4 + 质量报告；
    2. reconstruct  抽帧 → COLMAP 相机重建 → Brush 训练 → final.ply（可本地引擎，可远程服务）；
    3. isolate      按每帧人脸框重投影投票，把全场景点云裁剪为脸部点云；
    4. fit_makeup   多帧地标三角化 → canonical 脸模型配准 → 按妆容 spec 给 splat 上色。

后端抽象：reconstruct.ReconstructionBackend 当前实现 LocalEngineBackend（调用本机
FFmpeg/COLMAP/Brush，引擎发现规则与 ooosplat 一致）；部署服务器时新增
RemoteBackend（HTTP 提交视频、轮询 final.ply）即可，采集/贴合侧代码不变。
"""
from __future__ import annotations
