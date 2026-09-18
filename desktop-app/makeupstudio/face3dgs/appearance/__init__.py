"""appearance — 写真试妆新链路（R 升级）：真·3DGS 底模 + UV 妆容 + PBR 材质。

模块分工：
    frames      表情一致性选帧（3DGS 训练不崩塌的前提）
    train_base  gsplat 光度训练（densify 全开）→ base.ply
    uvbind      canonical UV/区域覆盖 → 点云一等属性
    makeup_uv   2048² UV 妆容目标场（参数化 + guidance 聚合 + 微观纹理）
    optimize    guidance 监督的可微外观优化（几何冻结）
    pbr         化妆品材质着色参考实现（rough/coat/sss/sheen）
    render_pbr  带材质的前向泼溅预览
    pipeline    全流程编排
"""
