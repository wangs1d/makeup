# 试妆 App 安装与启动

能力一（试妆渲染）与能力三（实时指导）需要一个试妆 App 作为 Bridge 的 `app` 端客户端。
协议是公开契约（见 `bridge-protocol.md`），任何实现都可以；本仓库附带的是
`unity-app/`（Windows 桌面版，Unity + 3D 人脸网格 + 高斯溅射渲染）。

## 方式 A：直接运行已构建版本（推荐给用户）

1. 从主项目发布页下载 `MakeupMirror-<平台>.zip`（或 skill 发布包附带的 `app/` 目录）。
2. 解压后运行 `MakeupMirror.exe`。
3. 首次运行允许摄像头权限；标题栏出现 `bridge: connected` 即就绪。

App 端口等参数可在其旁边的 `appsettings.json` 修改（`bridge_url`、`tracker_udp_port`）。

## 方式 B：从源码构建（开发者）

需要 Unity 2022.3 LTS（Windows 模块）：

1. Unity Hub 新建 3D 项目到 `unity-app/` 对应目录（或直接打开该工程）。
2. 安装依赖包（Package Manager → Add package from git URL）：
   - `com.unity.nuget.newtonsoft-json`
   - （可选，唇部/高光体积渲染）UnityGaussianSplatting：
     `https://github.com/aras-p/UnityGaussianSplatting.git`
3. 按 `unity-app/docs/assembly.md` 搭场景（约 10 分钟：拖 prefab、填引用）。
4. File → Build Settings → Windows x86_64 → Build。

## 人脸追踪 sidecar

App 本身不做人脸关键点检测；由 Python sidecar 提供关键点（468 点）+ 6DoF 头部姿态：

```bash
pip install mediapipe opencv-python numpy
python unity-app/tools/face_tracker.py --relay    # 推荐：接收 App 中继帧（单摄像头闭环）
python unity-app/tools/face_tracker.py            # 旧模式：sidecar 自开摄像头
python unity-app/tools/face_tracker.py --synthetic # 无摄像头/无 mediapipe 的联调模拟
```

先启动 sidecar 再启动 App；App 状态栏显示 `tracking: ok · N fps · pose ✓` 即正常
（无 `pose ✓` 说明还在走旧版 v1 JSON 兼容路径，妆容退回固定平面映射）。
sidecar 完全本地运行，不上传任何画面。

## 无 App 时的降级

- 能力二（解析素材）不依赖 App，可照常使用。
- 能力一/三：`setup_check.py` 检测到无 app 连接时，agent 应引导用户按上文安装；
  用户暂时不想装 App 时，可把解析出的 spec 与 analysis.md 以图文形式向用户讲解妆容。
