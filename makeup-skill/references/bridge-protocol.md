# Bridge WebSocket 协议（v1.1）

bridge_server 在 `ws://127.0.0.1:8765`（可改 `--port`）提供消息总线，两类客户端：

- **agent**：本 skill 的脚本（apply_spec / live_coach / …）
- **app**：试妆 App（Unity 客户端，同协议可接任意客户端实现）

消息均为单帧 JSON 文本。所有消息带 `type`；请求类消息带 `ref`（任意字符串），
响应用 `{"type":"ack","ref":<原ref>,"status":"ok"|"error","error":"..."}` 回执。

## v1.1 相对 v1 的变化

1. **精确路由**：agent 消息可带 `to`（app 的 client_id）定向投递；缺省广播给所有 app。
2. **ack/frame 按 ref 回投**：bridge 记录"哪个 agent 发出了带 ref 的请求"，app 的应答只回给
   那个 agent（v1 按"最近 agent"启发式，多 agent 并发会错投）。转发的 app 消息带 `from`
   字段（app 的 client_id）。
3. **资产 HTTP 侧车**：bridge 附带 `http://127.0.0.1:<asset-port>/assets/<key>`（默认 8768），
   `PUT` 上传 / `GET` 下载；`hello_ok` 与 `status` 下发 `asset_base_url`。大资产不再 base64
   内嵌单条 WebSocket 消息（上限受 max_size 32MB 约束）。
4. **coaching 带进度**：`step` / `steps` / `step_name` / `progress`（0~1），App 渲染步骤进度条。
5. **tracking_state 实测**：`fps` 为实测值、`landmarks` 为实际点数、`pose` 表示是否携带 6DoF
   姿态（v1 为硬编码常量）。
6. `frame` 回传带 `ref`（与 request_frame 匹配）与实际 `w`/`h`。

## 握手

连接后第一条消息必须声明角色：

```json
{"type": "hello", "role": "app", "client_id": "unity-mirror-01",
 "caps": ["render","frame","coaching","tracking_v2","assets_url","progress"]}
```

- `caps`（app 端）：`render`（可渲染妆容）、`frame`（可回传摄像头帧）、`coaching`（可显示指导）、
  `tracking_v2`（消费二进制追踪协议）、`assets_url`（支持 HTTP 资产下载）、`progress`（步骤进度条）。
- 重复 role+client_id 顶掉旧连接（close 4001）。
- `hello_ok` 回包：`{"type":"hello_ok","bridge":"makeup-bridge/1.1","apps":N,"app_ids":[...],
  "asset_base_url":"http://127.0.0.1:8768/assets/"}`（未开侧车时无 asset_base_url）。

## Agent → App

| type | 载荷 | 说明 |
|---|---|---|
| `apply_spec` | `{"spec": {...}, "intensity": 0.6?, "assets": {"名": "<base64>"}? 或 "assets_url": {"名": "url"}?, "to": "app-id"?}` | 下发妆容并渲染。资产 ≤ MAKEUP_ASSET_INLINE_MAX（默认 1MB）内嵌 base64，更大走 assets_url。spec 建议用 bake_assets 的 baked_spec（带 `baked` 引用与 `splat_layers`） |
| `clear_makeup` | `{}` | 卸妆还原 |
| `set_intensity` | `{"value": 0.0~1.0}` | 只调浓度不重发 spec |
| `coaching` | `{"text","area"?,"priority":"info"\|"warn","speak"?:true,"note"?:string,"step"?:int,"steps"?:int,"step_name"?:str,"progress"?:0~1}` | 实时指导提示；App 队列显示，warn 插队、同文 60s 去重；`speak` 且 warn/done 时 TTS 朗读 |
| `request_frame` | `{"quality": 75?}` | 请求一帧当前摄像头画面（App 回 ack 后回 `frame`） |
| `ping` | `{}` | 探活：无 `to` 时由 bridge 自答；带 `to` 转发给指定 app |

## App → Agent / Bridge

| type | 载荷 | 说明 |
|---|---|---|
| `frame` | `{"ref", "data": "<base64 JPEG>", "w":1280, "h":720}` | 按 ref 回给请求方（无 ref 兜底给最近请求者） |
| `tracking_state` | `{"ok": true, "fps": 29.5, "landmarks": 468, "pose": true}` | 每秒节流上报（实测值）；广播给所有 agent，带 `from` |
| `intensity_changed` | `{"value": 0.7}` | 用户拖 App 内滑杆时节流上报（0.25s） |
| `error` | `{"code": "...", "message": "..."}` | App 端错误（如渲染失败） |

## Bridge 内建行为

- 路由：agent 消息按 `to` 定向或广播；app 的 ack/frame 按 ref 精确回投；无主的 ack 记日志丢弃。
- 无 app 在线时：`ack status=error, error="no_app_connected"`；`to` 指定了不存在的 app 时
  `error="no_such_app:<id>"`。
- 心跳：WebSocket 传输层 ping（bridge 20s 间隔、15s 超时）；应用层 `ping`（无 to）回
  `ack ok` 并附 `apps`/`app_ids`。
- 状态查询：`{"type":"status"}` → `{"type":"status","apps","agents","app_ids","bridge",
  "asset_base_url"}`。

## 资产 HTTP 侧车（端口默认 8768，`--asset-port 0` 关闭）

| 方法 | 路径 | 行为 |
|---|---|---|
| `PUT`/`POST` | `/assets/<key>` | 存入内存 LRU（总容量 256MB），返回 `201 {"key","url","bytes"}` |
| `GET` | `/assets/<key>` | 返回资产字节（Content-Type 按扩展名） |
| `GET` | `/health` | `{"bridge","assets","bytes"}` |

key 约束 `^[A-Za-z0-9_\-./]{1,200}$`（客户端 `BridgeClient.upload_asset` 会用
`<sha256[:16]>/<basename>` 生成，天然去重且防路径穿越）。

## 人脸追踪 sidecar 协议（独立于 WebSocket，UDP）

见 `unity-app/tools/tracking_protocol.py` 头注释：v2 二进制（magic `MKT2`，4×4 姿态矩阵 +
头部局部关键点 + 图像坐标）、v1 JSON 兼容（`--legacy-json`）、App→sidecar 帧中继
（UDP :8767，magic `MKF1` 分片）。
