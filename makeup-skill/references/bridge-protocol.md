# Bridge WebSocket 协议（v1）

bridge_server 在 `ws://127.0.0.1:8765`（可改）提供消息总线，两类客户端：

- **agent**：本 skill 的脚本（apply_spec / live_coach / …）
- **app**：试妆 App（Unity 客户端，同协议可接任意客户端实现）

消息均为单帧 JSON 文本。所有消息带 `type`；请求类消息可带 `ref`（任意字符串），
响应用 `{"type":"ack","ref":<原ref>,"status":"ok"|"error","error":"..."}` 回执。

## 握手

连接后第一条消息必须声明角色：

```json
{"type": "hello", "role": "app", "client_id": "unity-mirror-01", "caps": ["render","frame","coaching"]}
```

`caps`（app 端）：`render`（可渲染妆容）、`frame`（可回传摄像头帧）、`coaching`（可显示指导）。
Bridge 收到重复 role+client_id 会顶掉旧连接。

## Agent → App

| type | 载荷 | 说明 |
|---|---|---|
| `apply_spec` | `{"spec": {...}, "intensity": 0.6?, "assets": {"文件名": "<base64 PNG>"}?}` | 下发妆容并渲染；intensity 缺省用 spec 内值。`assets` 为可选的烘焙纹理包（bake_assets 产物，App 先落盘再渲染，重名覆盖） |
| `clear_makeup` | `{}` | 卸妆还原 |
| `set_intensity` | `{"value": 0.0~1.0}` | 只调浓度不重发 spec |
| `coaching` | `{"text": "...", "area": "eye"?, "priority": "info"\|"warn", "speak": true?}` | 实时指导提示；App 显示文字，`speak` 时用 TTS 朗读 |
| `request_frame` | `{"quality": 75?}` | 请求一帧当前摄像头画面 |
| `ping` | `{}` | 探活，App 回 ack |

## App → Agent / Bridge

| type | 载荷 | 说明 |
|---|---|---|
| `frame` | `{"data": "<base64 JPEG>", "w":1280, "h":720}` | 响应 request_frame 回传一帧（只发给最近请求的 agent） |
| `tracking_state` | `{"ok": true, "fps": 29.5, "landmarks": 468}` | 追踪状态变化时上报；`ok:false` 时 agent 应提醒用户对准脸部 |
| `intensity_changed` | `{"value": 0.7}` | 用户在 App 拖滑杆时上报 |
| `error` | `{"code": "...", "message": "..."}` | App 端错误（如渲染失败） |

## Bridge 内建行为

- 按 role 路由：agent 发的 app 类消息广播给所有 app 连接；`frame` 只回给发起 request 的 agent。
- 无 app 在线时 agent 发 `apply_spec` 会收到 `ack status=error, error="no_app_connected"`。
- 心跳：WebSocket 传输层 ping（bridge 20s 间隔、15s 超时，客户端库自动应答）；应用层 `ping` 消息用于业务探活，回 `ack ok`。
- 状态查询：`{"type":"status"}` → `{"type":"status","apps":1,"agents":2}`。
