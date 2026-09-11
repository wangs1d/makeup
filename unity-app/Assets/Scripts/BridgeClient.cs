// BridgeClient.cs
// 与 makeup bridge_server 的 WebSocket 客户端（System.Net.WebSockets，无第三方依赖）。
// 协议见 makeup-skill/references/bridge-protocol.md：hello 握手、ref-ack 匹配、断线自动重连。
// 收到的业务消息通过 OnMessage 回调抛给 MakeupAppMain 处理；SendAck 按 ref 回执；SendJson 发事件。
using System;
using System.Collections;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace MakeupMirror
{
    public class BridgeClient : MonoBehaviour
    {
        public string bridgeUrl = "ws://127.0.0.1:8765";
        public string clientId = "unity-mirror-01";
        public float reconnectInterval = 3f;

        public bool Connected { get; private set; }
        /// hello_ok 里 bridge 给的资产 HTTP 基址（如 http://127.0.0.1:8768/assets/），可为 null
        public string AssetBaseUrl { get; private set; }
        public event Action<JObject> OnMessage;         // 全部业务消息
        public event Action<bool> OnConnectionChanged;

        private ClientWebSocket _ws;
        private Coroutine _loop;

        private void OnEnable() { _loop = StartCoroutine(RunLoop()); }
        private void OnDisable()
        {
            if (_loop != null) { StopCoroutine(_loop); _loop = null; }
            CloseQuiet();
        }

        private IEnumerator RunLoop()
        {
            while (true)
            {
                var connectTask = ConnectAsync();
                yield return new WaitUntil(() => connectTask.IsCompleted);
                if (!connectTask.Result)
                {
                    UpdateConnected(false);
                    yield return new WaitForSeconds(reconnectInterval);
                    continue;
                }
                var recvTask = ReceiveLoopAsync();
                yield return new WaitUntil(() => recvTask.IsCompleted);
                UpdateConnected(false);
                yield return new WaitForSeconds(reconnectInterval);
            }
        }

        private async Task<bool> ConnectAsync()
        {
            try
            {
                var ws = new ClientWebSocket();
                using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5)))
                    await ws.ConnectAsync(new Uri(bridgeUrl), cts.Token);
                _ws = ws;
                await SendJson(new JObject
                {
                    ["type"] = "hello",
                    ["role"] = "app",
                    ["client_id"] = clientId,
                    ["caps"] = new JArray("render", "frame", "coaching", "tracking_v2", "assets_url", "progress"),
                });
                UpdateConnected(true);
                Debug.Log($"[bridge] 已连接 {bridgeUrl}");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[bridge] 连接失败（{reconnectInterval}s 后重试）：{e.Message}");
                return false;
            }
        }

        private async Task ReceiveLoopAsync()
        {
            var buf = new byte[64 * 1024];
            try
            {
                while (_ws != null && _ws.State == WebSocketState.Open)
                {
                    var sb = new StringBuilder();
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await _ws.ReceiveAsync(new ArraySegment<byte>(buf), CancellationToken.None);
                        if (result.MessageType == WebSocketMessageType.Close)
                            return;
                        sb.Append(Encoding.UTF8.GetString(buf, 0, result.Count));
                    } while (!result.EndOfMessage);

                    if (result.Count == 0) continue;
                    try
                    {
                        var msg = JObject.Parse(sb.ToString());
                        if (msg.Value<string>("type") == "hello_ok")
                            AssetBaseUrl = msg.Value<string>("asset_base_url");
                        OnMessage?.Invoke(msg);
                    }
                    catch (JsonException e)
                    {
                        Debug.LogWarning($"[bridge] 消息解析失败：{e.Message}");
                    }
                }
            }
            catch (Exception) { /* 断开 → 外层重连 */ }
        }

        // ---------- 发送 ----------

        public async Task SendJson(JObject msg)
        {
            var ws = _ws;
            if (ws == null || ws.State != WebSocketState.Open) return;
            byte[] bytes = Encoding.UTF8.GetBytes(msg.ToString(Formatting.None));
            try
            {
                using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10)))
                    await ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text,
                                       true, cts.Token);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[bridge] 发送失败：{e.Message}");
            }
        }

        public async Task SendAck(string reference, bool ok, string error = null)
        {
            if (string.IsNullOrEmpty(reference)) return;
            var ack = new JObject
            {
                ["type"] = "ack",
                ["ref"] = reference,
                ["status"] = ok ? "ok" : "error",
            };
            if (!ok) ack["error"] = error ?? "unknown";
            await SendJson(ack);
        }

        private void CloseQuiet()
        {
            try { _ws?.Abort(); } catch { }
            _ws = null;
        }

        private void UpdateConnected(bool v)
        {
            if (Connected == v) return;
            Connected = v;
            OnConnectionChanged?.Invoke(v);
        }
    }
}
