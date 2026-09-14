// AvatarStationFlow.cs — 妆容台流程状态机（Bridge v1.2 画像会话）。
// 流程：用户上传 3DGS 画像 → 选妆在画像上预览 → 用户确认 → 进妆容台（辅助化妆）。
//   idle → registered（avatar_register：加载画像 ply）
//        → preview（avatar_preview：应用 tint + 附加溅射，画像居中展示）
//        → confirmed（avatar_confirm：锁定妆效）
//        → station（enter_station：画像移到侧栏作目标参照，摄像头画面保留给 VLM 指导；
//          真脸不再渲染任何妆容—— MakeupAppMain 的 legacy 真脸附妆默认关闭）
// 每次状态迁移广播 station_state 给 agent。
using System;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;

namespace MakeupMirror
{
    public class AvatarStationFlow : MonoBehaviour
    {
        public enum StationState
        {
            Idle = 0,
            Registered = 1,
            Preview = 2,
            Confirmed = 3,
            Station = 4,
        }

        [Header("引用（MakeupAppMain 接线）")]
        public GaussianAvatarRenderer avatarRenderer;
        public AvatarSplatRenderer avatarSplats;
        public BridgeClient bridge;

        [Header("画像布局（相机空间，世界=相机空间）")]
        [Tooltip("预览姿态：画像中心位置（正对用户）")]
        public Vector3 previewPosition = new Vector3(0f, 0f, 1.15f);
        [Tooltip("妆容台姿态：画像移到侧栏（右）作为目标参照")]
        public Vector3 stationPosition = new Vector3(0.42f, 0.02f, 1.0f);
        [Tooltip("画像显示尺寸：脸高（米）")]
        public float faceHeightMeters = 0.22f;
        [Tooltip("妆容台侧栏缩小系数")]
        public float stationScaleK = 0.75f;

        public StationState State { get; private set; } = StationState.Idle;
        public string AvatarId { get; private set; }

        /// <summary>状态迁移回调（MakeupAppMain 用于刷新状态栏）。</summary>
        public event Action<StationState> StateChanged;

        // ---------- Bridge 消息（MakeupAppMain 分发） ----------

        public bool Handles(string type) => type == "avatar_register" || type == "avatar_preview"
            || type == "avatar_confirm" || type == "enter_station" || type == "leave_station";

        public async Task HandleAsync(JObject msg, string reference)
        {
            string type = msg.Value<string>("type");
            try
            {
                switch (type)
                {
                    case "avatar_register":
                        await Register(msg);
                        break;
                    case "avatar_preview":
                        await Preview(msg);
                        break;
                    case "avatar_confirm":
                        if (State < StationState.Preview)
                            throw new InvalidOperationException("尚未进入画像预览，无法确认");
                        SetState(StationState.Confirmed);
                        break;
                    case "enter_station":
                        EnterStation(msg);
                        break;
                    case "leave_station":
                        SetState(AvatarId != null ? StationState.Confirmed : StationState.Idle);
                        ApplyLayout();
                        break;
                }
                if (reference != null) await bridge.SendAck(reference, true);
            }
            catch (Exception e)
            {
                Debug.LogError($"[station] 处理 {type} 失败：{e}");
                if (reference != null) await bridge.SendAck(reference, false, e.Message);
            }
        }

        private async Task Register(JObject msg)
        {
            string id = msg.Value<string>("avatar_id") ?? "avatar";
            byte[] ply = await FetchAsset(msg, "avatar.ply");
            if (ply == null) throw new InvalidOperationException("画像 ply 下载失败");
            avatarRenderer.Load(id, ply);
            AvatarId = id;
            avatarRenderer.transform.localScale = Vector3.one * faceHeightMeters;
            SetState(StationState.Registered);
            ApplyLayout();
        }

        private async Task Preview(JObject msg)
        {
            string id = msg.Value<string>("avatar_id");
            if (avatarRenderer.AvatarId != id || !avatarRenderer.IsLoaded)
            {
                // 未注册/换画像：先补拉画像
                await Register(msg);
            }
            byte[] tint = await FetchAsset(msg, "tint.bin")
                          ?? EncodeTint(msg["makeup"]?["tint_bin_b64"]);
            if (tint == null) throw new InvalidOperationException("缺少妆容 tint 数据");
            avatarRenderer.ApplyTint(tint);

            byte[] splats = await FetchAsset(msg, "add_splats.json");
            if (splats == null && msg["makeup"]?["add_splats"] is JArray inline)
                splats = System.Text.Encoding.UTF8.GetBytes(new JObject { ["splats"] = inline }.ToString());
            avatarSplats.Apply(splats);

            float inten = msg["look"]?.Value<float?>("intensity") ?? 0.8f;
            avatarRenderer.makeupIntensity = inten;
            avatarSplats.intensity = inten;
            SetState(StationState.Preview);
            ApplyLayout();
        }

        private void EnterStation(JObject msg)
        {
            if (avatarRenderer.IsLoaded && State < StationState.Confirmed)
            {
                // live_coach --station 直入妆容台（无显式 confirm）：视为已确认
                SetState(StationState.Confirmed);
            }
            float inten = msg["look"]?.Value<float?>("intensity") ?? -1f;
            if (inten >= 0f) avatarSplats.intensity = inten;
            SetState(StationState.Station);
            ApplyLayout();
        }

        public void ClearLook()
        {
            avatarRenderer.ClearTint();
            avatarSplats.Clear();
            if (State >= StationState.Preview) SetState(StationState.Registered);
        }

        // ---------- 布局 ----------

        private void ApplyLayout()
        {
            bool station = State == StationState.Station;
            avatarRenderer.transform.localPosition = station ? stationPosition : previewPosition;
            float s = faceHeightMeters * (station ? stationScaleK : 1f);
            avatarRenderer.transform.localScale = Vector3.one * s;
        }

        private void SetState(StationState s)
        {
            if (State == s) return;
            State = s;
            StateChanged?.Invoke(s);
            _ = bridge?.SendJson(new JObject { ["type"] = "station_state",
                ["state"] = s.ToString().ToLowerInvariant(), ["avatar_id"] = AvatarId ?? "" });
        }

        // ---------- 资产下载 ----------

        private static async Task<byte[]> FetchAsset(JObject msg, string name)
        {
            string url = msg["assets_url"]?[name]?.Value<string>();
            if (url == null) return null;
            using var req = UnityWebRequest.Get(url);
            req.timeout = 60;                 // 画像 ply 可达百 MB
            var op = req.SendWebRequest();
            while (!op.isDone) await Task.Yield();
            return req.result == UnityWebRequest.Result.Success ? req.downloadHandler.data : null;
        }

        private static byte[] EncodeTint(JToken makeup)
        {
            string b64 = makeup?["tint_bin_b64"]?.Value<string>();
            return b64 != null ? Convert.FromBase64String(b64) : null;
        }
    }
}
