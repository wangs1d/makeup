// MakeupAppMain.cs
// App 主入口：程序化搭建场景（背景 quad / 人脸 mesh / UI），接线 Bridge ↔ 渲染 ↔ 指导显示。
//
// 真·AR 对齐（P1）：Unity 相机固定在原点、朝 +Z、旋转恒等 —— 世界空间 = 相机空间。
//   · 相机 FOV 由追踪包的 focal_px/h 推出，与 sidecar 的投影模型一致，人脸网格投影落在视频人脸上；
//   · 背景 quad 按同一 FOV 与摄像头宽高比铺满视锥；
//   · 镜像用投影矩阵 x 翻转 + GL.invertCulling 一处实现（视频与 3D 同步镜像）。
// 处理 apply_spec（assets 内嵌 base64 或 assets_url HTTP 下载）、clear_makeup、set_intensity、
// coaching（含 progress/step）、request_frame（异步）、ping；上报实测 tracking_state、intensity_changed。
using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace MakeupMirror
{
    public class MakeupAppMain : MonoBehaviour
    {
        [Header("可选：场景里已搭好的组件（留空则程序化创建）")]
        public WebcamDisplay webcam;
        public UdpLandmarkReceiver tracker;
        public FaceMeshDeformer deformer;
        public MakeupLayerRenderer makeup;
        public SplatLayerRenderer splat;
        public BridgeClient bridge;
        public CoachingDisplay coaching;
        public Canvas uiCanvas;

        [Header("显示")]
        [Tooltip("镜像显示（照镜子习惯）：投影矩阵翻转，视频与妆容一起镜像")]
        public bool mirror = true;
        [Tooltip("背景 quad 距离（米），需大于人脸距离")]
        public float backgroundDistance = 2.5f;

        private Camera _cam;
        private Transform _bgQuad;
        private Slider _slider;
        private Text _statusText;
        private string _statusBridge = "connecting";
        private string _statusTracking = "waiting";
        private float _lastTrackingPush;
        private float _appliedFocal = -1f;
        private int _appliedW, _appliedH;
        private bool _sliderSilent;
        private float _lastIntensitySent = -1f;
        private float _lastIntensitySentTime;
        private float _pendingIntensity = -1f;

        private void Awake()
        {
            EnsureComponents();
        }

        private void EnsureComponents()
        {
            // 相机：原点、朝 +Z、恒等旋转（世界=相机空间，shader 里的光方向按此约定）
            if (Camera.main == null)
            {
                var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
                camGo.AddComponent<Camera>();
            }
            _cam = Camera.main;
            _cam.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            _cam.clearFlags = CameraClearFlags.SolidColor;
            _cam.backgroundColor = new Color(0.06f, 0.06f, 0.08f);
            _cam.nearClipPlane = 0.05f;
            _cam.farClipPlane = 10f;
            _cam.fieldOfView = 40f;
            _cam.allowHDR = false;
            _cam.allowMSAA = true;

            // 摄像头背景
            if (webcam == null)
            {
                var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
                Destroy(quad.GetComponent<Collider>());
                quad.name = "WebcamBackground";
                quad.transform.position = new Vector3(0f, 0f, backgroundDistance);
                quad.transform.localScale = new Vector3(2.3f, 1.3f, 1f);
                var mr = quad.GetComponent<MeshRenderer>();
                mr.sharedMaterial = new Material(Shader.Find("MakeupMirror/WebcamBackground"));
                webcam = quad.AddComponent<WebcamDisplay>();
                webcam.background = mr;
            }
            _bgQuad = webcam.transform;

            if (tracker == null) tracker = gameObject.AddComponent<UdpLandmarkReceiver>();
            if (deformer == null)
            {
                var faceGo = new GameObject("FaceMesh");
                faceGo.transform.SetParent(transform, false);
                deformer = faceGo.AddComponent<FaceMeshDeformer>();
                deformer.tracker = tracker;
                deformer.meshFilter = faceGo.AddComponent<MeshFilter>();
                var faceMr = faceGo.AddComponent<MeshRenderer>();
                faceMr.enabled = false;   // 底网格不画：妆容层共享它的 mesh，各自带材质
            }
            if (makeup == null)
            {
                makeup = gameObject.AddComponent<MakeupLayerRenderer>();
                makeup.deformer = deformer;
            }
            if (splat == null)
            {
                splat = gameObject.AddComponent<SplatLayerRenderer>();
                splat.deformer = deformer;
                splat.viewCamera = _cam;
            }
            if (coaching == null) coaching = gameObject.AddComponent<CoachingDisplay>();
            if (uiCanvas == null) uiCanvas = BuildUI();
            if (bridge == null) bridge = gameObject.AddComponent<BridgeClient>();
        }

        // ---------- UI ----------

        private Canvas BuildUI()
        {
            var canvasGo = new GameObject("UI");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1280, 720);
            canvasGo.AddComponent<GraphicRaycaster>();
            if (FindObjectOfType<UnityEngine.EventSystems.EventSystem>() == null)
            {
                var es = new GameObject("EventSystem");
                es.transform.SetParent(canvasGo.transform, false);
                es.AddComponent<UnityEngine.EventSystems.EventSystem>();
                es.AddComponent<UnityEngine.EventSystems.StandaloneInputModule>();
            }
            var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

            // 状态栏（左上）
            _statusText = MakeText(canvasGo.transform, "Status", font, 18, TextAnchor.UpperLeft,
                new Vector2(0.02f, 0.93f), new Vector2(0.7f, 0.99f));
            _statusText.color = new Color(1f, 1f, 1f, 0.85f);

            // 指导字幕（底部居中）
            var text = MakeText(canvasGo.transform, "CoachingText", font, 30, TextAnchor.LowerCenter,
                new Vector2(0.05f, 0.16f), new Vector2(0.95f, 0.28f));
            coaching.Bind(text);

            // 步骤进度条（字幕上方）
            var barBg = MakePanel(canvasGo.transform, "ProgressBar", new Vector2(0.3f, 0.29f), new Vector2(0.7f, 0.305f),
                new Color(1f, 1f, 1f, 0.18f));
            var fillGo = new GameObject("Fill");
            fillGo.transform.SetParent(barBg.transform, false);
            var fillRect = fillGo.AddComponent<RectTransform>();
            fillRect.anchorMin = Vector2.zero; fillRect.anchorMax = Vector2.one;
            fillRect.offsetMin = fillRect.offsetMax = Vector2.zero;
            var fill = fillGo.AddComponent<Image>();
            fill.color = new Color(0.94f, 0.43f, 0.55f, 0.95f);
            fill.type = Image.Type.Filled;
            fill.fillMethod = Image.FillMethod.Horizontal;
            fill.sprite = null;
            var progressLabel = MakeText(barBg.transform, "ProgressLabel", font, 18, TextAnchor.LowerCenter,
                new Vector2(0f, 1f), new Vector2(1f, 1f));
            var plRect = progressLabel.GetComponent<RectTransform>();
            plRect.offsetMin = new Vector2(0f, 6f);
            plRect.offsetMax = new Vector2(0f, 34f);
            progressLabel.color = new Color(1f, 1f, 1f, 0.9f);
            coaching.BindProgress(fill, progressLabel);

            // 强度滑杆（左下）
            _slider = MakeSlider(canvasGo.transform, font);
            _slider.onValueChanged.AddListener(OnSliderChanged);
            return canvas;
        }

        private static Text MakeText(Transform parent, string name, Font font, int size, TextAnchor anchor,
                                     Vector2 amin, Vector2 amax)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var rect = go.AddComponent<RectTransform>();
            rect.anchorMin = amin; rect.anchorMax = amax;
            rect.offsetMin = rect.offsetMax = Vector2.zero;
            var text = go.AddComponent<Text>();
            text.alignment = anchor;
            text.fontSize = size;
            text.font = font;
            text.horizontalOverflow = HorizontalWrapMode.Wrap;
            text.verticalOverflow = VerticalWrapMode.Overflow;
            text.color = Color.white;
            text.text = "";
            var outline = go.AddComponent<Outline>();
            outline.effectColor = new Color(0f, 0f, 0f, 0.85f);
            outline.effectDistance = new Vector2(1.5f, -1.5f);
            return text;
        }

        private static Image MakePanel(Transform parent, string name, Vector2 amin, Vector2 amax, Color color)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var rect = go.AddComponent<RectTransform>();
            rect.anchorMin = amin; rect.anchorMax = amax;
            rect.offsetMin = rect.offsetMax = Vector2.zero;
            var img = go.AddComponent<Image>();
            img.color = color;
            return img;
        }

        private Slider MakeSlider(Transform parent, Font font)
        {
            var root = MakePanel(parent, "IntensitySlider", new Vector2(0.03f, 0.05f), new Vector2(0.3f, 0.075f),
                new Color(1f, 1f, 1f, 0.16f));
            var slider = root.gameObject.AddComponent<Slider>();
            slider.minValue = 0f; slider.maxValue = 1f; slider.value = 0.8f;
            slider.transition = Selectable.Transition.None;

            var fillArea = new GameObject("FillArea");
            fillArea.transform.SetParent(root.transform, false);
            var far = fillArea.AddComponent<RectTransform>();
            far.anchorMin = Vector2.zero; far.anchorMax = Vector2.one; far.offsetMin = far.offsetMax = Vector2.zero;
            var fillGo = new GameObject("Fill");
            fillGo.transform.SetParent(fillArea.transform, false);
            var fillRect = fillGo.AddComponent<RectTransform>();
            fillRect.anchorMin = Vector2.zero; fillRect.anchorMax = Vector2.one; fillRect.offsetMin = fillRect.offsetMax = Vector2.zero;
            var fillImg = fillGo.AddComponent<Image>();
            fillImg.color = new Color(0.94f, 0.43f, 0.55f, 0.95f);
            slider.fillRect = fillRect;

            var handleArea = new GameObject("HandleArea");
            handleArea.transform.SetParent(root.transform, false);
            var har = handleArea.AddComponent<RectTransform>();
            har.anchorMin = Vector2.zero; har.anchorMax = Vector2.one; har.offsetMin = har.offsetMax = Vector2.zero;
            var handleGo = new GameObject("Handle");
            handleGo.transform.SetParent(handleArea.transform, false);
            var hr = handleGo.AddComponent<RectTransform>();
            hr.sizeDelta = new Vector2(22f, 22f);
            var hImg = handleGo.AddComponent<Image>();
            hImg.color = Color.white;
            slider.handleRect = hr;
            slider.targetGraphic = hImg;

            var label = MakeText(parent, "IntensityLabel", font, 20, TextAnchor.LowerLeft,
                new Vector2(0.03f, 0.08f), new Vector2(0.3f, 0.12f));
            label.text = "妆感 80%";
            _intensityLabel = label;
            return slider;
        }

        private Text _intensityLabel;

        private void OnSliderChanged(float v)
        {
            if (_intensityLabel != null) _intensityLabel.text = $"妆感 {Mathf.RoundToInt(v * 100)}%";
            if (_sliderSilent) return;
            makeup.SetIntensity(v);
            _pendingIntensity = v;   // 节流上报（Update 里发）
        }

        private void SetSliderSilently(float v)
        {
            if (_slider == null) return;
            _sliderSilent = true;
            _slider.value = Mathf.Clamp01(v);
            _sliderSilent = false;
        }

        // ---------- 生命周期 ----------

        private void OnEnable()
        {
            bridge.OnMessage += HandleMessage;
            bridge.OnConnectionChanged += HandleConnectionChanged;
            tracker.TrackingChanged += HandleTrackingChanged;
        }

        private void OnDisable()
        {
            bridge.OnMessage -= HandleMessage;
            bridge.OnConnectionChanged -= HandleConnectionChanged;
            tracker.TrackingChanged -= HandleTrackingChanged;
        }

        private void Start()
        {
            RefreshProjection(_cam.fieldOfView);
            FitBackground(16f / 9f);
        }

        private void HandleConnectionChanged(bool v) => _statusBridge = v ? "connected" : "reconnecting";
        private void HandleTrackingChanged(bool v) => _statusTracking = v ? "ok" : "no-face";

        private void Update()
        {
            // FOV / 背景与追踪包几何联动（focal 或分辨率变化时刷新一次）
            if (tracker.TryGetLatest(out var frame) && frame.faceOk && !frame.legacy && frame.focalPx > 1f
                && (Mathf.Abs(frame.focalPx - _appliedFocal) > 0.5f || frame.width != _appliedW || frame.height != _appliedH))
            {
                _appliedFocal = frame.focalPx; _appliedW = frame.width; _appliedH = frame.height;
                float vfov = 2f * Mathf.Atan(frame.height / (2f * frame.focalPx)) * Mathf.Rad2Deg;
                RefreshProjection(vfov);
                FitBackground(frame.width / (float)Mathf.Max(1, frame.height));
            }

            webcam.PublishAmbientGlobals(mirror);

            if (_statusText != null)
            {
                string pose = frame.hasPose ? "pose ✓" : (frame.legacy ? "legacy" : "no-pose");
                _statusText.text = $"bridge: {_statusBridge}   tracking: {_statusTracking} · {tracker.MeasuredFps:F0} fps · "
                                   + $"{tracker.LastLandmarkCount} pts · {pose}"
                                   + (makeup.CurrentSpecName != null ? $"   妆容: {makeup.CurrentSpecName}" : "")
                                   + (splat.Count > 0 ? $" · splat {splat.Count}" : "");
            }

            if (!bridge.Connected) return;
            // tracking_state 节流上报（实测 fps / 点数）
            if (Time.time - _lastTrackingPush > 1f)
            {
                _lastTrackingPush = Time.time;
                _ = bridge.SendJson(new JObject
                {
                    ["type"] = "tracking_state",
                    ["ok"] = deformer.HasFace,
                    ["fps"] = Mathf.Round(tracker.MeasuredFps * 10f) / 10f,
                    ["landmarks"] = tracker.LastLandmarkCount,
                    ["pose"] = frame.hasPose,
                });
            }
            // intensity_changed 节流上报（拖滑杆）
            if (_pendingIntensity >= 0f && Time.time - _lastIntensitySentTime > 0.25f
                && Mathf.Abs(_pendingIntensity - _lastIntensitySent) > 0.004f)
            {
                _lastIntensitySentTime = Time.time;
                _lastIntensitySent = _pendingIntensity;
                _ = bridge.SendJson(new JObject { ["type"] = "intensity_changed", ["value"] = Mathf.Round(_pendingIntensity * 100f) / 100f });
                _pendingIntensity = -1f;
            }
        }

        /// 相机 FOV + 镜像投影（投影矩阵 x 翻转；GL.invertCulling 让背面剔除仍正确）
        private void RefreshProjection(float vfovDeg)
        {
            _cam.ResetProjectionMatrix();
            _cam.fieldOfView = Mathf.Clamp(vfovDeg, 15f, 120f);
            var p = _cam.projectionMatrix;
            if (mirror) p = Matrix4x4.Scale(new Vector3(-1f, 1f, 1f)) * p;
            _cam.projectionMatrix = p;
            GL.invertCulling = mirror;
        }

        /// 背景 quad 铺满视锥（按摄像头宽高比，视锥外留黑边而不拉伸）
        private void FitBackground(float aspect)
        {
            if (_bgQuad == null) return;
            float h = 2f * backgroundDistance * Mathf.Tan(_cam.fieldOfView * 0.5f * Mathf.Deg2Rad);
            _bgQuad.position = new Vector3(0f, 0f, backgroundDistance);
            _bgQuad.rotation = Quaternion.identity;
            _bgQuad.localScale = new Vector3(h * aspect, h, 1f);
        }

        // ---------- Bridge 消息 ----------

        private async void HandleMessage(JObject msg)
        {
            string type = msg.Value<string>("type");
            string reference = msg.Value<string>("ref");
            try
            {
                switch (type)
                {
                    case "apply_spec":
                        await ApplySpec(msg);
                        await bridge.SendAck(reference, true);
                        break;
                    case "clear_makeup":
                        makeup.Clear();
                        splat.Clear();
                        coaching.SetProgress(null, 0, 0, 0f);
                        await bridge.SendAck(reference, true);
                        break;
                    case "set_intensity":
                    {
                        float v = msg.Value<float?>("value") ?? 0.8f;
                        makeup.SetIntensity(v);
                        SetSliderSilently(v);
                        await bridge.SendAck(reference, true);
                        break;
                    }
                    case "coaching":
                        coaching.Show(msg.Value<string>("text"),
                                      msg.Value<string>("priority") ?? "info",
                                      msg.Value<bool?>("speak") ?? false,
                                      msg.Value<string>("note"));
                        if (msg["progress"] != null)
                            coaching.SetProgress(msg.Value<string>("step_name") ?? msg.Value<string>("area"),
                                                 msg.Value<int?>("step") ?? 0, msg.Value<int?>("steps") ?? 0,
                                                 msg.Value<float?>("progress") ?? 0f);
                        break;
                    case "request_frame":
                    {
                        byte[] jpg = await CaptureAsync(msg.Value<int?>("quality") ?? 80);
                        if (jpg == null)
                        {
                            await bridge.SendAck(reference, false, "webcam_not_ready");
                            break;
                        }
                        await bridge.SendAck(reference, true);
                        _ = bridge.SendJson(new JObject
                        {
                            ["type"] = "frame",
                            ["data"] = Convert.ToBase64String(jpg),
                            ["w"] = webcam.Texture != null ? webcam.Texture.width : 0,
                            ["h"] = webcam.Texture != null ? webcam.Texture.height : 0,
                        });
                        break;
                    }
                    case "ping":
                        await bridge.SendAck(reference, true);
                        break;
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[bridge] 处理 {type} 失败：{e}");
                if (reference != null)
                    await bridge.SendAck(reference, false, e.Message);
            }
        }

        private Task<byte[]> CaptureAsync(int quality)
        {
            var tcs = new TaskCompletionSource<byte[]>();
            webcam.CaptureJpegAsync(quality, 960, b => tcs.TrySetResult(b));
            return tcs.Task;
        }

        private async Task ApplySpec(JObject msg)
        {
            var spec = (JObject)msg["spec"];
            var assets = new Dictionary<string, byte[]>();
            if (msg["assets"] is JObject assetsObj)
                foreach (var kv in assetsObj)
                    assets[kv.Key] = Convert.FromBase64String(kv.Value.Value<string>());
            if (msg["assets_url"] is JObject urls)
                foreach (var kv in urls)
                {
                    var bytes = await DownloadAsync(kv.Value.Value<string>());
                    if (bytes != null) assets[kv.Key] = bytes;
                    else Debug.LogWarning($"[bridge] 资产下载失败：{kv.Key}（用程序化兜底）");
                }

            await makeup.Apply(spec, assets);
            var intenTok = msg["intensity"] ?? spec["intensity"];
            float inten = intenTok?.Value<float>() ?? 0.8f;
            makeup.SetIntensity(inten);
            SetSliderSilently(inten);
            splat.Apply(spec["splat_layers"] as JArray);
        }

        private Task<byte[]> DownloadAsync(string url)
        {
            var tcs = new TaskCompletionSource<byte[]>();
            StartCoroutine(DownloadCoroutine(url, tcs));
            return tcs.Task;
        }

        private IEnumerator DownloadCoroutine(string url, TaskCompletionSource<byte[]> tcs)
        {
            using (var req = UnityWebRequest.Get(url))
            {
                req.timeout = 15;
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success)
                    tcs.TrySetResult(req.downloadHandler.data);
                else
                    tcs.TrySetResult(null);
            }
        }
    }
}
