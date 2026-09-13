// WebcamDisplay.cs
// 摄像头取流（WebCamTexture 铺满背景 quad）+ 三个运行时服务：
//   1) FrameRelay：把降采样帧经 UDP 分片发给 face_tracker sidecar（单摄像头闭环），
//      追踪用的帧和显示用的帧同源，姿态对齐才精确；
//   2) CaptureJpegAsync：异步截帧回传 Bridge（AsyncGPUReadback，不卡主线程）；
//   3) Ambient：周期估计环境光（色调/亮度图/主光方向），供妆容 shader 光影合成。
using System;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using Debug = UnityEngine.Debug;

namespace MakeupMirror
{
    public class WebcamDisplay : MonoBehaviour
    {
        [Tooltip("背景 quad 的 MeshRenderer（用 MakeupMirror/WebcamBackground 材质）")]
        public MeshRenderer background;
        [Tooltip("镜像显示（照镜子习惯）。默认由主入口用相机投影翻转实现，这里保持 false")]
        public bool mirror = false;
        public int requestedCameraIndex = 0;
        [Header("相机请求分辨率（设备不支持时由驱动降级）")]
        public int requestedWidth = 1920;
        public int requestedHeight = 1080;
        public int requestedFps = 30;

        [Header("帧中继（发给 face_tracker --relay）")]
        public bool relayEnabled = true;
        public int relayPort = 8767;
        [Tooltip("每 N 帧采样一次")]
        public int relayEveryNFrames = 2;
        public int relayWidth = 640;
        [Range(40, 90)] public int relayQuality = 65;

        [Header("环境光估计")]
        public bool ambientEnabled = true;
        public float ambientInterval = 0.4f;

        public WebCamTexture Texture { get; private set; }
        public bool IsReady => Texture != null && Texture.isPlaying && Texture.didUpdateThisFrame;

        /// 最近一次环境光估计（无效时 Valid=false）
        public struct Ambient
        {
            public bool Valid;
            public Color Tint;        // 色调（归一化到平均亮度=1）
            public float Luminance;   // 0~1 平均亮度
            public Vector3 LightDir;  // 世界（=视图）空间，指向光源
        }

        public Ambient LatestAmbient => _ambient;
        public int RelayFramesSent => _relaySent;

        private Material _mat;
        private Ambient _ambient;
        private UdpClientRelay _relay;
        private RenderTexture _relayRt, _probeRt;
        private Texture2D _lumTex;
        private float _lastAmbientTime;
        private int _frameCounter, _relaySent;
        private bool _relayBusy;
        private byte[] _jpeg;
        private readonly object _jpegLock = new object();

        private void Start()
        {
            _mat = background != null ? background.material : null;
            if (WebCamTexture.devices.Length == 0)
            {
                Debug.LogError("[webcam] 没有可用摄像头");
                return;
            }
            string name = WebCamTexture.devices[Mathf.Clamp(requestedCameraIndex, 0, WebCamTexture.devices.Length - 1)].name;
            Texture = new WebCamTexture(name, requestedWidth, requestedHeight, requestedFps);
            Texture.Play();
            if (_mat != null)
            {
                _mat.mainTexture = Texture;
                _mat.SetFloat("_Mirror", 0f);   // 镜像统一由主入口的相机投影翻转实现
            }
            if (relayEnabled) _relay = new UdpClientRelay("127.0.0.1", relayPort);
            _lumTex = new Texture2D(32, 18, TextureFormat.RGBA32, false, true)
            { wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
            Debug.Log($"[webcam] 使用摄像头：{name}（请求 {requestedWidth}x{requestedHeight}，实际 {Texture.width}x{Texture.height}，{WebCamTexture.devices.Length} 个可用）");
        }

        private void OnDestroy()
        {
            if (Texture != null) { Texture.Stop(); Destroy(Texture); }
            _relay?.Dispose();
            DestroyRenderTex(ref _relayRt);
            DestroyRenderTex(ref _probeRt);
            if (_lumTex != null) Destroy(_lumTex);
        }

        private static void DestroyRenderTex(ref RenderTexture rt)
        {
            if (rt != null) { rt.Release(); rt = null; }
        }

        private void Update()
        {
            if (Texture == null || !Texture.isPlaying || Texture.width == 0) return;
            _frameCounter++;

            // 1) 帧中继
            if (_relay != null && relayEveryNFrames > 0 && _frameCounter % relayEveryNFrames == 0 && !_relayBusy)
                StartCoroutine(RelayOneFrame());

            // 2) 环境光估计
            if (ambientEnabled && Time.time - _lastAmbientTime > ambientInterval)
            {
                _lastAmbientTime = Time.time;
                StartCoroutine(ProbeAmbient());
            }
        }

        // ---------- 帧中继 ----------

        private IEnumerator RelayOneFrame()
        {
            _relayBusy = true;
            int tw = relayWidth, th = Mathf.Max(2, Mathf.RoundToInt(relayWidth * (float)Texture.height / Texture.width));
            EnsureRt(ref _relayRt, tw, th);
            Graphics.Blit(Texture, _relayRt);
            var done = false;
            AsyncGPUReadback.Request(_relayRt, 0, TextureFormat.RGB24, r =>
            {
                if (!r.hasError)
                {
                    var raw = r.GetData<byte>().ToArray();   // 主线程拷出
                    lock (_jpegLock) { _jpeg = raw; }
                }
                done = true;
            });
            yield return new WaitUntil(() => done);
            int w = tw, h = th;
            byte[] rawCopy;
            lock (_jpegLock) { rawCopy = _jpeg; _jpeg = null; }
            if (rawCopy != null)
            {
                // 编码在主线程（640 宽 ≈2ms，丢帧不排队）
                byte[] jpg = ImageConversion.EncodeArrayToJPG(rawCopy, GraphicsFormat.R8G8B8_UNorm,
                    (uint)w, (uint)h, (uint)(w * 3), relayQuality);
                if (jpg != null && jpg.Length > 0) _relay.Send(jpg, w, h);
                _relaySent++;
            }
            _relayBusy = false;
        }

        // ---------- 环境光 ----------

        private IEnumerator ProbeAmbient()
        {
            EnsureRt(ref _probeRt, 32, 18);
            Graphics.Blit(Texture, _probeRt);
            var done = false;
            AsyncGPUReadback.Request(_probeRt, 0, TextureFormat.RGBA32, r =>
            {
                if (!r.hasError) ApplyAmbient(r);
                done = true;
            });
            yield return new WaitUntil(() => done);
        }

        private void ApplyAmbient(AsyncGPUReadbackRequest r)
        {
            try
            {
                var data = r.GetData<Color32>().ToArray();
                if (data.Length < 32 * 18) return;
                _lumTex.SetPixelData(data, 0);
                _lumTex.Apply(false);

                Vector3 sum = Vector3.zero;
                float wsum = 0f;
                foreach (var c in data)
                {
                    float lum = (0.299f * c.r + 0.587f * c.g + 0.114f * c.b) / 255f;
                    sum += new Vector3(c.r, c.g, c.b) * lum;
                    wsum += lum;
                }
                if (wsum < 1e-3f) return;
                var tint = new Color(sum.x / wsum / 255f, sum.y / wsum / 255f, sum.z / wsum / 255f);
                float lumMean = (tint.r + tint.g + tint.b) / 3f;
                tint /= Mathf.Max(lumMean, 1e-3f);          // 平均亮度归一 → 纯色调

                // 亮度质心 → 主光水平/竖直方向（画面里亮的一侧即来光方向）
                float cx = 0f, cy = 0f;
                for (int y = 0; y < 18; y++)
                    for (int x = 0; x < 32; x++)
                    {
                        var c = data[y * 32 + x];
                        float lum = (0.299f * c.r + 0.587f * c.g + 0.114f * c.b) / 255f;
                        cx += x * lum; cy += y * lum;
                    }
                cx = cx / wsum / 32f; cy = cy / wsum / 18f;
                // RT 行 0 = 图像底部（GPU 内存布局）→ y 已是向上语义；视图空间 x 右 y 上，光大致来自相机方向
                var lightDir = new Vector3((cx - 0.5f) * 1.6f, (cy - 0.5f) * 1.2f, -0.75f).normalized;

                _ambient = new Ambient
                {
                    Valid = true,
                    Tint = tint,
                    Luminance = lumMean,
                    LightDir = lightDir,
                };
            }
            catch (Exception) { /* 请求在销毁后完成等场景，忽略 */ }
        }

        private void EnsureRt(ref RenderTexture rt, int w, int h)
        {
            if (rt != null && (rt.width != w || rt.height != h)) DestroyRenderTex(ref rt);
            if (rt == null)
                rt = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32) { filterMode = FilterMode.Bilinear };
        }

        /// 发布环境光全局量（主入口每帧调用）：_EnvTint/_EnvLumTex/_LightDirWorld/_EnvMirror
        public void PublishAmbientGlobals(bool mirror)
        {
            if (!_ambient.Valid) return;
            Shader.SetGlobalVector("_EnvTint", _ambient.Tint);
            Shader.SetGlobalTexture("_EnvLumTex", _lumTex);
            Shader.SetGlobalVector("_LightDirWorld", _ambient.LightDir);
            Shader.SetGlobalFloat("_EnvLuminance", _ambient.Luminance);
            Shader.SetGlobalFloat("_EnvMirror", mirror ? 1f : 0f);
        }

        // ---------- 截帧回传 ----------

        /// 异步截当前帧为 JPEG（AsyncGPUReadback + 主线程编码，不卡渲染）
        public void CaptureJpegAsync(int quality, int maxWidth, Action<byte[]> onDone)
        {
            if (Texture == null || !Texture.isPlaying || Texture.width == 0)
            {
                onDone?.Invoke(null);
                return;
            }
            StartCoroutine(CaptureCoroutine(quality, maxWidth, onDone));
        }

        private IEnumerator CaptureCoroutine(int quality, int maxWidth, Action<byte[]> onDone)
        {
            int w = Texture.width, h = Texture.height;
            float scale = Mathf.Min(1f, maxWidth / (float)w);
            int tw = Mathf.RoundToInt(w * scale), th = Mathf.RoundToInt(h * scale);
            var rt = RenderTexture.GetTemporary(tw, th, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(Texture, rt);
            byte[] result = null;
            var done = false;
            AsyncGPUReadback.Request(rt, 0, TextureFormat.RGB24, r =>
            {
                try
                {
                    if (!r.hasError)
                    {
                        var raw = r.GetData<byte>().ToArray();
                        result = ImageConversion.EncodeArrayToJPG(raw, GraphicsFormat.R8G8B8_UNorm,
                            (uint)tw, (uint)th, (uint)(tw * 3), quality);
                    }
                }
                catch (Exception) { result = null; }
                done = true;
            });
            yield return new WaitUntil(() => done);
            RenderTexture.ReleaseTemporary(rt);
            onDone?.Invoke(result);
        }

        // ---------- UDP 分片发送 ----------

        /// 把 JPEG 帧按 ≤16000B 分片发往 sidecar（协议 magic "MKF1"，见 tracking_protocol.py）
        private class UdpClientRelay : IDisposable
        {
            private Socket _sock;
            private readonly IPEndPoint _target;
            private uint _frameId;
            public int Dropped { get; private set; }

            public UdpClientRelay(string host, int port)
            {
                _sock = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                _target = new IPEndPoint(IPAddress.Parse(host), port);
            }

            public void Send(byte[] jpeg, int w, int h)
            {
                if (_sock == null || jpeg == null || jpeg.Length == 0) return;
                const int payload = 16000;
                int count = (jpeg.Length + payload - 1) / payload;
                if (count > 65000) { Dropped++; return; }
                var header = new byte[20];
                header[0] = (byte)'M'; header[1] = (byte)'K'; header[2] = (byte)'F'; header[3] = (byte)'1';
                WriteU32(header, 4, ++_frameId);
                WriteU16(header, 8, 0); WriteU16(header, 10, (ushort)count);
                WriteU16(header, 12, (ushort)w); WriteU16(header, 14, (ushort)h);
                WriteU32(header, 16, (uint)jpeg.Length);
                try
                {
                    for (int i = 0; i < count; i++)
                    {
                        WriteU16(header, 8, (ushort)i);
                        int len = Math.Min(payload, jpeg.Length - i * payload);
                        var pkt = new byte[20 + len];
                        Buffer.BlockCopy(header, 0, pkt, 0, 20);
                        Buffer.BlockCopy(jpeg, i * payload, pkt, 20, len);
                        _sock.SendTo(pkt, _target);
                    }
                }
                catch (SocketException) { Dropped++; }
                catch (ObjectDisposedException) { }
            }

            private static void WriteU16(byte[] b, int off, ushort v)
            { b[off] = (byte)v; b[off + 1] = (byte)(v >> 8); }

            private static void WriteU32(byte[] b, int off, uint v)
            {
                b[off] = (byte)v; b[off + 1] = (byte)(v >> 8);
                b[off + 2] = (byte)(v >> 16); b[off + 3] = (byte)(v >> 24);
            }

            public void Dispose()
            {
                try { _sock?.Close(); } catch { }
                _sock = null;
            }
        }
    }
}
