// UdpLandmarkReceiver.cs
// 接收 face_tracker.py sidecar 的追踪数据流（UDP 127.0.0.1:8766）。
// 协议以 unity-app/tools/tracking_protocol.py 为准：
//   v2 二进制 "MKT2"：header + 4×4 姿态（厘米，OpenGL 相机系）+ 头部局部关键点（厘米）+ 图像坐标
//   v1 JSON（兼容）：{"ok","w","h","t","pts":[[x,y,z]×468]} 归一化图像坐标
// 后台线程收包解析，主线程 TryGetLatest 取最新帧；含帧龄/乱序检查。
// 坐标系转换在此完成：OpenGL(z 朝相机) → Unity(z 朝前)，厘米 → 米。
using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json.Linq;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace MakeupMirror
{
    public struct LandmarkFrame
    {
        public bool faceOk;
        public bool hasPose;          // v2 且 sidecar 求解了姿态
        public bool legacy;           // v1：points 为归一化图像坐标（x右 y下），无姿态
        public bool relaySource;
        public uint seq;
        public int width;
        public int height;
        public float focalPx;         // 像素焦距（决定相机 FOV）
        public double time;           // sidecar 时间戳
        public long recvTicks;        // Stopwatch 时戳（帧龄）
        public Vector3[] points;      // v2：头部局部坐标（Unity 米，(x,y,-z)×0.01）；v1：归一化图像坐标
        public Vector2[] imagePoints; // v2：归一化图像坐标（2D 叠加用）
        public Matrix4x4 pose;        // v2：Unity 相机空间（米）刚体变换
    }

    public class UdpLandmarkReceiver : MonoBehaviour
    {
        [Tooltip("face_tracker.py 的 UDP 端口")]
        public int port = 8766;
        [Tooltip("超过此帧龄的追踪包视为过期（秒）")]
        public float maxFrameAge = 0.25f;

        public bool TrackingOk { get; private set; }
        public event Action<bool> TrackingChanged;

        /// 最近 1 秒收到的有效追踪包数（tracking_state 上报用）
        public float MeasuredFps { get; private set; }
        public int LastLandmarkCount { get; private set; }
        public uint DroppedOutOfOrder { get; private set; }

        private UdpClient _udp;
        private Thread _thread;
        private volatile bool _running;
        private readonly object _lock = new object();
        private LandmarkFrame _latest;
        private bool _hasFrame;
        private uint _lastSeq;
        private bool _seqInit;
        private int _fpsCounter;
        private long _fpsWindowStart;

        private static readonly byte[] MagicV2 = Encoding.ASCII.GetBytes("MKT2");
        private const int HeaderSize = 30;
        private const ushort FlagFace = 1, FlagPose = 2, FlagRelay = 4;

        private void OnEnable() => StartReceive();
        private void OnDisable() => StopReceive();

        private void StartReceive()
        {
            try
            {
                _udp = new UdpClient(port);
            }
            catch (SocketException e)
            {
                Debug.LogError($"[tracker] UDP {port} 占用失败（sidecar 未启动或重复绑定）：{e.Message}");
                return;
            }
            _running = true;
            _fpsWindowStart = Stopwatch.GetTimestamp();
            _thread = new Thread(ReceiveLoop) { IsBackground = true, Name = "udp-landmarks" };
            _thread.Start();
            Debug.Log($"[tracker] 监听 udp://127.0.0.1:{port}（MKT2 二进制 / v1 JSON 自适应）");
        }

        private void StopReceive()
        {
            _running = false;
            try { _udp?.Close(); } catch { }
            _thread?.Join(500);
            _udp = null;
            _thread = null;
        }

        private void ReceiveLoop()
        {
            var remote = new IPEndPoint(IPAddress.Loopback, 0);
            while (_running)
            {
                try
                {
                    byte[] data = _udp.Receive(ref remote);
                    LandmarkFrame frame = IsV2(data) ? ParseV2(data) : ParseV1(data);
                    frame.recvTicks = Stopwatch.GetTimestamp();

                    if (_seqInit && frame.seq != 0 && frame.seq < _lastSeq && _lastSeq - frame.seq < 1000)
                    {
                        DroppedOutOfOrder++;
                        continue;   // 乱序旧包丢弃
                    }
                    _seqInit = true;
                    _lastSeq = frame.seq;

                    lock (_lock)
                    {
                        _latest = frame;
                        _hasFrame = true;
                    }
                    if (frame.faceOk)
                    {
                        _fpsCounter++;
                        LastLandmarkCount = frame.points?.Length ?? 0;
                    }
                    long now = Stopwatch.GetTimestamp();
                    double win = (now - _fpsWindowStart) / (double)Stopwatch.Frequency;
                    if (win >= 1.0)
                    {
                        MeasuredFps = (float)(_fpsCounter / win);
                        _fpsCounter = 0;
                        _fpsWindowStart = now;
                    }
                    if (frame.faceOk != TrackingOk)
                    {
                        TrackingOk = frame.faceOk;
                        TrackingChanged?.Invoke(frame.faceOk);
                    }
                }
                catch (SocketException) { break; }
                catch (ObjectDisposedException) { break; }
                catch (Exception e) { if (_running) Debug.LogWarning($"[tracker] 包解析失败：{e.Message}"); }
            }
        }

        private static bool IsV2(byte[] d) =>
            d.Length >= HeaderSize && d[0] == MagicV2[0] && d[1] == MagicV2[1] && d[2] == MagicV2[2] && d[3] == MagicV2[3];

        // ---------- v2 二进制 ----------

        private static LandmarkFrame ParseV2(byte[] d)
        {
            // <4sHIdHHfHH
            ushort flags = BitConverter.ToUInt16(d, 4);
            uint seq = BitConverter.ToUInt32(d, 6);
            double t = BitConverter.ToDouble(d, 10);
            int w = BitConverter.ToUInt16(d, 18);
            int h = BitConverter.ToUInt16(d, 20);
            float focal = BitConverter.ToSingle(d, 22);
            int n = BitConverter.ToUInt16(d, 26);
            var f = new LandmarkFrame
            {
                faceOk = (flags & FlagFace) != 0,
                hasPose = (flags & FlagPose) != 0,
                relaySource = (flags & FlagRelay) != 0,
                seq = seq, time = t, width = w, height = h, focalPx = focal,
                pose = Matrix4x4.identity,
            };
            if (!f.faceOk || n == 0) return f;
            int need = HeaderSize + 64 + n * 12 + n * 8;
            if (d.Length < need) throw new Exception($"MKT2 截断 {d.Length} < {need}");

            int off = HeaderSize;
            var m = new float[16];
            Buffer.BlockCopy(d, off, m, 0, 64);
            off += 64;
            f.pose = GlToUnity(m);

            var raw = new float[n * 3];
            Buffer.BlockCopy(d, off, raw, 0, n * 12);
            off += n * 12;
            var pts = new Vector3[n];
            for (int i = 0; i < n; i++)
                pts[i] = new Vector3(raw[i * 3] * 0.01f, raw[i * 3 + 1] * 0.01f, -raw[i * 3 + 2] * 0.01f);
            f.points = pts;

            var raw2 = new float[n * 2];
            Buffer.BlockCopy(d, off, raw2, 0, n * 8);
            var img = new Vector2[n];
            for (int i = 0; i < n; i++) img[i] = new Vector2(raw2[i * 2], raw2[i * 2 + 1]);
            f.imagePoints = img;
            return f;
        }

        /// 行主序 4×4（OpenGL 相机系：z 朝相机，厘米）→ Unity（z 朝前，米）。
        /// M_unity = F·M·F，F=diag(1,1,-1)：第 2 行/第 2 列元素变号（对角不变），平移 z 变号。
        private static Matrix4x4 GlToUnity(float[] m)
        {
            var u = new Matrix4x4();
            u.m00 = m[0]; u.m01 = m[1]; u.m02 = -m[2]; u.m03 = m[3] * 0.01f;
            u.m10 = m[4]; u.m11 = m[5]; u.m12 = -m[6]; u.m13 = m[7] * 0.01f;
            u.m20 = -m[8]; u.m21 = -m[9]; u.m22 = m[10]; u.m23 = -m[11] * 0.01f;
            u.m30 = 0f; u.m31 = 0f; u.m32 = 0f; u.m33 = 1f;
            return u;
        }

        // ---------- v1 JSON（兼容） ----------

        private static LandmarkFrame ParseV1(byte[] d)
        {
            var json = JObject.Parse(Encoding.UTF8.GetString(d));
            var f = new LandmarkFrame
            {
                legacy = true,
                faceOk = json.Value<bool>("ok"),
                width = json.Value<int>("w"),
                height = json.Value<int>("h"),
                time = json.Value<double>("t"),
                pose = Matrix4x4.identity,
            };
            if (f.faceOk && json["pts"] is JArray pts)
            {
                var v = new Vector3[pts.Count];
                for (int i = 0; i < pts.Count; i++)
                {
                    var p = (JArray)pts[i];
                    v[i] = new Vector3(p[0].Value<float>(), p[1].Value<float>(), p[2].Value<float>());
                }
                f.points = v;
            }
            return f;
        }

        /// 主线程取最新一帧（无更新时返回 false；过期帧 faceOk 置 false）
        public bool TryGetLatest(out LandmarkFrame frame)
        {
            lock (_lock)
            {
                if (!_hasFrame) { frame = default; return false; }
                frame = _latest;
            }
            double age = (Stopwatch.GetTimestamp() - frame.recvTicks) / (double)Stopwatch.Frequency;
            if (age > maxFrameAge && frame.faceOk)
                frame.faceOk = false;   // sidecar 停了/卡了：视为丢脸，交给 deformer 淡出
            return true;
        }
    }
}
