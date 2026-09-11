// FaceMeshDeformer.cs
// 用 sidecar 追踪数据驱动 canonical 人脸网格。姿态/表情解耦（v2 协议）：
//   根节点变换 = 追踪包 4×4 姿态（相机空间刚体：位置+朝向+距离），
//   网格顶点   = 头部局部关键点（表情层：张嘴/皱眉时唇形眼形跟随）。
// 兼容 v1（归一化图像点 → 相机前固定平面映射）。零每帧 GC；1€ 滤波消抖；丢脸整体淡出。
using System.Diagnostics;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace MakeupMirror
{
    public class FaceMeshDeformer : MonoBehaviour
    {
        [Header("引用")]
        public UdpLandmarkReceiver tracker;
        public MeshFilter meshFilter;

        [Header("平滑（1€ 滤波）")]
        [Tooltip("稳定：静止更平滑；敏捷：跟随更快")]
        [Range(0f, 1f)] public float responsiveness = 0.5f;

        [Header("v1 兼容映射（旧 sidecar 无姿态时）")]
        public float faceDistance = 0.55f;
        public float scaleX = 0.42f;
        public float scaleY = 0.42f;
        public float scaleZ = 0.42f;

        [Tooltip("无脸帧后开始淡出的延迟（秒）")]
        public float holdSeconds = 0.35f;
        [Tooltip("淡出时长（秒）")]
        public float fadeSeconds = 0.5f;

        /// 妆容/溅射层读这个做整体淡入淡出（0=完全消失）
        public float FacePresence { get; private set; } = 0f;
        public bool HasFace { get; private set; }
        public Vector3[] CurrentWorldPositions { get; private set; }
        public Vector3[] CurrentWorldNormals { get; private set; }

        private Mesh _mesh;
        private Vector3[] _baseVerts;         // canonical 原始顶点（无追踪时兜底显示）
        private Vector2[] _uvs;
        private CanonicalFaceModel _model;

        private Vector3[] _verts;             // 复用顶点缓冲（零 GC）
        private readonly System.Collections.Generic.List<Vector3> _normalList =
            new System.Collections.Generic.List<Vector3>(512);
        private Matrix4x4 _localToWorld;

        private OneEuroCloud _cloud;
        private PoseSmoother _poseSm;
        private double _lastFaceTime;
        private float _smoothMin, _smoothBeta;

        private void Start()
        {
            _model = CanonicalFaceModel.Load();
            if (_model == null || tracker == null) { enabled = false; return; }

            _mesh = Instantiate(_model.Mesh);
            _mesh.name = "FaceMesh_live";
            meshFilter.sharedMesh = _mesh;
            _baseVerts = _model.CanonicalVertices;
            _uvs = _model.UVs;

            int n = _mesh.vertexCount;
            _verts = new Vector3[n];
            CurrentWorldPositions = new Vector3[n];
            CurrentWorldNormals = new Vector3[n];
            System.Array.Copy(_baseVerts, _verts, n);

            // 初始摆在相机前（canonical 正面朝 -Z，即朝向原点处的相机）
            transform.position = new Vector3(0f, 0f, faceDistance);
            transform.rotation = Quaternion.LookRotation(Vector3.back, Vector3.up);
            RecomputeWorld();
        }

        private void Update()
        {
            UpdateSmoothingParams();
            if (!tracker.TryGetLatest(out var frame))
                return;

            if (!frame.faceOk)
            {
                if (HasFace && Time.realtimeSinceStartupAsDouble - _lastFaceTime > holdSeconds)
                    HasFace = false;
            }
            else
            {
                _lastFaceTime = Time.realtimeSinceStartupAsDouble;
                if (!HasFace) { _cloud?.Reset(); _poseSm.Reset(); }
                HasFace = true;

                float t = (float)Time.realtimeSinceStartupAsDouble;
                if (frame.legacy || !frame.hasPose)
                    ApplyLegacy(frame, t);
                else
                    ApplyPose(frame, t);
            }

            // 丢脸淡出 / 找回淡入
            float target = HasFace ? 1f : 0f;
            float speed = HasFace ? 4f : 1f / Mathf.Max(0.05f, fadeSeconds);
            FacePresence = Mathf.MoveTowards(FacePresence, target, speed * Time.deltaTime);
            if (FacePresence <= 0f && !HasFace) return;   // 全隐没后不再算世界坐标
            RecomputeWorld();
        }

        private void UpdateSmoothingParams()
        {
            if (_cloud == null || !Mathf.Approximately(_smoothMin, responsiveness))
            {
                _smoothMin = responsiveness;
                // responsiveness 0→稳（min_cutoff 低），1→跟手（min_cutoff 高）
                float minCutoff = Mathf.Lerp(0.6f, 2.2f, responsiveness);
                float beta = Mathf.Lerp(0.02f, 0.1f, responsiveness);
                _cloud = new OneEuroCloud(minCutoff, beta);
                _poseSm = new PoseSmoother();
            }
        }

        /// v2 姿态路径：根节点 = 姿态矩阵，顶点 = 头部局部点（表情）
        private void ApplyPose(LandmarkFrame frame, float t)
        {
            int n = Mathf.Min(_verts.Length, frame.points.Length);
            var src = frame.points;       // 接收器缓存不可就地改，先拷入自有缓冲再滤波
            for (int i = 0; i < n; i++) _verts[i] = src[i];
            for (int i = n; i < _verts.Length; i++) _verts[i] = _baseVerts[i];
            _cloud.Filter(_verts, t);

            Vector3 pos = frame.pose.GetColumn(3);
            Quaternion rot = Quaternion.LookRotation(frame.pose.GetColumn(2), frame.pose.GetColumn(1));
            (pos, rot) = _poseSm.Filter(pos, rot, t);
            transform.SetPositionAndRotation(pos, rot);
            WriteVertices();
        }

        /// v1 兼容路径：归一化图像点 → 相机前固定平面（无姿态解算的老 sidecar）
        private void ApplyLegacy(LandmarkFrame frame, float t)
        {
            var pts = frame.points;
            int n = Mathf.Min(_verts.Length, pts.Length);
            var work = new Vector3[n];   // v1 低频兜底路径，可接受分配
            for (int i = 0; i < n; i++)
            {
                Vector3 p = pts[i];
                work[i] = new Vector3((p.x - 0.5f) * scaleX, (0.5f - p.y) * scaleY, -p.z * scaleZ);
            }
            _cloud.Filter(work, t);
            for (int i = 0; i < n; i++) _verts[i] = work[i];
            for (int i = n; i < _verts.Length; i++) _verts[i] = _baseVerts[i];
            transform.SetPositionAndRotation(new Vector3(0f, 0f, faceDistance),
                                             Quaternion.LookRotation(Vector3.back, Vector3.up));
            WriteVertices();
        }

        private void WriteVertices()
        {
            _mesh.SetVertices(_verts);       // 无分配
            _mesh.RecalculateNormals();
        }

        private void RecomputeWorld()
        {
            _localToWorld = transform.localToWorldMatrix;
            _mesh.GetNormals(_normalList);
            int n = Mathf.Min(CurrentWorldPositions.Length, _normalList.Count);
            for (int i = 0; i < n; i++)
            {
                CurrentWorldPositions[i] = _localToWorld.MultiplyPoint3x4(_verts[i]);
                CurrentWorldNormals[i] = _localToWorld.MultiplyVector(_normalList[i]).normalized;
            }
        }

        /// 蒙版光栅化用：关键点索引 → canonical UV
        public bool TryGetUV(int landmarkIndex, out Vector2 uv)
        {
            if (_uvs != null && landmarkIndex >= 0 && landmarkIndex < _uvs.Length)
            {
                uv = _uvs[landmarkIndex];
                return true;
            }
            uv = default;
            return false;
        }
    }
}
