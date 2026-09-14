// GaussianAvatarRenderer.cs — 3DGS 画像渲染器（Avatar 试妆主视图）。
//   · GaussianAvatarParser 解析 PLY → ComputeBuffer（位置/协方差/颜色/妆容 tint）
//   · 排序：CPU 视深排序（相机动过阈值或每 resortIntervalFrames 帧），索引 buffer 上传
//   · 绘制：Graphics.DrawProceduralNow + MakeupMirror/GaussianAvatarSplat（EWA 投影在 shader 内）
// 12 万高斯量级 30fps+（排序 ~8ms，仅在视角变化时发生）；更大规模先在导入端抽稀。
// 布局由 AvatarStationFlow 控制（预览居中 / 妆容台右侧参照）。
using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace MakeupMirror
{
    public class GaussianAvatarRenderer : MonoBehaviour
    {
        [Header("引用")]
        [Tooltip("画像渲染用相机（留空取 Camera.main）")]
        public Camera viewCamera;
        [Tooltip("GaussianAvatarSplat shader（留空自动 Find）")]
        public Shader splatShader;

        [Header("排序与外观")]
        [Tooltip("排序间隔帧数（相机未动时也要周期性重排）")]
        public int resortIntervalFrames = 6;
        [Tooltip("相机位移超过该值(米)或转角超过该角度(度)立即重排")]
        public float resortMoveThreshold = 0.02f;
        public float resortAngleThreshold = 1.5f;
        [Range(0f, 1f)] public float makeupIntensity = 0.8f;

        public bool IsLoaded => _count > 0;
        public int Count => _count;
        public string AvatarId => _avatarId;

        private ComputeBuffer _positions, _covA, _covB, _colors, _tints, _order;
        private int[] _orderCpu = Array.Empty<int>();
        private float[] _depthKeys = Array.Empty<float>();
        private int _count;
        private string _avatarId;
        private Material _mat;
        private Vector3 _lastCamPos = Vector3.zero;
        private Quaternion _lastCamRot = Quaternion.identity;
        private int _framesSinceSort = int.MaxValue;
        private static readonly int RootMatrixId = Shader.PropertyToID("_RootMatrix");
        private static readonly int WorldToCamId = Shader.PropertyToID("_WorldToCamera");
        private static readonly int ProjId = Shader.PropertyToID("_CameraProjection");
        private static readonly int FocalId = Shader.PropertyToID("_FocalPx");
        private static readonly int ScreenId = Shader.PropertyToID("_ScreenSize");
        private static readonly int IntensityId = Shader.PropertyToID("_MakeupIntensity");
        private static readonly int EnvTintId = Shader.PropertyToID("_EnvTint");
        private static readonly int EnvStrengthId = Shader.PropertyToID("_EnvStrength");

        private void OnDestroy() => ReleaseBuffers();

        private void ReleaseBuffers()
        {
            foreach (var b in new[] { _positions, _covA, _covB, _colors, _tints, _order })
                b?.Release();
            _positions = _covA = _covB = _colors = _tints = _order = null;
            _count = 0;
            _framesSinceSort = int.MaxValue;
        }

        // ---------- 加载 ----------

        public void Load(string avatarId, byte[] plyBytes, byte[] tintBytes = null)
        {
            var cloud = GaussianAvatarParser.LoadPly(plyBytes);
            ReleaseBuffers();
            _avatarId = avatarId;
            _count = cloud.Count;
            _positions = new ComputeBuffer(_count, 12);
            _positions.SetData(cloud.Positions);
            _covA = new ComputeBuffer(_count, 16);
            _covA.SetData(cloud.CovA);
            _covB = new ComputeBuffer(_count, 16);
            _covB.SetData(cloud.CovB);
            _colors = new ComputeBuffer(_count, 16);
            _colors.SetData(cloud.Colors);
            _tints = new ComputeBuffer(_count, 16);
            _tints.SetData(BuildNeutralTint(_count));
            _orderCpu = new int[_count];
            for (int i = 0; i < _count; i++) _orderCpu[i] = i;
            _depthKeys = new float[_count];
            _order = new ComputeBuffer(_count, 4);
            _order.SetData(_orderCpu);
            _framesSinceSort = int.MaxValue;

            if (tintBytes != null) ApplyTint(tintBytes);
            if (splatShader == null) splatShader = Shader.Find("MakeupMirror/GaussianAvatarSplat");
            if (_mat == null)
            {
                _mat = new Material(splatShader);
            }
            Debug.Log($"[avatar] 画像已加载：{avatarId}（{_count} 高斯）");
        }

        public void Unload()
        {
            ReleaseBuffers();
            _avatarId = null;
        }

        private static Vector4[] BuildNeutralTint(int n)
        {
            var t = new Vector4[n];
            var zero = new Vector4(0, 0, 0, 0);
            for (int i = 0; i < n; i++) t[i] = zero;
            return t;
        }

        /// <summary>妆容 tint buffer（MKMKP1 tint.bin；行数必须与高斯数一致）。</summary>
        public void ApplyTint(byte[] tintBytes)
        {
            var tint = GaussianAvatarParser.LoadTint(tintBytes);
            if (tint.Length != _count)
            {
                Debug.LogError($"[avatar] tint 行数不符：tint={tint.Length} gaussians={_count}"
                               + "（画像抽稀设置与编译不一致？）");
                return;
            }
            _tints.SetData(tint);
            Debug.Log($"[avatar] 妆容 tint 已应用：覆盖 {CountTinted(tint)} 高斯");
        }

        public void ClearTint() => _tints.SetData(BuildNeutralTint(_count));

        private static int CountTinted(Vector4[] t)
        {
            int k = 0;
            for (int i = 0; i < t.Length; i++)
                if (t[i].w > 1e-4f) k++;
            return k;
        }

        // ---------- 渲染 ----------

        private void Update()
        {
            if (_mat != null) _mat.SetFloat(IntensityId, makeupIntensity);
        }

        private void OnRenderObject()
        {
            var cam = viewCamera != null ? viewCamera : Camera.main;
            if (cam == null || Camera.current != cam || _count == 0 || _mat == null) return;

            MaybeResort(cam);
            PushFrameGlobals(cam);
            _mat.SetPass(0);
            Graphics.DrawProceduralNow(MeshTopology.Triangles, _count * 6, 1);
        }

        private void MaybeResort(Camera cam)
        {
            bool moved = (cam.transform.position - _lastCamPos).sqrMagnitude
                             > resortMoveThreshold * resortMoveThreshold
                         || Quaternion.Angle(cam.transform.rotation, _lastCamRot) > resortAngleThreshold;
            if (!moved && _framesSinceSort < resortIntervalFrames)
            {
                _framesSinceSort++;
                return;
            }
            _lastCamPos = cam.transform.position;
            _lastCamRot = cam.transform.rotation;
            _framesSinceSort = 0;

            Vector3 fwd = cam.transform.forward;
            Vector3 pos = cam.transform.position;
            var cloudPos = new Vector3[_count];
            _positions.GetData(cloudPos);            // 12 万 × 12B ≈ 1.4MB 回读，排序帧 ~2ms
            var root = transform.localToWorldMatrix;
            for (int i = 0; i < _count; i++)
            {
                Vector3 w = root.MultiplyPoint3x4(cloudPos[i]);
                _orderCpu[i] = i;
                _depthKeys[i] = -Vector3.Dot(w - pos, fwd);   // 远 → 近
            }
            Array.Sort(_depthKeys, _orderCpu);
            _order.SetData(_orderCpu);
        }

        private void PushFrameGlobals(Camera cam)
        {
            _mat.SetMatrix(RootMatrixId, transform.localToWorldMatrix);
            _mat.SetMatrix(WorldToCamId, cam.worldToCameraMatrix);
            _mat.SetMatrix(ProjId, cam.projectionMatrix);
            float ih = cam.pixelHeight;
            float iw = cam.pixelWidth;
            var p = cam.projectionMatrix;
            _mat.SetVector(FocalId, new Vector2(p.m00 * iw * 0.5f, p.m11 * ih * 0.5f));
            _mat.SetVector(ScreenId, new Vector2(iw, ih));
            var envTint = Shader.GetGlobalVector("_EnvTint");
            var envStrength = Shader.GetGlobalFloat("_EnvStrength");
            _mat.SetVector(EnvTintId, envTint);
            _mat.SetFloat(EnvStrengthId, envStrength);
        }
    }
}
