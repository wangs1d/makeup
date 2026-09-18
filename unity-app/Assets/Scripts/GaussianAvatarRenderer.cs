// GaussianAvatarRenderer.cs — 3DGS 画像渲染器（Avatar 试妆主视图）。
//   · GaussianAvatarParser 解析 PLY → ComputeBuffer（位置/协方差/颜色/SH 高阶/妆容 tint/材质）
//   · 排序：GPU bitonic（GaussianAvatarSort.compute，每帧可重排无 pop）；
//     compute 不可用时回退 CPU 视深排序（相机动过阈值或每 resortIntervalFrames 帧）
//   · 绘制：Graphics.DrawProceduralNow + MakeupMirror/GaussianAvatarSplat
//     （EWA 投影 + SH 视角色在顶点，化妆品 PBR 逐像素在片元）
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
        [Tooltip("GPU 深度排序 compute（留空用 CPU 排序回退）")]
        public ComputeShader sortShader;

        [Header("排序与外观")]
        [Tooltip("排序间隔帧数（相机未动时也要周期性重排；GPU 排序时自动每帧）")]
        public int resortIntervalFrames = 6;
        [Tooltip("相机位移超过该值(米)或转角超过该角度(度)立即重排")]
        public float resortMoveThreshold = 0.02f;
        public float resortAngleThreshold = 1.5f;
        [Range(0f, 1f)] public float makeupIntensity = 0.8f;
        [Tooltip("化妆品 PBR 高光强度（rough/coat 材质通道，唇釉/珠光随视角流动）")]
        [Range(0f, 1f)] public float specStrength = 0.5f;
        [Tooltip("合成漫反射重打光量（0=纯叠加，烘焙资产保持真实光影；程序化模板资产可拉满）")]
        [Range(0f, 1f)] public float relight = 0f;
        [Tooltip("光源方向（世界系，化妆品高光用；light.bin sidecar 会覆盖）")]
        public Vector3 lightDir = new Vector3(0.30f, 0.55f, 0.80f);

        public bool IsLoaded => _count > 0;
        public int Count => _count;
        public string AvatarId => _avatarId;

        private ComputeBuffer _positions, _covA, _covB, _colors, _tints, _order;
        private ComputeBuffer _matNR, _matCS;      // PBR 材质 sidecar（MKMA v1）
        private ComputeBuffer _shRest, _shNone;    // SH 高阶（无则绑 1 元素占位）
        private ComputeBuffer _keysA, _keysB, _idxA, _idxB;  // GPU bitonic ping-pong
        private ComputeBuffer _gpuOrder;           // 当前绑给材质的排序索引
        private int _kernelDepth, _kernelBitonic;
        private bool _computeReady;
        private int[] _orderCpu = Array.Empty<int>();
        private float[] _depthKeys = Array.Empty<float>();
        private int _count;
        private uint _n2;
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
        private static readonly int SpecStrengthId = Shader.PropertyToID("_SpecStrength");
        private static readonly int RelightId = Shader.PropertyToID("_Relight");
        private static readonly int ShCountId = Shader.PropertyToID("_ShCount");
        private static readonly int LightDirId = Shader.PropertyToID("_LightDir");
        private static readonly int OrderId = Shader.PropertyToID("_Order");
        private static readonly int ShRestBufId = Shader.PropertyToID("_ShRest");

        private void OnDestroy() => ReleaseBuffers();

        private void ReleaseBuffers()
        {
            foreach (var b in new[] { _positions, _covA, _covB, _colors, _tints, _order,
                                      _matNR, _matCS, _shRest, _shNone,
                                      _keysA, _keysB, _idxA, _idxB })
                b?.Release();
            _positions = _covA = _covB = _colors = _tints = _order = null;
            _matNR = _matCS = _shRest = _shNone = null;
            _keysA = _keysB = _idxA = _idxB = null;
            _gpuOrder = null;
            _count = 0;
            _n2 = 0;
            _computeReady = false;
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
            if (cloud.ShRest != null)
            {
                _shRest = new ComputeBuffer(cloud.ShRest.Length, 12);
                _shRest.SetData(cloud.ShRest);
            }
            else
            {
                _shNone = new ComputeBuffer(1, 12);      // 占位：HLSL StructuredBuffer 必须可绑定
                _shRest = _shNone;
            }
            _orderCpu = new int[_count];
            for (int i = 0; i < _count; i++) _orderCpu[i] = i;
            _depthKeys = new float[_count];
            _order = new ComputeBuffer(_count, 4);
            _order.SetData(_orderCpu);
            _gpuOrder = _order;
            _matNR = new ComputeBuffer(_count, 16);
            _matNR.SetData(BuildNeutralMaterialNR(_count));
            _matCS = new ComputeBuffer(_count, 16);
            _matCS.SetData(BuildNeutralMaterialCS(_count));
            _framesSinceSort = int.MaxValue;

            if (tintBytes != null) ApplyTint(tintBytes);
            if (splatShader == null) splatShader = Shader.Find("MakeupMirror/GaussianAvatarSplat");
            if (_mat == null)
            {
                _mat = new Material(splatShader);
            }
            _mat.SetBuffer(ShRestBufId, _shRest);
            _mat.SetFloat(ShCountId, cloud.ShRest != null ? 8f : 0f);
            SetupCompute();
            Debug.Log($"[avatar] 画像已加载：{avatarId}（{_count} 高斯，"
                      + $"SH={(cloud.ShRest != null ? "deg2" : "dc-only")}，"
                      + $"排序={(_computeReady ? "GPU" : "CPU")}）");
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

        /// <summary>PBR 材质 sidecar（MKMA v1 material.bin；行数必须与高斯数一致）。</summary>
        public void ApplyMaterial(byte[] materialBytes)
        {
            Vector4[] nr, cs;
            GaussianAvatarParser.LoadMaterial(materialBytes, out nr, out cs);
            if (nr.Length != _count)
            {
                Debug.LogError($"[avatar] material 行数不符：{nr.Length} vs {_count}");
                return;
            }
            _matNR.SetData(nr);
            _matCS.SetData(cs);
            Debug.Log($"[avatar] 化妆品材质已应用：{_count} 高斯");
        }

        /// <summary>主光 sidecar（MKLT1 light.bin）：高光方向与烘焙光照对齐。</summary>
        public void ApplyLight(byte[] lightBytes)
        {
            try
            {
                Vector3 dir;
                float strength;
                Vector3 tint;
                GaussianAvatarParser.LoadLight(lightBytes, out dir, out strength, out tint);
                lightDir = dir;
                Debug.Log($"[avatar] 主光已应用：dir={dir} strength={strength:0.00}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[avatar] light.bin 解析失败：{e.Message}");
            }
        }

        private static Vector4[] BuildNeutralMaterialNR(int n)
        {
            var t = new Vector4[n];
            for (int i = 0; i < n; i++) t[i] = new Vector4(0f, 0f, 1f, 0.52f);
            return t;
        }

        private static Vector4[] BuildNeutralMaterialCS(int n)
        {
            // 中性 = 无化妆品薄层（纯叠加语义下不添加任何高光；皮肤高光已烘焙）
            var t = new Vector4[n];
            for (int i = 0; i < n; i++) t[i] = new Vector4(0f, 0f, 0f, 0f);
            return t;
        }

        private static int CountTinted(Vector4[] t)
        {
            int k = 0;
            for (int i = 0; i < t.Length; i++)
                if (t[i].w > 1e-4f) k++;
            return k;
        }

        // ---------- GPU 排序 ----------

        private void SetupCompute()
        {
            _computeReady = false;
            if (sortShader == null || _count == 0) return;
            try
            {
                _kernelDepth = sortShader.FindKernel("DepthKeys");
                _kernelBitonic = sortShader.FindKernel("Bitonic");
                _n2 = 1;
                while (_n2 < (uint)_count) _n2 <<= 1;
                _keysA = new ComputeBuffer((int)_n2, 4);
                _keysB = new ComputeBuffer((int)_n2, 4);
                _idxA = new ComputeBuffer((int)_n2, 4);
                _idxB = new ComputeBuffer((int)_n2, 4);
                _computeReady = true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[avatar] GPU 排序不可用（回退 CPU）：{e.Message}");
                _keysA = _keysB = _idxA = _idxB = null;
            }
        }

        /// <summary>GPU bitonic 深度排序；结果索引直接作为材质 _Order 绑定。</summary>
        private void GpuResort(Camera cam)
        {
            var root = transform.localToWorldMatrix;
            sortShader.SetMatrix(RootMatrixId, root);
            sortShader.SetVector("_CamPos", cam.transform.position);
            sortShader.SetVector("_CamFwd", cam.transform.forward);
            sortShader.SetInt("_Count", _count);
            sortShader.SetInt("_N2", (int)_n2);
            sortShader.SetBuffer(_kernelDepth, "_Positions", _positions);
            sortShader.SetBuffer(_kernelDepth, "_KeysA", _keysA);
            sortShader.SetBuffer(_kernelDepth, "_IdxA", _idxA);
            sortShader.Dispatch(_kernelDepth, Mathf.CeilToInt(_n2 / 256f), 1, 1);

            var keysIn = _keysA; var keysOut = _keysB;
            var idxIn = _idxA; var idxOut = _idxB;
            for (uint k = 2; k <= _n2; k <<= 1)
            {
                for (uint j = k >> 1; j > 0; j >>= 1)
                {
                    sortShader.SetBuffer(_kernelBitonic, "_KeysIn", keysIn);
                    sortShader.SetBuffer(_kernelBitonic, "_IdxIn", idxIn);
                    sortShader.SetBuffer(_kernelBitonic, "_KeysOut", keysOut);
                    sortShader.SetBuffer(_kernelBitonic, "_IdxOut", idxOut);
                    sortShader.SetInt("_K", (int)k);
                    sortShader.SetInt("_J", (int)j);
                    sortShader.Dispatch(_kernelBitonic, Mathf.CeilToInt(_n2 / 256f), 1, 1);
                    var tk = keysIn; keysIn = keysOut; keysOut = tk;
                    var ti = idxIn; idxIn = idxOut; idxOut = ti;
                }
            }
            _gpuOrder = idxIn;                 // 最后一轮的输出
        }

        // ---------- 渲染 ----------

        private void Update()
        {
            if (_mat != null)
            {
                _mat.SetFloat(IntensityId, makeupIntensity);
                _mat.SetFloat(RelightId, relight);
            }
        }

        private void OnRenderObject()
        {
            var cam = viewCamera != null ? viewCamera : Camera.main;
            if (cam == null || Camera.current != cam || _count == 0 || _mat == null) return;

            bool resorted = MaybeResort(cam);
            if (resorted && _computeReady) _mat.SetBuffer(OrderId, _gpuOrder);
            PushFrameGlobals(cam);
            _mat.SetPass(0);
            Graphics.DrawProceduralNow(MeshTopology.Triangles, _count * 6, 1);
        }

        /// <summary>返回本次是否实际重排（GPU 路径每帧重排；CPU 路径按阈值节流）。</summary>
        private bool MaybeResort(Camera cam)
        {
            if (_computeReady)
            {
                GpuResort(cam);
                return true;
            }
            bool moved = (cam.transform.position - _lastCamPos).sqrMagnitude
                             > resortMoveThreshold * resortMoveThreshold
                         || Quaternion.Angle(cam.transform.rotation, _lastCamRot) > resortAngleThreshold;
            if (!moved && _framesSinceSort < resortIntervalFrames)
            {
                _framesSinceSort++;
                return false;
            }
            _lastCamPos = cam.transform.position;
            _lastCamRot = cam.transform.rotation;
            _framesSinceSort = 0;

            Vector3 fwd = cam.transform.forward;
            Vector3 pos = cam.transform.position;
            var cloudPos = new Vector3[_count];
            _positions.GetData(cloudPos);            // CPU 回退：回读 + Array.Sort
            var root = transform.localToWorldMatrix;
            for (int i = 0; i < _count; i++)
            {
                Vector3 w = root.MultiplyPoint3x4(cloudPos[i]);
                _orderCpu[i] = i;
                _depthKeys[i] = -Vector3.Dot(w - pos, fwd);   // 远 → 近
            }
            Array.Sort(_depthKeys, _orderCpu);
            _order.SetData(_orderCpu);
            _gpuOrder = _order;
            return true;
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
            _mat.SetFloat(SpecStrengthId, specStrength);
            _mat.SetVector(LightDirId,
                lightDir.sqrMagnitude < 1e-6f ? new Vector3(0.3f, 0.55f, 0.8f) : lightDir.normalized);
        }
    }
}
