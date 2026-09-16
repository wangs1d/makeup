// AvatarSplatRenderer.cs — 画像上的静态附加溅射层（唇釉/闪片等 add_splats.json）。
// 与 SplatLayerRenderer（真脸锚点版）的区别：位置/法线/切轴直接由 Python 编译器
// 烘进 add_splats.json（画像局部空间，脸高=1），不需要关键点查询——挂在画像根节点
// 下随布局移动即可。每帧视深排序 + RenderMeshInstanced，复用 GaussianSplat shader。
using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace MakeupMirror
{
    public class AvatarSplatRenderer : MonoBehaviour
    {
        [Tooltip("高斯面片 shader（MakeupMirror/GaussianSplat，与真脸溅射共用）")]
        public Shader splatShader;
        [Range(0.5f, 2f)] public float sizeScale = 1f;
        [Range(0f, 0.9f)] public float viewBlend = 0.35f;
        [Tooltip("强度（预览确认值 enter_station 后由 look.intensity 驱动）")]
        [Range(0f, 1f)] public float intensity = 1f;
        [Tooltip("镜面强度总开关（唇釉/珠光的高光，逐帧按真实视角 CPU 计算）")]
        [Range(0f, 1f)] public float specStrength = 0.45f;
        [Tooltip("光源方向（世界系，Blinn-Phong 用）")]
        public Vector3 lightDir = new Vector3(0.30f, 0.55f, 0.80f);

        public int Count => _count;

        private struct Splat
        {
            public Vector3 pos;      // 画像局部空间
            public Vector3 normal;
            public Vector3 axis;     // 切向长轴
            public Vector2 sigma;    // 归一化单位（脸高=1）
            public Vector4 color;    // rgb + peak alpha
            public float gloss;      // 0..1 镜面强度（add_splats.json 的 gloss，缺省 0.35）
        }

        private const int Batch = 1023;
        private readonly List<Splat> _splats = new List<Splat>(512);
        private int _count;
        private Matrix4x4[] _matrices = Array.Empty<Matrix4x4>();
        private Vector4[] _colorsSorted = Array.Empty<Vector4>();
        private readonly Vector4[] _colorsBatch = new Vector4[Batch];
        private float[] _specSorted = Array.Empty<float>();
        private readonly float[] _specBatch = new float[Batch];
        private float[] _depthKeys = Array.Empty<float>();
        private int[] _order = Array.Empty<int>();
        private Mesh _quad;
        private Material _mat;
        private MaterialPropertyBlock _mpb;
        private static readonly int ColorId = Shader.PropertyToID("_Color");
        private static readonly int IntensityId = Shader.PropertyToID("_MakeupIntensity");
        private static readonly int SpecId = Shader.PropertyToID("_Spec");
        private static readonly int SpecStrengthId = Shader.PropertyToID("_SpecStrength");

        private void Awake()
        {
            if (splatShader == null) splatShader = Shader.Find("MakeupMirror/GaussianSplat");
            _quad = BuildQuad();
            _mpb = new MaterialPropertyBlock();
        }

        private void OnDestroy()
        {
            if (_mat != null) Destroy(_mat);
            if (_quad != null) Destroy(_quad);
        }

        /// <param name="splatsJson">add_splats.json 原始字节（{"splats":[...]}，画像局部空间）</param>
        public void Apply(byte[] splatsJson)
        {
            Clear();
            if (splatsJson == null || splatsJson.Length == 0) return;
            var root = JObject.Parse(System.Text.Encoding.UTF8.GetString(splatsJson));
            if (!(root["splats"] is JArray arr)) return;

            foreach (var t in arr)
            {
                if (!(t is JObject s)) continue;
                var col = Color.white;
                ColorUtility.TryParseHtmlString(s.Value<string>("color") ?? "#FFFFFF", out col);
                var sig = s["sigma"] as JArray;
                _splats.Add(new Splat
                {
                    pos = ToV3(s["pos"]),
                    normal = ToV3(s["normal"]).normalized,
                    axis = ToV3(s["axis"]).normalized,
                    sigma = new Vector2(sig?[0]?.Value<float?>() ?? 0.0015f,
                                        sig?[1]?.Value<float?>() ?? 0.0012f),
                    color = new Vector4(col.r, col.g, col.b, s.Value<float?>("alpha") ?? 0.7f),
                    gloss = s.Value<float?>("gloss") ?? 0.35f,
                });
            }
            _count = _splats.Count;
            _matrices = new Matrix4x4[_count];
            _colorsSorted = new Vector4[_count];
            _specSorted = new float[_count];
            _depthKeys = new float[_count];
            _order = new int[_count];
            if (_mat == null)
            {
                _mat = new Material(splatShader) { enableInstancing = true };
                _mat.renderQueue = 3150;
            }
            Debug.Log($"[avatar] 附加溅射就绪：{_count} 个高斯面片");
        }

        public void Clear()
        {
            _splats.Clear();
            _count = 0;
        }

        private static Vector3 ToV3(JToken t)
        {
            var a = t as JArray;
            if (a == null || a.Count < 3) return Vector3.zero;
            return new Vector3(a[0].Value<float>(), a[1].Value<float>(), a[2].Value<float>());
        }

        private void LateUpdate()
        {
            if (_count == 0 || _mat == null) return;
            var cam = Camera.main;
            if (cam == null) return;
            var l2w = transform.localToWorldMatrix;
            Vector3 camPos = cam.transform.position;
            Vector3 camFwd = cam.transform.forward;

            for (int i = 0; i < _count; i++)
            {
                var s = _splats[i];
                Vector3 p = l2w.MultiplyPoint3x4(s.pos);
                Vector3 n = l2w.MultiplyVector(s.normal).normalized;
                Vector3 a = l2w.MultiplyVector(s.axis);
                a -= n * Vector3.Dot(a, n);
                if (a.sqrMagnitude < 1e-10f) a = Vector3.Cross(n, Vector3.up);
                a.Normalize();
                Vector3 b = Vector3.Cross(n, a);
                // 侧向时向相机倾斜，避免边缘面片消失（与真脸溅射同策略）
                var toCam = (camPos - p).normalized;
                n = Vector3.Slerp(n, toCam, viewBlend).normalized;

                float sx = s.sigma.x * 6f * sizeScale;   // 全宽 = 6σ
                float sy = s.sigma.y * 6f * sizeScale;
                var m = Matrix4x4.identity;
                m.SetColumn(0, new Vector4(a.x * sx, a.y * sx, a.z * sx, 0f));
                m.SetColumn(1, new Vector4(b.x * sy, b.y * sy, b.z * sy, 0f));
                m.SetColumn(2, new Vector4(n.x * 0.001f, n.y * 0.001f, n.z * 0.001f, 0f));
                m.SetColumn(3, new Vector4(p.x, p.y, p.z, 1f));
                _order[i] = i;
                _depthKeys[i] = -Vector3.Dot(p - camPos, camFwd);
                _matrices[i] = m;
                _colorsSorted[i] = s.color;
                // Blinn-Phong 逐帧 CPU 计算（相机/光源每帧变化；数量 ≤ 数千，开销可忽略）
                var L = lightDir.sqrMagnitude < 1e-6f ? new Vector3(0.3f, 0.55f, 0.8f) : lightDir.normalized;
                var H = (L + toCam).normalized;
                float ndh = Mathf.Clamp01(Vector3.Dot(n, H));
                _specSorted[i] = s.gloss * Mathf.Pow(ndh, 90f);
            }
            Array.Sort(_depthKeys, _order);

            _mat.SetFloat(IntensityId, intensity);
            _mat.SetFloat(SpecStrengthId, specStrength);
            var bounds = new Bounds(transform.position, Vector3.one * 1.5f);
            var rp = new RenderParams(_mat)
            {
                worldBounds = bounds,
                shadowCastingMode = ShadowCastingMode.Off,
                receiveShadows = false,
                layer = gameObject.layer,
                matProps = _mpb,
            };
            if (SystemInfo.supportsInstancing)
            {
                for (int start = 0; start < _count; start += Batch)
                {
                    int n = Mathf.Min(Batch, _count - start);
                    for (int k = 0; k < n; k++)
                    {
                        _colorsBatch[k] = _colorsSorted[_order[start + k]];
                        _specBatch[k] = _specSorted[_order[start + k]];
                    }
                    _mpb.SetVectorArray(ColorId, _colorsBatch);
                    _mpb.SetFloatArray(SpecId, _specBatch);
                    Graphics.RenderMeshInstanced(rp, _quad, 0, _matrices, n, start);
                }
            }
            else
            {
                for (int i = 0; i < _count; i++)
                {
                    _mpb.SetVector(ColorId, _colorsSorted[_order[i]]);
                    Graphics.RenderMesh(rp, _quad, 0, _matrices[_order[i]]);
                }
            }
        }

        private static Mesh BuildQuad()
        {
            var m = new Mesh { name = "avatar_splat_quad" };
            m.SetVertices(new[]
            {
                new Vector3(-0.5f, -0.5f, 0f), new Vector3(0.5f, -0.5f, 0f),
                new Vector3(0.5f, 0.5f, 0f), new Vector3(-0.5f, 0.5f, 0f),
            });
            m.SetUVs(0, new[] { new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1), new Vector2(0, 1) });
            m.SetTriangles(new[] { 0, 2, 1, 0, 3, 2 }, 0);
            m.RecalculateBounds();
            return m;
        }
    }
}
