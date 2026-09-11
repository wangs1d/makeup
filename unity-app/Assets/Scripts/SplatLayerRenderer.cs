// SplatLayerRenderer.cs
// 3D 高斯溅射层（唇釉/水光高光的体积感）—— 排序实例化后端：
//   · 每个 splat 锚定到人脸关键点 + 沿法线抬升，随脸运动（位置来自 FaceMeshDeformer 世界坐标）
//   · 面片贴合切平面：长轴沿区域走向（同组下一锚点方向），短轴垂直——各向异性核不再是纯 billboard
//   · 每帧按视深从远到近排序，Graphics.RenderMeshInstanced 批量绘制（≤1023/批），零 GameObject
//   · 颜色/峰值 alpha 走 MaterialPropertyBlock 实例数组；强度/淡出/环境色温走 shader 全局量
// 若装了 aras-p/UnityGaussianSplatting 可另写后端接同一份 splat_layers 数据（见 docs/assembly.md 第 6 节）。
using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace MakeupMirror
{
    public class SplatLayerRenderer : MonoBehaviour
    {
        [Header("引用")]
        public FaceMeshDeformer deformer;
        public Camera viewCamera;
        [Tooltip("高斯面片 shader（MakeupMirror/GaussianSplat）")]
        public Shader splatShader;

        [Header("外观")]
        [Tooltip("面片尺寸整体缩放")]
        [Range(0.5f, 2f)] public float sizeScale = 1f;
        [Tooltip("与法线的最小夹角余弦：面片过于侧向时向相机方向倾斜，避免边缘消失")]
        [Range(0f, 0.9f)] public float viewBlend = 0.35f;

        public int Count => _count;

        private struct Splat
        {
            public int landmark;
            public int axisLandmark;    // 长轴参考锚点（同组相邻），-1 = 无
            public int toward;          // 位置内收目标关键点（如外唇→内唇），-1 = 无
            public float inset;         // 内收比例 0~1
            public float offset;        // 米
            public Vector2 sigma;       // 米
            public Vector4 color;       // rgb + 峰值 alpha
        }

        private const int Batch = 1023;
        private Splat[] _splats = Array.Empty<Splat>();
        private int _count;
        private Matrix4x4[] _matrices = Array.Empty<Matrix4x4>();
        private Vector4[] _colorsSorted = Array.Empty<Vector4>();
        private readonly Vector4[] _colorsBatch = new Vector4[Batch];
        private float[] _depthKeys = Array.Empty<float>();
        private int[] _order = Array.Empty<int>();
        private Mesh _quad;
        private Material _mat;
        private MaterialPropertyBlock _mpb;
        private bool _warnedNoInstancing;
        private static readonly int ColorId = Shader.PropertyToID("_Color");

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

        /// <param name="splatLayers">baked_spec.json 里的 splat_layers 数组</param>
        public void Apply(JArray splatLayers)
        {
            Clear();
            if (splatLayers == null) return;
            if (_mat == null)
            {
                _mat = new Material(splatShader) { enableInstancing = true };
                _mat.renderQueue = 3100;
            }

            var list = new List<Splat>(256);
            foreach (var token in splatLayers)
            {
                if (!(token is JObject layer)) continue;
                var anchors = layer["anchors"] as JArray;
                if (anchors == null) continue;

                // 同组内按 t 排序，长轴指向组内下一锚点
                var groups = new Dictionary<string, List<JObject>>();
                foreach (var t in anchors)
                {
                    if (!(t is JObject a)) continue;
                    string g = a.Value<string>("group") ?? "";
                    if (!groups.TryGetValue(g, out var lst)) groups[g] = lst = new List<JObject>();
                    lst.Add(a);
                }
                foreach (var kv in groups)
                {
                    var lst = kv.Value;
                    lst.Sort((x, y) => (x.Value<float?>("t") ?? 0f).CompareTo(y.Value<float?>("t") ?? 0f));
                    for (int i = 0; i < lst.Count; i++)
                    {
                        var a = lst[i];
                        var sigmaTok = a["sigma"] as JArray;
                        Color col = Color.white;
                        ColorUtility.TryParseHtmlString(a.Value<string>("color") ?? "#FFFFFF", out col);
                        float alpha = a.Value<float?>("alpha") ?? 0.7f;
                        int axis = -1;
                        if (lst.Count > 1)
                            axis = (i + 1 < lst.Count ? lst[i + 1] : lst[i - 1]).Value<int>("landmark");
                        list.Add(new Splat
                        {
                            landmark = a.Value<int>("landmark"),
                            axisLandmark = axis,
                            toward = a.Value<int?>("toward") ?? -1,
                            inset = Mathf.Clamp01(a.Value<float?>("inset") ?? 0.45f),
                            offset = a.Value<float?>("offset") ?? 0.0012f,
                            sigma = new Vector2(sigmaTok?[0]?.Value<float?>() ?? 0.0015f,
                                                sigmaTok?[1]?.Value<float?>() ?? 0.0012f),
                            color = new Vector4(col.r, col.g, col.b, alpha),
                        });
                    }
                }
            }

            _splats = list.ToArray();
            _count = _splats.Length;
            _matrices = new Matrix4x4[_count];
            _colorsSorted = new Vector4[_count];
            _depthKeys = new float[_count];
            _order = new int[_count];
            Debug.Log($"[splat] 溅射层就绪：{_count} 个高斯面片（排序实例化）");
        }

        public void Clear()
        {
            _splats = Array.Empty<Splat>();
            _count = 0;
        }

        private void LateUpdate()
        {
            if (_count == 0 || deformer == null || viewCamera == null || _mat == null) return;
            if (deformer.FacePresence <= 0.01f) return;
            var pos = deformer.CurrentWorldPositions;
            var nrm = deformer.CurrentWorldNormals;
            if (pos == null || nrm == null || pos.Length == 0) return;

            Vector3 camPos = viewCamera.transform.position;
            Vector3 camFwd = viewCamera.transform.forward;
            Vector3 toCam;

            // 1) 深度键（远→近）
            for (int i = 0; i < _count; i++)
            {
                int li = _splats[i].landmark;
                _order[i] = i;
                _depthKeys[i] = li < pos.Length ? -Vector3.Dot(pos[li] - camPos, camFwd) : 0f;
            }
            Array.Sort(_depthKeys, _order);   // 升序 = -depth 升序 = depth 降序（远的先画）

            // 2) 实例矩阵
            Bounds bounds = new Bounds(pos[Mathf.Min(1, pos.Length - 1)], Vector3.one * 1.5f);
            int written = 0;
            for (int k = 0; k < _count; k++)
            {
                ref var s = ref _splats[_order[k]];
                if (s.landmark >= pos.Length) continue;
                Vector3 basePos = pos[s.landmark];
                if (s.toward >= 0 && s.toward < pos.Length)
                    basePos = Vector3.Lerp(basePos, pos[s.toward], s.inset);   // 外唇线锚点内收到唇体
                Vector3 p = basePos + nrm[s.landmark] * s.offset;
                Vector3 n = nrm[s.landmark];
                toCam = (camPos - p).normalized;
                // 侧向时向相机方向倾斜一点，保证边缘面片不会变成一条线
                n = Vector3.Slerp(n, toCam, viewBlend).normalized;

                Vector3 a;
                if (s.axisLandmark >= 0 && s.axisLandmark < pos.Length)
                    a = pos[s.axisLandmark] - pos[s.landmark];
                else
                    a = Vector3.Cross(n, Vector3.up);
                a -= n * Vector3.Dot(a, n);
                if (a.sqrMagnitude < 1e-10f) a = Vector3.Cross(n, Vector3.right);
                a.Normalize();
                Vector3 b = Vector3.Cross(n, a);

                float sx = s.sigma.x * 6f * sizeScale;   // 全宽 = 6σ（±3σ）
                float sy = s.sigma.y * 6f * sizeScale;
                var m = new Matrix4x4();
                m.SetColumn(0, new Vector4(a.x * sx, a.y * sx, a.z * sx, 0f));
                m.SetColumn(1, new Vector4(b.x * sy, b.y * sy, b.z * sy, 0f));
                m.SetColumn(2, new Vector4(n.x * 0.001f, n.y * 0.001f, n.z * 0.001f, 0f));
                m.SetColumn(3, new Vector4(p.x, p.y, p.z, 1f));
                _matrices[written] = m;
                _colorsSorted[written] = s.color;
                written++;
            }
            if (written == 0) return;

            // 3) 绘制
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
                for (int start = 0; start < written; start += Batch)
                {
                    int n = Mathf.Min(Batch, written - start);
                    Array.Copy(_colorsSorted, start, _colorsBatch, 0, n);
                    _mpb.SetVectorArray(ColorId, _colorsBatch);
                    Graphics.RenderMeshInstanced(rp, _quad, 0, _matrices, n, start);
                }
            }
            else
            {
                if (!_warnedNoInstancing)
                {
                    _warnedNoInstancing = true;
                    Debug.LogWarning("[splat] 设备不支持 GPU 实例化，退回逐面片绘制");
                }
                for (int i = 0; i < written; i++)
                {
                    _mpb.SetVector(ColorId, _colorsSorted[i]);
                    Graphics.RenderMesh(rp, _quad, 0, _matrices[i]);
                }
            }
        }

        private static Mesh BuildQuad()
        {
            var m = new Mesh { name = "splat_quad" };
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
