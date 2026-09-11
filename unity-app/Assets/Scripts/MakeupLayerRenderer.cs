// MakeupLayerRenderer.cs
// 妆容图层管理：spec 的每个 layer → 独立 GameObject（共享人脸网格）+ 妆容材质。
// 蒙版由 RegionMaskBaker 在后台线程烘焙（主线程只建纹理）；色带/噪点来自 bridge 下发的烘焙资产，
// 缺资产时由 color_stops 程序化生成兜底。intensity / 脸缘羽化 / 面部存在度 走 shader 全局量。
using System.Collections.Generic;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace MakeupMirror
{
    public class MakeupLayerRenderer : MonoBehaviour
    {
        [Header("引用")]
        public FaceMeshDeformer deformer;
        [Tooltip("妆容层 shader（MakeupMirror/MakeupLayer）")]
        public Shader layerShader;

        [Header("外观")]
        [Range(0f, 1f)] public float intensity = 0.8f;
        [Tooltip("环境光合成强度（0 关闭 = 旧版 unlit 叠加）")]
        [Range(0f, 1f)] public float envStrength = 0.85f;

        public int LayerCount => _layers.Count;
        public string CurrentSpecName { get; private set; }

        private RegionMaskBaker _baker;
        private Texture2D _faceEdgeTex;
        private readonly List<LayerEntry> _layers = new List<LayerEntry>();
        private int _generation;

        private static readonly int MaskTex = Shader.PropertyToID("_MaskTex");
        private static readonly int RampTex = Shader.PropertyToID("_RampTex");
        private static readonly int GrainTex = Shader.PropertyToID("_GrainTex");
        private static readonly int GrainStrength = Shader.PropertyToID("_GrainStrength");
        private static readonly int Opacity = Shader.PropertyToID("_Opacity");
        private static readonly int Finish = Shader.PropertyToID("_Finish");
        private static readonly int GMakeupIntensity = Shader.PropertyToID("_MakeupIntensity");
        private static readonly int GFacePresence = Shader.PropertyToID("_FacePresence");
        private static readonly int GFaceEdgeTex = Shader.PropertyToID("_FaceEdgeTex");
        private static readonly int GEnvStrength = Shader.PropertyToID("_EnvStrength");

        private struct LayerEntry
        {
            public GameObject go;
            public string id;
            public float baseOpacity;
            public Material mat;
        }

        private void Awake()
        {
            if (layerShader == null)
                layerShader = Shader.Find("MakeupMirror/MakeupLayer");
            if (deformer != null)
                _baker = new RegionMaskBaker(deformer);
            Shader.SetGlobalFloat(GMakeupIntensity, intensity);
            Shader.SetGlobalFloat(GFacePresence, 0f);
        }

        private void Start()
        {
            if (_baker != null && _baker.Ready && _faceEdgeTex == null)
            {
                _faceEdgeTex = _baker.BakeFaceEdgeTexture();
                Shader.SetGlobalTexture(GFaceEdgeTex, _faceEdgeTex);
            }
        }

        private void Update()
        {
            if (deformer != null) Shader.SetGlobalFloat(GFacePresence, deformer.FacePresence);
            Shader.SetGlobalFloat(GEnvStrength, envStrength);
        }

        // ---------- 应用 / 清除 ----------

        /// <param name="spec">bridge 下发的 spec JObject</param>
        /// <param name="assets">烘焙资产（文件名 → PNG 字节），可为 null</param>
        public async Task Apply(JObject spec, Dictionary<string, byte[]> assets)
        {
            int gen = ++_generation;
            if (_baker == null || !_baker.Ready || deformer == null)
            {
                Debug.LogError("[makeup] 缺少 RegionMaskBaker/FaceMeshDeformer，无法渲染");
                return;
            }
            var layers = spec["layers"] as JArray;
            if (layers == null) { Debug.LogWarning("[makeup] spec 无 layers"); return; }

            // 后台线程顺序烘焙全部蒙版（主线程不卡）
            var enabled = new List<JObject>();
            foreach (var token in layers)
                if (token is JObject layer && (layer.Value<bool?>("enabled") ?? true))
                    enabled.Add(layer);
            var baker = _baker;
            MaskBake[] bakes = await Task.Run(() =>
            {
                var result = new MaskBake[enabled.Count];
                for (int i = 0; i < enabled.Count; i++)
                    result[i] = baker.BakeLayerMaskData(enabled[i]);
                return result;
            });
            if (gen != _generation) return;   // 期间来了更新的 spec，放弃本次

            Clear();
            for (int i = 0; i < enabled.Count; i++)
            {
                var layer = enabled[i];
                if (bakes[i] == null) continue;
                var mask = RegionMaskBaker.ToTexture(bakes[i]);

                var go = new GameObject($"makeup_{layer.Value<string>("id")}");
                go.transform.SetParent(deformer.transform, false);   // 跟随人脸根节点姿态
                var mf = go.AddComponent<MeshFilter>();
                mf.sharedMesh = deformer.GetComponent<MeshFilter>().sharedMesh;
                var mr = go.AddComponent<MeshRenderer>();
                mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                mr.receiveShadows = false;
                var mat = new Material(layerShader) { renderQueue = 3000 + Mathf.Min(_layers.Count, 80) };
                mat.SetTexture(MaskTex, mask);
                mat.SetTexture(RampTex, GetRamp(layer, assets));
                mat.SetTexture(GrainTex, GetGrain(layer, assets));
                mat.SetFloat(GrainStrength, layer.Value<float?>("texture_strength") ?? 0.3f);
                mat.SetFloat(Finish, FinishIndex(layer.Value<string>("finish")));
                float op = Mathf.Clamp01(layer.Value<float?>("opacity") ?? 0.7f);
                mat.SetFloat(Opacity, op);
                mr.sharedMaterial = mat;
                _layers.Add(new LayerEntry { go = go, id = layer.Value<string>("id"), baseOpacity = op, mat = mat });
            }
            CurrentSpecName = spec.Value<string>("name");
            Debug.Log($"[makeup] 应用妆容「{CurrentSpecName}」：{_layers.Count} 层 mesh");
        }

        public void Clear()
        {
            foreach (var l in _layers)
                if (l.go != null) Destroy(l.go);
            _layers.Clear();
            CurrentSpecName = null;
        }

        public void SetIntensity(float v)
        {
            intensity = Mathf.Clamp01(v);
            Shader.SetGlobalFloat(GMakeupIntensity, intensity);
        }

        private static float FinishIndex(string finish) => finish switch
        {
            "matte" => 0f,
            "satin" => 1f,
            "dewy" => 2f,
            "gloss" => 3f,
            _ => 1f,
        };

        // ---------- 资产获取（bridge 下发优先，程序化兜底） ----------

        private static Texture2D FromPng(byte[] png)
        {
            var tex = new Texture2D(4, 4, TextureFormat.RGBA32, false, true);
            if (ImageConversion.LoadImage(tex, png)) return tex;
            Destroy(tex);
            return null;
        }

        private Texture2D GetRamp(JObject layer, Dictionary<string, byte[]> assets)
        {
            string baked = layer["baked"]?["ramp"]?.Value<string>();
            if (baked != null && assets != null && assets.TryGetValue(baked, out var png))
            {
                var t = FromPng(png);
                if (t != null) { t.wrapMode = TextureWrapMode.Clamp; return t; }
            }
            // 兜底：color_stops → 256x1 渐变
            var tex = new Texture2D(256, 1, TextureFormat.RGBA32, false, true)
            { wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
            var stops = layer["color_stops"] as JArray;
            var px = new Color32[256];
            for (int x = 0; x < 256; x++)
                px[x] = SampleStops(stops, x / 255f);
            tex.SetPixels32(px);
            tex.Apply(false, true);
            return tex;
        }

        private static Color SampleStops(JArray stops, float t)
        {
            if (stops == null || stops.Count == 0) return new Color(0.85f, 0.6f, 0.5f);
            var pts = new List<(float at, Color col)>();
            foreach (var s in stops)
            {
                if (ColorUtility.TryParseHtmlString(s.Value<string>("hex"), out var c))
                    pts.Add((s.Value<float?>("at") ?? 0f, c));
            }
            if (pts.Count == 0) return new Color(0.85f, 0.6f, 0.5f);
            pts.Sort((a, b) => a.at.CompareTo(b.at));
            if (t <= pts[0].at) return pts[0].col;
            if (t >= pts[pts.Count - 1].at) return pts[pts.Count - 1].col;
            for (int i = 0; i + 1 < pts.Count; i++)
                if (t >= pts[i].at && t <= pts[i + 1].at)
                {
                    float f = Mathf.InverseLerp(pts[i].at, pts[i + 1].at, t);
                    return Color.Lerp(pts[i].col, pts[i + 1].col, f);
                }
            return pts[pts.Count - 1].col;
        }

        private Texture2D GetGrain(JObject layer, Dictionary<string, byte[]> assets)
        {
            string baked = layer["baked"]?["grain"]?.Value<string>();
            if (baked != null && assets != null && assets.TryGetValue(baked, out var png))
            {
                var t = FromPng(png);
                if (t != null) { t.wrapMode = TextureWrapMode.Repeat; return t; }
            }
            // 兜底：中性 128 灰（无粉感）
            var tex = new Texture2D(64, 64, TextureFormat.RGBA32, false, true) { wrapMode = TextureWrapMode.Repeat };
            var px = new Color32[64 * 64];
            for (int i = 0; i < px.Length; i++) px[i] = new Color32(128, 128, 128, 255);
            tex.SetPixels32(px);
            tex.Apply(false, true);
            return tex;
        }
    }
}
