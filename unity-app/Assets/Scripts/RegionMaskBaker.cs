// RegionMaskBaker.cs
// 妆容区域蒙版烘焙：landmark-regions.json 的关键点索引 → UV 空间蒙版。
// R 通道 = 覆盖率（羽化边缘），G 通道 = 向心度（区域中心 1 → 边缘 0，供渐变色带取色）。
// 纯软件光栅化；CPU 部分（BakeLayerMaskData）可在后台线程执行，纹理创建（ToTexture）须主线程。
// 与 makeup-skill/scripts/preview_render.py 的 RegionMasks 同一逻辑（Python 侧为等价参考实现）。
using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace MakeupMirror
{
    /// CPU 烘焙结果（线程安全的纯数据）
    public sealed class MaskBake
    {
        public float[] Coverage;
        public float[] Centrality;
        public int Size;
    }

    public class RegionMaskBaker
    {
        public const int MaskSize = 512;
        /// 脸缘羽化宽度（像素，UV 512 尺度）
        public const float EdgeFeatherPx = 26f;

        private JObject _regions;
        private readonly Dictionary<string, int[]> _indexCache = new Dictionary<string, int[]>();
        private readonly object _cacheLock = new object();
        private readonly FaceMeshDeformer _deformer;

        public bool Ready => _regions != null;

        public RegionMaskBaker(FaceMeshDeformer deformer)
        {
            _deformer = deformer;
            string path = Path.Combine(Application.streamingAssetsPath, "landmark-regions.json");
            if (!File.Exists(path))
            {
                Debug.LogError($"[masks] 找不到 {path}（从 makeup-skill/references/ 复制）");
                return;
            }
            _regions = JObject.Parse(File.ReadAllText(path));
            Debug.Log($"[masks] landmark-regions.json 已载入（{(_regions["regions"] as JObject)?.Count} 区域组）");
        }

        private int[] Indices(string group)
        {
            lock (_cacheLock)
            {
                if (_indexCache.TryGetValue(group, out var cached)) return cached;
                var arr = _regions?["regions"]?[group]?["indices"] as JArray;
                var list = new List<int>();
                if (arr != null)
                    foreach (var t in arr) list.Add(t.Value<int>());
                var result = list.ToArray();
                _indexCache[group] = result;
                return result;
            }
        }

        private int Anchor(string group)
        {
            var tok = _regions?["regions"]?[group]?["center_anchor"];
            return tok?.Value<int>() ?? -1;
        }

        // ---------- 对外入口 ----------

        /// 主线程便捷入口：CPU 烘焙 + 建纹理
        public Texture2D BakeLayerMask(JObject layer)
        {
            var data = BakeLayerMaskData(layer);
            return data == null ? null : ToTexture(data);
        }

        /// CPU 烘焙（可在后台线程调用）。未知 region 返回 null。
        public MaskBake BakeLayerMaskData(JObject layer)
        {
            string region = layer.Value<string>("region") ?? "";
            string side = layer.Value<string>("side") ?? "both";
            var shape = layer["shape"] as JObject ?? new JObject();

            var canvas = new float[MaskSize * MaskSize]; // coverage
            switch (region)
            {
                case "foundation": PaintPolygon(canvas, UvPolygon("foundation_face_oval"), 1f); break;
                case "concealer": PaintConcealer(canvas, side, shape); break;
                case "contour":
                    PaintStroke(canvas, UvPolygon("contour_forehead"), StrokeWidth(shape, 0.035f));
                    if (shape.Value<bool?>("jaw") ?? true)
                    {
                        PaintStroke(canvas, UvPolygon("contour_jaw_left"), StrokeWidth(shape, 0.03f));
                        PaintStroke(canvas, UvPolygon("contour_jaw_right"), StrokeWidth(shape, 0.03f));
                    }
                    if (shape.Value<bool?>("nose") ?? true)
                        PaintStroke(canvas, UvPolygon("contour_nose"), StrokeWidth(shape, 0.012f));
                    break;
                case "eyebrow": PaintEyebrow(canvas, side, shape); break;
                case "eyeshadow": PaintEyeshadow(canvas, side, shape); break;
                case "eyeliner": PaintEyeliner(canvas, side, shape, lashes: false); break;
                case "lashes": PaintEyeliner(canvas, side, shape, lashes: true); break;
                case "blush": PaintBlush(canvas, side, shape); break;
                case "highlight": PaintHighlight(canvas, shape); break;
                case "lipstick": PaintLips(canvas, shape); break;
                default:
                    Debug.LogWarning($"[masks] 未知 region：{region}，跳过");
                    return null;
            }

            // 羽化：按"到区域边缘的距离"软衰减（大面积腮红/眼影才有自然晕染），再 1 遍小核模糊抗锯齿。
            // 羽化宽度表与 makeup-skill/scripts/preview_render.py 的 FEATHER_PX 一致。
            float falloff = shape.Value<float?>("falloff") ?? 0.65f;
            float feather = FeatherPx(region, shape, falloff);
            float[] cov = canvas;
            if (feather > 0f)
            {
                float[] dist = DistanceFromOutside(canvas, 0.05f);
                cov = new float[canvas.Length];
                for (int i = 0; i < cov.Length; i++)
                    cov[i] = canvas[i] * Mathf.Clamp01(dist[i] / feather);
            }
            float[] blurred = BoxBlur(cov, 1);
            float[] centrality = Centrality(blurred);
            return new MaskBake { Coverage = blurred, Centrality = centrality, Size = MaskSize };
        }

        // 各部位羽化基准宽度（像素，512 UV 尺度）；0 = 细线条不做距离衰减（会被侵蚀）
        private static readonly Dictionary<string, float> FeatherBase = new Dictionary<string, float>
        {
            { "foundation", 10f }, { "concealer", 20f }, { "contour", 16f }, { "eyebrow", 0f },
            { "eyeshadow", 24f }, { "eyeliner", 0f }, { "lashes", 0f }, { "blush", 44f },
            { "highlight", 14f }, { "lipstick", 4f },
        };

        private static float FeatherPx(string region, JObject shape, float falloff)
        {
            float b = FeatherBase.TryGetValue(region, out var v) ? v : 8f;
            if (region == "lipstick") b = 3f + (shape.Value<float?>("blur") ?? 0.15f) * 30f;
            if (b <= 0f) return 0f;
            return b * (0.5f + Mathf.Clamp01(falloff));
        }

        /// 主线程：烘焙数据 → RGBA 纹理（R=覆盖，G=向心）
        public static Texture2D ToTexture(MaskBake data)
        {
            int n = data.Size;
            var tex = new Texture2D(n, n, TextureFormat.RGBA32, false, true)
            { wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
            var pixels = new Color32[n * n];
            for (int i = 0; i < pixels.Length; i++)
            {
                byte cov = (byte)(Mathf.Clamp01(data.Coverage[i]) * 255f);
                byte cen = (byte)(Mathf.Clamp01(data.Centrality[i]) * 255f);
                pixels[i] = new Color32(cov, cen, 0, 255);
            }
            tex.SetPixels32(pixels);
            tex.Apply(false, true);
            return tex;
        }

        /// 脸缘羽化图（R = 距脸廓内侧距离/羽化宽度，0 边缘 → 1 内部）。所有妆容层共用，消除"面具边缘"。
        public Texture2D BakeFaceEdgeTexture()
        {
            var oval = new float[MaskSize * MaskSize];
            PaintPolygon(oval, UvPolygon("foundation_face_oval"), 1f);
            float[] dist = DistanceFromOutside(oval);
            var tex = new Texture2D(MaskSize, MaskSize, TextureFormat.R8, false, true)
            { wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
            var px = new byte[MaskSize * MaskSize];
            for (int i = 0; i < px.Length; i++)
                px[i] = (byte)(Mathf.Clamp01(dist[i] / EdgeFeatherPx) * 255f);
            tex.SetPixelData(px, 0);
            tex.Apply(false, true);
            return tex;
        }

        private List<Vector2> UvPolygon(string group)
        {
            var pts = new List<Vector2>();
            foreach (int idx in Indices(group))
                if (_deformer != null && _deformer.TryGetUV(idx, out var uv))
                    pts.Add(uv);
            return pts;
        }

        private static IEnumerable<string> Sides(string side)
        {
            if (side == "left" || side == "right") { yield return side; yield break; }
            yield return "left";
            yield return "right";
        }

        // ---------- 各部位画法 ----------

        private void PaintEyebrow(float[] canvas, string side, JObject shape)
        {
            float w = 0.010f + (shape.Value<float?>("thickness") ?? 0.4f) * 0.006f;
            foreach (string one in Sides(side))
                PaintStroke(canvas, UvPolygon($"eyebrow_{one}"), w);
        }

        private void PaintEyeshadow(float[] canvas, string side, JObject shape)
        {
            float lower = shape.Value<float?>("lower_lid") ?? 0f;
            foreach (string one in Sides(side))
            {
                // 上睑：眼睑轮廓环 + 眼窝上界（extended）
                var pts = new List<Vector2>(UvPolygon($"eyelid_{one}"));
                var ext = _regions?["regions"]?[$"eyelid_{one}"]?["extended"] as JArray;
                if (ext != null)
                    foreach (var t in ext)
                        if (_deformer.TryGetUV(t.Value<int>(), out var uv)) pts.Add(uv);
                PaintPolygon(canvas, pts, 1f);
                if (lower > 0.01f)
                    PaintPolygon(canvas, UvPolygon($"lower_lid_{one}"), Mathf.Clamp01(lower));
            }
        }

        private void PaintEyeliner(float[] canvas, string side, JObject shape, bool lashes)
        {
            float width = (lashes ? 0.006f : 0.0f) + (shape.Value<float?>("thickness") ?? 0.25f) * 0.008f;
            float wing = shape.Value<float?>("wing") ?? 0.3f;
            foreach (string one in Sides(side))
            {
                PaintStroke(canvas, UvPolygon($"eyeliner_{one}"), width);
                if (!lashes && wing > 0.01f)
                {
                    var wingPts = WingPoints(one, wing);
                    if (wingPts.Count >= 2) PaintStroke(canvas, wingPts, width * 0.8f);
                }
            }
        }

        private List<Vector2> WingPoints(string one, float wing)
        {
            var tok = _regions?["regions"]?[$"eyeliner_{one}"]?["wing_anchor"] as JArray;
            var pts = new List<Vector2>();
            if (tok == null) return pts;
            foreach (var t in tok)
                if (_deformer.TryGetUV(t.Value<int>(), out var uv)) pts.Add(uv);
            if (pts.Count >= 2)
            {
                // 眼尾向外上拉：水平方向按眼线走向（内眼角→外眼角）的外侧取，左右对称
                var tail = pts[pts.Count - 1];
                var dir = (pts[pts.Count - 1] - pts[0]).normalized;
                float outward = dir.x >= 0f ? 1f : -1f;
                var lift = new Vector2(outward * 0.65f, 0.35f).normalized;
                pts.Add(tail + lift * (0.008f + 0.018f * wing));
            }
            return pts;
        }

        private void PaintBlush(float[] canvas, string side, JObject shape)
        {
            float radius = shape.Value<float?>("radius") ?? 0.12f;
            float angle = shape.Value<float?>("angle_deg") ?? 15f;
            var centerUv = shape["center_uv"] as JArray;
            foreach (string one in Sides(side))
            {
                Vector2 center;
                if (centerUv != null && one == "left")
                    center = new Vector2(centerUv[0].Value<float>(), centerUv[1].Value<float>());
                else if (centerUv != null)
                    center = new Vector2(1f - centerUv[0].Value<float>(), centerUv[1].Value<float>());
                else if (_deformer.TryGetUV(Anchor($"blush_{one}"), out var uv))
                    center = uv;
                else continue;
                float sign = one == "left" ? 1f : -1f;
                PaintEllipse(canvas, center, radius, radius * 1.25f, sign * angle, 1f);
            }
        }

        private void PaintHighlight(float[] canvas, JObject shape)
        {
            var areas = shape["areas"] as JArray ?? new JArray("cheek", "nose");
            var wanted = new HashSet<string>();
            foreach (var a in areas) wanted.Add(a.Value<string>());
            if (wanted.Contains("cheek"))
            {
                if (_deformer.TryGetUV(Anchor("highlight_cheek_left"), out var l)) PaintEllipse(canvas, l, 0.034f, 0.024f, 10f, 1f);
                if (_deformer.TryGetUV(Anchor("highlight_cheek_right"), out var r)) PaintEllipse(canvas, r, 0.034f, 0.024f, -10f, 1f);
            }
            if (wanted.Contains("nose")) PaintStroke(canvas, UvPolygon("highlight_nose"), 0.012f);
            if (wanted.Contains("cupid")) PaintStroke(canvas, UvPolygon("highlight_cupid"), 0.009f);
        }

        private void PaintConcealer(float[] canvas, string side, JObject shape)
        {
            if (!(shape.Value<bool?>("under_eye") ?? true)) return;
            float size = Mathf.Clamp(shape.Value<float?>("size") ?? 1f, 0.4f, 2f);
            foreach (string one in Sides(side))
            {
                var pts = UvPolygon($"lower_lid_{one}");
                if (pts.Count == 0) continue;
                var center = Vector2.zero;
                foreach (var p in pts) center += p;
                center /= pts.Count;
                PaintEllipse(canvas, center + new Vector2(0f, -0.012f), 0.055f * size, 0.028f * size, 0f, 1f);
            }
        }

        private void PaintLips(float[] canvas, JObject shape)
        {
            var outer = UvPolygon("lips_outer");
            PaintPolygon(canvas, outer, 1f);
            float overline = Mathf.Clamp01(shape.Value<float?>("overline") ?? 0f);
            if (overline > 0.01f)
                PaintStroke(canvas, outer, 0.004f + overline * 0.012f);   // 唇线外扩
        }

        // ---------- 图元 ----------

        private static float StrokeWidth(JObject shape, float def) => shape.Value<float?>("strength") is float s
            ? def * (0.6f + 0.8f * s) : def;

        private static void PaintPolygon(float[] canvas, List<Vector2> pts, float value)
        {
            if (pts.Count < 3) return;
            int n = pts.Count;
            var px = new float[n]; var py = new float[n];
            for (int i = 0; i < n; i++) { px[i] = pts[i].x * MaskSize; py[i] = pts[i].y * MaskSize; }
            float minY = float.MaxValue, maxY = float.MinValue;
            for (int i = 0; i < n; i++) { minY = Mathf.Min(minY, py[i]); maxY = Mathf.Max(maxY, py[i]); }
            int y0 = Mathf.Max(0, (int)minY - 1), y1 = Mathf.Min(MaskSize - 1, (int)maxY + 1);
            var xs = new List<float>(16);
            for (int y = y0; y <= y1; y++)
            {
                float cy = y + 0.5f;
                xs.Clear();
                for (int i = 0; i < n; i++)
                {
                    int j = (i + 1) % n;
                    if ((py[i] <= cy && py[j] > cy) || (py[j] <= cy && py[i] > cy))
                        xs.Add(px[i] + (cy - py[i]) / (py[j] - py[i]) * (px[j] - px[i]));
                }
                xs.Sort();
                for (int k = 0; k + 1 < xs.Count; k += 2)
                {
                    int x0 = Mathf.Max(0, (int)xs[k]), x1 = Mathf.Min(MaskSize - 1, (int)xs[k + 1]);
                    for (int x = x0; x <= x1; x++) canvas[y * MaskSize + x] = Mathf.Max(canvas[y * MaskSize + x], value);
                }
            }
        }

        private void PaintStroke(float[] canvas, List<Vector2> pts, float widthUv)
        {
            if (pts.Count < 2) { if (pts.Count == 1) PaintEllipse(canvas, pts[0], widthUv, widthUv, 0f, 1f); return; }
            float wpx = widthUv * MaskSize;
            var strip = new List<Vector2>(pts.Count * 2);
            for (int i = 0; i < pts.Count; i++)
            {
                Vector2 dir = i == 0 ? pts[1] - pts[0] : i == pts.Count - 1 ? pts[i] - pts[i - 1] : pts[i + 1] - pts[i - 1];
                dir = new Vector2(-dir.y, dir.x).normalized * (wpx * 0.5f / MaskSize);
                strip.Add(pts[i] + dir);
            }
            for (int i = pts.Count - 1; i >= 0; i--)
            {
                Vector2 dir = i == 0 ? pts[1] - pts[0] : i == pts.Count - 1 ? pts[i] - pts[i - 1] : pts[i + 1] - pts[i - 1];
                dir = new Vector2(dir.y, -dir.x).normalized * (wpx * 0.5f / MaskSize);
                strip.Add(pts[i] + dir);
            }
            PaintPolygon(canvas, strip, 1f);
        }

        private static void PaintEllipse(float[] canvas, Vector2 c, float rx, float ry, float angleDeg, float value)
        {
            rx = Mathf.Max(rx, 0.01f); ry = Mathf.Max(ry, 0.01f);
            float rad = -angleDeg * Mathf.Deg2Rad;
            float cos = Mathf.Cos(rad), sin = Mathf.Sin(rad);
            int x0 = Mathf.Max(0, (int)((c.x - (rx + ry)) * MaskSize));
            int x1 = Mathf.Min(MaskSize - 1, (int)((c.x + (rx + ry)) * MaskSize));
            int y0 = Mathf.Max(0, (int)((c.y - (rx + ry)) * MaskSize));
            int y1 = Mathf.Min(MaskSize - 1, (int)((c.y + (rx + ry)) * MaskSize));
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++)
                {
                    Vector2 d = new Vector2((x + 0.5f) / MaskSize - c.x, (y + 0.5f) / MaskSize - c.y);
                    float u = (d.x * cos - d.y * sin) / rx;
                    float v = (d.x * sin + d.y * cos) / ry;
                    if (u * u + v * v <= 1f) canvas[y * MaskSize + x] = Mathf.Max(canvas[y * MaskSize + x], value);
                }
        }

        // ---------- 后处理 ----------

        private static float[] BoxBlur(float[] src, int passes)
        {
            float[] a = (float[])src.Clone(), b = new float[src.Length];
            for (int p = 0; p < passes; p++)
            {
                for (int y = 0; y < MaskSize; y++)
                    for (int x = 0; x < MaskSize; x++)
                    {
                        float s = 0f; int cnt = 0;
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                int nx = Mathf.Clamp(x + dx, 0, MaskSize - 1);
                                int ny = Mathf.Clamp(y + dy, 0, MaskSize - 1);
                                s += a[ny * MaskSize + nx]; cnt++;
                            }
                        b[y * MaskSize + x] = s / cnt;
                    }
                (a, b) = (b, a);
            }
            return a;
        }

        /// 两遍 chamfer 距离场：区域内像素（coverage > threshold）到区域外的距离（像素）
        private static float[] DistanceFromOutside(float[] coverage, float threshold = 0.5f)
        {
            const float INF = 1e6f;
            var d = new float[coverage.Length];
            for (int i = 0; i < d.Length; i++) d[i] = coverage[i] > threshold ? INF : 0f;
            // 正向
            for (int y = 0; y < MaskSize; y++)
                for (int x = 0; x < MaskSize; x++)
                {
                    int i = y * MaskSize + x;
                    if (d[i] == 0f) continue;
                    float m = d[i];
                    if (x > 0) m = Mathf.Min(m, d[i - 1] + 1f);
                    if (y > 0) m = Mathf.Min(m, d[i - MaskSize] + 1f);
                    if (x > 0 && y > 0) m = Mathf.Min(m, d[i - MaskSize - 1] + 1.414f);
                    if (x < MaskSize - 1 && y > 0) m = Mathf.Min(m, d[i - MaskSize + 1] + 1.414f);
                    d[i] = m;
                }
            // 反向
            for (int y = MaskSize - 1; y >= 0; y--)
                for (int x = MaskSize - 1; x >= 0; x--)
                {
                    int i = y * MaskSize + x;
                    if (d[i] == 0f) continue;
                    float m = d[i];
                    if (x < MaskSize - 1) m = Mathf.Min(m, d[i + 1] + 1f);
                    if (y < MaskSize - 1) m = Mathf.Min(m, d[i + MaskSize] + 1f);
                    if (x < MaskSize - 1 && y < MaskSize - 1) m = Mathf.Min(m, d[i + MaskSize + 1] + 1.414f);
                    if (x > 0 && y < MaskSize - 1) m = Mathf.Min(m, d[i + MaskSize - 1] + 1.414f);
                    d[i] = m;
                }
            return d;
        }

        /// 向心度 = 1 - dist/max（区域中心 1 → 边缘 0）
        private static float[] Centrality(float[] coverage)
        {
            float[] d = DistanceFromOutside(coverage);
            float max = 0f;
            for (int i = 0; i < d.Length; i++) if (d[i] < 1e5f) max = Mathf.Max(max, d[i]);
            var outv = new float[coverage.Length];
            if (max < 1e-3f) return outv;
            for (int i = 0; i < d.Length; i++)
                outv[i] = coverage[i] > 0.01f ? Mathf.Clamp01(1f - d[i] / max) : 0f;
            return outv;
        }
    }
}
