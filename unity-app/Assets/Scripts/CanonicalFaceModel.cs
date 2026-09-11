// CanonicalFaceModel.cs
// 解析 StreamingAssets/canonical_face_model.obj（MediaPipe 468 点拓扑，v/vt/f），
// 构建 Unity Mesh 并提供 UV 查询（妆容蒙版光栅化用）。
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace MakeupMirror
{
    public class CanonicalFaceModel
    {
        public Mesh Mesh { get; private set; }
        public Vector2[] UVs { get; private set; }
        public int VertexCount { get; private set; }

        /// canonical 头部局部顶点（Unity 空间：米，(x,y,-z)），与追踪包 pts 同一约定
        public Vector3[] CanonicalVertices { get; private set; }

        public static CanonicalFaceModel Load(string streamingAssetsName = "canonical_face_model.obj")
        {
            string path = Path.Combine(Application.streamingAssetsPath, streamingAssetsName);
            if (!File.Exists(path))
            {
                Debug.LogError($"[face-model] 找不到 {path}（仓库 Assets/StreamingAssets 已随附）");
                return null;
            }
            var positions = new List<Vector3>(512);
            var texcoords = new List<Vector2>(512);
            var faces = new List<(int v, int vt)[]>(512);

            foreach (string raw in File.ReadLines(path))
            {
                if (raw.StartsWith("v "))
                {
                    var p = raw.Split(' ');
                    // canonical（右手系，厘米，+z 朝观察者）→ Unity（左手系，米，脸正面朝 -Z 即朝向相机）：
                    // (x, y, -z) × 0.01；该反射使 obj 原始绕向恰好需要 (0,2,1) 翻转（下方 tris）
                    positions.Add(new Vector3(
                        float.Parse(p[1], CultureInfo.InvariantCulture) * 0.01f,
                        float.Parse(p[2], CultureInfo.InvariantCulture) * 0.01f,
                        -float.Parse(p[3], CultureInfo.InvariantCulture) * 0.01f));
                }
                else if (raw.StartsWith("vt "))
                {
                    var p = raw.Split(' ');
                    texcoords.Add(new Vector2(
                        float.Parse(p[1], CultureInfo.InvariantCulture),
                        float.Parse(p[2], CultureInfo.InvariantCulture)));
                }
                else if (raw.StartsWith("f "))
                {
                    // f v1/vt1 v2/vt2 v3/vt3 —— canonical obj 的 vt 是图集置换编号，
                    // 与顶点编号不同；记录每角的 (顶点, UV) 对，之后折算成顶点索引 UV
                    var p = raw.Split(' ');
                    var corners = new (int, int)[3];
                    for (int i = 0; i < 3; i++)
                    {
                        var parts = p[i + 1].Split('/');
                        corners[i] = (int.Parse(parts[0]) - 1, int.Parse(parts[1]) - 1);
                    }
                    faces.Add(corners);
                }
            }

            // 顶点→UV 置换映射（每个顶点在所有面角中的 vt 编号一致，见 references/schema.md）
            var vertexUv = new Vector2[positions.Count];
            foreach (var corners in faces)
                foreach (var (v, vt) in corners)
                    vertexUv[v] = texcoords[vt];

            // 网格以米为单位的 canonical 头部局部坐标（供 v2 追踪路径直接替换顶点）
            model.CanonicalVertices = positions.ToArray();

            var tris = new List<int>(faces.Count * 3);
            foreach (var corners in faces)
            {
                // z 反射后 obj 原始绕向朝内 → 翻转 (0,2,1) 得 Unity 正面（朝向相机一侧）
                tris.Add(corners[0].v); tris.Add(corners[2].v); tris.Add(corners[1].v);
            }

            var model = new CanonicalFaceModel();
            model.VertexCount = positions.Count;
            model.UVs = vertexUv;
            var mesh = new Mesh { indexFormat = UnityEngine.Rendering.IndexFormat.UInt16 };
            mesh.SetVertices(positions);
            mesh.SetUVs(0, vertexUv);
            mesh.SetTriangles(tris, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            model.Mesh = mesh;
            Debug.Log($"[face-model] 载入 {positions.Count} 顶点 / {tris.Count / 3} 面（UV 按顶点置换映射）");
            return model;
        }
    }
}
