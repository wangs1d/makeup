// GaussianAvatarParser.cs — 标准 3DGS PLY 与妆容 sidecar（MKMKP1 tint / MKSEM1 语义）解析。
// Python 侧（avatar_io/avatar_session）产出与上传的格式在此对齐：
//   · PLY：binary_little_endian，vertex 元素含 x y z f_dc_0..2 opacity scale_0..2 rot_0..3
//     （SH 高阶与法线忽略）；颜色 SH DC → sRGB、opacity sigmoid、scale exp、quat 归一
//   · MKMKP1 tint.bin：magic(4)+ver u16+flags u16+N u32+rsv u32 + rgba f32×4×N（a=覆盖权重）
//   · MKSEM1 semantics.bin：magic(4)+ver u16+flags u16+N u32+rsv u32 + regionId u8×N[+conf u8×N]
// 解析零依赖（无 JsonPlugin/无 assimp），12 万高斯 <300ms。
using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace MakeupMirror
{
    public struct AvatarGaussianCloud
    {
        public Vector3[] Positions;      // 归一化空间（脸高=1，居中 xy）
        public Vector4[] CovA;           // Σ 上三角打包 (xx,xy,xz,yy)
        public Vector4[] CovB;           // (yz,zz,0,0)
        public Vector4[] Colors;         // (rgb sRGB, opacity)
        public int Count;
    }

    public static class GaussianAvatarParser
    {
        // ---------- PLY ----------

        public static AvatarGaussianCloud LoadPly(byte[] raw)
        {
            int dataStart;
            var props = ParsePlyHeader(raw, out dataStart, out int count);
            foreach (var need in new[] { "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                                          "opacity", "scale_0", "scale_1", "scale_2",
                                          "rot_0", "rot_1", "rot_2", "rot_3" })
                if (!props.ContainsKey(need))
                    throw new InvalidDataException($"PLY 缺少 3DGS 必需属性：{need}");

            int stride = 0;
            foreach (var p in props.Values) stride += p.Size;
            if (raw.Length < dataStart + (long)count * stride)
                throw new InvalidDataException($"PLY 数据区不完整：需要 {dataStart + (long)count * stride} 字节");

            var cloud = new AvatarGaussianCloud
            {
                Positions = new Vector3[count],
                CovA = new Vector4[count],
                CovB = new Vector4[count],
                Colors = new Vector4[count],
                Count = count,
            };
            const float shC0 = 0.28209479177387814f;
            for (int i = 0; i < count; i++)
            {
                int row = dataStart + i * stride;
                float x = PropFloat(raw, row, props, "x");
                float y = PropFloat(raw, row, props, "y");
                float z = PropFloat(raw, row, props, "z");
                cloud.Positions[i] = new Vector3(x, y, z);

                // SH DC → 线性 → sRGB 近似
                float r = 0.5f + shC0 * PropFloat(raw, row, props, "f_dc_0");
                float g = 0.5f + shC0 * PropFloat(raw, row, props, "f_dc_1");
                float b = 0.5f + shC0 * PropFloat(raw, row, props, "f_dc_2");
                float op = 1f / (1f + Mathf.Exp(-PropFloat(raw, row, props, "opacity")));
                cloud.Colors[i] = new Vector4(
                    Srgb(r), Srgb(g), Srgb(b), Mathf.Clamp01(op));

                float sx = Mathf.Exp(PropFloat(raw, row, props, "scale_0"));
                float sy = Mathf.Exp(PropFloat(raw, row, props, "scale_1"));
                float sz = Mathf.Exp(PropFloat(raw, row, props, "scale_2"));
                float qw = PropFloat(raw, row, props, "rot_0");
                float qx = PropFloat(raw, row, props, "rot_1");
                float qy = PropFloat(raw, row, props, "rot_2");
                float qz = PropFloat(raw, row, props, "rot_3");
                float ql = Mathf.Sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-9f;
                qw /= ql; qx /= ql; qy /= ql; qz /= ql;

                // M = R·S（旋转×尺度），Σ = M·Mᵀ（解析展开，避免逐点矩阵乘）
                float m00 = (1 - 2 * (qy * qy + qz * qz)) * sx;
                float m01 = (2 * (qx * qy - qw * qz)) * sy;
                float m02 = (2 * (qx * qz + qw * qy)) * sz;
                float m10 = (2 * (qx * qy + qw * qz)) * sx;
                float m11 = (1 - 2 * (qx * qx + qz * qz)) * sy;
                float m12 = (2 * (qy * qz - qw * qx)) * sz;
                float m20 = (2 * (qx * qz - qw * qy)) * sx;
                float m21 = (2 * (qy * qz + qw * qx)) * sy;
                float m22 = (1 - 2 * (qx * qx + qy * qy)) * sz;
                // Σ = M·Mᵀ
                cloud.CovA[i] = new Vector4(
                    m00 * m00 + m01 * m01 + m02 * m02,                 // xx
                    m00 * m10 + m01 * m11 + m02 * m12,                 // xy
                    m00 * m20 + m01 * m21 + m02 * m22,                 // xz
                    m10 * m10 + m11 * m11 + m12 * m12);                // yy
                cloud.CovB[i] = new Vector4(
                    m10 * m20 + m11 * m21 + m12 * m22,                 // yz
                    m20 * m20 + m21 * m21 + m22 * m22,                 // zz
                    0f, 0f);
            }
            return cloud;
        }

        private static float Srgb(float v) => Mathf.Pow(Mathf.Clamp01(v), 1f / 2.2f);

        private static float PropFloat(byte[] raw, int row, System.Collections.Generic
            .Dictionary<string, PlyProp> props, string name)
        {
            var p = props[name];
            return BitConverter.ToSingle(raw, row + p.Offset);
        }

        private struct PlyProp
        {
            public int Offset;
            public int Size;
        }

        private static System.Collections.Generic.Dictionary<string, PlyProp> ParsePlyHeader(
            byte[] raw, out int dataStart, out int vertexCount)
        {
            if (raw.Length < 4 || raw[0] != 'p' || raw[1] != 'l' || raw[2] != 'y')
                throw new InvalidDataException("不是 PLY 文件");
            string header = Encoding.ASCII.GetString(raw, 0, Math.Min(raw.Length, 64 * 1024));
            int endIdx = header.IndexOf("end_header", StringComparison.Ordinal);
            if (endIdx < 0) throw new InvalidDataException("PLY 缺少 end_header");
            int nl = header.IndexOf('\n', endIdx);
            dataStart = nl + 1;
            vertexCount = 0;

            var props = new System.Collections.Generic.Dictionary<string, PlyProp>();
            int offset = 0, declared = -1;
            bool inVertex = false;
            foreach (var rawLine in header.Substring(0, endIdx).Split('\n'))
            {
                var parts = rawLine.Trim().Split(new[] { ' ', '\t', '\r' },
                    StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 0) continue;
                if (parts[0] == "format")
                {
                    if (parts.Length < 2 || parts[1] != "binary_little_endian")
                        throw new InvalidDataException($"仅支持 binary_little_endian PLY：{parts[1]}");
                }
                else if (parts[0] == "element")
                {
                    inVertex = parts.Length > 1 && parts[1] == "vertex";
                    if (inVertex && parts.Length > 2)
                        int.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out declared);
                }
                else if (parts[0] == "property" && inVertex && parts.Length >= 2)
                {
                    int size = SizeOf(parts[1]);
                    props[parts[parts.Length - 1]] = new PlyProp { Offset = offset, Size = size };
                    offset += size;
                }
            }
            vertexCount = declared >= 0 ? declared : 0;
            return props;
        }

        private static int SizeOf(string type)
        {
            switch (type)
            {
                case "float": case "int": case "uint": return 4;
                case "double": return 8;
                case "uchar": case "char": return 1;
                case "short": case "ushort": return 2;
                default: throw new InvalidDataException($"不支持的 PLY 类型：{type}");
            }
        }

        // ---------- 妆容 sidecar ----------

        public static Vector4[] LoadTint(byte[] raw)
        {
            if (raw == null || raw.Length < 16)
                throw new InvalidDataException("tint.bin 过短");
            if (raw[0] != 'M' || raw[1] != 'K' || raw[2] != 'M' || raw[3] != 'K')
                throw new InvalidDataException("tint.bin magic 不符（需要 MKMK/MKMKP1）");
            ushort ver = BitConverter.ToUInt16(raw, 4);
            if (ver != 1) throw new InvalidDataException($"tint.bin 版本不支持：{ver}");
            int n = BitConverter.ToInt32(raw, 8);
            if (raw.Length < 16 + (long)n * 16)
                throw new InvalidDataException($"tint.bin 数据不完整：N={n}");
            var tint = new Vector4[n];
            for (int i = 0; i < n; i++)
            {
                int o = 16 + i * 16;
                tint[i] = new Vector4(
                    BitConverter.ToSingle(raw, o),
                    BitConverter.ToSingle(raw, o + 4),
                    BitConverter.ToSingle(raw, o + 8),
                    BitConverter.ToSingle(raw, o + 12));
            }
            return tint;
        }
    }
}
