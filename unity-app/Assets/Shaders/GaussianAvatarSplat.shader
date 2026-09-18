// GaussianAvatarSplat.shader — 3DGS 画像高斯原语渲染（DrawProceduralNow，六顶点展开）。
// GaussianAvatarRenderer 提供结构化 buffer（位置/3D 协方差/颜色/SH 高阶/妆容 tint/
// 排序索引/材质），顶点着色器做标准 EWA 投影 + SH 视角色求值；片元做逐像素
// 化妆品 PBR（rough/coat/sss/sheen，线性空间数学后转显示空间）。
// 写实底模的颜色已含烘焙光照 → 默认纯叠加（_Relight=0 只加高光/SSS/绒光），
// 拉满 _Relight 才启用合成 wrap diffuse（程序化模板资产用）。
Shader "MakeupMirror/GaussianAvatarSplat"
{
    Properties
    {
        _MakeupIntensity ("Makeup intensity", Range(0, 1)) = 0.8
        _Relight ("Synthetic relight", Range(0, 1)) = 0.0
    }
    SubShader
    {
        Tags { "Queue" = "Transparent+200" "RenderType" = "Transparent" "IgnoreProjector" = "True" }
        ZWrite Off
        ZTest LEqual
        Cull Off
        Blend One OneMinusSrcAlpha      // premultiplied alpha

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #include "UnityCG.cginc"

            StructuredBuffer<float3> _Positions;
            StructuredBuffer<float4> _CovA;      // (Σxx, Σxy, Σxz, Σyy)
            StructuredBuffer<float4> _CovB;      // (Σyz, Σzz, 0, 0)
            StructuredBuffer<float4> _Colors;    // (rgb sRGB, opacity)
            StructuredBuffer<float4> _Tints;     // (rgb, 覆盖权重) — a==0 表示无妆
            StructuredBuffer<uint>   _Order;     // 视深远→近
            StructuredBuffer<float4> _MatNR;     // (nx,ny,nz, rough) 画像局部空间
            StructuredBuffer<float4> _MatCS;     // (coat, sss, sheen, 0)
            StructuredBuffer<float3> _ShRest;    // SH 高阶 8×(n)（degree2，xyz=RGB 通道）

            float4x4 _RootMatrix;                // 画像根节点 local→world
            float4x4 _WorldToCamera;
            float4x4 _CameraProjection;
            float2   _FocalPx;                   // (fx, fy) 像素
            float2   _ScreenSize;
            float    _MakeupIntensity;
            float4   _EnvTint;
            float    _EnvStrength;
            float    _SpecStrength;              // 化妆品 PBR 高光总开关
            float    _Relight;                   // 合成漫反射重打光量（烘焙资产默认 0）
            float    _ShCount;                   // 每 splat SH 高阶条数（0=无）
            float3   _LightDir;                  // 世界系，指向光源

            static const float PixelBlur = 0.3;  // 屏幕 2D 协方差像素抗锯齿（标准 3DGS 同值）
            static const float QuadSigma = 2.5;  // 半边长 2.5σ（2σ 截断环在大 splat 上可见）
            static const float2 QuadCorners[6] = {
                float2(-1, -1), float2(1, -1), float2(1, 1),
                float2(-1, -1), float2(1, 1), float2(-1, 1)
            };

            // 2.2 近似 sRGB（与 GaussianAvatarParser.Srgb 同式，可逆）
            float3 SrgbToLinear(float3 c) { return pow(c, 2.2); }
            float3 LinearToSrgb(float3 c) { return pow(c, 1.0 / 2.2); }

            // SH degree 2 高阶求值（世界系方向；系数来自 gsplat 世界系训练）。
            // 与 3DGS/gsplat eval_sh 同式：sh[1..8] 逐通道在 _ShRest[b+0..7]。
            // 调用方必须先判 _ShCount（无 SH 时 _ShRest 是 1 元素占位，不可寻址）。
            float3 ShEvalDeg2(float3 d, uint idx)
            {
                uint b = idx * 8u;
                float x = d.x, y = d.y, z = d.z;
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;
                float3 c = -0.48860251190292f * y * _ShRest[b + 0]
                           + 0.48860251190292f * z * _ShRest[b + 1]
                           - 0.48860251190292f * x * _ShRest[b + 2];
                c += 1.09254843059211f * xy * _ShRest[b + 3]
                     - 1.09254843059211f * yz * _ShRest[b + 4]
                     + 0.31539156525252f * (2.0f * zz - xx - yy) * _ShRest[b + 5]
                     - 1.09254843059211f * xz * _ShRest[b + 6]
                     + 0.54627421529604f * (xx - yy) * _ShRest[b + 7];
                return c;
            }

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 dsig : TEXCOORD0;         // 主轴坐标系下的 σ 距离
                float4 col : TEXCOORD1;          // 显示空间 rgb + alpha（全精度插值器，HDR 高光不截断）
                float3 wpos : TEXCOORD2;         // 世界系位置（片元视线）
                float3 wnorm : TEXCOORD3;        // 世界系法线
                float4 mat : TEXCOORD4;          // (coat, sss, sheen, rough)
            };

            v2f vert(uint vid : SV_VertexID)
            {
                v2f o;
                uint gi = vid / 6u;
                float2 c = QuadCorners[vid % 6u];
                uint idx = _Order[gi];

                float3 world = mul(_RootMatrix, float4(_Positions[idx], 1)).xyz;
                float4 view = mul(_WorldToCamera, float4(world, 1));
                float vz = max(view.z, 0.05);    // Unity 相机空间 +Z 朝前

                // Σworld → Σview：Σv = W Σ Wᵀ
                float3x3 W = (float3x3)_WorldToCamera;
                float3x3 Sigma = float3x3(
                    _CovA[idx].x, _CovA[idx].y, _CovA[idx].z,
                    _CovA[idx].y, _CovA[idx].w, _CovB[idx].x,
                    _CovA[idx].z, _CovB[idx].x, _CovB[idx].y);
                float3x3 SigV = mul(W, mul(Sigma, transpose(W)));

                // 焦距雅可比 J（2×3，对角近似）→ 屏幕 2D 协方差：J Σv Jᵀ + 像素模糊
                float3 J0 = float3(_FocalPx.x / vz, 0, -_FocalPx.x * view.x / (vz * vz));
                float3 J1 = float3(0, _FocalPx.y / vz, -_FocalPx.y * view.y / (vz * vz));
                float a = dot(J0, mul(SigV, J0)) + PixelBlur;
                float b = dot(J0, mul(SigV, J1));
                float cc = dot(J1, mul(SigV, J1)) + PixelBlur;

                // 特征分解（2×2 解析）：长/短轴与特征值
                float det = max(a * cc - b * b, 1e-6);
                float mid = 0.5 * (a + cc);
                float disc = sqrt(max(mid * mid - det, 1e-6));
                float l1 = mid + disc;
                float l2 = max(mid - disc, 0.05);
                float2 e1 = normalize(abs(b) > 1e-9 ? float2(l1 - cc, b) : float2(1, 0));
                float2 e2 = float2(-e1.y, e1.x);

                // quad 展开：主/次轴 × QuadSigma·√λ（像素）→ NDC → clip
                float2 ext = QuadSigma * float2(sqrt(l1), sqrt(l2));
                float2 pxOffset = c.x * ext.x * e1 + c.y * ext.y * e2;
                float4 clip = mul(_CameraProjection, view);
                clip.xy += pxOffset * 2.0 / _ScreenSize * clip.w;

                o.pos = clip;

                // 基色：DC(sRGB) → 线性 + SH 视角项 → 显示空间（混合仍在显示空间，
                // 与训练预览/主流 3DGS 查看器一致）
                float3 dcLin = SrgbToLinear(_Colors[idx].rgb);
                if (_ShCount > 0.5)
                    dcLin = saturate(dcLin + ShEvalDeg2(normalize(world - _WorldSpaceCameraPos), idx));
                float4 tint = _Tints[idx];
                float3 base = lerp(LinearToSrgb(dcLin), tint.rgb,
                                   saturate(tint.a * _MakeupIntensity));

                o.col = float4(base, _Colors[idx].a);
                o.dsig = c * QuadSigma;          // 片元：主轴坐标系 σ 距离
                o.wpos = world;
                float4 nr = _MatNR[idx];
                o.wnorm = mul(_RootMatrix, float4(nr.xyz, 0)).xyz;
                float4 cs = _MatCS[idx];
                o.mat = float4(cs.x, cs.y, cs.z, clamp(nr.w, 0.03, 1.0));
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float g = exp(-0.5 * dot(i.dsig, i.dsig));
                float alpha = saturate(i.col.a * g);

                // 化妆品 PBR（与 appearance/pbr.py 同式）逐像素求值：
                // rough 决定高光宽度，coat 清漆（唇釉/珠光），sss 唇部透光红移，
                // sheen 掠射绒光。线性空间数学，结果转回显示空间参与混合。
                float3 rgb = i.col.rgb;
                float4 cs = i.mat;
                float3 nW = normalize(i.wnorm);
                float3 vW = normalize(_WorldSpaceCameraPos - i.wpos);
                float3 L = normalize(_LightDir);
                float3 H = normalize(L + vW);
                float ndl = saturate(dot(nW, L));
                float ndv = saturate(dot(nW, vW));
                float ndh = saturate(dot(nW, H));
                float hdv = saturate(dot(H, vW));

                float shin = clamp(2.0 / (cs.w * cs.w * cs.w * cs.w + 1e-4), 8, 1200);
                float fac = saturate((ndl + 0.25) / 1.25);           // wrap diffuse
                float fres = 0.028 + 0.972 * pow(1 - hdv, 5);
                float spec = pow(ndh, shin) * fres * cs.x;
                float sheen = cs.z * 0.35 * pow(1 - ndv, 3);
                float3 sss = cs.y * 0.30 * (1 - fac)
                             * float3(0.90, 0.22, 0.45);             // 唇部透光红移

                float3 albLin = SrgbToLinear(saturate(rgb));
                float3 envLin = lerp(1.0.xxx, saturate(SrgbToLinear(_EnvTint.rgb)),
                                     saturate(_EnvStrength));
                // relight=0（默认，烘焙资产）：漫反射恒 1，只叠加高光/SSS/绒光
                float wrap = lerp(1.0, 0.30 + 0.70 * fac, _Relight);
                albLin = albLin * wrap * envLin
                         + albLin * sss
                         + (spec + sheen) * _SpecStrength * envLin * 3.0;

                return float4(LinearToSrgb(saturate(albLin)) * alpha, alpha);  // premultiplied
            }
            ENDCG
        }
    }
    FallBack Off
}
