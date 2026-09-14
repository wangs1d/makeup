// GaussianAvatarSplat.shader — 3DGS 画像高斯原语渲染（DrawProceduralNow，六顶点展开）。
// GaussianAvatarRenderer 提供结构化 buffer（位置/3D 协方差/颜色/妆容 tint/排序索引），
// 顶点着色器做标准 EWA 投影：Σworld → 视空间 → 焦距雅可比 → 屏幕 2D 协方差 →
// 特征轴展开 quad；片元按 σ 距离评估高斯核，premultiplied 混合（远→近由 CPU 排序保证）。
// 妆容：_Tints per-Gaussian rgba（a=覆盖权重），与原色按 _MakeupIntensity 插值——
// 换妆不重建位置 buffer，只换 tint buffer 与全局强度。
Shader "MakeupMirror/GaussianAvatarSplat"
{
    Properties
    {
        _MakeupIntensity ("Makeup intensity", Range(0, 1)) = 0.8
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
            StructuredBuffer<float4> _Colors;    // (rgb, opacity)
            StructuredBuffer<float4> _Tints;     // (rgb, 覆盖权重) — a==0 表示无妆
            StructuredBuffer<int>   _Order;      // 视深远→近

            float4x4 _RootMatrix;                // 画像根节点 local→world
            float4x4 _WorldToCamera;
            float4x4 _CameraProjection;
            float2   _FocalPx;                   // (fx, fy) 像素
            float2   _ScreenSize;
            float    _MakeupIntensity;
            float4   _EnvTint;
            float    _EnvStrength;

            static const float PixelBlur = 0.3;  // 屏幕 2D 协方差像素抗锯齿（标准 3DGS 同值）
            static const float QuadSigma = 2.0;  // 半边长 = 2σ（覆盖核能量 ~98%）
            static const float2 QuadCorners[6] = {
                float2(-1, -1), float2(1, -1), float2(1, 1),
                float2(-1, -1), float2(1, 1), float2(-1, 1)
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 col : COLOR0;             // 直色 rgb + alpha（premultiplied 在 frag 做）
                float2 dsig : TEXCOORD0;         // 主轴坐标系下的 σ 距离
            };

            v2f vert(uint vid : SV_VertexID)
            {
                v2f o;
                uint gi = vid / 6u;
                float2 c = QuadCorners[vid % 6u];
                int idx = _Order[gi];

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
                float4 tint = _Tints[idx];
                float3 rgb = lerp(_Colors[idx].rgb, tint.rgb, saturate(tint.a * _MakeupIntensity));
                rgb *= lerp(1.0, saturate(_EnvTint.rgb), saturate(_EnvStrength));
                o.col = float4(rgb, _Colors[idx].a);
                o.dsig = c * QuadSigma;          // 片元：主轴坐标系 σ 距离
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float g = exp(-0.5 * dot(i.dsig, i.dsig));
                float alpha = saturate(i.col.a * g);
                return float4(i.col.rgb * alpha, alpha);   // premultiplied
            }
            ENDCG
        }
    }
    FallBack Off
}
