// MakeupLayer.shader — 妆容图层：蒙版覆盖率 × 渐变色带 × 粉感噪点 × 环境光合成 × 质感高光。
// 渲染在人脸网格上（透明叠加在摄像头画面之上），蒙版 R=覆盖率（已羽化），G=向心度（取色带用）。
//
// 光影合成（P2）：妆容不是无光照贴片——
//   · 环境色调 _EnvTint（摄像头帧估计）给妆色染上房间色温；
//   · 屏幕空间亮度图 _EnvLumTex 让妆容随脸上真实光影明暗（光照一侧亮、阴影一侧暗）；
//   · 主光方向 _LightDirWorld 做 wrap diffuse（次表面近似）与双层高光（sheen + clearcoat）。
// 全局量：_MakeupIntensity（滑杆）、_FacePresence（丢脸淡出）、_FaceEdgeTex（脸缘羽化）。
Shader "MakeupMirror/MakeupLayer"
{
    Properties
    {
        _MaskTex ("Mask (R=coverage, G=centrality)", 2D) = "white" {}
        _RampTex ("Color Ramp", 2D) = "white" {}
        _GrainTex ("Grain", 2D) = "gray" {}
        _GrainStrength ("Grain Strength", Range(0, 1)) = 0.3
        _Opacity ("Opacity", Range(0, 1)) = 0.7
        _Finish ("Finish (0 matte / 1 satin / 2 dewy / 3 gloss)", Range(0, 3)) = 1
    }
    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" "IgnoreProjector" = "True" }
        ZWrite Off
        ZTest LEqual
        Cull Back
        Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0
            #include "UnityCG.cginc"

            sampler2D _MaskTex;
            sampler2D _RampTex;
            sampler2D _GrainTex;
            float _GrainStrength;
            float _Opacity;
            float _Finish;

            // 全局（MakeupLayerRenderer / WebcamDisplay 设置）
            float _MakeupIntensity;
            float _FacePresence;
            sampler2D _FaceEdgeTex;
            float4 _EnvTint;
            sampler2D _EnvLumTex;
            float4 _LightDirWorld;
            float _EnvLuminance;
            float _EnvStrength;
            float _EnvMirror;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 normal : TEXCOORD1;
                float3 viewDir : TEXCOORD2;
                float4 screenPos : TEXCOORD3;
                float3 objPos : TEXCOORD4;
            };

            v2f vert(appdata_base v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.texcoord;
                o.normal = UnityObjectToWorldNormal(v.normal);
                o.viewDir = normalize(_WorldSpaceCameraPos.xyz - mul(unity_ObjectToWorld, v.vertex).xyz);
                o.screenPos = ComputeScreenPos(o.pos);
                o.objPos = v.vertex.xyz;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float4 mask = tex2D(_MaskTex, i.uv);
                float coverage = mask.r;
                if (coverage < 0.004) discard;

                // 渐变取色：向心度（中心 1 → 边缘 0）映射到色带
                fixed3 col = tex2D(_RampTex, float2(mask.g, 0.5)).rgb;

                // 粉感噪点：用物体空间坐标平铺（近似切线空间），转头时颗粒不在脸上"游走"
                float2 grainUv = i.objPos.xy * 90.0 + i.objPos.z * 30.0;
                float grain = tex2D(_GrainTex, grainUv).r - 0.5;
                col *= 1.0 + grain * _GrainStrength * 0.6;
                float alpha = coverage * _Opacity * (1.0 + grain * _GrainStrength * 0.25);

                // ---------- 环境光合成 ----------
                float3 n = normalize(i.normal);
                float3 v = normalize(i.viewDir);
                float3 l = normalize(_LightDirWorld.xyz + float3(0, 0, -1e-4));
                float env = saturate(_EnvStrength);

                // 房间色温
                float3 tint = lerp(float3(1, 1, 1), saturate(_EnvTint.rgb), env);
                col *= tint;

                // 屏幕空间亮度：妆容跟随脸上真实光影（光照侧亮、阴影侧暗）
                float2 suv = i.screenPos.xy / max(i.screenPos.w, 1e-4);
                suv.x = lerp(suv.x, 1.0 - suv.x, _EnvMirror);
                float lum = dot(tex2D(_EnvLumTex, suv).rgb, float3(0.299, 0.587, 0.114));
                float lumRef = max(_EnvLuminance, 0.05);
                float shade = lum / lumRef;                       // 1 = 平均亮度
                shade = lerp(1.0, clamp(shade, 0.45, 1.6), env * 0.8);
                col *= shade;

                // wrap diffuse（次表面散射近似：阴影侧不死黑）
                float wrap = 0.35;
                float ndl = saturate((dot(n, l) + wrap) / (1.0 + wrap));
                float diffuse = lerp(1.0, 0.55 + 0.45 * ndl, env);
                col *= diffuse;

                // ---------- 双层高光 ----------
                float3 h = normalize(l + v);
                float ndh = saturate(dot(n, h));
                float fres = pow(1.0 - saturate(dot(n, v)), 3.0);
                // 底层 sheen（粉质广域微光）：matte 0.04 → gloss 0.52
                float sheenStrength = 0.04 + _Finish * 0.16;
                float sheenSharp = 8.0 + _Finish * 24.0;
                float sheen = fres * sheenStrength + pow(fres, sheenSharp) * (0.15 + _Finish * 0.35);
                // 清漆层（dewy/gloss 才有）：窄而亮的镜面高光
                float clearcoat = saturate(_Finish - 1.0) * 0.5;            // dewy 0.5, gloss 1.0
                float spec = pow(ndh, 60.0 + _Finish * 60.0) * clearcoat * lerp(0.4, 1.0, env);
                col += (sheen + spec).xxx * lerp(0.6, 1.0, shade);

                // ---------- 脸缘羽化 / 全局强度 / 丢脸淡出 ----------
                float edge = tex2D(_FaceEdgeTex, i.uv).r;
                alpha *= smoothstep(0.0, 1.0, edge);
                alpha *= _MakeupIntensity * _FacePresence;

                return fixed4(saturate(col), saturate(alpha));
            }
            ENDCG
        }
    }
    FallBack Off
}
