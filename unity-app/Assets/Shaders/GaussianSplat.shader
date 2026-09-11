// GaussianSplat.shader — 高斯溅射面片（GPU 实例化）：各向异性高斯核 alpha 衰减。
// SplatLayerRenderer 每帧按视深排序后用 Graphics.RenderMeshInstanced 批量绘制：
// 每实例矩阵 = 切平面朝向（长轴沿唇线/区域走向，短轴垂直）× 3σ 尺寸；
// 每实例颜色 _Color（a=核峰值 alpha）走 MaterialPropertyBlock 数组。
// 全局：_MakeupIntensity（滑杆）、_FacePresence（丢脸淡出）、_EnvTint（环境色温）。
Shader "MakeupMirror/GaussianSplat"
{
    Properties
    {
        _Color ("Color (a = peak alpha)", Color) = (0.8, 0.3, 0.35, 0.7)
        _Sharpness ("Kernel sharpness", Range(0.5, 2.0)) = 1.0
    }
    SubShader
    {
        Tags { "Queue" = "Transparent+100" "RenderType" = "Transparent" "IgnoreProjector" = "True" }
        ZWrite Off
        ZTest LEqual
        Cull Off
        Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma target 3.0
            #include "UnityCG.cginc"

            float _Sharpness;
            float _MakeupIntensity;
            float _FacePresence;
            float4 _EnvTint;
            float _EnvStrength;

            UNITY_INSTANCING_BUFFER_START(Props)
                UNITY_DEFINE_INSTANCED_PROP(fixed4, _Color)
            UNITY_INSTANCING_BUFFER_END(Props)

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(i);
                fixed4 c = UNITY_ACCESS_INSTANCED_PROP(Props, _Color);
                // 面片全宽 = 3σ：uv 中心 0.5 ↔ 0σ，边缘 ↔ 1.5σ（长短轴由实例矩阵缩放决定）
                float2 d = (i.uv - 0.5) * 3.0;
                float r2 = dot(d, d) * _Sharpness;
                float g = exp(-r2 * 0.5);
                c.rgb *= lerp(float3(1, 1, 1), saturate(_EnvTint.rgb), saturate(_EnvStrength));
                c.a = saturate(c.a * g) * _MakeupIntensity * _FacePresence;
                return c;
            }
            ENDCG
        }
    }
    FallBack Off
}
