// WebcamBackground.shader — 摄像头画面背景（可镜像，照镜子习惯）。
Shader "MakeupMirror/WebcamBackground"
{
    Properties
    {
        _MainTex ("Webcam", 2D) = "black" {}
        _Mirror ("Mirror (1=左右翻转)", Float) = 1
    }
    SubShader
    {
        Tags { "Queue" = "Background" "RenderType" = "Opaque" }
        Cull Off
        ZWrite Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float _Mirror;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata_base v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.texcoord;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float2 uv = i.uv;
                uv.x = lerp(uv.x, 1.0 - uv.x, saturate(_Mirror));
                return tex2D(_MainTex, uv);
            }
            ENDCG
        }
    }
    FallBack Off
}
