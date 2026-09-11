// OneEuroFilter.cs
// 1€ 滤波（Casiez 2012）：自适应低通，静止时强平滑、快速运动时低延迟。
// 与 unity-app/tools/tracking_protocol.py 的参考实现同一算法（参数含义一致）。
using UnityEngine;

namespace MakeupMirror
{
    /// 标量 1€ 滤波器
    public struct OneEuroScalar
    {
        public float minCutoff;   // 静止截止频率（越小越稳）
        public float beta;        // 速度→截止增益（越大越跟手）
        public float dCutoff;

        private bool _init;
        private float _x, _dx, _t;

        public OneEuroScalar(float minCutoff, float beta, float dCutoff = 1f) : this()
        {
            this.minCutoff = minCutoff;
            this.beta = beta;
            this.dCutoff = dCutoff;
        }

        private static float Alpha(float cutoff, float dt)
        {
            float tau = 1f / (2f * Mathf.PI * cutoff);
            return 1f / (1f + tau / dt);
        }

        public float Filter(float x, float t)
        {
            if (!_init || t <= _t)
            {
                _init = true;
                _x = x; _dx = 0f; _t = t;
                return x;
            }
            float dt = t - _t;
            _t = t;
            float dx = (x - _x) / dt;
            float aD = Alpha(dCutoff, dt);
            _dx = aD * dx + (1f - aD) * _dx;
            float cutoff = minCutoff + beta * Mathf.Abs(_dx);
            float a = Alpha(cutoff, dt);
            _x = a * x + (1f - a) * _x;
            return _x;
        }

        public void Reset() => _init = false;
    }

    /// 向量 1€ 滤波（三分量独立滤波）
    public struct OneEuroVec3
    {
        private OneEuroScalar _x, _y, _z;

        public OneEuroVec3(float minCutoff, float beta, float dCutoff = 1f) : this()
        {
            _x = new OneEuroScalar(minCutoff, beta, dCutoff);
            _y = new OneEuroScalar(minCutoff, beta, dCutoff);
            _z = new OneEuroScalar(minCutoff, beta, dCutoff);
        }

        public Vector3 Filter(Vector3 v, float t) => new Vector3(
            _x.Filter(v.x, t), _y.Filter(v.y, t), _z.Filter(v.z, t));

        public void Reset() { _x.Reset(); _y.Reset(); _z.Reset(); }
    }

    /// 关键点阵 1€ 滤波（468×3），共享同一套参数。
    /// 注意尺度：点数据是米——beta 的单位是 Hz/(m/s)，典型头部运动 0.02m/s 需要几十才有效
    public class OneEuroCloud
    {
        private OneEuroScalar[] _f;
        private readonly float _minCutoff, _beta, _dCutoff;

        public OneEuroCloud(float minCutoff = 0.8f, float beta = 75f, float dCutoff = 1f)
        {
            _minCutoff = minCutoff; _beta = beta; _dCutoff = dCutoff;
        }

        public void Ensure(int count)
        {
            if (_f != null && _f.Length >= count * 3) return;
            _f = new OneEuroScalar[count * 3];
            for (int i = 0; i < _f.Length; i++)
                _f[i] = new OneEuroScalar(_minCutoff, _beta, _dCutoff);
        }

        public void Filter(Vector3[] pts, float t)
        {
            if (_f == null || _f.Length < pts.Length * 3) Ensure(pts.Length);
            for (int i = 0; i < pts.Length; i++)
            {
                int k = i * 3;
                pts[i] = new Vector3(_f[k].Filter(pts[i].x, t),
                                     _f[k + 1].Filter(pts[i].y, t),
                                     _f[k + 2].Filter(pts[i].z, t));
            }
        }

        public void Reset()
        {
            if (_f == null) return;
            for (int i = 0; i < _f.Length; i++) _f[i].Reset();
        }
    }

    /// 姿态平滑：平移用 1€，旋转用自适应 Slerp（角速度越快越贴近原始值）
    public struct PoseSmoother
    {
        private OneEuroScalar _tx, _ty, _tz;
        private Quaternion _q;
        private bool _init;

        public void Reset() => _init = false;

        public (Vector3 pos, Quaternion rot) Filter(Vector3 pos, Quaternion rot, float t)
        {
            if (!_init)
            {
                _init = true;
                _tx = new OneEuroScalar(1.0f, 75f);
                _ty = new OneEuroScalar(1.0f, 75f);
                _tz = new OneEuroScalar(1.0f, 75f);
                _q = rot;
                return (pos, rot);
            }
            var p = new Vector3(_tx.Filter(pos.x, t), _ty.Filter(pos.y, t), _tz.Filter(pos.z, t));
            float angle = Quaternion.Angle(_q, rot);
            // 3°/帧以内平滑追赶；突变（>60°）直接采纳，防止滤波"拖影"
            float alpha = Mathf.Clamp01(angle / 60f);
            _q = Quaternion.Slerp(_q, rot, Mathf.Max(alpha, 0.35f));
            return (p, _q);
        }
    }
}
