// CoachingDisplay.cs
// 实时指导显示：真正的消息队列（warn 插队、同文 60s 去重、可提前推进），info/warn 两级配色，淡出；
// 步骤进度条（live_coach 的 coaching.progress/step 字段）；
// TTS：Windows 上优先反射加载 System.Speech（进程内、低延迟），失败退回 PowerShell SAPI 子进程；
// 其它平台仅日志。speak=true 且 priority=warn/done 才播报，避免絮叨。
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

namespace MakeupMirror
{
    public class CoachingDisplay : MonoBehaviour
    {
        [Header("显示")]
        public float holdSeconds = 4.0f;
        public float fadeSeconds = 0.6f;
        [Tooltip("新消息到来时，当前消息已显示超过此秒数则提前推进")]
        public float minShowSeconds = 1.8f;

        [Header("语音")]
        public bool ttsEnabled = true;
        [Tooltip("只播报 warn 与步骤完成，info 静音（false 则全部播报）")]
        public bool ttsWarnAndDoneOnly = true;
        [Tooltip("同一句话在此秒数内不重复显示/播报")]
        public float dedupeSeconds = 60f;

        public int QueueLength => _queue.Count;
        public string TtsBackend { get; private set; } = "none";

        private struct Msg
        {
            public string text;
            public string priority;
            public bool speak;
            public string note;
        }

        private Text _text;
        private Image _progressFill;
        private Text _progressLabel;
        private readonly List<Msg> _queue = new List<Msg>();
        private readonly Dictionary<string, float> _recent = new Dictionary<string, float>();
        private Coroutine _runner;
        private float _shownAt = -100f;
        private bool _showing;

        private object _synth;             // System.Speech.Synthesis.SpeechSynthesizer（反射）
        private MethodInfo _speakAsync;
        private bool _synthTried;

        public void Bind(Text text) => _text = text;

        public void BindProgress(Image fill, Text label)
        {
            _progressFill = fill;
            _progressLabel = label;
            SetProgress(null, 0, 0, 0f);
        }

        // ---------- 队列 ----------

        public void Show(string message, string priority, bool speak, string note = null)
        {
            if (_text == null || string.IsNullOrWhiteSpace(message)) return;
            priority = priority ?? "info";
            note = note ?? "";

            // 去重：同一句 dedupeSeconds 内不再打断
            if (_recent.TryGetValue(message, out float last) && Time.time - last < dedupeSeconds && note != "done")
                return;
            _recent[message] = Time.time;
            if (_recent.Count > 64) PruneRecent();

            var m = new Msg { text = message, priority = priority, speak = speak, note = note };
            if (priority == "warn") _queue.Insert(0, m);   // 警示插队
            else _queue.Add(m);
            if (_queue.Count > 4) _queue.RemoveAt(_queue.Count - 1);

            // 当前消息已经看够了 → 提前推进
            if (_showing && Time.time - _shownAt > minShowSeconds)
            {
                if (_runner != null) StopCoroutine(_runner);
                _showing = false;
                _runner = StartCoroutine(Runner());
            }
            else if (!_showing)
            {
                _runner = StartCoroutine(Runner());
            }
        }

        private void PruneRecent()
        {
            var stale = new List<string>();
            foreach (var kv in _recent)
                if (Time.time - kv.Value > dedupeSeconds) stale.Add(kv.Key);
            foreach (var k in stale) _recent.Remove(k);
        }

        private IEnumerator Runner()
        {
            while (_queue.Count > 0)
            {
                var m = _queue[0];
                _queue.RemoveAt(0);
                _showing = true;
                _shownAt = Time.time;
                _text.text = m.text;
                _text.color = m.priority == "warn" ? new Color(1f, 0.62f, 0.2f)
                            : m.note == "done" ? new Color(0.55f, 0.95f, 0.6f)
                            : Color.white;
                SetAlpha(1f);
                if (m.speak && ttsEnabled && (!ttsWarnAndDoneOnly || m.priority == "warn" || m.note == "done"))
                    Speak(m.text);

                yield return new WaitForSeconds(holdSeconds);
                float t = 0f;
                while (t < fadeSeconds)
                {
                    if (_queue.Count > 0) break;   // 有新消息：立刻切换
                    t += Time.deltaTime;
                    SetAlpha(1f - t / fadeSeconds);
                    yield return null;
                }
            }
            SetAlpha(0f);
            _showing = false;
            _runner = null;
        }

        private void SetAlpha(float a)
        {
            if (_text == null) return;
            var c = _text.color; c.a = a;
            _text.color = c;
        }

        // ---------- 进度 ----------

        /// 步骤进度（live_coach 推送）；stepName 为空则隐藏
        public void SetProgress(string stepName, int step, int total, float progress01)
        {
            if (_progressFill == null) return;
            bool visible = !string.IsNullOrEmpty(stepName) && total > 0;
            _progressFill.transform.parent.gameObject.SetActive(visible);
            if (!visible) return;
            _progressFill.fillAmount = Mathf.Clamp01(progress01);
            if (_progressLabel != null)
                _progressLabel.text = $"步骤 {step}/{total} · {stepName} · {Mathf.RoundToInt(progress01 * 100)}%";
        }

        // ---------- TTS ----------

        private void Speak(string text)
        {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            if (!_synthTried) TrySystemSpeech();
            if (_synth != null && _speakAsync != null)
            {
                try { _speakAsync.Invoke(_synth, new object[] { text }); return; }
                catch (Exception e) { Debug.LogWarning($"[tts] System.Speech 失败，改用 SAPI 子进程：{e.Message}"); _synth = null; }
            }
            TtsBackend = "powershell-sapi";
            ThreadPool.QueueUserWorkItem(_ => SpeakViaPowerShell(text));
#else
            TtsBackend = "log";
            Debug.Log($"[coaching TTS] {text}");
#endif
        }

#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        private void TrySystemSpeech()
        {
            _synthTried = true;
            try
            {
                var asm = Assembly.Load("System.Speech");
                var type = asm?.GetType("System.Speech.Synthesis.SpeechSynthesizer");
                if (type == null) return;
                _synth = Activator.CreateInstance(type);
                type.GetMethod("SetOutputToDefaultAudioDevice")?.Invoke(_synth, null);
                _speakAsync = type.GetMethod("SpeakAsync", new[] { typeof(string) });
                if (_speakAsync != null) TtsBackend = "system.speech";
            }
            catch (Exception)
            {
                _synth = null;   // Unity 的 .NET 配置未带 System.Speech：走 PowerShell
            }
        }

        private static void SpeakViaPowerShell(string text)
        {
            try
            {
                string safe = text.Replace("'", "''");
                var psi = new ProcessStartInfo
                {
                    FileName = "powershell",
                    Arguments = "-NoProfile -NonInteractive -WindowStyle Hidden -Command "
                                + "\"Add-Type -AssemblyName System.Speech; "
                                + "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                                + $"$s.Rate = 1; $s.Speak('{safe}')\"",
                    CreateNoWindow = true,
                    UseShellExecute = false,
                };
                using (var p = Process.Start(psi)) { p?.WaitForExit(15000); }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[tts] PowerShell SAPI 失败：{e.Message}");
            }
        }
#endif
    }
}
