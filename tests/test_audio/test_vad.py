"""VAD 组件测试 (直接 python 运行, 不依赖 pytest)。

覆盖:
  (a) 纯静音 -> []
  (b) 两个分明的音调爆发(各1秒), 间隔 2 秒静音 -> 检出 2 段, 段位近似覆盖发声区
  (c) 短促噪声(<min_utterance_ms=300ms) -> 被过滤
  (d) 复合长段(持续音 + 短暂静音 gated) -> hangover 吸收内部顿挫, 保持 1 段
  (e) 确定性: 同一输入两次调用输出逐位一致
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.audio.vad import compute_energy, detect_utterances  # noqa: E402

SR = 16000


def synth_tone(start_s, dur_s, freq, amp):
    """合成一段已知频率的正弦音(浮点 PCM)。"""
    n = int(round(dur_s * SR))
    t = (np.arange(n) + start_s * SR) / SR
    return amp * np.sin(2 * np.pi * freq * t)


ALL_FAILURES = []


def run_case(name):
    print(f"\n=== {name} ===")

    def check(cond, msg):
        mark = "PASS" if cond else "FAIL"
        print(f"  [{mark}] {msg}")
        if not cond:
            ALL_FAILURES.append(msg)

    return check


# ---------- (a) 纯静音 ----------
check = run_case("(a) 纯静音 -> []")
silence = np.zeros(1 * SR, dtype=np.float64)
res = detect_utterances(silence, SR)
print("  result:", res)
check(res == [], "纯静音应检出为空列表")

# ---------- (b) 两个音调爆发 ----------
check = run_case("(b) 两个音调爆发, 间隔 2 秒静音")
seg = np.concatenate([
    synth_tone(0.0, 1.0, 440.0, 0.5),
    np.zeros(int(2.0 * SR), dtype=np.float64),
    synth_tone(0.0, 1.0, 880.0, 0.5),
])
res = detect_utterances(seg, SR)
print("  result:", res)
check(len(res) == 2, f"应检出 2 段, 实际 {len(res)} 段")
if len(res) >= 1:
    s0, e0 = res[0]
    print(f"  段1: [{s0:.3f},{e0:.3f}]")
    check(0.0 <= s0 <= 0.3, f"段1起点应贴近发声区起点, 实际 {s0:.3f}")
    check(0.9 <= e0 <= 1.5, f"段1终点应覆盖拖尾, 实际 {e0:.3f}")
if len(res) >= 2:
    s1, e1 = res[1]
    print(f"  段2: [{s1:.3f},{e1:.3f}]")
    check(2.8 <= s1 <= 3.3, f"段2起点应贴近第二发声区, 实际 {s1:.3f}")
    check(3.9 <= e1 <= (4.0 + 1e-6), f"段2终点不应超音频长度, 实际 {e1:.3f}")
    check((s1 - e0) > 1.5, "两段间隔应远大于 min_silence, 不被合并")

# ---------- (c) 短促噪声 ----------
check = run_case("(c) 短促噪声 (<300ms) -> 被过滤")
short_noise = np.concatenate([
    synth_tone(0.0, 0.08, 2000.0, 1.0),
    np.zeros(int(0.5 * SR), dtype=np.float64),
])
res = detect_utterances(short_noise, SR)
print("  result:", res)
check(res == [], "短于 min_utterance_ms 的孤立微段应被丢弃")

# ---------- (d) 复合长段 ----------
check = run_case("(d) 复合长段(持续音 + 短暂静音 gated)")
dur_a = 1.0
gap = 0.12          # 内部短暂静音 < hold(250ms) 与 min_silence(200ms)
dur_b = 1.38
long_sig = np.concatenate([
    synth_tone(0.0, dur_a, 300.0, 0.5),
    np.zeros(int(gap * SR), dtype=np.float64),
    synth_tone(0.0, dur_b, 600.0, 0.5),
])
res = detect_utterances(long_sig, SR)
print("  result:", res)
check(len(res) == 1, f"内部顿挫被 hangover 吸收, 应只 1 段, 实际 {len(res)} 段")
if res:
    s0, e0 = res[0]
    print(f"  段: [{s0:.3f},{e0:.3f}] 时长 {e0-s0:.3f}s")
    check(s0 <= 0.2, f"段起点应贴近起始发声, 实际 {s0:.3f}")
    check(e0 >= 2.3 and e0 <= (dur_a + gap + dur_b + 1e-6),
          f"段终点应覆盖全部发声并夹住内部墩挫, 实际 {e0:.3f}")

# ---------- (e) 确定性 ----------
check = run_case("(e) 确定性")
sig = np.concatenate([
    synth_tone(0.0, 1.0, 440.0, 0.5),
    np.zeros(int(2.0 * SR), dtype=np.float64),
    synth_tone(0.0, 1.0, 880.0, 0.5),
])
r1 = detect_utterances(sig, SR)
r2 = detect_utterances(sig, SR)
print("  r1:", r1)
print("  r2:", r2)
check(r1 == r2, "同一输入两次调用应逐位一致")

if ALL_FAILURES:
    print("\nFAILURES:")
    for x in ALL_FAILURES:
        print("  -", x)
    print("FAIL: 存在失败断言")
    sys.exit(1)
else:
    print("\nALL PASS")
    sys.exit(0)