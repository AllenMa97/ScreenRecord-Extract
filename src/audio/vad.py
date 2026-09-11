"""自适应能量语音活动检测 (Adaptive Energy Voice Activity Detection, VAD)

用途 (Purpose):
  对已重采样为 16000Hz 单声道 float64 PCM 的音轨, 把"连续语音"切成若干语音段,
  每段返回 (start_秒, end_秒)。这些段按视频时间轴映射回画面, 供逐段 ASR 转写。
  ASR 服务端不返回时间戳, 因此切段完全由本模块在客户端完成。

算法思想 (Algorithm idea):
  1) 分帧: 按 frame_ms 将样本切成固定长帧, 逐帧计算 RMS 能量。
  2) 自适应阈值: 不做绝对能量硬编码(真实语音音量差异大)。用能量分布估计噪声基底,
     取帧能量最低的一个小比例(trimmed-min), 语音阈值 = 噪声基底 × factor。
     这样对"全静音"输入阈值仍很低, 但能量恒等于 0, 不会触发任何语音检出。
  3) 状态机: silent→speech 需连续若干帧越阈(去抖, 防单帧误触);
     speech→silent 需连续低于阈值 hold_ms 对应帧数(hangover/拖尾),
     使语音中间几帧能量低谷或短停顿不被切段。
  4) 过滤: 丢弃时长 < min_utterance_ms 的孤立微段(空拍/点触声)。
  5) 合并: 相邻段间静音 ≤ min_silence_ms 时合并成一段。

纯函数、确定性、CPU 友好: 逐帧逻辑用 numpy 向量化, 无随机数,
入参相同则输出逐位一致。
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

# 自适应阈值的内部常量(确定性, 可通过测试设置)
NOISE_PERCENTILE_FRACTION = 0.02   # 能量最低的 2% 均值作为噪声基底
THRESHOLD_FACTOR = 10.0            # 语音阈值 = 噪声基底 × factor
ABSOLUTE_FLOOR = 1e-6              # 绝对保护下限, 防止浮点噪声误触
DEBOUNCE_PRE_FRAMES = 3            # 判定进入语音前需连续越阈的帧数(去抖)


def _frame_len(sample_rate: int, frame_ms: int) -> int:
    """由采样率与帧毫秒数计算每帧样本长度(至少为 1)。"""
    return max(1, int(round(sample_rate * frame_ms / 1000.0)))


def _frame_energies(samples: np.ndarray, frame_len: int) -> np.ndarray:
    """将样本按 frame_len 分帧, 返回逐帧 RMS 能量(标量 float64 数组)。"""
    n = samples.size
    if n == 0 or frame_len <= 0:
        return np.zeros(0, dtype=np.float64)
    n_frames = (n + frame_len - 1) // frame_len
    pad = n_frames * frame_len - n
    if pad:
        # 尾部不足一帧则补零, 保证每一帧长度一致, 便于向量化求 RMS
        samples = np.concatenate([samples, np.zeros(pad, dtype=samples.dtype)])
    squared = samples.reshape(n_frames, frame_len) ** 2
    return np.sqrt(squared.mean(axis=1))


def compute_energy(samples: np.ndarray, frame_ms: int = 20) -> np.ndarray:
    """将浮点 PCM 按 frame_ms 分帧, 返回逐帧 RMS 能量(标量数组)。"""
    arr = np.asarray(samples, dtype=np.float64)
    flen = _frame_len(16000, frame_ms)
    return _frame_energies(arr, flen)


def _estimate_threshold(energies: np.ndarray,
                        energy_floor: Optional[float]) -> float:
    """由帧能量分布估计语音阈值(自适应, 无需外部校准)。

    取能量最低的 2% 均值作为噪声基底, 乘以 factor。因为真实"静音/背景噪声"
    是能量分布的极低尾, 取较小比例能稳定落在噪声区; 即便信号几乎恒定,
    只要存在极少量低能量帧, 也能正确把噪声基底拉回近零。
    """
    n = energies.size
    n_noise = max(1, int(round(n * NOISE_PERCENTILE_FRACTION)))
    n_noise = min(n_noise, max(1, n))
    sorted_e = np.sort(energies)
    noise_floor = float(np.mean(sorted_e[:n_noise]))

    floor = ABSOLUTE_FLOOR
    if energy_floor is not None and float(energy_floor) > floor:
        floor = float(energy_floor)
    return max(floor, noise_floor * THRESHOLD_FACTOR)


def detect_utterances(
    samples: np.ndarray,
    sample_rate: int = 16000,
    *,
    frame_ms: int = 20,
    energy_floor: Optional[float] = None,
    hold_ms: int = 250,          # 语音结束后需连续静音这么久才判段结束(拖尾吸收短停顿)
    min_utterance_ms: int = 300, # 短于此的候选段被丢弃
    min_silence_ms: int = 200,   # 两段语音间静音少于此时长则合并为一段
) -> List[Tuple[float, float]]:
    """自适应能量 VAD, 返回 [(start_秒, end_秒), ...]。

    空输入或整段无语音均返回 []。end 不超过音频总时长。
    """
    if samples is None:
        return []
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    total = arr.size
    if total == 0:
        return []

    flen = _frame_len(sample_rate, frame_ms)
    energies = _frame_energies(arr, flen)
    threshold = _estimate_threshold(energies, energy_floor)
    voiced = energies > threshold

    n_frames = energies.size
    debounce_pre = max(1, int(DEBOUNCE_PRE_FRAMES))
    hangover = max(1, int(round(hold_ms / float(frame_ms))))
    min_utt_frames = max(1, int(round(min_utterance_ms / float(frame_ms))))
    min_sil_frames = max(1, int(round(min_silence_ms / float(frame_ms))))

    # ---------- 状态机(基于"连续同值区间"run 扫描, 帧数级循环开销极小) ----------
    runs: List[Tuple[bool, int, int]] = []
    if n_frames > 0:
        start = 0
        cur = bool(voiced[0])
        for i in range(1, n_frames):
            if bool(voiced[i]) != cur:
                runs.append((cur, start, i))
                cur = bool(voiced[i])
                start = i
        runs.append((cur, start, n_frames))

    # segments 元素为 (start_frame, end_frame, core_end)。
    # core_end 是本段实际越阈语音的结束帧(hangover 之前), 用于 min_utterance 过滤:
    # 拖尾会把孤立微段"撑胖", 若按整体 span 过滤会漏掉, 故按语音核心长度判丢弃。
    segments: List[Tuple[int, int, int]] = []
    open_start: Optional[int] = None
    for is_voice, a, b in runs:
        length = b - a
        if is_voice:
            if open_start is None:
                if length >= debounce_pre:
                    # 去抖: 足够长的连续越阈才真正判为语音起始
                    open_start = a
            # 已在段内: 该语音区间顺延段长(即便很短也不切断, 吸收语速快的短句)
        else:
            if open_start is not None and length >= hangover:
                # 拖尾: 连续静音达到 hold_ms 才结束该段, 并保留 hangover 尾帧;
                # 该静音段起点 a 即语音核心的结束帧。
                segments.append((open_start, a + hangover, a))
                open_start = None
    if open_start is not None:
        segments.append((open_start, n_frames, n_frames))

    # ---------- 合并: 相邻段静音 <= min_silence_ms ----------
    merged: List[Tuple[int, int, int]] = []
    for s, e, ce in segments:
        if merged and (s - merged[-1][1]) <= min_sil_frames:
            merged[-1] = (merged[-1][0], e, ce)
        else:
            merged.append((s, e, ce))

    # ---------- 过滤: 丢弃过短的孤立微段(空拍/点触声), 按语音核心长度判定 ----------
    dt = float(frame_ms) / 1000.0
    audio_dur = total / float(sample_rate)
    out: List[Tuple[float, float]] = []
    for s, e, ce in merged:
        if (ce - s) < min_utt_frames:
            continue
        start_s = float(s) * dt
        end_s = min(float(e) * dt, audio_dur)
        if end_s > start_s:
            out.append((start_s, end_s))
    return out