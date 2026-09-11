"""
音频转写编排器 (Audio Transcriptor)

把一段媒体(视频或纯音频)变成带视频时间轴锚点的逐段语音转写文本:

    decode → VAD 切段 → (超长段按时长细分) → 并发逐段 ASR → VoiceUtterance[]

设计要点:
  - ASR 服务端不返回时间戳, 因此时间轴完全由客户端 VAD 切段决定; 逐段独立转写
    即可把每段文本精确锚定到视频时间轴, 后续与视觉侧"语音气泡"按时间就近配对。
  - 服务端单请求音频过大(实测 >25MB)会 CUDA OOM, 故对超长段按 max_segment_seconds
    硬切成更小段, 保证每请求体积在安全区间。
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

import numpy as np

from .asr_client import AsrClient
from .audio_loader import AudioLoadError, decode_audio, has_audio_track
from .data_types import AudioDecoded, VoiceUtterance
from . import vad as _vad

logger = logging.getLogger(__name__)


class TranscriptResult:
    """一次音频转写任务的结果。"""

    def __init__(self, source: str, duration: float = 0.0,
                 has_audio: bool = False, reason: str = "",
                 segments: Optional[List[VoiceUtterance]] = None,
                 vad_utterances: int = 0,
                 rms_level: float = 0.0):
        self.source = source
        self.duration = duration
        self.has_audio = has_audio
        self.reason = reason          # 空串=正常; 否则为跳过原因
        self.segments = segments or []
        self.vad_utterances = vad_utterances
        self.rms_level = rms_level    # 全片均方根能量(线性), 供前端区分静音/噪声

    def to_dict(self) -> dict:
        text_segments = sum(
            1 for s in self.segments if (s.text or "").strip()
        )
        return {
            "source": self.source,
            "duration_s": self.duration,
            "has_audio": self.has_audio,
            "reason": self.reason,
            "enabled": bool(self.has_audio and self.segments),
            "vad_utterances": self.vad_utterances,
            "rms_level": self.rms_level,
            "text_segments": text_segments,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text, "events": s.events}
                for s in self.segments
            ],
        }


def _split_long(utt: tuple, max_seconds: float,
                max_bytes: Optional[int] = None, sr: int = 16000) -> List[tuple]:
    """把 (t0,t1) 超长语音段按时长/字节上限硬切为子段。

    字节上限也作为切分依据: 即使时长在客户端范围内, 一旦编码成 16bit wav
    超过单请求字节上限, 仍要切成更小段(防服务端 OOM, 见 M3)。
    """
    t0, t1 = utt
    max_sub = max_seconds
    if max_bytes and sr > 0:
        # 16bit 单声道: 每样本 2 字节 → 每请求最多 max_bytes/2/sr 秒
        byte_limit_s = max_bytes / (2.0 * sr)
        if byte_limit_s > 0:
            max_sub = min(max_sub, byte_limit_s)
    if t1 - t0 <= max_sub:
        return [(t0, t1)]
    out = []
    cur = t0
    while cur < t1:
        nxt = min(t1, cur + max_sub)
        out.append((cur, nxt))
        cur = nxt
    return out


def _slice(samples: np.ndarray, sr: int, t0: float, t1: float) -> np.ndarray:
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(samples), int(round(t1 * sr)))
    return samples[i0:i1] if i1 > i0 else samples[0:0]


def transcribe_media(
    path: str,
    asr_client: AsrClient,
    *,
    sample_rate: int = 16000,
    max_segment_seconds: float = 360.0,
    max_request_bytes: Optional[int] = None,
    concurrency: int = 2,
    vad_kwargs: Optional[dict] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> TranscriptResult:
    """对媒体文件做"VAD切段 → 逐段转写", 返回完整结果。

    参数:
        path: 媒体文件路径(视频或纯音频)。
        asr_client: 已配置好模型/密钥的 ASR 客户端。
        sample_rate: 解码/转写使用的采样率(与 ASR 服务期望一致, 默认16k)。
        max_segment_seconds: 单请求最长语音段时长(更长则硬切)。默认360s。
        max_request_bytes: 单请求上传字节上限(超过则按时长/字节硬切, 防服务端OOM)。
        concurrency: 并发转写请求数。默认2(GPUStack 显存有限, 保守不并发过大)。
        vad_kwargs: 透传给 vad.detect_utterances 的参数。
        progress: 可选回调 (done, total), 供上层展示进度。
        is_cancelled: 可选回调; 返回 True 时中途停止调度剩余的 ASR 段
            (用于 Web 用户删除运行中会话时, 避免已删会话继续占用 ASR/GPU)。

    返回: TranscriptResult(包含带时间锚点的语音段列表)。
    """
    base = os.path.basename(path or "")
    if not path or not os.path.exists(path):
        return TranscriptResult(source=base, reason=f"文件不存在: {path}")

    try:
        decoded = decode_audio(path, sample_rate)
    except AudioLoadError as e:
        return TranscriptResult(source=base, reason=f"音轨加载失败: {e}")
    if not decoded.has_audio:
        return TranscriptResult(source=base, duration=0.0, has_audio=False,
                                reason="媒体无音轨或音轨不可解码")

    samples: np.ndarray = decoded.samples
    sr = decoded.sample_rate
    duration = decoded.duration
    vk = dict(vad_kwargs or {})
    max_bytes = max_request_bytes or getattr(asr_client, "max_request_bytes", None)
    try:
        utts = _vad.detect_utterances(samples, sr, **vk)
    except Exception as e:
        logger.warning("VAD 失败, 回退整段转写: %s", e)
        utts = [(0.0, min(duration, max_segment_seconds))]

    # 超长段硬切(按时长与单请求字节上限)
    tasks = []  # [(t0,t1)]
    for u in utts:
        tasks.extend(_split_long(u, max_segment_seconds, max_bytes, sr))

    # 整段无语音: 若文件很短则整段转写一次, 捕捉纯音乐/其余标记
    if not tasks and duration <= max_segment_seconds:
        tasks = [(0.0, duration)]

    total = len(tasks)
    results: List[VoiceUtterance] = []

    def _one(seg):
        t0, t1 = seg
        seg_pcm = _slice(samples, sr, t0, t1)
        try:
            return asr_client.transcribe_pcm(seg_pcm, sr, t0, t1)
        except Exception as e:
            logger.warning("段[%.2f,%.2f] 转写失败: %s", t0, t1, e)
            return VoiceUtterance(start=t0, end=t1, text="", events=[])

    if total <= 1:
        if is_cancelled is not None and is_cancelled():
            return TranscriptResult(source=base, duration=duration,
                                    has_audio=True, segments=[],
                                    vad_utterances=len(utts),
                                    rms_level=0.0, reason="已取消")
        for seg in tasks:
            results.append(_one(seg))
    else:
        # 用 submit+shutdown(cancel_futures) 手动编排, 便于取消时不再调度
        # 剩余 ASR 段(shutdown 不等待正在运行的请求, wait=False 直接返回)。
        ex = ThreadPoolExecutor(max_workers=max(1, concurrency))
        try:
            fut_by_seg = [(seg, ex.submit(_one, seg)) for seg in tasks]
            done = 0
            for seg, fut in fut_by_seg:
                if is_cancelled is not None and is_cancelled():
                    break
                results.append(fut.result())
                done += 1
                if progress:
                    try:
                        progress(done, total)
                    except Exception:
                        pass
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    results.sort(key=lambda u: u.start)
    # 全片均方根能量(线性): 供上层区分"静音"(≈0) 与 "有声音(含噪声/人声)"。
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    return TranscriptResult(
        source=base, duration=duration, has_audio=True,
        segments=results, vad_utterances=len(utts),
        rms_level=rms,
    )