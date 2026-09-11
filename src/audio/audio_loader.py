"""
音频加载器 (Audio Loader)

用 PyAV(自带完整 FFmpeg)解码任意媒体文件(视频 mp4/mov/mkv 或纯音频 mp3/wav/...)
的音轨, 重采样为单声道 float64 PCM, 供下游 VAD/ASR 使用。

背景: 本项目系统 PATH 里的 ffmpeg 是精简版(仅视频解码, 无音频能力),
OpenCV 也只读视频帧。因此音轨解码统一走 PyAV 打包的完整 FFmpeg。
"""
from __future__ import annotations

import numpy as np
import av

from .data_types import AudioDecoded


class AudioLoadError(RuntimeError):
    """音轨加载失败。"""


def has_audio_track(path: str) -> bool:
    """判定媒体文件是否含有可解码的音轨(不真正解码音频样本)。"""
    try:
        container = av.open(str(path))
        try:
            return any(s.type == "audio" for s in container.streams)
        finally:
            container.close()
    except Exception:
        return False


def decode_audio(path: str, target_rate: int = 16000) -> AudioDecoded:
    """解码媒体文件音轨 → 单声道 float64 PCM(重采样到 target_rate)。

    返回 AudioDecoded, samples 时间轴与视频时间轴对齐
    (ts = sample_index / sample_rate)。

    若媒体无音轨, 返回 has_audio=False 的 AudioDecoded(空 samples)。
    若文件打不开或解码出错, 抛 AudioLoadError。
    """
    try:
        container = av.open(str(path))
    except Exception as e:
        raise AudioLoadError(f"无法打开媒体文件 / Cannot open media: {path}: {e}")

    audio_streams = [s for s in container.streams if s.type == "audio"]
    if not audio_streams:
        container.close()
        return AudioDecoded(samples=np.zeros(0, dtype=np.float64),
                            sample_rate=target_rate,
                            audio_duration=0.0,
                            has_audio=False)

    # 选"样本数最多"的音轨(通常是主音轨; 部分文件同时含多音轨)
    stream = max(audio_streams, key=lambda s: s.duration or 0)

    resampler = av.AudioResampler(format="dbl", layout="mono", rate=target_rate)
    out = []
    try:
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                arr = rf.to_ndarray()      # shape (1, n) after mono resample
                out.append(arr.reshape(-1))
    except Exception as e:
        raise AudioLoadError(f"音轨解码失败 / Audio decode failed: {path}: {e}")
    finally:
        # 刷出剩余缓冲
        for rf in resampler.resample(None):
            out.append(rf.to_ndarray().reshape(-1))
        container.close()

    if not out:
        return AudioDecoded(samples=np.zeros(0, dtype=np.float64),
                            sample_rate=target_rate,
                            audio_duration=0.0,
                            has_audio=True)
    samples = np.concatenate(out)
    return AudioDecoded(samples=samples,
                        sample_rate=target_rate,
                        audio_duration=len(samples) / float(target_rate),
                        has_audio=True)


def extract_segment(samples: np.ndarray, sample_rate: int,
                    start_s: float, dur_s: float) -> np.ndarray:
    """从完整 PCM 中截取 [start_s, start_s+dur_s) 片段(用于分片上传 ASR)。"""
    i0 = max(0, int(round(start_s * sample_rate)))
    i1 = min(len(samples), int(round((start_s + dur_s) * sample_rate)))
    if i1 <= i0:
        return samples[0:0]
    return samples[i0:i1]