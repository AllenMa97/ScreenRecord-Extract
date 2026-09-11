"""
ASR 语音转文字客户端 (Speech-to-Text Client)

调用 GPUStack 提供的 OpenAI 兼容接口:
    POST {base_url}/audio/transcriptions  (multipart: file + model)

实测特性:
  - 返回纯文本 {"text": "..."}, 不支持 verbose_json 分段时间戳 → 时间轴由
    客户端 VAD 切段得到, 见 transcriptor.py。
  - 单请求音频过大时, 服务端会 CUDA OOM(实测 >25MB 开始崩溃), 需按
    max_request_bytes / max_chunk_duration 分片, 见 chunk 逻辑。
  - 对静音/纯音乐返回 "." 或含 🎼/🎵 等非语音标记。
"""
from __future__ import annotations

import io
import logging
import time
import wave

import numpy as np
import requests

from .data_types import VoiceUtterance

logger = logging.getLogger(__name__)

# SenseVoice 会在结果里插入的非语音符号(音乐/掌声/笑声等)
NON_SPEECH_MARKERS = {"🎼", "🎵", "🎶", "♪", "♫", "😂", "😅", "🎵"}


class AsrError(RuntimeError):
    """ASR 服务调用失败(非瞬时, 重试无效)。"""


class AsrClient:
    """OpenAI 兼容的 ASR 客户端。"""

    def __init__(self, base_url: str, api_key: str, model: str = "sensevoice-small",
                 max_request_bytes: int = 16 * 1024 * 1024,
                 timeout: float = 600.0, max_retries: int = 3,
                 retry_backoff: float = 2.0):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self.max_request_bytes = max_request_bytes
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def _transcribe_bytes(self, wav_bytes: bytes) -> str:
        """上传一段 wav bytes, 返回原始文本。带重试与退避。"""
        if len(wav_bytes) > self.max_request_bytes:
            # M3: 单请求字节上限真正生效, 防服务端 CUDA OOM(实测 >25MB 崩溃).
            raise AsrError(
                f"分段过大: {len(wav_bytes)}B > 上限 {self.max_request_bytes}B, "
                f"需减小分片 / chunk too large"
            )
        url = f"{self._base_url}/audio/transcriptions"
        last_exc = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = requests.post(
                    url, headers=self._headers,
                    files={"file": ("seg.wav", wav_bytes, "audio/wav")},
                    data={"model": self._model, "language": "auto"},
                    timeout=self._timeout,
                )
                if resp.status_code == 413:
                    # Payload Too Large: 单请求过大, 调用方需减小分片
                    raise AsrError(f"413: 音频过大需减小分片 / payload too large: {resp.text[:200]}")
                resp.raise_for_status()
                payload = resp.json()
                text = payload.get("text", "") if isinstance(payload, dict) else ""
                return text.strip()
            except requests.exceptions.HTTPError as e:
                if getattr(e.response, "status_code", None) in (429, 500, 502, 503):
                    backoff = self._retry_backoff * (2 ** (attempt - 1))
                    logger.warning("ASR 瞬时失败(HTTP %s) 第%d次, %s后重试: %s",
                                   getattr(e.response, "status_code", "?"), attempt,
                                   round(backoff, 1), e)
                    last_exc = e
                    time.sleep(backoff)
                    continue
                raise AsrError(f"ASR HTTP 错误: {e}")
            except requests.exceptions.RequestException as e:
                last_exc = e
                logger.warning("ASR 请求异常 第%d次: %s", attempt, e)
                time.sleep(self._retry_backoff * (2 ** (attempt - 1)))
        raise AsrError(f"ASR 调用重试耗尽 / exhausted retries: {last_exc}")

    def transcribe_slice(self, samples: np.ndarray, sample_rate: int,
                         start_s: float, end_s: float) -> VoiceUtterance:
        """转写 PCM samples 中 [start_s, end_s) 这一段, 返回带时间锚点的语音段。

        samples 为完整 PCM(单声道 float64); 切片在内部完成再编码为 wav 上传。
        """
        i0 = max(0, int(round(start_s * sample_rate)))
        i1 = min(len(samples), int(round(end_s * sample_rate)))
        if i1 <= i0:
            return VoiceUtterance(start=min(i0, len(samples) - 1) / sample_rate if len(samples) else start_s,
                                  end=start_s, text="")
        seg = samples[i0:i1]
        wav_bytes = self.pcm_to_wav(seg, sample_rate)
        raw = self._transcribe_bytes(wav_bytes)
        return self._parsed_utterance(raw, start_s, end_s)

    def transcribe_pcm(self, samples: np.ndarray, sample_rate: int,
                       start_s: float, end_s: float) -> VoiceUtterance:
        """转写一段已切好的 PCM(wav 由该段直接编码), 时间锚点为 [start_s, end_s)。"""
        wav_bytes = self.pcm_to_wav(samples, sample_rate)
        raw = self._transcribe_bytes(wav_bytes)
        return self._parsed_utterance(raw, start_s, end_s)

    @staticmethod
    def pcm_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        """float64 PCM → 内存 wav(bytes), 16bit 单声道。"""
        arr = np.asarray(samples, dtype=np.float64)
        if arr.size == 0:
            arr = np.zeros(1, dtype=np.float64)
        clipped = np.clip(arr, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm16.tobytes())
        return buf.getvalue()

    @staticmethod
    def _parsed_utterance(raw_text: str, start_s: float, end_s: float) -> VoiceUtterance:
        """把 ASR 原始文本解析成 VoiceUtterance, 剥离非语音标记到 events。"""
        text = (raw_text or "").strip()
        events = [m for m in NON_SPEECH_MARKERS if m in text]
        # 纯非语音(如仅音乐/静音标记). 去除标记只留实际语言部分
        cleaned = "".join(ch for ch in text if ch not in NON_SPEECH_MARKERS).strip()
        cleaned = cleaned.strip("。" ",.，")
        # SenseVoice 对纯音乐/空白常返回 '.', 判为纯事件
        if cleaned in ("", ".", "…", "..."):
            return VoiceUtterance(start=start_s, end=end_s, text="",
                                  events=events, confidence=0.0)
        return VoiceUtterance(start=start_s, end=end_s, text=cleaned,
                              events=events, confidence=1.0)

    def check_status(self) -> bool:
        """探测服务与模型是否可用(GET /v1/models)。"""
        try:
            r = requests.get(f"{self._base_url}/models",
                             headers=self._headers, timeout=10)
            models = [m.get("id") for m in (r.json().get("data") or [])] if r.ok else []
            return self._model in models
        except Exception:
            return False