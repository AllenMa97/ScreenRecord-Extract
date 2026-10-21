"""
逐条语音抽取器 (Per-Voice-Message Extractor)

从"拍屏录制的微信长对话视频"中, 按视觉/OCR 检测到的每条语音气泡事件,
从整段视频音轨中切出对应的语音, 逐个保存为独立 WAV 文件, 并可选逐条转写,
为后续业务(逐条转写喂 NLP、逐条声纹建档)提供干净输入。

数据流:
    video -> decode_audio(整轨 PCM, 与视频时间轴对齐)
        -> 对每个 VoiceEvent: extract_segment(ts, dur) -> PCM
        -> write wav (磁盘)        [逐条语音文件]
        -> ASR 转写 (逐条文本)      [逐条转写 -> NLP]
        -> 声纹接口 (预留)          [逐条语音 -> 声纹特征]

说明 / Notes:
    语音的实际发声窗口 = [气泡timestamp, timestamp + 语音时长]。
    语音时长通常来自 OCR 时长标签(audio_duration)。若该事件无 duration,
    则回退用 (end_time - timestamp)(气泡停留时长), 但后者可能含非语音留白。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .asr_client import AsrClient
from .audio_loader import decode_audio, extract_segment, has_audio_track
from .data_types import VoiceEvent


@dataclass
class ExtractedVoice:
    """一条被切出的语音(已落盘), 含转写与事件锚点。"""
    index: int                     # 序号: 1-based, 文件名里可见
    path: str                      # 磁盘 wav 绝对路径
    timestamp: float               # 视频时间轴出现时刻(秒)
    duration: float                # 实际切出的语音时长(秒)
    side: str = ""                 # '发' / '收'
    category: str = ""             # 业务侧可打标(如 victim/suspect)预留
    text: str = ""                 # 逐条转写文本(未转写则为空)
    confidence: float = 0.0        # 转写置信度(服务端暂返回 1.0)
    extra: dict = field(default_factory=dict)


class VoiceExtractor:
    """按语音气泡事件从视频音轨切出逐条语音。

    用法 / Usage::
        ex = VoiceExtractor(model="sensevoice-small")
        out = ex.extract(video_path, events, out_dir="voices", transcribe=True)
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: str = "sensevoice-small", sample_rate: int = 16000):
        self._sample_rate = sample_rate
        self._asr = AsrClient(
            base_url=base_url or os.environ.get("ASR_BASE_URL", ""),
            api_key=api_key or os.environ.get("ASR_API_KEY", ""),
            model=model,
        ) if (base_url or os.environ.get("ASR_BASE_URL")) else None

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def extract(self, video_path: str, events: List[VoiceEvent],
                out_dir: str = "voices", transcribe: bool = True,
                side_meta: Optional[dict] = None) -> List[ExtractedVoice]:
        """按事件切出逐条语音并落盘, 可选逐条转写。

        side_meta: {side: category} 的映射, 例如 {'发': 'victim', '收': 'suspect'},
                   用于给每条语音打业务标签(声纹建档侧会用到)。默认 '发'/'收'。
        """
        os.makedirs(out_dir, exist_ok=True)

        if not has_audio_track(video_path):
            return []

        decoded = decode_audio(video_path, target_rate=self._sample_rate)
        if not decoded.has_audio or decoded.samples.size == 0 or not events:
            return []

        results: List[ExtractedVoice] = []
        for i, ev in enumerate(sorted(events, key=lambda e: e.timestamp), start=1):
            dur = self._resolve_duration(ev)
            seg = extract_segment(decoded.samples, self._sample_rate,
                                  ev.timestamp, dur)
            if seg.size == 0:
                continue

            path = self._write_wav(seg, out_dir, i, ev)
            category = self._category_for(ev.side, side_meta)
            text, conf = "", 0.0
            if transcribe and self._asr:
                utt = self._asr.transcribe_pcm(seg, self._sample_rate,
                                               ev.timestamp, ev.timestamp + dur)
                text, conf = utt.text, utt.confidence

            results.append(ExtractedVoice(
                index=i, path=path,
                timestamp=ev.timestamp, duration=dur,
                side=ev.side, category=category, text=text,
                confidence=conf,
                extra={"source": ev.source, "end_time": ev.end_time}))
        return results

    # ------------------------------------------------------------------
    # 时长解析 / 落盘 / 标签
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_duration(ev: VoiceEvent) -> float:
        """/ 语音时长优先级: OCR 时长标签(audio_duration) > 气泡停留时长(end_time-ts)。"""
        if getattr(ev, "audio_duration", None) and ev.audio_duration > 0:
            # OCR 时长标签: 语音真实播放时长, 最准(不含非语音留白)
            return float(ev.audio_duration)
        if ev.end_time is not None and ev.end_time > ev.timestamp:
            # 气泡停留时长可能含非语音留白, 但无标签时的可用回退
            return max(0.0, ev.end_time - ev.timestamp)
        return 3.0  # 兜底默认时长

    @staticmethod
    def _write_wav(seg: np.ndarray, out_dir: str, index: int,
                   ev: VoiceEvent) -> str:
        """把 float64 PCM 写为 16bit 单声道 wav, 文件名含序号与事件锚点。"""
        from .asr_client import AsrClient  # 复用其 wav 编码, 避免重复实现
        fname = "voice_{:03d}_t{:.1f}s_{}.wav".format(index, ev.timestamp, ev.side or "u")
        path = os.path.join(out_dir, fname)
        with open(path, "wb") as f:
            f.write(AsrClient.pcm_to_wav(seg, 16000))
        return os.path.abspath(path)

    @staticmethod
    def _category_for(side: str, side_meta: Optional[dict] = None) -> str:
        if side_meta and side in side_meta:
            return side_meta[side]
        return side

    # ------------------------------------------------------------------
    # 声纹预留接口
    # ------------------------------------------------------------------
    def extract_embedding(self, seg: np.ndarray, sample_rate: int) -> np.ndarray:
        """逐条语音 -> 声纹特征向量(预留)。

        当前为空实现: 声纹提取需引入专门模型(如 Resemblyzer / ECAPA-TDNN),
        未接入前调用返回空向量。接入后在此处:
            model = load_speaker_model()          # 加载声纹模型
            emb = model.embed(seg, sample_rate)   # 转为定长特征
        """
        return np.zeros(0, dtype=np.float32) if seg.size == 0 else np.zeros(256, dtype=np.float32)


def write_manifest(results: List[ExtractedVoice], out_path: str) -> None:
    """把逐条语音清单写为 JSONL, 供下游(NLP/声纹)直接消费。"""
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "index": r.index, "path": r.path,
                "timestamp": r.timestamp, "duration": r.duration,
                "side": r.side, "category": r.category,
                "text": r.text, "confidence": r.confidence,
                "extra": r.extra,
            }, ensure_ascii=False) + "\n")