"""
逐条语音落盘 + 审听 URL 服务 (Per-voice audio service)

在提取完成后, 用 VoiceExtractor 按 voice_events(audio_duration) 从视频音轨切出
逐条语音 wav, 存到 `output/voices/`, 并把可访问的相对 URL 回填进结果 json 的
voice_events.audio_url, 供前端"找到并审听"每条语音。

向后端暴露:
    GET /api/sessions/{sid}/voices/{name}   -> 返回单个 wav(见 sessions.py)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from src.audio.voice_extractor import VoiceExtractor, ExtractedVoice
from . import session_store as store


VOICES_DIR = "voices"   # 相对 output_dir 的目录名


def clean_voices(out_dir: Path) -> None:
    """清理上次提取产生的逐条语音 / Clear previous per-voice wavs."""
    d = out_dir / VOICES_DIR
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)


def build_voice_audio(sid: str, video_path: str,
                      result: Dict) -> Dict:
    """按 result.voice_events 切出逐条语音并回填 audio_url(原地修改并返回 result)。

    幂等: 无视频/无事件/无音轨均安全返回原 result。切段落盘到 output/voices/,
    随后把每个事件补上 audio_url("/api/sessions/{sid}/voices/{file}")。
    """
    events = result.get("voice_events") or []
    if not events or not video_path:
        return result

    # 转回 VoiceEvent 对象供 VoiceExtractor 使用(带 OCR 时长/坐标/收发)
    from src.audio.data_types import VoiceEvent
    ve_events = []
    for d in events:
        ve_events.append(VoiceEvent(
            timestamp=float(d.get("timestamp", 0.0)),
            end_time=d.get("end_time"),
            side=d.get("side") or "",
            confidence=float(d.get("confidence", 0.0)),
            matched_text=d.get("text") or "",
            box=d.get("box") or None,
            source=d.get("source") or "vision",
            audio_duration=d.get("audio_duration"),
        ))

    out = store.output_dir(sid) / VOICES_DIR
    ex = VoiceExtractor()  # 不转写(转写已在 report_merge 完成), 只做切片落盘
    extracted: List[ExtractedVoice] = ex.extract(
        video_path, ve_events, out_dir=str(out), transcribe=False)

    # 按 index(1-based) 映射回 result 的 voice_events(事件已按时间排序)
    by_index = {r.index: r for r in extracted}
    for i, d in enumerate(events, start=1):
        r = by_index.get(i)
        if r is None:
            d["audio_url"] = None
            continue
        d["audio_url"] = f"/api/sessions/{sid}/voices/{Path(r.path).name}"

    # 把回填后的 voice_events 同步进落盘的 extracted_content_*.json,
    # 使 /result(json) 与 memory 内 result 一致。
    _rewrite_result_json(sid, result)
    return result


def _rewrite_result_json(sid: str, result: Dict) -> None:
    """把带 audio_url 的最新结果覆盖到 extracted_content_*.json(取最新一份)。"""
    try:
        out_dir = store.output_dir(sid)
        files = sorted(
            out_dir.glob("extracted_content_*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return
        with open(files[0], "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except OSError:
        pass