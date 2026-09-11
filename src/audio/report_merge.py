"""
报告合并 (Report Merge)

把音频侧(带时间锚点的逐段转写)与视觉侧(语音气泡事件)合并进最终报告 dict:

  1. result["audio"]       : 完整逐段音频转写(视频时间轴锚点)。
  2. result["voice_events"]: 视觉语音气泡 与 音频段 按时间就近配对 → 还原"这条
                             语音说的内容"; 未配对的(静音/被提前滚走)保留空文本。
  3. 注入聊天段落          : 把配好对的语音消息以 "- +Ns [语音][报/收] ：text"
                             形式按时间插入对应视觉段落的 content, 使语音内容真正
                             进聊天记录流水。
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .data_types import VoiceEvent, VoiceUtterance
from .transcriptor import TranscriptResult

logger = logging.getLogger(__name__)


def _fmt_video(sec: float) -> str:
    """视频秒数 → 报告里可读的 +Ns 形式(与视觉时间戳兜底约定一致)。"""
    s = max(0, int(round(sec)))
    return f"{s}"


# 相邻语音段间隔小于该秒数时视为同一条语音(气泡可能被 VAD 按长时间停顿切开)
SEG_JOIN_GAP_S = 0.9


def _collect_cluster(segs, start_idx, t, max_match_gap_s, consumed):
    """从事件时刻 t 出发, 收集一条"语音簇": 与 t 时间重叠的段及其相邻连续段。

    VAD 会按 >250ms 停顿把一条带停顿的语音切成多段, 故把与 t 时间上有交集、
    且彼此间隔 < SEG_JOIN_GAP_S 的段合并为一条, 避免后半句丢失; 已被其它
    事件消费的段跳过(M1: 一条语音只注入一次)。返回 (段下标列表, 拼接文本)。
    """
    n = len(segs)
    # 第一阶段: 找与事件时间窗重叠的种子段
    seeds = [i for i in range(n)
             if not consumed[i] and segs[i].start - max_match_gap_s <= t <= segs[i].end + max_match_gap_s]
    if not seeds:
        return None, None
    # 第二阶段: 从种子向两端扩展连续段(同一条语音被停顿切开的部分)
    chosen = set(seeds)
    changed = True
    while changed:
        changed = False
        lo = min(chosen)
        hi = max(chosen)
        # 向左扩展
        if lo > 0 and not consumed[lo - 1] and (lo - 1) not in chosen:
            if segs[lo].start - segs[lo - 1].end <= SEG_JOIN_GAP_S:
                chosen.add(lo - 1)
                changed = True
        # 向右扩展
        if hi < n - 1 and not consumed[hi + 1] and (hi + 1) not in chosen:
            if segs[hi + 1].start - segs[hi].end <= SEG_JOIN_GAP_S:
                chosen.add(hi + 1)
                changed = True
    idx = sorted(chosen)
    text = "".join(segs[i].text for i in idx if segs[i].text)
    return idx, text


def pair_voice_events(events: List[VoiceEvent],
                      segments: List[VoiceUtterance],
                      max_match_gap_s: float = 1.5) -> List[dict]:
    """把每个语音气泡事件匹配到音频转写段(一条语音簇)。

    匹配: 收集与事件时刻时间上重叠的段及其相邻连续段, 合并为一条语音;
    一条语音段最多被一个事件消费(避免重复注入); 未命中任何段 → 文本为空。

    空文本段不参与匹配: 整段静音/无有效人声时(VAD返回空段, 转写为空串),
    该段不能虚拟命中气泡(消除了气泡"假配对+空文本"的伪命中, 见静音视频场景)。
    返回可按时间排序的配对 dict 列表。
    """
    segs = [s for s in sorted(segments, key=lambda s: s.start)
            if (s.text or "").strip()]
    consumed = [False] * len(segs)
    out = []
    for ev in sorted(events, key=lambda e: e.timestamp):
        t = ev.timestamp
        idx, text = _collect_cluster(segs, None, t, max_match_gap_s, consumed)
        matched = idx is not None
        if matched:
            for i in idx:
                consumed[i] = True
        out.append({
            "timestamp": ev.timestamp,
            "end_time": ev.end_time,
            "side": ev.side,
            "confidence": ev.confidence,
            "source": getattr(ev, "source", "vision"),
            "duration": (ev.matched_text or ""),
            "text": text or "",
            "matched": matched,
            "consumed_ids": idx or [],
        })
    return out


def _section_window(sections: List[dict], t: float) -> Optional[int]:
    """返回应承载时刻 t 对应语音消息的段落下标(时间窗最近的), 无则 None。"""
    if not sections:
        return None
    stamps = [float(s.get("timestamp", 0.0)) for s in sections]
    if t <= stamps[0]:
        return 0
    for i in range(len(stamps) - 1):
        if stamps[i] <= t < stamps[i + 1]:
            return i
    return len(sections) - 1


def merge_into_result(result: dict, transcript: TranscriptResult,
                      events: List[VoiceEvent],
                      max_match_gap_s: float = 1.5,
                      inject_chat: bool = True) -> dict:
    """把音频转写 + 语音气泡配对写入 result dict 并注入聊天段落。

    参数:
        result: 视觉侧组装出的结果 dict(含 sections)。
        transcript: 音频转写结果。
        events: 视觉语音气泡出现事件。
        max_match_gap_s: 配对最大时间容差(秒)。
        inject_chat: 是否把语音消息注入视觉段落 content。
    """
    result["audio"] = transcript.to_dict()
    if not transcript.has_audio:
        result["voice_events"] = []
        return result

    paired = pair_voice_events(events, transcript.segments, max_match_gap_s)
    result["voice_events"] = paired

    matched = [p for p in paired if p["matched"]]
    unmatched = [p for p in paired if not p["matched"]]
    stats = result.setdefault("statistics", {})
    stats["voice_events"] = len(paired)
    stats["voice_matched"] = len(matched)
    stats["voice_unmatched"] = len(unmatched)
    stats["audio_segments"] = len(transcript.segments)

    # 注入聊天段落: 语音消息以独立行插入对应时间窗口的段落末尾
    if inject_chat and result.get("sections"):
        sections = result["sections"]
        by_time = {}
        for p in paired:
            if not p["text"]:
                continue
            idx = _section_window(sections, p["timestamp"])
            if idx is None:
                continue
            side = f"[{p['side']}] " if p["side"] else ""
            line = f"- +{_fmt_video(p['timestamp'])}s [语音] {side}：{p['text']}"
            by_time.setdefault(idx, []).append((p["timestamp"], line))
        if by_time:
            offset = -1
            for idx, arr in sorted(by_time.items()):
                arr.sort()
                lines = [ln for _, ln in arr]
                cur = sections[idx]["content"].rstrip()
                for ln in lines:
                    cur = cur + ("\n" if cur else "") + ln
                sections[idx]["content"] = cur

    # 更新摘要
    total_segs = len(transcript.segments)
    text_segs = sum(
        1 for s in transcript.segments if (s.text or "").strip()
    )
    summary_text = (
        f", 音频{total_segs}段{text_segs}段可读"
        if total_segs and text_segs
        else (", 有音轨但未采集到人声" if total_segs else "")
    )
    if summary_text:
        base = str(result.get("summary", ""))
        base = base.rstrip("。")
        result["summary"] = (f"{base}{summary_text}").strip()
    return result