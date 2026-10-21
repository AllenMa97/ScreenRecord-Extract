# -*- coding: utf-8 -*-
"""冒烟测试: 合并逻辑(视觉为准 + OCR 框内佐证)在 171653 上应输出 25 条,
不被界面时钟污染成 88 条, 且时长标签召回 >= 15/25(修复坐标系错位前仅 4/25)。
Usage: python tools/smoke_merge.py
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.voice_bubble_scanner import (VoiceBubbleScanner,
                                            merge_ocr_and_vision,
                                            probe_media_duration)
from src.audio.ocr_voice_scanner import OcrVoiceScanner

VIDEO = r"D:\手机录屏内容提取\web\videos\20260831_203323_fc42\source.mp4"
ROI = (0, 60, 890, 1740)
CACHE = r"D:\手机录屏内容提取\web\videos\20260831_203323_fc42\output\cache\source_494b917fb9e807ba.jsonl"

scanner = VoiceBubbleScanner(roi_box=ROI, voice_threshold=0.55,
                             sample_interval_s=0.2, expire_s=2.0)
vision = scanner.scan(VIDEO)

ocr = OcrVoiceScanner(frame_width=ROI[2])
# 关键: OCR 缓存的 bbox 是 ROI 局部坐标, 必须传 roi_offset 换算到原始帧坐标
hits = ocr.scan_duration_hits(CACHE, roi_offset=(ROI[0], ROI[1]))

merged = merge_ocr_and_vision(
    vision, hits, media_duration_s=probe_media_duration(VIDEO))

print(f"视觉事件数          = {len(vision)}")
print(f"OCR 时长标签观测    = {len(hits)} (已剔除 >60s 的时钟/进度等非语音标签)")
print(f"合并后事件数        = {len(merged)}  (期望 >=25 且远小于 88)")
n_hy = sum(1 for e in merged if e.source == "hybrid")
n_vi = sum(1 for e in merged if e.source == "vision")
n_dur = sum(1 for e in merged if e.audio_duration)
print(f"  其中 hybrid(OCR佐证)={n_hy}, 纯 vision={n_vi}")
print(f"  拿到时长标签        = {n_dur}/{len(merged)}  (修复前 4/25)")

assert len(merged) == len(vision), "必须以视觉清单为准, 不得新增事件"
assert len(merged) >= 25, "不应低于肉眼核对的 25 条"
assert len(merged) < 60, "不应被时钟污染膨胀"
assert n_dur >= 15, f"时长标签召回不足: {n_dur}/{len(merged)}"
for i, e in enumerate(merged, 1):
    d = f"{e.audio_duration:5.1f}s" if e.audio_duration else "   -- "
    print(f"  EV{i:02d} t={e.timestamp:6.2f}s {e.side} [{e.source}] 时长={d} "
          f"{e.matched_text or ''}")
print("\nSMOKE OK")