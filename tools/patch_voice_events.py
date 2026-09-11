# -*- coding: utf-8 -*-
"""把已运行会话的 stored result 语音气泡更新为"OCR+视觉混合"结果,
与新版 pipeline._scan_voice_bubbles + merge_into_result 完全等价,
以立即在 UI 呈现修复后的召回(无需整片重跑)。仅用于演示/回填。
Usage: python tools/patch_voice_events.py <session_dir> <video_path> <roi>
"""
import sys, json, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.voice_bubble_scanner import VoiceBubbleScanner, merge_ocr_and_vision
from src.audio.ocr_voice_scanner import OcrVoiceScanner
from src.audio.report_merge import merge_into_result
from src.audio.transcriptor import TranscriptResult


def main():
    sdir, video, roi = sys.argv[1], sys.argv[2], tuple(int(v) for v in sys.argv[3].split(","))
    outdir = Path(sdir) / "output"
    files = sorted(outdir.glob("extracted_content_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("未找到 extracted_content_*.json"); return
    path = files[0]
    result = json.load(open(path, encoding="utf-8"))
    cache = None
    for c in outdir.joinpath("cache").glob("*.jsonl"):
        cache = str(c); break
    if cache is None:
        print("未找到 OCR 缓存"); return

    ocr = OcrVoiceScanner(
        frame_width=result.get("video", {}).get("width", 890)).scan_from_cache(cache)
    vision = VoiceBubbleScanner(roi_box=roi, voice_threshold=0.55,
                                sample_interval_s=0.2).scan(video)
    merged = merge_ocr_and_vision(ocr, vision)

    a = result.get("audio") or {}
    tr = TranscriptResult(has_audio=a.get("has_audio", False),
                          source=str(a.get("source", "")))
    tr.segments = []

    r = merge_into_result(result, tr, merged, max_match_gap_s=1.5, inject_chat=False)
    # 保留原始 segments, 仅替换 voice_events
    r["audio"] = result.get("audio") or {}
    json.dump(r, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    from collections import Counter
    src_counter = Counter(v2.get("source") for v2 in r.get("voice_events", []))
    print(f"已更新 {path.name}: voice_events={len(r.get('voice_events', []))} "
          f"source={dict(src_counter)} 未采集={sum(1 for v in r['voice_events'] if not v['matched'])}")


if __name__ == "__main__":
    main()