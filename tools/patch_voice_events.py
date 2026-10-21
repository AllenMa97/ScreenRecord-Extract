# -*- coding: utf-8 -*-
"""用**修复后的**时长标签召回重算已运行会话的 voice_events, 并重切逐条语音 wav。

为什么需要它: 本次修复(坐标系对齐 + ≤60s 领域约束 + 生命周期窗口匹配)只影响
"语音气泡扫描"这一段。对已跑过 ASR 的会话无需整片重跑(重跑 ASR 很贵), 只要:
  1. 重扫视觉气泡 + 重读 OCR 时长标签(传 roi_offset) -> 新 voice_events;
  2. 用新清单整体替换落盘 extracted_content_*.json 里的 voice_events
     (旧条目带的是错位匹配出的假时长, 如 154s/172s, 必须整段替换而非合并);
  3. 清理并重建 output/voices/*.wav, 回填 audio_url。

Usage:
  python tools/patch_voice_events.py <session_id> [x,y,w,h]
ROI 缺省时按标准边距推算: (0, 60, W, H-180)。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.voice_bubble_scanner import (VoiceBubbleScanner,
                                            merge_ocr_and_vision,
                                            probe_media_duration)
from src.audio.ocr_voice_scanner import OcrVoiceScanner
from web.backend.services import session_store as store
from web.backend.services import voice_audio

# 与 src/roi_manager 的默认边距一致: 顶部状态栏 60, 底部输入栏 120
MARGIN_TOP, MARGIN_BOTTOM = 60, 120


def latest_result(out_dir: Path) -> Path:
    files = sorted(out_dir.glob("extracted_content_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise SystemExit(f"未找到 extracted_content_*.json: {out_dir}")
    return files[0]


def find_ocr_cache(out_dir: Path) -> Path:
    files = sorted((out_dir / "cache").glob("*.jsonl"),
                   key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise SystemExit(f"未找到 OCR 缓存: {out_dir / 'cache'}")
    return files[0]


def default_roi(sid: str):
    """按标准边距从视频实际尺寸推算 ROI。"""
    import cv2
    cap = cv2.VideoCapture(str(store.video_path(sid)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    return (0, MARGIN_TOP, int(w), int(h) - MARGIN_TOP - MARGIN_BOTTOM)


def main():
    sid = sys.argv[1]
    video = store.video_path(sid)
    if not video:
        raise SystemExit(f"会话缺少视频文件: {sid}")
    out_dir = store.output_dir(sid)
    path = latest_result(out_dir)

    roi = (tuple(int(v) for v in sys.argv[2].split(","))
           if len(sys.argv) > 2 else default_roi(sid))
    if not roi or roi[2] <= 0 or roi[3] <= 0:
        raise SystemExit("无法确定 ROI, 请以 <x,y,w,h> 形式传入")

    cache = find_ocr_cache(out_dir)
    result = json.loads(path.read_text(encoding="utf-8"))
    print(f"会话    : {sid}")
    print(f"视频    : {video.name}")
    print(f"结果    : {path.name}")
    print(f"ROI     : {roi}")
    print(f"OCR缓存 : {cache.name}")

    vision = VoiceBubbleScanner(roi_box=tuple(int(v) for v in roi),
                                voice_threshold=0.55,
                                sample_interval_s=0.2).scan(str(video))
    hits = OcrVoiceScanner(frame_width=int(roi[2])).scan_duration_hits(
        str(cache), roi_offset=(int(roi[0]), int(roi[1])))
    merged = merge_ocr_and_vision(
        vision, hits, media_duration_s=probe_media_duration(str(video)))

    old = result.get("voice_events") or []
    n_old_dur = sum(1 for d in old if d.get("audio_duration"))
    print(f"\n视觉气泡 {len(vision)} -> 合并后 {len(merged)}; "
          f"原存储 {len(old)} 条(其中带时长 {n_old_dur})")

    # 整体替换: 旧条目的 source/时长来自坐标错位的旧匹配(曾出现 154s/172s), 无保留价值
    events = []
    for ev in merged:
        events.append({
            "timestamp": ev.timestamp,
            "end_time": ev.end_time,
            "side": ev.side,
            "confidence": ev.confidence,
            "source": ev.source,
            "duration": "",
            "audio_duration": ev.audio_duration,
            "text": ev.matched_text or "",
            "matched": False,
            "consumed_ids": [],
        })
    result["voice_events"] = events
    n_dur = sum(1 for d in events if d.get("audio_duration"))
    print(f"新 voice_events = {len(events)} 条, 带时长 {n_dur} 条 (修复前 4)")
    for i, d in enumerate(events, 1):
        if d["audio_duration"]:
            print(f"  #{i:02d} t={d['timestamp']:7.2f}s {d['side']} "
                  f"时长={d['audio_duration']:5.1f}s")

    # 同步统计, 避免前端仍显示旧口径
    st = result.get("statistics")
    if isinstance(st, dict):
        st["voice_events"] = len(events)

    # 重切逐条语音(旧 wav 一律清掉, 避免时长修正后残留错长度的文件)
    voice_audio.clean_voices(out_dir)
    voice_audio.build_voice_audio(sid, str(video), result)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    n_url = sum(1 for d in result["voice_events"] if d.get("audio_url"))
    print(f"\n已写回 {path.name}; audio_url 回填 = {n_url}/{len(result['voice_events'])}")


if __name__ == "__main__":
    main()
