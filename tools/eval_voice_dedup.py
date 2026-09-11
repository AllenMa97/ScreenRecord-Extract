# -*- coding: utf-8 -*-
"""对比 旧(时间窗合并) vs 新(位置感知合并) 的语音事件数, 用于召回核查。
Usage: python tools/eval_voice_dedup.py <video> <roi x,y,w,h>
"""
import sys, time
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.voice_bubble_scanner import VoiceBubbleScanner

def main():
    video = sys.argv[1]
    roi = tuple(int(v) for v in sys.argv[2].split(","))
    scanner = VoiceBubbleScanner(roi_box=roi, voice_threshold=0.55,
                                 sample_interval_s=0.2, expire_s=2.0)
    t0 = time.time()
    events = scanner.scan(video)
    print(f"新逻辑(尺寸签名+去整宽条): {len(events)} 个事件  ({time.time()-t0:.1f}s)")
    # 同数据上复算旧逻辑(纯时间窗)作对照
    old = scanner._dedup_events(list(events), merge_window_s=2.0,
                                size_tol=1e9, cx_tol=1e9)
    print(f"对照 纯时间窗(huge容差≈旧): {len(old)} 个事件")
    for e in events:
        b = tuple(round(v) for v in e.box) if e.box else ()
        print(f"  t={e.timestamp:6.2f}s {e.side or '?':<2} box={b}")

if __name__ == "__main__":
    main()