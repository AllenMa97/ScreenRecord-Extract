# -*- coding: utf-8 -*-
"""扫描在不同 voice_threshold 下的最终事件数, 找贴近用户计数(~25)的设定。
Usage: python tools/sweep_voice_threshold.py <video> <roi>
"""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.voice_bubble_scanner import VoiceBubbleScanner

def main():
    video = sys.argv[1]; roi = tuple(int(v) for v in sys.argv[2].split(","))
    for thr in [0.50, 0.52, 0.55, 0.58]:
        s = VoiceBubbleScanner(roi_box=roi, voice_threshold=thr,
                               sample_interval_s=0.2, expire_s=2.0)
        evs = s.scan(video)
        print(f"threshold={thr}: {len(evs)} 事件")

if __name__ == "__main__":
    main()