# -*- coding: utf-8 -*-
"""在当前检测器上打印各气泡的 voice_score 与关键特征, 核对 reweight 效果。"""
import sys, collections
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visual.voice_bubble_detector import (detect_voice_bubbles, _ink_mask,
                                              _downsample, _voice_score,
                                              _green_mask, _white_mask)

def main():
    video = sys.argv[1]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, int(round(0.2 * fps)))
    scores = collections.Counter()
    shown = 0
    fid = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fid % skip != 0:
            fid += 1
            continue
        ts = fid / fps
        for b in detect_voice_bubbles(fr):
            key = f"vs={b.voice_score:.2f} side={b.side or '?'} {b.w}x{b.h}"
            scores[key] += 1
            if shown < 48:
                print(f"t={ts:5.2f} side={b.side or '?'} vs={b.voice_score:.2f} "
                      f"cf={b.confidence:.2f} box=({b.x},{b.y},{b.w}x{b.h})")
                shown += 1
        fid += 1
    cap.release()
    print("\n=== score 分布(去重后去放大) ===")
    for k, c in sorted(scores.items()):
        print(f"  {c:4d}x  {k}")

if __name__ == "__main__":
    main()