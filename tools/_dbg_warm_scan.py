# -*- coding: utf-8 -*-
"""临时诊断: 复现 v3_warm 首个语音气泡事件缺失的根因。用完即删。"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visual.voice_bubble_detector import (detect_voice_bubbles, _ink_mask,
                                              _voice_score, _downsample,
                                              _green_mask, _white_mask,
                                              _collect_from_mask)
from src.audio.voice_bubble_scanner import VoiceBubbleScanner

VID = "synthetic/voice_chat/videos/v3_warm.mp4"

cap = cv2.VideoCapture(VID)
fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
ok, f0 = cap.read()
cap.release()
print("fps", fps, "frames", n, "size_hw", f0.shape[:2], "dur", round(n / fps, 2))

H, W = f0.shape[:2]
MT, MB = 60, 120
roi = (0, MT, W, H - MT - MB)
print("margin_roi", roi)

scan = VoiceBubbleScanner(roi_box=roi, voice_threshold=0.0,
                          sample_interval_s=0.2).scan(VID)
print("\n=== ALL DETECTED EVENTS (threshold=0) n=%d ===" % len(scan))
for ev in scan[:60]:
    print(f"  t={ev.timestamp:5.1f} side={ev.side} conf={ev.confidence:.2f} box={ev.box} src={ev.source}")

print("\n=== FIRST 12s: raw detect_voice_bubbles on ROI ===")
cap = cv2.VideoCapture(VID)
fid = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    ts = fid / fps
    if ts > 12.0:
        break
    if fid % 3 == 0:
        roi_img = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        bs = detect_voice_bubbles(roi_img)
        keep = [b for b in bs if b.voice_score >= 0.0]
        if keep:
            print(f"t={ts:4.1f} " + "; ".join(
                f"[{b.side} vs={b.voice_score:.2f} cf={b.confidence:.2f} {b.x},{b.y},{b.w},{b.h}]"
                for b in keep))
    fid += 1
cap.release()
