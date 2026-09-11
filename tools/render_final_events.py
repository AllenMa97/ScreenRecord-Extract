# -*- coding: utf-8 -*-
"""渲染 scanner 最终输出事件的裁剪图, 逐个人眼核验。"""
import sys, os
import cv2, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.voice_bubble_scanner import VoiceBubbleScanner

def imwrite(p, img):
    cv2.imencode(".jpg", img)[1].tofile(p)

def main():
    video, roi, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    roi = tuple(int(v) for v in roi.split(","))
    x0, y0 = roi[0], roi[1]
    os.makedirs(outdir, exist_ok=True)
    scanner = VoiceBubbleScanner(roi_box=roi, voice_threshold=0.55,
                                 sample_interval_s=0.2, expire_s=2.0)
    events = scanner.scan(video)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for i, e in enumerate(events, 1):
        if not e.box: continue
        bx, by, bw, bh = [int(v) for v in e.box]
        pad = 6
        x1 = max(x0+bx-pad, 0); y1 = max(y0+by-pad, 0)
        x2 = min(x0+bx+bw+pad, 890); y2 = min(y0+by+bh+pad, 1920)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(e.timestamp*fps))
        ok, fr = cap.read()
        if not ok: continue
        crop = fr[y1:y2, x1:x2]
        sc = {"发":"send","收":"recv","":"unk"}.get(e.side,"unk")
        p = os.path.join(outdir, f"e{i:02d}_t{e.timestamp:05.1f}_{sc}_w{bw}_h{bh}.jpg")
        imwrite(p, crop)
        print(f"{i:02d} t={e.timestamp:6.2f} {e.side or '?':<2} w={bw} h={bh} -> {os.path.basename(p)}")
    cap.release()

if __name__ == "__main__":
    main()