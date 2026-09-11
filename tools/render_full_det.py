# -*- coding: utf-8 -*-
"""全屏+检测框标注帧, 用于直接肉眼对比 真实/检测 语音气泡。
Usage: python tools/render_full_det.py <video> <roi x,y,w,h> <outdir> <ts_csv>
"""
import sys, os
import cv2, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visual.voice_bubble_detector import detect_voice_bubbles

def imwrite(p, img):
    cv2.imencode(".jpg", img)[1].tofile(p)

def main():
    video, roi, outdir, ts_csv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    x0, y0, w0, h0 = [int(v) for v in roi.split(",")]
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for ts in [float(x) for x in ts_csv.split(",") if x.strip()]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts*fps))
        ok, fr = cap.read()
        if not ok: continue
        robj = fr[y0:y0+h0, x0:x0+w0]
        bubbles = detect_voice_bubbles(robj)
        vo = [b for b in bubbles if b.voice_score >= 0.55]
        base = fr.copy()
        for b in vo:
            px0, py0 = x0+b.x, y0+b.y
            col = (0, 220, 0) if b.side == "发" else (0, 220, 220)
            cv2.rectangle(base, (px0, py0), (px0+b.w, py0+b.h), col, 3)
            cv2.putText(base, f"{b.side} {b.voice_score:.2f}", (px0, max(0,py0-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        p = os.path.join(outdir, f"dt_{ts:04.1f}s_{len(vo)}n.jpg")
        imwrite(p, base)
        print(f"{p} -> {len(vo)} det")
    cap.release()

if __name__ == "__main__":
    main()