# -*- coding: utf-8 -*-
"""抽取全屏关键帧存档, 供肉眼统计真实语音气泡数(ground truth)。
Usage: python tools/extract_full_frames.py <video> <outdir> <ts_csv>
"""
import sys, os
import cv2, numpy as np

video, outdir, ts_csv = sys.argv[1:4]
os.makedirs(outdir, exist_ok=True)
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
def imwrite(p, img):
    cv2.imencode(".jpg", img)[1].tofile(p)
for ts in [float(x) for x in ts_csv.split(",") if x.strip()]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts*fps))
    ok, fr = cap.read()
    if not ok: continue
    p = os.path.join(outdir, f"full_{ts:04.1f}s.jpg")
    imwrite(p, fr)
    print(p)
cap.release()