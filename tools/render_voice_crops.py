# -*- coding: utf-8 -*-
"""把每个"尺寸签名聚类"的首个检测框裁剪出来存图, 供肉眼核查真伪。
Usage: python tools/render_voice_crops.py <video> <roi> <outdir>
"""
import sys, os
from pathlib import Path
import cv2
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.voice_bubble_scanner import VoiceBubbleScanner

def main():
    video, roi, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    roi = tuple(int(v) for v in roi.split(","))
    os.makedirs(outdir, exist_ok=True)
    scanner = VoiceBubbleScanner(roi_box=roi, voice_threshold=0.55,
                                 sample_interval_s=0.2, expire_s=2.0)
    events = scanner.scan(video)
    h, w = 1920, 890
    real = [e for e in events if e.box and e.box[2] < w*0.75 and e.box[3] < h*0.25]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 按 (side, w桶, h桶) 分组, 每组取一个代表性事件
    sig = defaultdict(list)
    for e in real:
        bx, by, bw, bh = e.box
        sig[(e.side, int(round(bw/20)*20), int(round(bh/10)*10))].append(e)
    n = 0
    for key, lst in sorted(sig.items(), key=lambda kv: kv[0][2]):
        e = lst[0]
        bx, by, bw, bh = e.box
        # 把裁剪框稍微外扩, 含完整气泡
        pad = 8
        x0 = max(roi[0]+bx-pad, 0); y0 = max(roi[1]+by-pad, 0)
        x1 = min(roi[0]+bx+bw+pad, w); y1 = min(roi[1]+by+bh+pad, h)
        fid = int(e.timestamp*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok: continue
        crop = frame[y0:y1, x0:x1]
        n += 1
        scode = {"发": "send", "收": "recv", "": "unk"}.get(key[0], "unk")
        fname = os.path.join(outdir, f"b{n:02d}_t{e.timestamp:04.1f}_{scode}_w{int(bw)}_h{int(bh)}.jpg")
        cv2.imwrite(fname, crop)
        print(f"{fname}")
    cap.release()

if __name__ == "__main__":
    main()