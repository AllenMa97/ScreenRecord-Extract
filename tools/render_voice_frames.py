# -*- coding: utf-8 -*-
"""渲染视觉语音气泡检测的整帧标注图, 供人工核对 25 条。
Usage: python tools/render_voice_frames.py
输出: outdir/voice_evNN_tXXXX.jpg  (整帧缩略 + 气泡框)
"""
import sys, os
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.voice_bubble_scanner import VoiceBubbleScanner

VIDEO = r"D:\手机录屏内容提取\web\videos\20260831_203323_fc42\source.mp4"
ROI = (0, 60, 890, 1740)     # (x,y,w,h): 去掉顶部状态栏60 + 底部输入框120
OUT = r"D:\手机录屏内容提取\web\videos\20260831_203323_fc42\voice_check"
os.makedirs(OUT, exist_ok=True)

def imwrite(p, img):
    cv2.imencode(".jpg", img)[1].tofile(p)

scanner = VoiceBubbleScanner(roi_box=ROI, voice_threshold=0.55,
                             sample_interval_s=0.2, expire_s=2.0)
events = scanner.scan(VIDEO)
print(f"视觉检出事件数 = {len(events)}")

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
rw, rh = ROI[2], ROI[3]
SIDE = {"发": "send", "收": "recv", "": "unk"}

for i, e in enumerate(events, 1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(e.timestamp * fps))
    ok, fr = cap.read()
    if not ok:
        continue
    # 整帧缩小便于拼图查看
    h, w = fr.shape[:2]
    scale = 360 / rh
    nw, nh = int(w * scale), int(h * scale)
    disp = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)
    # 气泡框(ROI 相对坐标 -> 全帧坐标 -> 缩略坐标)
    bx, by, bw, bh = [int(v) for v in e.box]
    gx = (ROI[0] + bx) * scale
    gy = (ROI[1] + by) * scale
    gx2 = (ROI[0] + bx + bw) * scale
    gy2 = (ROI[1] + by + bh) * scale
    col = (0, 200, 0) if e.side == "发" else (0, 120, 255)
    cv2.rectangle(disp, (int(gx), int(gy)), (int(gx2), int(gy2)), col, 2)
    cv2.putText(disp, f"#{i} t{e.timestamp:.1f} {e.side}", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    p = os.path.join(OUT, f"ev{i:02d}_t{e.timestamp:06.1f}_{SIDE.get(e.side,'unk')}.jpg")
    imwrite(p, disp)
    print(f"  EV{i:02d} t={e.timestamp:6.2f} side={e.side or '?'} box={[round(v) for v in e.box]}")

cap.release()
print(f"\n完成, 标注图目录: {OUT}")