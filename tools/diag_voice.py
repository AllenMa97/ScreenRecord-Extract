# -*- coding: utf-8 -*-
"""
语音气泡召回诊断: 复制 VoiceBubbleScanner 的逐帧 IoU 状态机, 在"时间窗合并"之前
输出每个"首次出现事件"的 (time, side, box)。据此区分召回丢失的两类根因:
  A) 误合并: 时间窗合并把"同侧、间隔<1s 的两条不同语音"并成一条(需看原始簇的间距/位置);
  B) 真漏检: 该气泡从未被 detect_voice_bubbles 报出(需与帧内容比对)。
Usage: python tools/diag_voice.py <video_path> <roi 'x,y,w,h'>
"""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visual.voice_bubble_detector import detect_voice_bubbles

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx)); iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix*iy; union = aw*ah + bw*bh - inter
    return inter/union if union else 0.0

def main():
    video = sys.argv[1]; x0,y0,w0,h0 = [int(v) for v in sys.argv[2].split(",")]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = 0.2
    skip = max(1, int(round(interval*fps)))
    active = []                 # [(box, side, conf, vs, first_ts, last_ts)]
    events = []                 # raw first-appearance events with box
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if fid % skip != 0:
            fid += 1; continue
        ts = fid / fps
        roi = frame[y0:y0+h0, x0:x0+w0]
        bubbles = detect_voice_bubbles(roi)
        cands = []
        for b in bubbles:
            if b.voice_score < 0.55: continue
            cands.append(((x0+b.x, y0+b.y, b.w, b.h), b.side, b.confidence, b.voice_score))
        # IoU 匹配 (same as _update_active)
        unmatched = list(cands)
        for st in active:
            best=None; bst=0.0
            for i,c in enumerate(unmatched):
                io = iou(st[0], c[0])
                if io>bst: bst=io; best=i
            if best is not None and bst>=0.35:
                c = unmatched.pop(best)
                active = [ (c[0], c[1] or st[1], c[2], c[3], st[4], ts) if s is st else s
                           for s in active ]
                unmatched[best] if False else None
        # 新出现 -> raw event
        for c in unmatched:
            active.append((c[0], c[1], c[2], c[3], ts, ts))
            events.append((ts, c[1], c[0], c[2], c[3]))
        # expire 2s
        active = [s for s in active if ts - s[5] <= 2.0]
        fid += 1
    cap.release()
    events.sort(key=lambda e: e[0])
    print(f"=== 原始首次出现事件(raw) = {len(events)} ===")
    for e in events:
        ts, side, box, cf, vs = e
        print(f"  t={ts:6.2f}s side={side or '?':<2} score={vs:.2f} conf={cf:.2f} box=({box[0]},{box[1]},{box[2]}x{box[3]})")
    # 按位置强聚类: 只看"同位置(+IoU)"归并, 不分时间窗
    grouped = []
    for e in events:
        box = e[2]
        found = None
        for g in grouped:
            if iou(box, g[-1][2]) > 0.5:
                found = g; break
        if found is None:
            grouped.append([e])
        else:
            found.append(e)
    print(f"\n=== 按位置(IoU>0.5)聚类后的不同气泡数 = {len(grouped)} ===")
    for g in grouped:
        ts0 = min(e[0] for e in g); ts1 = max(e[0] for e in g)
        side = g[0][1]; box = g[0][2]
        print(f"  t={ts0:6.2f}~{ts1:6.2f}s side={side or '?':<2} 出现{len(g)}次 box=({box[0]},{box[1]},{box[2]}x{box[3]})")

if __name__ == "__main__":
    main()