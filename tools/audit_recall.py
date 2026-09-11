# -*- coding: utf-8 -*-
"""召回审计: 渲染 scanner 当前输出的每个事件气泡裁剪 + 额外的低置信度候选,
供人眼/视觉模型核验召回与精确率。
Usage: python tools/audit_recall.py <video> <roi> <outdir>
"""
import sys, os
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.voice_bubble_scanner import VoiceBubbleScanner
from src.visual.voice_bubble_detector import detect_voice_bubbles


def imwrite(p, img):
    cv2.imencode(".jpg", img)[1].tofile(p)


def crop_at(cap, ts, box, roi_xy):
    x0, y0 = roi_xy
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
    ok, fr = cap.read()
    if not ok:
        return None
    bx, by, bw, bh = [int(v) for v in box]
    pad = 6
    H, W = fr.shape[:2]
    x1 = max(x0 + bx - pad, 0); y1 = max(y0 + by - pad, 0)
    x2 = min(x0 + bx + bw + pad, W); y2 = min(y0 + by + bh + pad, H)
    return fr[y1:y2, x1:x2]


def main():
    video, roi, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    roi = tuple(int(v) for v in roi.split(","))
    os.makedirs(outdir, exist_ok=True)
    x0, y0 = roi[0], roi[1]

    scanner = VoiceBubbleScanner(roi_box=roi, voice_threshold=0.55,
                                 sample_interval_s=0.2, expire_s=2.0)
    events = scanner.scan(video)
    cap = cv2.VideoCapture(video)

    for i, e in enumerate(events, 1):
        img = crop_at(cap, e.timestamp, e.box, (x0, y0))
        if img is None:
            continue
        sc = {"发": "send", "收": "recv", "": "unk"}.get(e.side, "unk")
        p = os.path.join(outdir, f"ev{i:02d}_t{e.timestamp:05.1f}_{sc}.jpg")
        imwrite(p, img)
        print(f"EV {i:02d} t={e.timestamp:6.2f} end={e.end_time or e.timestamp:6.2f} "
              f"side={e.side or '?'} box={[round(v) for v in e.box]} -> {os.path.basename(p)}")

    # 额外候选: 原始扫描(不做 _dedup 合并)的低置信度事件, 找 detector 可能漏掉的
    raw = VoiceBubbleScanner(roi_box=roi, voice_threshold=0.30,
                             sample_interval_s=0.2, expire_s=2.0,
                             merge_window_s=0.0).scan(video)
    print(f"\n低阈值(0.30, 不合并)事件数 = {len(raw)}")
    t_events = {tuple(round(v, 2) for v in e.box) for e in events}
    extra = 0
    for j, e in enumerate(raw, 1):
        if tuple(round(v, 2) for v in e.box) in t_events:
            continue
        extra += 1
        img = crop_at(cap, e.timestamp, e.box, (x0, y0))
        if img is None:
            continue
        sc = {"发": "send", "收": "recv", "": "unk"}.get(e.side, "unk")
        p = os.path.join(outdir, f"ex{j:02d}_t{e.timestamp:05.1f}_{sc}.jpg")
        imwrite(p, img)
        print(f"EX {j:02d} t={e.timestamp:6.2f} side={e.side or '?'} "
              f"box={[round(v) for v in e.box]} score={e.confidence:.2f} -> {os.path.basename(p)}")
    print(f"\n额外候选(可能漏检) = {len(raw)} 中 {extra} 个不在主事件框集中")
    cap.release()


if __name__ == "__main__":
    main()