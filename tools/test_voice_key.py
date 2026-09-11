# -*- coding: utf-8 -*-
"""验证"尺寸+中心x 判定同气泡"的假设: 按 (side, w容差, h容差, cx容差, 时间窗)
合并事件, 观察气泡数与配对数, 与用户肉眼计数(~25)对照。
Usage: python tools/test_voice_key.py <video> <roi>
"""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.voice_bubble_scanner import VoiceBubbleScanner
from src.audio.data_types import VoiceEvent

def same_bubble(a: VoiceEvent, b: VoiceEvent, tw=4.0, w_tol=0.18, h_tol=0.15, cx_tol=0.12):
    """判定两个事件是否为同一物理气泡(滚动只动 y, w/h/x 稳定)。"""
    if a.side != b.side:
        return False
    if abs(a.timestamp - b.timestamp) > tw:
        return False
    if not (a.box and b.box):
        return True
    ax, ay, aw, ah = a.box; bx, by, bw, bh = b.box
    if abs(aw - bw) / max(aw, bw) > w_tol:
        return False
    if abs(ah - bh) / max(ah, bh) > h_tol:
        return False
    acx, bcx = ax + aw / 2, bx + bw / 2
    if abs(acx - bcx) / max(aw, bw) > cx_tol:
        return False
    return True

def dedup_key(events, sweeps):
    n = len(events)
    for tw, wt, ht, ct in sweeps:
        merged = [False]*n
        kept = []
        for i, ev in enumerate(events):
            if merged[i]:
                continue
            kept.append(ev)
            for j in range(i+1, n):
                if merged[j]: continue
                if any(same_bubble(ev, events[j], tw, wt, ht, ct) for ev in kept[-1:]):
                    merged[j] = True
        yield (tw, wt, ht, ct), len(kept)

def main():
    video = sys.argv[1]; roi = tuple(int(v) for v in sys.argv[2].split(","))
    scanner = VoiceBubbleScanner(roi_box=roi, voice_threshold=0.55,
                                 sample_interval_s=0.2, expire_s=2.0)
    events = scanner.scan(video)
    h, w = 1920, 890
    real = [e for e in events if e.box and e.box[2] < w*0.75 and e.box[3] < h*0.25]
    print(f"位置感知扫描事件={len(events)}, 常规条={len(real)}  (用户肉眼约25)")
    sweeps = [
        (4.0, .18, .15, .12), (4.0, .25, .20, .15), (2.0, .25, .20, .15),
        (6.0, .25, .20, .15), (4.0, .30, .25, .20),
    ]
    for (tw, wt, ht, ct), c in dedup_key(real, sweeps):
        print(f"  时间窗{tw}s w容差{wt} h容差{ht} cx容差{ct} → {c} 个气泡")

if __name__ == "__main__":
    main()