# -*- coding: utf-8 -*-
"""分析语音事件: 同一物理气泡在滚动时 y 位移, 但尺寸(w,h)不变。
按 (side, h, 附近w) 做尺寸签名聚类, 观察不同物理气泡数与尺寸分布。
Usage: python tools/analyze_voice_sizes.py <video> <roi>
"""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.voice_bubble_scanner import VoiceBubbleScanner

def main():
    video = sys.argv[1]
    roi = tuple(int(v) for v in sys.argv[2].split(","))
    scanner = VoiceBubbleScanner(roi_box=roi, voice_threshold=0.55,
                                 sample_interval_s=0.2, expire_s=2.0)
    events = scanner.scan(video)
    # 过滤: 全宽条(疑似输入栏/顶栏)并非普通聊天气泡
    h, w = 1920, 890
    real = [e for e in events if e.box and e.box[2] < w*0.75 and e.box[3] < h*0.25]
    print(f"扫描事件(全)= {len(events)}, 常规尺寸条= {len(real)}")
    # 尺寸签名: (side, h, w_rounded)
    from collections import defaultdict
    sig = defaultdict(list)
    for e in real:
        bx, by, bw, bh = e.box
        key = (e.side, int(round(bw/20)*20), int(round(bh/10)*10))
        sig[key].append((e.timestamp, e.box))
    print(f"\n尺寸签名簇数 = {len(sig)}")
    for (side, wb, hb), lst in sorted(sig.items(), key=lambda kv: kv[0][2]):
        ts = [x[0] for x in lst]
        print(f"  side={side or '?':<2} w~{wb} h~{hb}: {len(lst):2d}次  t={min(ts):6.2f}~{max(ts):6.2f} (首例 box={tuple(round(v) for v in lst[0][1])})")

if __name__ == "__main__":
    main()