# -*- coding: utf-8 -*-
"""扫描 merge_window_s 对最终事件数的影响 (threshold=0.55)。
关键认知: 检测层已足够(85 raw / 54 位置簇), 召回损失全部在 _dedup_events
的 merge_window_s(默认2.0)过度合并。找贴近用户肉眼计数(~25)的窗口设定。
Usage: python tools/sweep_window.py <video> <roi>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.voice_bubble_scanner import VoiceBubbleScanner

def main():
    video = sys.argv[1]; roi = tuple(int(v) for v in sys.argv[2].split(","))
    for wnd in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0]:
        s = VoiceBubbleScanner(roi_box=roi, voice_threshold=0.55,
                               sample_interval_s=0.2, expire_s=2.0,
                               merge_window_s=wnd)
        evs = s.scan(video)
        print(f"merge_window_s={wnd}: {len(evs)} 事件")

if __name__ == "__main__":
    main()