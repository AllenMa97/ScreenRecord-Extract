# -*- coding: utf-8 -*-
"""统计 ground truth 中相邻采样帧的文本重叠率, 判断各视频的时间冗余结构。
高重叠 = 滚动拷贝/静止重复(应被去重); 低重叠 = 内容真正快速更新。
Usage: python tools/gt_overlap.py"""
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

def norm(t):
    return " ".join(t.split())

def main():
    for gt in sorted((ROOT / "ground_truth").glob("*.jsonl")):
        frames = {}
        with open(gt, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                frames[int(r["frame_id"])] = {
                    norm(li.get("text","")) for li in r.get("lines",[]) if li.get("text")}
        ids = sorted(frames)
        if len(ids) < 2:
            print(f"{gt.stem:<34} 帧数太少 {len(ids)}")
            continue
        overlaps = []
        for a, b in zip(ids, ids[1:]):
            A, B = frames[a], frames[b]
            if not B:
                continue
            inter = len(A & B)
            ov = inter / max(1, len(B))
            overlaps.append(ov)
        n = len(overlaps)
        avg = sum(overlaps) / n
        high = sum(1 for o in overlaps if o >= 0.5) / n
        # 平均每帧行数与变化
        nlines = [len(frames[i]) for i in ids]
        print(f"{gt.stem:<34} 帧{len(ids):<4} 重叠率avg={avg:.2f} "
              f"high(>=0.5)占比={high:.0%} 平均行数={sum(nlines)/len(nlines):.1f}")

if __name__ == "__main__":
    main()