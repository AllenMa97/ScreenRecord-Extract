# -*- coding: utf-8 -*-
"""诊断 behavior_log.jsonl: 归纳视觉去重/滚动去重的行为结构
Diagnose behavior_log.jsonl: summarize visual-dedup / scroll behavior."""

import json
import sys
from collections import Counter, defaultdict

LOG = r"D:\手机录屏内容提取\web\videos\20260826_134450_355d\output\behavior_log.jsonl"


def main():
    rows = []
    with open(LOG, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))

    print(f"总事件数 / total events: {len(rows)}")

    # 模块事件分布
    mod_ev = Counter((r.get("module"), r.get("event")) for r in rows)
    print("\n== 模块/事件统计 ==")
    for (m, e), c in sorted(mod_ev.items(), key=lambda x: -x[1]):
        print(f"  {m:>16}.{e:<22} {c}")

    # row_fingerprint compare 判定分布
    compares = [r for r in rows if r.get("module") == "row_fingerprint" and r.get("event") == "compare"]
    verdicts = Counter(c.get("extra", {}).get("verdict") for c in compares)
    print("\n== row_fingerprint 判定分布 ==")
    for v, c in verdicts.most_common():
        print(f"  {v:<10} {c}")

    # visual_dedup 决策分布
    vd = [r for r in rows if r.get("module") == "visual_dedup"]
    vd_ev = Counter(r.get("event") for r in vd)
    print("\n== visual_dedup 决策分布 ==")
    for e, c in vd_ev.most_common():
        print(f"  {e:<22} {c}")

    # 按帧追踪 compare 的 new_rows / verdict 序列
    print("\n== 每帧 compare 序列 (帧id→verdict, 新行数) ==")
    seq = []
    for c in compares:
        fid = c.get("frame_id")
        ex = c.get("extra", {})
        nr = ex.get("new_rows", [])
        seq.append((fid, ex.get("verdict"), len(nr), ex.get("aligned_ratio"), ex.get("displacement")))
    # 只打印 verdict 变化的位置(压缩静止重复)
    prev_verdict = None
    run = 0
    for fid, v, n, ratio, disp in seq:
        if v != prev_verdict:
            if prev_verdict is not None:
                print(f"    ... (x{run} 帧 {prev_verdict})")
            print(f"  帧{fid}: {v} 新行={n} 对齐率={ratio:.2f} 位移={disp}")
            prev_verdict = v
            run = 1
        else:
            run += 1
    if run:
        print(f"    ... (x{run} 帧 {prev_verdict})")

    # sample 的滚动/静止分布
    samples = [r for r in rows if r.get("module") == "frame_sampler" and r.get("event") == "sample"]
    sample_state = Counter(s.get("extra", {}).get("state") for s in samples)
    print("\n== 采样帧 state 分布 ==")
    for s, c in sample_state.most_common():
        print(f"  {s:<10} {c}")


if __name__ == "__main__":
    main()