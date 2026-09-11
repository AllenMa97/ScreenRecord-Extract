"""分析行为日志: 去重决策质量诊断 / Diagnose dedup decisions in a behavior log.
用法 / Usage:  python tools/diag_log.py <behavior_log.jsonl>
"""
import json
import sys
from collections import Counter, defaultdict


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows


def main(path):
    rows = load(path)
    state = {}                          # frame_id -> frame_sampler state
    for r in rows:
        if r.get("module") == "frame_sampler" and r.get("event") == "sample":
            state[r.get("frame_id")] = (r.get("extra") or {}).get("state")

    # ---- 1. OCR 触发帧的采样状态分布 ----
    ocr_by_state = Counter()
    ocr_frames = []
    for r in rows:
        ev = r.get("event") or ""
        if ev.endswith("_recognize") or ev == "high_conf_new":
            st = state.get(r.get("frame_id"), "?")
            ocr_by_state[st] += 1
            ocr_frames.append((r.get("frame_id"), st, r.get("elapsed_sec"),
                               r.get("module")))
    print("== OCR(识别/high_conf_new) 触发帧的采样状态分布 ==")
    for k, v in ocr_by_state.most_common():
        print(f"  {k:<12} {v}")
    print(f"  OCR 总数 = {sum(ocr_by_state.values())}")

    # ---- 2. rf 判定分布(去重器自身) ----
    rf = Counter()
    for r in rows:
        if r.get("module") == "row_fingerprint" and r.get("event") == "compare":
            rf[r.get("extra", {}).get("verdict")] += 1
    print("\n== row_fingerprint 判定分布 ==")
    for k, v in rf.most_common():
        print(f"  {k:<8} {v}")

    # ---- 3. 滚动状态下的 OCR 连续触发(相邻 OCR 是否都在滚动中、间隔多久) ----
    print("\n== 滚动态触发的 OCR 帧序列(检查是否在滚动中反复送OCR) ==")
    scrolling_ocr = [f for f in ocr_frames if f[1] == "scrolling"]
    print(f"  滚动态 OCR 共 {len(scrolling_ocr)} 次")
    if len(scrolling_ocr) >= 2:
        for i in range(1, len(scrolling_ocr)):
            prev, cur = scrolling_ocr[i - 1], scrolling_ocr[i]
            gap = cur[2] - prev[2] if isinstance(cur[2], (int, float)) and isinstance(prev[2], (int, float)) else 0
            print(f"  frame={prev[0]} -> frame={cur[0]}  时间间隔={gap:.2f}s ({prev[3]})")

    # ---- 4. 重复剔除(滚动记忆库)效果 ----
    dropped = sum((r.get("extra", {}) or {}).get("dropped_revisit", 0)
                  for r in rows if r.get("event") == "compare")
    lib_sizes = [r.get("extra", {}).get("lib_size", 0)
                 for r in rows if r.get("event") == "compare"]
    print(f"\n== 滚动记忆库 ==")
    print(f"  复现剔除累计 = {dropped}")
    print(f"  库大小范围   = min={min(lib_sizes) if lib_sizes else 0}, "
          f"max={max(lib_sizes) if lib_sizes else 0}")

    # ---- 5. rf_new 判定但最终被下游判定为复现/dup 的情况 ----
    # 对比 row_fingerprint 判定 与 最终 visual_dedup 决策
    final = {}
    for r in rows:
        m, ev = r.get("module"), r.get("event")
        if m == "visual_dedup" and ev in ("rf_dup", "rf_new", "high_conf_new", "high_conf_dup"):
            final[(r.get("elapsed_sec"))] = ev
    print(f"\n== visual_dedup 最终决策计数 ==")
    vd = Counter()
    for r in rows:
        if r.get("module") == "visual_dedup" and r.get("event") in (
                "rf_dup", "rf_new", "high_conf_new", "high_conf_dup"):
            vd[r.get("event")] += 1
    for k, v in vd.most_common():
        print(f"  {k:<16} {v}")


if __name__ == "__main__":
    main(sys.argv[1])