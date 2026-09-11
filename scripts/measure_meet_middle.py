"""
head+tail策略潜力测量 / Measure head+tail strategy potential.

用内容缓存(唯一内容指纹 + 首次/末次出现)分析htkf视频:
  1. 内容时间分布: 首次出现帧号分布 → 单调滚动 vs 静止/回访结构
  2. 回访内容: 同一内容再次出现(间隔>1s)的比例 → head+tail/分片去重可获益部分
  3. 头尾覆盖曲线: 头扫描与尾扫描各自覆盖全部唯一内容所需的帧数
Using the content cache (unique fingerprint + first/last appearance), measure:
  1. temporal distribution of first appearances → monotonic vs static/revisit
  2. revisit ratio (same content reappears after >1s) → portion head+tail dedup wins
  3. head-only and tail-only coverage curves → meet-in-the-middle savings bound
"""
import os
import sys
import json
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CACHE = os.path.join(PROJECT_ROOT, "output", "cache", "htkf_c91485fc9ec63d1b.jsonl")

# hash16 -> (first_fid, last_fid, first_ts, last_ts, obs)
first = {}
last = {}
obs = defaultdict(int)
with open(CACHE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("kind") != "content":
            continue
        h = int(rec["hash"])
        fid, ts = int(rec["fid"]), rec.get("ts", 0.0)
        if h not in first:
            first[h] = (fid, ts)
        last[h] = (fid, ts)
        obs[h] += 1

n = len(first)
print(f"唯一内容: {n}")
fids = sorted(fid for fid, _ in first.values())
print(f"首次出现帧号: min={fids[0]} max={fids[-1]}")

# 回访判定: 末次-首次时间 > 1s 且 观测>=2 → 内容在时间线上重复出现
revisit = []
for h, (fid_f, ts_f) in first.items():
    fid_l, ts_l = last[h]
    if ts_l - ts_f > 1.0 and obs[h] >= 2:
        revisit.append((ts_f, ts_l, ts_l - ts_f, obs[h], h))
print(f"回访内容(重复出现): {len(revisit)}/{n} = {len(revisit)/n*100:.1f}%")
for ts_f, ts_l, span, o, h in sorted(revisit)[:15]:
    print(f"  首现{ts_f:.2f}s 末现{ts_l:.2f}s 跨度{span:.2f}s 观测{o}次 hash={h}")

# 观测次数分布
multi = sum(1 for h in obs if obs[h] >= 2)
print(f"观测>=2次的内容: {multi}/{n} = {multi/n*100:.1f}%")

# 头覆盖曲线: 按首次出现帧号累积唯一内容
sorted_first = sorted((fid, h) for h, (fid, _) in first.items())
total_frames = sorted_first[-1][0]
covered = 0
pcts = []
for pct in (25, 50, 75, 90, 95, 100):
    target = n * pct / 100
    for fid, h in sorted_first:
        covered = sum(1 for _, hh in sorted_first if _ <= fid)
        if covered >= target:
            break
    pcts.append((pct, fid, covered))
print("头向覆盖(需扫到帧号才能看到X%唯一内容):")
for pct, fid, c in pcts:
    print(f"  {pct}%: 帧{fid} ({fid/total_frames*100:.0f}%视频)")
