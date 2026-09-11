# -*- coding: utf-8 -*-
"""物理气泡计数: 完全忽略 OCR 时长文本, 仅按时间轴连续可见将时长标签聚成真实物理气泡。
判据: 时长标签按其首次可见时间排序, 相邻两次观察若时间间隔超过 gap_s 视为不同物理气泡。
原因: 同一物理气泡滚动重看时可见是连续的; 不同气泡(或跨过滚动消失再出现)才有时间断层。
输出: 物理气泡总数 + 每个气泡的时间窗与采样到的时长文本(用于人工核对)。
"""
import json, re
from pathlib import Path

_DUR = re.compile(r'^(\d{1,2}):(\d{1,2})$')

def is_dur(t):
    m = _DUR.match((t or '').strip())
    if not m: return None
    if int(m.group(1)) > 2 or int(m.group(2)) < 1: return None
    return m.group(0)

cache = r"D:\手机录屏内容提取\web\videos\20260831_203323_fc42\output\cache\source_494b917fb9e807ba.jsonl"

# 收集所有被确认为时长标签的出现 (ts, dur_text)
rows = []
for line in open(cache, encoding='utf-8'):
    d = json.loads(line)
    ts = float(d.get('ts', 0))
    if ts > 70:          # 视频只有65.7s, 过滤越界帧
        continue
    for l in d.get('lines', []):
        dur = is_dur(l.get('text', ''))
        if not dur:
            continue
        rows.append((ts, dur))
rows.sort()

if not rows:
    print('无时长标签')
    raise SystemExit

# 时间主导的物理聚类
GAP_S = 0.6             # 两次观察间隔超过0.6s -> 新物理气泡
bubbles = []
cur_ts0, cur_ts1, cur_durs, cur_gap = rows[0][0], rows[0][0], [rows[0][1]], 0.0
for ts, dur in rows[1:]:
    if ts - cur_ts1 > GAP_S:
        bubbles.append((cur_ts0, cur_ts1, cur_durs))
        cur_ts0, cur_ts1, cur_durs = ts, ts, [dur]
    else:
        cur_ts1 = ts
        if dur != cur_durs[-1]:
            cur_durs.append(dur)
bubbles.append((cur_ts0, cur_ts1, cur_durs))

total_dur_labels = len(rows)
print(f'总时长标签观察数 = {total_dur_labels}')
print(f'物理气泡数(时间连续跟随) = {len(bubbles)}\n')
for i, (t0, t1, durs) in enumerate(bubbles, 1):
    print(f'  #{i:2d}  视频 {t0:6.1f}s ~ {t1:6.1f}s  文本序列={durs}')