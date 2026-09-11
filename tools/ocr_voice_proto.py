# -*- coding: utf-8 -*-
"""原型: 从 OCR 缓存(每帧 line text+bbox) 提取语音气泡事件(时长标签 mm:ss)。
验证召回(可识别出的不同语音消息数)与聚类合理性。
Usage: python tools/ocr_voice_proto.py <ocr_cache.jsonl>
"""
import sys, json, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


DUR = re.compile(r'^(\d{1,2}):([0-5]\d)$')


def is_duration(t: str):
    m = DUR.match(t.strip())
    if not m:
        return False
    mm, ss = int(m.group(1)), int(m.group(2))
    return 0 <= mm <= 15 and not (mm == 0 and ss == 0)


def main():
    cache = sys.argv[1]
    width = 890
    active = []  # (box, side, first_ts, last_ts)
    events = []  # dict
    expire_s = 1.0
    last_ts = [0.0]
    order = []

    def flush_events():
        pass

    frames = 0; labels = 0
    for line in open(cache, encoding='utf-8'):
        d = json.loads(line)
        frames += 1
        ts = d['ts']; last_ts[0] = ts
        cands = []
        for l in d.get('lines', []):
            text = l.get('text', '')
            if not is_duration(text):
                continue
            bx0, by0, bx1, by1 = l['bbox']
            bx, by, bw, bh = bx0, by0, bx1 - bx0, by1 - by0
            side = '收' if (bx0 + bx1) / 2 < width / 2 else '发'
            labels += 1
            cands.append({'box': (bx, by, bw, bh), 'side': side, 'ts': ts, 'text': text})
        # match active
        unmatched = list(cands)
        for st in active:
            best = None; best_iou = 0.0
            for i, c in enumerate(unmatched):
                v = iou(st['box'], c['box'])
                if v > best_iou:
                    best_iou = v; best = i
            if best is not None and best_iou >= 0.35:
                c = unmatched.pop(best)
                st['box'] = c['box']; st['last_ts'] = ts
                st['side'] = c['side'] or st['side']
                st['seen'] += 1
        for c in unmatched:
            active.append({'box': c['box'], 'side': c['side'],
                           'first_ts': ts, 'last_ts': ts, 'seen': 1, 'text': c['text']})
        # expire & emit
        alive = []
        for st in active:
            if ts - st['last_ts'] > expire_s:
                events.append({
                    'timestamp': st['first_ts'],
                    'end_time': st['last_ts'],
                    'side': st['side'],
                    'box': list(st['box']),
                    'frames': st['seen'],
                    'text': st['text'],
                })
            else:
                alive.append(st)
        active = alive
    for st in active:
        events.append({'timestamp': st['first_ts'], 'end_time': st['last_ts'],
                       'side': st['side'], 'box': list(st['box']),
                       'frames': st['seen'], 'text': st['text']})

    events.sort(key=lambda e: e['timestamp'])
    print(f'OCR 帧={frames}, 时长标签共 {labels} 次, 事件数(expire={expire_s}) = {len(events)}')
    from collections import Counter
    sides = Counter(e['side'] for e in events)
    print('  side:', dict(sides))
    long = [e for e in events if (e['end_time'] - e['timestamp']) > 6]
    print(f'  存活>6s 的事件 = {len(long)}')
    uniq_text = Counter(e['text'] for e in events)
    print(f'  唯一时长文本数 = {len(uniq_text)}')
    for e in events[:60]:
        print(f"t={e['timestamp']:6.2f} end={e['end_time']:6.2f} {e['side']} "
              f"box={[round(v) for v in e['box']]} frames={e['frames']} {e['text']}")


if __name__ == '__main__':
    main()