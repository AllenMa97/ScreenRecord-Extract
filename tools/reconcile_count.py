# -*- coding: utf-8 -*-
"""核查: OCR 时长标签按"物理位置连续跟随"聚类, 数真实出现的语音气泡数。
对比 (side,duration) 文本去重 vs 物理跟随 两种口径, 定位 88 与~25 的差在哪。
"""
import json, re
from pathlib import Path

_DUR = re.compile(r'^(\d{1,2}):(\d{1,2})$')
def is_dur(t):
    m = _DUR.match((t or '').strip())
    if not m: return None
    if int(m.group(1)) > 2 or int(m.group(2)) < 1: return None
    return m

def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ix=max(0,min(ax+aw,bx+bw)-max(ax,bx)); iy=max(0,min(ay+ah,by+bh)-max(ay,by))
    inter=ix*iy; union=aw*ah+bw*bh-inter
    return inter/union if union>0 else 0.0

cache=r"D:\手机录屏内容提取\web\videos\20260831_203323_fc42\output\cache\source_494b917fb9e807ba.jsonl"
W=890
# 按 (side,dur) 归并向量的出现段(连续可见 streak)
streaks={}   # (side,dur) -> list of (t0,t1, box)
active={}    # (side,dur) -> current streak
for line in open(cache,encoding='utf-8'):
    d=json.loads(line); ts=float(d.get('ts',0))
    frame_keys=set()
    for l in d.get('lines',[]):
        m=is_dur(l.get('text',''))
        if not m: continue
        cb=l.get('bbox'); 
        if not cb: continue
        bx0,by0,bx1,by1=cb
        side='收' if (bx0+bx1)/2<W/2 else '发'
        key=(side,m.group(0))
        frame_keys.add((key,(bx0,by0,bx1-bx0,by1-by0)))
    # 结束未在本帧出现的 streaks
    for k in list(active.keys()):
        if k not in frame_keys:
            s=active.pop(k); s['t1']=ts; streaks.setdefault(k,[]).append(s)
    for k,box in frame_keys:
        if k in active:
            active[k]['box']=box
        else:
            active[k]={'t0':ts,'t1':ts,'box':box}
for k,s in active.items():
    s['t1']=float('inf'); streaks.setdefault(k,[]).append(s)

# 口径A: 文本去重 (side,dur)
print('口径A: 唯一 (side,时长) 键 =', len(streaks))
# 口径B: 物理跟随——每键内, 若出现段间在时间与位置上"断开"则算作新的物理气泡
def physical_count(k, segs):
    cnt=0; last=None
    for s in sorted(segs,key=lambda x:x['t0']):
        if last is None:
            cnt+=1; last=s; continue
        gap=s['t0']-last['t1']
        boxgap = iou(last['box'], s['box']) if (last['box'] and s['box']) else 1
        if gap>1.5 and boxgap<0.2:   # 长时间消失 + 位置大跳 -> 新气泡(或重新回看)
            cnt+=1
        last=s
    return cnt
phys=0
for k,segs in streaks.items():
    c=physical_count(k,segs)
    phys+=c
    if c>1:
        print(f'  {k}: {len(segs)}段 -> {c}个物理气泡 (duration={k[1]})')
print('口径B: 物理跟随总数 =', phys)