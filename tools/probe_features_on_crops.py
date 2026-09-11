# -*- coding: utf-8 -*-
"""在 scanner 实际检测到的 box 上测量 _voice_score 特征, 输出到表格供人工核对
"哪些是真语音(波形)、哪些是文本"。并用不同阈值统计事件数, 供设判别阈值。
Usage: python tools/probe_features_on_crops.py <video> <roi>
"""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.voice_bubble_scanner import VoiceBubbleScanner
from src.visual.voice_bubble_detector import _ink_mask, _voice_score

def feats(bgr, scale):
    small = cv2.resize(bgr, (max(1,int(bgr.shape[1]*scale)), max(1,int(bgr.shape[0]*scale))),
                       interpolation=cv2.INTER_AREA)
    ink = _ink_mask(small)
    h, w = ink.shape
    inkb = (ink > 0).astype(np.uint8)
    nib = int(inkb.sum())
    if nib < 16:
        return dict(nib=0, density=0.0, span=0.0, bars=0.0, align=0.0,
                    narrow=0.0, sparse=0.0, score=0.0)
    total = float(nib)
    density = total / (h * w)
    v_open = cv2.morphologyEx(inkb, cv2.MORPH_OPEN, np.ones((9, 1), np.uint8))
    h_open = cv2.morphologyEx(inkb, cv2.MORPH_OPEN, np.ones((1, 9), np.uint8))
    vert_ret = v_open.sum() / total
    bars = min(vert_ret / 0.55, 1.0)
    rows = np.where(inkb.sum(axis=1) > 0)[0]
    span = (rows.max()-rows.min()+1)/max(h,1) if rows.size else 0.0
    narrow = max(0.0, 1.0 - span*1.6)
    sparse = 1.0 - min(density/0.42, 1.0)
    colbottoms = [np.where(inkb[:,c]>0)[0].max() for c in range(w) if np.where(inkb[:,c]>0)[0].size]
    align = max(0.0, 1.0 - (float(np.std(colbottoms))/max(h,1)*4.0)) if colbottoms else 0.0
    vs = _voice_score(ink)
    return dict(nib=nib, density=density, span=span, bars=bars, align=align,
                narrow=narrow, sparse=sparse, score=vs)

def main():
    video = sys.argv[1]; roi = tuple(int(v) for v in sys.argv[2].split(","))
    x0, y0 = roi[0], roi[1]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = (320.0/W) if W > 320 else 1.0
    scanner = VoiceBubbleScanner(roi_box=roi, voice_threshold=0.55,
                                 sample_interval_s=0.2, expire_s=2.0,
                                 merge_window_s=1.2)
    events = scanner.scan(video)
    rows = []
    for i, e in enumerate(events, 1):
        if not e.box: continue
        bx, by, bw, bh = [int(v) for v in e.box]
        fid = int(e.timestamp*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, fr = cap.read()
        if not ok: continue
        bgr = fr[y0+by:y0+by+bh, x0+bx:x0+bx+bw]
        f = feats(bgr, scale)
        rows.append((i, e.timestamp, e.side, bw, bh, f))
    cap.release()
    # 打印按 score 排序, 并把阈值切点标出
    print("idx  t      side  w x h    nib  dens  span  bars align narr sparse score")
    seq = sorted(rows, key=lambda r: r[5]["score"])
    for i, t, side, bw, bh, f in seq:
        print(f"{i:3d} {t:6.2f} {side or '?':<3} {bw:3d}x{bh:<3} "
              f"{f['nib']:5d} {f['density']:5.3f} {f['span']:4.2f} "
              f"{f['bars']:4.2f} {f['align']:4.2f} {f['narrow']:4.2f} "
              f"{f['sparse']:4.2f} {f['score']:5.2f}")
    print("\n--- 按不同阈值保留事件数(仅依据 score>=thr 的粗筛) ---")
    for thr in [0.55, 0.58, 0.60, 0.62, 0.65]:
        n = sum(1 for _,_,_,_,_,f in rows if f["score"] >= thr)
        print(f"  threshold>={thr}: {n} 事件")

if __name__ == "__main__":
    main()