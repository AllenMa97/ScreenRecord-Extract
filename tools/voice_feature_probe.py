# -*- coding: utf-8 -*-
"""对比 语音气泡 与 文字气泡 的形态特征(孤立条列/高度方差/间隔), 找稳定判别量。
需要人工指定: <video> <roi> <ts,side,w,h>... 每次读该框的 ink mask 输出特征。
"""
import sys
import cv2, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visual.voice_bubble_detector import _ink_mask, _downsample

def feat(ink):
    ink = (ink > 0).astype(np.uint8)
    h, w = ink.shape
    total = int(ink.sum())
    # 每列高度
    colh = ink.sum(axis=0)
    # 孤立条列: 该列有黑, 且左右相邻列都基本空
    cols_on = colh > 0
    isolated = 0
    for c in range(1, w-1):
        if cols_on[c] and not cols_on[c-1] and not cols_on[c+1]:
            isolated += 1
    # 有内容列
    n_cols = int((cols_on).sum())
    # 列高方差(规整的波形条高度变化小?)
    hstd = float(colh[cols_on].std()) if n_cols else 0.0
    hmean = float(colh[cols_on].mean()) if n_cols else 0.0
    # 填充率
    density = total / (h*w)
    return dict(total=total, w=w, h=h, n_cols=n_cols, isolated=isolated,
                iso_ratio=isolated/max(n_cols,1), hstd=hstd, hmean=hmean,
                density=density)

def main():
    video = sys.argv[1]; x0,y0,w0,h0 = [int(v) for v in sys.argv[2].split(",")]
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for arg in sys.argv[3:]:
        ts, side, w, h = arg.split(":")
        ts, bw, bh = float(ts), int(w), int(h)
        # 从该时刻找 ROI 内尺度最接近 (bw,bh) 的气泡
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts*fps))
        ok, frame = cap.read()
        if not ok: continue
        roi = frame[y0:y0+h0, x0:x0+w0]
        small, scale = _downsample(roi)
        best=None; bestd=1e9
        import numpy as _np
        from src.visual.voice_bubble_detector import _green_mask, _white_mask, _collect_from_mask
        for b in _collect_from_mask(small, _green_mask(small), "g", scale) + _collect_from_mask(small, _white_mask(small), "w", scale):
            d = abs(b.w-bw)+abs(b.h-bh)
            if d<bestd: bestd=d; best=b
        if best is None:
            print(f"{ts}s: no candidate"); continue
        box = small[int(best.y*scale):int((best.y+best.h)*scale),
                    int(best.x*scale):int((best.x+best.w)*scale)]
        ink = _ink_mask(box)
        print(f"{ts}s side={best.side} box={best.w}x{best.h} score={best.voice_score:.2f} R={best.voice_score>0.5}")
        print(f"    {feat(ink)}")
    cap.release()

if __name__ == "__main__":
    main()