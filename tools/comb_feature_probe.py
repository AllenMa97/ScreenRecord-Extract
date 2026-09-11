# -*- coding: utf-8 -*-
"""对"已知真语音"与"已知文本/墙纸"的 crop 计算一组判别特征, 寻找能分开两者的信号。
真语音(e01/e05/e06): 一排等高的细竖条(声波条), 底部对齐、条间有空隙(梳状)。
文本/墙纸: 连续笔画/装饰图案, 无"等高空隙梳状"结构。
Usage: python tools/comb_feature_probe.py <video>
"""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visual.voice_bubble_detector import _ink_mask, _downsample

# (标签, t秒, box) —— 由 render_final_events 实际输出, 已人工确认真/假语音
KNOWN = [
    # (label, t, box, is_voice)
    ("voice_e01",  2.47, (14, 1566, 517, 164), True),
    ("voice_e05", 11.21, (28, 222, 503, 128), True),
    ("voice_e06", 16.14, (28, 1004, 317, 70), True),
    ("text_e07", 17.85, (17, 323, 551, 72), False),     # 过来吗? 文本
    ("text_e08", 18.04, (578, 1343, 178, 200), False),  # 文本气泡
    ("text_e13", 28.30, (17, 1215, 476, 72), False),    # 墙纸/背景
    ("text_e09", 19.18, (17, 1546, 439, 72), False),    # 我看一下今晚方不方便
]

def comb_features(ink_binary):
    """梳状声波条特征: 统计列, 找"等高的细竖条"占比。"""
    h, w = ink_binary.shape
    col = np.sum(ink_binary, axis=0)          # 每列墨水高度
    bar = (col > 0).astype(np.uint8)          # 有墨水列
    # 列间空隙 → 条形结构数量
    # 找连续"条"(run of columns with ink), 计算条宽
    runs = []
    start = None
    for c in range(w):
        if bar[c] and start is None:
            start = c
        elif not bar[c] and start is not None:
            runs.append((start, c - 1))
            start = None
    if start is not None:
        runs.append((start, w - 1))
    widths = [e - s + 1 for s, e in runs]
    # 竖条高度统计(有墨列的高度)
    heights = col[col > 0]
    # 底部对齐: 每列最低有墨水行的 y, 方差小 → 波形底沿齐
    lows = []
    for c in range(w):
        yy = np.where(ink_binary[:, c] > 0)[0]
        if yy.size:
            lows.append(yy.max())
    # 顶部对齐: 波形条顶部常齐平(微起伏)
    pads = [np.where(ink_binary[:, 0] > 0)[0] for _ in range(1)]
    # 归一化高度差: 各条高度相对集中(条高近似相等)
    inh = np.array(heights, float) if len(heights) else np.array([0.0])
    h_mean = inh.mean() + 1e-6
    h_rel = inh / h_mean
    height_uniform = np.exp(-float(np.std(h_rel)))
    # 托底对齐: 底沿散布小
    low_align = 1.0 - min(float(np.std(lows)) / max(h, 1) * 6.0, 1.0) if lows else 0.0
    # 条宽集中(声波条为细窄条, 条宽小且一致)
    wr = np.array(widths, float) if widths else np.array([0.0])
    narrow_run = float(np.mean(wr <= 3)) if len(wr) else 0.0
    # 梳状占比: 细窄条内墨水占比
    comb_ratio = float(np.sum(bar)) / float(w)
    return dict(runs=len(widths), narrow_run=narrow_run, comb_ratio=comb_ratio,
                height_uniform=height_uniform, low_align=low_align,
                col_mean=float(col.mean()), nib=int(ink_binary.sum()))

def main():
    video = sys.argv[1]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = (320.0 / W) if W > 320 else 1.0
    print("label          runs nrun comb hUni lowAlign nib")
    for label, t, box, is_voice in KNOWN:
        x, y, w, h = box
        fid = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, fr = cap.read()
        if not ok:
            print(label, "readfail"); continue
        bgr = fr[y:y + h, x:x + w]
        small = cv2.resize(bgr, (max(1, int(w*scale)), max(1, int(h*scale))),
                           interpolation=cv2.INTER_AREA)
        ink = (_ink_mask(small) > 0).astype(np.uint8)
        if ink.sum() < 16:
            print(f"{label}  (nib<16)"); continue
        f = comb_features(ink)
        tag = " VOICE" if is_voice else " text "
        print(f"{label} [{tag}] runs={f['runs']:3d} nrun={f['narrow_run']:.2f} "
              f"comb={f['comb_ratio']:.2f} hUni={f['height_uniform']:.2f} "
              f"lowAlign={f['low_align']:.2f} nib={f['nib']}")
    cap.release()

if __name__ == "__main__":
    main()