"""
截图质量量化 / Quantify screenshot quality.

对送OCR的帧计算拉普拉斯方差(清晰度)分布, 并关联帧MAE/采样状态,
判断视觉侧是否把"滚动中/模糊"的帧送给了OCR。
Compute Laplacian variance (sharpness) distribution for OCR-submitted frames
and correlate with frame MAE/sampling state to judge whether the visual side
submits mid-scroll/blurry frames to OCR.
"""
import os
import sys
import json

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FRAMES_DIR = os.path.join(PROJECT_ROOT, "output", "frames", "htkf")

def lap_score(path: str) -> float:
    data = open(path, "rb").read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

# 读取帧索引 / load frame index
idx = {}
idx_path = os.path.join(FRAMES_DIR, "frames_index.jsonl")
if os.path.exists(idx_path):
    for line in open(idx_path, encoding="utf-8"):
        r = json.loads(line)
        idx[int(r.get("frame_id", -1))] = r

scores = []
for fn in os.listdir(FRAMES_DIR):
    if not fn.endswith("_ocr.jpg"):
        continue
    path = os.path.join(FRAMES_DIR, fn)
    s = lap_score(path)
    if s < 0:
        continue
    fid = int(fn.split("_")[0][1:])
    rec = idx.get(fid, {})
    scores.append({
        "fid": fid, "ts": rec.get("timestamp", 0),
        "lap": s, "mae": rec.get("mae", -1),
        "state": rec.get("state", "?"),
    })

scores.sort(key=lambda x: x["fid"])
laps = [s["lap"] for s in scores]
print(f"送OCR帧数: {len(scores)}")
print(f"拉普拉斯方差: min={min(laps):.0f} p25={np.percentile(laps,25):.0f} "
      f"中位={np.median(laps):.0f} p75={np.percentile(laps,75):.0f} max={max(laps):.0f}")
for thr in (100, 200, 300, 500):
    n = sum(1 for s in scores if s["lap"] < thr)
    print(f"  清晰度<{thr}: {n} ({n/len(scores)*100:.0f}%)")
from collections import Counter
states = Counter(s["state"] for s in scores)
print("\n送OCR帧的采样状态分布(static=静止/确认帧, rapid+scrolling=滚动动画帧):")
for k, v in states.most_common():
    print(f"  {k}: {v} ({v/len(scores)*100:.0f}%)")
# 滚动动画帧(内容半截/未settle) / mid-scroll frames
mid = sum(1 for s in scores if s["state"] in ("rapid", "scrolling"))
print(f"\n滚动动画中送OCR: {mid}/{len(scores)} = {mid/len(scores)*100:.0f}% "
      f"(这些帧文本半截, OCR结果质量差, 下游去重要额外兜底)")
print("\n最模糊的10帧(可能是滚动中截图):")
for s in sorted(scores, key=lambda x: x["lap"])[:10]:
    print(f"  帧{s['fid']} ts={s['ts']:.2f}s lap={s['lap']:.0f} mae={s['mae']:.1f} "
          f"state={s['state']}")
