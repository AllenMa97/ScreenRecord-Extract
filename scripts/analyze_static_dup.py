"""
用原始视频帧验证dHash16噪声水平 / Validate dHash16 noise on RAW video frames.

从视频直接解码f300-f329区域, 计算:
  1. 相邻采样帧的MAE与hd16(原始帧, 无JPEG重编码)
  2. 与f304(同内容)的hd16
  3. 与远段落帧(不同内容)的hd16
"""
import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.image_utils import resize_frame, compute_dhash, hamming_distance, compute_mae


def prep_gray(gray: np.ndarray, w: int = 160) -> np.ndarray:
    if gray.shape[1] > w:
        scale = w / gray.shape[1]
        gray = resize_frame(gray, w, max(1, int(gray.shape[0] * scale)))
    return gray


def hd16(gray_a, gray_b) -> int:
    return hamming_distance(
        compute_dhash(prep_gray(gray_a), 16),
        compute_dhash(prep_gray(gray_b), 16))


video = os.path.join(PROJECT_ROOT, "videos", "htkf.mp4")
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"fps={fps}")

# 收集目标帧索引/时间 / target frames: fid 300-329
targets = {300: 6.082, 304: 6.173, 305: 6.191, 306: 6.209, 308: 6.246,
           311: 6.30, 314: 6.355, 317: 6.409, 322: 6.518, 326: 6.591, 329: 6.645}
grays = {}
idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = idx / fps
    if 5.5 <= t <= 8.0:  # f300~f329区域 / region
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays[idx] = g
    idx += 1
cap.release()
print(f"解码{len(grays)}帧")

# 帧号与索引的映射: 采样fid与视频帧号一致? 用时间定位 / locate by timestamp
def get_by_t(t0, tol=0.01):
    for i, g in grays.items():
        if abs(i / fps - t0) < tol:
            return i, g
    return None, None

f304 = get_by_t(6.173)
print("f304:", "找到" if f304[1] is not None else "未找到")

print("\n== 同内容: 与f304的MAE/hd16 (原始帧) ==")
base_i, base_g = f304
for name, t in targets.items():
    if name == 304:
        continue
    i, g = get_by_t(t)
    if g is None:
        print(f"  f{name}({t}s): 未找到"); continue
    mae = compute_mae(prep_gray(base_g), prep_gray(g))
    hd = hd16(base_g, g)
    print(f"  f{name}({t}s): mae={mae:.3f}, hd16={hd}/256")

print("\n== 相邻OCR采样帧: MAE/hd16 (原始帧) ==")
prev_t = None
for name, t in sorted(targets.items()):
    i, g = get_by_t(t)
    if prev_t is not None and g is not None:
        pi, pg = prev_t
        mae = compute_mae(prep_gray(pg), prep_gray(g))
        hd = hd16(pg, g)
        print(f"  f{name}: vs上一采样帧 mae={mae:.3f}, hd16={hd}/256")
    prev_t = (i, g)

print("\n== 不同内容: 远段落 vs f304 (原始帧) ==")
for t in [1.0, 2.5, 4.0, 5.0, 9.0, 12.0, 15.0]:
    i, g = get_by_t(t, tol=0.05)
    if g is None:
        print(f"  {t}s: 未找到"); continue
    hd = hd16(base_g, g)
    print(f"  t={t}s: hd16={hd}/256")
