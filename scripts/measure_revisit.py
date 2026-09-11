"""
结构级head+tail潜力测量(无需OCR) / Structural head+tail potential (no OCR needed).

对任意视频: 均匀采样, 计算16x16 dHash(160宽灰度), 统计:
  1. 唯一内容数 & 回访率(同一内容在时间线上再次出现)
  2. 头向/尾向覆盖曲线 → meet-in-the-middle 提前终止的理论节省
For any video: uniform sample, 16x16 dHash on 160-wide gray, measure:
  1. unique contents & revisit ratio
  2. head/tail coverage curves → theoretical early-termination saving
"""
import os
import sys
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.image_utils import to_gray, compute_dhash

video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    PROJECT_ROOT, "videos", "xxr.mp4")
sample_step = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5  # 秒 / sec

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    print("无法打开 / cannot open:", video)
    sys.exit(1)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"视频: {os.path.basename(video)} fps={fps:.1f} 总帧={total} "
      f"时长={total/fps:.1f}s 采样间隔={sample_step}s")

first = {}
last = {}
obs = {}
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
idx = 0
while True:
    t_target = idx * sample_step
    f_target = int(round(t_target * fps))
    if f_target >= total:
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_target)
    ok, frame = cap.read()
    if not ok:
        break
    # 降采样到160宽灰度(与流水线dedup一致) / downsample like pipeline dedup
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (160, int(h * 160 / w)), interpolation=cv2.INTER_AREA)
    gray = to_gray(small)
    h16 = compute_dhash(gray, 16)
    if h16 not in first:
        first[h16] = (f_target, t_target)
    last[h16] = (f_target, t_target)
    obs[h16] = obs.get(h16, 0) + 1
    idx += 1
cap.release()

n = len(first)
print(f"采样帧: {idx}  唯一内容: {n}")

# 回访: 末次-首次 > 1.5s (容忍同内容短暂动画) / revisit if span > 1.5s
revisit = [h for h in first if last[h][0] - first[h][0] > 1.5 * fps and obs[h] >= 2]
print(f"回访内容: {len(revisit)}/{n} = {len(revisit)/n*100:.1f}%")

# 头向覆盖 / head coverage
sf = sorted((first[h][0], h) for h in first)
total_f = total - 1
for pct in (50, 75, 90, 95, 100):
    target = n * pct / 100
    fid = None
    for fid, h in sf:
        if sum(1 for f0, _ in sf if f0 <= fid) >= target:
            break
    print(f"  头向{pct}%: 帧{fid} ({fid/total_f*100:.0f}%视频)")
