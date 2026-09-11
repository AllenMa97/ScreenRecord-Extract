"""定量分析滚动段帧间真实差异: MAE / 行指纹共同行 / 甘迪位移。
判断: 滚动是"逐行平移(应重叠)"还是"快速跳变(无重叠)"。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import av
import numpy as np

from src.utils.image_utils import (
    to_gray, resize_frame, compute_dhash, hamming_distance,
    compute_row_diff_profile)
from src.row_fingerprint import RowFingerprintAligner


class NullLogger:
    def log(self, module, event, detail, **kw):
        pass


def build_config():
    return {
        "rf_strip_height": 16, "rf_dhash_size": 8, "rf_gap_penalty": -0.3,
        "rf_dup_ratio": 0.8, "rf_new_ratio": 0.4, "rf_noise_floor": 14,
        "rf_history_len": 3, "rf_rigid_roll_ratio": 0.7, "rf_rigid_over_ratio": 0.5,
        "compute_width": 160,
    }


def hd(a, b):
    return bin(a ^ b).count('1')


def seq_of(roi, strip_h=16, w=160):
    g = to_gray(roi)
    h, ww = g.shape[:2]
    scale = w / ww
    small = resize_frame(g, w, max(1, int(h * scale)))
    n = max(1, small.shape[0] // strip_h)
    return [compute_dhash(small[i*strip_h:(i+1)*strip_h], 8) for i in range(n)]


def main():
    src = r"D:\手机录屏内容提取\web\videos\20260826_134450_355d\source.mp4"
    roi = (0, 60, 576, 1100)
    cont = av.open(src)
    fps = float(cont.streams.video[0].average_rate)
    frame_map = {}
    for i, fr in enumerate(cont.decode(video=0)):
        frame_map[i] = fr.to_ndarray(format="bgr24")
    cont.close()

    pairs = [(10, 11), (13, 14), (14, 15), (15, 16), (16, 18), (18, 20),
             (25, 30), (40, 50), (0, 2), (2, 5)]
    print(f"{'帧对':>10} {'MAE':>8} {'共同行(≤14)':>12} {'总行':>4} {'重叠率':>6} {'位移窗口':>8}")
    # 枚举位移即可算重叠率(粗暴): 用最小平均hd
    cache = {}
    for a, b in pairs:
        if a not in cache:
            cache[a] = seq_of(frame_map[a][roi[1]:roi[3], roi[0]:roi[2]])
        if b not in cache:
            cache[b] = seq_of(frame_map[b][roi[1]:roi[3], roi[0]:roi[2]])
        sa, sb = cache[a], cache[b]
        gb = to_gray(frame_map[b][roi[1]:roi[3], roi[0]:roi[2]][:, :1920])
        ga = to_gray(frame_map[a][roi[1]:roi[3], roi[0]:roi[2]])
        # MAE(在原始ROI)
        mae = float(np.abs(ga.astype(np.float32) - gb.astype(np.float32)).mean())
        # 找最佳位移: 对每个 cur 行找与所有 prev 行的最小 hd
        nb, na = len(sb), len(sa)
        mins = []
        for j in range(nb):
            m = min(hd(sb[j], sa[i]) for i in range(na))
            mins.append(m)
        common = sum(1 for m in mins if m <= 14)
        overlap = common / max(1, nb)
        # 位移窗口: 找使重叠最大的偏移
        best_off, best_over = 0, 0
        for off in range(-6, 7):
            cnt = 0
            for j in range(nb):
                i = j - off
                if 0 <= i < na and hd(sb[j], sa[i]) <= 14:
                    cnt += 1
            if cnt > best_over:
                best_over, best_off = cnt, off
        print(f"{a}->{b:<6} {mae:>8.1f} {common:>12} {nb:>4} {overlap:>6.2f} off={best_off}(重叠{best_over})")


if __name__ == "__main__":
    main()