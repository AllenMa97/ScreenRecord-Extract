# -*- coding: utf-8 -*-
"""检测层召回上限: 不做去重, 把所有帧检测到的高分气泡按"尺寸签名+w容差"聚类,
统计整片视频能看到的不同语音气泡数(即去重后理论上限)。并与用户肉眼计数对照。
Usage: python tools/recall_ceiling.py <video> <roi>
"""
import sys
from pathlib import Path
import cv2
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visual.voice_bubble_detector import detect_voice_bubbles

def main():
    video = sys.argv[1]; x0,y0,w0,h0 = [int(v) for v in sys.argv[2].split(",")]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, int(round(0.2*fps)))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 收集所有帧的高分气泡检测 (按 时间窗 但只做尺寸签名聚类——位置无关)
    seen = []  # (ts, box, side, score)
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if fid % skip != 0:
            fid += 1; continue
        ts = fid / fps
        roi = frame[y0:y0+h0, x0:x0+w0]
        for b in detect_voice_bubbles(roi):
            if b.voice_score < 0.55: continue
            if b.w < W*0.75 or b.h < H*0.25:  # 过滤整宽状态栏
                seen.append((ts, (b.x, b.y, b.w, b.h), b.side, b.voice_score))
        fid += 1
    cap.release()
    print(f"跨帧高分气泡实例 = {len(seen)} (threshold=0.55, 去整宽条)")
    # 尺寸签名聚类(位置无关), 只按 (side, w桶, h桶)
    sig = defaultdict(list)
    for ts, box, side, sc in seen:
        bx, by, bw, bh = box
        sig[(side, int(round(bw/15)*15), int(round(bh/8)*8))].append(ts)
    print(f"尺寸签名簇数(检测层) = {len(sig)}")
    for (side, wb, hb), ts in sorted(sig.items(), key=lambda kv: kv[0][2]):
        print(f"  {side or '?':<2} w~{wb} h~{hb}: {len(ts)}次 t={min(ts):6.2f}~{max(ts):6.2f}")

if __name__ == "__main__":
    main()