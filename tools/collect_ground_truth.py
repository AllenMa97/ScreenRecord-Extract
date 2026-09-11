# -*- coding: utf-8 -*-
"""
视频 ground truth 采集器 / Video ground-truth collector

用途: 把 videos 目录下每个视频按固定时间间隔抽帧, 送真实 OCR(API)服务,
      把"可能的细粒度"屏幕内容(文本行 + bbox + 置信度 + 时间戳)保存成 JSONL,
      作为后续视觉去重算法调整的参照基准(ground truth)。

输出 / Output:
  ground_truth/<video_stem>.jsonl
  每条记录对应一个被采样的帧:
    {"video","frame_id","timestamp","lines":[{"text","bbox":[x,y,w,h],"conf"}]}

采样 / Sampling:
  --interval 秒间隔抽帧(默认 0.5s)。配合并行分块(并发=配置 ocr.concurrent,
  API 在途 job 数受此限制, 过高会被 AI Studio 限流)。

断点续传 / Resumable:
  已完成的帧 ID 记录在 <video_stem>.cursor 中; 重跑自动跳过已完成帧并从此处继续,
  不会重复调用 OCR。

用法 / Usage:
  python tools/collect_ground_truth.py [--videos DIR] [--interval 0.5]
  python tools/collect_ground_truth.py --videos "D:/手机录屏内容提取/videos"
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 项目根目录加入路径 / add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import av
import cv2
import numpy as np

from src.ocr_client import OcrClient
from src.config_loader import Config


class _NullLogger:
    def log(self, *a, **k):
        pass


def _load_ocr_config(cfg: Config) -> dict:
    """从 Config 中提取 OcrClient 需要的 ocr 段字典"""
    c = cfg.get("ocr") if isinstance(cfg.get("ocr"), dict) else {}
    return c


def _roi_box(frame: np.ndarray, cfg: Config) -> tuple:
    """ROI: 剔除状态栏/导航栏(顶部/底部 margin)"""
    h, w = frame.shape[:2]
    top = int(cfg.get("roi", "margin_top", 60) or 60)
    bottom = int(cfg.get("roi", "margin_bottom", 120) or 120)
    y0 = min(top, h - 1)
    y1 = max(y0 + 1, h - bottom)
    return (0, y0, w, y1 - y0)


def _encode_jpg(roi: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes() if ok else b""


def _collect(video_path: Path, cfg: Config, interval: float) -> int:
    client = OcrClient(_load_ocr_config(cfg), _NullLogger())
    if client.mode != "api":
        print(f"[skip] {video_path.name}: OCR 非 API 模式(mode={client.mode}), 无法采集真实 ground truth")
        return 0

    concurrent = int(cfg.get("ocr", "concurrent", 4) or 4)
    out_dir = ROOT / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    out_jsonl = out_dir / f"{stem}.jsonl"
    cursor_file = out_dir / f"{stem}.cursor"

    # 断点: 已完成的帧 ID
    done_ids = set()
    if cursor_file.exists():
        for ln in cursor_file.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln.isdigit():
                done_ids.add(int(ln))
    # 已有输出中的帧也视为完成(兼容只写jsonl没写cursor的情况)
    if out_jsonl.exists():
        for ln in out_jsonl.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                done_ids.add(int(json.loads(ln)["frame_id"]))
            except Exception:
                pass
    print(f"[{stem}] 已有/完成帧 {len(done_ids)} 个, 从 {max(done_ids, default=-1)+1} 继续")

    cont = av.open(str(video_path))
    stream = cont.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 25.0
    step = max(1, int(round(fps * interval)))
    print(f"[{stem}] fps={fps:.1f} step={step}并发={concurrent}")

    def ocr_one(meta):
        frame_id, ts, roi = meta
        lines = client.recognize(roi, frame_id=frame_id, timestamp=ts)
        return {
            "video": video_path.name,
            "frame_id": frame_id,
            "timestamp": round(ts, 4),
            "lines": [
                {"text": l.text, "bbox": list(l.bbox), "conf": round(float(l.confidence), 4)}
                for l in lines
            ],
        }

    lock = threading.Lock()
    out_handle = open(out_jsonl, "a", encoding="utf-8")
    cursor_handle = open(cursor_file, "a", encoding="utf-8")
    record_count = 0

    batch = []
    chunk = max(2, concurrent * 2)

    def flush():
        nonlocal batch, record_count
        if not batch:
            return
        with ThreadPoolExecutor(max_workers=concurrent) as ex:
            results = list(ex.map(ocr_one, batch))
        for r in results:
            out_handle.write(json.dumps(r, ensure_ascii=False) + "\n")
            cursor_handle.write(str(r["frame_id"]) + "\n")
            record_count += 1
        out_handle.flush()
        cursor_handle.flush()
        print(f"[{stem}] 已完成 {record_count} 帧采集 (累计), 最新帧 {batch[-1][0]}")
        batch = []

    try:
        for i, fr in enumerate(cont.decode(video=0)):
            if not stream.frames or i > stream.frames * 2:  # 安全上界
                break
            if i % step != 0:
                continue
            if i in done_ids:
                continue
            img = fr.to_ndarray(format="bgr24")
            x, y, w, h = _roi_box(img, cfg)
            roi = img[y:y + h, x:x + w]
            ts = i / fps if fps > 0 else 0.0
            batch.append((i, ts, roi))
            if len(batch) >= chunk:
                flush()
                done_ids.update(r[0] for r in batch)
        if batch:
            flush()
    finally:
        out_handle.close()
        cursor_handle.close()
        cont.close()
    print(f"[{stem}] 采集完成 / done, 共 {record_count} 帧")
    return record_count


def main():
    parser = argparse.ArgumentParser(description="采集视频 OCR ground truth")
    parser.add_argument("--videos", default="videos", help="视频目录 / video dir")
    parser.add_argument("--interval", type=float, default=0.5, help="采样间隔(秒) / sample interval")
    parser.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    args = parser.parse_args()

    cfg = Config(args.config)
    vpath = Path(args.videos)
    if not vpath.is_absolute():
        vpath = ROOT / vpath

    if vpath.is_file():
        videos = [vpath]
    else:
        exts = ("*.mp4", "*.mov", "*.avi", "*.mkv")
        videos = []
        for e in exts:
            videos.extend(vpath.glob(e))
        videos = sorted(set(videos))
    if not videos:
        print("未找到视频 / no videos found:", vpath)
        sys.exit(1)

    grand = 0
    for vp in videos:
        print("=" * 60)
        grand += _collect(vp, cfg, args.interval)
    print("=" * 60)
    print(f"全部完成 / all done, 总计新增 {grand} 帧采集")


if __name__ == "__main__":
    main()