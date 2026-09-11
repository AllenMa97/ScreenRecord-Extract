# -*- coding: utf-8 -*-
"""
框架级去重验证: 以真实 OCR ground truth 为基准, 评估当前行指纹对齐去重器(RFSA)
在真实视频上的 召回/冗余 权衡。

用法 / Usage:
  python tools/validate_dedup.py <video_stem>

对 ground_truth/<stem>.jsonl 中每个被采样帧(真实OCR文本, 已按时间序):
  1. 重开视频, 在相同 frame_id 取 ROI, 喂给 RowFingerprintAligner.compare(当前框架)。
  2. compare 返回 OCR(判 new/gray) 或 skip(判 dup)。
  3. 度量三件事:
      OCR 调用次数      = 成本
      丢失度(missed)   = 只出现在"被跳过帧"、从未被任何 OCR 帧捕获的文本行
                         (框架把真实新内容误判为 dup → 召回损失)
      冗余(wasted)     = 判 OCR 但未带来任何"此前未捕获"新文本的帧数
                         (滚动反复触发无价值 OCR → 资源浪费)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import av

from src.config_loader import Config
from src.row_fingerprint import RowFingerprintAligner


class _NullLogger:
    def log(self, *a, **k):
        return None


def _norm(t: str) -> str:
    return " ".join(t.split())


def _roi_box(img, cfg) -> tuple:
    h, w = img.shape[:2]
    top = int(cfg.get("roi", "margin_top", 60) or 60)
    bottom = int(cfg.get("roi", "margin_bottom", 120) or 120)
    y0 = min(top, h - 1)
    y1 = max(y0 + 1, h - bottom)
    return (0, y0, w, y1 - y0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--no-phase", action="store_true",
                    help="关闭相位归一化(对比相位敏感旧指纹)")
    args = ap.parse_args()

    cfg = Config(str(ROOT / "config.yaml"))
    vd = cfg.get("visual_dedup")
    rf_cfg = dict(vd)
    rf_cfg["compute_width"] = vd.get("compute_width", 160)
    if args.no_phase:
        rf_cfg["rf_phase_enabled"] = False

    tag = "Phase-ON " if not args.no_phase else "Phase-OFF"

    gt_path = ROOT / "ground_truth" / f"{args.stem}.jsonl"
    gt = {}
    with open(gt_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            lines = {_norm(li.get("text", "")) for li in r.get("lines", []) if li.get("text")}
            gt[int(r["frame_id"])] = lines
    ids = sorted(gt.keys())
    print(f"[{args.stem}] ground truth 帧数 = {len(ids)}")

    aligner = RowFingerprintAligner(rf_cfg, _NullLogger())
    video = ROOT / "videos" / f"{args.stem}.mp4"
    cont = av.open(str(video))
    stream = cont.streams.video[0]

    id_set = set(ids)
    decisions = {}
    captured = set()
    ocr_calls = 0
    wasted = 0
    idx = 0
    for fr in cont.decode(video=0):
        if idx in id_set:
            img = fr.to_ndarray(format="bgr24")
            x, y, w, h = _roi_box(img, cfg)
            roi = img[y:y + h, x:x + w]
            res = aligner.compare(roi)
            if res.verdict == "dup":
                decisions[idx] = "skip"
            else:
                decisions[idx] = "ocr"
                ocr_calls += 1
                lines = gt[idx]
                if not (lines - captured):
                    wasted += 1            # 判 OCR 但全部是已见文本 → 冗余
                captured |= lines
        idx += 1
        if idx > ids[-1]:
            break

    all_lines = set().union(*gt.values()) if gt else set()
    # 丢失 = 存在于"被跳过帧"、且从未被任何 OCR 帧捕获的行
    lost = set()
    for fid in ids:
        if decisions.get(fid) != "ocr":
            lost |= (gt[fid] - captured)
    recall = len(captured) / len(all_lines) if all_lines else 1.0

    print(f"\n== 评估结果 [{tag}] =="
          f"\n  采样帧      = {len(ids)}"
          f"\n  OCR 调用     = {ocr_calls}   (节省 {len(ids) - ocr_calls} 帧)"
          f"\n  唯一文本行   = {len(all_lines)}  已捕获 = {len(captured)}"
          f"\n  召回率       = {recall:.1%}"
          f"\n  丢内容(行)   = {len(lost)}"
          f"\n  冗余OCR帧    = {wasted}  / {ocr_calls}")

    if lost:
        print("\n  丢失内容样例(被误判 dup 跳过) :")
        shown = 0
        for fid in ids:
            if decisions.get(fid) != "ocr":
                ls = sorted(gt[fid] & lost)
                if ls:
                    print(f"    frame={fid}: {ls[:3]}")
                    shown += 1
                    if shown >= 8:
                        break
    # 判定分布
    from collections import Counter
    print("\n  skip/ocr 帧数:", Counter(decisions.values()))


if __name__ == "__main__":
    main()