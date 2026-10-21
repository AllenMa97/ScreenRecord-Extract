# -*- coding: utf-8 -*-
"""
公开数据集适配器 / Public-dataset adapter

把一个 HuggingFace 上的**带标注截图**数据集, 通过 datasets-server REST 接口取若干样本,
转成本项目的标签格式(labels.jsonl + meta.json), 从而可直接用 tools/eval_synthetic.py
评测跟踪/装配层(无需改评测代码)。

工作方式 / How:
  1. 取 N 行样本(每行 = 一张截图 + 逐元素的 bbox/labels/texts);
  2. 逐元素转成真值行 {text, bbox}; 空文本元素跳过; bbox 裁到画面内;
  3. 合成 --frames 帧: 第 0 帧用原始框, 其余帧对框加 +-jitter 抖动(模拟 OCR 框位方差),
     文本保持不变 => 既可评"并簇是否正确", 也可评"抖动下的稳健性";
  4. 输出 synthetic/public/<slug>.labels.jsonl 与 <slug>.meta.json, 并保存原图。

默认数据集 / Default dataset:
  docling-project/screenparse  (网页截图, 每元素 bbox+类别+文本, 约 144 万行)

用法 / Usage:
  python tools/fetch_public_sample.py --count 5 --proxy http://127.0.0.1:10809
  python tools/eval_synthetic.py --labels synthetic/public
"""
import argparse
import json
import os
import random
import re
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ROWS_API = "https://datasets-server.huggingface.co/rows"


def _build_opener(proxy: str):
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    handlers.append(urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    return urllib.request.build_opener(*handlers)


def _get(url: str, opener, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "trae-public-adapter/1.0"})
    with opener.open(req, timeout=timeout) as resp:
        return resp.read()


def fetch_rows(dataset: str, config: str, split: str, offset: int, count: int, opener):
    """取样本行 / fetch sample rows"""
    url = ROWS_API + "?" + urllib.parse.urlencode({
        "dataset": dataset, "config": config, "split": split,
        "offset": offset, "length": count,
    })
    data = json.loads(_get(url, opener).decode("utf-8"))
    if "error" in data:
        raise SystemExit(f"HF 返回错误: {data['error']}")
    return data.get("rows", [])


def _slugify(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")
    return (s[:max_len] or "sample")


def build_scenario(row, ds_slug: str, frames: int, jitter: int,
                   out_dir: Path, opener, save_image: bool) -> int:
    """把一行样本转成一个场景, 返回真值行数 / build one scenario, return #truth lines"""
    r = row["row"]
    W, H = int(r["width"]), int(r["height"])
    slug = _slugify(r.get("id", row.get("row_idx", 0)))

    # 逐元素 -> 真值行 (x,y,w,h), 裁进画面 / elements -> truth lines, clamped
    lines = []
    for box, text in zip(r.get("bboxes", []), r.get("texts", [])):
        text = (text or "").strip()
        if not text or len(box) < 4:
            continue
        x, y, w, h = (int(v) for v in box[:4])
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        lines.append((text, x, y, w, h))

    if not lines:
        print(f"  [skip] {slug}: 无有效文本元素")
        return 0

    fps = 10
    rng = random.Random(abs(hash(slug)) % (2 ** 31))
    label_path = out_dir / f"{slug}.labels.jsonl"
    with open(label_path, "w", encoding="utf-8") as f:
        for fid in range(frames):
            ts = round(fid / fps, 4)
            for text, x, y, w, h in lines:
                if fid == 0:
                    bx, by = x, y
                else:  # 抖动模拟 OCR 框位方差 / jitter to mimic框位 variance
                    bx = max(0, min(W - w, x + rng.randint(-jitter, jitter)))
                    by = max(0, min(H - h, y + rng.randint(-jitter, jitter)))
                f.write(json.dumps({"text": text, "bbox": [bx, by, w, h],
                                    "frame_id": fid, "timestamp": ts},
                                   ensure_ascii=False) + "\n")

    meta = {
        "video_id": slug,
        "fps": fps,
        "resolution": [W, H],
        "duration_sec": round(frames / fps, 3),
        "behavior_profile": "public_screenshot",
        "case_id": ds_slug,
        "style_title": str(r.get("url", "")),
        "title_height": 0,      # 截图即全画面, 无标题栏/导航栏
        "nav_height": 0,
        "frame_count": frames,
        "scroll_seq": [0] * frames,
        "frame_types": ["chat"] * frames,
        "messages": [
            {"idx": i, "sender_id": "public", "is_received": True,
             "msg_type": "text", "text": t, "voice_transcribed": ""}
            for i, (t, *_rest) in enumerate(lines)
        ],
        "source": {"dataset": ds_slug, "row_idx": row.get("row_idx"),
                   "url": r.get("url", ""), "num_elements": r.get("num_elements")},
    }
    (out_dir / f"{slug}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if save_image:
        src = (r.get("image") or {}).get("src")
        if src:
            img_dir = out_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            try:
                (img_dir / f"{slug}.jpg").write_bytes(_get(src, opener))
            except Exception as exc:  # 图片失败不影响标签 / image failure is non-fatal
                print(f"  [warn] {slug}: 图片下载失败 {exc}")

    print(f"  [ok] {slug}: 真值行 {len(lines)}, 帧 {frames}, 画面 {W}x{H}")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="docling-project/screenparse")
    ap.add_argument("--config", default="default")
    ap.add_argument("--split", default="train")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--frames", type=int, default=8,
                    help="每张截图合成的帧数(用于跟踪层)")
    ap.add_argument("--jitter", type=int, default=2, help="非首帧 bbox 抖动像素")
    ap.add_argument("--out", default=str(ROOT / "synthetic" / "public"))
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY", ""))
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    opener = _build_opener(args.proxy)
    ds_slug = _slugify(args.dataset.replace("/", "_"))

    print(f"[{args.dataset}] offset={args.offset} count={args.count} "
          f"frames={args.frames} jitter={args.jitter} proxy={args.proxy or '-'}")
    rows = fetch_rows(args.dataset, args.config, args.split,
                      args.offset, args.count, opener)
    total = 0
    for row in rows:
        total += build_scenario(row, ds_slug, args.frames, args.jitter,
                               out_dir, opener, not args.no_images)
    print(f"\n完成: {len(rows)} 个场景, 共 {total} 条真值行 -> {out_dir}")


if __name__ == "__main__":
    main()
