"""
实测内容指纹OCR复用 / Measure content-fingerprint OCR reuse on a fresh run.

用临时输出目录 + mock OCR 跑完整流水线, 统计:
  - ocr_calls_made: 实际OCR调用数(复用前/后对比基准)
  - ocr_content_reused: 内容复用跳过OCR的帧数
  - frames_ocr_processed: 消费帧数(含复用)
"""
import os
import sys
import tempfile
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import Config
from src.pipeline import ExtractionPipeline

tmp = tempfile.mkdtemp(prefix="htkf_reuse_")
config = Config(os.path.join(PROJECT_ROOT, "config.yaml"))
config._data["output"]["dir"] = tmp
config._data["ocr"]["mode"] = "mock"
config._data["llm"]["enabled"] = False

video = os.path.join(PROJECT_ROOT, "videos", "htkf.mp4")
p = ExtractionPipeline(config)
r = p.run(video)

ri = r.get("run_info", {})
print("\n===== 实测结果 / Measurement =====")
for k in ["sampled_frames", "frames_ocr_processed", "frames_dedup_skipped",
          "frames_cv_dup_skipped", "frames_scroll_thinned", "ocr_calls_made",
          "ocr_content_reused", "ocr_cache_hit", "content_reuse_max_hd",
          "gray_zone_count", "calibration_count"]:
    print(f"  {k}: {ri.get(k)}")
print(f"  htkf视频({ri.get('total_video_frames')}帧): "
      f"OCR调用{ri.get('ocr_calls_made')}次, "
      f"内容复用{ri.get('ocr_content_reused')}帧, "
      f"消费{ri.get('frames_ocr_processed')}帧")

# 缓存内容条目检查 / check content entries in cache
cache_path = ri.get("ocr_cache_path")
if cache_path and os.path.exists(cache_path):
    n_fid, n_content = 0, 0
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if json.loads(line).get("kind") == "content":
                n_content += 1
            else:
                n_fid += 1
    print(f"  缓存: {n_fid}条帧条目, {n_content}条内容指纹")

print(f"  临时输出: {tmp}")
