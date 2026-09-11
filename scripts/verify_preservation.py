"""
用真实缓存验证内容保全 / Verify content preservation using the real OCR cache.

mock模式 + 真实输出目录(命中真实缓存): 重放真实OCR结果, 叠加内容复用,
检查提取内容与上次运行一致。
Mock mode + real output dir (real cache): replay real OCR results with content
reuse, verify extracted content matches the previous run.
"""
import os
import sys
import json
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import Config
from src.pipeline import ExtractionPipeline

# 备份当前输出 / backup current outputs
out_dir = os.path.join(PROJECT_ROOT, "output")
bak = out_dir + "_bak_before_reuse"
if os.path.exists(bak):
    shutil.rmtree(bak)
shutil.copytree(out_dir, bak, ignore=shutil.ignore_patterns("frames", "cache"))

config = Config(os.path.join(PROJECT_ROOT, "config.yaml"))
config._data["ocr"]["mode"] = "mock"     # 不调用API, 只重放缓存 / no API, replay cache
config._data["llm"]["enabled"] = False

video = os.path.join(PROJECT_ROOT, "videos", "htkf.mp4")
p = ExtractionPipeline(config)
r = p.run(video)

ri = r.get("run_info", {})
print("\n===== 真实缓存重放 / Real-cache replay =====")
for k in ["sampled_frames", "ocr_processed", "ocr_calls_made",
          "ocr_content_reused", "ocr_cache_hit", "frames_dedup_skipped",
          "frames_cv_dup_skipped", "frames_scroll_thinned", "gaps_detected"]:
    print(f"  {k}: {ri.get(k)}")
print(f"  summary: {r.get('summary')}")

# 与备份对比summary / compare with backup
bak_json = os.path.join(bak, "extracted_content.json")
if os.path.exists(bak_json):
    with open(bak_json, encoding="utf-8") as f:
        old = json.load(f)
    print(f"  上次summary: {old.get('summary')}")
    print(f"  本次summary: {r.get('summary')}")
