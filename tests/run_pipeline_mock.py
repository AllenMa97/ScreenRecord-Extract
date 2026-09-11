"""用当前代码跑一次 mock 管线, 验证滚动记忆库/滚动门控是否激活 / Run the pipeline once in mock mode with current code to verify the rolling-memory library & gate activate."""
import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import Config
from src.pipeline import ExtractionPipeline

VIDEO = r"D:\手机录屏内容提取\web\videos\20260826_134450_355d\source.mp4"

def main():
    cfg = Config(os.path.join(PROJECT_ROOT, "config.yaml"),
                 overrides={"output": {"save_frames": True},
                            "ocr": {"mode": "mock", "cache_persist_mock": True}})
    print("rf_lib_enabled =", cfg.get("visual_dedup", "rf_lib_enabled"))
    print("rf_match_floor =", cfg.get("visual_dedup", "rf_match_floor"))
    print("rf_enabled     =", cfg.get("visual_dedup", "rf_enabled"))
    out = tempfile.mkdtemp(prefix="mockrun_")
    events = []
    pipeline = ExtractionPipeline(cfg, output_dir=out, save_frames=True,
                                  progress_callback=lambda e: events.append(e),
                                  live_log=True)
    result = pipeline.run(VIDEO)
    print("\n=== run done ===")
    print("result keys:", list(result.keys()))
    log = os.path.join(out, "behavior_log.jsonl")
    print("log path:", log, "exists:", os.path.exists(log))
    if os.path.exists(log):
        import json
        rows = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
        bfs = [r for r in rows if r.get("module") == "row_fingerprint" and r.get("event") == "compare"]
        libs = [r["extra"].get("lib_size") for r in bfs]
        drop = sum((r["extra"].get("dropped_revisit") or 0) for r in bfs)
        verdict = {}
        for r in bfs:
            v = r["extra"].get("verdict")
            verdict[v] = verdict.get(v, 0) + 1
        print(f"\ncompare 事件 {len(bfs)}:")
        print(f"  累积库大小 max={max(libs) if libs else 0} (应为>0 说明已激活)")
        print(f"  复现剔除累计 dropped={drop}")
        print(f"  判定分布 = {verdict}")
        print("\n关键帧数=3 的样本(+同步滚动一致性字段) => 属新代码")
        sample = [r["extra"] for r in bfs[:8]]
        for s in sample:
            print("  ", {k: s.get(k) for k in ("lib_size","dropped_revisit","rolling_consistent","verdict","aligned_ratio")})

if __name__ == "__main__":
    main()