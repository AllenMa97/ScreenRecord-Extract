"""
手机录屏内容提取 - 主入口 / Main Entry Point

用法 / Usage:
  python main.py --video <视频路径> [--config <配置文件路径>]
  python main.py --video <video_path> [--config <config_path>]

示例 / Example:
  python main.py --video recording.mp4
  python main.py --video recording.mp4 --config config.yaml
"""

import argparse
import sys
import os

# 将项目根目录加入路径 / Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config_loader import Config
from src.pipeline import ExtractionPipeline
from src.report_render import save_reports


def main():
    parser = argparse.ArgumentParser(
        description="手机录屏内容提取 / Screen Recorder Content Extraction"
    )
    parser.add_argument(
        "--video", "-v", default=None,
        help="视频文件路径(默认取config.yaml的video.input_path) / "
             "Video file path (default: video.input_path in config.yaml)"
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml",
        help="配置文件路径 (默认: config.yaml) / Config file path (default: config.yaml)"
    )
    args = parser.parse_args()

    # 检查配置文件 / Check config file
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在 / Error: Config file not found: {config_path}")
        sys.exit(1)

    # 加载配置 / Load config
    config = Config(config_path)

    # 解析视频路径: 优先--video, 否则取config.yaml的video.input_path
    # Resolve video path: prefer --video, fallback to video.input_path in config.yaml
    video_path = args.video or config.get("video", "input_path", "")
    if not video_path:
        print("错误: 未指定视频路径, 请使用--video参数或在config.yaml配置video.input_path "
              "/ Error: No video path. Use --video or set video.input_path in config.yaml")
        sys.exit(1)

    # 相对路径基于项目根目录 / Relative path resolved against project root
    if not os.path.isabs(video_path):
        video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), video_path)

    # 解析待处理视频列表: 目录=全部视频(逐个处理), 文件=单个视频
    # Resolve video list: directory = all videos (processed one by one), file = single video
    if os.path.isdir(video_path):
        video_files = sorted(
            f for f in os.listdir(video_path)
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        )
        if not video_files:
            print(f"错误: 目录中没有视频文件 / Error: No video file in dir: {video_path}")
            sys.exit(1)
        video_paths = [os.path.join(video_path, f) for f in video_files]
        print(f"目录扫描到 {len(video_paths)} 个视频 / Found {len(video_paths)} videos in dir")
    else:
        video_paths = [video_path]

    # 逐个处理视频 / Process each video
    exit_code = 0
    for idx, vp in enumerate(video_paths, start=1):
        print(f"\n{'=' * 60}\n[{idx}/{len(video_paths)}] 开始处理: {vp}\n{'=' * 60}")

        # 检查视频文件 / Check video file
        if not os.path.exists(vp):
            print(f"错误: 视频文件不存在 / Error: Video file not found: {vp}")
            exit_code = 1
            continue

        # 创建并运行流水线 / Create and run pipeline
        pipeline = ExtractionPipeline(config)
        result = pipeline.run(vp)

        # 输出结果摘要 / Print result summary
        print("\n" + "=" * 60)
        print(f"[{idx}/{len(video_paths)}] 处理完成 / Processing Complete")
        print("=" * 60)

        if "error" in result:
            print(f"错误: {result['error']}")
            exit_code = 1
            continue

        run_info = result.get("run_info", {})
        stats = result.get("statistics", {})

        print(f"视频: {run_info.get('video_path', 'N/A')}")
        print(f"耗时: {run_info.get('elapsed_seconds', 0):.1f}秒")
        print(f"采样帧数: {run_info.get('sampled_frames', 0)}")
        print(f"模糊跳过: {run_info.get('blur_skipped', 0)}")
        print(f"去重跳过: {run_info.get('dedup_skipped', 0)}")
        print(f"OCR处理: {run_info.get('ocr_processed', 0)}")
        print(f"挂起帧: {run_info.get('suspended_frames', 0)}")
        print(f"内容缺口: {run_info.get('gaps_detected', 0)}")
        print(f"跟踪文本行: {stats.get('total_tracked_lines', 0)}")
        print(f"LLM增强: {'是' if stats.get('llm_enhanced') else '否'}")
        print(f"输出文件: {run_info.get('output_path', 'N/A')}")
        print(f"行为日志: {run_info.get('behavior_log_path', 'N/A')}")

        # 生成人类可读报告(md/html) / Generate human-readable reports
        video_stem = os.path.splitext(os.path.basename(vp))[0]
        output_dir = os.path.dirname(run_info.get("output_path", "")) or "output"
        md_path, html_path = save_reports(
            result, output_dir, f"extracted_content_{video_stem}")
        print(f"Markdown报告: {os.path.abspath(md_path)}")
        print(f"HTML报告: {os.path.abspath(html_path)}")
        print("=" * 60)

    print(f"\n全部处理结束 / All done. 退出码 / exit code: {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
