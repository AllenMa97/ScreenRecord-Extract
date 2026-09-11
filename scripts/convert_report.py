"""
结果报告转换器 / Report Converter

将extracted_content_*.json转换为更易读的Markdown和HTML报告,
输出文件名与输入JSON同名(如 extracted_content_zy.json -> .md/.html)。
Converts extracted_content_*.json into human-friendly Markdown and HTML
reports, named after the input JSON (e.g. extracted_content_zy.json -> .md/.html).

用法 / Usage:
  python scripts/convert_report.py --input output/extracted_content_zy.json
  python scripts/convert_report.py --input output/extracted_content_zy.json --output-dir reports
"""

import argparse
import json
import os
import sys

# 允许从项目根目录导入src包 / Allow importing src package from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.report_render import save_reports  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="结果报告转换器 / Report Converter")
    parser.add_argument("--input", required=True,
                        help="结果JSON路径 / Result JSON path "
                             "(e.g. output/extracted_content_zy.json)")
    parser.add_argument("--output-dir", default=None,
                        help="输出目录(默认与JSON同目录) / Output directory "
                             "(default: same dir as the JSON)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"错误: 找不到结果文件 / Error: Result file not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(input_path))
    stem = os.path.splitext(os.path.basename(input_path))[0]
    md_path, html_path = save_reports(data, output_dir, stem)

    print(f"Markdown报告: {os.path.abspath(md_path)}")
    print(f"HTML报告: {os.path.abspath(html_path)}")


if __name__ == "__main__":
    main()
