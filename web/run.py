"""
录屏内容提取 Web 服务便捷启动脚本 / Convenience launcher.

用法 / Usage:
  python web/run.py [--port 8000] [--host 127.0.0.1]
"""

import argparse
import sys
from pathlib import Path

# 项目根目录加入路径(确保 import src.* 与 web.backend.* 均可用)
# Add project root to sys.path so both src.* and web.backend.* import.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="启动录屏提取Web服务 / Run the web demo")
    parser.add_argument("--host", default="0.0.0.0",
                        help="监听地址: 0.0.0.0=局域网可访问(演示用) / "
                             "127.0.0.1=仅本机 / bind host")
    parser.add_argument("--port", type=int, default=8000,
                        help="监听端口 / bind port")
    parser.add_argument("--reload", action="store_true",
                        help="开发热重载 / hot reload (dev)")
    args = parser.parse_args()

    print("=" * 56)
    print("手机录屏内容提取 - 演示服务 / Screen-Recorder Extraction Demo")
    print(f"访问 / open: http://{args.host}:{args.port}")
    print("=" * 56)

    uvicorn.run("web.backend.main:app",
                host=args.host,
                port=args.port,
                reload=args.reload,
                workers=1)


if __name__ == "__main__":
    main()
