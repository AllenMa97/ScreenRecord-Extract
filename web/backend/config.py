"""
Web 服务路径与常量配置 / Web service paths & constants.

目录结构 / Layout:
  web/
    backend/      FastAPI 后端 / backend package
    frontend/dist 前端构建产物 / frontend build output (Vue)
    videos/{sid}/ 会话数据(视频+输出+元数据) / session data
  src/ config.yaml 算法流水线(复用) / algorithm pipeline (reused)
"""

import sys
from pathlib import Path

# web/backend/config.py -> web/ -> 项目根 / project root
WEB_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEB_ROOT.parent
BACKEND_ROOT = WEB_ROOT / "backend"
VIDEOS_ROOT = WEB_ROOT / "videos"
FRONTEND_DIST = WEB_ROOT / "frontend" / "dist"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# 保证可导入算法源码(src.*) / Ensure the algorithm source (src.*) is importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 上传限制 / Upload limits
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_MB = 1024            # 1GB, 内部演示用 / internal demo
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# 支持的OCR模式(演示/测试) / OCR modes (demo/test)
OCR_MODES = {"api", "mock", "local"}

# 会话状态 / Session status
STATUS_UPLOADING = "uploading"
STATUS_READY = "ready"
STATUS_EXTRACTING = "extracting"
STATUS_DONE = "done"
STATUS_ERROR = "error"

# 流水线阶段(前端 Stepper 展示) / Pipeline stages (frontend stepper)
STAGES = ["opening", "roi", "warmup", "steady", "postprocessing", "done"]
