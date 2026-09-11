"""
视频元数据读取 / Video metadata extraction.

使用 OpenCV 读取格式/时长/fps/帧数/分辨率等, 供前端"视频信息"区展示。
Uses OpenCV to read format/duration/fps/frame-count/resolution for the
frontend "video info" panel.
"""

import os
from pathlib import Path
from typing import Dict, Optional

import cv2


def _fmt_bytes(n: int) -> str:
    """人类可读文件大小 / Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def read_video_meta(path: Path) -> Optional[Dict]:
    """
    读取视频元数据 / Read video metadata.

    返回 / Returns: dict or None(无法打开)
      filename, format, size_bytes, size_human,
      duration_sec, fps, total_frames, width, height, codec
    """
    if not path.is_file():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    finally:
        cap.release()

    codec = ""
    if fourcc:
        try:
            codec = "".join(chr(int(int(fourcc) >> (8 * i)) & 0xFF)
                            for i in range(4)).strip()
        except ValueError:
            codec = ""

    duration = total / fps if fps > 0 else 0.0
    size = path.stat().st_size
    return {
        "filename": path.name,
        "format": path.suffix.lstrip(".").upper(),
        "size_bytes": size,
        "size_human": _fmt_bytes(size),
        "duration_sec": round(duration, 2),
        "fps": round(fps, 2),
        "total_frames": total,
        "width": width,
        "height": height,
        "codec": codec,
    }
