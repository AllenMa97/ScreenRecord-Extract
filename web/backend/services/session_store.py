"""
会话存储 / Session store.

每个会话一个目录: web/videos/{sid}/
  source.{ext}  视频文件 / uploaded video
  meta.json     会话元数据 / session metadata
  output/       流水线输出(结果/日志/帧/缓存) / pipeline output
"""

import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import VIDEOS_ROOT, STATUS_READY

_META_NAME = "meta.json"
_VIDEO_NAMES = ("source.mp4", "source.mov", "source.avi",
                "source.mkv", "source.webm", "source.m4v")


def new_sid() -> str:
    """生成会话ID / Generate a session id."""
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def session_dir(sid: str) -> Path:
    return VIDEOS_ROOT / sid


def _find_video(d: Path) -> Optional[Path]:
    """在会话目录中查找视频文件 / Find the video file in a session dir."""
    if not d.is_dir():
        return None
    for name in _VIDEO_NAMES:
        p = d / name
        if p.is_file():
            return p
    # 兼容任意扩展名 / fallback: any file with a video-like extension
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi",
                                                ".mkv", ".webm", ".m4v"):
            return p
    return None


def create_session(display_name: str, ext: str) -> str:
    """创建会话并返回 sid / Create a session, return sid."""
    sid = new_sid()
    d = session_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "output").mkdir(exist_ok=True)
    meta = {
        "sid": sid,
        "name": display_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": STATUS_READY,
        "stage": "ready",
        "stage_detail": "",
        "progress": None,
        "error": None,
        "video": {
            "filename": f"source{ext.lower()}",
            "path": str(d / f"source{ext.lower()}"),
        },
    }
    write_meta(sid, meta)
    return sid


def video_path(sid: str) -> Optional[Path]:
    """会话视频文件路径 / Video file path of a session."""
    return _find_video(session_dir(sid))


def output_dir(sid: str) -> Path:
    return session_dir(sid) / "output"


def meta_path(sid: str) -> Path:
    return session_dir(sid) / _META_NAME


def write_meta(sid: str, meta: Dict[str, Any]) -> None:
    meta["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = meta_path(sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta(sid: str) -> Optional[Dict[str, Any]]:
    p = meta_path(sid)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def update_meta(sid: str, **updates: Any) -> Optional[Dict[str, Any]]:
    """加载-合并-保存会话元数据 / Load-merge-save session metadata."""
    meta = load_meta(sid)
    if meta is None:
        return None
    meta.update(updates)
    write_meta(sid, meta)
    return meta


def list_sessions() -> List[Dict[str, Any]]:
    """按更新时间倒序列出所有会话 / List sessions, newest first."""
    out: List[Dict[str, Any]] = []
    if not VIDEOS_ROOT.is_dir():
        return out
    for d in sorted(VIDEOS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime
                    if p.is_dir() else 0, reverse=True):
        if not d.is_dir() or not (d / _META_NAME).is_file():
            continue
        meta = load_meta(d.name)
        if meta is None:
            continue
        out.append(summarize(meta, d))
    return out


def summarize(meta: Dict[str, Any], d: Path) -> Dict[str, Any]:
    """精简会话信息(列表用) / Compact session info for list view."""
    vid = meta.get("video", {})
    progress = meta.get("progress") or {}
    return {
        "sid": meta.get("sid"),
        "name": meta.get("name"),
        "status": meta.get("status"),
        "stage": meta.get("stage"),
        "stage_detail": meta.get("stage_detail"),
        "error": meta.get("error"),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
        "video": {
            "filename": vid.get("filename"),
            "size_bytes": vid.get("size_bytes"),
            "duration_sec": vid.get("duration_sec"),
            "total_frames": vid.get("total_frames"),
            "fps": vid.get("fps"),
        },
        "progress": {
            "frame_id": progress.get("frame_id"),
            "total_frames": progress.get("total_frames"),
            "percent": progress.get("percent"),
            "elapsed_sec": progress.get("elapsed_sec"),
        },
        "statistics": meta.get("statistics"),
    }


def delete_session(sid: str) -> bool:
    d = session_dir(sid)
    if not d.is_dir():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


_lock = threading.Lock()


def lock() -> threading.Lock:
    """跨模块共享锁(元数据读改写串行化) / Shared lock for meta read-modify-write."""
    return _lock
