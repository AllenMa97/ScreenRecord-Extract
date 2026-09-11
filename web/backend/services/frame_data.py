"""
帧数据服务 / Frame data service.

从流水线落盘产物读取数据, 供前端帧检视:
  - frames_index.jsonl : 采样帧索引(帧ID/时间/状态/MAE/判定/文件名)
  - output/cache/*.jsonl : OCR断点缓存(逐帧文本行)
  - behavior_log.jsonl : 行为日志流
Reads pipeline on-disk artifacts for the frontend frame inspector:
  frames_index.jsonl / OCR checkpoint cache / behavior log.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from . import session_store as store

# 判定分组(前端筛选chips) / verdict groups for frontend filter chips
_OCR_VERDICTS = {"ocr", "ocr_reused"}
_VALUE_VERDICTS = {"value_break", "value_fallback"}
_SUPPRESSED_VERDICTS = {"suppressed", "long_scroll_sample"}
GROUPED_VERDICTS = {
    "all": None,
    "ocr": _OCR_VERDICTS,
    "dup": {"dup"},
    "blur": {"blur"},
    "suppressed": _SUPPRESSED_VERDICTS,
    "value_break": _VALUE_VERDICTS,
    "thinned": {"thinned"},
}

# 轻量缓存: (路径, mtime_ns, size) -> 解析结果 / tiny memo: key -> parsed
_OCR_CACHE: Dict[tuple, Dict[int, Dict]] = {}
_INDEX_CACHE: Dict[tuple, List[dict]] = {}
# 兜底抽帧结果缓存: 避免同一帧反复seek源视频(帧图反复查看/滚动时很常见)
# memo for source-video frame extraction so repeated views don't re-seek
_FALLBACK_CACHE: Dict[tuple, bytes] = {}
_FALLBACK_MAX = 256


def _cache_key(p: Path) -> Optional[tuple]:
    if not p.is_file():
        return None
    st = p.stat()
    return (str(p), st.st_mtime_ns, st.st_size)


def frames_index_path(sid: str) -> Optional[Path]:
    """定位 frames_index.jsonl / Locate the frames index file."""
    frames_root = store.output_dir(sid) / "frames"
    if not frames_root.is_dir():
        return None
    # 按修改时间取最新视频的索引 / newest video's index by mtime
    idx_files = sorted(frames_root.glob("*/frames_index.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    return idx_files[0] if idx_files else None


def load_frames_index(sid: str) -> List[dict]:
    """读取全部采样帧索引 / Load all sampled-frame index entries."""
    p = frames_index_path(sid)
    if p is None:
        return []
    key = _cache_key(p)
    if key is None:
        return []
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    out: List[dict] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    _INDEX_CACHE[key] = out
    if len(_INDEX_CACHE) > 64:  # 防无界增长 / cap to avoid unbounded growth
        _INDEX_CACHE.clear()
    return out


def ocr_cache_path(sid: str) -> Optional[Path]:
    """定位 OCR 断点缓存 / Locate the OCR checkpoint cache (newest)."""
    cache_root = store.output_dir(sid) / "cache"
    if not cache_root.is_dir():
        return None
    files = sorted(cache_root.glob("*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_ocr_cache(sid: str) -> Dict[int, Dict]:
    """读取 OCR 缓存: fid -> {"ts", "lines": [{"text","bbox","confidence"}]}"""
    p = ocr_cache_path(sid)
    if p is None:
        return {}
    key = _cache_key(p)
    if key is None:
        return {}
    cached = _OCR_CACHE.get(key)
    if cached is not None:
        return cached
    out: Dict[int, Dict] = {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fid = item.get("fid")
                if fid is None:
                    continue
                lines = [{
                    "text": t.get("text", ""),
                    "bbox": t.get("bbox", [0, 0, 0, 0]),
                    "confidence": round(float(t.get("confidence", 1.0)), 4),
                } for t in (item.get("lines") or []) if isinstance(t, dict)]
                # 帧条目优先; content 条目兜底 / frame entry wins, content fallback
                kind = item.get("kind")
                if kind == "content" and fid in out:
                    continue
                out[fid] = {
                    "ts": item.get("ts", 0.0),
                    "lines": lines,
                    "reused": kind == "content",
                }
    except OSError:
        return {}
    _OCR_CACHE[key] = out
    if len(_OCR_CACHE) > 64:  # 防无界增长 / cap to avoid unbounded growth
        _OCR_CACHE.clear()
    return out


def _has_image(sid: str, entry: dict) -> bool:
    """帧图像是否可取 / Whether a frame image is obtainable.

    诊断图存在优先; 否则只要源视频存在, 即可通过抽帧兜底返回图像。
    Diagnostic jpg first; otherwise the source video allows frame extraction.
    """
    fname = entry.get("file")
    if fname:
        frames_root = store.output_dir(sid) / "frames"
        for sub in frames_root.iterdir() if frames_root.is_dir() else []:
            if sub.is_dir() and (sub / fname).is_file():
                return True
    video = store.video_path(sid)
    return video is not None


def frame_image_fallback(sid: str, fid: int) -> Optional[bytes]:
    """从源视频按帧号抽帧(诊断图缺失时兜底), 带内存缓存避免反复seek。
    Extract frame from source video (fallback), memoized to avoid re-seek."""
    key = (sid, int(fid))
    hit = _FALLBACK_CACHE.get(key)
    if hit is not None:
        _FALLBACK_CACHE.pop(key)
        _FALLBACK_CACHE[key] = hit  # LRU: 移到最后 / move to tail
        return hit
    buf = _frame_image_fallback_uncached(sid, fid)
    if buf is not None:
        _FALLBACK_CACHE.pop(key, None)
        _FALLBACK_CACHE[key] = buf
        if len(_FALLBACK_CACHE) > _FALLBACK_MAX:
            _FALLBACK_CACHE.pop(next(iter(_FALLBACK_CACHE)))  # 淘汰最久 / evict oldest
    return buf


def _frame_image_fallback_uncached(sid: str, fid: int) -> Optional[bytes]:
    """从源视频按帧号抽帧(诊断图缺失时兜底) / Extract frame from source video.

    返回 JPEG 字节; 失败返回 None。帧号从1开始(与frames_index一致)。
    Returns JPEG bytes, or None on failure. frame ids start at 1.
    """
    video = store.video_path(sid)
    if video is None or not video.is_file():
        return None
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pos = max(0, min(int(fid) - 1, max(total - 1, 0)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            return None
        h, w = frame.shape[:2]
        if w > 720:  # 限制宽度, 减小传输开销 / limit width to cut transfer size
            scale = 720.0 / w
            frame = cv2.resize(frame, (720, max(1, int(h * scale))))
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buf.tobytes() if ok else None
    finally:
        cap.release()


def frame_image_path(sid: str, fid: int) -> Optional[Path]:
    """按帧ID定位图像文件 / Locate a frame image by frame id."""
    entry = next((e for e in load_frames_index(sid)
                  if e.get("frame_id") == fid), None)
    if not entry or not entry.get("file"):
        return None
    frames_root = store.output_dir(sid) / "frames"
    if not frames_root.is_dir():
        return None
    for sub in frames_root.iterdir():
        if sub.is_dir():
            p = sub / entry["file"]
            if p.is_file():
                return p
    return None


def list_frames(sid: str, page: int = 1, page_size: int = 30,
                group: str = "all") -> Dict:
    """分页返回采样帧(含OCR预览) / Paginated sampled frames with OCR preview."""
    entries = load_frames_index(sid)
    group_set = GROUPED_VERDICTS.get(group)
    if group_set is not None:
        entries = [e for e in entries
                   if (e.get("verdict") or "") in group_set]
    total = len(entries)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    items = entries[start:start + page_size]

    ocr_cache = load_ocr_cache(sid)
    out_items = []
    for e in items:
        fid = e.get("frame_id")
        ocr = ocr_cache.get(fid)
        preview = None
        if ocr:
            lines = ocr["lines"]
            preview = {
                "line_count": len(lines),
                "preview": [l["text"] for l in lines[:6]],
                "reused": ocr.get("reused", False),
            }
        out_items.append({
            "frame_id": fid,
            "timestamp": e.get("timestamp"),
            "state": e.get("state"),
            "mae": e.get("mae"),
            "verdict": e.get("verdict"),
            "file": e.get("file"),
            "has_image": _has_image(sid, e),
            "ocr": preview,
        })
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "items": out_items,
    }


def frame_detail(sid: str, fid: int) -> Optional[Dict]:
    """单帧详情(索引 + OCR行) / Single-frame detail (index + OCR lines)."""
    entry = next((e for e in load_frames_index(sid)
                  if e.get("frame_id") == fid), None)
    if entry is None:
        return None
    ocr_cache = load_ocr_cache(sid)
    ocr = ocr_cache.get(fid)
    return {
        "frame_id": fid,
        "timestamp": entry.get("timestamp"),
        "state": entry.get("state"),
        "mae": entry.get("mae"),
        "verdict": entry.get("verdict"),
        "file": entry.get("file"),
        "has_image": _has_image(sid, entry),
        "ocr": {
            "ts": ocr.get("ts") if ocr else None,
            "lines": ocr["lines"] if ocr else [],
            "reused": ocr.get("reused", False) if ocr else False,
        } if ocr else None,
    }


def behavior_log_path(sid: str) -> Path:
    return store.output_dir(sid) / "behavior_log.jsonl"


def read_logs(sid: str, offset: int = 0, limit: int = 500) -> Dict:
    """按行号读取行为日志 / Read behavior log entries by line offset."""
    p = behavior_log_path(sid)
    entries: List[dict] = []
    total = 0
    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = f.readlines()
            total = len(lines)
            for line in lines[offset:offset + limit]:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
    return {"total": total, "offset": offset, "limit": limit,
            "entries": entries, "next_offset": min(offset + limit, total)}
