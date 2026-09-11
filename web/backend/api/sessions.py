"""
会话 API / Session APIs.

会话生命周期: 上传视频(建会话) -> 提取 -> 查看帧/结果/日志 -> 切换/删除
Session lifecycle: upload (create) -> extract -> inspect frames/result/log
"""

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from .. import config as web_config
from ..services import frame_data, session_store as store
from ..services.extractor import manager
from ..services.video_meta import read_video_meta

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


# ----------------------------------------------------------------------
# 会话列表 / session list
# ----------------------------------------------------------------------
@router.get("")
def list_sessions():
    """所有会话(按更新时间倒序) / All sessions, newest first."""
    sessions = store.list_sessions()
    # 实时状态修正: 若服务重启前会话卡在 extracting, 修正为可重跑状态
    # Fix stale "extracting" status after a server restart.
    for s in sessions:
        if s["status"] == web_config.STATUS_EXTRACTING \
                and s["sid"] not in manager.running_sids():
            meta = store.update_meta(s["sid"], status=web_config.STATUS_READY,
                                     stage="ready", error="服务重启, 任务中断")
            if meta:
                s["status"] = meta.get("status")
    return {"sessions": sessions}


# ----------------------------------------------------------------------
# 上传视频(创建会话) / upload video (create session)
# ----------------------------------------------------------------------
@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """上传视频并创建会话 / Upload a video and create a session."""
    filename = (file.filename or "recording").strip()
    ext = Path(filename).suffix.lower()
    if ext not in web_config.ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            400,
            f"不支持的视频格式: {ext or '(无扩展名)'}. 支持: "
            f"{', '.join(sorted(web_config.ALLOWED_VIDEO_EXTS))}")

    # 先落盘, 再校验大小(UploadFile 不可seek) / write first, then size check
    sid = store.create_session(filename, ext)
    dest = Path(store.load_meta(sid)["video"]["path"])
    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > web_config.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"文件超过上限 {web_config.MAX_UPLOAD_MB}MB")
                out.write(chunk)
    except HTTPException:
        store.delete_session(sid)
        raise
    except Exception as exc:
        store.delete_session(sid)
        raise HTTPException(500, f"保存文件失败: {exc}") from exc

    if size == 0:
        store.delete_session(sid)
        raise HTTPException(400, "上传文件为空 / empty file")

    try:
        meta = read_video_meta(dest)
    except Exception:
        meta = None
    if meta is None:
        store.delete_session(sid)
        raise HTTPException(400, "无法解码视频(可能是损坏文件或非视频)")

    store.update_meta(sid, video=meta)
    return store.load_meta(sid)


# ----------------------------------------------------------------------
# 会话详情 / session detail
# ----------------------------------------------------------------------
def _get_meta(sid: str):
    meta = store.load_meta(sid)
    if meta is None:
        raise HTTPException(404, "会话不存在 / session not found")
    return meta


@router.get("/{sid}")
def get_session(sid: str):
    """会话完整信息 / Full session info."""
    return _get_meta(sid)


@router.get("/{sid}/video")
def get_video(sid: str):
    """视频文件(浏览器播放用) / Video file (for browser playback)."""
    _get_meta(sid)
    path = store.video_path(sid)
    if path is None or not path.is_file():
        raise HTTPException(404, "视频不存在")
    return FileResponse(str(path), media_type="video/mp4")


# ----------------------------------------------------------------------
# 提取 / extraction
# ----------------------------------------------------------------------
@router.post("/{sid}/extract")
def start_extract(sid: str, ocr_mode: Optional[str] = Query(
        default="api", description="OCR模式: api/mock/local")):
    """启动提取任务(默认真实OCR) / Start extraction task (real OCR by default)."""
    _get_meta(sid)
    if ocr_mode and ocr_mode not in web_config.OCR_MODES:
        raise HTTPException(400, f"未知OCR模式: {ocr_mode}")
    started = manager.start(sid, ocr_mode=ocr_mode)
    return {"sid": sid, "started": started}


@router.delete("/{sid}")
def delete_session_api(sid: str):
    """删除会话(运行中允许: 先中止流水线再删, 见前端确认提示) /
    Delete a session (allowed while running: cancel pipeline first)."""
    _get_meta(sid)
    if manager.is_running(sid):
        # 请求中止正在跑的流水线(尽快停止 OCR/ASR, 避免占用资源后仍在已删会话上写)
        manager.cancel(sid)
    store.delete_session(sid)
    manager.drop(sid)
    return {"deleted": True}


# ----------------------------------------------------------------------
# 帧检视 / frame inspection
# ----------------------------------------------------------------------
@router.get("/{sid}/frames")
def list_frames(sid: str,
                page: int = Query(1, ge=1),
                page_size: int = Query(30, ge=1, le=200),
                group: str = Query("all")):
    """分页采样帧列表(含OCR预览) / Paginated sampled frames with OCR preview."""
    _get_meta(sid)
    if group not in frame_data.GROUPED_VERDICTS:
        raise HTTPException(400, f"未知筛选分组: {group}")
    return frame_data.list_frames(sid, page=page, page_size=page_size,
                                  group=group)


@router.get("/{sid}/frames/{fid}")
def frame_detail(sid: str, fid: int):
    """单帧详情(索引+OCR行) / Single-frame detail."""
    _get_meta(sid)
    detail = frame_data.frame_detail(sid, fid)
    if detail is None:
        raise HTTPException(404, f"帧 {fid} 不存在")
    return detail


@router.get("/{sid}/frames/{fid}/image")
def frame_image(sid: str, fid: int):
    """帧图像(诊断ROI图; 缺失时从源视频抽帧兜底) / Frame image.

    Uses the pipeline diagnostic jpg when available; otherwise extracts the
    frame from the source video (dup/blur frames are not saved as jpgs).
    """
    _get_meta(sid)
    path = frame_data.frame_image_path(sid, fid)
    if path is not None and path.is_file():
        return FileResponse(str(path), media_type="image/jpeg")
    buf = frame_data.frame_image_fallback(sid, fid)
    if buf is None:
        raise HTTPException(404, "该帧无法获取图像")
    return Response(content=buf, media_type="image/jpeg")


# ----------------------------------------------------------------------
# 结果与日志 / result & logs
# ----------------------------------------------------------------------
@router.get("/{sid}/result")
def get_result(sid: str):
    """最终提取结果 / Final extraction result."""
    _get_meta(sid)
    out_dir = store.output_dir(sid)
    # 结果按视频名输出 extracted_content_<视频名>.json, 取最新一份
    # Results are saved per video as extracted_content_<video_name>.json;
    # serve the newest one.
    files = sorted(
        out_dir.glob("extracted_content_*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise HTTPException(404, "尚无结果, 请先执行提取")
    import json
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/{sid}/logs")
def get_logs(sid: str,
             offset: int = Query(0, ge=0),
             limit: int = Query(500, ge=1, le=5000)):
    """行为日志(按行号分页) / Behavior log (paginated by line offset)."""
    _get_meta(sid)
    return frame_data.read_logs(sid, offset=offset, limit=limit)
