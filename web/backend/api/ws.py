"""
WebSocket 实时事件流 / WebSocket realtime event stream.

推送三类数据给前端:
  - manager 事件队列(阶段/帧进度/OCR帧)
  - behavior_log.jsonl 尾部增量(行为日志流)
  - 会话状态快照(连接时发送)
Pushes three data kinds to the frontend: manager events (stage / frame
progress / OCR frames), behavior-log tail increments, and an initial
session snapshot.
"""

import asyncio
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services import frame_data, session_store as store
from ..services.extractor import manager

router = APIRouter(tags=["ws"])

_LOG_TAIL_INTERVAL = 0.4       # 日志尾部轮询间隔(秒) / log tail poll interval
_MAX_DRAIN = 200               # 单次最多转发事件数 / max events forwarded per pass


@router.websocket("/api/sessions/{sid}/events")
async def session_events(ws: WebSocket, sid: str):
    """会话实时事件流 / Realtime event stream for a session."""
    await ws.accept()
    if store.load_meta(sid) is None:
        await ws.send_json({"type": "error", "detail": "会话不存在"})
        await ws.close()
        return

    # 初始快照: 会话状态 + 当前进度 / initial snapshot: status + progress
    meta = store.load_meta(sid)
    progress = meta.get("progress") or {}
    await ws.send_json({
        "type": "snapshot",
        "status": meta.get("status"),
        "stage": meta.get("stage"),
        "stage_detail": meta.get("stage_detail"),
        "progress": progress,
        "running": manager.is_running(sid),
    })

    log_path = frame_data.behavior_log_path(sid)
    offset = _file_size(log_path)
    events_q = manager.event_queue(sid)
    last_status_ts = time.time()

    try:
        while True:
            # 1) 转发 manager 事件 / forward manager events
            if events_q is not None:
                drained = []
                for _ in range(_MAX_DRAIN):
                    try:
                        drained.append(events_q.popleft())
                    except IndexError:
                        break
                for evt in drained:
                    await ws.send_json({"type": "event", "event": evt})

            # 2) 增量推送行为日志 / tail behavior log
            new_entries = _tail_log(log_path, offset)
            if new_entries:
                for entry in new_entries:
                    await ws.send_json({"type": "log", "entry": entry})

            # 3) 周期状态心跳(防断线+状态同步) / periodic status heartbeat
            now = time.time()
            if now - last_status_ts >= 2.0:
                last_status_ts = now
                meta = store.load_meta(sid)
                await ws.send_json({
                    "type": "status",
                    "status": meta.get("status"),
                    "stage": meta.get("stage"),
                    "stage_detail": meta.get("stage_detail"),
                    "progress": meta.get("progress"),
                    "error": meta.get("error"),
                    "running": manager.is_running(sid),
                })

            # 4) 等待客户端消息(超时即继续循环, 兼容ping/断线)
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


def _file_size(path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _tail_log(path, offset: int):
    """读取日志文件新增行(按字节偏移) / Read new log lines by byte offset."""
    size = _file_size(path)
    if size == offset:
        return []
    # 文件被重写且变短(运行结束save重写): 从头重读 / file rewritten & shrunk:
    # (save() rewrites at end) -> re-read from head
    if size < offset:
        offset = 0
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.seek(offset)
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
    return out
