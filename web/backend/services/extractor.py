"""
提取任务管理器 / Extraction task manager.

在后台线程运行算法流水线, 通过 progress_callback 接收实时事件:
  - 更新会话元数据(状态/阶段/进度)
  - 事件推入每会话队列(WebSocket 实时转发)
Runs the algorithm pipeline in a background thread; receives realtime
events via progress_callback: updates session metadata and pushes
events into a per-session queue (forwarded over WebSocket).
"""

import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

from .. import config as web_config
from ..config import CONFIG_PATH
from src.config_loader import Config
from src.pipeline import ExtractionPipeline

from . import session_store as store
from . import voice_audio


class ExtractionManager:
    """会话提取任务管理 / Per-session extraction task manager."""

    def __init__(self):
        self._tasks: Dict[str, Dict] = {}   # sid -> {"thread", "events", "lock", "last_push"}
        self._lock = threading.Lock()
        self._meta_lock = store.lock()

    # ------------------------------------------------------------------
    # 任务查询 / task queries
    # ------------------------------------------------------------------
    def is_running(self, sid: str) -> bool:
        with self._lock:
            task = self._tasks.get(sid)
            return bool(task and task["thread"].is_alive())

    def running_sids(self) -> List[str]:
        with self._lock:
            return [sid for sid, t in self._tasks.items()
                    if t["thread"].is_alive()]

    def event_queue(self, sid: str) -> Optional[Deque[dict]]:
        """获取会话事件队列(WS 转发用) / Get a session's event queue."""
        with self._lock:
            task = self._tasks.get(sid)
            return task["events"] if task else None

    # ------------------------------------------------------------------
    # 启动 / start
    # ------------------------------------------------------------------
    def start(self, sid: str, ocr_mode: Optional[str] = None) -> bool:
        """启动提取任务(幂等) / Start extraction (idempotent)."""
        if self.is_running(sid):
            return False
        events: Deque[dict] = deque(maxlen=2000)
        started_at = time.time()
        thread = threading.Thread(target=self._run,
                                  args=(sid, ocr_mode, events),
                                  daemon=True,
                                  name=f"extract-{sid}")
        with self._lock:
            self._tasks[sid] = {
                "thread": thread,
                "events": events,
                "last_push": 0.0,
                "started_at": started_at,
                "max_fid": 0,   # 进度单调峰值(重置关键帧会回头重采) / monotonic peak frame id
                "cancel_event": threading.Event(),
            }
        thread.start()
        return True

    # ------------------------------------------------------------------
    # 取消 / 摘除 / cancel & drop
    # ------------------------------------------------------------------
    def cancel(self, sid: str) -> None:
        """请求中止该会话的流水线(删除运行中会话时调用)。"""
        with self._lock:
            task = self._tasks.get(sid)
            if task:
                task["cancel_event"].set()

    def drop(self, sid: str) -> None:
        """从任务表移除该会话(线程为后台线程, 删除后不再写已删会话)。"""
        with self._lock:
            self._tasks.pop(sid, None)

    # ------------------------------------------------------------------
    # 主运行逻辑 / main runner
    # ------------------------------------------------------------------
    def _run(self, sid: str, ocr_mode: Optional[str], events: Deque[dict]) -> None:
        video = store.video_path(sid)
        if video is None:
            self._fail(sid, events, "会话视频不存在 / video not found")
            return
        out_dir = store.output_dir(sid)

        # 清理上次提取产物(缓存/帧/日志/结果), 保证每次提取干净重跑。
        # 否则 mock 运行写下的 OCR 断点缓存会被 api 重跑误加载(跳过真实OCR),
        # 导致结果退化为旧 mock 文本。
        # Clean previous artifacts so every run starts fresh; otherwise OCR
        # checkpoint caches written by a mock run get reloaded by a real run
        # and the output degrades to stale mock text.
        self._clean_output(out_dir)

        # 会话输出目录隔离, 强制保存诊断帧 + mock也落盘OCR缓存
        # (会话隔离下无交叉污染, 且保证帧级OCR结果完成后仍可回溯)
        # Session output isolation: force diagnostic frames; persist OCR cache
        # in mock mode too (no cross-session pollution; frame-level OCR stays
        # inspectable after the run).
        overrides = {
            "output": {"save_frames": True},
            "ocr": {"cache_persist_mock": True},
        }
        if ocr_mode and ocr_mode in web_config.OCR_MODES:
            overrides["ocr"]["mode"] = ocr_mode

        store.update_meta(sid, status=web_config.STATUS_EXTRACTING,
                          stage="opening", stage_detail="准备启动算法流水线",
                          error=None, audio_progress=None)

        def on_event(evt: dict) -> None:
            self._on_event(sid, events, evt)

        try:
            config = Config(str(CONFIG_PATH), overrides=overrides)
            cancel_event = None
            with self._lock:
                task = self._tasks.get(sid)
                if task:
                    cancel_event = task["cancel_event"]
            pipeline = ExtractionPipeline(
                config,
                output_dir=str(out_dir),
                save_frames=True,
                progress_callback=on_event,
                live_log=True,
                cancel_event=cancel_event,
            )
            result = pipeline.run(str(video))
        except Exception as exc:  # noqa: BLE001 - 任何异常都转为会话错误
            self._fail(sid, events, f"{exc}")
            return

        if "error" in result:
            self._fail(sid, events, str(result["error"]))
            return

        # 提取完成后: 按 voice_events 切出逐条语音 wav 并回填 audio_url,
        # 供前端"找到并审听"每条语音(voice_audio.build_voice_audio 幂等)。
        # Post-extract: slice each voice message into a wav and attach audio_url.
        try:
            voice_audio.build_voice_audio(sid, str(video), result)
        except Exception as exc:  # noqa: BLE001 - 语音切片失败不阻断结果保存
            import logging
            logging.getLogger(__name__).warning("逐条语音切片失败: %s", exc)

        # 完成时进度写满, 避免"已done但进度停在中间"的观感 / mark 100% on done
        cur_progress = (store.load_meta(sid) or {}).get("progress") or {}
        meta_updates = {
            "status": web_config.STATUS_DONE,
            "stage": "done",
            "stage_detail": "提取完成",
            "progress": {
                "frame_id": cur_progress.get("frame_id"),
                "total_frames": cur_progress.get("total_frames"),
                "percent": 100.0,
                "elapsed_sec": cur_progress.get("elapsed_sec"),
            },
            "run_info": result.get("run_info"),
            "statistics": result.get("statistics"),
            "summary": result.get("summary"),
            "gaps": result.get("gaps"),
            "sections": result.get("sections"),
            "sections_count": len(result.get("sections", [])),
            "llm_enhanced": bool(result.get("llm_enhanced")),
            # 音频侧概览: 供会话状态/摘要展示, 完整数据经 /result 获取
            # Audio-side overview for status/summary; full data via /result
            "audio": result.get("audio"),
            "voice_events": result.get("voice_events"),
        }
        store.update_meta(sid, **meta_updates)
        events.append({"type": "status", "status": web_config.STATUS_DONE,
                       "stage": "done", "detail": "提取完成"})

    def _clean_output(self, out_dir: Path) -> None:
        """清空上次提取产物 / Clear previous extraction artifacts."""
        import shutil
        for name in ("cache", "frames"):
            p = out_dir / name
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        # 清理上次的逐条语音 (build_voice_audio 会在下次重新切)
        try:
            voice_audio.clean_voices(out_dir)
        except Exception:
            pass
        # 结果/报告按视频名输出, 用通配符整体清理
        # Results/reports are per-video; clean them via glob patterns.
        for pattern in ("behavior_log.jsonl",
                        "extracted_content_*.json",
                        "extracted_content_*.md",
                        "extracted_content_*.html"):
            for p in out_dir.glob(pattern):
                if p.is_file():
                    try:
                        p.unlink()
                    except OSError:
                        pass

    def _fail(self, sid: str, events: Deque[dict], msg: str) -> None:
        store.update_meta(sid, status=web_config.STATUS_ERROR,
                          stage="error", stage_detail=msg[:500],
                          error=msg[:500])
        events.append({"type": "status", "status": web_config.STATUS_ERROR,
                       "stage": "error", "detail": msg[:500]})

    # ------------------------------------------------------------------
    # 事件处理 / event handling (pipeline 多线程回调, 必须快且线程安全)
    # ------------------------------------------------------------------
    def _on_event(self, sid: str, events: Deque[dict], evt: dict) -> None:
        etype = evt.get("type")

        if etype == "stage":
            stage = evt.get("stage", "")
            store.update_meta(sid, stage=stage,
                              stage_detail=evt.get("detail", ""))
            events.append({"type": "stage", "stage": stage,
                           "detail": evt.get("detail", "")})
            return

        if etype == "frame":
            total = evt.get("total_frames") or 0
            fid = evt.get("frame_id") or 0
            now = time.time()
            with self._lock:
                task = self._tasks.get(sid)
                started_at = task.get("started_at", now) if task else now
                last = task["last_push"] if task else 0.0
                max_fid = 0
                if task:
                    # 进度用单调峰值: 流水线"重置关键帧"会回头重采已处理的帧,
                    # 直接取当前frame_id会导致百分比回退
                    # monotonic peak: keyframe-reset re-walks earlier frames,
                    # so use the max frame id seen so far to keep % monotonic
                    task["max_fid"] = max(task.get("max_fid", 0), fid)
                    max_fid = task["max_fid"]
            percent = round(max_fid / total * 100, 1) if total else 0.0
            # 事件全量入队(WS实时); 附上单调进度供前端直接使用
            # events always enqueued (realtime WS); monotonic percent included
            events.append({"type": "frame",
                           "frame_id": fid,
                           "max_frame_id": max_fid,
                           "total_frames": total,
                           "percent": percent,
                           "timestamp": evt.get("timestamp")})
            if now - last < 0.4 or task is None:
                return          # 未到节流间隔, 不写meta / within throttle window
            with self._lock:
                task["last_push"] = now
            store.update_meta(sid, progress={
                "frame_id": fid,
                "max_frame_id": max_fid,
                "total_frames": total,
                "percent": percent,
                "elapsed_sec": round(now - started_at, 1),
            })
            return

        if etype == "ocr_frame":
            events.append({"type": "ocr_frame",
                           "frame_id": evt.get("frame_id"),
                           "timestamp": evt.get("timestamp"),
                           "lines_count": len(evt.get("lines", []))})
            return

        if etype == "audio_progress":
            # 音频转写实时进度: 入队推送(Ws实时)并持久化到 meta(供轮询/快照兜底)。
            # 事件稀疏(逐语音段, 非逐帧), 无需节流。
            # Real-time audio-progress: enqueue for Ws + persist to meta for
            # poll/snapshot fallback. Events are sparse (per VAD segment),
            # so no throttling needed.
            ap = {
                "phase": evt.get("phase", ""),
                "done": int(evt.get("done") or 0),
                "total": int(evt.get("total") or 0),
                "detail": evt.get("detail", ""),
            }
            events.append({"type": "audio_progress", **ap})
            store.update_meta(sid, audio_progress=ap)
            return

        # 其他类型(progress 等)直接入队 / others pass through
        events.append(evt)


# 模块级单例 / module-level singleton
manager = ExtractionManager()
