"""
语音气泡扫描器 (Voice-Bubble Scanner)

对话记录屏做一次**独立的后处理第二遍扫描**: 用 cv2 重开视频, 按时间抽样帧,
对 ROI 区域跑轻量气泡检测, 归并出"每个语音气泡何时出现在屏幕上"的事件
(含方向 '发'/'收'), 作为把音频转写配对回聊天记录的时间锚点。

与 OCR producer 解耦: 不触碰既有采样/去重/OCR 流程, 低风险; 抽样(默认每
0.2s 一帧)成本低。
"""
from __future__ import annotations

import bisect
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .data_types import VoiceEvent
from src.visual.voice_bubble_detector import VoiceBubble, detect_voice_bubbles

logger = logging.getLogger(__name__)

ROI_TYPE = Tuple[int, int, int, int]  # (x, y, w, h), 原始帧坐标


class VoiceBubbleScanner:
    """对视频做抽样扫描, 产生语音气泡"出现事件"时间线。"""

    def __init__(self, roi_box: ROI_TYPE, voice_threshold: float = 0.55,
                 sample_interval_s: float = 0.2, min_bubble_ms: int = 0,
                 expire_s: float = 2.0, merge_window_s: float = 1.2):
        self._roi_box = roi_box
        self._voice_threshold = voice_threshold
        self._interval = sample_interval_s
        self._min_bubble_ms = min_bubble_ms
        self._expire_s = expire_s
        self._merge_window_s = merge_window_s
        self._active: List[VoiceBubbleState] = []

    def scan(self, video_path: str, progress=None) -> List[VoiceEvent]:
        """扫描视频, 返回语音气泡出现事件(list[VoiceEvent], 按时间排序)。

        同一气泡持续多帧: 只在其"首次出现"或"方向变化"时发射事件,
        避免一帧一个事件刷屏。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("无法打开视频做气泡扫描: %s", video_path)
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_skip = max(1, int(round(self._interval * fps)))
        events: List[VoiceEvent] = []
        self._active = []
        x, y, w, h = self._roi_box
        if w <= 0 or h <= 0:
            cap.release()
            return []

        fid = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fid % frame_skip != 0:
                fid += 1
                continue
            ts = fid / fps
            roi_bgr = frame[y:y + h, x:x + w]
            bubbles = detect_voice_bubbles(roi_bgr)
            # 投影回原始帧坐标, 过滤低 voice_score
            cands = []
            for b in bubbles:
                if b.voice_score < self._voice_threshold:
                    continue
                bx = VoiceBubble(x + b.x, y + b.y, b.w, b.h, b.side,
                                 b.confidence, b.voice_score)
                cands.append(bx)
            self._update_active(cands, ts, events, progress)
            fid += 1
            if total and fid > total:
                break
        cap.release()
        events.sort(key=lambda e: e.timestamp)
        return self._dedup_events(events, merge_window_s=self._merge_window_s)

    def _dedup_events(self, events: List[VoiceEvent],
                      merge_window_s: float = 1.2,
                      size_tol: float = 0.25,
                      cx_tol: float = 0.15) -> List[VoiceEvent]:
        """合并"同一物理语音气泡被重复发射"的事件簇。

        现象: 气泡内波形随播放/轻微滚动而略有变化, 导致相邻采样帧(默认0.2s)
        的IoU匹配偶发失败 → 同一气泡被当作新气泡重复发射, 时间戳间隔≈采样周期。
        判定: 时间窗内同侧别、且**尺寸签名一致**的事件才视为同一个气泡。

        为何用"尺寸+中心x"而非位置IoU: 气泡在滚动时 y 坐标随之整体平移,
        但同一气泡的宽/高/中心x 保持稳定。若按IoU合并, 滚动一下 y 错位,
        IoU 骤降 → 被误判成新气泡(过度分裂); 若只按"同侧+时间"(旧逻辑),
        会把同侧紧邻的两条不同语音误并(召回丢失)。尺寸签名兼顾两者。
        """
        COLOR_W, COLOR_H = 890, 1920   # ROI原始坐标下整屏尺寸(用于过滤状态栏条)
        real = [e for e in events
                if not (e.box and e.box[2] >= COLOR_W * 0.75
                        and e.box[3] < COLOR_H * 0.25)]  # 剔除整宽状态栏/输入条
        kept: List[VoiceEvent] = []
        for ev in real:
            merged = False
            for k in kept:
                if k.side != ev.side:
                    continue
                if abs(k.timestamp - ev.timestamp) > merge_window_s:
                    continue
                if not _same_size_key(k, ev, size_tol, cx_tol):
                    continue
                # 同一气泡(时间近 + 尺寸签名同): 仅延长消亡时间, 不新增事件
                if ev.end_time is not None:
                    k.end_time = max(k.end_time or k.timestamp, ev.end_time)
                # 首次发射保留; 定价 confidence 取更高者
                k.confidence = max(k.confidence, ev.confidence)
                if not k.matched_text:
                    k.matched_text = ev.matched_text
                merged = True
                break
            if not merged:
                kept.append(ev)
        kept.sort(key=lambda e: e.timestamp)
        return kept

    # -- 内部状态机: 把"每帧检测到的气泡集合"变成"事件流" --

    def _update_active(self, cands, ts, events, progress):
        # 先尝试与已活跃气泡匹配(IoU), 更新其观察
        unmatched = list(cands)
        for st in self._active:
            best = None
            best_iou = 0.0
            for i, c in enumerate(unmatched):
                iou = _iou(st.last_box, (c.x, c.y, c.w, c.h))
                if iou > best_iou:
                    best_iou = iou
                    best = i
            if best is not None and best_iou >= 0.35:
                c = unmatched.pop(best)
                st.update(c, ts)
        # 新出现的活跃气泡 → 发事件
        for c in unmatched:
            st = VoiceBubbleState(c)
            ev = VoiceEvent(timestamp=ts, end_time=ts, side=c.side,
                            confidence=c.confidence, frame_id=int(ts * 30),
                            box=(c.x, c.y, c.w, c.h))
            st.event_ref = ev
            self._active.append(st)
            events.append(ev)
        # L2: 剔除超时未再命中的气泡状态(防止长视频下活跃列表无限增长导致
        # 旧气泡与新气泡高IoU误匹配、事件重复; 并在消亡时补记 end_time)
        expired = [st for st in self._active
                   if st.last_ts is not None and ts - st.last_ts > self._expire_s]
        for st in expired:
            for ev in events:
                if ev is st.event_ref:
                    ev.end_time = st.last_ts
        self._active = [st for st in self._active if st not in expired]
        if progress:
            try:
                progress(fid=len(cands), ts=ts)
            except Exception:
                pass


class VoiceBubbleState:
    """单个活跃语音气泡的多帧观察状态。"""

    def __init__(self, bubble: VoiceBubble):
        self.side = bubble.side
        self.last_box = (bubble.x, bubble.y, bubble.w, bubble.h)
        self.confidence = bubble.confidence
        self.first_ts = self.last_ts = None
        self.seen_frames = 0
        self.event_ref = None   # 关联的 VoiceEvent, 用于消亡时补记 end_time

    def update(self, bubble: VoiceBubble, ts: float):
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts
        self.last_box = (bubble.x, bubble.y, bubble.w, bubble.h)
        self.side = bubble.side or self.side
        self.seen_frames += 1


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def merge_ocr_and_vision(vision_events: List[VoiceEvent],
                         ocr_hits,
                         match_window_s: float = 0.8,
                         box_tol: float = 0.45) -> List[VoiceEvent]:
    """以视觉气泡清单为主, 用 OCR 时长标签在**气泡框内**做双源佐证。

    视觉形状检测给出权威且稳定的气泡清单(已与肉眼核对一致)。OCR 时长标签仅当
    在时间窗内且其框与视觉气泡框空间重叠足够时, 才把该事件标记为 source='hybrid'
    (双源确认), 否则保持 source='vision'。**绝不新增独立事件**, 杜绝全局 mm:ss
    把界面时钟/进度误判成语音气泡导致的计数爆炸。

    ocr_hits: OcrVoiceScanner.scan_duration_hits 的产物(list[DurationHit])。
    box_tol: 空间重叠判定用 IoU 阈值; 标签框与气泡框部分相交即可(标签在气泡内)。
    """
    merged: List[VoiceEvent] = []
    if not ocr_hits:
        merged = [e for e in vision_events]
        merged.sort(key=lambda e: e.timestamp)
        return merged
    for ev in vision_events:
        for h in ocr_hits:
            if abs(h.timestamp - ev.timestamp) > match_window_s:
                continue
            if ev.side and h.side != ev.side:
                continue
            if not _bubble_hit_overlap(ev.box, h.box, box_tol):
                continue
            ev.source = "hybrid"
            break
        merged.append(ev)
    merged.sort(key=lambda e: e.timestamp)
    return merged


def _bubble_hit_overlap(bubble_box, hit_box, box_tol: float) -> bool:
    """视觉气泡框与 OCR 时长标签框是否空间上足够重合。

    标签(如 00:03)通常位于气泡内部/紧邻, 二者 IoU 或包含度达阈值即视为佐证。
    无框数据时退化为不设空间约束(仅时间窗)。
    """
    if not hit_box:
        return True
    if not bubble_box:
        return True
    ax, ay, aw, ah = bubble_box
    bbx, bby, bbw, bbh = hit_box
    # 交叠面积 / min(气泡框, 标签框) 面积 -> 标签需显著落在气泡内
    ix = max(0, min(ax + aw, bbx + bbw) - max(ax, bbx))
    iy = max(0, min(ay + ah, bby + bbh) - max(ay, bby))
    inter = ix * iy
    if inter <= 0:
        return False
    denom = min(aw * ah, bbw * bbh)
    return inter / denom >= box_tol


def _any_within(ts_sorted: List[float], t: float, window: float) -> bool:
    i = bisect.bisect_left(ts_sorted, t)
    if i < len(ts_sorted) and ts_sorted[i] - t <= window:
        return True
    if i > 0 and t - ts_sorted[i - 1] <= window:
        return True
    return False


def _same_size_key(a: VoiceEvent, b: VoiceEvent,
                   size_tol: float = 0.25, cx_tol: float = 0.15) -> bool:
    """两事件是否属同一物理气泡: 滚动只动 y, 宽/高/中心x 稳定。
    任一缺失框则退化按时间合并(兼容旧行为)。"""
    if not (a.box and b.box):
        return True
    ax, ay, aw, ah = a.box
    bx, by, bw, bh = b.box
    # 宽/高相对差在容差内
    if abs(aw - bw) / max(aw, bw, 1) > size_tol:
        return False
    if abs(ah - bh) / max(ah, bh, 1) > size_tol:
        return False
    # 中心x 相差不超过宽度比例(同侧气泡贴边一致)
    acx, bcx = ax + aw / 2, bx + bw / 2
    if abs(acx - bcx) / max(aw, bw, 1) > cx_tol:
        return False
    return True