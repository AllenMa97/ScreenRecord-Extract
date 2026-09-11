"""
OCR 语音气泡扫描器 (OCR Voice-Bubble Scanner)

视觉形状检测已能给出准确的语音气泡清单(在真实测试屏上数为 25 条, 与肉眼一致)。
本扫描器不再独立计数: 而是基于流水线已落盘的 **OCR 缓存** 输出每个可能"时长标签"
的**原始观测**时间与位置, 供上层把视觉气泡**框内**的时长作为佐证(双源确认), 不新增
独立事件——避免全局 `mm:ss` 正则把界面时钟/进度数字误判为语音气泡导致的计数爆炸
(曾在含实时时钟的录屏上错误写出 88 条, 而真实语音仅 25 条)。

CD-ROM: emit raw duration-label observations (ts/text/box), not aggregated events.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# 时长标签: 1~2 位分钟 + 秒, 分钟 ≤ 2(语音上限), 秒 ≥ 1。
# 排除整份时钟(如 "18:35", 分钟远超 2)。Voice duration like "00:03"/"1:05".
_DURATION_RE = re.compile(r'^(\d{1,2}):(\d{1,2})$')


def _is_duration(text: str) -> bool:
    m = _DURATION_RE.match((text or "").strip())
    if not m:
        return False
    return int(m.group(1)) <= 2 and int(m.group(2)) >= 1


@dataclass
class DurationHit:
    """一次在 OCR 缓存中捕获到的时长标签观测。Single observed duration label."""
    timestamp: float        # 出现时刻(秒) / video time (s)
    text: str               # 时长文本如 "00:03" / duration text
    box: Tuple[float, float, float, float]  # (x, y, w, h) 原始帧坐标
    confidence: float       # OCR 置信度 / OCR confidence
    side: str               # '发' / '收' by horizontal position


class OcrVoiceScanner:
    """从 OCR 缓存输出原始时长标签观测, 供视觉气泡佐证使用。"""

    def __init__(self, frame_width: int = 890, min_confidence: float = 0.5):
        self._width = frame_width
        self._min_conf = min_confidence

    def scan_duration_hits(self, cache_path: str) -> List[DurationHit]:
        """读取 OCR 缓存(NDJSON: {fid, ts, lines:[{text,bbox,confidence}]}),
        返回所有被识别为时长标签的原始观测列表(不做任何聚合/去重), 按时间排序。

        是否作为气泡佐证由上层用**视觉气泡框**判断, 此处不引入计数语义。
        """
        if not cache_path or not os.path.exists(cache_path):
            logger.warning("OCR 缓存不存在: %s", cache_path)
            return []
        hits: List[DurationHit] = []
        rev = self._width / 2.0
        for line in open(cache_path, "r", encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts = float(d.get("ts", 0.0))
            for l in d.get("lines", []):
                if not _is_duration(l.get("text", "")):
                    continue
                cb = l.get("bbox")
                if not cb or len(cb) != 4:
                    continue
                conf = float(l.get("confidence", 1.0) or 1.0)
                if conf < self._min_conf:
                    continue
                bx0, by0, bx1, by1 = cb
                side = "收" if (bx0 + bx1) / 2.0 < rev else "发"
                hits.append(DurationHit(
                    timestamp=ts,
                    text=str(l.get("text", "")).strip(),
                    box=(float(bx0), float(by0),
                         float(bx1 - bx0), float(by1 - by0)),
                    confidence=conf,
                    side=side,
                ))
        hits.sort(key=lambda h: h.timestamp)
        return hits