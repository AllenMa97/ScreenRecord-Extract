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

# 时长标签有两种真实写法, 必须都认 / two real-world forms, both must be parsed:
#
#   ① 带冒号的 mm:ss —— 部分机型/端上把时长写成 "00:03"。
#   ② 不带冒号的 N"  ——**实测主流微信渲染就是这种**: 数字后紧跟一个引号, 无冒号
#      (见 tools/render_voice_crops.py 的裁图 OCR 结果: `4"` / `5"` / `6"…` / `8"`)。
#      引号还可能被 OCR 判读成 ” / ’ / ′ / ″ / '' , 末尾常带省略号。
#      注意**不把 °(度数符号)算进来**: 它更常见于"今天37°"这类正常文本, 收益
#      (少数裁图把引号认成 °)远小于误报代价, 故只收引号族字形。
#
# 旧实现只认 ①, 在真实录屏上召回为 0(被测录屏 20 条语音的标签全部走 ②), 使
# voice_extractor 退化成按"气泡停留时长"切音轨 —— 这正是 GT
# `known_failure_modes_to_watch` 首条预警的情形。
# 省略号可能出现在**前面**: 气泡被屏幕边缘裁切/OCR 只吃到半截时, 标签会读成
# `…4"`(实测 自行模拟对话记录4.mp4), 漏掉前缀会让它当正文行混进聊天面板。
# A leading ellipsis occurs when the bubble is clipped at the frame edge.
_ELLIPSIS = r'(?:…+|\.{2,3}|\u2026)?\s*'
_DURATION_RE_COLON = re.compile(
    r'^' + _ELLIPSIS + r'(\d{1,2})\s*[:：]\s*(\d{1,2})' + _ELLIPSIS + r'$')
_DURATION_RE_QUOTE = re.compile(
    r'^' + _ELLIPSIS + r'(\d{1,2})\s*["\'”’′″]{1,2}\s*' + _ELLIPSIS + r'$')

# 语音消息时长上限(秒): 微信单条语音最长 60s。Voice-message ceiling.
MAX_VOICE_SECONDS = 60.0


def parse_duration_seconds(text: str) -> Optional[float]:
    """把时长标签文本解析成秒数; 非时长标签或超出语音上限返回 None。

    领域约束: 冒号写法秒位必须 ∈ [1,59](排除 "00:00"/"18:75"), 两种写法总时长
    都必须 ≤ 60s —— 超 60s 的 mm:ss 必然是界面时钟/视频进度, 用该约束直接剔除
    (实测 907 个 mm:ss 命中里有 468 个 >60s)。
    """
    s = (text or "").strip()
    m = _DURATION_RE_COLON.match(s)
    if m:
        minutes, seconds = int(m.group(1)), int(m.group(2))
        if seconds < 1 or seconds >= 60:
            return None
        total = float(minutes * 60 + seconds)
    else:
        m = _DURATION_RE_QUOTE.match(s)
        if not m:
            return None
        # 引号写法里的数字**本身就是秒**(不是分钟), 不折算。
        total = float(int(m.group(1)))
    if total <= 0 or total > MAX_VOICE_SECONDS:
        return None
    return total


def _is_duration(text: str) -> bool:
    """文本是否为**可能的语音时长标签**(已套用 ≤60s 领域约束)。"""
    return parse_duration_seconds(text) is not None


@dataclass
class DurationHit:
    """一次在 OCR 缓存中捕获到的时长标签观测。Single observed duration label."""
    timestamp: float        # 出现时刻(秒) / video time (s)
    text: str               # 时长文本如 "00:03" / duration text
    box: Tuple[float, float, float, float]  # (x, y, w, h) **原始帧坐标** / frame coords
    confidence: float       # OCR 置信度 / OCR confidence
    side: str               # '发' / '收' by horizontal position


@dataclass
class TextHit:
    """一次在 OCR 缓存中捕获到的**普通文本行**观测(非时长标签)。

    Purpose / 用途: 语音气泡内部只有波形(以及可选的时长标签), **没有正文文字**;
    反过来, 一个候选框里若被 OCR 认出了成句的正文, 它就不是语音气泡。故把普通
    文本行(排除时长标签)也导出为原始观测, 供上层做"框内含文本"的质量门。

    box 为**原始帧坐标**(调用方传 roi_offset 后已换算), 与视觉气泡框同坐标系。
    """
    timestamp: float
    text: str
    box: Tuple[float, float, float, float]  # (x, y, w, h) 原始帧坐标 / frame coords
    confidence: float
    side: str


class OcrVoiceScanner:
    """从 OCR 缓存输出原始时长标签观测, 供视觉气泡佐证使用。"""

    def __init__(self, frame_width: int = 890, min_confidence: float = 0.5):
        self._width = frame_width
        self._min_conf = min_confidence

    def _iter_lines(self, cache_path: str, roi_offset: Tuple[int, int]):
        """逐条产出 OCR 缓存里的文本行观测(dict), 不做时长/正文语义过滤。

        统一在这里做"坐标系换算 + 侧别判定", 避免各调用方各写一遍而口径漂移。
        OCR 缓存里的 bbox 是 **ROI 局部坐标**(见 data_types 契约), 而视觉气泡框是
        **原始帧坐标**; 必须加 ROI 左上角偏移统一坐标系, 否则 y 方向固定错开
        ROI.y(常见 60px)而文本行高仅 30~50px, 重叠恒为 0 → 框内判定几乎全失效。

        偏移**按帧取自缓存记录里的 `roi` 字段**: 自适应 ROI 会在预热结束时挪动原点
        (实测 (0,60) → (0,302)), 同一个缓存文件里混着两套局部坐标系。只用调用方给的
        终版偏移会把预热期帧整体平移 242px, 正文质量门随之全数失效(实测
        `自行模拟对话记录4.mp4`: 10 个候选气泡一个都没被剔除)。`roi_offset` 因此退化
        为**旧缓存(无 roi 字段)的回退值**。
        """
        fb_x, fb_y = int(roi_offset[0]), int(roi_offset[1])
        rev = self._width / 2.0   # 左右判定在 ROI 局部空间做(与框原始坐标系无关)
        for line in open(cache_path, "r", encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts = float(d.get("ts", 0.0))
            roi = d.get("roi")
            if roi and len(roi) >= 2:
                try:
                    ox, oy = int(roi[0]), int(roi[1])
                except (TypeError, ValueError):
                    ox, oy = fb_x, fb_y
            else:
                ox, oy = fb_x, fb_y
            for l in d.get("lines", []):
                cb = l.get("bbox")
                if not cb or len(cb) != 4:
                    continue
                conf = float(l.get("confidence", 1.0) or 1.0)
                if conf < self._min_conf:
                    continue
                bx0, by0, bx1, by1 = cb
                side = "收" if (bx0 + bx1) / 2.0 < rev else "发"
                # ROI 局部坐标 -> 原始帧坐标(与视觉气泡框同坐标系)
                bx0, by0, bx1, by1 = bx0 + ox, by0 + oy, bx1 + ox, by1 + oy
                yield {
                    "timestamp": ts,
                    "text": str(l.get("text", "")).strip(),
                    "box": (float(bx0), float(by0),
                            float(bx1 - bx0), float(by1 - by0)),
                    "confidence": conf,
                    "side": side,
                }

    def scan_duration_hits(self, cache_path: str,
                           roi_offset: Tuple[int, int] = (0, 0)) -> List[DurationHit]:
        """读取 OCR 缓存(NDJSON: {fid, ts, lines:[{text,bbox,confidence}]}),
        返回所有被识别为时长标签的原始观测列表(不做任何聚合/去重), 按时间排序。

        是否作为气泡佐证由上层用**视觉气泡框**判断, 此处不引入计数语义。
        """
        if not cache_path or not os.path.exists(cache_path):
            logger.warning("OCR 缓存不存在: %s", cache_path)
            return []
        hits = [DurationHit(**r) for r in self._iter_lines(cache_path, roi_offset)
                if _is_duration(r["text"])]
        hits.sort(key=lambda h: h.timestamp)
        return hits

    def scan_text_hits(self, cache_path: str,
                       roi_offset: Tuple[int, int] = (0, 0)) -> List[TextHit]:
        """读取 OCR 缓存, 返回**普通文本行**(已排除时长标签)的原始观测, 按时间排序。

        供"候选气泡框内含正文 → 判为文本气泡"的质量门使用(见
        `voice_bubble_scanner.drop_text_bubbles`)。时长标签必须排除, 因为它本来
        就长在语音气泡里, 保留会把真语音气泡误杀。
        """
        if not cache_path or not os.path.exists(cache_path):
            logger.warning("OCR 缓存不存在: %s", cache_path)
            return []
        hits = [TextHit(**r) for r in self._iter_lines(cache_path, roi_offset)
                if r["text"] and not _is_duration(r["text"])]
        hits.sort(key=lambda h: h.timestamp)
        return hits