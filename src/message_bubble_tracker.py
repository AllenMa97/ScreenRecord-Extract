"""
文本行跟踪器

核心创新: 用空间感知的逐行跟踪替代整帧 LCS 拼接。

问题背景:
  原方案将整帧OCR文本拼接为一个字符串再做Levenshtein比对, 存在:
  1. 单行OCR错误影响整体相似度
  2. LCS要求精确子序列匹配, OCR错一个字就断链
  3. 丢弃了bbox空间信息, 无法判断哪些行是新内容

解决方案:
  利用OCR返回的bbox坐标, 将每行文本作为独立实例进行跨帧跟踪:
  1. 匹配: bbox IoU + 中心距离 + 文本相似度组合判定
  2. 去重: 匹配成功的行视为已见内容, 不重复输出
  3. 拼接: 未匹配的行视为新内容, 按空间位置有序输出
  4. 纠错: 多帧投票 + 置信度加权纠正OCR错误

  此方法借鉴视频文本跟踪领域(Video Text Tracking)的思路,
  将文本实例在帧间的对应关系显式建模。

参考思路:
  - 使用appearance + geometry + text特征建立帧间对应关系
"""

import uuid
from typing import List, Optional

from src.bubble_merger import STAMP_RE
from src.data_types import TextLine, TrackedLine
from src.utils.text_utils import (
    bbox_iou, bbox_center_distance, text_similarity, normalize_text
)
from src.utils.behavior_logger import BehaviorLogger


class MessageBubbleTracker:
    """空间感知的文本行跟踪器"""

    def __init__(self, config: dict, logger: BehaviorLogger):
        self._iou_threshold: float = config.get("iou_threshold", 0.4)
        self._text_sim_threshold: float = config.get("text_sim_threshold", 0.80)
        self._expire_seconds: float = config.get("line_expire_seconds", 3.0)
        self._voting_window: int = config.get("voting_window", 5)
        self._center_dist_ratio: float = config.get("center_distance_ratio", 0.15)
        self._logger = logger

        self._active_lines: List[TrackedLine] = []
        # 已滚动出屏而过期的行: 不丢弃, 保留到最终组装。
        # 滚动出屏的行仍是视频中出现的真实内容, 若过期即删会导致召回损失。
        self._completed_lines: List[TrackedLine] = []
        self._roi_height: float = 1.0  # 由pipeline设置

    def set_roi_dimensions(self, roi_height: int, roi_width: int) -> None:
        """设置ROI尺寸(用于中心距离归一化)"""
        self._roi_height = max(float(roi_height), 1.0)

    def update(self, frame_id: int, timestamp: float,
               ocr_lines: List[TextLine],
               allow_new: bool = True) -> List[TextLine]:
        """
        用新帧的OCR结果更新跟踪器。

        参数:
            frame_id: 当前帧ID
            timestamp: 当前时间戳
            ocr_lines: 当前帧OCR识别的文本行列表
            allow_new: 是否允许未匹配行创建新跟踪行。滚动帧(内容残片/错字)置False,
                       只参与匹配投票, 不污染输出; 完整内容会由后续静止帧创建。
        返回:
            新发现的文本行(未匹配到已有跟踪行)
        """
        # 过滤无效行
        valid_lines = [
            line for line in ocr_lines
            if line.text.strip() and line.bbox[2] > 0 and line.bbox[3] > 0
        ]

        new_lines: List[TextLine] = []
        matched_count = 0

        for line in valid_lines:
            line.text = normalize_text(line.text)
            matched_line = self._find_match(line, frame_id)

            if matched_line is not None:
                # 匹配成功: 更新跟踪行(去重, 不输出)
                prev_text = matched_line.text   # 投票前文本(纠错效果对比用)
                matched_line.update_with(line)
                matched_line.is_new = False
                matched_count += 1
                self._log_vote(matched_line, line, prev_text, frame_id)
            elif allow_new:
                # 未匹配: 新内容(仅允许新建的帧)
                tracked = TrackedLine(
                    line_id=uuid.uuid4().hex[:12],
                    text=line.text,
                    bbox=line.bbox,
                    first_bbox=tuple(line.bbox),
                    confidence=line.confidence,
                    first_seen_frame=frame_id,
                    last_seen_frame=frame_id,
                    first_timestamp=timestamp,
                    last_timestamp=timestamp,
                    observations=[line],
                    is_new=True
                )
                self._active_lines.append(tracked)
                new_lines.append(line)
            # else: 滚动帧未匹配行忽略(残片不输出)

        # 过期清理
        self._expire_lines(timestamp)

        self._logger.log("text_tracker", "update",
                         f"帧{frame_id}: OCR={len(valid_lines)}行, "
                         f"匹配={matched_count}, 新增={len(new_lines)}, "
                         f"活跃跟踪={len(self._active_lines)}",
                         frame_id=frame_id,
                         extra={"ocr_lines": len(valid_lines),
                                "matched": matched_count,
                                "new": len(new_lines),
                                "active_tracked": len(self._active_lines)})

        return new_lines

    def _find_match(self, line: TextLine, frame_id: int) -> Optional[TrackedLine]:
        """
        在活跃跟踪行中查找匹配。

        匹配条件(满足任一空间条件 AND 文本相似度):
          - OR 中心距离 < center_dist_ratio * roi_height
          - AND 文本相似度 > text_sim_threshold

        时间戳行隔离(修复时间戳行帧内滚雪球吸收):
          聊天历史里每行消息上方的小时间戳(如"20:13")高度相似(差异仅分钟位)、
          且彼此垂直间距常小于宽松的中心距离阈值(0.3*ROI高度≈522px), 若不加
          隔离, 同一帧内十几个时间戳行会被第一条跟踪行依次吸收, 投票滚雪球
          后只剩一条, 消息与聊天时间的关联全错。隔离规则:
          - 时间戳行与正文行互不匹配(类型隔离), 正文行不得并入时间戳跟踪行;
          - 同帧内出现的时间戳行彼此独立: 同一帧同时显示的不同时间戳必然
            是不同界面元素(各自属于不同消息), 绝不合并; 仅跨帧出现的同一
            元素(如顶部时钟/滚动后复现的历史时间戳)才按空间+文本匹配。
        """
        line_is_stamp = bool(STAMP_RE.fullmatch((line.text or "").strip()))

        best_match: Optional[TrackedLine] = None
        best_score: float = 0.0

        for tracked in self._active_lines:
            tracked_is_stamp = bool(STAMP_RE.fullmatch((tracked.text or "").strip()))
            # 时间戳行与正文行互不匹配
            if line_is_stamp != tracked_is_stamp:
                continue
            # 同帧内的时间戳行各自独立, 不合并
            if line_is_stamp and tracked.last_seen_frame == frame_id:
                continue

            # 空间匹配
            iou = bbox_iou(line.bbox, tracked.bbox)
            center_dist = bbox_center_distance(line.bbox, tracked.bbox)
            center_threshold = self._center_dist_ratio * self._roi_height

            spatial_match = (iou > self._iou_threshold or
                             center_dist < center_threshold)

            if not spatial_match:
                continue

            # 文本匹配
            sim = text_similarity(line.text, tracked.text)
            if sim < self._text_sim_threshold:
                continue

            # 综合评分: 空间+文本
            score = iou * 0.4 + sim * 0.6
            if score > best_score:
                best_score = score
                best_match = tracked

        return best_match

    def _log_vote(self, tracked: TrackedLine, line: TextLine,
                  prev_text: str, frame_id: int) -> None:
        """
        时序一致性投票日志(纠错效果验证)。

        每条观测都写结构化行为日志(behavior_log.jsonl), 供离线核对;
        控制台仅打印纠错相关事件(投票结果变更 / 本次观测与投票结果不一致),
        便于实时验证多帧投票纠正OCR错误的效果。
        """
        # 票数统计
        counts: dict = {}
        for obs in tracked.observations:
            counts[obs.text] = counts.get(obs.text, 0) + 1
        changed = prev_text != tracked.text          # 投票结果被本次观测改变
        agrees = line.text == tracked.text           # 本次观测与当前投票一致
        conf = line.confidence

        # 结构化行为日志(全量)
        self._logger.log(
            "text_tracker", "vote",
            f"帧{frame_id} 跟踪行{tracked.line_id[:8]}: "
            f"观测'{line.text}' conf={conf:.2f} → 投票'{tracked.text}' "
            f"(变更前'{prev_text}') 观测数={len(tracked.observations)} "
            f"{'[结果变更]' if changed else ('[吸收噪声]' if not agrees else '')}",
            frame_id=frame_id,
            extra={
                "line_id": tracked.line_id,
                "obs_text": line.text,
                "obs_conf": round(float(conf), 3),
                "voted_text": tracked.text,
                "prev_text": prev_text,
                "changed": changed,
                "agrees": agrees,
                "obs_count": len(tracked.observations),
                "vote_counts": counts,
                "avg_conf": round(tracked.confidence, 3),
            })

        # 控制台: 仅纠错相关事件
        if changed:
            print(f"[投票] 帧{frame_id} 行{tracked.line_id[:8]}: "
                  f"结果变更 '{prev_text}' → '{tracked.text}' "
                  f"(票数{counts}, 观测{len(tracked.observations)}次, "
                  f"平均置信{tracked.confidence:.2f})", flush=True)
        elif not agrees:
            print(f"[投票] 帧{frame_id} 行{tracked.line_id[:8]}: "
                  f"吸收噪声观测 '{line.text}' (conf={conf:.2f}), "
                  f"投票保持 '{tracked.text}' 票数{counts}", flush=True)

    def _expire_lines(self, current_timestamp: float) -> None:
        """清理过期跟踪行, 过期行移入完成列表(不丢弃)。

        行滚动出屏后会过期, 但其内容仍是视频中出现的真实内容, 必须保留
        到最终组装, 否则会丢内容(召回率损失)。过期的行不再参与匹配。
        """
        kept = []
        expired = []
        for line in self._active_lines:
            if current_timestamp - line.last_timestamp <= self._expire_seconds:
                kept.append(line)
            else:
                expired.append(line)
        self._active_lines = kept
        self._completed_lines.extend(expired)
        if expired:
            self._logger.log("text_tracker", "expire",
                             f"过期转完成: {len(expired)}行, "
                             f"活跃{len(self._active_lines)}行, "
                             f"累计完成{len(self._completed_lines)}行",
                             extra={"expired": len(expired),
                                    "active": len(self._active_lines),
                                    "completed": len(self._completed_lines)})

    def get_all_tracked(self) -> List[TrackedLine]:
        """获取所有跟踪行(活跃+已完成, 按首次出现时间+首次位置排序)。

        排序用首次bbox而非最新bbox: 滚动会把首屏各行更新到不同位置, 用最新
        bbox会让同屏共享同一first_timestamp的行乱序, 破坏阅读顺序与气泡合并。
        """
        all_lines = self._active_lines + self._completed_lines
        return sorted(all_lines,
                      key=lambda x: (x.first_timestamp,
                                     (x.first_bbox or x.bbox)[1]))

    def get_new_lines(self) -> List[TrackedLine]:
        """获取标记为新内容的跟踪行"""
        return [line for line in self._active_lines if line.is_new]

    def compute_overlap_ratio(self, current_lines: List[TextLine],
                              previous_lines: List[TextLine]) -> float:
        """
        计算当前帧与上一帧的文本行重叠率。

        用于检测内容缺口: 重叠率低表示快速滑动导致内容断裂。
        """
        if not previous_lines:
            return 0.0 if current_lines else 1.0

        matched = 0
        for curr in current_lines:
            for prev in previous_lines:
                iou = bbox_iou(curr.bbox, prev.bbox)
                sim = text_similarity(curr.text, prev.text)
                if iou > self._iou_threshold or sim > 0.85:
                    matched += 1
                    break

        total = max(len(current_lines), len(previous_lines))
        return matched / total if total > 0 else 1.0
