"""
时序状态机

核心思路: 用OCR输出统计特征驱动状态机, 无需训练分类器。

设计动机:
  原方案建议训练UI页面分类器, 但实际没有标注数据。
  替代方案: 直接利用OCR返回的文本行数量、文本密度等统计特征
  判断当前是否处于目标页面(聊天记录/参数页)。

状态转换:
  NORMAL → SUSPENDED: 连续N帧OCR文本行数 < min_text_lines
                       或文本密度下降到基线的density_ratio_threshold以下
  SUSPENDED → RESUMING: OCR文本行数恢复到 >= min_text_lines
  RESUMING → NORMAL: 内容重叠率 >= overlap_threshold (回到同一页面)
                     或新内容(不同页面但仍是有效文本页)
  RESUMING → SUSPENDED: 仍然文本行数不足

内容缺口检测:
  在NORMAL状态下, 如果相邻两帧的文本行重叠率 < gap_overlap_threshold,
  标记为内容缺口(快速滑动导致中间内容丢失)。
"""

from typing import List, Optional

from src.data_types import TextLine, SystemState, StateAction
from src.utils.text_utils import bbox_iou, text_similarity
from src.utils.behavior_logger import BehaviorLogger


class StateMachine:
    """OCR驱动的时序状态机"""

    def __init__(self, config: dict, logger: BehaviorLogger):
        self._min_text_lines: int = config.get("min_text_lines", 3)
        self._suspend_frames: int = config.get("suspend_confirm_frames", 3)
        self._density_ratio: float = config.get("density_ratio_threshold", 0.3)
        self._overlap_threshold: float = config.get("overlap_threshold", 0.25)
        self._gap_overlap_threshold: float = config.get("gap_overlap_threshold", 0.15)
        self._logger = logger

        self._state: SystemState = SystemState.NORMAL
        self._abnormal_count: int = 0
        self._baseline_line_count: float = 0.0  # 正常状态文本行数基线
        self._baseline_density: float = 0.0     # 正常状态文本密度基线
        self._baseline_frame_count: int = 0     # 基线更新帧数

        self._last_normal_lines: Optional[List[TextLine]] = None  # 上一正常帧的文本行
        self._pre_suspend_lines: Optional[List[TextLine]] = None  # 挂起前最后的文本行
        self._last_frame_id: int = 0

    def update(self, frame_id: int, timestamp: float,
               ocr_lines: List[TextLine],
               overlap_ratio: float = 0.0) -> StateAction:
        """
        更新状态机。

        参数:
            frame_id: 帧ID
            timestamp: 时间戳
            ocr_lines: 当前帧OCR文本行
            overlap_ratio: 与上一正常帧的文本重叠率
        返回:
            StateAction: 状态机输出动作
        """
        self._last_frame_id = frame_id
        line_count = len(ocr_lines)
        text_density = self._compute_density(ocr_lines)

        action = StateAction(state=self._state, overlap_ratio=overlap_ratio)

        if self._state == SystemState.NORMAL:
            action = self._handle_normal(frame_id, timestamp, ocr_lines,
                                         line_count, text_density, overlap_ratio)
        elif self._state == SystemState.SUSPENDED:
            action = self._handle_suspended(frame_id, timestamp, ocr_lines,
                                            line_count, text_density)
        elif self._state == SystemState.RESUMING:
            action = self._handle_resuming(frame_id, timestamp, ocr_lines,
                                           line_count, text_density)

        return action

    def _handle_normal(self, frame_id: int, timestamp: float,
                       ocr_lines: List[TextLine], line_count: int,
                       text_density: float, overlap_ratio: float) -> StateAction:
        """处理NORMAL状态"""
        is_abnormal = self._is_abnormal(line_count, text_density)

        if is_abnormal:
            self._abnormal_count += 1
            self._logger.log("state_machine", "abnormal_detected",
                             f"异常帧: lines={line_count}, density={text_density:.4f}, "
                             f"consecutive={self._abnormal_count}/{self._suspend_frames}",
                             frame_id=frame_id,
                             extra={"line_count": line_count,
                                    "density": round(text_density, 4),
                                    "consecutive": self._abnormal_count})

            if self._abnormal_count >= self._suspend_frames:
                # 进入挂起状态
                self._state = SystemState.SUSPENDED
                self._pre_suspend_lines = self._last_normal_lines
                self._logger.log("state_machine", "suspend",
                                 f"状态转换: NORMAL → SUSPENDED (连续{self._abnormal_count}帧异常)",
                                 frame_id=frame_id, extra={"new_state": "suspended"})
                return StateAction(
                    state=SystemState.SUSPENDED,
                    should_process=False,
                    is_state_change=True
                )
            return StateAction(
                state=SystemState.NORMAL,
                should_process=False  # 异常帧不处理
            )
        else:
            # 正常帧
            self._abnormal_count = 0
            self._update_baseline(line_count, text_density)

            # 内容缺口检测
            is_gap = False
            gap_detail = ""
            if self._last_normal_lines is not None and len(self._last_normal_lines) > 0:
                if overlap_ratio < self._gap_overlap_threshold:
                    is_gap = True
                    gap_detail = (f"内容缺口: 重叠率={overlap_ratio:.2f} < "
                                  f"{self._gap_overlap_threshold}")
                    self._logger.log("state_machine", "gap_detected",
                                     gap_detail, frame_id=frame_id,
                                     extra={"overlap_ratio": round(overlap_ratio, 3)})

            self._last_normal_lines = list(ocr_lines)

            return StateAction(
                state=SystemState.NORMAL,
                should_process=True,
                is_gap=is_gap,
                gap_detail=gap_detail,
                overlap_ratio=overlap_ratio
            )

    def _handle_suspended(self, frame_id: int, timestamp: float,
                          ocr_lines: List[TextLine], line_count: int,
                          text_density: float) -> StateAction:
        """处理SUSPENDED状态"""
        is_abnormal = self._is_abnormal(line_count, text_density)

        if not is_abnormal:
            # 文本恢复, 进入恢复状态
            self._state = SystemState.RESUMING
            self._abnormal_count = 0
            self._logger.log("state_machine", "resume_attempt",
                             f"状态转换: SUSPENDED → RESUMING (lines={line_count})",
                             frame_id=frame_id, extra={"new_state": "resuming"})
            return self._handle_resuming(frame_id, timestamp, ocr_lines,
                                         line_count, text_density)
        else:
            self._logger.log("state_machine", "still_suspended",
                             f"仍挂起: lines={line_count}, density={text_density:.4f}",
                             frame_id=frame_id)
            return StateAction(
                state=SystemState.SUSPENDED,
                should_process=False
            )

    def _handle_resuming(self, frame_id: int, timestamp: float,
                         ocr_lines: List[TextLine], line_count: int,
                         text_density: float) -> StateAction:
        """处理RESUMING状态"""
        is_abnormal = self._is_abnormal(line_count, text_density)

        if is_abnormal:
            # 再次异常, 回到挂起
            self._state = SystemState.SUSPENDED
            self._logger.log("state_machine", "resume_failed",
                             "状态转换: RESUMING → SUSPENDED (恢复失败)",
                             frame_id=frame_id, extra={"new_state": "suspended"})
            return StateAction(
                state=SystemState.SUSPENDED,
                should_process=False,
                is_state_change=True
            )

        # 检查内容重叠
        overlap = 0.0
        if self._pre_suspend_lines is not None and len(self._pre_suspend_lines) > 0:
            overlap = self._compute_content_overlap(ocr_lines, self._pre_suspend_lines)
            if overlap >= self._overlap_threshold:
                # 成功恢复到同一页面
                self._state = SystemState.NORMAL
                self._abnormal_count = 0
                self._last_normal_lines = list(ocr_lines)
                self._pre_suspend_lines = None
                self._update_baseline(line_count, text_density)
                self._logger.log("state_machine", "resume_success",
                                 f"状态转换: RESUMING → NORMAL (重叠率={overlap:.2f})",
                                 frame_id=frame_id, extra={"new_state": "normal",
                                                           "overlap": round(overlap, 3)})
                return StateAction(
                    state=SystemState.NORMAL,
                    should_process=True,
                    is_state_change=True,
                    overlap_ratio=overlap
                )
            else:
                # 不同页面但仍有文本 → 可能是新的聊天/参数页
                self._state = SystemState.NORMAL
                self._abnormal_count = 0
                self._last_normal_lines = list(ocr_lines)
                self._pre_suspend_lines = None
                self._update_baseline(line_count, text_density)
                self._logger.log("state_machine", "resume_new_page",
                                 f"恢复到新页面(重叠率={overlap:.2f}), 标记缺口",
                                 frame_id=frame_id, extra={"new_state": "normal",
                                                           "overlap": round(overlap, 3),
                                                           "is_gap": True})
                return StateAction(
                    state=SystemState.NORMAL,
                    should_process=True,
                    is_state_change=True,
                    is_gap=True,
                    gap_detail=f"恢复到不同页面, 重叠率={overlap:.2f}",
                    overlap_ratio=overlap
                )
        else:
            # 无挂起前数据, 直接恢复
            self._state = SystemState.NORMAL
            self._abnormal_count = 0
            self._last_normal_lines = list(ocr_lines)
            self._pre_suspend_lines = None
            self._update_baseline(line_count, text_density)
            self._logger.log("state_machine", "resume_no_data",
                             "无挂起前数据, 直接恢复 / No pre-suspend data, direct resume",
                             frame_id=frame_id, extra={"new_state": "normal"})
            return StateAction(
                state=SystemState.NORMAL,
                should_process=True,
                is_state_change=True
            )

    def _is_abnormal(self, line_count: int, text_density: float) -> bool:
        """
        判断当前帧是否异常(不在目标页面)。

        异常条件(满足任一):
          - 文本行数 < min_text_lines
          - 文本密度 < 基线 * density_ratio (当基线已建立时)
        """
        if line_count < self._min_text_lines:
            return True
        if self._baseline_density > 0 and text_density < self._baseline_density * self._density_ratio:
            return True
        return False

    def _compute_density(self, ocr_lines: List[TextLine]) -> float:
        """计算文本密度(总字符数 / 总bbox面积)"""
        if not ocr_lines:
            return 0.0
        total_chars = sum(len(line.text) for line in ocr_lines)
        total_area = sum(line.bbox[2] * line.bbox[3] for line in ocr_lines)
        if total_area <= 0:
            return 0.0
        return total_chars / total_area * 1000  # 缩放因子方便比较

    def _update_baseline(self, line_count: int, text_density: float) -> None:
        """更新正常状态基线(指数移动平均)"""
        alpha = 0.1
        if self._baseline_frame_count == 0:
            self._baseline_line_count = float(line_count)
            self._baseline_density = text_density
        else:
            self._baseline_line_count = (1 - alpha) * self._baseline_line_count + alpha * line_count
            self._baseline_density = (1 - alpha) * self._baseline_density + alpha * text_density
        self._baseline_frame_count += 1

    def _compute_content_overlap(self, current: List[TextLine],
                                 previous: List[TextLine]) -> float:
        """计算两帧文本行的内容重叠率"""
        if not previous:
            return 0.0
        matched = 0
        for curr in current:
            for prev in previous:
                iou = bbox_iou(curr.bbox, prev.bbox)
                sim = text_similarity(curr.text, prev.text)
                if iou > 0.3 or sim > 0.80:
                    matched += 1
                    break
        return matched / max(len(current), len(previous))

    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def baseline_line_count(self) -> float:
        return self._baseline_line_count

    @property
    def baseline_density(self) -> float:
        return self._baseline_density
