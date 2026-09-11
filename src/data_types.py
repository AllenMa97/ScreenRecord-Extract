"""
数据类型定义

定义整个流水线中各模块间传递的核心数据结构。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


@dataclass
class TextLine:
    """
    OCR 识别的单行文本。

    bbox 格式: (x, y, w, h), 相对于 ROI 裁剪图的坐标。
    ROI (Region of Interest) = 感兴趣区域: 剔除状态栏/导航栏后真正有内容的屏幕区域。
    """
    text: str                              # 识别文本
    bbox: Tuple[int, int, int, int]        # 边界框
    confidence: float = 1.0                # 置信度
    frame_id: int = 0                      # 来源帧 ID
    timestamp: float = 0.0                 # 时间戳


@dataclass
class FrameOcrResult:
    """一帧的 OCR 结果"""
    frame_id: int
    timestamp: float
    lines: List[TextLine] = field(default_factory=list)
    text_density: float = 0.0              # 文本密度(字符数/ROI 面积)


@dataclass
class TrackedLine:
    """
    被跟踪的文本行。

    跟踪同一文本行在连续帧中的多次 OCR 观测, 用于去重和投票纠错。
    """
    line_id: str                           # 唯一标识
    text: str                              # 当前最佳文本(投票结果)
    bbox: Tuple[int, int, int, int]        # 最新 bbox
    first_bbox: Optional[Tuple[int, int, int, int]] = None
    # 首次出现时的bbox。滚动会更新bbox, 而首屏截屏(视频开头完整聊天历史)的
    # 各行共享同一first_timestamp, 若用"最新bbox"排序/合并, 行被滚动后的位置
    # 会打乱阅读顺序、破坏气泡几何关系。首次bbox才是这些行同屏时的真实位置,
    # 组装阶段的排序与气泡合并必须以它为准(无则回退bbox)。
    confidence: float = 0.0                # 平均置信度
    first_seen_frame: int = 0              # 首次出现帧
    last_seen_frame: int = 0               # 最后出现帧
    first_timestamp: float = 0.0           # 首次时间戳
    last_timestamp: float = 0.0            # 最后时间戳
    observations: List[TextLine] = field(default_factory=list)  # 所有观测
    is_new: bool = True                    # 是否为新内容(首次出现)

    def update_with(self, line: TextLine) -> None:
        """用新观测更新跟踪行"""
        self.observations.append(line)
        self.last_seen_frame = line.frame_id
        self.last_timestamp = line.timestamp
        self.bbox = line.bbox
        # 多帧投票: 选择出现次数最多的文本
        text_counts: dict = {}
        for obs in self.observations:
            text_counts[obs.text] = text_counts.get(obs.text, 0) + 1
        self.text = max(text_counts, key=text_counts.get)
        confidences = [o.confidence for o in self.observations]
        self.confidence = sum(confidences) / len(confidences)

    def observation_count(self) -> int:
        return len(self.observations)


class SystemState(Enum):
    """系统状态"""
    NORMAL = "normal"          # 正常提取
    SUSPENDED = "suspended"    # 挂起(误触退出/页面切换)
    RESUMING = "resuming"      # 恢复中


@dataclass
class StateAction:
    """状态机输出动作"""
    state: SystemState
    should_process: bool = True       # 是否处理当前帧 OCR
    is_state_change: bool = False     # 是否发生状态转换
    is_gap: bool = False              # 是否检测到内容缺口
    gap_detail: str = ""              # 缺口描述
    overlap_ratio: float = 0.0        # 与前一正常帧的内容重叠率
