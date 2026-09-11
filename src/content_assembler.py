"""
内容组装器

负责将文本跟踪器的输出组装为最终结构化结果。

流程:
  1. 收集所有跟踪行和新内容行
  2. 按时间顺序排列
  3. 标注内容缺口
  4. 可选: 调用LLM进行OCR纠错和结构化
  5. 输出最终JSON结果
"""

import json
import os
from collections import OrderedDict
from typing import List

from src.bubble_merger import classify_side, classify_side_line, merge_wrapped_lines
from src.data_types import TrackedLine, TextLine
from src.llm_client import LlmClient
from src.screenshot_filter import (collapse_screenshot_clusters,
                                   detect_screenshot_clusters)
from src.utils.behavior_logger import BehaviorLogger


class ContentAssembler:
    """内容组装器"""

    def __init__(self, config: dict, logger: BehaviorLogger,
                 output_dir: str = "output"):
        self._logger = logger
        self._output_dir = output_dir
        self._llm = LlmClient(config, logger)
        chat_cfg = (config or {}).get("chat_format", {})
        self._chat_enabled: bool = chat_cfg.get("enabled", True)
        self._center_margin: int = int(chat_cfg.get("center_margin", 20))
        self._all_new_lines: List[TextLine] = []
        self._all_tracked: List[TrackedLine] = []
        self._gaps: List[dict] = []
        self._roi_width: int = 0
        self._frame_provider = None   # 气泡像素验证用帧提供器

    def set_roi_width(self, width: int) -> None:
        """设置ROI宽度(聊天收发定位用)"""
        self._roi_width = int(width or 0)
        self._llm.set_roi_width(self._roi_width)

    def set_frame_provider(self, provider) -> None:
        """设置按帧序号(1基)取ROI帧的回调, 供气泡像素验证/颜色判定使用。

        同一屏的多行共享 first_seen_frame, 随机读帧(seek+decode)较慢, 这里
        套一层小容量LRU缓存, 避免每行重复解码同一帧。
        """
        self._frame_provider = self._cached_provider(provider)
        self._llm.set_frame_provider(self._frame_provider)

    def _cached_provider(self, provider):
        """给原始帧提供器加LRU缓存(小容量, 同屏帧复用即可)"""
        if provider is None:
            return None
        cache: OrderedDict = OrderedDict()
        max_size = 30

        def cached(frame_id: int):
            if frame_id in cache:
                cache.move_to_end(frame_id)
                return cache[frame_id]
            frame = provider(frame_id)
            if frame is not None and frame.size:
                cache[frame_id] = frame
                while len(cache) > max_size:
                    cache.popitem(last=False)
            return frame
        return cached

    def add_new_lines(self, lines: List[TextLine]) -> None:
        """添加新发现的文本行"""
        self._all_new_lines.extend(lines)

    def add_gap(self, timestamp: float, detail: str) -> None:
        """添加内容缺口记录"""
        self._gaps.append({"timestamp": timestamp, "detail": detail})

    def collect_tracked(self, tracker) -> None:
        """从文本跟踪器收集所有跟踪行"""
        self._all_tracked = tracker.get_all_tracked()

    def assemble(self) -> dict:
        """
        组装最终结果。

        返回:
            结构化结果字典
        """
        self._logger.log("content_assembler", "start",
                         f"开始组装: {len(self._all_tracked)}跟踪行, "
                         f"{len(self._all_new_lines)}新行, {len(self._gaps)}缺口")

        # 气泡内换行合并(CV侧几何方案, 先于LLM): 一条长消息被界面自动换行后,
        # OCR返回多行, 若不合并会被拆成多条语义不完整的结果并带偏后续消息的
        # 合并/时间/归属。几何合并独立于LLM, 两条组装路径统一受益。
        merged, merged_count = merge_wrapped_lines(
            self._all_tracked,
            lambda line: classify_side_line(line, self._roi_width,
                                            self._center_margin,
                                            self._chat_enabled,
                                            self._frame_provider),
            self._frame_provider)
        if merged_count:
            self._logger.log("content_assembler", "wrapped_merge",
                             f"气泡内换行合并: {merged_count}处碎片合并为完整消息",
                             extra={"merged": merged_count})
        self._all_tracked = merged

        # 截图消息过滤(几何合并后、LLM增强前): 收发双方发送的 App 截图内部
        # 文字会被OCR识别成聊天行, 检测到截图簇后整簇折叠为 [图片消息] 占位,
        # 避免截图文字污染聊天正文; 占位行方向用纯几何位置(右=发, 左=收)。
        shot_clusters = detect_screenshot_clusters(self._all_tracked,
                                                   self._roi_width)
        if shot_clusters:
            shot_count = sum(len(c) for c in shot_clusters)
            self._all_tracked = collapse_screenshot_clusters(
                self._all_tracked, shot_clusters,
                lambda bbox: classify_side(bbox, self._roi_width,
                                           self._center_margin,
                                           self._chat_enabled))
            self._logger.log("content_assembler", "screenshot_filter",
                             f"截图过滤: {len(shot_clusters)}个截图簇, "
                             f"{shot_count}行截图文字折叠为[图片消息]",
                             extra={"clusters": len(shot_clusters),
                                    "rows": shot_count})

        # 使用LLM增强(如果启用)或基础组装
        result = self._llm.enhance_result(self._all_tracked, self._gaps)

        # 添加统计信息
        result["statistics"] = {
            "total_tracked_lines": len(self._all_tracked),
            "total_new_lines": len(self._all_new_lines),
            "total_gaps": len(self._gaps),
            "llm_enhanced": result.get("llm_enhanced", False),
        }

        self._logger.log("content_assembler", "complete",
                         f"组装完成: {result['statistics']}")
        return result

    def save(self, result: dict, video_name: str = "") -> str:
        """保存结果到文件。

        按视频名输出 extracted_content_<视频名>.json, 多视频结果互不覆盖;
        不再生成通用 extracted_content.json, 避免多视频下最后运行覆盖混淆。
        """
        os.makedirs(self._output_dir, exist_ok=True)
        filename = f"extracted_content_{video_name}.json" if video_name else "extracted_content.json"
        path = os.path.join(self._output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return path
