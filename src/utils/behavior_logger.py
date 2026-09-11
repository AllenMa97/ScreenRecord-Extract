"""
行为日志器

记录系统在处理过程中的每一步决策, 产出可读的行为日志(非 LLM thinking 结果)。

日志格式为 JSON Lines (jsonl), 每行一条记录, 包含:
  - timestamp: 时间戳
  - module: 模块名
  - event: 事件类型
  - detail: 详细信息
  - frame_id: 关联帧 ID (可选)
"""

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional


class BehaviorLogger:
    """行为日志记录器"""

    def __init__(self, output_dir: str = "output", filename: str = "behavior_log.jsonl",
                 live_append: bool = False):
        self._logs: List[Dict[str, Any]] = []
        self._output_dir = output_dir
        self._filename = filename
        self._start_time = time.time()
        self._live_append = live_append
        self._lock = threading.Lock()
        self._fh = None

    def log(self, module: str, event: str, detail: str = "",
            frame_id: Optional[int] = None, extra: Optional[Dict] = None) -> None:
        """
        记录一条日志。

        参数:
            module: 模块名(如 frame_sampler, blur_detector 等)
            event: 事件类型(如 sample, skip, detect, suspend 等)
            detail: 人类可读的描述
            frame_id: 关联的帧 ID
            extra: 额外结构化数据
        """
        entry = {
            "elapsed_sec": round(time.time() - self._start_time, 3),
            "module": module,
            "event": event,
            "detail": detail,
        }
        if frame_id is not None:
            entry["frame_id"] = frame_id
        if extra:
            entry["extra"] = extra
        with self._lock:
            self._logs.append(entry)
            # 实时落盘(Web 演示用): 追加写文件, 便于外部实时读取日志流。
            # 默认关闭, 不影响原行为。
            if self._live_append:
                self._append_file(entry)

    def _append_file(self, entry: Dict[str, Any]) -> None:
        """增量追加一条日志到文件"""
        try:
            if self._fh is None:
                os.makedirs(self._output_dir, exist_ok=True)
                path = os.path.join(self._output_dir, self._filename)
                self._fh = open(path, "a", encoding="utf-8")
            self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._fh.flush()
        except OSError:
            pass

    def save(self) -> str:
        """保存日志到文件。返回文件路径。"""
        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, self._filename)
        with open(path, "w", encoding="utf-8") as f:
            for entry in self._logs:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        return path

    def get_summary(self) -> Dict[str, int]:
        """获取各模块事件计数摘要。"""
        summary: Dict[str, int] = {}
        for entry in self._logs:
            key = f"{entry['module']}.{entry['event']}"
            summary[key] = summary.get(key, 0) + 1
        return summary

    def to_list(self) -> List[Dict[str, Any]]:
        """返回所有日志记录。"""
        return self._logs
