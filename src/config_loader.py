"""
配置加载器

负责从 YAML 文件加载配置并提供类型安全的访问接口。
"""

import os
from typing import Any, Dict

import yaml


def _deep_merge(base: dict, overrides: dict) -> dict:
    """递归合并覆盖字典(Web 服务按会话覆盖配置用)。"""
    out = dict(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = "config.yaml",
                 overrides: dict = None):
        self._data: Dict[str, Any] = {}
        self._load(config_path)
        if overrides:
            self._data = _deep_merge(self._data, overrides)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在 / Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f) or {}

    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """获取配置项"""
        section_data = self._data.get(section, {})
        if key is None:
            return section_data
        return section_data.get(key, default)

    @property
    def sampling(self) -> dict:
        return self._data.get("sampling", {})

    @property
    def roi(self) -> dict:
        return self._data.get("roi", {})

    @property
    def blur(self) -> dict:
        return self._data.get("blur", {})

    @property
    def visual_dedup(self) -> dict:
        return self._data.get("visual_dedup", {})

    @property
    def ocr(self) -> dict:
        return self._data.get("ocr", {})

    @property
    def text_tracker(self) -> dict:
        return self._data.get("text_tracker", {})

    @property
    def state_machine(self) -> dict:
        return self._data.get("state_machine", {})

    @property
    def llm(self) -> dict:
        return self._data.get("llm", {})

    @property
    def audio(self) -> dict:
        return self._data.get("audio", {})

    @property
    def calibration(self) -> dict:
        return self._data.get("calibration", {})

    @property
    def output(self) -> dict:
        return self._data.get("output", {})
