"""
文本工具模块

提供文本相似度计算、bbox 操作、文本规范化等工具。
"""

import re
from typing import Tuple
import numpy as np


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算 Levenshtein 编辑距离。

    使用动态规划, O(N*M) 复杂度。
    对长文本(>200 字符)截断计算以控制耗时。
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    max_len = 200
    s1_trunc = s1[:max_len]
    s2_trunc = s2[:max_len]

    previous = list(range(len(s2_trunc) + 1))
    for i, c1 in enumerate(s1_trunc):
        current = [i + 1]
        for j, c2 in enumerate(s2_trunc):
            insertions = previous[j + 1] + 1
            deletions = current[j] + 1
            substitutions = previous[j] + (c1 != c2)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def text_similarity(s1: str, s2: str) -> float:
    """
    计算文本相似度 (0~1)。

    基于归一化编辑距离: similarity = 1 - ED
    """
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    ed = levenshtein_distance(s1, s2)
    return 1.0 - ed / max_len


def bbox_iou(bbox1: Tuple, bbox2: Tuple) -> float:
    """
    计算两个 bbox 的 IoU。

    bbox 格式: (x, y, w, h)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    if xa >= xb or ya >= yb:
        return 0.0

    inter = (xb - xa) * (yb - ya)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_center(bbox: Tuple) -> Tuple[float, float]:
    """计算 bbox 中心点"""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def bbox_center_distance(bbox1: Tuple, bbox2: Tuple) -> float:
    """计算两个 bbox 中心点之间的欧氏距离"""
    cx1, cy1 = bbox_center(bbox1)
    cx2, cy2 = bbox_center(bbox2)
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def normalize_text(text: str) -> str:
    """
    规范化文本: 去除多余空白, 统一标点。
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_meaningful_text(text: str, min_length: int = 2) -> bool:
    """
    判断文本是否有意义(非纯符号/数字)。
    """
    cleaned = re.sub(r"[\s\d\W]", "", text)
    return len(cleaned) >= min_length
