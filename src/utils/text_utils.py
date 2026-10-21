"""
文本工具模块

提供文本相似度计算、bbox 操作、文本规范化等工具。
"""

import re
import unicodedata
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


# ---- 数字锚点 / digit anchor ------------------------------------------------
# 模板消息(订单SN0010 / 订单SN0014、您的快递8812 / 8816)的文字部分完全一致,
# 唯一区别是数字位。编辑距离相似度对这类"仅数字不同"的近重复给出的分值极高
# (11字差1字 → 0.91), 任何基于相似度的判据都分不开它们, 只有把数字单独抽出来
# 逐位比对才行 —— 这就是"数字锚点"。
# Template messages differ only in their digits, and edit-distance similarity
# scores them near-identically; comparing the digit runs positionally is the
# only reliable discriminator.

def digit_anchor(text: str) -> str:
    """数字锚点: NFKC 归一后所有数字字符的逐位拼接。

    用"拼接"而非"按标点分组", 以容忍 OCR/ASR 在标点上的抖动
    ("repro-20261016-0037.md" 与 "repro202610160037MD" 锚点相同)。
    """
    s = unicodedata.normalize("NFKC", text or "")
    return "".join(re.findall(r"\d", s))


def digit_skeleton(text: str) -> str:
    """去掉数字段后的文本骨架(用于判断"差异是否只在数字位")。"""
    s = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\d+", "", s)


def _is_subsequence(short: str, long: str) -> bool:
    """short 是否为 long 的子序列(不要求连续)。"""
    it = iter(long)
    return all(ch in it for ch in short)


def anchors_compatible(a: str, b: str, min_len: int = 4) -> bool:
    """两个数字锚点是否相容 / whether two digit anchors can be the same message.

    相容 = 完全相等, 或短锚点是长锚点的**子序列**(容忍 OCR/ASR 多读或漏读一位
    数字)。子序列仅在长锚点达到 min_len 时放行, 防 "12"⊂"123" 之类偶然包含。
    任一侧无数字时视为相容(不构成冲突)。
    """
    if not a or not b or a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(long) >= min_len and _is_subsequence(short, long)


def template_digit_conflict(a: str, b: str,
                            skeleton_sim: float = 0.90,
                            anchor_min_len: int = 4) -> bool:
    """两文本是否"同模板、数字不同"→ 应判为**不同消息**。

    判据(同时满足才算冲突 / both must hold):
      1) 两侧都含数字, 且数字锚点不相容(见 `anchors_compatible`);
      2) 去掉数字后的骨架相似度 ≥ skeleton_sim —— 即差异**只在数字位**。
    条件 2 把判据收紧到模板近重复, 避免误伤"数字不同且内容也不同"的正常匹配。

    >>> template_digit_conflict("订单SN0010已确认收款", "订单SN0014已确认收款")
    True
    >>> template_digit_conflict("订单SN0010已确认收款", "订单SN00l0已确认收款")
    False
    """
    if anchors_compatible(digit_anchor(a), digit_anchor(b), anchor_min_len):
        return False
    return text_similarity(digit_skeleton(a), digit_skeleton(b)) >= skeleton_sim
