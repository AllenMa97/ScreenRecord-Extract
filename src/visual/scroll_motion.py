"""
帧级滚动位移估计 (Frame-level vertical scroll estimation).

用途 / Purpose
-------------
聊天记录录屏是滚动式的: 同一行消息在不同时刻的屏幕 `y` 不同, 必须换算成
"文档纵坐标"(y + 累计滚动量)才能还原真实聊天顺序。现有主路径用相邻 OCR 帧的
公共文本行求 Δy; 但 OCR 缓存是**按内容稀疏存储**的, 滚动剧烈的一段可能连续
多帧只识别到顶部常驻横幅(被静止行规则排除), 于是整段滚动被漏记(实测
t=98.5~101.4 漏记 218px), 后续所有气泡/文本的文档坐标整体错位。

本模块从**图像侧**补这个信号: 对相邻采样帧(默认 0.2s)在聊天区做多横带
`cv2.matchTemplate`, 取各横带纵向位移的中位数作为该步位移, 逐帧累计成曲线。
只依赖帧间相对运动, 不做跨长间隔匹配(长间隔内容已大幅变化, 匹配会错)。

约束 / Constraints
------------------
  * 只在相邻近帧之间估计(默认 0.2s), 快速滚动单步位移仍远小于搜索范围;
  * 多条横带取中位数, 个别横带被气泡动画(播放中的波形)干扰不影响结果;
  * 匹配得分或横带一致性不足时该步记为"不可靠", 累计量保持不变。
"""

from __future__ import annotations

import bisect
from typing import List, Optional, Tuple

import numpy as np

try:  # 与视觉模块一致: 优先使用项目已依赖的 opencv
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


SAMPLE_INTERVAL_S = 0.2   # 采样间隔(与气泡扫描一致)
WORK_WIDTH = 320          # 工作宽度; 先降采样再匹配, 降内存与噪声
STRIP_COUNT = 6           # 横向条带数
STRIP_H_RATIO = 0.10      # 每条带高度占 ROI 高比例
MIN_MATCH_SCORE = 0.80    # 单条带模板匹配得分下限
MIN_STRIPS = 3            # 至少多少条带参与(且得分达标)才认为该步可靠
MAX_STRIP_SPREAD = 4.0    # 条带位移的绝对中位差上限(px), 超过视为不可靠
MAX_DY_RATIO = 0.45       # 单步位移上限(相对工作高度), 防止匹配到错误重复结构


def _work_gray(frame: np.ndarray, roi_box: Tuple[int, int, int, int],
               work_width: int = WORK_WIDTH) -> Optional[np.ndarray]:
    """裁 ROI → 降采样 → 灰度 float32; 空帧返回 None。"""
    if cv2 is None:
        return None
    x, y, w, h = (int(v) for v in roi_box)
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0 or roi.ndim != 3:
        return None
    if roi.shape[1] > work_width:
        scale = work_width / float(roi.shape[1])
        roi = cv2.resize(roi, (work_width, int(round(roi.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)


def estimate_step_dy(prev_gray: np.ndarray,
                     cur_gray: np.ndarray) -> Optional[float]:
    """估计相邻两帧内容的纵向位移 dy(px, 正=内容下移, 负=内容上移)。

    多条横带各自做模板匹配, 取位移中位数; 可靠时返回 dy, 否则 None。
    """
    if cv2 is None or prev_gray is None or cur_gray is None:
        return None
    if prev_gray.shape != cur_gray.shape:
        return None
    H, W = prev_gray.shape
    band_h = max(8, int(round(H * STRIP_H_RATIO)))
    if H < band_h * 2 or W < 8:
        return None
    max_dy = int(round(H * MAX_DY_RATIO))
    dys: List[float] = []
    for i in range(STRIP_COUNT):
        center = int(round(H * (i + 0.5) / STRIP_COUNT))
        y0 = min(max(0, center - band_h // 2), H - band_h)
        tpl = prev_gray[y0:y0 + band_h, :]
        if float(tpl.std()) < 3.0:      # 纯色条带, 匹配无意义
            continue
        res = cv2.matchTemplate(cur_gray, tpl, cv2.TM_CCOEFF_NORMED)
        if res.size == 0:
            continue
        # res 的列对应水平位置(模板与图等宽 → 只有一列), 行对应纵向偏移
        col = res[:, 0]
        j = int(np.argmax(col))
        score = float(col[j])
        if score < MIN_MATCH_SCORE:
            continue
        dy = float(j - y0)
        if abs(dy) > max_dy:
            continue
        dys.append(dy)
    if len(dys) < MIN_STRIPS:
        return None
    med = float(np.median(dys))
    spread = float(np.median(np.abs(np.array(dys) - med)))
    if spread > MAX_STRIP_SPREAD:
        return None
    return med


def estimate_scroll_curve(video_path: str,
                          roi_box: Tuple[int, int, int, int],
                          interval_s: float = SAMPLE_INTERVAL_S
                          ) -> List[Tuple[float, float]]:
    """扫描视频, 返回图像侧累计滚动曲线 [(t, offset)]。

    offset 与 OCR 路径同向: 内容上滚(屏幕上移)时 offset 增大。
    无法可靠估计的步不改变 offset(视为静止), 与 OCR 路径的兜底一致。
    """
    if cv2 is None:
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = max(1, int(round(interval_s * fps)))
    curve: List[Tuple[float, float]] = []
    offset = 0.0
    prev_gray = None
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % frame_skip != 0:
            fid += 1
            continue
        ts = fid / fps
        gray = _work_gray(frame, roi_box)
        if gray is not None:
            if prev_gray is not None and prev_gray.shape == gray.shape:
                dy = estimate_step_dy(prev_gray, gray)
                if dy is not None:
                    offset += -dy          # 内容上移(dy<0) → 偏移增大
            curve.append((ts, offset))
            prev_gray = gray
        fid += 1
    cap.release()
    return curve


def offset_at(curve: List[Tuple[float, float]], t: float) -> float:
    """线性插值取累计偏移; 曲线为空返回 0。"""
    if not curve:
        return 0.0
    times = [c[0] for c in curve]
    i = bisect.bisect_right(times, float(t)) - 1
    if i < 0:
        return 0.0
    if i + 1 >= len(curve):
        return curve[-1][1]
    t0, o0 = curve[i]
    t1, o1 = curve[i + 1]
    if t1 <= t0:
        return o1
    ratio = min(max((float(t) - t0) / (t1 - t0), 0.0), 1.0)
    return o0 + (o1 - o0) * ratio
