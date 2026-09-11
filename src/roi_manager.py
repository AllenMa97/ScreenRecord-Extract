"""
ROI管理器

负责检测和维护感兴趣区域(ROI), 剔除状态栏、导航栏等干扰区域。

设计原则:
  - ROI是屏幕上的固定视口区域(非内容区域)
  - 内容在ROI内部滚动, ROI边界固定
  - ROI裁剪后保留原始分辨率送入OCR
  - 仅在计算帧差/拉普拉斯时降采样

自适应内容区检测:
  ROI = 持续变化的行带。顶部的OS状态栏与底部的输入框/导航栏在时序上
  "近似静止"(时钟跳动/图标出现属低频事件, 活动率极低), 因此预热期内
  逐帧累计行级活动率, 预热结束时取最长连续高活动带作为内容区, 自然
  排除顶栏/底栏。数据不足或结果不可信时回退到配置margin。
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.image_utils import to_gray
from src.utils.behavior_logger import BehaviorLogger


class RoiManager:
    """ROI管理器"""

    def __init__(self, config: dict, logger: BehaviorLogger):
        self._margin_top: int = config.get("margin_top", 60)
        self._margin_bottom: int = config.get("margin_bottom", 120)
        self._margin_left: int = config.get("margin_left", 0)
        self._margin_right: int = config.get("margin_right", 0)
        self._recheck_interval: float = config.get("recheck_interval", 5.0)
        self._logger = logger
        self._roi_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._last_recheck_time: float = 0.0

        # 自适应内容区检测参数
        self._adaptive: bool = bool(config.get("adaptive", False))
        self._compute_width: int = int(config.get("compute_width", 160))
        self._noise_floor: float = float(config.get("noise_floor", 2.0))
        self._ratio_min: float = float(config.get("activity_ratio_min", 0.05))
        self._min_content_ratio: float = float(config.get("min_content_ratio", 0.3))
        self._pad_rows: int = int(config.get("pad_rows", 8))

        # 时序活动率缓冲
        self._activity: Optional[np.ndarray] = None   # 每行活跃帧计数
        self._pairs: int = 0                           # 累计帧对数
        self._ref_gray: Optional[np.ndarray] = None    # 上一帧降采样灰度
        self._frame_size: Tuple[int, int] = (0, 0)     # (w, h) 帧尺寸
        self._finalized: bool = False

    def _margin_box(self) -> Tuple[int, int, int, int]:
        """按配置margin计算基准ROI框"""
        w, h = self._frame_size
        x = self._margin_left
        y = self._margin_top
        roi_w = w - self._margin_left - self._margin_right
        roi_h = h - self._margin_top - self._margin_bottom

        if roi_w <= 0 or roi_h <= 0:
            self._logger.log("roi_manager", "warning",
                             f"ROI尺寸异常 / Abnormal ROI size: w={roi_w}, h={roi_h}")
            roi_w = max(roi_w, 100)
            roi_h = max(roi_h, 100)
        return (x, y, roi_w, roi_h)

    def detect_roi(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        从帧中检测ROI

        使用配置的边距裁剪状态栏和导航栏(首帧基准, 预热结束后由
        自适应检测修正)。Uses configured margins to crop status bar and

        返回 (x, y, w, h) 格式的ROI框
        """
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        self._roi_box = self._margin_box()
        self._logger.log("roi_manager", "detect",
                         f"ROI: x={self._roi_box[0]}, y={self._roi_box[1]}, "
                         f"w={self._roi_box[2]}, h={self._roi_box[3]}, "
                         f"frame_size={w}x{h}")
        return self._roi_box

    @property
    def roi_box(self) -> Tuple[int, int, int, int]:
        return self._roi_box

    def feed_frame(self, frame: np.ndarray) -> None:
        """
        预热期逐帧累计行级活动率

        与上一帧比较: 行平均绝对像素差超过噪声地板且整帧平均差超过
        噪声地板(排除压缩噪声)时, 该行记为一次活跃。活动率 = 活跃帧数 / 帧对数。
        """
        if not self._adaptive or self._finalized:
            return
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        if self._activity is None or self._activity.shape[0] != h:
            self._activity = np.zeros(h, dtype=np.int32)
            self._ref_gray = None
            self._pairs = 0

        gray = to_gray(frame)
        gray = cv2.resize(gray, (self._compute_width, h),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._ref_gray is not None:
            diff = np.abs(gray - self._ref_gray).mean(axis=1)
            self._pairs += 1
            if float(diff.mean()) >= self._noise_floor:
                self._activity += (diff >= self._noise_floor).astype(np.int32)
        self._ref_gray = gray

    def finalize(self) -> Tuple[int, int, int, int]:
        """
        预热结束: 由行活动率求解自适应内容区

        内容区 = 活动率平滑后超过阈值的最长连续行带, 顶边界≈状态栏下沿,
        底边界≈输入框/导航栏上沿。无足够数据或内容带过小(结果不可信)
        时回退到配置margin。Content = the longest contiguous band whose

        返回 (x, y, w, h) 格式的ROI框
        """
        self._finalized = True
        margin_box = self._margin_box()
        if not self._adaptive or self._pairs < 10:
            return margin_box

        h = self._activity.shape[0]
        ratio = self._activity.astype(np.float32) / self._pairs

        # 短窗口平滑: 消除单行噪声/时间跳动导致的孤立活跃行
        win = max(1, h // 64)
        kernel = np.ones(win, dtype=np.float32) / win
        smooth = np.convolve(ratio, kernel, mode="same")
        active = smooth >= self._ratio_min

        # 最长连续活跃带
        best_start, best_len = 0, 0
        cur_start: Optional[int] = None
        for i, a in enumerate(active):
            if a:
                if cur_start is None:
                    cur_start = i
            elif cur_start is not None:
                if i - cur_start > best_len:
                    best_start, best_len = cur_start, i - cur_start
                cur_start = None
        if cur_start is not None and h - cur_start > best_len:
            best_start, best_len = cur_start, h - cur_start

        # 内容带过小 → 结果不可信, 回退margin(如全静止视频/整屏无变化)
        if best_len < self._min_content_ratio * h:
            self._logger.log(
                "roi_manager", "adaptive",
                f"内容带过小({best_len}/{h}), 回退margin / Band too small, "
                f"fallback to margins",
                extra={"best_len": int(best_len), "frame_h": int(h),
                       "pairs": self._pairs})
            return margin_box

        top = max(0, best_start - self._pad_rows)
        bottom = min(h, best_start + best_len + self._pad_rows)
        x, _, w, _ = margin_box
        box = (x, top, w, bottom - top)
        self._roi_box = box
        self._logger.log(
            "roi_manager", "adaptive",
            f"自适应ROI: {box} (内容带 {best_start}~{best_start + best_len}, "
            f"pairs={self._pairs}, ratio_p50={float(np.median(ratio)):.3f}, "
            f"ratio_max={float(ratio.max()):.3f})",
            extra={"roi": list(box), "band_top": int(best_start),
                   "band_bottom": int(best_start + best_len),
                   "pairs": self._pairs,
                   "ratio_p50": round(float(np.median(ratio)), 3),
                   "ratio_max": round(float(ratio.max()), 3)})
        return box

    def validate_roi(self, frame: np.ndarray, current_time: float) -> bool:
        """
        校验ROI边界是否仍然有效

        每隔recheck_interval秒执行一次, 检查ROI边缘是否有预期的文本结构。

        返回True表示ROI有效
        """
        if current_time - self._last_recheck_time < self._recheck_interval:
            return True
        self._last_recheck_time = current_time

        x, y, w, h = self._roi_box
        roi_gray = to_gray(frame[y:y + h, x:x + w])

        # 检查ROI区域是否有足够的边缘信息(文本存在)
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / (w * h)

        is_valid = edge_density > 0.01
        if not is_valid:
            self._logger.log("roi_manager", "validate",
                             f"ROI边缘密度低 / Low edge density: {edge_density:.4f}",
                             extra={"edge_density": round(edge_density, 4)})
        return is_valid

    def should_recheck(self, current_time: float) -> bool:
        """是否需要重新检测ROI"""
        return current_time - self._last_recheck_time >= self._recheck_interval
