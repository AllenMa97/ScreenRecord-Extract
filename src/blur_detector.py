"""
拉普拉斯 + 梯度方向熵(GDE)双证据模糊检测器

核心思想:
  单用拉普拉斯方差判模糊在算法层面不可靠: 它与文本密度强相关
  (稀疏页面分数天然低, xxr 后期 875 vs 预热 5000+, OCR 却全部清晰),
  基线/阈值都在"猜内容密度"而非"猜模糊", 泛化性差。

  本版改为双证据 AND 判定, 成本不对称(宁多一次OCR, 不丢内容):
    blurry = 拉普拉斯低  AND  梯度方向熵低
    - 拉普拉斯(弱证据, 对文本密度敏感): 感知"分数下降"
    - GDE(强证据, 与内容密度解耦): 确认"方向性拖影"
  实测: 真实清晰帧 GDE 稳定在 3.25~3.44 (与拉普拉斯 875~9000 无关),
  方向性拖影 <2.6, 纯空白=0。稀疏页拉普拉斯低但 GDE 高 → 放行;
  拖影两证据齐 → 过滤; 高斯模糊(各向同性)拉普拉斯极低仍被 GDE 拦截。
  只有两个证据同时成立才判模糊, 单一证据(内容稀疏或轻微降质)
  一律放行交给下游(视觉去重 + OCR置信度)兜底。

启发式破锁:
  1. 连续全滤强制重置: 连续 N 帧全判模糊 → 内容密度可能切换导致基线失真,
     用最近分数重建基线并强制放行一帧(failsafe, 防永久过滤)
  2. 窗口过期重置: 窗口长期未更新(如长时间空白) → 清窗重填, 恢复后立即适应新内容。
  3. EWMA 平滑: 全滤重置时若历史不足, 用指数滑动平均作为初始基线。

参考文献:
  - 拉普拉斯方差: 自动对焦/图像质量评估经典方法
  - GDE: 无参考图像质量评价中利用梯度方向统计检测运动模糊
"""

import time
from collections import deque
from typing import Optional

import numpy as np

from src.utils.image_utils import (
    to_gray, resize_frame, compute_laplacian_variance,
    compute_gradient_direction_entropy
)
from src.utils.behavior_logger import BehaviorLogger


class BlurDetector:
    """双证据模糊检测器(拉普拉斯+GDE, AND判定, 启发式破锁)"""

    def __init__(self, config: dict, logger: BehaviorLogger):
        self._window_size: int = config.get("window_size", 10)
        self._blur_ratio: float = config.get("blur_ratio", 0.3)
        self._min_absolute: float = config.get("min_absolute_score", 10.0)
        self._compute_w: int = config.get("compute_width", 320)

        # GDE: 与内容密度解耦的强证据(绝对下限, 无需自适应校准)
        self._gde_bins: int = config.get("gde_bins", 36)
        self._gde_floor: float = config.get("gde_floor", 2.6)
        self._gde_weak_ratio: float = config.get("gde_weak_ratio", 0.1)

        # 启发式破锁
        self._unlock_consecutive: int = config.get("unlock_consecutive", 5)
        self._stale_seconds: float = config.get("stale_seconds", 8.0)
        self._ewma_alpha: float = config.get("ewma_alpha", 0.2)

        self._logger = logger

        self._score_window: deque = deque(maxlen=self._window_size)
        self._baseline: Optional[float] = None
        # 预热阶段收集的原始Score
        self._warmup_scores: list = []
        self._collect_warmup: bool = True
        # 破锁状态
        self._recent_scores: deque = deque(maxlen=max(self._window_size * 2, 16))
        self._consecutive_blurry: int = 0
        self._last_update_ts: Optional[float] = None
        self._ewma: Optional[float] = None

    def compute_score(self, roi_frame: np.ndarray) -> float:
        """
        计算拉普拉斯方差 Score。

        在降采样灰度图上计算以加速。
        """
        gray = self._to_gray_resized(roi_frame)
        return compute_laplacian_variance(gray)

    def _to_gray_resized(self, roi_frame: np.ndarray) -> np.ndarray:
        """降采样灰度预处理"""
        gray = to_gray(roi_frame)
        h, w = gray.shape[:2]
        if w > self._compute_w:
            scale = self._compute_w / w
            new_h = max(1, int(h * scale))
            gray = resize_frame(gray, self._compute_w, new_h)
        return gray

    def _compute_features(self, roi_frame: np.ndarray) -> tuple:
        """计算双特征(拉普拉斯, GDE)"""
        gray = self._to_gray_resized(roi_frame)
        lap = compute_laplacian_variance(gray)
        gde = compute_gradient_direction_entropy(
            gray, self._gde_bins, self._gde_weak_ratio)
        return lap, gde

    def _reset_window(self, lap: float) -> None:
        """用最近分数重建基线窗口"""
        recent = list(self._recent_scores)
        pool = [s for s in recent if s >= self._min_absolute]
        if not pool:
            pool = recent or []
        pool.append(lap)
        # 取低分位作为基线: 反映"当前内容密度", 而非历史密集页
        pool_sorted = sorted(pool)
        seed = pool_sorted[min(len(pool_sorted) - 1, max(0, len(pool_sorted) // 4))]
        self._score_window.clear()
        for s in recent[-self._window_size:]:
            if s >= self._min_absolute:
                self._score_window.append(s)
        if not self._score_window:
            self._score_window.append(max(seed, lap))
        self._baseline = float(np.median(list(self._score_window)))
        self._last_update_ts = time.time()

    def is_blurry(self, roi_frame: np.ndarray) -> bool:
        """
        判断当前帧是否模糊。

        判定逻辑:
          1. 双特征: 拉普拉斯方差(lap, 弱证据) + 梯度方向熵(gde, 强证据)
          2. 弱证据: lap < max(基线*blur_ratio, 绝对下限)
          3. 强证据: gde < gde_floor(绝对, 与内容密度解耦)
          4. AND: 两证据同时成立才判模糊(成本不对称: 单证据一律放行)
          5. 破锁: 连续全滤/窗口过期 → 重建基线并放行
          6. 分数不低于绝对下限的帧入窗更新基线(滑动中位数, 跟随内容密度)
        """
        lap, gde = self._compute_features(roi_frame)

        if self._collect_warmup:
            self._warmup_scores.append(lap)
        self._recent_scores.append(lap)
        now = time.time()

        # 过期重置: 窗口长期未更新 → 重建, 恢复后立即适应新内容
        if (self._last_update_ts is not None
                and now - self._last_update_ts > self._stale_seconds):
            self._logger.log("blur_detector", "stale_reset",
                             f"窗口过期({now - self._last_update_ts:.1f}s)重置基线")
            self._reset_window(lap)

        if len(self._score_window) == 0:
            # 初始/重置后首帧: 用绝对下限兜底
            lap_low = lap < self._min_absolute
            gde_low = gde < self._gde_floor
            is_blurry = lap_low and gde_low
            mode = "init_threshold"
            baseline = None
        else:
            baseline = float(np.median(list(self._score_window)))
            threshold = max(baseline * self._blur_ratio, self._min_absolute)
            lap_low = lap < threshold
            gde_low = gde < self._gde_floor
            is_blurry = lap_low and gde_low
            mode = "dual_cue"

        # 窗口更新: 分数不低于绝对下限即入窗(基线跟随内容密度, 防全滤死锁)
        if lap >= self._min_absolute:
            self._score_window.append(lap)
            self._baseline = float(np.median(list(self._score_window)))
            self._last_update_ts = now
            self._ewma = (self._ewma_alpha * lap
                          + (1 - self._ewma_alpha) * self._ewma) \
                if self._ewma is not None else float(lap)

        # 启发式破锁: 连续全滤 → 重建基线 + 强制放行一帧
        if is_blurry:
            self._consecutive_blurry += 1
            if self._consecutive_blurry >= self._unlock_consecutive:
                self._logger.log(
                    "blur_detector", "unlock_forced",
                    f"连续{self._consecutive_blurry}帧全滤, 重建基线并强制放行",
                    extra={"lap": round(lap, 2), "gde": round(gde, 3)})
                self._reset_window(lap)
                self._consecutive_blurry = 0
                is_blurry = False  # 成本不对称: 宁多一次OCR, 不丢内容
        else:
            self._consecutive_blurry = 0

        self._logger.log(
            "blur_detector", "check",
            f"score={lap:.2f}, gde={gde:.3f}, "
            f"baseline={'None' if baseline is None else f'{baseline:.2f}'} "
            f"(mode={mode}), lap_low={lap_low}, gde_low={gde_low}, "
            f"blurry={is_blurry}",
            extra={"score": round(lap, 2), "gde": round(gde, 3),
                   "baseline": None if baseline is None else round(baseline, 2),
                   "lap_low": lap_low, "gde_low": gde_low,
                   "blurry": is_blurry})
        return is_blurry

    def finalize_calibration(self, floor_pct: float = 5.0) -> None:
        """
        预热结束时预填基线窗口。

        由流水线在探索阶段结束后调用。
        用预热期分数预填基线窗口, 使稳态一开始就有可靠基线。
        注意: 不再用预热期 p5 抬升绝对下限 min_absolute——拉普拉斯方差与
        文本密度强相关, 预热期密集页面(如长对话)的 p5 远高于后期稀疏页面
        (认证记录/留白多)的清晰分数, 若抬升为地板会把清晰但稀疏的页面
        全部误杀(实测 xxr: 后期 score 875~4090 vs 预热 p5=5093, OCR 却
        全部清晰可读)。绝对下限仅兜底全黑/全白等极端帧。
        GDE 是绝对阈值且与内容密度解耦, 无需预热校准。
        """
        self._collect_warmup = False
        if not self._warmup_scores:
            return
        # 预填窗口: 分数不低于绝对下限(兜底)的预热帧
        clear_scores = [s for s in self._warmup_scores if s >= self._min_absolute]
        for s in clear_scores[: self._window_size]:
            self._score_window.append(s)
        if self._score_window:
            self._baseline = float(np.median(list(self._score_window)))
            self._last_update_ts = time.time()
            self._ewma = float(np.median(list(self._score_window)))
        self._logger.log("blur_detector", "calibrate",
                         f"预热校准: 绝对下限保持 {self._min_absolute:.1f}, "
                         f"GDE地板 {self._gde_floor:.2f}, "
                         f"基线预填{len(self._score_window)}帧",
                         extra={"min_absolute": round(self._min_absolute, 2),
                                "gde_floor": round(self._gde_floor, 3),
                                "baseline_frames": len(self._score_window)})

    @property
    def baseline(self) -> Optional[float]:
        return self._baseline

    @property
    def min_absolute(self) -> float:
        """当前绝对下限"""
        return self._min_absolute

    @property
    def gde_floor(self) -> float:
        """GDE 地板"""
        return self._gde_floor
