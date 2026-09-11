"""
帧价值估计器

核心思想:
  送检决策的通用化: 用连续价值分数替代"状态硬分类"。

  帧价值 V = 视觉新颖度(novelty) × 清晰度(clarity):

    - 新颖度 = max(内容替换证据, 位移免疫的新条带比例):
      * 内容替换证据: 相位相关峰 pc_peak 低 = 两帧不是简单平移 = 内容替换
        (弹窗/toast/页面切换/新消息)。非刚体运动无需像素对齐, 直接强证据。
      * 位移免疫的新条带比例: 垂直位移搜索(PC dy 引导)对齐后, 条带级像素
        差异超阈值的新条带比例——纯滚动帧对齐后只有滚入/滚出的条带算新,
        位移天然免疫。引导搜索范围只需覆盖 PC 累积位移的残差(±8px), 快速
        滚动(帧间位移>16px)不再失效。
      * 融合取 max: 任一证据强即视为新内容(效果优先, 成本可多花)。

    - 清晰度: 梯度方向熵(GDE)闸门, 运动拖影帧价值归零。

  参考帧维护: producer 每次送检后 update_reference 并重置累计位移。
"""

from typing import Optional, Tuple

import numpy as np

from src.utils.image_utils import compute_gradient_direction_entropy
from src.utils.behavior_logger import BehaviorLogger


class FrameValueEstimator:
    """帧价值估计器"""

    def __init__(self, config: dict, logger: BehaviorLogger):
        # 新颖度参数
        self._strip_count: int = int(config.get("value_strip_count", 8))
        # 位移引导搜索半宽(px): 只需覆盖PC dy累积残差
        self._shift_tol: int = int(config.get("value_shift_tol", 8))
        # 位移clamp上限(px)的兜底: 实际上限 = max(此值, 屏高h), 保证滚动整屏
        # 也能被搜索范围覆盖。快速滚动帧间位移可到30px+。
        self._max_shift: int = int(config.get("value_max_shift", 60))
        # 条带差异阈值: 对齐后单条带MAE超过此值判"新条带"(灰度0-255)
        self._strip_mae_threshold: float = float(
            config.get("value_strip_mae_threshold", 6.0))
        # 内容替换证据的最小贡献: pc_peak 低于此值视为强替换证据
        self._peak_replace: float = float(config.get("value_peak_replace", 0.5))
        # 清晰度参数
        self._gde_min: float = float(config.get("value_gde_min", 2.6))
        self._logger = logger
        # 最近送检参考帧(降采样灰度)
        self._ref_gray: Optional[np.ndarray] = None
        # 参考帧→当前帧的累积位移估计(px, 由PC dy积分)
        self._est_shift: float = 0.0
        # 统计
        self._last_novelty: float = 0.0
        self._last_clarity: float = 1.0

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def estimate(self, fd) -> dict:
        """
        评估帧价值

        参数:
            fd: FrameData(需含 gray_small
        返回:
            {"novelty": float, "clarity": float, "value": float}
        """
        gray_small = fd.gray_small
        if (self._ref_gray is None
                or gray_small.shape != self._ref_gray.shape):
            # 无参考帧(首帧) → 高价值, 正常送检
            self._last_novelty = 1.0
            self._last_clarity = 1.0
            return {"novelty": 1.0, "clarity": 1.0, "value": 1.0,
                    # 诊断字段(弹窗/toast/切换排查)
                    "diag": {"novelty_src": "no_ref",
                             "pc_state": getattr(fd, "pc_state", "static"),
                             "pc_peak": round(float(getattr(fd, "pc_peak", 0.0)), 3),
                             "pc_dy": round(float(getattr(fd, "pc_dy", 0.0)), 2),
                             "est_shift": round(self._est_shift, 2),
                             "replace_evidence": 0.0,
                             "gde": None}}

        # 清晰度闸门
        gde = compute_gradient_direction_entropy(gray_small)
        clarity = 1.0 if gde >= self._gde_min else 0.0

        # 累计位移: 帧间PC dy积分(内容替换段无平移意义, 重置)
        pc_dy = float(getattr(fd, "pc_dy", 0.0))
        if getattr(fd, "pc_state", "static") == "content_change":
            self._est_shift = 0.0
        else:
            self._est_shift += pc_dy
            # clamp上限需至少覆盖整屏: 快速滚动帧间位移可达30px+, 若上限只有
            # 半屏高, 位移被截断且永久触发"半屏短路"误判全屏全新。
            cap = max(self._max_shift, gray_small.shape[0])
            self._est_shift = max(-cap, min(cap, self._est_shift))

        # 内容替换证据: pc_peak 低 = 非刚体内容替换(弹窗/toast/切换)
        pc_peak = float(getattr(fd, "pc_peak", 0.0))
        replace_evidence = max(0.0, 1.0 - pc_peak / self._peak_replace)
        replace_evidence = min(1.0, replace_evidence)

        # 新颖度按运动类型分流
        #   - 滚动(scrolling): 位移比例 = 滚入新内容比例 = |est_shift|/h。
        #     滚动动画帧本身像素差异大(运动模糊+滚入内容), 条带对齐后的
        #     重叠区MAE也远超阈值, 逐条带判新必然饱和; 位移比例免疫这一切,
        #     物理意义明确(滚动半屏≈滚入50%新内容才值得送检)。
        #   - 静止/内容替换(static/content_change): 条带级对齐新颖度。
        pc_state = getattr(fd, "pc_state", "static")
        if pc_state == "scrolling":
            h = gray_small.shape[0]
            novelty = min(1.0, abs(self._est_shift) / max(1, h))
            novelty_src = "shift_ratio"
            strip_diag: dict = {}
        else:
            novelty, strip_diag = self._shift_novelty(gray_small)
            novelty_src = "strip_aligned"

        # 融合取max: 任一证据强即新内容
        novelty_final = max(replace_evidence, novelty)
        self._last_novelty = novelty_final
        self._last_clarity = clarity
        return {"novelty": novelty_final, "clarity": clarity,
                "value": novelty_final * clarity,
                # 诊断字段: 完整还原评分路径(弹窗/toast/切换/新屏排查用)
                "diag": {"novelty_src": novelty_src,
                         "pc_state": pc_state,
                         "pc_peak": round(pc_peak, 3),
                         "pc_dy": round(pc_dy, 2),
                         "est_shift": round(self._est_shift, 2),
                         "replace_evidence": round(replace_evidence, 3),
                         "novelty_branch": round(novelty, 3),
                         "gde": round(gde, 3),
                         **strip_diag}}

    def update_reference(self, gray_small: np.ndarray) -> None:
        """更新送检参考帧并重置累计位移"""
        self._ref_gray = gray_small.copy()
        self._est_shift = 0.0

    @property
    def last_novelty(self) -> float:
        return self._last_novelty

    @property
    def last_clarity(self) -> float:
        return self._last_clarity

    # ------------------------------------------------------------------
    # 位移免疫新颖度
    # ------------------------------------------------------------------

    def _shift_novelty(self, cur: np.ndarray) -> Tuple[float, dict]:
        """
        位移引导对齐后的条带级新颖度

        返回 (新颖度, 诊断字段)

        搜索中心 = 累积位移(PC dy积分), 范围 ±shift_tol: 只需覆盖残差。
        PC不可用时(est_shift=0且无信号)退化为全范围粗搜。
        """
        ref = self._ref_gray
        h, w = cur.shape

        # 位移过大 → 内容基本全新(滚动满屏才全新; 半屏只滚入50%新内容,
        # 由条带对齐给出比例, 不应短路)。快速滚动帧间位移可达30px+, 若
        # 半屏(60px)就短路, 2-3帧累积即触发, 滚动段每帧都误判全新。
        if abs(self._est_shift) >= h:
            return 1.0, {"short_circuit": True,
                         "best_shift": int(round(self._est_shift)),
                         "best_mae": None,
                         "new_strips": self._strip_count}

        # 1. 位移引导搜索
        # 搜索范围clamp需覆盖整屏(est_shift可累积到h), 否则快速滚动时
        # 搜索中心=est_shift=100+会被_max_shift=60截断, 找不到真实对齐。
        center = int(round(self._est_shift))
        cap = max(self._max_shift, h)
        lo = max(-cap, center - self._shift_tol)
        hi = min(cap, center + self._shift_tol)
        if self._shift_tol <= 0:
            # 全范围粗搜(stride 4)
            lo, hi = -cap, cap
        cur_f = cur.astype(np.float64)
        ref_f = ref.astype(np.float64)
        best_shift = 0
        best_mae = float("inf")
        step = 1 if self._shift_tol > 0 else 4
        for s in range(lo, hi + 1, step):
            if s >= 0:
                a = cur_f[s:, :]
                b = ref_f[:h - s, :]
            else:
                a = cur_f[:h + s, :]
                b = ref_f[-s:, :]
            mae = float(np.mean(np.abs(a - b)))
            if mae < best_mae:
                best_mae = mae
                best_shift = s
        # 无条件精调: 粗搜步长4px会错过精确位移(快速滚动帧位移可达-32),
        # 且粗搜命中点 abs>8 时旧代码不精调导致对齐残差大、新颖度饱和。
        # 粗搜后始终在 ±8 内细搜, 对齐到像素级。
        if self._shift_tol <= 0 and step > 1:
            for s in range(max(-cap, best_shift - 8),
                           min(cap, best_shift + 8) + 1):
                if s >= 0:
                    a = cur_f[s:, :]
                    b = ref_f[:h - s, :]
                else:
                    a = cur_f[:h + s, :]
                    b = ref_f[-s:, :]
                mae = float(np.mean(np.abs(a - b)))
                if mae < best_mae:
                    best_mae = mae
                    best_shift = s

        # 2. 条带级新颖度
        strip_h = max(1, h // self._strip_count)
        new_strips = 0
        for i in range(self._strip_count):
            y0 = i * strip_h
            y1 = min((i + 1) * strip_h, h)
            ref_y0 = y0 - best_shift   # cur行r对应ref行 r-best_shift
            ref_y1 = y1 - best_shift
            if ref_y0 < 0 or ref_y1 > h:
                # 非重叠区: 滚入/滚出的新内容
                new_strips += 1
                continue
            diff = np.abs(cur_f[y0:y1, :] - ref_f[ref_y0:ref_y1, :])
            if float(np.mean(diff)) > self._strip_mae_threshold:
                new_strips += 1

        return (new_strips / self._strip_count,
                {"short_circuit": False,
                 "best_shift": best_shift,
                 "best_mae": round(best_mae, 2),
                 "new_strips": new_strips,
                 "strip_total": self._strip_count})
