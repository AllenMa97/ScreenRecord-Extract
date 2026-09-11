"""
视觉去重器

核心创新: 用"行差分剖面 + 行条带dHash"替代全局pHash

方案A+B升级:
  方案A - 置信度软化: 不再做硬二分判定, 而是输出三档置信度:
    - HIGH_CONF_DUP: 高置信度重复, 跳过OCR(省算力)
    - HIGH_CONF_NEW: 高置信度新内容, 直接送OCR
    - GRAY_ZONE: 灰色地带, 送OCR + 标记需LLM语义判定

  方案B - 在线阈值校准: 根据OCR文本反馈动态调整视觉阈值:
    - OCR后发现文本完全重复 → 视觉阈值太松, 收紧diff_dup_thresh
    - OCR后发现文本完全不同 → 视觉阈值太紧, 放宽diff_dup_thresh
    - 使视觉模态逐渐适应当前视频的内容特征

问题背景:
  全局pHash捕获图像低频结构, 但聊天记录UI在不同滚动位置的布局
  高度相似, 导致pHash误判。行差分剖面+条带dHash解决了这个问题,
  但在灰色地带(正常滑动)仍有误判风险, 需LLM语义判定兜底。

  串行级联的致命缺陷: 视觉判DUPLICATE → 直接跳过OCR → 文本模态
  没有机会纠正。方案A通过三档置信度解决: 只有高置信度重复才跳过,
  灰色地带仍送OCR让LLM判定。
"""

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import List, Optional, Tuple

import numpy as np

from src.utils.image_utils import (
    to_gray, resize_frame, split_into_strips,
    compute_dhash, hamming_distance, compute_row_diff_profile, percentile
)
from src.row_fingerprint import RowFingerprintAligner
from src.utils.behavior_logger import BehaviorLogger


class DedupVerdict(Enum):
    """去重判定结果"""
    DUPLICATE = "duplicate"        # 重复(跳过)
    NEW_CONTENT = "new_content"    # 新内容(保留)
    OVERLAP = "overlap"            # 部分重叠(保留)


class ConfidenceTier(Enum):
    """
    置信度档位
    方案A核心: 视觉去重输出三档, 灰色地带需LLM仲裁。
    """
    HIGH_CONF_DUP = "high_conf_dup"    # 高置信度重复, 跳过OCR
    HIGH_CONF_NEW = "high_conf_new"    # 高置信度新内容, 直接送OCR
    GRAY_ZONE = "gray_zone"            # 灰色地带, 送OCR + LLM判定


@dataclass
class DedupResult:
    """去重判定结果"""
    verdict: DedupVerdict
    confidence_tier: ConfidenceTier = ConfidenceTier.HIGH_CONF_NEW
    total_diff: float = 0.0
    changed_strip_count: int = 0
    total_strips: int = 0
    strip_distances: Optional[List[int]] = None
    needs_llm_arbitration: bool = False  # 是否需要LLM语义仲裁


class VisualDedup:
    """
    视觉去重器(行差分剖面 + 行条带dHash + 置信度三档 + 在线校准)
    """

    def __init__(self, config: dict, logger: BehaviorLogger):
        self._strip_count: int = config.get("strip_count", 8)
        self._dhash_size: int = config.get("dhash_size", 8)
        self._strip_threshold: int = config.get("strip_distance_threshold", 5)
        self._changed_ratio: float = config.get("changed_strip_ratio", 0.5)

        # 原始阈值(配置值)
        self._orig_diff_dup_thresh: float = config.get("diff_duplicate_threshold", 300)
        self._orig_diff_new_thresh: float = config.get("diff_new_content_threshold", 3000)

        # 当前生效阈值(会被在线校准调整)
        self._diff_dup_thresh: float = self._orig_diff_dup_thresh
        self._diff_new_thresh: float = self._orig_diff_new_thresh

        # 方案A: 灰色地带的边界倍率
        # 高置信度重复: diff < dup_thresh * dup_tight_ratio
        # 灰色地带: dup_thresh * dup_tight_ratio <= diff <= new_thresh * new_loose_ratio
        # 高置信度新内容: diff > new_thresh * new_loose_ratio
        self._dup_tight_ratio: float = config.get("dup_tight_ratio", 0.5)
        self._new_loose_ratio: float = config.get("new_loose_ratio", 0.6)

        # 方案B: 在线校准参数
        self._calibration_enabled: bool = config.get("calibration_enabled", True)
        self._calibration_rate: float = config.get("calibration_rate", 0.1)
        self._calibration_min_dup: float = config.get("calibration_min_dup", 50.0)
        self._calibration_max_new: float = config.get("calibration_max_new", 10000.0)
        self._calibration_history: List[Tuple[float, str]] = []
        # 校准历史跨线程访问锁(采集线程写check, 消费线程读calibrate)
        self._cal_lock = Lock()

        self._compute_w: int = config.get("compute_width", 160)
        self._logger = logger

        # 方案C: 行指纹序列对齐(RFSA)灰色地带精筛。滚动是刚体垂直平移,
        # 同一行文本滚动后落入不同条带, 旧strip dHash位置绑定会误判新内容;
        # RFSA把帧编码为行指纹序列, 用带gap罚分的全局对齐, 位移天然免疫。
        # 回弹/滚动位移 → 判dup跳过OCR; 有新行 → 送OCR保召回。
        self._rf_enabled: bool = config.get("rf_enabled", False)
        self._row_aligner: Optional[RowFingerprintAligner] = None
        if self._rf_enabled:
            self._row_aligner = RowFingerprintAligner(config, logger)

        self._last_keyframe_gray: Optional[np.ndarray] = None
        self._last_keyframe_strips: Optional[List[np.ndarray]] = None
        self._last_keyframe_hashes: Optional[List[int]] = None

    def _prepare_gray(self, roi_frame: np.ndarray) -> np.ndarray:
        """准备降采样灰度图"""
        gray = to_gray(roi_frame)
        h, w = gray.shape[:2]
        if w > self._compute_w:
            scale = self._compute_w / w
            new_h = max(1, int(h * scale))
            gray = resize_frame(gray, self._compute_w, new_h)
        return gray

    def _compute_strip_hashes(self, gray: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """计算各条带的dHash"""
        strips = split_into_strips(gray, self._strip_count)
        hashes = [compute_dhash(s, self._dhash_size) for s in strips]
        return strips, hashes

    def check(self, roi_frame: np.ndarray) -> DedupResult:
        """检查当前帧是否为重复内容(仅返回判定)"""
        result, _ = self.check_with_gray(roi_frame)
        return result

    def check_with_gray(self, roi_frame: np.ndarray
                        ) -> Tuple[DedupResult, np.ndarray]:
        """
        检查当前帧是否为重复内容, 输出三档置信度, 并返回降采样灰度。

        判定逻辑:
          1. 行差分 < dup_thresh * dup_tight_ratio → HIGH_CONF_DUP (跳过OCR)
          2. 行差分 > new_thresh * new_loose_ratio → HIGH_CONF_NEW (送OCR)
          3. 中间区域 → 条带dHash精筛 → 仍不确定 → GRAY_ZONE (送OCR+LLM)
        """
        gray = self._prepare_gray(roi_frame)

        # 第一帧: 无比较对象
        if self._last_keyframe_gray is None:
            strips, hashes = self._compute_strip_hashes(gray)
            self._last_keyframe_gray = gray
            self._last_keyframe_strips = strips
            self._last_keyframe_hashes = hashes
            # 同步RFSA关键帧(与主关键帧同基准)
            if self._row_aligner is not None:
                self._row_aligner.update_keyframe(gray)
            self._logger.log("visual_dedup", "first_frame",
                             "首帧作为关键帧 / First frame as keyframe")
            return DedupResult(
                verdict=DedupVerdict.NEW_CONTENT,
                confidence_tier=ConfidenceTier.HIGH_CONF_NEW
            ), gray

        # Step 1: 行差分剖面
        profile = compute_row_diff_profile(gray, self._last_keyframe_gray)
        total_diff = float(profile.sum())

        # 方案A: 三档置信度判定
        high_dup_boundary = self._diff_dup_thresh * self._dup_tight_ratio
        high_new_boundary = self._diff_new_thresh * self._new_loose_ratio

        # 高置信度重复: 行差分极小
        if total_diff < high_dup_boundary:
            self._logger.log("visual_dedup", "high_conf_dup",
                             f"行差分={total_diff:.1f} < {high_dup_boundary:.1f}, "
                             f"高置信度重复, 跳过OCR",
                             extra={"total_diff": round(total_diff, 1),
                                    "tier": "high_conf_dup"})
            with self._cal_lock:
                self._calibration_history.append((total_diff, "dup"))
            return DedupResult(
                verdict=DedupVerdict.DUPLICATE,
                confidence_tier=ConfidenceTier.HIGH_CONF_DUP,
                total_diff=total_diff
            ), gray

        # Step 1.5: RFSA位移免疫主通道(锚点去重)。行差分对"内容上移/滚动位移"
        # 敏感(消息弹出使画面整体平移1-2px即产生大diff), 会把内容相同的帧误判
        # 为high_conf_new→重复OCR。RFSA把帧编码为行指纹序列做全局对齐, 位移
        # 天然免疫: 内容重叠但位移的帧(滚动/回弹/重复快照)对齐率高→判dup拦截;
        # 真正的新内容(如翻页/新消息批次)对齐率低→判new/gray继续。
        if self._rf_enabled and self._row_aligner is not None:
            rf = self._row_aligner.compare(gray)
            if rf.verdict == "dup":
                # 位移免疫判重复: 与任一历史关键帧对齐率>=dup_ratio且无实质新行
                # (内容重叠但画面平移)。这是"锚点去重"的核心拦截点。
                self._logger.log(
                    "visual_dedup", "rf_dup",
                    f"RFSA判重复(位移免疫): 对齐率={rf.aligned_ratio:.2f}, "
                    f"位移={rf.displacement}, 行差分={total_diff:.1f}, 跳过OCR",
                    extra={"aligned_ratio": round(rf.aligned_ratio, 3),
                           "displacement": rf.displacement,
                           "total_diff": round(total_diff, 1),
                           "tier": "high_conf_dup"})
                with self._cal_lock:
                    self._calibration_history.append((total_diff, "dup"))
                return DedupResult(
                    verdict=DedupVerdict.DUPLICATE,
                    confidence_tier=ConfidenceTier.HIGH_CONF_DUP,
                    total_diff=total_diff
                ), gray
            if rf.verdict == "new":
                # 大量新行(翻页/整屏内容替换/新消息批次) → 高置信度新内容
                self._update_keyframe(gray)
                self._logger.log(
                    "visual_dedup", "rf_new",
                    f"RFSA判新内容: 对齐率={rf.aligned_ratio:.2f}, "
                    f"新行={rf.new_rows}, 送OCR",
                    extra={"aligned_ratio": round(rf.aligned_ratio, 3),
                           "new_rows": rf.new_rows,
                           "total_diff": round(total_diff, 1),
                           "tier": "high_conf_new"})
                with self._cal_lock:
                    self._calibration_history.append((total_diff, "new"))
                return DedupResult(
                    verdict=DedupVerdict.NEW_CONTENT,
                    confidence_tier=ConfidenceTier.HIGH_CONF_NEW,
                    total_diff=total_diff
                ), gray
            # gray: 少量新行+位移混合 → 落入行差分阈值与旧条带dHash精筛兜底

        # 高置信度新内容: 行差分极大
        if total_diff > high_new_boundary:
            self._update_keyframe(gray)
            self._logger.log("visual_dedup", "high_conf_new",
                             f"行差分={total_diff:.1f} > {high_new_boundary:.1f}, "
                             f"高置信度新内容, 直接送OCR",
                             extra={"total_diff": round(total_diff, 1),
                                    "tier": "high_conf_new"})
            with self._cal_lock:
                self._calibration_history.append((total_diff, "new"))
            return DedupResult(
                verdict=DedupVerdict.NEW_CONTENT,
                confidence_tier=ConfidenceTier.HIGH_CONF_NEW,
                total_diff=total_diff
            ), gray

        # Step 2: 灰色地带精筛(条带dHash, 位移敏感, 兜底)。
        # RFSA已在Step 1.5执行(判dup/new直接返回; 判gray落至此)。
        # 此处仅当RFSA判gray或RFSA禁用时兜底, 用旧条带dHash估changed_ratio。
        curr_strips, curr_hashes = self._compute_strip_hashes(gray)
        strip_distances = [
            hamming_distance(curr_hashes[i], self._last_keyframe_hashes[i])
            for i in range(self._strip_count)
        ]
        changed_count = sum(1 for d in strip_distances if d > self._strip_threshold)
        changed_ratio = changed_count / self._strip_count

        # 灰色地带判定
        if changed_ratio >= self._changed_ratio:
            # 多数条带变化 → 倾向新内容, 但置信度不足 → GRAY_ZONE
            verdict = DedupVerdict.NEW_CONTENT
            tier = ConfidenceTier.GRAY_ZONE
            needs_llm = True
            self._update_keyframe(gray)
        elif changed_ratio <= (1.0 - self._changed_ratio):
            # 多数条带相同 → 倾向重复, 但置信度不足 → GRAY_ZONE
            verdict = DedupVerdict.DUPLICATE
            tier = ConfidenceTier.GRAY_ZONE
            needs_llm = True
        else:
            # 部分重叠 → OVERLAP, 送OCR处理
            verdict = DedupVerdict.OVERLAP
            tier = ConfidenceTier.GRAY_ZONE
            needs_llm = True
            self._update_keyframe(gray)

        self._logger.log("visual_dedup", "gray_zone",
                         f"灰色地带: diff={total_diff:.1f}, "
                         f"changed_strips={changed_count}/{self._strip_count}, "
                         f"ratio={changed_ratio:.2f}, verdict={verdict.value}, "
                         f"needs_llm={needs_llm}",
                         extra={"total_diff": round(total_diff, 1),
                                "changed_strips": changed_count,
                                "total_strips": self._strip_count,
                                "strip_distances": strip_distances,
                                "verdict": verdict.value,
                                "tier": tier.value,
                                "needs_llm": needs_llm})

        with self._cal_lock:
            self._calibration_history.append((total_diff, "gray"))
        return DedupResult(
            verdict=verdict,
            confidence_tier=tier,
            total_diff=total_diff,
            changed_strip_count=changed_count,
            total_strips=self._strip_count,
            strip_distances=strip_distances,
            needs_llm_arbitration=needs_llm
        ), gray

    def update_keyframe(self, roi_frame: np.ndarray) -> None:
        """更新关键帧缓存"""
        gray = self._prepare_gray(roi_frame)
        self._update_keyframe(gray)

    def _update_keyframe(self, gray: np.ndarray) -> None:
        """更新关键帧缓存(内部)"""
        strips, hashes = self._compute_strip_hashes(gray)
        self._last_keyframe_gray = gray
        self._last_keyframe_strips = strips
        self._last_keyframe_hashes = hashes
        # 同步RFSA关键帧(160宽灰度, 内部降采样幂等)
        if self._row_aligner is not None:
            self._row_aligner.update_keyframe(gray)

    def update_keyframe_with_gray(self, gray: np.ndarray) -> None:
        """
        用已降采样的灰度更新关键帧(供并行流水线的状态恢复使用)
        """
        self._update_keyframe(gray)

    def force_update_keyframe(self, roi_frame: np.ndarray) -> None:
        """强制更新关键帧(如状态恢复后)"""
        gray = self._prepare_gray(roi_frame)
        self._update_keyframe(gray)
        self._logger.log("visual_dedup", "force_update",
                         "强制更新关键帧 / Force updated keyframe")

    # ==========================================================================
    # 方案B: 在线阈值校准
    # ==========================================================================

    def calibrate_from_text_feedback(self, text_verdict: str) -> None:
        """
        根据OCR文本反馈校准视觉阈值

        方案B核心: 让文本模态的判定结果反向影响视觉模态的阈值。

        参数:
            text_verdict: 文本模态判定结果
                "text_dup" - OCR后发现文本重复(视觉阈值太松, 应收紧)
                "text_new" - OCR后发现文本不同(视觉阈值太紧, 应放宽)
                "text_same_as_corrected" - LLM纠错后判定为同一内容
        """
        if not self._calibration_enabled:
            return
        if not self._calibration_history:
            return

        # 取最近一条灰色地带记录的diff值
        recent = None
        with self._cal_lock:
            history = list(self._calibration_history)
        for diff, category in reversed(history):
            if category == "gray":
                recent = diff
                break
        if recent is None:
            return

        rate = self._calibration_rate

        if text_verdict == "text_dup":
            # OCR发现文本重复 → 视觉把"重复"判成了灰色地带 → 阈值太松
            # 应提高dup_thresh, 使类似diff值落入高置信度重复
            new_thresh = self._diff_dup_thresh + (recent - self._diff_dup_thresh) * rate
            new_thresh = max(self._calibration_min_dup, new_thresh)
            old = self._diff_dup_thresh
            self._diff_dup_thresh = new_thresh
            self._logger.log("visual_dedup", "calibrate",
                             f"方案B校准: 文本反馈=重复, "
                             f"dup_thresh {old:.0f} → {new_thresh:.0f} (收紧)",
                             extra={"text_verdict": text_verdict,
                                    "old_dup_thresh": round(old, 1),
                                    "new_dup_thresh": round(new_thresh, 1)})

        elif text_verdict == "text_new":
            # OCR发现文本不同 → 视觉把"新内容"判成了灰色地带 → 阈值太紧
            # 应降低new_thresh, 使类似diff值落入高置信度新内容
            new_thresh = self._diff_new_thresh - (self._diff_new_thresh - recent) * rate
            new_thresh = min(self._calibration_max_new, new_thresh)
            old = self._diff_new_thresh
            self._diff_new_thresh = new_thresh
            self._logger.log("visual_dedup", "calibrate",
                             f"方案B校准: 文本反馈=新内容, "
                             f"new_thresh {old:.0f} → {new_thresh:.0f} (放宽)",
                             extra={"text_verdict": text_verdict,
                                    "old_new_thresh": round(old, 1),
                                    "new_new_thresh": round(new_thresh, 1)})

    # ==========================================================================
    # 预热/探索阶段校准
    # ==========================================================================

    def finalize_calibration(self, pairs: List[Tuple[float, float]],
                             static_floor: float = 2.0,
                             dup_pct: float = 50.0,
                             new_pct: float = 50.0) -> None:
        """
        预热结束时用(diff, velocity)观测校准dup/new阈值

        由流水线在探索阶段结束后调用
        核心思想: 用帧间变化速度做弱标注, 把观测diff按"静止/变化"分桶,
        再取分位数作为阈值种子, 使diff阈值适应当前视频的分辨率与内容密度,
        后续再由方案B(文本反馈)持续漂移修正。
        """
        dup_diffs = sorted(d for d, v in pairs if v < static_floor)
        new_diffs = sorted(d for d, v in pairs if v >= static_floor)

        old_dup, old_new = self._diff_dup_thresh, self._diff_new_thresh

        if len(dup_diffs) >= 5:
            dup = percentile(dup_diffs, dup_pct)
            self._diff_dup_thresh = max(dup, self._calibration_min_dup)

        if len(new_diffs) >= 5:
            new = percentile(new_diffs, new_pct)
            new = max(new, self._diff_dup_thresh * 2.0)
            self._diff_new_thresh = min(new, self._calibration_max_new)

        self._logger.log("visual_dedup", "calibrate_warmup",
                         f"预热校准: dup {old_dup:.0f} → {self._diff_dup_thresh:.0f}, "
                         f"new {old_new:.0f} → {self._diff_new_thresh:.0f} "
                         f"(静止{len(dup_diffs)}条, 变化{len(new_diffs)}条)",
                         extra={"old_dup": round(old_dup, 1),
                                "new_dup": round(self._diff_dup_thresh, 1),
                                "old_new": round(old_new, 1),
                                "new_new": round(self._diff_new_thresh, 1),
                                "static_obs": len(dup_diffs),
                                "changed_obs": len(new_diffs)})

    @property
    def current_dup_threshold(self) -> float:
        return self._diff_dup_thresh

    @property
    def current_new_threshold(self) -> float:
        return self._diff_new_thresh

    @property
    def gray_zone_stats(self) -> dict:
        """获取灰色地带统计"""
        total = len(self._calibration_history)
        gray_count = sum(1 for _, c in self._calibration_history if c == "gray")
        dup_count = sum(1 for _, c in self._calibration_history if c == "dup")
        new_count = sum(1 for _, c in self._calibration_history if c == "new")
        return {
            "total": total,
            "high_conf_dup": dup_count,
            "high_conf_new": new_count,
            "gray_zone": gray_count,
            "gray_zone_ratio": gray_count / total if total > 0 else 0.0,
        }
