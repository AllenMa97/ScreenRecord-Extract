"""
行指纹序列对齐去重器

核心思想:
  滚动录屏是刚体垂直平移: 同一段文本在不同帧只发生y方向位移, 内容不变。
  把"水平切条带 → 每条带算指纹"看成把一帧编码成一个行指纹序列:
    滚动 = 序列的位移(offset)
    新消息 = 序列的插入(局部新增)
    静止 = 序列相同
  两帧比较退化为两个行指纹序列的对齐(带gap罚分的动态规划), 位移天然免疫,
  无需任何像素阈值。Comparing two frames becomes sequence alignment; shift

  替代旧的"行差分剖面sum + 条带dHash + changed_strip_ratio"灰色地带精筛:
    - 条带dHash位置绑定 → 滚动把同一行从条带3挪到条带2被误判为新内容。
    - 对齐允许位移 → 滚动/回弹直接判重复。Alignment allows shift; scrolls

参考文献:
  - 序列对齐: Needleman-Wunsch 全局比对算法 (1970), 生物信息学经典。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.utils.image_utils import to_gray, resize_frame, compute_dhash, hamming_distance
from src.utils.behavior_logger import BehaviorLogger


@dataclass
class RowAlignResult:
    """行对齐结果"""
    displacement: int = 0        # 序列整体位移(滚动距离, 指纹级)
    new_rows: List[int] = field(default_factory=list)  # 当前帧未对齐行索引
    aligned_ratio: float = 1.0   # 对齐行占比
    verdict: str = "dup"         # "dup" / "new" / "gray"
    total_rows: int = 0          # 当前帧行数


@dataclass
class RollingGateResult:
    """滚动一致性门控纯函数的结果"""
    substantive_new: List[int] = field(default_factory=list)
    disp_median: int = 0        # 主位移(匹配对位移中位数)
    main_ratio: float = 1.0     # 主位移占比
    over_ratio: float = 0.0     # 失真(超噪声阈)行占比
    rigid_roll: bool = False    # 是否判定为"刚性滚动"(触发去重门控)


def _classify_substantive_new(
    prev_seq: List[int], cur_seq: List[int],
    matched_pairs: List[Tuple[int, int]],
    *,
    noise_floor: float,
    rigid_roll_ratio: float,
    rigid_over_ratio: float,
) -> RollingGateResult:
    """位移免疫的实质新行判定(滚动去重核心, 位移/窗口免疫)。

    前提: matched_pairs 只含"干净匹配"(相似度>=match_floor 的真实同内容行),
    完全不同的行不会混入匹配(DP 匹配语义已在 _align 修复)。本函数据此:
      - 无干净锚点 → 无法稳健估计刚性位移 → 保守保留所有未对齐行(整屏翻页/替换)。
      - 有锚点 → 主位移占比高判"刚性滚动"; 主动滚动(|d|>0)时, 位移预测"带内"
        (0<=c-d<n)的未对齐行是被该位移重新呈现的已见行(滚动条带错位→dHash失真
        →没干净匹配), 属滚动拖影→抑制; 只有带外露出带才可能是真正新内容。滚动
        中被抑制的边缘新内容由后续 settle 静止帧兜底召回(pipeline 已保证)。
      - 非刚性 或 d==0(内容原位相同: 静止帧/原位插入)→ 不抑制, 保留召回。
    相比旧版(主位移占比+失真行占比双判据): 失真行占比判据已被 match_floor 的
    干净匹配天然取代——失真(半条带)难成干净匹配, 不会混入 matched_pairs。
    """
    n, m = len(prev_seq), len(cur_seq)
    matched_cur = {cj for _pi, cj in matched_pairs}
    if not matched_pairs:
        return RollingGateResult(
            substantive_new=[c for c in range(m)],
            disp_median=0, main_ratio=0.0, over_ratio=0.0, rigid_roll=False)

    shifts = [cj - pi for pi, cj in matched_pairs]
    disp_median = int(np.median(shifts))
    if len(shifts) >= 3:
        main_count = float(np.sum(np.abs(np.asarray(shifts) - disp_median) <= 1))
        main_ratio = main_count / len(shifts)
    else:
        main_ratio = 1.0
    # 最少需要3个干净锚点才信任"刚性滚动"判定, 防止1-2个偶然同构图元(如共享
    # 表头)骗取刚性判定而误抑真实新内容。
    rigid_shift = (len(matched_pairs) >= 3) and (main_ratio >= rigid_roll_ratio)

    if rigid_shift and disp_median != 0:
        substantive_new = [
            c for c in range(m)
            if c not in matched_cur and not (0 <= c - disp_median < n)
        ]
    else:
        substantive_new = [c for c in range(m) if c not in matched_cur]

    return RollingGateResult(
        substantive_new=substantive_new,
        disp_median=disp_median,
        main_ratio=main_ratio,
        over_ratio=0.0,
        rigid_roll=rigid_shift,
    )


class RowFingerprintAligner:
    """行指纹序列对齐去重器"""

    def __init__(self, config: dict, logger: BehaviorLogger):
        # 条带高度(px, 在降采样灰度上): 接近文本行高, 保证同一行落到同一
        # 条带内。Strip height (px, on downsampled gray): near text-line height.
        self._strip_height: int = config.get("rf_strip_height", 16)
        self._dhash_size: int = config.get("rf_dhash_size", 8)   # 8x8=64bit
        self._gap_penalty: float = config.get("rf_gap_penalty", -0.3)
        self._dup_ratio: float = config.get("rf_dup_ratio", 0.8)
        self._new_ratio: float = config.get("rf_new_ratio", 0.4)
        # 匹配行最低相似度(DP 对角匹配门槛): 相似度低于此值的两行视为"不构成
        # 匹配", 对角不可用→只能走gap→未对齐行进入新行候选。0.85≈要求 hd<=9:
        # 同内容(含滚动/条带噪声)hd<=6 通过, 完全不同内容 hd>=16 被拒。太低
        # (如0.6, hd<=25)会把完全不同的行也强行匹配, DP 择优走伪路径产生大量
        # 伪匹配/对齐率虚高。这是DP匹配语义的框架参数(非启发式超参)。
        # Min row similarity for a DP diagonal match: below this the pair is not
        # "a match" (diagonal unavailable -> becomes a gap/new-row candidate).
        # 0.85 ~ hd<=9: same content (with scroll/strip noise, hd<=6) passes,
        # fully-different content (hd>=16) is rejected. Too low (0.6 -> hd<=25)
        # force-matches unrelated rows into spurious alignments.
        self._match_floor: float = config.get("rf_match_floor", 0.85)
        # 实质新行的绝对噪声阈值(64bit汉明距离): 同内容滚动/静止的匹配行
        # hd通常<=6(压缩噪声), 完全不同文本hd>=16(实测"Message line 01" vs
        # 完全不同的中文文本 hd=24)。绝对阈值而非"匹配对中位数相对地板":
        # 两帧内容不同时hd普遍大, 中位数被抬高→地板被抬高→所有行都被判
        # 匹配→恒判dup(真实帧实测对齐率全为1.0的根因)。
        self._noise_floor: int = int(config.get("rf_noise_floor", 14))
        # 滚动一致性门控阈值: 匹配对位移主占比达此值视为刚性滚动(整屏刚体平移),
        # 此时"位移一致但dHash超阈"的行视为滚动失真而非新内容。滚动时同段文本
        # 落入条带边界被截断→dHash失真→hd超阈, 若不门控会把整屏滚动误判为批次
        # 新内容反复送OCR。位移分散(真翻页)时主占比低, 门控失效, 保留召回。
        self._rigid_roll_ratio: float = config.get("rf_rigid_roll_ratio", 0.7)
        # 滚动一致性门控的第二判据: 失真行占比上限。滚动=刚体平移、内容不变,
        # 仅少数因条带截断而失真(dHash超阈); 整屏替换/翻页时几乎所有行都超阈。
        # 仅当"位移主占比高 且 失真行占少数"同时成立才判定为滚动失真去重,
        # 防止把整屏内容替换误判为滚动而漏掉新内容。
        self._rigid_over_ratio: float = config.get("rf_rigid_over_ratio", 0.5)
        self._compute_w: int = config.get("compute_width", 160)
        # 相位归一化开关(框架级): 把条带网格平移到文本行对齐, 使同一行文本在任意
        # 滚动位移下指纹一致 → 滚动退化为整数条带位移 → DP 干净匹配恢复去重。
        # 0=关闭(退化为相位敏感的旧指纹, 真实滚动几乎失去去重能力)。
        # Phase normalization (framework): aligns the strip grid to text rows so the
        # same line has the same fingerprint at any scroll position; scroll reduces
        # to integer strip offsets and clean DP matching (dedup) is restored.
        self._phase_enabled: bool = bool(config.get("rf_phase_enabled", True))
        # 多关键帧历史长度: 滚动回看时旧行在上一关键帧中已滚出, 单关键帧
        # 无法识别"部分内容重现"; 维护最近N帧关键帧, compare与全部对齐取最优,
        # 重现的旧行能匹配更早的关键帧 → 判dup; 真正的新内容与所有历史都不
        # 匹配 → 判new。
        self._history_len: int = max(1, config.get("rf_history_len", 3))
        # 行指纹累积库(滚动记忆 / rolling memory): 持久记录"历史上见过并OCR过"
        # 的内容行指纹, 与关键帧窗口无关。滚动把内容滚出最近N帧窗口(回看/往返)
        # 时, 新关键帧无法匹配, 会把这些"旧内容重现"误判为全新而反复OCR; 累积库
        # 让新颖度变成"对全片记忆的判据"而非"对最近窗口的判据"——重现行在库中
        # 有近邻即视为回看, 位移/窗口免疫。库只在 pipeline 接受内容帧(update_
        # keyframe)时提交, 滚动失真/中间帧从不入库, 不会污染。
        # Persistent seen-row library (rolling memory): remembers every content row
        # fingerprint ever OCR'd, independent of the keyframe window. When scroll
        # pushes content out of the recent N-frame window (round-trips/revisits),
        # a fresh window can't match those rows and re-judges the revisit as new;
        # the library makes novelty a criterion on the whole-video memory rather
        # than the recent window. Rows are committed only when the pipeline accepts
        # a content frame (update_keyframe), so scroll smear never pollutes it.
        self._lib_enabled: bool = bool(config.get("rf_lib_enabled", True))
        self._lib_max: int = int(config.get("rf_lib_max", 2000))
        self._lib_match_hd: int = int(config.get("rf_lib_match_hd", 6))
        # 稀疏条带的"退化"墨水占比下限(暗像素占条带比例)。占比低于此值的条带
        # 几乎全是背景, 其 dHash 由压缩噪声而非文字决定 → 与库中任意背景行都能
        # 在模糊半径内碰撞(实测: 仅有一个 3 字短气泡的条带 ink=0.004, 与库行
        # hd=5 假碰撞), 使真正的新短消息被当成"回看"剔除 → 漏检。故退化条带
        # 只认**精确**命中(hd==0), 不做模糊近邻匹配。
        # Degenerate-strip ink floor: a strip below this dark-pixel ratio is almost
        # pure background, so its dHash is decided by compression noise, not glyphs,
        # and fuzzily collides with any background row in the library. Such strips
        # are matched exactly (hd==0) only.
        self._degenerate_ink: float = float(config.get("rf_degenerate_ink", 0.01))
        self._logger = logger

        # 关键帧指纹序列队列(最近N帧)
        self._keyframes: List[List[int]] = []
        # 最近一次关键帧序列(诊断/兼容)
        self._last_sequence: Optional[List[int]] = None
        # 最近一次对齐的滚动一致性(诊断/日志)
        self._last_main_ratio: float = 1.0
        self._last_rigid_roll: bool = False
        self._last_over_ratio: float = 0.0
        # 最近一次文本相位偏移(px, 诊断)
        self._last_phase: int = 0
        # 滚动记忆累积库: 已见内容行指纹集合(去重)
        self._seen_rows: set = set()

    # ------------------------------------------------------------------
    # 指纹提取
    # ------------------------------------------------------------------

    def _estimate_text_phase(self, small: np.ndarray) -> int:
        """估计文本的亚条带垂直相位(px): 使文本行尽可能落在条带内部而非被边界截断。

        滚动是刚体垂直平移, 但每次滚动位移是连续像素(常为亚条带的非整数倍)。
        固定条带网格下, 同一段文本滚动后落入不同相位: 文本行被条带边界截断 →
        该条的 dHash 与"完整文本行"完全不同 → DP 找不到干净锚点 → 整屏被误判为
        新内容反复 OCR。这是旧方案在真实滚动上几乎失去去重能力的根因。
        相位归一化: 每帧估计一个全局相位偏移, 把条带网格平移到与文本行对齐,
        使"同一行文本"在其占用的条带内保持相同相位 → dHash 位移不变 → 滚动退化为
        整数条带位移, 由 DP 对齐干净匹配, 去重恢复。真插入的新文本行仍无匹配 → 保留召回。
        Estimating an optimal phase keeps text rows inside strip interiors so dHash
        becomes (sub-strip) phase-invariant; scrolls reduce to integer strip offsets.
        """
        H = self._strip_height
        if H <= 1:
            return 0
        small_h = small.shape[0]
        # 文本近似为"暗像素": 聊天 UI 正文多为深色文字(无论浅/彩气泡背景)。
        # 用暗像素的逐行投影作为文本行活动剖面。
        gray8 = small
        mask = gray8 < 128
        profile = mask.sum(axis=1).astype(np.int64)
        if small_h <= H or profile.sum() == 0:
            return 0
        # 平滑去噪, 只保留真实文本行轮廓(去掉孤立噪声点)
        kernel = np.ones(3, dtype=np.float64) / 3.0
        prof = np.convolve(profile, kernel, mode="same")

        best_s, best_cost = 0, None
        for s in range(H):
            cost = 0
            b = s
            while b < small_h:
                lo = max(0, b - 1)
                hi = min(small_h, b + 2)
                cost += int(prof[lo:hi].sum())   # 落在边界±1内的暗像素 = 被截断风险
                b += H
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_s = s
        return best_s

    def _extract_sequence(self, roi_frame: np.ndarray
                          ) -> Tuple[List[int], List[float]]:
        """把帧编码为行指纹序列(相位归一化: 先估相位再切条带)

        同时返回每条带的墨水占比(暗像素占条带比例), 供滚动记忆库区分"有内容的
        条带"与"退化的近空条带"(见 _degenerate_ink)。
        / Returns (row fingerprints, per-strip dark-pixel ratios).
        """
        gray = to_gray(roi_frame)
        h, w = gray.shape[:2]
        scale = self._compute_w / w
        small = resize_frame(gray, self._compute_w, max(1, int(h * scale)))
        small_h = small.shape[0]
        H = self._strip_height
        phase = self._estimate_text_phase(small) if self._phase_enabled else 0
        self._last_phase = phase
        n = max(1, (small_h - phase) // H)
        seq: List[int] = []
        inks: List[float] = []
        for i in range(n):
            y0 = phase + i * H
            strip = small[y0: y0 + H]
            if strip.shape[0] < H:
                strip = np.pad(strip, ((0, H - strip.shape[0]), (0, 0)),
                               mode="edge")
            seq.append(compute_dhash(strip, self._dhash_size))
            inks.append(float(np.mean(strip < 128)))
        return seq, inks

    # ------------------------------------------------------------------
    # 滚动记忆累积库(位移/窗口免疫的新颖度判据)
    # ------------------------------------------------------------------

    def _row_seen_in_lib(self, fp: int, ink: float) -> bool:
        """该行指纹是否已在累积库中有近邻(滚动往返去重)。
        先做精确匹配(快路径), 再按噪声半径做模糊近邻(秒同内容微量扰动)。
        退化条带(ink < _degenerate_ink, 几乎无内容)只认精确匹配: 其 dHash 由
        噪声决定, 模糊近邻会把新短消息的背景条带误判成"已见"。
        / Degenerate (near-empty) strips match exactly only; their dHash carries
        no glyph signal and fuzzy matching there yields false "seen" collisions.
        """
        if fp in self._seen_rows:
            return True
        if ink < self._degenerate_ink:
            return False
        if not self._lib_match_hd or not self._seen_rows:
            return False
        r = self._lib_match_hd
        for lib_fp in self._seen_rows:
            if hamming_distance(fp, lib_fp) <= r:
                return True
        return False

    def _commit_subsequences(self, seq: List[int]) -> None:
        """把内容帧全部行指纹并入滚动记忆库(去重, 达上限后停止以防不可控膨胀)。
        只在 pipeline 接受内容帧时调用——滚动失真/中间帧不入库。
        """
        if not self._lib_enabled:
            return
        for fp in seq:
            if len(self._seen_rows) >= self._lib_max:
                return
            self._seen_rows.add(fp)

    def _filter_seen_rows(self, seq: List[int], inks: List[float],
                          new_rows: List[int]) -> Tuple[List[int], int]:
        """把"相对窗口是新的、但累积库中早已见过"的重现/回看行剔除。
        返回 (过滤后的新行, 剔除的复现行数)。这就是位移/窗口免疫的实质——
        新颖度对"全片记忆"判定, 而非只对最近几帧。
        """
        if not self._lib_enabled or not self._seen_rows:
            return new_rows, 0
        kept: List[int] = []
        dropped = 0
        for cj in new_rows:
            if self._row_seen_in_lib(seq[cj], inks[cj]):
                dropped += 1
            else:
                kept.append(cj)
        return kept, dropped

    # ------------------------------------------------------------------
    # 序列对齐(带gap罚分的Needleman-Wunsch)
    # ------------------------------------------------------------------

    def _align(self, prev_seq: List[int],
               cur_seq: List[int]) -> Tuple[float, int, List[int]]:
        """
        全局对齐两指纹序列

        返回:
            aligned_ratio: 匹配行数 / 当前帧行数 / matched / cur rows
            displacement: 匹配对中 (cur_idx - prev_idx) 的中位数
            new_rows: 当前帧中未对齐的行索引
        """
        n, m = len(prev_seq), len(cur_seq)
        # DP表: D[i][j] = 前i个prev与前j个cur对齐的最高得分
        neg_inf = -1e9
        floor = self._match_floor
        # 不可匹配的"对角"得分: 相似度低于 match_floor 的两行视为"不构成匹配"
        # (不等同于内容有差异被记为新行, 而是这次对齐根本不存在这条连线)。否则
        # 任意两行都会给正分(≈1-hd/64≈0.75)恒大于gap罚分(-0.3), DP会被迫把
        # 完全不同的行强行匹配、制造大量伪匹配, 使对齐率虚高/新行误判。这是DP
        # 匹配语义的框架修复: 只有"足够相似"的行才允许对角匹配。
        # Framework match-semantics fix: a diagonal match requires similarity
        # >= match_floor; otherwise any unrelated pair scores ~0.75 > gap and
        # the DP force-matches fully different rows into spurious alignments.
        D = np.full((n + 1, m + 1), neg_inf, dtype=np.float64)
        D[0, 0] = 0.0
        for i in range(1, n + 1):
            D[i, 0] = i * self._gap_penalty
        for j in range(1, m + 1):
            D[0, j] = j * self._gap_penalty

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sim = 1.0 - hamming_distance(prev_seq[i - 1],
                                             cur_seq[j - 1]) / 64.0
                diag_val = sim if sim >= floor else neg_inf
                D[i, j] = max(
                    D[i - 1, j - 1] + diag_val,
                    D[i - 1, j] + self._gap_penalty,
                    D[i, j - 1] + self._gap_penalty,
                )

        # 回溯
        i, j = n, m
        matched_pairs: List[Tuple[int, int]] = []   # (prev_idx, cur_idx)
        gap_cur_rows: List[int] = []                # cur-only 行(cur有prev无)
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                sim = 1.0 - hamming_distance(prev_seq[i - 1],
                                             cur_seq[j - 1]) / 64.0
                diag_val = sim if sim >= floor else neg_inf
                diag = D[i - 1, j - 1] + diag_val
                if abs(D[i, j] - diag) < 1e-9:
                    matched_pairs.append((i - 1, j - 1))
                    i -= 1
                    j -= 1
                    continue
            if i > 0 and abs(D[i, j] - (D[i - 1, j] + self._gap_penalty)) < 1e-9:
                i -= 1                      # prev有prev独有的行(滚动顶部滚出)
            else:
                gap_cur_rows.append(j - 1)  # cur有cur独有的行(新内容/滚入)
                j -= 1

        # 实质新行判定: 对齐匹配对中, 内容差异显著大于"同内容噪声"的行视为
        # 新内容。基准 = 绝对噪声阈值(同内容滚动/静止hd<=6, 完全不同hd>=16)。
        # 滚动一致性门控(Rolling-consistency gate): 滚动是刚体垂直平移, 同一
        # 段文本只发生位移, 内容不变。但滚动后文本行落入条带边界被截断, dHash
        # 因"半气泡+半背景"而失真→hd 超噪声阈, 被误判为"实质新行", 导致整屏
        # 滚动被当成批次新内容反复送OCR。门控的判据与结果见 module 级纯函数
        # classify_substantive_new, 这里直接使用。
        gate = _classify_substantive_new(
            prev_seq, cur_seq, matched_pairs, noise_floor=self._noise_floor,
            rigid_roll_ratio=self._rigid_roll_ratio,
            rigid_over_ratio=self._rigid_over_ratio)
        substantive_new = gate.substantive_new
        self._last_main_ratio = gate.main_ratio
        self._last_rigid_roll = gate.rigid_roll
        self._last_over_ratio = gate.over_ratio
        # 记录本次对齐的滚动一致性(供上级日志展示/诊断)
        main_ratio = gate.main_ratio
        over_ratio = gate.over_ratio

        # 实质新行 = 位移免疫门控筛出的未对齐行(带内拖影已抑制)。
        # 注意不能取 gap_cur_rows ∪ substantive_new: gap_cur_rows 已含全部未对齐
        # 行, 会把被门控抑制的带内拖影又加回来, 使抑制失效。
        new_rows = sorted(set(substantive_new))
        # 对齐率 = 实质匹配行 / 当前帧行数(重标记的新行不计为匹配), 归一到[0,1]
        # (无干净锚点时 substantive_new 可超过 matched_pairs, 对齐率为0)。
        rematched = max(0, len(matched_pairs) - len(substantive_new))
        aligned_ratio = rematched / max(1, m)

        # 位移 = 匹配对 (cur_idx - prev_idx) 的中位数
        if matched_pairs:
            shifts = [cj - pi for pi, cj in matched_pairs]
            displacement = int(np.median(shifts))
        else:
            displacement = 0
        return aligned_ratio, displacement, new_rows

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def compare(self, roi_frame: np.ndarray) -> RowAlignResult:
        """
        与最近N帧关键帧比较(取最优对齐)

        首帧视为关键帧并返回 dup(全匹配)。
        """
        seq, inks = self._extract_sequence(roi_frame)
        if not self._keyframes:
            self._keyframes.append(seq)
            self._last_sequence = seq
            return RowAlignResult(
                displacement=0, new_rows=[], aligned_ratio=1.0,
                verdict="dup", total_rows=len(seq))

        # 与所有历史关键帧对齐, 取 (对齐率, -新行数) 字典序最优: 回看重现
        # 能匹配更早关键帧(高对齐率), 新内容在所有历史上都低。
        best: Optional[Tuple[float, int, List[int]]] = None
        best_key: Optional[Tuple[float, int]] = None
        for kf in self._keyframes:
            aligned_ratio, displacement, new_rows = self._align(kf, seq)
            key = (aligned_ratio, -len(new_rows))
            if best_key is None or key > best_key:
                best_key = key
                best = (aligned_ratio, displacement, new_rows)
        aligned_ratio, displacement, new_rows = best

        # 滚动记忆过滤: 把"相对窗口是新、但累积库早已见过"的回看/复现行剔除。
        # 这使得新颖度对"全片记忆+位移对齐"判定, 滚动往返/长滚回看不再反复OCR。
        raw_new_count = len(new_rows)
        new_rows, dropped_revisit = self._filter_seen_rows(seq, inks, new_rows)

        # 判定: 有实质新行 → 至少送OCR(gray/new); 无新行且高对齐 → dup。
        if new_rows:
            if aligned_ratio <= self._new_ratio:
                verdict = "new"
            else:
                verdict = "gray"
        elif aligned_ratio >= self._dup_ratio:
            verdict = "dup"
        else:
            verdict = "gray"

        rigid_tag = f", 滚动一致性={self._last_rigid_roll}"
        self._logger.log(
            "row_fingerprint", "compare",
            f"对齐率={aligned_ratio:.2f}, 位移={displacement}, "
            f"新行={new_rows}, 复现剔除={dropped_revisit}, 判定={verdict} "
            f"(历史关键帧数={len(self._keyframes)}; 累积库={len(self._seen_rows)})"
            f"{rigid_tag}",
            extra={"aligned_ratio": round(aligned_ratio, 3),
                   "displacement": displacement,
                   "new_rows": new_rows,
                   "raw_new_count": raw_new_count,
                   "dropped_revisit": dropped_revisit,
                   "verdict": verdict,
                   "keyframes": len(self._keyframes),
                   "lib_size": len(self._seen_rows),
                   "rolling_consistent": self._last_rigid_roll,
                   "main_ratio": round(self._last_main_ratio, 3),
                   "over_ratio": round(self._last_over_ratio, 3)})
        return RowAlignResult(
            displacement=displacement, new_rows=new_rows,
            aligned_ratio=aligned_ratio, verdict=verdict,
            total_rows=len(seq))

    def update_keyframe(self, roi_frame: np.ndarray) -> None:
        """更新关键帧指纹序列(入队, 保留最近N帧), 并把内容行并入滚动记忆库"""
        seq, _inks = self._extract_sequence(roi_frame)
        # 与最近关键帧相同则跳过(重复关键帧无信息量)
        if not self._keyframes or seq != self._keyframes[-1]:
            self._keyframes.append(seq)
            if len(self._keyframes) > self._history_len:
                self._keyframes.pop(0)
        # 滚动记忆: 内容帧的全部行指纹入库(创造性帧才走此路径, 滚动中间帧不入库)
        self._commit_subsequences(seq)
        self._last_sequence = seq

    def force_update(self, roi_frame: np.ndarray) -> None:
        """强制更新关键帧(如状态恢复后): 清空窗口历史并重设, 保留滚动记忆库"""
        seq, _inks = self._extract_sequence(roi_frame)
        self._keyframes = [seq]
        self._commit_subsequences(seq)   # 恢复后的当前内容也纳入滚动记忆
        self._last_sequence = seq
        self._logger.log("row_fingerprint", "force_update",
                         "强制更新关键帧(清空历史) / Force updated keyframe (history cleared)")
