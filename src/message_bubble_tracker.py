"""
文本行跟踪器

核心创新: 用空间感知的逐行跟踪替代整帧 LCS 拼接。

问题背景:
  原方案将整帧OCR文本拼接为一个字符串再做Levenshtein比对, 存在:
  1. 单行OCR错误影响整体相似度
  2. LCS要求精确子序列匹配, OCR错一个字就断链
  3. 丢弃了bbox空间信息, 无法判断哪些行是新内容

解决方案:
  利用OCR返回的bbox坐标, 将每行文本作为独立实例进行跨帧跟踪:
  1. 匹配: bbox IoU + 中心距离 + 文本相似度组合判定
  2. 去重: 匹配成功的行视为已见内容, 不重复输出
  3. 拼接: 未匹配的行视为新内容, 按空间位置有序输出
  4. 纠错: 多帧投票 + 置信度加权纠正OCR错误

  此方法借鉴视频文本跟踪领域(Video Text Tracking)的思路,
  将文本实例在帧间的对应关系显式建模。

参考思路:
  - 使用appearance + geometry + text特征建立帧间对应关系
"""

import uuid
from typing import List, Optional

from src.bubble_merger import STAMP_RE
from src.data_types import TextLine, TrackedLine
from src.utils.text_utils import (
    bbox_iou, bbox_center_distance, text_similarity, normalize_text,
    template_digit_conflict
)
from src.utils.behavior_logger import BehaviorLogger


class MessageBubbleTracker:
    """空间感知的文本行跟踪器"""

    def __init__(self, config: dict, logger: BehaviorLogger):
        self._iou_threshold: float = config.get("iou_threshold", 0.4)
        self._text_sim_threshold: float = config.get("text_sim_threshold", 0.80)
        self._expire_seconds: float = config.get("line_expire_seconds", 3.0)
        self._voting_window: int = config.get("voting_window", 5)
        self._center_dist_ratio: float = config.get("center_distance_ratio", 0.15)
        # 强空间证据: bbox 几乎重合(同一框)时放宽文本门限, 兜住 OCR 噪声
        self._iou_strong: float = config.get("iou_strong_threshold", 0.6)
        self._text_sim_strong: float = config.get("text_sim_strong_threshold", 0.50)
        # 休眠行重挂门限: 休眠过久的行需强空间证据才可吸收观测(防身份顶替)
        self._dormant_frames: int = int(config.get("dormant_rematch_frames", 10))
        # 模板消息数字锚点守卫的骨架相似度门限(0 表示关闭)
        self._digit_guard_sim: float = float(
            config.get("template_digit_guard_sim", 0.90))
        # 过期行重挂宽限: 过期后该窗口内仍可匹配并复活
        self._reattach_grace: float = float(
            config.get("reattach_grace_seconds", self._expire_seconds))
        # 相邻行排他: 同列内垂直边距 < ratio×行高 的两行视为"相邻"; 一个观测只能
        # 归给其中空间最贴近的合格行。0 表示关闭。
        self._adjacent_gap: float = float(config.get("adjacent_gap_ratio", 1.0))
        self._logger = logger

        self._active_lines: List[TrackedLine] = []
        # 已滚动出屏而过期的行: 不丢弃, 保留到最终组装。
        # 滚动出屏的行仍是视频中出现的真实内容, 若过期即删会导致召回损失。
        self._completed_lines: List[TrackedLine] = []
        # 过期宽限行: 已过 expire 但仍在 reattach_grace 内, 可凭强空间证据重新匹配
        # (否则"短暂失配→过期→重建新行"会造成 ID 切换)。
        self._grace_lines: List[TrackedLine] = []
        self._roi_height: float = 1.0  # 由pipeline设置
        self._roi_width: float = 1.0

    def set_roi_dimensions(self, roi_height: int, roi_width: int) -> None:
        """设置ROI尺寸(用于中心距离归一化)"""
        self._roi_height = max(float(roi_height), 1.0)
        self._roi_width = max(float(roi_width), 1.0)

    def update(self, frame_id: int, timestamp: float,
               ocr_lines: List[TextLine],
               allow_new: bool = True) -> List[TextLine]:
        """
        用新帧的OCR结果更新跟踪器。

        参数:
            frame_id: 当前帧ID
            timestamp: 当前时间戳
            ocr_lines: 当前帧OCR识别的文本行列表
            allow_new: 是否允许未匹配行创建新跟踪行。滚动帧(内容残片/错字)置False,
                       只参与匹配投票, 不污染输出; 完整内容会由后续静止帧创建。
        帧内一一对应 / One-to-one within a frame:
            同一帧内, 一条跟踪行最多吸收一行, 一行最多归入一条跟踪行。否则"模板相似、
            位置相邻"的不同消息(如 SN0001/SN0002、"认证一/二/三"、"13:32/13:33")会被
            同一条跟踪行依次吞并, 造成严重漏召回。
        相邻行排他 / Adjacent-row exclusivity:
            同列内垂直边距 < adjacent_gap_ratio × 行高 的两条跟踪行视为"相邻",
            一个观测只能归给其中**空间最贴近**的合格行。噪声下相邻模板消息
            (SN0001/SN0002)对彼此的空间+文本都会过门限, 仅按评分取最大会把
            邻行的观测吞进自己簇里(误并 → 纯度下降); 排他仲裁按归属距离纠正。
        返回:
            新发现的文本行(未匹配到已有跟踪行)
        """
        # 过滤无效行
        valid_lines = [
            line for line in ocr_lines
            if line.text.strip() and line.bbox[2] > 0 and line.bbox[3] > 0
        ]
        for line in valid_lines:
            line.text = normalize_text(line.text)

        # 1) 为本帧每行找候选匹配(含综合评分), 标注优先级:
        #    pass1 = 活跃且非休眠行(身份仍然"热"的行优先认领);
        #    pass2 = 休眠行 + 宽限行(只拿 pass1 剩下的观测)。
        #    这样滚动中"另一条行刚好滚进休眠行原位置"时, 那条行自己的跟踪行
        #    (非休眠)先认领, 休眠行无法顶替他人身份 —— 直接消除跨帧身份误并。
        candidates = []  # (pass, score, line_idx, tracked, in_grace)
        for i, line in enumerate(valid_lines):
            found = self._find_match_scored(line, frame_id)
            if found is None:
                continue
            tracked, score, in_grace = found
            dormant = (frame_id - tracked.last_seen_frame) > self._dormant_frames
            prio = 1 if (in_grace or dormant) else 0
            candidates.append((prio, score, i, tracked, in_grace))

        # 2) 两段式全局贪心: 先 pass1 再 pass2, 段内按评分降序;
        #    行与跟踪行在整帧内都只能占用一次(帧内一一对应)。
        candidates.sort(key=lambda c: (c[0], -c[1]))
        assigned = {}          # line_idx -> (TrackedLine, in_grace)
        claimed = set()        # id(TrackedLine) 已被本帧占用
        for _prio, score, i, tracked, in_grace in candidates:
            if i in assigned or id(tracked) in claimed:
                continue
            assigned[i] = (tracked, in_grace)
            claimed.add(id(tracked))

        # 3) 相邻跟踪行排他仲裁: 一条观测被相邻行认领时, 若"相邻行组"里存在
        #    **更贴近且文本证据更强**的合格行, 观测改判给它。
        #    噪声下相邻模板消息(如 SN0001/SN0002)对彼此的空间+文本都会过门限,
        #    评分最高者未必是观测真正所属的那条行, 于是把邻行的观测吞进自己簇里
        #    (误并 → 纯度下降)。仲裁条件(两者同时成立才改判, 避免无谓抖动):
        #      a) 存在相邻、合格且**更贴近**该观测的行;
        #      b) 该行对这条观测的**文本证据更强**(外观文本库最大相似度更高)。
        #    只按距离改判会把观测交给"空间更近但文本不符"的行, 反而制造 ID 抖动,
        #    故用文本证据做锚。
        adj_moves = 0
        if self._adjacent_gap > 0:
            pool = self._candidate_pool()
            for i, line in enumerate(valid_lines):
                hit = assigned.get(i)
                if hit is None:
                    continue
                cur, _cur_grace = hit
                best_d = bbox_center_distance(line.bbox, cur.bbox)
                cur_ev = self._text_evidence(line, cur)
                alt = alt_grace = None
                for tracked, in_grace in pool:
                    if tracked is cur or id(tracked) in claimed:
                        continue
                    if not self._adjacent_rows(cur, tracked):
                        continue
                    if self._match_strength(line, tracked, in_grace,
                                            frame_id) is None:
                        continue
                    d = bbox_center_distance(line.bbox, tracked.bbox)
                    if d < best_d and self._text_evidence(line, tracked) > cur_ev:
                        best_d, alt, alt_grace = d, tracked, in_grace
                if alt is not None:
                    claimed.discard(id(cur))
                    claimed.add(id(alt))
                    assigned[i] = (alt, alt_grace)
                    adj_moves += 1

        new_lines: List[TextLine] = []
        matched_count = 0
        reattached = 0

        for i, line in enumerate(valid_lines):
            hit = assigned.get(i)
            if hit is not None:
                matched_line, in_grace = hit
                if in_grace:
                    # 过期宽限行重新匹配: 从宽限列表复活回活跃列表(复用同一
                    # line_id, 消除"短暂失配→过期→重建"造成的 ID 切换)
                    self._grace_lines.remove(matched_line)
                    self._active_lines.append(matched_line)
                    reattached += 1
                # 匹配成功: 更新跟踪行(去重, 不输出)
                prev_text = matched_line.text   # 投票前文本(纠错效果对比用)
                matched_line.update_with(line)
                matched_line.is_new = False
                matched_count += 1
                self._log_vote(matched_line, line, prev_text, frame_id)
            elif allow_new:
                # 未匹配: 新内容(仅允许新建的帧)
                tracked = TrackedLine(
                    line_id=uuid.uuid4().hex[:12],
                    text=line.text,
                    bbox=line.bbox,
                    first_bbox=tuple(line.bbox),
                    confidence=line.confidence,
                    first_seen_frame=frame_id,
                    last_seen_frame=frame_id,
                    first_timestamp=timestamp,
                    last_timestamp=timestamp,
                    observations=[line],
                    is_new=True
                )
                self._active_lines.append(tracked)
                new_lines.append(line)
            # else: 滚动帧未匹配行忽略(残片不输出)

        # 过期清理(活跃 → 宽限 → 完成)
        self._expire_lines(timestamp)

        self._logger.log("text_tracker", "update",
                         f"帧{frame_id}: OCR={len(valid_lines)}行, "
                         f"匹配={matched_count}, 新增={len(new_lines)}, "
                         f"重挂={reattached}, 邻行改判={adj_moves}, "
                         f"活跃={len(self._active_lines)}",
                         frame_id=frame_id,
                         extra={"ocr_lines": len(valid_lines),
                                "matched": matched_count,
                                "new": len(new_lines),
                                "reattached": reattached,
                                "adj_moves": adj_moves,
                                "active_tracked": len(self._active_lines)})

        return new_lines

    def _max_appearance_sim(self, text: str, tracked: TrackedLine,
                            stop_at: float) -> float:
        """外观文本库: 取"观测文本"与该行文本的最大相似度, 达 stop_at 即短路。

        用最近 min(voting_window, 观测数) 条观测的文本(去重) + 当前投票文本
        作为外观库。相比只比投票文本, 它对单帧噪声更稳 —— 库里总有一两条
        接近真值的观测, 使本行的正确观测不被拒(否则会逐帧新建成碎片行)。
        """
        best = text_similarity(text, tracked.text)
        if best >= stop_at:
            return best
        obs = tracked.observations[-max(1, self._voting_window):]
        seen = {tracked.text}
        for o in obs:
            if o.text in seen:
                continue
            seen.add(o.text)
            v = text_similarity(text, o.text)
            if v > best:
                best = v
                if best >= stop_at:
                    return best
        return best

    def _text_evidence(self, line: TextLine, tracked: TrackedLine) -> float:
        """观测与跟踪行的文本证据: 外观文本库的**最大**相似度(不设短路门限)。

        供相邻行排他仲裁比较"哪条相邻行与这条观测的文字更相符"。
        """
        return self._max_appearance_sim(line.text, tracked, 1.01)

    def _find_match_scored(self, line: TextLine, frame_id: int):
        """
        查找匹配, 返回 (最佳跟踪行, 综合评分, 是否为宽限行) 或 None。

        匹配条件(满足任一空间条件 AND 文本相似度):
          - OR 中心距离 < center_dist_ratio * roi_height
          - AND 文本相似度 > text_sim_threshold
              若 bbox IoU >= iou_strong_threshold(同一框), 文本门限放宽到
              text_sim_strong_threshold, 以兜住 OCR 噪声造成的同行拆散。

        休眠行防身份顶替:
          休眠(超 dormant_rematch_frames 帧未见)的跟踪行, 模板文本可能与滚动
          进入同一像素位置的**另一条**行相似(如模板消息 SN0002 vs SN0004),
          仅凭文本 + 中心距会跨帧顶替他人身份, 造成误并(纯度下降)。
          故休眠行需 IoU >= iou_strong 的强空间证据才可吸收观测。
          但消息高度统一时, 滚动整一条会让下一条消息 IoU≈1 地落进原框, 强空间
          证据也会失效; 对这类"同模板、数字锚点不同"的观测另设守卫直接拒绝
          (见 _match_strength 中的数字锚点守卫)。

        宽限行: 已过 expire 但仍在 reattach_grace 内, 同样需强空间证据,
                匹配成功后复用原 line_id 复活(消除 ID 切换)。

        时间戳行隔离(修复时间戳行帧内滚雪球吸收):
          聊天历史里每行消息上方的小时间戳(如"20:13")高度相似(差异仅分钟位)、
          且彼此垂直间距常小于宽松的中心距离阈值, 若不加隔离, 同一帧内
          十几个时间戳行会被第一条跟踪行依次吸收。隔离规则:
          - 时间戳行与正文行互不匹配(类型隔离);
          - 同帧内出现的时间戳行彼此独立(各自属于不同消息), 不合并。

        相邻行排他(见 update 第 3 步): 本方法给出"评分最高"的匹配, 但若同一帧
        内存在与它相邻且更贴近该观测的合格行, 该观测最终归那一条。
        """
        best_match: Optional[TrackedLine] = None
        best_score: float = 0.0
        best_grace: bool = False

        for tracked, in_grace in self._candidate_pool():
            score = self._match_strength(line, tracked, in_grace, frame_id)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_match = tracked
                best_grace = in_grace

        if best_match is None:
            return None
        return best_match, best_score, best_grace

    def _candidate_pool(self):
        """候选跟踪行池: 活跃行(身份仍"热") + 宽限行(可在宽限窗口内复活)。"""
        return [(t, False) for t in self._active_lines] + \
               [(t, True) for t in self._grace_lines]

    def _match_strength(self, line: TextLine, tracked: TrackedLine,
                        in_grace: bool, frame_id: int) -> Optional[float]:
        """单对(观测, 跟踪行)的匹配强度: 合格返回综合评分, 不合格返回 None。

        由 _find_match_scored(取评分最高者) 与相邻行排他仲裁(取空间最贴近者)
        共用, 保证两处的合格判据完全一致。
        """
        line_is_stamp = bool(STAMP_RE.fullmatch((line.text or "").strip()))
        tracked_is_stamp = bool(STAMP_RE.fullmatch((tracked.text or "").strip()))
        # 时间戳行与正文行互不匹配
        if line_is_stamp != tracked_is_stamp:
            return None
        # 同帧内的时间戳行各自独立, 不合并
        if line_is_stamp and tracked.last_seen_frame == frame_id:
            return None

        # 空间匹配
        iou = bbox_iou(line.bbox, tracked.bbox)
        center_dist = bbox_center_distance(line.bbox, tracked.bbox)
        center_threshold = self._center_dist_ratio * self._roi_height
        if not (iou > self._iou_threshold or center_dist < center_threshold):
            return None

        # 休眠行/宽限行: 需强空间证据(IoU 同框), 防止模板文本顶替他人身份。
        is_dormant = (not in_grace and
                      (frame_id - tracked.last_seen_frame) > self._dormant_frames)
        if (in_grace or is_dormant) and iou < self._iou_strong:
            return None

        # 休眠行/宽限行的数字锚点守卫: 强空间证据**不足以**证明身份。
        # 聊天里消息高度基本一致, 滚动整一条后**下一条消息**会精确落进上一条的框
        # (IoU≈1); 若两条又是同模板仅数字不同(订单SN0010 / 订单SN0014), 文本相似
        # 度也高达 0.91, 远超同框放宽门限(0.50) → 休眠行吸收它并**改判**, 使后出现
        # 的消息继承更早的 first_timestamp, 输出顺序颠倒(实测
        # `自行模拟对话记录4.mp4` 帧123/帧178 各一次, SN0014 被排到 SN0010 之前)。
        # 故再加一道判据: 观测与本行**外观库全部文本**都构成"同模板、数字不同"时,
        # 判为另一条消息, 拒绝吸收(让它自建新行, 拿到属于自己的时间戳)。
        # 只约束休眠/宽限行: 热行的数字差异更可能是 OCR 噪声, 应留给投票层吸收。
        if (in_grace or is_dormant) and self._digit_guard_sim > 0:
            appearance = [tracked.text]
            appearance += [o.text for o in
                           tracked.observations[-max(1, self._voting_window):]]
            appearance = [t for t in appearance if t]
            if appearance and all(
                    template_digit_conflict(line.text, t, self._digit_guard_sim)
                    for t in appearance):
                self._logger.log(
                    "text_tracker", "digit_guard",
                    f"帧{frame_id} 跟踪行{tracked.line_id[:8]}: 拒绝吸收"
                    f"'{line.text}'(与本行'{tracked.text}'同模板而数字锚点不同, "
                    f"IoU={iou:.2f}, 休眠={is_dormant}, 宽限={in_grace})",
                    frame_id=frame_id,
                    extra={"line_id": tracked.line_id, "obs_text": line.text,
                           "tracked_text": tracked.text, "iou": round(iou, 3),
                           "dormant": is_dormant, "in_grace": in_grace})
                return None

        # 文本匹配: 对跟踪行的**外观文本库**(近期观测文本)求最大相似度。
        # 只比"当前投票文本"会因投票被单帧噪声拖偏而拒掉本行的正确观测
        # (噪声下文本门限 0.8 常常过不去 → 每帧新建 → 碎片化)。
        # 观测库是外观特征: 它总含若干低噪声观测, 保持匹配稳定。
        sim_required = (self._text_sim_strong if iou >= self._iou_strong
                        else self._text_sim_threshold)
        sim = self._max_appearance_sim(line.text, tracked, sim_required)
        if sim < sim_required:
            return None

        # 综合评分: 空间+文本
        return iou * 0.4 + sim * 0.6

    def _adjacent_rows(self, a: TrackedLine, b: TrackedLine) -> bool:
        """两跟踪行是否"相邻": 同一列(水平有重叠) 且垂直边距 < ratio × 行高。

        用**边到边**垂直间距而非中心距: 相邻堆叠的消息行间距常常只有十几像素,
        而中心距已接近一个行高; 边距口径同时覆盖"紧邻堆叠的两条消息"与
        "滚动一行后顶到自己原位置"两种情况。_adjacent_gap <= 0 时约束关闭。
        """
        if self._adjacent_gap <= 0:
            return False
        ax, ay, aw, ah = a.bbox
        bx, by, bw, bh = b.bbox
        # 同一列: 水平投影需有重叠
        if min(ax + aw, bx + bw) - max(ax, bx) <= 0:
            return False
        gap = max(ay - (by + bh), by - (ay + ah))
        return gap < self._adjacent_gap * max(ah, bh)

    def _log_vote(self, tracked: TrackedLine, line: TextLine,
                  prev_text: str, frame_id: int) -> None:
        """
        时序一致性投票日志(纠错效果验证)。

        每条观测都写结构化行为日志(behavior_log.jsonl), 供离线核对;
        控制台仅打印纠错相关事件(投票结果变更 / 本次观测与投票结果不一致),
        便于实时验证多帧投票纠正OCR错误的效果。
        """
        # 票数统计
        counts: dict = {}
        for obs in tracked.observations:
            counts[obs.text] = counts.get(obs.text, 0) + 1
        changed = prev_text != tracked.text          # 投票结果被本次观测改变
        agrees = line.text == tracked.text           # 本次观测与当前投票一致
        conf = line.confidence

        # 结构化行为日志(全量)
        self._logger.log(
            "text_tracker", "vote",
            f"帧{frame_id} 跟踪行{tracked.line_id[:8]}: "
            f"观测'{line.text}' conf={conf:.2f} → 投票'{tracked.text}' "
            f"(变更前'{prev_text}') 观测数={len(tracked.observations)} "
            f"{'[结果变更]' if changed else ('[吸收噪声]' if not agrees else '')}",
            frame_id=frame_id,
            extra={
                "line_id": tracked.line_id,
                "obs_text": line.text,
                "obs_conf": round(float(conf), 3),
                "voted_text": tracked.text,
                "prev_text": prev_text,
                "changed": changed,
                "agrees": agrees,
                "obs_count": len(tracked.observations),
                "vote_counts": counts,
                "avg_conf": round(tracked.confidence, 3),
            })

        # 控制台: 仅纠错相关事件
        if changed:
            print(f"[投票] 帧{frame_id} 行{tracked.line_id[:8]}: "
                  f"结果变更 '{prev_text}' → '{tracked.text}' "
                  f"(票数{counts}, 观测{len(tracked.observations)}次, "
                  f"平均置信{tracked.confidence:.2f})", flush=True)
        elif not agrees:
            print(f"[投票] 帧{frame_id} 行{tracked.line_id[:8]}: "
                  f"吸收噪声观测 '{line.text}' (conf={conf:.2f}), "
                  f"投票保持 '{tracked.text}' 票数{counts}", flush=True)

    def _expire_lines(self, current_timestamp: float) -> None:
        """过期清理分两级: 活跃 → 宽限 → 完成(不丢弃)。

        1) 活跃行超 expire_seconds → 宽限期(_grace_lines):
           仍可在宽限窗口内凭强空间证据重新匹配并复用原 line_id 复活,
           消除"短暂失配 → 过期 → 重建新行"造成的 ID 切换。
        2) 宽限行超 reattach_grace_seconds → 完成列表(_completed_lines):
           保留到最终组装, 因为滚动出屏的行仍是视频中出现的真实内容。
        """
        kept = []
        to_grace = []
        for line in self._active_lines:
            if current_timestamp - line.last_timestamp <= self._expire_seconds:
                kept.append(line)
            else:
                to_grace.append(line)
        self._active_lines = kept

        kept_g = []
        to_done = []
        # 宽限退出线: 过期后的再宽限时长, 必须 > 进入线(expire), 否则宽限期为 0
        grace_deadline = self._expire_seconds + max(0.0, self._reattach_grace)
        for line in self._grace_lines:
            if current_timestamp - line.last_timestamp <= grace_deadline:
                kept_g.append(line)
            else:
                to_done.append(line)
        self._grace_lines = kept_g + to_grace

        self._completed_lines.extend(to_done)

        if to_grace or to_done:
            self._logger.log("text_tracker", "expire",
                             f"过期: 活跃→宽限 {len(to_grace)}行, "
                             f"宽限→完成 {len(to_done)}行, "
                             f"活跃{len(self._active_lines)}行, "
                             f"宽限{len(self._grace_lines)}行, "
                             f"完成{len(self._completed_lines)}行",
                             extra={"to_grace": len(to_grace),
                                    "to_completed": len(to_done),
                                    "active": len(self._active_lines),
                                    "grace": len(self._grace_lines),
                                    "completed": len(self._completed_lines)})

    def get_all_tracked(self) -> List[TrackedLine]:
        """获取所有跟踪行(活跃+已完成, 按首次出现时间+首次位置排序)。

        排序用首次bbox而非最新bbox: 滚动会把首屏各行更新到不同位置, 用最新
        bbox会让同屏共享同一first_timestamp的行乱序, 破坏阅读顺序与气泡合并。
        """
        all_lines = self._active_lines + self._grace_lines + self._completed_lines
        return sorted(all_lines,
                      key=lambda x: (x.first_timestamp,
                                     (x.first_bbox or x.bbox)[1]))

    def get_new_lines(self) -> List[TrackedLine]:
        """获取标记为新内容的跟踪行"""
        return [line for line in self._active_lines if line.is_new]

    def compute_overlap_ratio(self, current_lines: List[TextLine],
                              previous_lines: List[TextLine]) -> float:
        """
        计算当前帧与上一帧的文本行重叠率。

        用于检测内容缺口: 重叠率低表示快速滑动导致内容断裂。
        """
        if not previous_lines:
            return 0.0 if current_lines else 1.0

        matched = 0
        for curr in current_lines:
            for prev in previous_lines:
                iou = bbox_iou(curr.bbox, prev.bbox)
                sim = text_similarity(curr.text, prev.text)
                if iou > self._iou_threshold or sim > 0.85:
                    matched += 1
                    break

        total = max(len(current_lines), len(previous_lines))
        return matched / total if total > 0 else 1.0
