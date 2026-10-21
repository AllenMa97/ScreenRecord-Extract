"""
语音气泡扫描器 (Voice-Bubble Scanner)

对话记录屏做一次**独立的后处理第二遍扫描**: 用 cv2 重开视频, 按时间抽样帧,
对 ROI 区域跑轻量气泡检测, 归并出"每个语音气泡何时出现在屏幕上"的事件
(含方向 '发'/'收'), 作为把音频转写配对回聊天记录的时间锚点。
方向由视觉层按气泡**水平位置**(相对会话区中线: 右侧'发'/左侧'收')判定, 颜色
仅作辅助, 故非绿气泡也能得到方向。

与 OCR producer 解耦: 不触碰既有采样/去重/OCR 流程, 低风险; 抽样(默认每
0.2s 一帧)成本低。
"""
from __future__ import annotations

import bisect
import logging
import os
import statistics
from collections import Counter
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .data_types import VoiceEvent
from src.visual.scroll_motion import estimate_step_dy
from src.visual.voice_bubble_detector import VoiceBubble, detect_voice_bubbles

logger = logging.getLogger(__name__)

ROI_TYPE = Tuple[int, int, int, int]  # (x, y, w, h), 原始帧坐标

# 跨帧匹配阈值 / Cross-frame matching thresholds
IOU_MATCH = 0.35            # 强匹配(IoU)阈值: 静止/小幅抖动时的主要路径
SCROLL_SIZE_TOL = 0.25      # 平移兜底: 宽/高相对容差(与 _dedup_events 同口径)
SCROLL_CX_TOL = 0.15        # 平移兜底: 中心x 相对容差
SCROLL_MAX_DY = 2.5         # 平移兜底: 纵向位移上限(以气泡高度归一)。
#   追加式滚动一帧的位移≈最新消息行高(最大约 2.3 倍语音气泡高), 取 2.5 容纳之;
#   同时挡住"自身气泡已滚出屏幕的陈旧状态"去认领屏幕另一端的新气泡(位移可达
#   十几倍气泡高), 否则新气泡不发事件、其音频会配到旧气泡的收发方向上。
SCROLL_RESID_TOL = 0.5      # 平移兜底: 与"全局平移估计"的残差上限(以气泡高度归一)
SCROLL_SHIFT_CLUSTER_PX = 12   # 全局平移估计: Δy 聚类容差(像素)
SCROLL_SHIFT_MIN_VOTES = 3     # 全局平移估计: 最少票数, 不足则视为"无滚动"

# 同一采样帧内检测到的多个候选**必然是不同物理气泡**, 不得按几何签名合并。
# / Candidates detected in the SAME sample frame are necessarily distinct bubbles.
SAME_FRAME_EPS_S = 1e-6
SAME_FRAME_IOU_MERGE = 0.5     # 例外: 同帧且框几乎重合 → 同一气泡的重复框

# "屏幕静止"判定: 相邻采样帧做相位相关(图像侧帧级纵向运动), 位移≈0 且响应足够
# 高 → 该帧为静止帧。只有静止帧上的气泡 y 才与滚动偏移曲线共基准; 滚动途中的
# 观测会与曲线错位一整步滚动量(实测单步可达 300px)。
# / Static-frame detection via phase correlation; only static-frame y values share
# a baseline with the scroll-offset curve.
PC_STATIC_SKIP_TOP = 90        # 跳过顶部固定叠加层(诈骗横幅/状态栏), 它们不随内容滚动
PC_STATIC_MAX_DY = 1.0         # 缩放后像素; 超过即视为滚动中
PC_STATIC_MIN_RESP = 0.55      # 相位相关响应下限
PC_DOWNSCALE = 2               # 相位相关降采样倍数(速度优先, 精度足够判静止)


def probe_media_duration(video_path: str) -> Optional[float]:
    """用 cv2 读视频总时长(秒), 供时长标签的可行性约束使用; 失败返回 None。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()
    if fps <= 0 or frames <= 0:
        return None
    return frames / fps


class VoiceBubbleScanner:
    """对视频做抽样扫描, 产生语音气泡"出现事件"时间线。"""

    def __init__(self, roi_box: ROI_TYPE, voice_threshold: float = 0.55,
                 sample_interval_s: float = 0.2, min_bubble_ms: int = 0,
                 expire_s: float = 2.0, merge_window_s: float = 1.2):
        self._roi_box = roi_box
        self._voice_threshold = voice_threshold
        self._interval = sample_interval_s
        self._min_bubble_ms = min_bubble_ms
        self._expire_s = expire_s
        self._merge_window_s = merge_window_s
        self._active: List[VoiceBubbleState] = []
        # 静止帧上的**逐帧原始检测**(不做跨帧身份合并): [(t, box, side, voice_score,
        # hue_std)]。跨帧身份跟踪在快速滚动时会把"上一只已移出屏的气泡"错认成
        # "新进入屏的另一只"(全局平移兜底), 使单个事件的 y 观测横跨两只气泡;
        # 而文档坐标排序只需要"某只气泡在某一静止屏上的位置", 用原始观测按文档
        # 坐标聚类反而更稳(同一物理气泡在多屏上的文档坐标基本重合)。
        self.static_observations: List[dict] = []
        # 图像侧累计滚动曲线 [(t, offset)]: 用相邻采样帧的多横带模板匹配估计
        # 纵向位移并累计(原始 ROI 像素)。OCR 公共文本行在缓存稀疏时会漏记整段
        # 滚动(见 `report_merge.build_scroll_curve`), 该曲线作为其可信度兜底。
        self.scroll_curve: List[Tuple[float, float]] = []

    def scan(self, video_path: str, progress=None) -> List[VoiceEvent]:
        """扫描视频, 返回语音气泡出现事件(list[VoiceEvent], 按时间排序)。

        同一气泡持续多帧: 只在其"首次出现"或"方向变化"时发射事件,
        避免一帧一个事件刷屏。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("无法打开视频做气泡扫描: %s", video_path)
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_skip = max(1, int(round(self._interval * fps)))
        events: List[VoiceEvent] = []
        self._active = []
        self.static_observations = []
        self.scroll_curve = []
        scroll_offset = 0.0
        x, y, w, h = self._roi_box
        if w <= 0 or h <= 0:
            cap.release()
            return []

        fid = 0
        prev_gray = None
        pc_win = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fid % frame_skip != 0:
                fid += 1
                continue
            ts = fid / fps
            roi_bgr = frame[y:y + h, x:x + w]
            # 屏幕是否静止: 相位相关给出帧级纵向运动, 只用来**分类**当前帧是否
            # 在滚动(静止帧的气泡 y 才与滚动曲线共基准)。累计位移不走这条路,
            # 快速滚动下它会混叠(实测少记 270px), 仍由 OCR 公共文本行给出。
            is_static = True
            chat = roi_bgr[PC_STATIC_SKIP_TOP:, :]
            if chat.size:
                g = cv2.cvtColor(chat, cv2.COLOR_BGR2GRAY)
                if PC_DOWNSCALE > 1:
                    g = cv2.resize(g, None, fx=1.0 / PC_DOWNSCALE,
                                   fy=1.0 / PC_DOWNSCALE,
                                   interpolation=cv2.INTER_AREA)
                g = g.astype(np.float32)
                g -= g.mean()
                if prev_gray is not None and prev_gray.shape == g.shape:
                    if pc_win is None:
                        pc_win = cv2.createHanningWindow(
                            (g.shape[1], g.shape[0]), cv2.CV_32F)
                    (_dx, dy), resp = cv2.phaseCorrelate(prev_gray, g, pc_win)
                    is_static = (resp >= PC_STATIC_MIN_RESP
                                 and abs(dy) <= PC_STATIC_MAX_DY)
                    # 图像侧滚动信号: 相位相关只够判静止/滚动, 累计位移由多横带
                    # 模板匹配给出(对气泡动画/局部变化更稳)。静止帧位移≈0, 跳过
                    # 匹配以省算力; 逐步累计到原始 ROI 像素。
                    if not is_static:
                        step = estimate_step_dy(prev_gray, g)
                        if step is not None:
                            scroll_offset += -float(step) * PC_DOWNSCALE
                prev_gray = g
            self.scroll_curve.append((ts, scroll_offset))
            bubbles = detect_voice_bubbles(roi_bgr)
            # 投影回原始帧坐标, 过滤低 voice_score
            cands = []
            for b in bubbles:
                if b.voice_score < self._voice_threshold:
                    continue
                if is_static:
                    # 静止帧的原始观测(不参与跨帧身份合并): 只保留**完整可见**的框,
                    # 贴边截断的框高度失真, 其 y 无法换算成可靠的文档坐标。
                    if b.y >= 2 and b.y + b.h <= h - 2:
                        self.static_observations.append({
                            "t": ts,
                            "box": (float(x + b.x), float(y + b.y),
                                    float(b.w), float(b.h)),
                            "side": b.side,
                            "voice_score": float(b.voice_score),
                            "confidence": float(b.confidence),
                        })
                bx = VoiceBubble(x + b.x, y + b.y, b.w, b.h, b.side,
                                 b.confidence, b.voice_score)
                cands.append(bx)
            self._update_active(cands, ts, events, progress, is_static=is_static)
            fid += 1
            if total and fid > total:
                break
        cap.release()
        events.sort(key=lambda e: e.timestamp)
        return self._dedup_events(events, merge_window_s=self._merge_window_s)

    def _dedup_events(self, events: List[VoiceEvent],
                      merge_window_s: float = 1.2,
                      size_tol: float = 0.25,
                      cx_tol: float = 0.15) -> List[VoiceEvent]:
        """合并"同一物理语音气泡被重复发射"的事件簇。

        现象: 气泡内波形随播放/轻微滚动而略有变化, 导致相邻采样帧(默认0.2s)
        的IoU匹配偶发失败 → 同一气泡被当作新气泡重复发射, 时间戳间隔≈采样周期。
        判定: 时间窗内 **几何签名一致** 的事件视为同一个气泡, 不再要求侧别相同。

        侧别翻转容忍(side-flip tolerance):
          视觉层方向现以气泡水平位置为主判据(右侧'发'/左侧'收'), 同一气泡相邻
          帧通常不会再翻转; 但气泡内波形随播放/轻微滚动变化时, 检测框仍可能轻微
          抖动。旧逻辑要求同侧别才合并, 一旦侧别偶发翻转, 同一句语音会被发射成
          2~3 条事件(如 t=0.0[收] / 0.2[发] / 0.4[发])。这里改为只按几何判定,
          侧别在合并后按出现次数表决(见下), 从根上消除翻转造成的重复。
          安全性: 收发方向本质由气泡的水平位置决定(发靠右/收靠左), 真不同的
          左右气泡中心x 相差极大, 会被 cx_tol 拒绝, 不会被误并。

        为何用"尺寸+中心x"而非位置IoU: 气泡在滚动时 y 坐标随之整体平移,
        但同一气泡的宽/高/中心x 保持稳定。若按IoU合并, 滚动一下 y 错位,
        IoU 骤降 → 被误判成新气泡(过度分裂); 若只按"同侧+时间"(旧逻辑),
        会把同侧紧邻的两条不同语音误并(召回丢失)。几何签名兼顾两者。

        链式合并: 时间窗以簇内"最近一次成员时刻"为基准(而非首次时刻), 使
        持续翻转/抖动的同一气泡能被连续并入同一簇, 不会因首次时刻过远而
        在窗口外再开一个新簇。

        同帧互斥(关键约束): 同一采样帧内被检测到的两个候选**不可能是同一个
        物理气泡**, 因此即使宽/高/中心x 完全一致也不得合并。否则"预置静态聊
        天"录屏首帧里同列同宽的 4 只气泡会被压成 1 只, 事件数锐减, 后续顺序
        对齐整体错位(实测 `自行模拟对话记录3.mp4`: 首帧 7 只 → 误并为 2 只)。
        / Same-frame mutual exclusion: two candidates seen in one frame are
        distinct bubbles and must never be merged by the geometry signature.
        """
        COLOR_W, COLOR_H = 890, 1920   # ROI原始坐标下整屏尺寸(用于过滤状态栏条)
        real = [e for e in events
                if not (e.box and e.box[2] >= COLOR_W * 0.75
                        and e.box[3] < COLOR_H * 0.25)]  # 剔除整宽状态栏/输入条
        kept: List[VoiceEvent] = []
        last_ts: List[float] = []      # 各簇最近成员时刻(链式合并基准)
        votes: List[Counter] = []      # 各簇侧别票数(翻转下按多数裁决)
        for ev in real:
            merged = False
            for i, k in enumerate(kept):
                if ev.timestamp - last_ts[i] > merge_window_s:
                    continue
                # 同帧互斥: 同一采样帧的两个候选是不同气泡, 除非框几乎重合
                if (abs(ev.timestamp - k.timestamp) < SAME_FRAME_EPS_S
                        and k.box and ev.box
                        and _iou(k.box, ev.box) < SAME_FRAME_IOU_MERGE):
                    continue
                if not _same_bubble_geom(k, ev, size_tol, cx_tol):
                    continue
                # 同一气泡(时间近 + 几何签名同): 仅延长消亡时间, 不新增事件
                k.end_time = max(k.end_time or k.timestamp,
                                 ev.end_time if ev.end_time is not None
                                 else ev.timestamp)
                # 首次发射保留; 置信度取更高者
                k.confidence = max(k.confidence, ev.confidence)
                if not k.matched_text:
                    k.matched_text = ev.matched_text
                last_ts[i] = max(last_ts[i], ev.timestamp)
                if ev.side:
                    votes[i][ev.side] += 1
                merged = True
                break
            if not merged:
                kept.append(ev)
                last_ts.append(ev.timestamp)
                vote = Counter()
                if ev.side:
                    vote[ev.side] += 1
                votes.append(vote)
        # 侧别表决: 同一气泡的掩码/位置翻转按出现次数取多数; 平票时优先 '发'。
        # (方向已改由水平位置主判, 翻转极少; 此表决仅为兼容旧行为的兜底。)
        for k, vote in zip(kept, votes):
            if not vote:
                continue
            top = max(vote.values())
            for side in ("发", "收"):
                if vote.get(side, 0) == top:
                    k.side = side
                    break
        kept.sort(key=lambda e: e.timestamp)
        return kept

    # -- 内部状态机: 把"每帧检测到的气泡集合"变成"事件流" --

    def _update_active(self, cands, ts, events, progress, is_static: bool = True):
        # 两阶段跨帧匹配, 把"每帧检测到的气泡集合"接到既有活跃状态上。
        #
        # 为什么需要两阶段 / Why two stages:
        #   聊天是**追加式滚动**——新消息到达时视图整体上移, 气泡在相邻采样帧之间
        #   的纵向位移常超过自身高度, 使位置 IoU 归零。若只用 IoU 匹配, 同一个气泡
        #   会被当作"新气泡"重复发射(每个新消息一行), 这些重复事件按时间排序后可能
        #   排在真正新气泡之前, 把本该属于新气泡的音频段抢走 → 收发方向错乱。
        #   故保留 IoU 强匹配, 再用"尺寸+中心x"平移不变签名兜底(滚动只动 y)。
        #   / Append-style scrolling shifts bubbles vertically by more than their own
        #   height, dropping IoU to zero; without a scroll-tolerant fallback the same
        #   bubble is re-emitted as new and steals the audio pairing of the real one.
        cand_used = [False] * len(cands)
        st_used = [False] * len(self._active)

        # 阶段1: IoU 强匹配(静止/小幅抖动的主路径, 保持既有行为)
        for si, st in enumerate(self._active):
            best = -1
            best_iou = IOU_MATCH
            for i, c in enumerate(cands):
                if cand_used[i]:
                    continue
                iou = _iou(st.last_box, (c.x, c.y, c.w, c.h))
                if iou > best_iou:
                    best_iou = iou
                    best = i
            if best >= 0:
                cand_used[best] = True
                st_used[si] = True
                st.update(cands[best], ts)
                st.touch_event(cands[best], ts, is_static)

        # 阶段2: 平移兜底匹配 —— 先估计"帧级全局纵向平移", 再在该平移下做一对一匹配。
        #
        # 为什么不能直接按 |Δy| 最小匹配 / Why not plain nearest-|Δy| matching:
        #   追加式滚动会让**整屏气泡同步平移同一像素量**。当一帧的平移量≈气泡行间距
        #   (实测一次快速滚动为 ~60px, 而语音气泡高 44px、行距也 ~60px)时, 某只气泡
        #   的**新位置**会比其**旧位置**更靠近"隔壁气泡的旧位置", 于是按 |Δy| 最小贪心
        #   配对会把同侧一串气泡互相错认(实测一次滚动使 4 只气泡错配), 这些错配把旧
        #   状态顶掉后又把真气泡当"新生"重复发射, 重复事件按时间排序抢走新消息的音频。
        #   改为: 用兼容配对的 Δy 众数(聚类)估计全局平移 g, 再按 |Δy - g| 最小配对;
        #   静止帧 g≈0, 行为与旧的 |Δy| 最小一致, 故为纯增量。
        #   / Estimate the frame-level global vertical shift g (mode of compatible Δy's),
        #   then match by |Δy - g| — nearest-|Δy| is ambiguous when the shift ≈ row pitch.
        pairs = []
        for si, st in enumerate(self._active):
            if st_used[si]:
                continue
            ah = st.last_box[3]
            for i, c in enumerate(cands):
                if cand_used[i]:
                    continue
                dy_px = _scroll_delta_px(st.last_box, (c.x, c.y, c.w, c.h))
                if dy_px is None:
                    continue
                if abs(dy_px) / max(ah, c.h, 1) > SCROLL_MAX_DY:
                    continue
                pairs.append((dy_px, si, i))
        shift = _estimate_scroll_shift(pairs)
        if shift is None:
            shift = 0.0
        pairs.sort(key=lambda p: abs(p[0] - shift))
        for dy_px, si, i in pairs:
            if st_used[si] or cand_used[i]:
                continue
            h = max(self._active[si].last_box[3], cands[i].h, 1)
            if abs(dy_px - shift) > SCROLL_RESID_TOL * h:
                break   # 已按残差升序, 后者只会更大
            st_used[si] = True
            cand_used[i] = True
            self._active[si].update(cands[i], ts)
            self._active[si].touch_event(cands[i], ts, is_static)

        # 阶段3: 仍未匹配的候选 → 新出现的活跃气泡, 发事件
        for i, c in enumerate(cands):
            if cand_used[i]:
                continue
            st = VoiceBubbleState(c)
            # 记录首见时刻, 使"此后从未再被匹配到"的状态也能按 expire_s 正常消亡;
            # 否则 last_ts 恒为 None 的状态永不超时, 会长期滞留并认领远处的新气泡。
            st.first_ts = st.last_ts = ts
            ev = VoiceEvent(timestamp=ts, end_time=ts, side=c.side,
                            confidence=c.confidence, frame_id=int(ts * 30),
                            box=(c.x, c.y, c.w, c.h),
                            last_timestamp=ts, last_box=(c.x, c.y, c.w, c.h),
                            stable_timestamp=(ts if is_static else None),
                            stable_box=((c.x, c.y, c.w, c.h) if is_static else None))
            st.event_ref = ev
            self._active.append(st)
            events.append(ev)
        # L2: 剔除超时未再命中的气泡状态(防止长视频下活跃列表无限增长导致
        # 旧气泡与新气泡高IoU误匹配、事件重复; 并在消亡时补记 end_time)
        expired = [st for st in self._active
                   if st.last_ts is not None and ts - st.last_ts > self._expire_s]
        for st in expired:
            for ev in events:
                if ev is st.event_ref:
                    ev.end_time = st.last_ts
        self._active = [st for st in self._active if st not in expired]
        if progress:
            try:
                progress(fid=len(cands), ts=ts)
            except Exception:
                pass


class VoiceBubbleState:
    """单个活跃语音气泡的多帧观察状态。"""

    def __init__(self, bubble: VoiceBubble):
        self.side = bubble.side
        self.last_box = (bubble.x, bubble.y, bubble.w, bubble.h)
        self.confidence = bubble.confidence
        self.first_ts = self.last_ts = None
        self.seen_frames = 0
        self.event_ref = None   # 关联的 VoiceEvent, 用于消亡时补记 end_time

    def update(self, bubble: VoiceBubble, ts: float):
        if self.first_ts is None:
            self.first_ts = ts
        self.last_ts = ts
        self.last_box = (bubble.x, bubble.y, bubble.w, bubble.h)
        self.side = bubble.side or self.side
        self.seen_frames += 1

    def touch_event(self, bubble: VoiceBubble, ts: float, is_static: bool = True):
        """把最近一次观测同步到事件上(供"稳定位置"排序使用)。

        `stable_*` 只记录**第一次静止帧观测**: 滚动途中的 y 与滚动偏移曲线错位
        一整步滚动量, 不能用于跨屏比较; 而首次静止观测离气泡上屏时刻最近,
        最能代表它在聊天里的位置。
        """
        ev = self.event_ref
        if ev is None:
            return
        ev.last_timestamp = ts
        ev.last_box = (bubble.x, bubble.y, bubble.w, bubble.h)
        if is_static and ev.stable_timestamp is None:
            ev.stable_timestamp = ts
            ev.stable_box = (bubble.x, bubble.y, bubble.w, bubble.h)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _parse_duration_seconds(text: str) -> Optional[float]:
    """把 OCR 时长标签(如 "00:03" / "1:05")解析成秒数; 非时长标签返回 None。

    统一走 ocr_voice_scanner 的解析(内含 ≤60s 语音上限领域约束)。
    """
    from .ocr_voice_scanner import parse_duration_seconds
    return parse_duration_seconds(text)


def match_duration_label(ev: VoiceEvent, hits: List["DurationHit"],
                         pad_s: float = 2.5,
                         x_tol: float = 0.05,
                         y_tol: float = 1.20,
                         media_duration_s: Optional[float] = None
                         ) -> Optional[float]:
    """在一个视觉气泡的**生命周期窗口 + 框邻域**内, 众数投票选出时长标签。

    为什么不用 IoU: 时长标签是气泡内的一个小文本块(高 34~45px), 与整只气泡框
    做 IoU 上限很低, 气泡内波形滚动/换行还会让框轻微抖动 → IoU 阈值几乎必然
    失败(实测各偏移下重叠率恒为 1.2%)。改为"标签中心点是否落在气泡框的
    容差扩张域内"更鲁棒。

    pad_s: 生命周期窗口的时间留白(气泡淡入淡出/OCR 采样时刻与视觉采样错位)。
    x_tol / y_tol: 以气泡宽/高为单位的中心点容差(相对量, 适配不同分辨率)。
    media_duration_s: 音/视频总时长, 用于**可行性约束**——语音不可能比"从气泡
        出现到录制结束"的剩余时间还长; 超过则丢弃该候选并退到次高票的取值。
    """
    if not ev.box:
        return None
    ax, ay, aw, ah = ev.box
    end = ev.end_time if ev.end_time is not None else ev.timestamp
    t0, t1 = ev.timestamp - pad_s, max(end, ev.timestamp) + pad_s
    ecx, ecy = ax + aw / 2.0, ay + ah / 2.0
    votes = Counter()
    dist = {}   # sec -> 该时长候选里最靠近气泡中心的归一化距离(用于平票时择优)
    for h in hits:
        if not (t0 <= h.timestamp <= t1):
            continue
        if ev.side and h.side != ev.side:
            continue
        sec = _parse_duration_seconds(h.text)
        if not sec:
            continue
        if not h.box:
            continue
        hx, hy, hw, hh = h.box
        cx, cy = hx + hw / 2.0, hy + hh / 2.0
        if not (ax - aw * x_tol <= cx <= ax + aw * (1 + x_tol)):
            continue
        if not (ay - ah * y_tol <= cy <= ay + ah * (1 + y_tol)):
            continue
        votes[sec] += 1
        d = (((cx - ecx) / max(aw, 1.0)) ** 2
             + ((cy - ecy) / max(ah, 1.0)) ** 2) ** 0.5
        if sec not in dist or d < dist[sec]:
            dist[sec] = d
    if not votes:
        return None
    # 先按票数、再按"离气泡中心更近"排序(避免 most_common 的任意顺序), 然后取
    # 第一个**时长放得下**的候选(可行性约束, 1s 容差吸收端点误差)。
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], dist.get(kv[0], 0.0)))
    for sec, _n in ranked:
        if media_duration_s and ev.timestamp + sec > media_duration_s + 1.0:
            continue
        return sec
    return None


# 正文质量门(剔除"框内含正文"的伪语音气泡) / text-content quality gate
# 语音气泡内部只有波形 + 可选时长标签, **没有正文**; 因此候选框内若被 OCR 认出了
# 一条正文(非时长标签), 它其实是文本气泡。标定与实测见 `drop_text_bubbles`。
TEXT_GATE_CONTAIN = 0.90    # 正文行需被候选框"基本包住"的面积比(相对正文行自身面积)
# 正文覆盖候选框自身面积的比例(兜住 OCR 框大于气泡的情形)。取 0.50 而非 0.35:
# 聊天区里的系统提示块(如"亲密度达到100℃…"四行)与本屏语音气泡在纵向相邻,
# 其末行会与气泡框部分重叠, 覆盖度 0.36 恰好越过 0.35 把真语音气泡误杀
# (实测 doc=1292.5 的"你下午三点前能补上吗")。真文本气泡的正文几乎整条在框内,
# 由 `TEXT_GATE_CONTAIN` 判据命中, 不依赖本阈值。
TEXT_GATE_COVER = 0.50
TEXT_GATE_PAD_S = 1.5       # 候选框生命周期窗口两侧的时间留白(秒)


def _center_within(inner, outer) -> bool:
    """内部框(正文行)的中心点是否落在外部框(候选气泡)内。

    两个框必须在**同一坐标系**: 有滚动曲线时都由调用方换算到文档坐标, 否则都用
    屏幕坐标。 / Whether the inner box's center lies inside the outer box.
    """
    ax, ay, aw, ah = outer
    hx, hy, hw, hh = inner
    return (ax <= hx + hw / 2.0 <= ax + aw and ay <= hy + hh / 2.0 <= ay + ah)


def _overlap_area(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def drop_text_bubbles(vision_events: List[VoiceEvent],
                      text_hits,
                      contain: float = TEXT_GATE_CONTAIN,
                      cover: float = TEXT_GATE_COVER,
                      pad_s: float = TEXT_GATE_PAD_S,
                      duration_hits=None,
                      scroll_curve=None
                      ) -> List[VoiceEvent]:
    """质量门: 剔除"框内含有可识别正文"的候选气泡。

    为什么 / Why:
      视觉形状检测无法可靠区分"语音气泡(波形)"与"短文文本气泡"——检测器的
      `voice_score` 阈值被刻意降到 0.45 以适配新拟真 UI(见 pipeline 注释), 结果
      短文本气泡也会被放行。实测(2026-10-16, `自行模拟对话记录3.mp4`): 去重后的
      气泡清单为 12'发'/12'收', 而 GT 只有 7'发'/13'收' —— **多出的 5 条全部是
      右侧(发)的短文本消息**(如 "哪三个" / "订单SN0010已确认收款"), 它们按顺序
      插入清单后把后续所有气泡与音频段的**位置配对整体错位**。

    判据 / Criterion(三重约束, 宁漏勿误杀):
      1. 时间: 正文行出现的时刻落在候选框生命周期窗口(± pad_s)内;
      2. 侧别: 正文行与候选框同侧(按水平位置), 否则不算(防跨列误伤);
      3. 位置: 正文行的**中心点落在候选框内**, 且满足下列任一覆盖度:
         a) 正文行面积 ≥ `contain` 比例被候选框覆盖 —— "整条正文都在框里";
         b) 正文覆盖候选框自身面积 ≥ `cover` —— OCR 框常比气泡略大, 单看 (a)
            会漏掉"框内明显有正文"的短气泡(实测 "哪三个"/"订单SN0010…")。
      4. 反证优先: 若框内同时检出**时长标签**(`duration_hits`), 该候选按语音处理,
         不因正文覆盖而剔除 —— 微信"语音转文字"会在同一气泡里同时出现正文与时长。
      时长标签**不参与**正文判据(它本就长在语音气泡内), 见
      `OcrVoiceScanner.scan_text_hits`。

    坐标口径 / Coordinate basis:
      传 `scroll_curve` 时, 正文行与候选框都先换算成**文档纵坐标**再比较。窗口内
      跨帧的屏幕 y 会相差一整步滚动量(实测 100+px), 只用屏幕 y 会漏判"同一行、
      不同帧"的情形(实测 "订单SN0010已确认收款" 在 t=39.07 的屏幕位置与 t=39.6
      的候选框差 56px, 未被剔除, 多出一只'发'气泡把后续顺序整体顶偏一位)。

    对 GT 的稳健收益(清单构成 = 稳定口径, 位置配对口径对单条噪声极敏感):
      清单偏离 |Δ发|+|Δ收| 由 6 降到 3(发 12→9, GT 7); 位置一致率口径本身噪声
      达 ±2/20, 不作为判据。
    / Drop candidates whose box CONTAINS a recognized text line (same side, same
    time) — voice bubbles carry a waveform, not prose.
    """
    if not text_hits:
        return list(vision_events)
    # 延迟导入: 与静止观测聚类共用同一套"屏幕 y → 文档 y"换算, 避免口径漂移。
    from .report_merge import scroll_offset_at
    use_doc = bool(scroll_curve) and len(scroll_curve) >= 2

    def _geo(box, ts) -> Tuple[float, float, float, float]:
        """把屏幕坐标框换算成文档坐标框(有曲线时); 无曲线时原样返回。"""
        if not use_doc:
            return box
        return (float(box[0]),
                float(box[1]) + scroll_offset_at(scroll_curve, float(ts)),
                float(box[2]), float(box[3]))

    ts_sorted = [h.timestamp for h in text_hits]
    dur = list(duration_hits or [])
    kept: List[VoiceEvent] = []
    dropped = 0
    for ev in vision_events:
        if not getattr(ev, "box", None) or ev.side not in ("发", "收"):
            kept.append(ev)
            continue
        # 位置判据必须用**静止帧观测框**: 候选框在滚动途中被首次检出时, 框的位置与
        # 同时刻 OCR 正文行不共基准(实测差 100+ px), 会把隔壁行的正文当成"框内正文"
        # 而误杀真语音气泡。静止帧的框才与该帧正文行同基准。
        gate_box = getattr(ev, "stable_box", None) or ev.box
        ax, _, aw, ah = gate_box
        anchor_t = getattr(ev, "stable_timestamp", None)
        if anchor_t is None:
            anchor_t = ev.timestamp
            end = ev.end_time if ev.end_time is not None else ev.timestamp
            t0, t1 = ev.timestamp - pad_s, max(end, ev.timestamp) + pad_s
        else:
            # 静止观测时刻的正文行才与 gate_box 同基准; 生命周期窗口常横跨滚动期,
            # 其中途的正文行位置多变, 会引入错位证据。
            t0, t1 = anchor_t - pad_s, anchor_t + pad_s
        # 候选框的文档坐标(顶边); `cluster_static_observations` 构造 stable_box 时
        # 保证 doc == stable_box.y + offset(stable_timestamp), 故可直接反算。
        gate_geo = _geo(gate_box, anchor_t)
        lo = bisect.bisect_left(ts_sorted, t0)
        is_text = False
        max_cover = 0.0
        for j in range(lo, len(text_hits)):
            h = text_hits[j]
            if h.timestamp > t1:
                break
            if h.side and h.side != ev.side:
                continue
            # 同一 OCR 帧内用屏幕坐标即可; 跨帧(窗口内不同时刻)必须用文档坐标,
            # 否则滚动量(实测单步 100+px)会把同一条消息判成"不重叠"。
            h_geo = _geo(h.box, h.timestamp)
            if not _center_within(h_geo, gate_geo):
                continue
            inter = _overlap_area(gate_geo, h_geo)
            if inter / max(1.0, h.box[2] * h.box[3]) >= contain:
                is_text = True
                break
            max_cover = max(max_cover, inter / max(1.0, aw * ah))
        if not is_text and max_cover >= cover:
            # 反证优先: 框内已检出的时长标签说明这是(可能带转写的)语音气泡
            has_label = any(
                (d.side == ev.side or not d.side)
                and t0 <= d.timestamp <= t1
                and getattr(d, "box", None)
                and _center_within(_geo(d.box, d.timestamp), gate_geo)
                for d in dur
            )
            is_text = not has_label
        if is_text:
            dropped += 1
        else:
            kept.append(ev)
    if dropped:
        logger.info("正文质量门: 剔除 %d 个框内含正文的伪语音气泡", dropped)
    return kept


# 贴纸/表情包宽度门 / sticker (text-free image) width gate
# 无文字贴纸会被形状检测当成语音气泡(实测 `自行模拟对话记录3.mp4` 的贴纸为 70x49,
# 而同屏语音气泡宽 138~219)。判据用**同批气泡宽度中位数**的相对比例, 不写死像素,
# 以便与分辨率/UI 解耦(不同 UI 的语音气泡宽度下限不同, 但同一 UI 内贴纸显著更窄)。
STICKER_WIDTH_RATIO = 0.60
# 高度门 / height gate: 与宽度同口径的相对判据。UI 装饰条/时长标签残片会被形状
# 检测当成语音气泡, 它们**宽度与真气泡一致**(贴着气泡左右边缘), 只有高度显著偏小
# (实测 `自行模拟对话记录4.mp4`: 真气泡 h=63~68, 残片 h=20; `记录3`: 真气泡
# h=47~50, 残片 h=26)。只看宽度抓不到这类残片, 而它在文档坐标排序里会插到中段,
# 吃掉一个配对槽位使其后所有气泡的收发方向整体错一位(实测 video4 因此 4/6)。
STICKER_HEIGHT_RATIO = 0.60
STICKER_MIN_SAMPLES = 5


def drop_narrow_bubbles(vision_events: List[VoiceEvent],
                        ratio: float = STICKER_WIDTH_RATIO,
                        min_samples: int = STICKER_MIN_SAMPLES,
                        height_ratio: float = STICKER_HEIGHT_RATIO
                        ) -> List[VoiceEvent]:
    """剔除宽**或高**显著小于同批气泡的候选(无文字贴纸/表情包、UI 装饰残片)。

    为什么 / Why: 顺序对齐是"第 k 段音频 ↔ 顺序里第 k 只气泡"的一一映射, 多出的
    一只贴纸会把其后所有气泡的收发方向整体错一位。贴纸没有正文, 正文质量门抓不到,
    只能靠几何: 同一录屏里语音气泡宽高都由 UI 决定且存在明显下限, 贴纸/图片明显更窄,
    装饰条/标签残片明显更矮。
    / Drop text-free stickers and UI fragments: they carry no prose (the text gate
    cannot see them) but are far narrower or shorter than voice bubbles of the same UI.
    """
    boxes = [e.box for e in vision_events
             if getattr(e, "box", None) and e.side in ("发", "收")]
    if len(boxes) < min_samples:
        return list(vision_events)
    w_limit = statistics.median([b[2] for b in boxes]) * ratio
    h_limit = statistics.median([b[3] for b in boxes]) * height_ratio
    kept = [e for e in vision_events
            if not (getattr(e, "box", None) and e.side in ("发", "收")
                    and (e.box[2] < w_limit or e.box[3] < h_limit))]
    if len(kept) != len(vision_events):
        logger.info("贴纸尺寸门: 剔除 %d 个小气泡(宽 < %.0fpx 或 高 < %.0fpx)",
                    len(vision_events) - len(kept), w_limit, h_limit)
    return kept


# 静止观测聚类的文档坐标容差(px)。同一物理气泡在不同静止屏上的 doc 抖动实测
# <=12px(检测框高度在 47~59 间浮动), 而相邻两条消息的 doc 间距约 81px(一个行槽),
# 故 24px 既能并掉抖动、又不会把相邻消息串在一起。
# / Doc-coordinate tolerance for clustering static-frame observations.
STATIC_CLUSTER_DOC_PX = 24.0
# 一个簇至少需要的观测帧数: 5fps 下停留 0.4s 即达标; 单帧簇多是滚动途中那几帧
# 静止判定的瞬时误检, 不作为气泡。
STATIC_CLUSTER_MIN_OBS = 2


def cluster_static_observations(observations: List[dict],
                                scroll_curve: Optional[List[Tuple[float, float]]] = None,
                                doc_tol: float = STATIC_CLUSTER_DOC_PX,
                                min_obs: int = STATIC_CLUSTER_MIN_OBS
                                ) -> List[VoiceEvent]:
    """把静止帧上的逐帧原始检测聚成"气泡清单", 并换算成文档坐标锚点。

    为什么不复用跨帧身份跟踪 / Why not reuse cross-frame identity tracking:
      滚动式录屏里, 同一气泡在滚动前后会被跟踪器当成"消亡 + 新生"(位移超过
      `SCROLL_MAX_DY` 时平移兜底也失效), 于是单个事件的 y 观测横跨两只气泡;
      而**静止帧上的原始检测**本身就极稳(实测同一物理气泡在多屏上的文档坐标
      只抖 ±12px)。文档坐标排序只需要"某只气泡在某一静止屏上的位置", 不需要
      "它从哪一帧开始存在", 故直接按文档坐标聚类, 绕开身份跟踪的错接。

    输出事件的 `stable_timestamp`/`stable_box` 精确复现该簇的文档坐标
    (stable_box.y = doc - 该时刻滚动偏移), 使 `report_merge._bubble_anchor`
    取到的 doc 与本簇一致; `timestamp`/`end_time` 保留首/末静止观测时刻。
    / Cluster per-frame static detections by document y, bypassing identity tracking.
    """
    if not observations:
        return []
    # 延迟导入: 本模块是纯视觉扫描器, 不应在模块顶层耦合报告合并层(曲线查询)。
    from .report_merge import scroll_offset_at
    curve = scroll_curve or []
    items = []
    for o in observations:
        box = o.get("box")
        if not box:
            continue
        t = float(o.get("t", 0.0))
        items.append((float(box[1]) + scroll_offset_at(curve, t), t, o))
    items.sort(key=lambda it: it[0])
    groups: List[List[Tuple[float, float, dict]]] = []
    last_doc: Optional[float] = None
    for it in items:
        if groups and last_doc is not None and it[0] - last_doc <= doc_tol:
            groups[-1].append(it)
        else:
            groups.append([it])
        last_doc = it[0]
    out: List[VoiceEvent] = []
    for g in groups:
        if len(g) < min_obs:
            continue
        doc = float(statistics.median([it[0] for it in g]))
        obs = [it[2] for it in g]
        # 锚点时刻取**首个静止观测**: 它离该屏最早的一帧 OCR 最近, 正文质量门
        # (按 stable_timestamp ± pad_s 取正文行)才有证据可用。OCR 缓存是按内容
        # 指纹稀疏存储的(本视频 152s 只有 26 帧), 若取簇的中位时刻, 很多气泡
        # 附近根本没有 OCR 帧 → 文本气泡漏过门控(实测 25 个簇一个都没被剔除)。
        # 同一静止屏内偏移恒定, 取首帧不影响文档坐标。
        t_anchor = float(min(it[1] for it in g))
        xs = [float(o["box"][0]) for o in obs]
        ys = [float(o["box"][1]) for o in obs]
        ws = [float(o["box"][2]) for o in obs]
        hs = [float(o["box"][3]) for o in obs]
        votes = Counter(o.get("side") or "" for o in obs)
        votes.pop("", None)
        side = ""
        if votes:
            top = max(votes.values())
            for s in ("发", "收"):
                if votes.get(s, 0) == top:
                    side = s
                    break
        x_mid = float(statistics.median(xs))
        w_mid = float(statistics.median(ws))
        h_mid = float(statistics.median(hs))
        # 锚点 y 由簇的文档坐标反解: stable_box.y + offset(t_anchor) == doc
        y_stable = doc - scroll_offset_at(curve, t_anchor)
        conf = max((float(o.get("confidence", 0.0)) for o in obs), default=0.0)
        out.append(VoiceEvent(
            timestamp=float(min(it[1] for it in g)),
            end_time=float(max(it[1] for it in g)),
            side=side, confidence=conf,
            frame_id=int(t_anchor * 30),
            box=(x_mid, float(statistics.median(ys)), w_mid, h_mid),
            last_timestamp=t_anchor,
            last_box=(x_mid, y_stable, w_mid, h_mid),
            stable_timestamp=t_anchor,
            stable_box=(x_mid, y_stable, w_mid, h_mid),
            source="vision"))
    logger.info("静止观测聚类: %d 次观测 → %d 个气泡簇",
                len(observations), len(out))
    return out


def merge_ocr_and_vision(vision_events: List[VoiceEvent],
                         ocr_hits,
                         match_window_s: float = 2.5,
                         box_tol: float = 0.45,
                         x_tol: float = 0.05,
                         y_tol: float = 1.20,
                         media_duration_s: Optional[float] = None
                         ) -> List[VoiceEvent]:
    """以视觉气泡清单为主, 用 OCR 时长标签在**气泡框内**做双源佐证。

    视觉形状检测给出权威且稳定的气泡清单(已与肉眼核对一致)。OCR 时长标签仅当
    在气泡生命周期时间窗内、同侧别、且其中心点落在气泡框容差域内时, 才把该事件
    标记为 source='hybrid'(双源确认)并回填 audio_duration; 否则保持
    source='vision'。**绝不新增独立事件**, 杜绝全局 mm:ss 把界面时钟/进度误判成
    语音气泡导致的计数爆炸。

    ocr_hits: OcrVoiceScanner.scan_duration_hits 的产物(list[DurationHit]),
        **其 box 必须已是原始帧坐标**(调用方需传 roi_offset)。
    match_window_s: 生命周期窗口两侧的时间留白(秒)。
    box_tol: 保留参数(旧 IoU 阈值), 现由 x_tol/y_tol 的中心点容差取代。
    media_duration_s: 音/视频总时长, 传给 match_duration_label 做可行性约束。
    """
    merged: List[VoiceEvent] = []
    if not ocr_hits:
        merged = [e for e in vision_events]
        merged.sort(key=lambda e: e.timestamp)
        return merged
    for ev in vision_events:
        dur = match_duration_label(ev, ocr_hits, pad_s=match_window_s,
                                   x_tol=x_tol, y_tol=y_tol,
                                   media_duration_s=media_duration_s)
        if dur and dur > 0:
            ev.audio_duration = dur
            ev.source = "hybrid"
        merged.append(ev)
    merged.sort(key=lambda e: e.timestamp)
    return merged


def _any_within(ts_sorted: List[float], t: float, window: float) -> bool:
    i = bisect.bisect_left(ts_sorted, t)
    if i < len(ts_sorted) and ts_sorted[i] - t <= window:
        return True
    if i > 0 and t - ts_sorted[i - 1] <= window:
        return True
    return False


def _same_bubble_geom(a: VoiceEvent, b: VoiceEvent,
                      size_tol: float = 0.25,
                      cx_tol: float = 0.15) -> bool:
    """两事件是否指向同一物理气泡(与侧别无关)。

    判据二选一:
      1. 框重叠(IoU>=0.3): 绿/白掩码交替命中同一气泡时, 框会有微小差异
         (白掩码可能只圈到气泡内浅色部分), 尺寸签名可能超出容差, 用重叠兜住;
      2. 尺寸+中心x 签名一致: 滚动使 y 平移时位置IoU会失效, 但宽/高/中心x
         稳定, 用于跨滚动的同一气泡判定。

    真不同的左右气泡(发靠右/收靠左)框几乎不重叠且中心x 相差极大, 两种判据
    都不会误并。
    """
    if not (a.box and b.box):
        return True
    if _iou(a.box, b.box) >= 0.3:
        return True
    return _same_size_key(a, b, size_tol, cx_tol)


def _scroll_delta_px(a_box, b_box) -> Optional[float]:
    """跨帧"平移兜底"候选: 尺寸与中心x 一致(纵向平移不变)时返回**带符号**的像素 Δy。

    追加式滚动使气泡整体平移, 位置 IoU 归零但宽/高/中心x 稳定, 故用它兜住同一气泡;
    返回的带符号 Δy(b_center_y - a_center_y)既用于估计"帧级全局平移"(整屏气泡同值),
    也用于在该平移下做一对一配对。不满足尺寸/中心x 一致性时返回 None(不匹配)。

    / Scroll-tolerant cross-frame candidate: same size + same center-x (translation
    invariant) → signed pixel Δy, or None if unmatched.
    """
    ax, ay, aw, ah = a_box
    bx, by, bw, bh = b_box
    if abs(aw - bw) / max(aw, bw, 1) > SCROLL_SIZE_TOL:
        return None
    if abs(ah - bh) / max(ah, bh, 1) > SCROLL_SIZE_TOL:
        return None
    if abs((ax + aw / 2) - (bx + bw / 2)) / max(aw, bw, 1) > SCROLL_CX_TOL:
        return None
    return (by + bh / 2) - (ay + ah / 2)


def _estimate_scroll_shift(pairs) -> Optional[float]:
    """从兼容配对的 Δy(像素)估计"帧级全局纵向平移" g。

    滚动时整屏气泡同步平移同一像素量, 真实配对的 Δy 会聚在同一值附近; 取"邻域内
    配对最多"的那个值(簇内中位数)作为 g。票数不足(非滚动场景)返回 None, 调用方
    退回 g=0(即按 |Δy| 最小配对, 与旧行为一致)。

    / Estimate the global frame-level vertical shift as the mode (within `cluster_px`)
    of compatible Δy's; None when evidence is insufficient.
    """
    if len(pairs) < SCROLL_SHIFT_MIN_VOTES:
        return None
    best_g, best_n = 0.0, 0
    for dy, _si, _i in pairs:
        n = 0
        for d2, _s2, _i2 in pairs:
            if abs(d2 - dy) <= SCROLL_SHIFT_CLUSTER_PX:
                n += 1
        if n > best_n:
            best_n, best_g = n, dy
    if best_n < SCROLL_SHIFT_MIN_VOTES:
        return None
    members = sorted(d for d, _s, _i in pairs
                     if abs(d - best_g) <= SCROLL_SHIFT_CLUSTER_PX)
    return members[len(members) // 2]


def _same_size_key(a: VoiceEvent, b: VoiceEvent,
                   size_tol: float = 0.25, cx_tol: float = 0.15) -> bool:
    """两事件是否属同一物理气泡: 滚动只动 y, 宽/高/中心x 稳定。
    任一缺失框则退化按时间合并(兼容旧行为)。
    """
    if not (a.box and b.box):
        return True
    ax, ay, aw, ah = a.box
    bx, by, bw, bh = b.box
    # 宽/高相对差在容差内
    if abs(aw - bw) / max(aw, bw, 1) > size_tol:
        return False
    if abs(ah - bh) / max(ah, bh, 1) > size_tol:
        return False
    # 中心x 相差不超过宽度比例(同侧气泡贴边一致)
    acx, bcx = ax + aw / 2, bx + bw / 2
    if abs(acx - bcx) / max(aw, bw, 1) > cx_tol:
        return False
    return True