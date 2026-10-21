"""
报告合并 (Report Merge)

把音频侧(带时间锚点的逐段转写)与视觉侧(语音气泡事件)合并进最终报告 dict:

  1. result["audio"]       : 完整逐段音频转写(视频时间轴锚点)。
  2. result["voice_events"]: 视觉语音气泡 与 音频段 按时间就近配对 → 还原"这条
                             语音说的内容"; 未配对的(静音/被提前滚走)保留空文本。
  3. 注入聊天段落          : 把配好对的语音内容以 "- +Ns [语音][报/收] ：text"
                             形式按时间插入对应视觉段落的 content, 使语音内容真正
                             进聊天记录流水。注入**按转写段(一句一次)**进行, 而不是
                             按气泡把整簇拼接成一行: 一个气泡的时间窗经常横跨相邻
                             若干条消息(气泡检测窗过宽), 拼接会把多条不同消息粘成
                             一条; 逐段注入保证每条内容仍可单独对齐。气泡级的聚合
                             文本仍完整保留在 result["voice_events"] 里。
                             注入前做**视觉×音频跨路去重**: 屏幕上已有同一句(经 OCR
                             与 LLM 组装出的视觉行)时不再重复注入, 避免同一消息两路
                             各出一条导致输出冗余。
"""
from __future__ import annotations

import bisect
import logging
import re
import statistics
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple

from .data_types import VoiceEvent, VoiceUtterance
from .transcriptor import TranscriptResult
from src.visual.scroll_motion import offset_at as _image_offset_at

logger = logging.getLogger(__name__)


def _fmt_video(sec: float) -> str:
    """视频秒数 → 报告里可读的 +Ns 形式(与视觉时间戳兜底约定一致)。"""
    s = max(0, int(round(sec)))
    return f"{s}"


# ---------------------------------------------------------------------------
# 视觉 × 音频 跨路去重 / Cross-path (visual × audio) de-duplication
# ---------------------------------------------------------------------------
# 同一句话在两条路径上各出一份: 视觉侧来自屏幕(气泡"转文字"/OCR, 经 LLM 组装成
# "- 18:57 [收] ：…"), 音频侧来自 ASR(注入成 "- +Ns [语音] [收] ：…")。两路都
# 有这条消息时最终输出会重复, 使输出行数虚高、准确率被压低。
# 本节在**注入音频行之前**先与已有视觉行做近似判重, 命中则不注入(视觉行已经是
# 该消息的代表, 且经过 LLM 纠错通常更干净)。
# Same utterance arriving from both paths must be emitted once: skip the audio
# line when a near-duplicate visual line already exists in the assembled content.
_CROSS_PATH_DUP_SIM = 0.60     # 归一化相似度门限(ASR 与 OCR/LLM 的差异常剩 0.62~0.95)
_CROSS_PATH_LEN_RATIO = 0.6    # 长度比门限(min/max): 长短悬殊不做判重, 防偶然相似
_CROSS_PATH_MIN_LEN = 4        # 过短文本(如"好。")不做跨路判重, 避免误杀
# "开头相同、后半段悬殊"保护 / shared-prefix, lopsided-tail guard:
# 两路文本若共享了绝大部分**起始**内容(公共前缀 >= `_CROSS_PATH_PREFIX_RATIO` × 较短串),
# 且各自的后半段长度悬殊(较短尾段 < `_CROSS_PATH_TAIL_RATIO` × 较长尾段), 则判为
# **两条不同消息** —— 它们只是开头相同(同一话题/同一份文档名/同一句开场), 后半段才是
# 各自的内容; 同一句话的两路转写噪声不会让"半句话整段消失"(只会零散错字), 故长度
# 悬殊的尾部 = 另一条消息的判别信息, 不作为判重依据。
# 实测(2026-10-16, 17 个录屏共 80 次判重决策): 仅该规则能救回被误杀的 1 条语音
# (GT 语音"上次那份接口文档里少了三个字段" vs 屏上长消息 #29 的残缺视觉行
# "上次那份接口文档用字段", 相似度 0.77 → 原文被吞), 且不影响任何一条真·跨路重复。
# Two different messages that merely share the same opening have a much shorter tail
# on one side; transcript noise of *one* utterance never removes half of it entirely.
_CROSS_PATH_PREFIX_RATIO = 0.70   # 公共前缀 / 较短串 长度比
_CROSS_PATH_TAIL_RATIO = 0.60     # 较短尾段 / 较长尾段 长度比
# 数字锚点"子序列"放宽: ASR 会多念/少念一个数字字符(如把 URL 尾号 "v2" 漏读成 "v"),
# 使两路锚点只差**插入/删除**而非**替换**。真·不同数字(8812/8816)是替换关系, 不满足
# 子序列, 仍被拦下。仅当纯数字锚点足够长(≥该值)才放行, 防 "12"⊂"123" 之类偶然包含。
# 注意: 该长度保护只对**含真实阿拉伯数字**的锚点生效 —— 见 `_anchor_compatible`。
_ANCHOR_SUBSEQ_MIN = 5
# ASR 会把听到的 URL 标点"念"成汉字(":"→"冒号", "/"→"斜杠"), 与视觉侧的符号写法
# 面貌差异过大。判重前剔除这些"口述标点"词(仅用于比对, 不改输出正文)。
_SPOKEN_PUNCT_RE = re.compile(r"冒号|反斜杠|斜杠|下划线|井号|星号|顿号|引号|括号|方括号")
# 注入行前缀("- +52s [语音] [收] ："): 用于把音频注入行排除在视觉基准之外
_INJECTED_HEAD_RE = re.compile(r"^-\s*\+\d+s\s*\[语音\]")
# 输出行前缀("- 18:57 [收] ：" / "- +52s [语音] [收] ："): 剥离后才是消息正文
_LINE_PREFIX_RE = re.compile(
    r"^-\s*(?:(?:\+\d+s?)|(?:\d{1,2}:\d{2}(?::\d{2})?))?"
    r"\s*(?:\[[^\]]*\]\s*)*[:：]?\s*"
)
_DIGITS_RE = re.compile(r"\d+")
# 常见中文数字: ASR 常写"2块"、屏幕/LLM 常写"两块", 折算后数字锚点才能对齐
_CN_NUM = {"零": "0", "一": "1", "两": "2", "二": "2", "三": "3", "四": "4",
           "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}


def _strip_line_prefix(line: str) -> str:
    """剥掉输出行前缀, 返回消息正文 / strip the output line prefix."""
    return _LINE_PREFIX_RE.sub("", (line or "").strip())


def _loose(s: str) -> str:
    """去空白标点、口述标点词、只留字母数字汉字 / NFKC-normalized."""
    s = unicodedata.normalize("NFKC", s or "")
    s = _SPOKEN_PUNCT_RE.sub("", s)
    return re.sub(r"[^\w\u4e00-\u9fff]", "", s)


def _is_subsequence(short: str, long_: str) -> bool:
    """short 是否为 long_ 的子序列(只允许插入/删除, 不允许替换)。"""
    it = iter(long_)
    return all(ch in it for ch in short)


def _common_prefix_len(a: str, b: str) -> int:
    """两串的公共前缀长度 / length of the longest common prefix."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _shared_prefix_with_lopsided_tail(a: str, b: str) -> bool:
    """是否"开头相同、后半段长短悬殊"(→ 两条不同消息, 见 `_CROSS_PATH_PREFIX_RATIO`)。

    其中一条是另一条的前缀时(公共前缀 == 较短串长)判为 False: 那是**纯截断**
    (视觉行被气泡折行/OCR 截尾), 属同一句。
    """
    m = min(len(a), len(b))
    pre = _common_prefix_len(a, b)
    if pre < _CROSS_PATH_PREFIX_RATIO * m or pre >= m:
        return False
    tail_a, tail_b = len(a) - pre, len(b) - pre
    return min(tail_a, tail_b) < _CROSS_PATH_TAIL_RATIO * max(tail_a, tail_b)


def _anchor_compatible(a: tuple, b: tuple) -> bool:
    """两路数字锚点是否相容 / whether two digit anchors are compatible.

    锚点是 (字面锚点, 折算锚点) 二元组(见 `_digit_anchor`)。判定分三种情形:

      * 折算锚点相等 → 相容;
      * 折算锚点不等, 但**至少一方写了真实阿拉伯数字** → 额外数字必须足够长
        (≥`_ANCHOR_SUBSEQ_MIN`)且为子序列才算"多念/漏念": 真实数字是硬证据
        ("8812" vs "8816" 是替换, "12"⊂"123" 是偶然包含, 都不同消息);
      * 折算锚点不等且**双方都没有真实阿拉伯数字** → 数字全部来自口语中文数字的
        折算, 差异本身就是噪声: ASR 把"也一起"漏听成"起"(或反过来多听一个"一"),
        折算锚点就从 "2" 变成 "21" —— 这是同一条消息, 不受长度保护限制。
    """
    a_lit, a_fold = a
    b_lit, b_fold = b
    if a_fold == b_fold:
        return True
    lo, hi = (a_fold, b_fold) if len(a_fold) <= len(b_fold) else (b_fold, a_fold)
    if not _is_subsequence(lo, hi):
        return False
    return (len(lo) >= _ANCHOR_SUBSEQ_MIN) if (a_lit or b_lit) else True


def _digit_anchor(s: str) -> tuple:
    """数字锚点 → (字面锚点, 折算锚点) / (literal, folded) digit anchors.

    字面锚点 = 原文里**本就存在**的阿拉伯数字(拼接所有数字串), 是硬证据: 两路都
    写真实数字时, 差异必是真实差异。
    折算锚点 = 再把常见中文数字折算成阿拉伯数字后拼出的数字串(ASR 写"2块"、屏幕/
    LLM 写"两块"时靠它对齐)。但中文数字里有大量非数字用法("一起/一直/一样"),
    折算会引入孤立噪声数字, 因此折算锚点只做宽松比对, 不能当作硬约束。

    用"拼接"而不是"按标点分组"是为了容忍两路在标点上的抖动: 视觉
    "repro-20261016-0037.md" 与 ASR "repro202610160037MD" 的字面锚点都是
    "202610160037"; 同时仍能区分"数字本身不同"的模板消息
    ("您的快递8812" vs "您的快递8816" → 字面锚点 "8812" vs "8816")。
    Digit anchor: concatenate all digit runs so punctuation jitter between the two
    paths does not break alignment, while genuinely different numbers still differ.
    """
    s = unicodedata.normalize("NFKC", s or "")
    literal = "".join(_DIGITS_RE.findall(s))
    for a, b in _CN_NUM.items():
        s = s.replace(a, b)
    return literal, "".join(_DIGITS_RE.findall(s))


def _visual_texts(sections: List[dict]) -> List[tuple]:
    """取出各段落中**视觉侧**(非音频注入)的正文, 返回 (归一化文本, 数字锚点)。"""
    out = []
    for sec in sections:
        for raw in str(sec.get("content", "")).split("\n"):
            raw = raw.strip()
            if not raw or _INJECTED_HEAD_RE.match(raw):
                continue
            body = _strip_line_prefix(raw)
            t = _loose(body)
            if len(t) >= _CROSS_PATH_MIN_LEN:
                out.append((t, _digit_anchor(body)))
    return out


def _is_cross_path_duplicate(audio_text: str, visual_texts: List[tuple]) -> bool:
    """音频文本是否已被某条视觉行覆盖(近似重复)。

    数字串不同的两条判为**不同消息**(如"您的快递8812已到驿站"与"您的快递8814
    已到驿站"模板同形但确实是两条), 不做合并 —— 仅靠文本相似度无法区分这类
    模板近重复, 数字锚点是必要约束。
    """
    a = _loose(audio_text)
    if len(a) < _CROSS_PATH_MIN_LEN:
        return False
    a_digits = _digit_anchor(audio_text)
    for v, v_digits in visual_texts:
        # 数字锚点不相容 → 不同消息 / incompatible digit anchor => different message
        if not _anchor_compatible(a_digits, v_digits):
            continue
        # 长度悬殊 → 不做判重(长文本与短文本易偶然相似) / skip lopsided lengths
        if min(len(a), len(v)) < _CROSS_PATH_LEN_RATIO * max(len(a), len(v)):
            continue
        # 开头相同但后半段长短悬殊 → 两条不同消息(见 `_shared_prefix_with_lopsided_tail`)
        if _shared_prefix_with_lopsided_tail(a, v):
            continue
        if SequenceMatcher(None, a, v).ratio() >= _CROSS_PATH_DUP_SIM:
            return True
    return False


# 相邻语音段间隔小于该秒数时视为同一条语音(气泡可能被 VAD 按长时间停顿切开)
SEG_JOIN_GAP_S = 0.9

# 兜底配对的**最大时间间隔**(秒) / fallback pairing max endpoint gap (seconds)
# 主循环只认领"落在事件时刻 t 附近 max_match_gap_s 内"的段; 但气泡检测器常把事件
# 时刻锚成 0.0 或明显偏移(真实气泡窗口 [timestamp, end_time] 很宽), 于是有段无人
# 认领 → 内容丢失。第二遍按"段与事件窗口的端点间距"再配一次, 只接受该间距不超过本
# 门限者。
# 标定: 一条语音段通常 ≤~7s(实测本场景 20 段时长 4~6.9s), 主循环已覆盖事件起点
# ±1.5s; 误锚事件靠重叠(间距 0)兜底, 非重叠情形用 6s ≈ 一条语音长度即可跨过检测
# 时刻偏差, 又不会去借相邻(方向可能不同)气泡的内容 —— 再大则易把本该属于另一条
# 消息的语音错配过来并带错收发方向。本场景 9 个未认领段到最近事件间距均为 0,
# 6s 足够全部认领; 取 6.0 而非更大值是保守权衡。
# The main loop only claims segments near the event's exact timestamp `t`; when the
# detector mis-anchors an event (timestamp 0.0 while the real window is wide), its
# segments are left unclaimed. This fallback re-matches leftovers by nearest window
# endpoint gap, accepting only gaps <= this threshold. 6s ~= one utterance length:
# enough to bridge a detection-time offset, small enough to avoid borrowing a
# neighboring bubble's content (which could carry the wrong '发'/'收' side).
FALLBACK_GAP_S = 6.0

# 顺序对齐的"屏"间隔阈值(秒) / screen gap for content-order alignment
# 见 `_ordered_bubbles`: 相邻气泡首次出现时刻间隔超过该值即视为进入下一屏。
# 标定: 同屏气泡的首次出现间隔 ≤ 采样周期级(0.2s)或一次滚动翻页(实测 ≤2.6s),
# 而不同消息/不同翻页之间是数秒~数十秒; 取 3s 落在两者之间, 且实测 2~5s 结果一致。
ORDER_SCREEN_GAP_S = 3.0

# "预置静态聊天"判据: 前几段音频的任一时刻在屏气泡数达到该值即认定 /
# Static-overlay detection: peak concurrent bubbles over the first few segments.
STATIC_OVERLAY_MIN_ON_SCREEN = 3
STATIC_OVERLAY_PROBE_SEGS = 3


def _window_gap(a_start: float, a_end: float,
                b_start: float, b_end: float) -> float:
    """两时间区间的最短端点间距(重叠时为 0) / shortest endpoint gap (0 on overlap)."""
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0.0


def _fallback_rank(gap: float, ev: VoiceEvent) -> tuple:
    """兜底候选事件的排序键: 距离优先, 等距时 side 非空者优先, 再等距取时间窗更窄者。

    side 非空优先: 有收发方向的候选信息更完整(方向是聊天记录的关键标注);
    时间窗更窄优先: 窗越窄说明该时刻越确定, 宽窗多来自误检/滚动残留。
    / Rank: gap, then non-empty side, then narrower event window.
    """
    end = ev.end_time if ev.end_time is not None else ev.timestamp
    width = abs(end - ev.timestamp)
    return (gap, 0 if ev.side else 1, width)


def _text_segments(segments: List[VoiceUtterance]) -> List[VoiceUtterance]:
    """按时间排序、剔除空文本后的转写段(配对与注入两处同口径)。

    空文本段不参与任何环节: 整段静音/无有效人声时(VAD返回空段), 它既不能
    虚拟命中气泡, 也不应被注入成空行。
    """
    return [s for s in sorted(segments, key=lambda s: s.start)
            if (s.text or "").strip()]


def _collect_cluster(segs, start_idx, t, max_match_gap_s, consumed):
    """从事件时刻 t 出发, 收集一条"语音簇": 与 t 时间重叠的段及其相邻连续段。

    VAD 会按 >250ms 停顿把一条带停顿的语音切成多段, 故把与 t 时间上有交集、
    且彼此间隔 < SEG_JOIN_GAP_S 的段合并为一条, 避免后半句丢失; 已被其它
    事件消费的段跳过(M1: 一条语音只注入一次)。返回 (段下标列表, 拼接文本)。
    """
    n = len(segs)
    # 第一阶段: 找与事件时间窗重叠的种子段
    seeds = [i for i in range(n)
             if not consumed[i] and segs[i].start - max_match_gap_s <= t <= segs[i].end + max_match_gap_s]
    if not seeds:
        return None, None
    # 第二阶段: 从种子向两端扩展连续段(同一条语音被停顿切开的部分)
    chosen = set(seeds)
    changed = True
    while changed:
        changed = False
        lo = min(chosen)
        hi = max(chosen)
        # 向左扩展
        if lo > 0 and not consumed[lo - 1] and (lo - 1) not in chosen:
            if segs[lo].start - segs[lo - 1].end <= SEG_JOIN_GAP_S:
                chosen.add(lo - 1)
                changed = True
        # 向右扩展
        if hi < n - 1 and not consumed[hi + 1] and (hi + 1) not in chosen:
            if segs[hi + 1].start - segs[hi].end <= SEG_JOIN_GAP_S:
                chosen.add(hi + 1)
                changed = True
    idx = sorted(chosen)
    text = "".join(segs[i].text for i in idx if segs[i].text)
    return idx, text


def _bubble_anchor(ev: Optional[VoiceEvent]) -> Tuple[Optional[float], Optional[float]]:
    """返回气泡的**稳定位置**锚点 (t, y), 用于与同一屏的文本行比较。

    首见位置是"滚动途中某一屏"的临时坐标: 预置/滚动式聊天里气泡被首次检测到时
    可能正处于滚动中途, 与文本行不共基准。改用最后一次在屏观测(稳定停留后)的
    框与时刻, 文档坐标才与视觉行同基准。
    优序: **首次静止帧观测** → 最后观测 → 首见框。静止帧上的 y 才与滚动偏移
    曲线共基准(滚动途中观测会错位一整步滚动量, 实测可达 300px)。
    / Preference: first static-frame observation, then last observation.
    """
    if ev is None:
        return None, None
    t = getattr(ev, "stable_timestamp", None)
    box = getattr(ev, "stable_box", None)
    if t is None:
        t = getattr(ev, "last_timestamp", None)
        box = getattr(ev, "last_box", None)
    if t is None:
        t = ev.timestamp
    if not box:
        box = getattr(ev, "box", None)
    return t, (box[1] if box else None)


# 同一物理气泡被重复发射时, 其文档纵坐标几乎相同 → 合并。容差取 6px:
# 相邻两条消息的文档间距至少一条气泡高(~60px), 而同一气泡的锚点误差只有数像素。
DOC_DUP_PX = 6.0


def _dedup_by_doc(doc_keyed: List[Tuple[float, float, VoiceEvent]]
                  ) -> List[VoiceEvent]:
    """按文档纵坐标合并"同一物理气泡"的重复事件(输入已按 doc 升序)。

    重复来源: 检测分数在阈值附近抖动, 气泡短暂消失超过 `expire_s` 后被当作新气泡
    再发射一次(实测同一只气泡在静止屏上被发射 3 次, 把后续顺序整体顶偏)。
    只按紧容差 + 同侧别合并, 保证不会把相邻的不同消息并掉。
    """
    kept: List[Tuple[float, float, VoiceEvent]] = []
    for doc, ts, ev in doc_keyed:
        if any(abs(k[0] - doc) <= DOC_DUP_PX and k[2].side == ev.side
               for k in kept):
            continue
        kept.append((doc, ts, ev))
    return [k[2] for k in kept]


def _ordered_bubbles(events: List[VoiceEvent],
                     scroll_curve: Optional[List[Tuple[float, float]]] = None
                     ) -> List[VoiceEvent]:
    """把气泡事件按**聊天顺序**排序: 先按"屏"分组, 屏内按屏幕纵向位置(从上到下)。

    为什么不是简单的 (时间, y) / Why not plain (time, y):
      纯 (首次出现时刻, y) 只在"气泡逐条追加、每条独占一段时间"时成立。当录屏是
      **预置/滚动式**(整屏气泡一次性进入视野, 之后才被滚动翻页)时, 同一屏内多只气泡
      的首次出现时刻只差零点几秒, 用毫秒级时间做首键会把屏内顺序打乱成"检测抖动顺序",
      而不是"聊天先后顺序"; 而屏内真正的聊天先后由 **y(上→下)** 给出。
      实测(2026-10-16, GT `gt/chat30_voice.gt.json` + `自行模拟对话记录3.mp4`,
      20 条语音方向): 简单 (时间, y) 11/20, 按"屏内 y 优先"后 13/20, 且在
      间隔阈值 2~5s 上结果一致(非刀尖阈值)。

    分组规则: 依首次出现时刻升序扫描, 相邻间隔 **> `ORDER_SCREEN_GAP_S`**(默认 3s)即为
    换屏——"同时上屏"的气泡间隔≈采样周期(0.2s)或滚动翻页的一瞬(实测同屏 ≤2.6s),
    而真正的新消息/新翻页之间是数秒到数十秒。这与 `_looks_static_overlay` 的场景一致:
    该函数只在本函数被采用(预置静态聊天 + 叠加播放)时才返回真。
    / Group bubbles into "screens" by first-appearance gaps; within a screen the
    chat order is the vertical order (top-to-bottom), across screens it is time.
    """
    def _y(e: VoiceEvent) -> float:
        return e.box[1] if getattr(e, "box", None) else 0.0

    eligible = [e for e in events
                if e.side in ("发", "收") and getattr(e, "box", None)]
    # 首选: 全部气泡都有稳定位置锚点 + 有滚动偏移曲线 → 直接按**文档纵坐标**全局
    # 排序。滚动式录屏里"换屏"分组只是滚动动作的副产物, 文档坐标才是聊天顺序本身。
    # / Preferred: global document-order sort when stable anchors are available.
    if scroll_curve:
        doc_keyed = []
        for e in eligible:
            ts, y = _bubble_anchor(e)
            if y is None:
                doc_keyed = []
                break
            doc_keyed.append((y + scroll_offset_at(scroll_curve, ts),
                              e.timestamp, e))
        if doc_keyed:
            doc_keyed.sort(key=lambda x: (x[0], x[1]))
            return _dedup_by_doc(doc_keyed)
    eligible.sort(key=lambda e: (e.timestamp, _y(e)))
    placed = []
    screen_idx = -1
    prev_ts = None
    for e in eligible:
        if prev_ts is None or e.timestamp - prev_ts > ORDER_SCREEN_GAP_S:
            screen_idx += 1
        prev_ts = e.timestamp
        placed.append((screen_idx, e))
    placed.sort(key=lambda t: (t[0], _y(t[1])))
    return [e for _, e in placed]


def _looks_static_overlay(events: List[VoiceEvent],
                          segs: List[VoiceUtterance]) -> bool:
    """判断录屏是否为"预置静态聊天 + 音频按时序叠加播放"（→ 需顺序对齐）。

    正常逐条录制: 某条语音播放时, 屏上通常只有 1~2 只气泡(刚发出的那条 + 上一条)。
    预置静态聊天: 音频开始播放前屏上已堆着多只气泡, 其"上屏时刻"是录制者滚动到的
    时刻, 与音频时间轴无对应关系 → 任何按时间的配对都系统性错边。
    判据: **前几段**音频各自开始时, 屏上并存的气泡数峰值 >= 3, 且气泡总数与段数
    大体相当(够对齐)。取峰值而非只看首段一瞬间: 首段时刻的在屏数受单帧检测抖动
    影响(实测 `自行模拟对话记录3.mp4` 修复前首段只有 2 只、误判为非静态场景,
    退化成时间配对后整份结果错位), 而滚动前后任一时刻的并存峰值更能代表"屏上
    已堆着多条消息"这一事实。
    / Heuristic: peak concurrent bubbles at the start of the first few utterances
    >= 3, and the bubble count is comparable to the segment count.
    """
    if not segs or not events:
        return False
    if len(events) < 0.6 * len(segs):
        return False
    peak = 0
    for seg in segs[:STATIC_OVERLAY_PROBE_SEGS]:
        t0 = seg.start
        on_screen = 0
        for e in events:
            end = e.end_time if e.end_time is not None else e.timestamp
            if e.timestamp <= t0 + 0.3 and end >= t0 - 0.3:
                on_screen += 1
        peak = max(peak, on_screen)
    return peak >= STATIC_OVERLAY_MIN_ON_SCREEN


def pair_voice_events_by_order(events: List[VoiceEvent],
                               segments: List[VoiceUtterance],
                               scroll_curve: Optional[List[Tuple[float, float]]] = None,
                               order_bubbles: Optional[List[VoiceEvent]] = None
                               ) -> List[dict]:
    """**内容顺序对齐**配对: 第 k 段音频 ↔ 聊天顺序里第 k 条语音气泡。

    适用"预置静态聊天 + 音频按时序叠加播放"的录屏(见 `_looks_static_overlay`):
    此时音频时间轴与消息上屏时刻无对应关系, 按时间就近配对必然错边; 但两者的**顺序**
    是对应的 —— 第 k 段音频就是聊天里第 k 条语音消息, 方向取该气泡的屏幕位置。

    order_bubbles: 由静止帧原始观测按文档坐标聚类得到的气泡清单(见
        `voice_bubble_scanner.cluster_static_observations`)。滚动式录屏里跨帧
        身份跟踪会错接, 事件的稳定位置锚点大量缺失; 该清单的文档坐标直接来自
        静止观测, 顺序更可靠。为空时退回事件自身的锚点。
    / Content-order alignment: k-th segment <-> k-th bubble in chat order.
    """
    segs = _text_segments(segments)
    ordered = _ordered_bubbles(order_bubbles if order_bubbles else events,
                               scroll_curve)
    out = []
    for k, seg in enumerate(segs):
        ev = ordered[k] if k < len(ordered) else None
        anchor_t, anchor_y = _bubble_anchor(ev)
        out.append({
            "timestamp": seg.start,
            "end_time": seg.end,
            "side": ev.side if ev else None,
            "confidence": ev.confidence if ev else 0.0,
            "source": getattr(ev, "source", "vision") if ev else "audio",
            "duration": "",
            "audio_duration": None,
            "text": (seg.text or ""),
            "matched": ev is not None,
            "consumed_ids": [k],
            # 气泡侧锚点: 供聊天面板把语音行插到正确位置(见 merge_into_result)。
            # 顺序对齐时 timestamp 是音频段时刻, 与气泡上屏时刻无关, 必须单独保留;
            # 位置取**稳定停留后**的观测, 才能与同一屏的文本行共基准。
            "bubble_timestamp": anchor_t,
            "bubble_y": anchor_y,
        })
    logger.info("顺序对齐配对: %d 段音频 ↔ %d 只气泡", len(segs), len(ordered))
    return out


def pair_voice_events(events: List[VoiceEvent],
                      segments: List[VoiceUtterance],
                      max_match_gap_s: float = 1.5,
                      order_align: str = "auto",
                      scroll_curve: Optional[List[Tuple[float, float]]] = None,
                      order_bubbles: Optional[List[VoiceEvent]] = None
                      ) -> List[dict]:
    """把每个语音气泡事件匹配到音频转写段(一条语音簇)。

    匹配: 收集与事件时刻时间上重叠的段及其相邻连续段, 合并为一条语音;
    一条语音段最多被一个事件消费(避免重复注入); 未命中任何段 → 文本为空。

    空文本段不参与匹配: 整段静音/无有效人声时(VAD返回空段, 转写为空串),
    该段不能虚拟命中气泡(消除了气泡"假配对+空文本"的伪命中, 见静音视频场景)。
    主循环后追加**兜底配对**: 仍未认领的段按与事件窗口最近者再配一次(见 FALLBACK_GAP_S)。
    返回可按时间排序的配对 dict 列表。

    order_align: "auto"(默认, 命中静态叠加场景才改走顺序对齐) / "on" / "off"。
    """
    segs = _text_segments(segments)
    if order_align == "on" or (order_align == "auto"
                               and _looks_static_overlay(events, segs)):
        return pair_voice_events_by_order(events, segments, scroll_curve,
                                          order_bubbles=order_bubbles)
    consumed = [False] * len(segs)
    out = []

    def _order_key(e: VoiceEvent):
        # 同刻多事件时优先「屏幕更靠下」者: 追加式聊天里最新到达(正在被朗读)的消息
        # 总在底部; 而位于屏幕上方者多是滚动导致的旧气泡重复发射(如转账卡片误检),
        # 若让它先消费音频, 会把新消息的收发方向配错。
        # / On ties prefer the bottom-most bubble (newest arrival in append-style chat).
        y = e.box[1] if getattr(e, "box", None) else 0
        return (e.timestamp, -float(y))

    for ev in sorted(events, key=_order_key):
        t = ev.timestamp
        idx, text = _collect_cluster(segs, None, t, max_match_gap_s, consumed)
        matched = idx is not None
        if matched:
            for i in idx:
                consumed[i] = True
        anchor_t, anchor_y = _bubble_anchor(ev)
        out.append({
            "timestamp": ev.timestamp,
            "end_time": ev.end_time,
            "side": ev.side,
            "confidence": ev.confidence,
            "source": getattr(ev, "source", "vision"),
            "duration": (ev.matched_text or ""),
            "audio_duration": getattr(ev, "audio_duration", None),
            "text": text or "",
            "matched": matched,
            "consumed_ids": idx or [],
            "bubble_timestamp": anchor_t,
            "bubble_y": anchor_y,
        })

    # 第二遍(兜底): 主循环只认领"与事件时刻 t 距离 < max_match_gap_s"的段, 时刻被
    # 误锚的事件(如 timestamp=0.0 而真实窗口很宽)会导致大量段无人认领 → 内容丢失。
    # 这里对仍未 consumed 的段, 按"段与事件窗口 [timestamp, end_time] 的最短端点间距"
    # 再找一次最近事件(重叠即 0); 间距在 FALLBACK_GAP_S 内则单独追加一条配对条目。
    # 每段只认领一次(主循环已消费的段不再动), 避免重复注入。
    # / Second (fallback) pass for segments the main loop left unclaimed.
    for i in range(len(segs)):
        if consumed[i]:
            continue
        seg = segs[i]
        best_ev = None
        best_rank = None
        for ev in events:
            gap = _window_gap(seg.start, seg.end, ev.timestamp,
                              ev.end_time if ev.end_time is not None else ev.timestamp)
            rank = _fallback_rank(gap, ev)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_ev = ev
        if best_ev is None or best_rank[0] > FALLBACK_GAP_S:
            continue
        consumed[i] = True
        anchor_t, anchor_y = _bubble_anchor(best_ev)
        out.append({
            "timestamp": seg.start,
            "end_time": seg.end,
            "side": best_ev.side,
            "confidence": best_ev.confidence,
            "source": "audio",
            "duration": "",
            "audio_duration": None,
            "text": (seg.text or ""),
            "matched": True,
            "consumed_ids": [i],
            "bubble_timestamp": anchor_t,
            "bubble_y": anchor_y,
        })
    return out


def _section_window(sections: List[dict], t: float) -> Optional[int]:
    """返回应承载时刻 t 对应语音消息的段落下标(时间窗最近的), 无则 None。"""
    if not sections:
        return None
    stamps = [float(s.get("timestamp", 0.0)) for s in sections]
    if t <= stamps[0]:
        return 0
    for i in range(len(stamps) - 1):
        if stamps[i] <= t < stamps[i + 1]:
            return i
    return len(sections) - 1


# 同屏判定: 组内各条目首次出现的时刻跨度不超过该值 → 视为"同一屏", 按纵向位置
# 排序; 否则按出现时刻排序(逐条追加/单调滚动场景)。/ same-screen tick span.
SCREEN_SAME_TICK_S = 1.0

# 滚动曲线的时间查询容差: 首见时刻与 OCR 帧时刻是两条时间基准, 同帧差 ~0.4ms。
SCROLL_SNAP_S = 0.001

# 相邻曲线点间隔不超过该值时, 查询按线性插值取偏移(快速滚动中气泡在 5fps 下的
# 观测时刻常落在两帧之间, 阶梯保持会系统性偏小); 更长的间隔说明屏幕基本静止,
# 用前一帧的偏移(阶梯保持)更稳。
SCROLL_INTERP_MAX_GAP_S = 2.0

# OCR 变体配对的相似度下限与最短长度。OCR 会把同一行读成不同文本(实测
# "…接口文档用" vs "…接口文档里，字段"), 精确同键会漏掉这唯一的公共行, 使整段
# 滚动未被记录(实测最后一屏少记 218px, 文本/气泡的文档坐标整体错位)。
# 过短的键(如"字段")歧义大, 不参与模糊配对。
FUZZY_PAIR_MIN_SIM = 0.80
FUZZY_PAIR_MIN_LEN = 4


def _fuzzy_deltas(prev: Dict[str, float], cur: Dict[str, float],
                  static: set) -> List[float]:
    """在精确同键未命中的行之间做**一对一**模糊配对, 返回 Δy 列表。

    贪心按相似度从高到低接受, 每个键只能配对一次; 仅接受两边都够长且相似度
    达标的配对, 避免"订单SN0010/订单SN0014"这类近重复模板互相顶替时产生错误位移
    (它们的 Δy 通常相同, 误配也不会改变中位数)。
    """
    cand = []
    prev_keys = [k for k in prev if k not in static and len(k) >= FUZZY_PAIR_MIN_LEN]
    cur_keys = [k for k in cur if k not in static and len(k) >= FUZZY_PAIR_MIN_LEN]
    for a in prev_keys:
        for b in cur_keys:
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio >= FUZZY_PAIR_MIN_SIM:
                cand.append((ratio, a, b))
    cand.sort(reverse=True)
    used_a, used_b, deltas = set(), set(), []
    for ratio, a, b in cand:
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        deltas.append(prev[a] - cur[b])
    return deltas


def build_scroll_curve(frames: Iterable[Tuple[float, List]],
                       image_curve: Optional[List[Tuple[float, float]]] = None
                       ) -> List[Tuple[float, float]]:
    """用相邻 OCR 帧的**公共文本行**估计"滚动偏移曲线" [(t, offset)], 把屏幕
    纵坐标换算成文档纵坐标。

    `image_curve`(可选)是图像侧累计滚动曲线(原始像素, 见
    `src.visual.scroll_motion`)。OCR 连一行公共文本都没有时(缓存稀疏, 实测
    t=98.5~101.4 整段滚动漏记 218px), 用图像侧位移补齐; 有公共行时仍以 OCR
    为准 —— 单行配对在这类录屏里是可信的(实测"收到"y 320→153 确为 167px
    慢速滚动), 而图像侧在动画/长时间段会低估位移。

    问题: 聊天气泡/文本行的 `y` 是**当前屏幕**坐标, 滚动后整体平移, 直接比较
    不同时刻的 y 会得到错的阅读顺序(例如先出现、但已被滚到顶部的行 y 很小,
    会被误判成最新消息)。

    做法: 同一文本行在相邻两帧都出现时, 其 `y` 的减少量(= 内容向上滚动的位移)
    就是这两帧之间的滚动量; 对**该帧所有公共行**取中位数作为该步位移, 逐帧累计
    即得 offset(t)。文档坐标 = y(t) + offset(t), 与时刻无关, 可直接排序。

    按**文本**而非跟踪行 ID 做跨帧配对: 滚动会让跟踪行重新建簇(ID 切换),
    用 ID 会漏掉大量公共行, 累计出的偏移严重偏小(实测 293px vs 真实 555px)。

    两条必要的鲁棒性处理:
      1) **同帧重复文本**(如两段完全相同的系统提示)无法确定是哪一条, 直接不参与
         该帧配对 —— 否则 text->y 的映射会互相覆盖, 给出错误的位移;
      2) **全局静止行**(顶部常驻诈骗提示横幅、状态栏)在整段视频里 y 从不变动,
         其 Δy 恒为 0; 当某帧对只有它和另一条真实行公共时, 中位数会被拖到 0,
         滚动被整个漏记(实测漏掉 500+ px)。

    参数: `/ frames` = [(timestamp, [TextLine, ...]), ...], 帧按时间升序。
    """
    per_time: Dict[float, Tuple[float, Dict[str, float]]] = {}
    for t, lines in frames or []:
        d: Dict[str, float] = {}
        dup = set()
        for ln in lines or []:
            key = _loose(getattr(ln, "text", ""))
            box = getattr(ln, "bbox", None)
            if not key or not box:
                continue
            if key in d:
                dup.add(key)
            d[key] = float(box[1])
        for k in dup:
            d.pop(k, None)
        if d:
            # 键用四舍五入去重, 值保留**原始时刻**: 曲线点若用舍入后的时间, 查询
            # 37.6699 会落不到 37.67 这个点上而退回上一帧的偏移(实测差 167px)。
            per_time[round(float(t), 3)] = (float(t), d)
    times = sorted(per_time)
    if not times:
        return []
    # 全局静止行: 出现 >=3 帧且 y 从不移动 → 固定叠加层, 不参与位移投票
    y_hist: Dict[str, List[float]] = {}
    for _, d in per_time.values():
        for k, y in d.items():
            y_hist.setdefault(k, []).append(y)
    static = {k for k, ys in y_hist.items()
              if len(ys) >= 3 and max(ys) - min(ys) <= 2.0}
    curve: List[Tuple[float, float]] = [(per_time[times[0]][0], 0.0)]
    offset = 0.0
    prev = per_time[times[0]][1]
    prev_t = per_time[times[0]][0]
    has_image = bool(image_curve) and len(image_curve) >= 2
    for t in times[1:]:
        raw_t, cur = per_time[t]
        deltas = [prev[k] - cur[k] for k in cur
                  if k in prev and k not in static]
        if deltas:
            # 有公共行就以 OCR 为准(已排除全局静止行)。
            offset += statistics.median(deltas)
        elif has_image:
            # 一行公共文本都没有: 该段 OCR 漏记, 用图像侧帧级运动补齐。
            offset += (_image_offset_at(image_curve, raw_t)
                       - _image_offset_at(image_curve, prev_t))
        else:
            # 图像曲线缺失时, 用 OCR 变体配对兜底(同义文本被读成不同写法)。
            fuzzy = _fuzzy_deltas(prev, cur, static)
            if fuzzy:
                offset += statistics.median(fuzzy)
        curve.append((raw_t, offset))
        prev, prev_t = cur, raw_t
    return curve


def scroll_offset_at(curve: List[Tuple[float, float]], t: float) -> float:
    """取时刻 t 的累计滚动偏移: 短间隔两点线性插值, 长间隔阶梯保持。

    查询容差 `SCROLL_SNAP_S`: 跟踪行的首见时刻与 OCR 缓存帧时刻来自两条时间基准,
    同一帧会有 ~0.4ms 的差异; 不容差会让 37.6666 落不到 37.667 这个曲线点上,
    退回上一帧的偏移(实测造成 167px 误差)。

    插值条件 `SCROLL_INTERP_MAX_GAP_S`: 快速滚动中气泡在 5fps 采样下被观测到的
    时刻常落在两帧之间, 阶梯保持会系统性取到偏小的偏移(落后一整步滚动量);
    而长间隔(>2s)代表屏幕基本静止, 此时阶梯保持更稳(避免把静止期的噪声斜率
    外推到查询时刻)。
    """
    if not curve:
        return 0.0
    times = [c[0] for c in curve]
    i = bisect.bisect_right(times, float(t) + SCROLL_SNAP_S) - 1
    if i < 0:
        return 0.0
    if i + 1 < len(curve):
        t0, o0 = curve[i]
        t1, o1 = curve[i + 1]
        gap = t1 - t0
        if 0 < gap <= SCROLL_INTERP_MAX_GAP_S:
            ratio = (float(t) - t0) / gap
            ratio = min(max(ratio, 0.0), 1.0)
            return o0 + (o1 - o0) * ratio
    return curve[i][1]


def _match_anchor(body: str, anchors: List[dict]) -> Optional[dict]:
    """为一条输出行找到它的视觉锚点(正文、首见时刻、首见纵坐标)。

    先做规范化后的**精确**匹配(多数行原样来自跟踪行), 再做包含匹配兜底
    (LLM 可能把同一消息改写了首尾)。包含匹配要求两边都不短于 4 字, 避免
    "好的"之类短消息误配到长句。
    """
    key = _loose(body)
    if not key:
        return None
    for a in anchors:
        if a["loose"] == key:
            return a
    best = None
    for a in anchors:
        al = a["loose"]
        if len(al) < 4 or len(key) < 4:
            continue
        if key in al or al in key:
            if best is None or len(al) > len(best["loose"]):
                best = a
    return best


def _screen_order(entries: List[dict]) -> List[dict]:
    """按"同屏分组 + 屏内纵向位置"排序带 (t, y) 的条目。

    - 先按 (t, y) 线性扫描分组, 相邻条目时间间隔 > `ORDER_SCREEN_GAP_S` 即新的一屏;
    - 组内若时间跨度 <= `SCREEN_SAME_TICK_S`, 说明是同一帧/同一屏被同时看到的多条
      内容(预置静态聊天), 此时时间先后只是检测抖动, **纵向位置**才是聊天顺序;
    - 否则按出现时刻排序(逐条追加场景)。
    """
    if not entries:
        return []
    ordered = sorted(entries, key=lambda e: (e["t"], e["y"], e["seq"]))
    groups, cur = [], [ordered[0]]
    for e in ordered[1:]:
        if e["t"] - cur[-1]["t"] > ORDER_SCREEN_GAP_S:
            groups.append(cur)
            cur = [e]
        else:
            cur.append(e)
    groups.append(cur)
    out = []
    for g in groups:
        if g[-1]["t"] - g[0]["t"] <= SCREEN_SAME_TICK_S:
            out.extend(sorted(g, key=lambda e: (e["y"], e["seq"])))
        else:
            out.extend(sorted(g, key=lambda e: (e["t"], e["y"], e["seq"])))
    return out


def _interleave_lines(lines: List[str], voice_entries: List[dict],
                      anchors: List[dict]) -> Optional[str]:
    """把语音行按"聊天顺序"插入视觉行之间; 无法锚定时返回 None(调用方回退追加)。

    视觉行本身已按聊天顺序排列(文本跟踪器输出), 缺的只是"某条语音夹在哪两条
    视觉消息之间"。排序键优先用**文档纵坐标**(`doc`, 已消掉滚动位移): 它才是
    与时间无关的阅读顺序。没有 `doc`(未估计出滚动曲线)时退回"首见时刻 + 同屏
    纵坐标"的旧口径。
    """
    matched = [_match_anchor(_strip_line_prefix(ln), anchors) for ln in lines]
    if not any(matched):
        return None
    entries, last, last_doc = [], None, None
    for i, (ln, a) in enumerate(zip(lines, matched)):
        if a is not None:
            last = (a["t"], a["y"])
            last_doc = a.get("doc")
        t_y = last if last is not None else (0.0, -1e9)
        entries.append({"t": t_y[0], "y": t_y[1], "doc": last_doc,
                        "seq": i, "line": ln})
    base = len(lines)
    for j, v in enumerate(voice_entries):
        entries.append({"t": v["t"], "y": v["y"], "doc": v.get("doc"),
                        "seq": base + j, "line": v["line"]})
    if all(e.get("doc") is not None for e in entries):
        return "\n".join(e["line"] for e in
                         sorted(entries, key=lambda e: (e["doc"], e["seq"])))
    return "\n".join(e["line"] for e in _screen_order(entries))



def merge_into_result(result: dict, transcript: TranscriptResult,
                      events: List[VoiceEvent],
                      max_match_gap_s: float = 1.5,
                      inject_chat: bool = True,
                      order_align: str = "auto",
                      visual_lines: Optional[List[dict]] = None,
                      scroll_curve: Optional[List[Tuple[float, float]]] = None,
                      order_bubbles: Optional[List[VoiceEvent]] = None
                      ) -> dict:
    """把音频转写 + 语音气泡配对写入 result dict 并注入聊天段落。

    参数:
        result: 视觉侧组装出的结果 dict(含 sections)。
        transcript: 音频转写结果。
        events: 视觉语音气泡出现事件。
        max_match_gap_s: 配对最大时间容差(秒)。
        inject_chat: 是否把语音消息注入视觉段落 content。
        order_align: 配对策略, 见 `pair_voice_events`。
        visual_lines: 视觉文本行锚点 [{"text","t","y"}] 或
            [{"text","t","y","doc"}]。t=首见时刻, y=首见纵坐标(原始帧坐标),
            doc=文档纵坐标(已消滚动位移, 见 `build_scroll_curve`)。
        scroll_curve: 滚动偏移曲线, 用于把气泡的屏幕纵坐标同样换算成文档坐标。
        order_bubbles: 顺序对齐用的气泡清单(静止观测聚类产物), 为空时用 events。
    """
    result["audio"] = transcript.to_dict()
    if not transcript.has_audio:
        result["voice_events"] = []
        return result

    paired = pair_voice_events(events, transcript.segments, max_match_gap_s,
                               order_align=order_align,
                               scroll_curve=scroll_curve,
                               order_bubbles=order_bubbles)
    result["voice_events"] = paired

    matched = [p for p in paired if p["matched"]]
    unmatched = [p for p in paired if not p["matched"]]
    stats = result.setdefault("statistics", {})
    stats["voice_events"] = len(paired)
    stats["voice_matched"] = len(matched)
    stats["voice_unmatched"] = len(unmatched)
    stats["audio_segments"] = len(transcript.segments)

    # 注入聊天段落: 逐转写段(一句一行), 按聊天顺序插入视觉行之间
    if inject_chat and result.get("sections"):
        sections = result["sections"]
        segs = _text_segments(transcript.segments)
        # 视觉侧判重基准(注入前快照, 之后才追加音频行, 故基准恒为纯视觉内容)
        visual_texts = _visual_texts(sections)
        anchors = []
        for a in (visual_lines or []):
            key = _loose(a.get("text", ""))
            if key:
                t0 = float(a.get("t", 0.0))
                y0 = float(a.get("y", 0.0))
                doc = a.get("doc")
                anchors.append({"loose": key, "t": t0, "y": y0,
                                "doc": (float(doc) if doc is not None else
                                        (y0 + scroll_offset_at(scroll_curve, t0)
                                         if scroll_curve else None))})
        by_sec = {}
        skipped_dup = 0
        for p in paired:
            side = f"[{p['side']}] " if p["side"] else ""
            for i in (p.get("consumed_ids") or []):
                if i < 0 or i >= len(segs):
                    continue
                seg = segs[i]
                text = (seg.text or "").strip()
                if not text:
                    continue
                idx = _section_window(sections, seg.start)
                if idx is None:
                    continue
                # 跨路去重: 视觉行已有同一句 → 不重复注入
                if _is_cross_path_duplicate(text, visual_texts):
                    skipped_dup += 1
                    continue
                line = f"- +{_fmt_video(seg.start)}s [语音] {side}：{text}"
                # 气泡侧锚点优先(顺序对齐时 p['timestamp'] 是音频时刻, 与上屏无关)
                anchor_t = p.get("bubble_timestamp")
                anchor_y = p.get("bubble_y")
                at = float(anchor_t if anchor_t is not None else seg.start)
                ay = float(anchor_y if anchor_y is not None else 0.0)
                doc = (ay + scroll_offset_at(scroll_curve, at)
                       if (scroll_curve and anchor_y is not None) else None)
                by_sec.setdefault(idx, []).append({
                    "t": at,
                    "y": ay,
                    "doc": doc,
                    "line": line,
                })
        if skipped_dup:
            stats["voice_cross_path_dups"] = skipped_dup
        for idx, entries in sorted(by_sec.items()):
            cur = sections[idx].get("content", "")
            lines = cur.split("\n") if cur.strip() else []
            merged = _interleave_lines(lines, entries, anchors) if anchors else None
            if merged is None:
                # 回退: 无法用视觉锚点定位 → 按音频时刻顺序追加(旧行为)
                entries.sort(key=lambda e: e["t"])
                merged = "\n".join([ln for ln in lines]
                                   + [e["line"] for e in entries])
            sections[idx]["content"] = merged

    # 更新摘要
    total_segs = len(transcript.segments)
    text_segs = sum(
        1 for s in transcript.segments if (s.text or "").strip()
    )
    summary_text = (
        f", 音频{total_segs}段{text_segs}段可读"
        if total_segs and text_segs
        else (", 有音轨但未采集到人声" if total_segs else "")
    )
    if summary_text:
        base = str(result.get("summary", ""))
        base = base.rstrip("。")
        result["summary"] = (f"{base}{summary_text}").strip()
    return result