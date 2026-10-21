"""
字符级多观测共识投票 (Character-level Multi-Observation Consensus)

背景 / Background:
    同一文本行会在多帧里被 OCR 反复识别, 结果常有细微差异(全/半角括号、形近字、
    漏字多字)。旧的多帧投票是"整行精确字符串"多数表决: 错一个字就变成另一张票,
    投票形同虚设(3 次识别得到 3 个不同的串 → 每个都只有 1 票)。

算法 / Algorithm:
    1. 快速路径: 所有观测完全一致 → 直接返回(绝大多数情况)。
    2. 参考串 R = 权重和最大的观测(平票时取更长者)。
    3. 用最小编辑对齐(difflib.SequenceMatcher)把每个观测 T 贴到 R, 按对齐块投票:
       - equal   : 对齐位的字符各投一票;
       - replace : 等长时逐位投票; 不等长时按下标配对投票, 多出的字符记为插入票、
                 短缺的参考字符记为删除票;
       - insert  : 记为"在锚点处插入该片段"的票;
       - delete  : 记为"该参考字符应删除"的票。
    4. 重建结果: 逐位取票数最高的字符(plurality); 插入/删除需拿到过半总权重才采纳
       (严格多数), 避免把少数派的幻觉噪声写进结果。

设计取向 / Design notes:
    - 与"多数表决"相比, 本算法只在字符粒度做决策, 因此能吸收"每次错在不同字"的
      抖动; 对完全不同的观测(不同行)退化为逐位投票, 不引入比原方案更差的走样。
    - 权重默认取 OCR 置信度: 置信度高的观测票更重。
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

# 插入片段需超过"总权重 x 此比例"才采纳(严格多数)。
_INSERT_ADOPT_RATIO = 0.5
# 删除某参考字符需超过"总权重 x 此比例"才采纳(严格多数)。
_DELETE_ADOPT_RATIO = 0.5


def consensus_text(texts: Sequence[str],
                   weights: Optional[Sequence[float]] = None) -> str:
    """对同一文本行的多次观测做字符级共识投票, 返回共识串。

    参数:
        texts: 多帧观测到的文本(可能含空串, 会被忽略)。
        weights: 与 texts 等长的权重(默认各为 1.0); 通常传 OCR 置信度。
    返回:
        共识文本; 无有效观测时返回空串。
    """
    obs: List[tuple] = []
    for i, t in enumerate(texts):
        if not t:
            continue
        w = float(weights[i]) if weights is not None and i < len(weights) else 1.0
        obs.append((t, max(w, 0.0)))
    if not obs:
        return ""
    if len({t for t, _ in obs}) == 1:
        return obs[0][0]

    total_w = sum(w for _, w in obs)
    # 每个不同文本的权重和
    text_w: Dict[str, float] = defaultdict(float)
    for t, w in obs:
        text_w[t] += w
    # 参考串: 权重和最大; 平票取更长者(更完整)
    ref = max(text_w, key=lambda t: (text_w[t], len(t)))

    n = len(ref)
    # 逐位字符票
    pos_votes: List[Dict[str, float]] = [defaultdict(float) for _ in range(n)]
    # 删除票: 该参考字符应被删除的权重
    del_votes: List[float] = [0.0] * n
    # 插入票: 在参考下标 i 之前插入某片段的权重 (含尾部锚点 n)
    ins_votes: List[Dict[str, float]] = [defaultdict(float) for _ in range(n + 1)]

    # 参考串自身的字符, 由"与参考串相同的观测"投出
    for i, ch in enumerate(ref):
        pos_votes[i][ch] += text_w[ref]

    for t, w in text_w.items():
        if t == ref:
            continue
        sm = difflib.SequenceMatcher(None, ref, t, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    pos_votes[i1 + k][t[j1 + k]] += w
            elif tag == "replace":
                r_len, t_len = i2 - i1, j2 - j1
                paired = min(r_len, t_len)
                for k in range(paired):
                    pos_votes[i1 + k][t[j1 + k]] += w
                for k in range(paired, r_len):        # 参考更长 → 删除票
                    del_votes[i1 + k] += w
                if t_len > r_len:                     # 观测更长 → 插入票
                    ins_votes[i2][t[j1 + paired:j2]] += w
            elif tag == "insert":
                ins_votes[i2][t[j1:j2]] += w
            elif tag == "delete":
                for k in range(i1, i2):
                    del_votes[k] += w

    out: List[str] = []
    for i in range(n):
        # 先处理该锚点处被严格多数采纳的插入片段
        best_seg, best_seg_w = "", 0.0
        for seg, wv in ins_votes[i].items():
            if wv > _INSERT_ADOPT_RATIO * total_w and wv > best_seg_w:
                best_seg, best_seg_w = seg, wv
        if best_seg:
            out.append(best_seg)
        # 该参考位: 若删除票过半则丢弃, 否则取票数最高的字符
        if del_votes[i] > _DELETE_ADOPT_RATIO * total_w:
            continue
        best_ch, best_ch_w = ref[i], pos_votes[i].get(ref[i], 0.0)
        for ch, wv in pos_votes[i].items():
            if wv > best_ch_w:
                best_ch, best_ch_w = ch, wv
        out.append(best_ch)
    # 尾部锚点的插入片段
    best_seg, best_seg_w = "", 0.0
    for seg, wv in ins_votes[n].items():
        if wv > _INSERT_ADOPT_RATIO * total_w and wv > best_seg_w:
            best_seg, best_seg_w = seg, wv
    if best_seg:
        out.append(best_seg)

    return "".join(out)


def plurality_share(texts: Sequence[str],
                    weights: Optional[Sequence[float]] = None) -> float:
    """返回共识串在观测中的占比(未加权), 用于度量共识的稳定性。"""
    valid = [t for t in texts if t]
    if not valid:
        return 0.0
    cons = consensus_text(texts, weights)
    return sum(1 for t in valid if t == cons) / len(valid)
