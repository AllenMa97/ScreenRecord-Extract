# -*- coding: utf-8 -*-
"""单元测试: 模板消息的数字锚点守卫(休眠/宽限行不得吸收"同模板、数字不同"的观测)。

Unit test for the template digit-anchor guard: a dormant/grace tracked line must
NOT absorb an observation that is the same template with different digits.

复现 `自行模拟对话记录4.mp4` 的 #8/#10 顺序颠倒: 聊天消息高度统一, 滚动一整条后
`订单SN0014` 精确落进 `订单SN0010` 跟踪行的框(IoU≈1), 而两者文本相似度 0.91 又远超
"同框放宽门限" 0.50 → 休眠行吸收该观测并改判, 使后出现的 SN0014 继承了更早的
first_timestamp, 最终输出把它排到 SN0010 之前。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_types import TextLine, TrackedLine                      # noqa: E402
from src.message_bubble_tracker import MessageBubbleTracker           # noqa: E402
from src.utils.text_utils import (anchors_compatible, digit_anchor,   # noqa: E402
                                  template_digit_conflict)

_ok = True

SN10 = "订单SN0010已确认收款"
SN14 = "订单SN0014已确认收款"
BOX = (115, 500, 184, 63)      # 消息高度统一 → 滚动一整条后新消息落进同一个框


def check(name, cond, detail=""):
    global _ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        _ok = False


class StubLogger:
    """BehaviorLogger 的最小替身: 只收集记录, 不落盘。"""

    def __init__(self):
        self.records = []

    def log(self, module, event, detail, frame_id=None, extra=None):
        self.records.append((module, event, detail, extra or {}))


def make_tracker(**cfg):
    t = MessageBubbleTracker(cfg, StubLogger())
    t.set_roi_dimensions(1100, 576)
    return t


def make_tracked(text, frame_id=10, bbox=BOX):
    line = TextLine(text=text, bbox=bbox, confidence=1.0,
                    frame_id=frame_id, timestamp=1.0)
    return TrackedLine(line_id="a" * 12, text=text, bbox=bbox,
                       first_bbox=bbox, confidence=1.0,
                       first_seen_frame=frame_id, last_seen_frame=frame_id,
                       first_timestamp=1.0, last_timestamp=1.0,
                       observations=[line], is_new=False)


def main():
    # ---- 1) 数字锚点判据本身 ----
    check("锚点抽取", digit_anchor(SN10) == "0010", digit_anchor(SN10))
    check("同模板不同数字 → 冲突", template_digit_conflict(SN10, SN14))
    check("完全相同 → 不冲突", not template_digit_conflict(SN10, SN10))
    # OCR 漏读一位数字: '001' 是 '0010' 的子序列 → 相容, 不能当成另一条消息
    check("漏读一位数字 → 不冲突",
          not template_digit_conflict(SN10, "订单SN001已确认收款"))
    check("锚点子序列相容", anchors_compatible("001", "0010"))
    check("锚点 0010/0014 不相容", not anchors_compatible("0010", "0014"))
    # 无数字的普通行不受影响
    check("无数字 → 不冲突", not template_digit_conflict("在吗", "收到，你说"))
    # 骨架不同(不是模板近重复) → 交给常规相似度判据, 守卫不介入
    check("骨架不同 → 不冲突",
          not template_digit_conflict("您的快递8812已到", "订单SN0014已确认收款"))

    # ---- 2) 匹配层: 休眠行拒绝, 热行放行 ----
    tr = make_tracker()
    obs14 = TextLine(text=SN14, bbox=BOX, confidence=1.0,
                     frame_id=60, timestamp=2.0)
    tracked10 = make_tracked(SN10, frame_id=10)

    check("休眠行(IoU=1)拒绝同模板异数字观测",
          tr._match_strength(obs14, tracked10, False, 60) is None)
    check("守卫关闭时同一对可匹配(证明判据来自守卫)",
          make_tracker(template_digit_guard_sim=0)._match_strength(
              obs14, tracked10, False, 60) is not None)
    # 热行(刚见过)的数字差异更可能是 OCR 噪声, 仍交给投票层吸收
    hot = make_tracked(SN10, frame_id=59)
    check("热行不被守卫拦截",
          tr._match_strength(obs14, hot, False, 60) is not None)
    # 宽限行同样受约束
    check("宽限行拒绝同模板异数字观测",
          tr._match_strength(obs14, tracked10, True, 60) is None)
    # 同锚点的噪声观测(OCR 把 '已' 读成 '己')在休眠行上仍可匹配
    noisy = TextLine(text="订单SN0010己确认收款", bbox=BOX, confidence=1.0,
                     frame_id=60, timestamp=2.0)
    check("休眠行仍吸收同锚点的噪声观测",
          tr._match_strength(noisy, tracked10, False, 60) is not None)
    # 外观库里已出现过该锚点(行本身已被污染) → 不再拒绝, 避免抖动
    mixed = make_tracked(SN10, frame_id=10)
    mixed.observations.append(TextLine(text=SN14, bbox=BOX, confidence=1.0,
                                       frame_id=20, timestamp=1.5))
    check("外观库含该锚点时不拒绝",
          tr._match_strength(obs14, mixed, False, 60) is not None)

    # ---- 3) 端到端: 顺序不再颠倒 ----
    def run(guard):
        """两条消息先后落进同一个框; 返回按 first_timestamp 排序的 (时间, 文本)。

        置信度取 0.99/1.00 与实录一致: 字符级共识的参考串是"权重和最大者", 后到的
        SN0014 权重更高 → 整行被改判成 SN0014, 而 first_timestamp 仍是 SN0010 的。
        """
        t = make_tracker(**({} if guard else {"template_digit_guard_sim": 0}))
        t.update(10, 1.0, [TextLine(text=SN10, bbox=BOX, confidence=0.99,
                                    frame_id=10, timestamp=1.0)])
        t.update(60, 2.0, [TextLine(text=SN14, bbox=BOX, confidence=1.0,
                                    frame_id=60, timestamp=2.0)])
        lines = t._active_lines + t._grace_lines + t._completed_lines
        return sorted(((l.first_timestamp, l.text) for l in lines
                       if l.text in (SN10, SN14)))

    off = run(guard=False)
    check("对照(守卫关闭)复现顺序颠倒: SN0014 继承 t=1.0 且 SN0010 被吞",
          off == [(1.0, SN14)], str(off))
    on = run(guard=True)
    check("守卫开启后两条各自独立且顺序正确",
          on == [(1.0, SN10), (2.0, SN14)], str(on))

    print("\n" + ("ALL PASS" if _ok else "SOME FAILED"))
    return 0 if _ok else 1


if __name__ == "__main__":
    sys.exit(main())
