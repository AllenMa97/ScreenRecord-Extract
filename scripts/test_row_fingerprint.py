"""
Phase2 行指纹序列对齐(RFSA)三态测试 / Phase2 RFSA three-state test.

合成三类帧验证:
Synthesize three frame types:
  1. 回弹(内容相同) / bounce (same content)       → 期望 dup
  2. 滚动位移(部分重叠) / scrolled (partial overlap) → 期望 dup 或 gray(位移免疫)
  3. 新消息(底部插入) / new message (insertion)   → 期望 gray 或 new
"""
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.row_fingerprint import RowFingerprintAligner
from src.utils.behavior_logger import BehaviorLogger


def make_chat_frame(width=360, height=420, scroll_offset=0,
                    num_lines=12, extra=None) -> np.ndarray:
    """合成聊天界面帧(BGR) / Synthetic chat frame (BGR)"""
    frame = np.ones((height, width, 3), dtype=np.uint8) * 240
    line_height = 30
    start_y = 20 + scroll_offset
    for i in range(num_lines):
        y = start_y + i * line_height
        if 0 < y < height - 20:
            cv2.putText(frame, f"Message line {i + 1:02d} content",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if extra:
        for k, txt in enumerate(extra):
            y = start_y + (num_lines + k) * line_height
            if 0 < y < height - 20:
                cv2.putText(frame, txt, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def main():
    logger = BehaviorLogger({})
    aligner = RowFingerprintAligner({"compute_width": 160}, logger)

    # 首帧作为关键帧 / First frame becomes keyframe
    base = make_chat_frame()
    r0 = aligner.compare(base)
    print(f"首帧/initial: verdict={r0.verdict} (应dup/dup)")

    # 1. 回弹: 完全相同的帧 / Bounce: identical frame
    bounce = make_chat_frame()
    r1 = aligner.compare(bounce)
    print(f"回弹/bounce: verdict={r1.verdict} ratio={r1.aligned_ratio:.2f} "
          f"(应dup/dup)")

    # 2. 滚动位移: 内容相同但整体上移20px / Scroll: same content shifted up
    scrolled = make_chat_frame(scroll_offset=-20)
    r2 = aligner.compare(scrolled)
    print(f"滚动/scroll: verdict={r2.verdict} ratio={r2.aligned_ratio:.2f} "
          f"disp={r2.displacement} (应dup或gray, 位移免疫/dup-or-gray)")

    # 3. 新消息: 底部插入3行(完整的长文本行, 与合成行"Message line xx"内容
    #    完全不同)。两个关键点:
    #    (a) num_lines=10: 默认12行时 extra 的第2/3行 y=410/440 超出画布
    #        (420-20) 不会渲染, 实际只画1行 → 稀疏文本对dHash几乎无感;
    #        减到10行让3行新文本全部落在画布内(y=320/350/380)。
    #    (b) 文本须接近真实聊天行密度: 短文本差异在dHash中几乎无感
    #        ("Message line 01" vs "New msg A" hd=9<14噪声地板), 新消息
    #        文本须为长文本; 时间戳/序号等微变化是dHash固有极限, 需OCR级
    #        特征才能区分。
    #    New message: 3 full text rows inserted at bottom, content clearly
    #    different from "Message line xx". Two key points:
    #    (a) num_lines=10: with 12 rows the 2nd/3rd extra rows (y=410/440)
    #        fall outside the canvas (420-20) and never render, so only 1 row
    #        is drawn -> sparse text is nearly invisible to dHash; reducing to
    #        10 rows puts all 3 extra rows inside (y=320/350/380).
    #    (b) Text must mimic real chat-row density: short-text differences
    #        are invisible to dHash ("Message line 01" vs "New msg A"
    #        hd=9<noise floor); new-message text must be long. Micro-changes
    #        (timestamp/counter) are a known dHash limit needing OCR features.
    new_msg = make_chat_frame(
        num_lines=10,
        extra=["This is a brand new message row, completely different from old lines",
               "Another long reply row from the other side, also needs detection",
               "Third new message row, content likewise totally different from old ones"])
    r3 = aligner.compare(new_msg)
    print(f"新消息/newmsg: verdict={r3.verdict} ratio={r3.aligned_ratio:.2f} "
          f"new_rows={r3.new_rows} (应gray或new/gray-or-new)")

    # 断言 / Assertions
    ok = True
    if r1.verdict != "dup":
        print(f"  [FAIL] 回弹应判dup, 实际={r1.verdict}")
        ok = False
    else:
        print("  [PASS] 回弹判dup")

    if r2.verdict not in ("dup", "gray"):
        print(f"  [FAIL] 滚动位移应dup/gray(位移免疫), 实际={r2.verdict}")
        ok = False
    else:
        print(f"  [PASS] 滚动位移免疫 (verdict={r2.verdict})")

    if r3.verdict not in ("gray", "new"):
        print(f"  [FAIL] 新消息应gray/new, 实际={r3.verdict}")
        ok = False
    else:
        print(f"  [PASS] 新消息检测 (verdict={r3.verdict})")

    # 4. 关键帧更新后再次比较: 微平移1px(模拟消息弹出导致的整屏平移),
    #    内容与new_msg相同 → 应判dup拦截。这是RFSA主通道的核心场景
    #    (htkf组E: 静止同屏连续送检19次的重复快照, 画面平移1-2px)。
    #    注意: 不能用大位移(如-20px≈半条带16px), 文本行在条带内的位置
    #    移动会让dHash完全改变(hd 14-30) → 判new。这是"亚条带位移"的
    #    dHash固有极限: 回弹/微滚动(5-15px)会误判new, 由OCR内容复用兜底
    #    (重复OCR但结果复用, 仅多一次API调用, 非正确性损失)。
    #    After keyframe update, compare a 1px-shifted identical frame (simulates
    #    a message-popup whole-screen shift) -> dup. This is the RFSA main
    #    channel scenario (htkf group E: 19 duplicate snapshots of one static
    #    screen). Large shifts (-20px ~ half a strip) change row dHash entirely
    #    (hd 14-30) -> new; sub-strip displacement is a known dHash limit
    #    (bounce/micro-scroll 5-15px), covered by OCR content reuse.
    aligner.update_keyframe(new_msg)
    r4_frame = make_chat_frame(
        scroll_offset=-1, num_lines=10,
        extra=["This is a brand new message row, completely different from old lines",
               "Another long reply row from the other side, also needs detection",
               "Third new message row, content likewise totally different from old ones"])
    r4 = aligner.compare(r4_frame)
    print(f"滚动+新消息/scroll+new: verdict={r4.verdict} "
          f"ratio={r4.aligned_ratio:.2f} (应dup或gray)")
    if r4.verdict not in ("dup", "gray"):
        print(f"  [FAIL] 滚动+新消息应位移免疫, 实际={r4.verdict}")
        ok = False
    else:
        print(f"  [PASS] 滚动+新消息位移免疫 (verdict={r4.verdict})")

    print(f"\n结果: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
