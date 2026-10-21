"""字符级多观测共识投票测试 (直接 python 运行, 不依赖 pytest)。

覆盖:
  (a) 全一致            -> 原串
  (b) 精确多数          -> 多数变体胜出
  (c) 每次错不同字      -> 逐位投票恢复正确串(旧"整行多数"在此失败)
  (d) 全/半角混用       -> 多数形态胜出
  (e) 单字插入          -> 仅严格多数时采纳
  (f) 单字删除          -> 仅严格多数时采纳
  (g) 空输入            -> ""
  (h) 权重              -> 高置信观测票更重
  (i) 确定性            -> 同输入两次输出一致
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.text_consensus import consensus_text  # noqa: E402

ALL_FAILURES = []


def run_case(name):
    print(f"\n=== {name} ===")

    def check(cond, msg):
        mark = "PASS" if cond else "FAIL"
        print(f"  [{mark}] {msg}")
        if not cond:
            ALL_FAILURES.append(f"{name}: {msg}")

    return check


def _old_exact_majority(texts):
    """旧行为: 整行精确字符串多数表决(保留用于对照)。"""
    counts = {}
    for t in texts:
        counts[t] = counts.get(t, 0) + 1
    return max(counts, key=counts.get)


def main():
    # (a) 全一致
    check = run_case("a) 全一致")
    check(consensus_text(["好的", "好的", "好的"]) == "好的", "三个相同观测 -> 原串")

    # (b) 精确多数
    check = run_case("b) 精确多数")
    check(consensus_text(["ABC", "ABC", "ABD"]) == "ABC", "2:1 -> 多数串")

    # (c) 每次错不同字: 旧整行多数失败, 新逐位投票恢复
    check = run_case("c) 每次错不同字")
    obses = ["逾期吗（包括逾期已还的)",   # 全角括号
             "逾期吗(包括逾期己还的)",   # 己/已 形近
             "逾期吗(包括逾期已还的)"]   # 正确
    got = consensus_text(obses)
    check(got == "逾期吗(包括逾期已还的)", f"逐位投票恢复正确串, got={got!r}")
    check(_old_exact_majority(obses) != "逾期吗(包括逾期已还的)",
          "对照: 旧整行多数在此无法恢复(说明新算法确有增益)")

    # (d) 全/半角括号混用
    check = run_case("d) 全/半角混用")
    got = consensus_text(["(请您打", "（请您打", "(请您打", "(请您打"])
    check(got == "(请您打", f"半角 3:1 -> 半角形态, got={got!r}")

    # (e) 单字插入仅严格多数采纳
    check = run_case("e) 单字插入")
    check(consensus_text(["明白了", "明白了", "明白了", "明折白了"]) == "明白了",
          "1/4 支持插入 -> 不采纳")
    check(consensus_text(["明折白了", "明折白了", "明折白了", "明白了"]) == "明折白了",
          "3/4 支持插入 -> 采纳(严格多数)")

    # (f) 单字删除仅严格多数采纳
    check = run_case("f) 单字删除")
    check(consensus_text(["还款只需归还本金", "还款只需归还本金",
                          "还款只需归还本金", "还款需归还本金"]) == "还款只需归还本金",
          "1/4 支持删除 -> 不采纳")
    check(consensus_text(["还款需归还本金", "还款需归还本金",
                          "还款需归还本金", "还款只需归还本金"]) == "还款需归还本金",
          "3/4 支持删除 -> 采纳(严格多数)")

    # (g) 空输入
    check = run_case("g) 空输入")
    check(consensus_text([]) == "", "空列表 -> 空串")
    check(consensus_text(["", ""]) == "", "全空串 -> 空串")

    # (h) 权重
    check = run_case("h) 权重")
    got = consensus_text(["甲", "乙", "乙"], [0.95, 0.05, 0.05])
    check(got == "甲", f"高置信单票压过低置信双票, got={got!r}")

    # (i) 确定性
    check = run_case("i) 确定性")
    seq = ["逾期吗（包括逾期已还的)", "逾期吗(包括逾期己还的)", "逾期吗(包括逾期已还的)"]
    check(consensus_text(seq) == consensus_text(seq), "同输入两次输出一致")

    print("\n" + "=" * 40)
    if ALL_FAILURES:
        print(f"FAILED: {len(ALL_FAILURES)} 项")
        for m in ALL_FAILURES:
            print(f"  - {m}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
