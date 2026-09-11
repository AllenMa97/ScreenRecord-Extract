"""滚动一致性门控(Rolling-consistency gate)算法验证

直接构造"行指纹序列"驱动 RowFingerprintAligner._align, 验证核心三类场景:
  1. 真滚动(同内容垂直平移): 相同行指纹序列位移 → 无实质新行 → dup。
  2. 滚动 + 底部挤入新消息: 位移一致 + 少量全新行 → 不整屏判 new。
  3. 整屏替换(位移分散): 行指纹差异巨大且位移不一致 → 门控失效, 判 new 保召回。
  4. 同内容静止 + 压缩噪声(小 bit 翻转): 判 dup。

不同"内容"用高16位不同的 int 编码, 保证 hamming_distance > noise_floor(14);
同一内容位移则用相同 int, 位移对应序列 offset。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.row_fingerprint import RowFingerprintAligner


class NullLogger:
    def __init__(self):
        self._buffer = []

    def log(self, module, event, detail, **kw):
        self._buffer.append((module, event, detail))
        return None


def build_config(**over):
    cfg = {
        "rf_strip_height": 16,
        "rf_dhash_size": 8,
        "rf_gap_penalty": -0.3,
        "rf_match_floor": 0.85,
        "rf_dup_ratio": 0.8,
        "rf_new_ratio": 0.4,
        "rf_noise_floor": 14,
        "rf_history_len": 3,
        "rf_rigid_roll_ratio": 0.7,
        "rf_rigid_over_ratio": 0.5,
        "compute_width": 160,
    }
    cfg.update(over)
    return cfg


def content(k):
    """把内容标识 k 编码为 64-bit 行指纹: 高32位 = 确定性伪随机(期望 hd≈16 > 14),
    低位 = 固定噪声空间。不同 k -> 高32位随机 → 满足"不同文本行 hd>=16"。"""
    import random
    rng = random.Random(f"content-{k}")
    return (rng.getrandbits(32) << 32) | 0x00000000000015


def ctx(idx, k):
    """同一内容 k 在不同位移位置(idx): 高32位 = 内容k指纹, 低位编码 idx。
    同内容不同位置 → 高32位相同, hd 小; 不同内容 → 高32位随机, hd 大。"""
    return (content(k) & 0xFFFFFFFF00000000) | (idx & 0xFF)


def noise(fp, nflip, seed):
    """同内容静止压缩噪声: 仅翻转低16位(内容指纹在高32位保持不变)。"""
    import random
    rng = random.Random(seed)
    for _ in range(nflip):
        fp ^= (1 << rng.randint(0, 15))
    return fp


def main():
    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name} {detail}")
        if not cond:
            failures.append(name)

    cfg = build_config()

    # ---------------- 场景1: 真滚动(同内容位移, 内容未增) ----------------
    print("\n== 场景1: 真滚动(同内容垂直平移, 内容未增) ==")
    # prev 5 行内容 C1..C5; cur = C2..C5 (顶部 C1 滚出), 位移 -1
    prev1 = [ctx(0, 11), ctx(0, 12), ctx(0, 13), ctx(0, 14), ctx(0, 15)]
    cur1 = [ctx(1, 12), ctx(1, 13), ctx(1, 14), ctx(1, 15)]
    a1 = RowFingerprintAligner(cfg, NullLogger())
    ratio, disp, new_rows = a1._align(prev1, cur1)
    print(f"  对齐率={ratio:.2f}, 位移={disp}, 新行={new_rows}, "
          f"滚动一致性={a1._last_rigid_roll}")
    check("真滚动无新内容(内容未增)", not new_rows,
          f"got new_rows={new_rows}")
    check("真滚动对齐率高(判dup)", ratio >= 0.75,
          f"got ratio={ratio:.2f}")

    # ---------------- 场景2: 滚动 + 底部挤入新行, 位移一致 ----------------
    print("\n== 场景2: 滚动 + 底部挤入一件新内容 ==")
    prev2 = [ctx(0, 21), ctx(0, 22), ctx(0, 23), ctx(0, 24), ctx(0, 25)]
    # cur: C22..C25 位移 + 底部挤入全新内容 C99
    cur2 = [ctx(1, 22), ctx(1, 23), ctx(1, 24), ctx(1, 25),
            ctx(0, 26), ctx(0, 27)]  # 2 行位移+2新
    a2 = RowFingerprintAligner(cfg, NullLogger())
    ratio, disp, new_rows = a2._align(prev2, cur2)
    print(f"  对齐率={ratio:.2f}, 位移={disp}, 新行={new_rows}, "
          f"滚动一致性={a2._last_rigid_roll}")
    check("滚动+新增: 门控不整屏误判", len(new_rows) <= 2,
          f"got new_rows={new_rows}")

    # ---------------- 场景3: 整屏替换(位移分散, 门控失效保召回) ----------------
    print("\n== 场景3: 整屏内容替换(位移分散) ==")
    prev3 = [content(31), content(32), content(33), content(34)]
    cur3 = [content(51), content(52), content(53), content(54)]  # 完全不同内容
    a3 = RowFingerprintAligner(cfg, NullLogger())
    ratio, disp, new_rows = a3._align(prev3, cur3)
    print(f"  对齐率={ratio:.2f}, 位移={disp}, 新行={new_rows}, "
          f"滚动一致性={a3._last_rigid_roll}")
    check("整屏替换判为新内容(门控不误伤召回)",
          len(new_rows) >= 3 or ratio == 0.0,
          f"got new_rows={new_rows} ratio={ratio:.2f}")

    # ---------------- 场景4: 同内容静止 + 压缩噪声 ----------------
    print("\n== 场景4: 同内容静止 + 轻微压缩噪声 ==")
    prev4 = [content(61), content(62), content(63), content(64),
             content(65), content(66)]
    cur4 = [noise(content(61), 1, 1), noise(content(62), 2, 2),
            noise(content(63), 1, 3), noise(content(64), 2, 4),
            noise(content(65), 1, 5), noise(content(66), 2, 6)]
    a4 = RowFingerprintAligner(cfg, NullLogger())
    ratio, disp, new_rows = a4._align(prev4, cur4)
    print(f"  对齐率={ratio:.2f}, 位移={disp}, 新行={new_rows}, "
          f"滚动一致性={a4._last_rigid_roll}")
    check("静止加噪判重复(无实质新行)", not new_rows,
          f"got new_rows={new_rows}")

    # -------- 场景5: 滚动记忆库(累积行指纹) — 滚动往返/回看内容去重 --------
    print("\n== 场景5: 滚动记忆库 - 滚动往返回看(已在库中)不再判新 ==")
    a5 = RowFingerprintAligner(cfg, NullLogger())
    # 先"见过"内容 {10,11,12}(如早先滚上屏并OCR过)
    a5._commit_subsequences([content(10), content(11), content(12)])
    check("库已接收3个内容行", len(a5._seen_rows) == 3,
          f"got lib={len(a5._seen_rows)}")

    # 当前帧 cur: 3行是回看(10,11,12已在库) + 1行全新(20)
    cur5 = [content(10), content(11), content(12), content(20)]
    # 假设相对窗口这4行都被判为"新"(滚动滚出窗口后, 窗口无法匹配)
    kept, dropped = a5._filter_seen_rows(cur5, [0, 1, 2, 3])
    check("回看3行被剔除、仅新行保留",
          kept == [3] and dropped == 3,
          f"got kept={kept} dropped={dropped}")

    # -------- 场景6: 滚回历史屏 = 全库已见行 → 零新行 → 判dup --------
    print("\n== 场景6: 完整滚回历史屏(所有行都已在库) → 无实质新行 ==")
    a6 = RowFingerprintAligner(cfg, NullLogger())
    a6._commit_subsequences([content(31), content(32), content(33)])
    cur6 = [content(33), content(31), content(32)]   # 打乱顺位滚动回来
    kept, dropped = a6._filter_seen_rows(cur6, [0, 1, 2])
    check("历史屏回看不产生新行", kept == [] and dropped == 3,
          f"got kept={kept} dropped={dropped}")

    # -------- 场景7: 位移免疫 — 相同内容出现在不同位移仍被库识别 --------
    print("\n== 场景7: 位移/条带错位免疫 - 库近邻判定掩盖同内容位移 ==")
    a7 = RowFingerprintAligner(cfg, NullLogger())
    # 在库中内容41出现在位移idx=1
    a7._commit_subsequences([ctx(1, 41)])
    # 回看时同一内容41出现在不同位移idx=7(低字节变化, 高32位内容指纹不变)
    cur7 = [ctx(7, 41)]
    seen = a7._row_seen_in_lib(cur7[0])
    check("同内容不同位移仍判已见(位移免疫)", seen is True,
          f"got seen={seen}")

    print("\n================ 结果汇总 ================")
    if failures:
        print(f"FAILED: {len(failures)} 项未通过 -> {failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())