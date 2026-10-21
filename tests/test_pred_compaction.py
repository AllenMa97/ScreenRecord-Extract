"""Pred 压实 (输出冗余抑制) 测试 — 直接 python 运行, 不依赖 pytest。

覆盖两层:
  A. 同一消息的 OCR 变体判重 `_is_same_message` / `_variant_key` / `_digit_anchor`
     - 正例: 空白差异、全半角差异、OCR 错字(.Ig/.Ic/.lc)、滚动截断读法
     - 反例(必须分开): 仅数字不同的模板近重复消息、短回复被长消息包含、
       不同消息的偶然相似
  B. `_rejoin_llm_sections` 端到端: LLM 把同一消息输出成多条(变体/残影),
     确定性还原后应折叠为一条, 保留首次时间 + 最完整读法
  C. `_clean_ocr_lines` 低置信度噪声行过滤(气泡波形图标 / 滚动半截行)

Run: python tests/test_pred_compaction.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_types import TextLine, TrackedLine  # noqa: E402
from src.audio.report_merge import _digit_anchor as _cross_anchor  # noqa: E402
from src.audio.report_merge import (_anchor_compatible,  # noqa: E402
                                    _is_cross_path_duplicate, _loose)
from src.llm_client import (LlmClient, _digit_anchor, _is_same_message,  # noqa: E402
                            _variant_key)

ALL_FAILURES = []


def run_case(name):
    print(f"\n=== {name} ===")

    def check(cond, msg):
        mark = "PASS" if cond else "FAIL"
        print(f"  [{mark}] {msg}")
        if not cond:
            ALL_FAILURES.append(f"{name}: {msg}")

    return check


FULL_A = ("有的，我导出成incident-20261016-0042.Ig，里面能看到线程池被打满之后"
          "重试队列一直，最后把连接池的句柄耗尽了，这块你重点看下超时和重试配置")
FULL_B = ("有的，我导出成 incident-20261016-0042.Ic，里面能看到线程池被打满之后"
          "重试队列一直，最后把连接池的句柄耗尽了，这块你重点看下超时和重试配置")
FULL_C = ("有的，我导出成 incident-20261016-0042.lc，里面能看到线程池被打满之后"
          "重试队列一直，最后把连接池的句柄耗尽了，这块你重点看下超时和重试配置")
TRUNC = "有的，我导出成 incident-20261016-0042.Ic"
# video4 GT#12 实测: 长消息完整读法 vs 滚动回看时只剩尾部的读法(真后缀)
FULL_12 = "上次那份接口文档里，字段说明是不是漏了几个？...麻烦你抽空核对一下原始设计稿"
TAIL_12 = "明是不是漏了几个？...麻烦你抽空核对一下原始设计稿"


def test_same_message():
    chk = run_case("A. 同一消息的 OCR 变体判重")
    # 正例
    chk(_is_same_message("那我调到8次试试", "那我调到8 次试试"),
        "空白差异 → 同一条")
    chk(_is_same_message("换了，http://a/v2 这个", "换了，http://a/v2这个"),
        "URL 尾随空白差异 → 同一条")
    chk(_is_same_message("环境变量里那个 retry.max 是不是写小了",
                         "环境变量里那个retry.max是不是写小了"),
        "中英混排空白差异 → 同一条")
    chk(_is_same_message(FULL_A, FULL_B), "OCR 错字 .Ig/.Ic → 同一条")
    chk(_is_same_message(FULL_B, FULL_C), "OCR 错字 .Ic/.lc → 同一条")
    chk(_is_same_message(TRUNC, FULL_B), "滚动截断读法(前缀) → 同一条")
    chk(_is_same_message(TAIL_12, FULL_12), "滚动回看读法(后缀) → 同一条")
    chk(_is_same_message("abc", "abc"), "完全相同 → 同一条")
    # 反例
    chk(not _is_same_message("您的快递8812已到驿站，请凭取件码签收",
                             "您的快递8816已到驿站，请凭取件码签收"),
        "仅数字不同(8812/8816) → 必须分开")
    chk(not _is_same_message("刚核对过了，订单SN0014已确认收款",
                             "订单SN0010已确认收款"),
        "后缀同形但数字锚点不同(SN0014/SN0010) → 必须分开")
    chk(not _is_same_message("我三点前弄完", "明白了，我三点前弄完"),
        "短回复恰为长消息后缀(不足8字) → 必须分开")
    chk(not _is_same_message("你的快递8814也到了，我顺手帮你带上来",
                             "你的快递8818也到了，我顺手帮你带上来"),
        "仅数字不同(8814/8818) → 必须分开")
    chk(not _is_same_message("好", "好，我下午一起看"),
        "极短回复被长消息包含 → 必须分开")
    chk(not _is_same_message("收到，我先改一版", "收到，我更新一下本地配置"),
        "同前缀不同消息 → 必须分开")
    chk(not _is_same_message("有堆栈吗？发我一份", "你把复现步骤发我"),
        "无关消息 → 必须分开")
    chk(not _is_same_message("那我调到8次试试", "换了，http://a/v2 这个"),
        "数字锚点相同但内容无关 → 必须分开")
    chk(not _is_same_message("", FULL_B), "空文本 → 不判重")


def test_keys():
    chk = run_case("A2. 归一化键与数字锚点")
    chk(_variant_key("那 我 调 到 8 次") == _variant_key("那我调到8次"),
        "去空白后键一致")
    chk(_variant_key("ＡＢＣ１２３") == _variant_key("abc123"),
        "NFKC 全半角 + 小写统一")
    chk(_digit_anchor("repro-20261016-0037.md") == _digit_anchor(
        "repro202610160037MD"), "标点抖动不改变数字锚点")
    chk(_digit_anchor("8812") != _digit_anchor("8816"), "数字本身不同 → 锚点不同")


def _mk_tracked(text: str, ts: float, bbox=(50, 100, 400, 120)) -> TrackedLine:
    return TrackedLine(line_id=f"L{ts}", text=text, bbox=bbox, first_bbox=bbox,
                       confidence=0.99, first_seen_frame=1, last_seen_frame=1,
                       first_timestamp=ts, last_timestamp=ts, observations=[])


def test_rejoin_collapse():
    chk = run_case("B. _rejoin_llm_sections 折叠同一消息的重复输出")
    llm = LlmClient.__new__(LlmClient)
    llm._roi_width = 0
    llm._roi_height = 0
    llm._chat_enabled = False          # _classify_side 直接返回 ""
    llm._center_margin = 20
    llm._frame_provider = None

    # 权威行: 同一条消息被跟踪成 4 个变体(不同时间戳) + 另一条独立消息
    tracked = [
        _mk_tracked("环境变量里那个 retry.max是不是写小了", 20.0),
        _mk_tracked("环境变量里那个retry.max 是不是写小了", 30.0),
        _mk_tracked("环境变量里那个retry.max是不是写小了", 40.0),
        _mk_tracked("环境变量里那个 retry.max 是不是写小了", 50.0),
        _mk_tracked("您的快递8812已到驿站，请凭取件码签收", 60.0),
        _mk_tracked("您的快递8816已到驿站，请凭取件码签收", 70.0),
    ]
    sections = [{"timestamp": 20.0, "content": "\n".join([
        "- +20s [收] ：环境变量里那个 retry.max是不是写小了",
        "- +30s [收] ：环境变量里那个retry.max 是不是写小了",
        "- +40s [收] ：环境变量里那个retry.max是不是写小了",
        "- +50s [收] ：环境变量里那个 retry.max 是不是写小了",
        "- +60s [收] ：您的快递8812已到驿站，请凭取件码签收",
        "- +70s [收] ：您的快递8816已到驿站，请凭取件码签收",
    ])}]
    llm._rejoin_llm_sections(sections, tracked)
    out = sections[0]["content"].split("\n")
    chk(len(out) == 3, f"4 个变体折叠为 1 条 + 数字模板近重复保留 2 条 (实际 {len(out)})")
    chk(sum("retry.max" in ln.replace(" ", "") for ln in out) == 1,
        "retry.max 消息只输出 1 次")
    chk(any("8812" in ln for ln in out) and any("8816" in ln for ln in out),
        "8812 与 8816 两条都保留")
    chk(out[0].startswith("- +20s"), f"保留首次出现的时间 (实际 {out[0][:12]!r})")


def test_rejoin_prefers_fullest():
    chk = run_case("B2. 重复时保留最完整读法")
    llm = LlmClient.__new__(LlmClient)
    llm._roi_width = 0
    llm._roi_height = 0
    llm._chat_enabled = False
    llm._center_margin = 20
    llm._frame_provider = None

    tracked = [
        _mk_tracked(TRUNC, 10.0),
        _mk_tracked(FULL_B, 30.0),
    ]
    sections = [{"timestamp": 10.0, "content": "\n".join([
        f"- +10s [发] ：{TRUNC}",
        f"- +30s [发] ：{FULL_B}",
    ])}]
    llm._rejoin_llm_sections(sections, tracked)
    out = sections[0]["content"].split("\n")
    chk(len(out) == 1, f"截断读法与完整读法折叠为 1 条 (实际 {len(out)})")
    chk("线程池被打满" in out[0], "保留的是最完整读法")
    chk(out[0].startswith("- +10s"), "时间仍取首次出现")


def test_ocr_conf_filter():
    chk = run_case("C. 低置信度 OCR 噪声行过滤")
    from src.pipeline import ExtractionPipeline
    pipe = ExtractionPipeline.__new__(ExtractionPipeline)
    pipe._ocr_min_conf = 0.85
    pipe._scroll_conf = 0.65

    lines = [
        TextLine(text="川咖", bbox=(81, 79, 132, 112), confidence=0.7182),
        TextLine(text="川ll", bbox=(400, 730, 450, 760), confidence=0.767),
        TextLine(text="的和重试直", bbox=(433, 1, 529, 8), confidence=0.738),
        TextLine(text="境支重全那个Tey.aX是环是与小", bbox=(55, 1, 338, 8),
                 confidence=0.7009),
        TextLine(text="那 我调到 8 次试试", bbox=(53, 218, 189, 239),
                 confidence=0.9515),
        TextLine(text="环境变量里那个retry.max 是不是写小了",
                 bbox=(54, 380, 343, 397), confidence=0.9797),
    ]
    kept = pipe._clean_ocr_lines(lines, scroll_frame=False, roi_w=540, roi_h=780)
    texts = [l.text for l in kept]
    chk(len(kept) == 2, f"仅保留 2 条正文 (实际 {len(kept)}: {texts})")
    chk(all("川" not in t and "的和重试直" not in t for t in texts),
        "波形图标/半截残片被过滤")


def test_wrap_tail_leftover():
    chk = run_case("E. 换行尾片段归属 (滚动后父行滚出、只剩尾行可见)")

    def side(line):
        x1, _, x2, _ = line.first_bbox or line.bbox
        return "发" if (540 - x2) + 20 < x1 else "收"

    from src.bubble_merger import absorb_wrap_tail_leftovers

    parent = _mk_tracked("行，我下午发，顺便把那两个小的兼容性问题起写上",
                         30.0, bbox=(177, 698, 533, 735))
    tail = _mk_tracked("起写上", 95.7, bbox=(478, 0, 532, 15))
    kept, n = absorb_wrap_tail_leftovers([parent, tail], side)
    chk(n == 1 and kept == [parent], f"尾行归回父消息 (吸收 {n} 条)")

    # 反例 1: 时序相反 —— 短消息先出现、长消息后出现, 后缀关系成立但不吸收
    short_first = _mk_tracked("晚点说", 10.0, bbox=(60, 100, 140, 120))
    long_later = _mk_tracked("行，那晚点说", 30.0, bbox=(60, 200, 200, 220))
    kept, n = absorb_wrap_tail_leftovers([short_first, long_later], side)
    chk(n == 0 and len(kept) == 2, f"短消息先出现 → 不吸收 (吸收 {n} 条)")

    # 反例 2: 只是前缀/无关 → 不吸收
    a = _mk_tracked("好的，我提前把文档整理一下", 30.0, bbox=(60, 100, 400, 120))
    b = _mk_tracked("好的", 40.0, bbox=(60, 200, 130, 220))
    kept, n = absorb_wrap_tail_leftovers([a, b], side)
    chk(n == 0, f"前缀(非后缀) → 不吸收 (吸收 {n} 条)")

    # 反例 3: 后缀成立但收发方向不同 → 不吸收
    c = _mk_tracked("行，那晚点说", 10.0, bbox=(60, 100, 200, 120))       # 收
    d = _mk_tracked("晚点说", 30.0, bbox=(400, 200, 520, 220))            # 发
    kept, n = absorb_wrap_tail_leftovers([c, d], side)
    chk(n == 0, f"方向不同 → 不吸收 (吸收 {n} 条)")

    # 正例 2: 多行卡片(转账)滚到 ROI 边缘被遮挡、只露出一行 → 子行归回卡片
    card = _mk_tracked("¥200.00已收款\\n微信转账", 51.7, bbox=(129, 74, 220, 149))
    amount = _mk_tracked("¥200.00", 125.4, bbox=(129, 74, 220, 101))
    kept, n = absorb_wrap_tail_leftovers([card, amount], side)
    chk(n == 1 and kept == [card], f"卡片子行归回卡片 (吸收 {n} 条)")

    # 反例 4: 单行父消息中片段是子串(非后缀) → 不吸收(可能是两条独立消息)
    single = _mk_tracked("这周末我去爬山，你要不要一起", 30.0, bbox=(60, 100, 400, 120))
    frag = _mk_tracked("我去爬山", 40.0, bbox=(60, 200, 200, 220))
    kept, n = absorb_wrap_tail_leftovers([single, frag], side)
    chk(n == 0, f"单行父消息中的子串 → 不吸收 (吸收 {n} 条)")


def test_cross_path():
    chk = run_case("D. 视觉×音频 跨路去重")

    def vis(*texts):
        return [(_loose(t), _cross_anchor(t)) for t in texts]

    chk(_is_cross_path_duplicate("有我写成repro202610160037MD了",
                                 vis("有，我写成 repro-20261016-0037.md了")),
        "ASR(去标点/大写) vs 视觉(带连字符) → 判重")
    chk(_is_cross_path_duplicate("还有个是，那个取件马叫燕藕限会失败",
                                 vis("还有个事，那个取件码校验偶现会失败")),
        "ASR 同音错字 → 判重")
    chk(not _is_cross_path_duplicate("您的快递8816已到驿站，请凭取件码签收",
                                     vis("您的快递8812已到驿站，请凭取件码签收")),
        "数字锚点不同(8812/8816) → 不判重")
    chk(not _is_cross_path_duplicate("你的快递8818也到了，我顺手帮你带上来",
                                     vis("你的快递8814也到了，我顺手帮你带上来")),
        "数字锚点不同(8814/8818) → 不判重")
    chk(not _is_cross_path_duplicate("收到，我更新一下本地配置",
                                     vis("收到，我先改一版")),
        "同前缀不同消息(相似度不足) → 不判重")
    chk(not _is_cross_path_duplicate("好", vis("好，我下午一起看")),
        "过短文本 → 不判重")
    chk(_cross_anchor("repro-20261016-0037.md") == _cross_anchor("repro202610160037MD"),
        "标点抖动不改变跨路数字锚点")

    # --- 数字锚点"子序列"放宽(容忍 ASR 多念/少念一个数字字符) ---
    # 锚点是 (字面锚点, 折算锚点), 故用 _cross_anchor 生成后再比对。
    chk(_anchor_compatible(_cross_anchor("2026101600421"), _cross_anchor("20261016004211")),
        "真实数字锚点只差一个插入字符 → 相容")
    chk(_anchor_compatible(_cross_anchor("102030428080"), _cross_anchor("1020304280802")),
        "URL 尾号被 ASR 漏读(v2→VR) → 相容")
    chk(not _anchor_compatible(_cross_anchor("8812"), _cross_anchor("8816")),
        "数字**替换**(8812/8816) → 不相容")
    chk(not _anchor_compatible(_cross_anchor("8876540"), _cross_anchor("8876546")),
        "长数字替换(8876540/8876546) → 不相容")
    chk(not _anchor_compatible(_cross_anchor("12"), _cross_anchor("123")),
        "短锚点偶然包含(12⊂123) → 不相容(长度保护)")
    chk(_anchor_compatible(_cross_anchor("顺便把那两个小的兼容性问题也一起写上"),
                           _cross_anchor("顺便把那两个小的兼容性问题起写上")),
        "纯口语中文数字折算噪声(\"一起\"多出孤立数字) → 相容")

    # --- 两类真实未命中的跨路重复(ASR 与视觉面貌差异大) ---
    chk(_is_cross_path_duplicate(
        "有的我导出成incident202610160042log，里面能看到县城池被打满之后，重试队列"
        "一直堆积，最后把连接池的聚柄耗尽了。这块你重点看一下超时和重试配置",
        vis("有的，我导出成incident-20261016-0042.lc，里面能看到线程池被打满之后"
            "重试队列一直，最后把连接池的句柄耗尽了，这块你重点看下超时和重试配置")),
        "ASR 同音错字+多念'一' → 判重")
    chk(_is_cross_path_duplicate(
        "换了HTTP冒号斜杠斜杠10.20.30.428080APIVR这个是新的",
        vis("换了，http://10.20.30.42:8080/api/v2 这个")),
        "ASR 口述 URL 标点(冒号/斜杠) → 判重")


def main():
    test_same_message()
    test_keys()
    test_rejoin_collapse()
    test_rejoin_prefers_fullest()
    test_ocr_conf_filter()
    test_wrap_tail_leftover()
    test_cross_path()
    print("\n" + "=" * 60)
    print(f"FAIL: {len(ALL_FAILURES)}")
    for f in ALL_FAILURES:
        print("  -", f)
    print("全部通过 / ALL PASSED" if not ALL_FAILURES else "存在失败 / FAILURES")
    print("=" * 60)
    return 1 if ALL_FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
