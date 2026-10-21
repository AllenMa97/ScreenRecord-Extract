# -*- coding: utf-8 -*-
"""单元测试: 居中系统提示过滤的「行宽守卫」(宽折行正文不得被当系统提示删除)。

Unit test for the line-width guard in the centered-system-prompt filter: a wide
wrapped line of a real message must NOT be dropped as a system prompt.

复现 `自行模拟对话记录4.mp4` 的 GT#12 长消息主体丢失: 一条长消息被界面折成三行,
宽气泡的折行横向铺满 ROI(左右留白 129/135, 差 6px < center_margin 20) → classify_side
判不出方向; 水平中心 285 又贴着 ROI 中心 288(容差 46) → 判为「居中」; 文本 13/15 字
<= _SYS_MAX_LEN(20) → 走短文本捷径, 头行与中行被 drop_system_prompts 删掉, 整条长
消息只剩尾行(尾行较窄偏左, classify_side 判出「收」而幸存)。

守卫口径: 短文本捷径只对**窄行**生效; 措辞明确的系统提示仍走 SYSTEM_PROMPT_RE,
与行宽无关(否则长的「你已添加对方为好友…」会漏网)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bubble_merger import (SYSTEM_PROMPT_RE, _SYS_MAX_WIDTH_RATIO,  # noqa: E402
                               classify_side, drop_system_prompts,
                               is_centered_system_prompt)
from src.data_types import TrackedLine                            # noqa: E402

_ok = True

ROI_W = 576          # video4 实测 ROI 宽度
CM = 20              # center_margin

# video4 GT#12 三折行的真实 bbox(帧163/178, 取自 OCR 缓存)
HEAD_BOX = (129, 767, 441, 795)     # 宽占比 0.542
MID_BOX = (129, 653, 433, 678)      # 宽占比 0.528
TAIL_BOX = (128, 688, 393, 712)     # 宽占比 0.460, 左右留白 128/183 → 判「收」
HEAD_TXT = "上次那份接口文档里，字段说"
MID_TXT = "明是不是漏了几个？...麻烦你"
TAIL_TXT = "抽空核对一下原始设计稿"

# 阈值边界宽度: int(576*0.45)=259 → 259/576=0.4497 <阈值; 260/576=0.4514 >=阈值
W_BELOW = int(ROI_W * _SYS_MAX_WIDTH_RATIO)
W_ABOVE = W_BELOW + 1


def check(name, cond, detail=""):
    global _ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        _ok = False


def centered_box(width: int, y: int = 500, h: int = 28):
    """构造一个水平居中的 bbox(给定像素宽度)。"""
    x1 = (ROI_W - width) // 2
    return (x1, y, x1 + width, y + h)


def tracked(text, bbox, ts=1.0, fid=10):
    return TrackedLine(line_id=f"L{fid}", text=text, bbox=bbox, first_bbox=bbox,
                       confidence=0.99, first_seen_frame=fid, last_seen_frame=fid,
                       first_timestamp=ts, last_timestamp=ts)


def ratio(bbox):
    return (bbox[2] - bbox[0]) / float(ROI_W)


def main():
    print("=== 1. 前置: 三折行的几何判据确实全部指向「居中系统提示」 ===")
    for name, txt, box in (("头行", HEAD_TXT, HEAD_BOX),
                           ("中行", MID_TXT, MID_BOX)):
        side = classify_side(box, ROI_W, CM, True)
        check(f"{name} classify_side 判不出方向(左白{box[0]}/右白{ROI_W - box[2]})",
              side == "", f"got {side!r}")
        check(f"{name} 短文本捷径命中(len={len(txt)} <= 20) 且不匹配 SYSTEM_PROMPT_RE",
              len(txt) <= 20 and not SYSTEM_PROMPT_RE.search(txt))
        check(f"{name} 宽占比 {ratio(box):.3f} >= 守卫阈值 {_SYS_MAX_WIDTH_RATIO}",
              ratio(box) >= _SYS_MAX_WIDTH_RATIO)
    check("尾行有明确方向「收」(故本就幸存)",
          classify_side(TAIL_BOX, ROI_W, CM, True) == "收")

    print("\n=== 2. 守卫生效: 宽折行正文不得被删 ===")
    for name, txt, box in (("头行", HEAD_TXT, HEAD_BOX),
                           ("中行", MID_TXT, MID_BOX),
                           ("尾行", TAIL_TXT, TAIL_BOX)):
        got = is_centered_system_prompt(txt, box, ROI_W, CM, True)
        check(f"{name} is_centered_system_prompt == False", not got,
              f"{txt!r} box={box} got={got}")

    print("\n=== 3. 真系统提示仍被删(守卫不得放过) ===")
    cases = [
        ("窄居中日期条", "星期一", centered_box(72)),
        ("窄居中新消息提示", "以下为新消息", centered_box(144)),
        ("撤回提示", "对方撤回了一条消息", centered_box(216)),
        ("长且宽+命中RE(不受行宽守卫影响)", "你已添加对方为好友，现在可以开始聊天",
         centered_box(425)),
        ("长且宽+新增RE(通过了…验证)", "对方通过了你的朋友验证请求", centered_box(312)),
        ("正在输入", "对方正在输入…", centered_box(150)),
    ]
    for name, txt, box in cases:
        got = is_centered_system_prompt(txt, box, ROI_W, CM, True)
        check(f"{name}: {txt!r} 宽占比={ratio(box):.3f} → {got}", got is True)

    print("\n=== 4. 其他既有行为不回归 ===")
    check("HH:MM 时间条不滤(由 STAMP_RE 另行处理)",
          not is_centered_system_prompt("20:15", centered_box(70), ROI_W, CM, True))
    check("左气泡正常消息不滤",
          not is_centered_system_prompt("在吗", (30, 500, 100, 528), ROI_W, CM, True))
    check("右气泡正常消息不滤",
          not is_centered_system_prompt("在的", (470, 500, 546, 528), ROI_W, CM, True))
    check("chat_enabled=False 时一律不滤",
          not is_centered_system_prompt("星期一", centered_box(72), ROI_W, CM, False))
    check("宽居中+长(>20字)+无系统措辞 → 不滤(与守卫前一致)",
          not is_centered_system_prompt(
              "这是一段很长很长的居中文字但是没有任何系统提示的措辞在里面",
              centered_box(500), ROI_W, CM, True))
    check(f"阈值上侧: 宽{W_ABOVE}px(占比{ratio(centered_box(W_ABOVE)):.4f}) → 不滤",
          not is_centered_system_prompt("边界宽度的居中文本行",
                                        centered_box(W_ABOVE), ROI_W, CM, True))
    check(f"阈值下侧: 宽{W_BELOW}px(占比{ratio(centered_box(W_BELOW)):.4f}) → 仍滤",
          is_centered_system_prompt("星期一", centered_box(W_BELOW), ROI_W, CM, True))
    check("bbox 缺失时不滤(保守)",
          not is_centered_system_prompt("星期一", None, ROI_W, CM, True))

    print("\n=== 5. drop_system_prompts 端到端: GT#12 三折行全部保留 ===")
    lines = [tracked(HEAD_TXT, HEAD_BOX, 29.467, 163),
             tracked(MID_TXT, MID_BOX, 29.967, 178),
             tracked(TAIL_TXT, TAIL_BOX, 29.967, 178),
             tracked("星期一", centered_box(72, 300), 5.0, 30)]
    kept, dropped = drop_system_prompts(lines, ROI_W, CM, True)
    check(f"剔除 1 条(仅「星期一」), 保留 {len(kept)} 条",
          dropped == 1 and len(kept) == 3,
          f"dropped={dropped} kept={[l.text for l in kept]}")
    check("三折行文本按序全在保留集中",
          [l.text for l in kept] == [HEAD_TXT, MID_TXT, TAIL_TXT],
          str([l.text for l in kept]))

    print("\n" + ("ALL PASS" if _ok else "SOME FAILED"))
    return 0 if _ok else 1


if __name__ == "__main__":
    sys.exit(main())
