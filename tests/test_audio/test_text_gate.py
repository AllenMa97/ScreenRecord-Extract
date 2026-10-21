# -*- coding: utf-8 -*-
"""单元测试: 正文质量门(候选框内含正文 → 判为文本气泡) + 尺寸门。

Unit test for the text-content quality gate: a candidate bubble that CONTAINS a
recognized (non-duration) text line is a text bubble, not a voice bubble.
末尾另测 `drop_narrow_bubbles` 的尺寸门(贴纸过窄 / UI 残片过矮)。
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.audio.data_types import VoiceEvent
from src.audio.ocr_voice_scanner import OcrVoiceScanner
from src.audio.voice_bubble_scanner import (drop_narrow_bubbles,
                                            drop_text_bubbles)

_ok = True

ROI_OFFSET = (0, 60)   # ROI 局部 -> 原始帧: y += 60
FRAME_W = 432


def check(name, cond, detail=""):
    global _ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        _ok = False


def _write_cache(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "ocr.jsonl")
        # bbox 为 **ROI 局部坐标**(与生产契约一致); 中线 216 -> 右半判 '发'
        _write_cache(cache, [
            {"fid": 3, "ts": 5.0, "lines": [
                {"text": "哪三个", "bbox": [300, 100, 380, 130], "confidence": 0.95},
                {"text": "收到", "bbox": [40, 100, 120, 130], "confidence": 0.95},
                # 时长标签: 长在语音气泡里, 不是正文
                {"text": '6"', "bbox": [360, 240, 380, 260], "confidence": 0.95},
                # 前导省略号的时长标签: 气泡被屏幕边缘裁切时 OCR 只吃到半截
                # (实测 自行模拟对话记录4.mp4 的 `…4"`), 必须同样判为时长标签
                {"text": '…4"', "bbox": [360, 400, 380, 420], "confidence": 0.95},
                # 低置信应被过滤
                {"text": "噪声", "bbox": [300, 300, 380, 330], "confidence": 0.2},
            ]},
        ])

        sc = OcrVoiceScanner(frame_width=FRAME_W)
        texts = sc.scan_text_hits(cache, roi_offset=ROI_OFFSET)
        durs = sc.scan_duration_hits(cache, roi_offset=ROI_OFFSET)
        check("scan_text_hits 排除时长标签与低置信", len(texts) == 2,
              f"got {[t.text for t in texts]}")
        check("scan_duration_hits 命中 2 条时长(含前导省略号)", len(durs) == 2,
              f"got {[d.text for d in durs]}")
        check("box 已换算到原始帧坐标(y+60)",
              all(abs(t.box[1] - 160.0) < 1e-6 for t in texts),
              str([t.box for t in texts]))
        check("侧别按 ROI 中线判定", {t.side for t in texts} == {"发", "收"},
              str([(t.text, t.side) for t in texts]))

        # 框住 "哪三个"(帧坐标 [300,160,80,30]) -> 应被剔除
        text_bubble = VoiceEvent(timestamp=5.0, end_time=5.4, side="发",
                                 box=(298, 158, 84, 34))
        # 同侧、同时间, 但框内无正文 -> 语音气泡, 必须保留
        voice_bubble = VoiceEvent(timestamp=5.0, end_time=5.4, side="发",
                                  box=(300, 460, 130, 44))
        # 跨列: 左侧正文(收)不能剔除右侧(发)气泡
        cross_side = VoiceEvent(timestamp=5.0, end_time=5.4, side="发",
                                box=(40, 158, 84, 34))
        # 内置时长标签的语音气泡: 标签不是正文, 必须保留
        dur_bubble = VoiceEvent(timestamp=5.0, end_time=5.4, side="发",
                                box=(356, 296, 30, 28))
        # 时间窗外的气泡不受影响
        far_away = VoiceEvent(timestamp=30.0, end_time=30.4, side="发",
                              box=(298, 158, 84, 34))

        kept = drop_text_bubbles([text_bubble, voice_bubble, cross_side,
                                  dur_bubble, far_away], texts)
        ids = [id(e) for e in kept]
        check("框内含同侧正文 -> 剔除", id(text_bubble) not in ids)
        check("无正文的语音气泡 -> 保留", id(voice_bubble) in ids)
        check("跨列(异侧)正文不误杀", id(cross_side) in ids)
        check("内置时长标签不误杀", id(dur_bubble) in ids)
        check("生命周期窗外不误杀", id(far_away) in ids)

        check("无 text_hits 时原样返回",
              len(drop_text_bubbles([text_bubble, voice_bubble], [])) == 2)

        # 跨帧滚动: 正文行在 t=5.0 的屏幕 y=160, 气泡的静止观测在 t=5.4, 期间
        # 屏幕上滚 56px, 气泡屏幕 y=104。屏幕坐标下二者"不重叠", 但文档坐标一致
        # (160 == 104 + 56), 必须仍判为文本气泡。
        cache2 = os.path.join(tmp, "ocr2.jsonl")
        _write_cache(cache2, [
            {"fid": 3, "ts": 5.0, "lines": [
                {"text": "订单SN0010已确认收款",
                 "bbox": [138, 100, 336, 122], "confidence": 0.95},
            ]},
        ])
        texts2 = sc.scan_text_hits(cache2, roi_offset=ROI_OFFSET)
        curve = [(5.0, 0.0), (5.4, 56.0)]
        scrolled_text = VoiceEvent(timestamp=5.0, end_time=5.4, side="发",
                                   box=(130, 104, 219, 50),
                                   stable_timestamp=5.4,
                                   stable_box=(130, 104, 219, 50))
        scrolled_voice = VoiceEvent(timestamp=5.0, end_time=5.4, side="发",
                                    box=(130, 250, 140, 50),
                                    stable_timestamp=5.4,
                                    stable_box=(130, 250, 140, 50))
        kept_screen = drop_text_bubbles([scrolled_text, scrolled_voice], texts2)
        check("无滚动曲线时跨帧正文判不出(基线)",
              id(scrolled_text) in [id(e) for e in kept_screen])
        kept_doc = drop_text_bubbles([scrolled_text, scrolled_voice], texts2,
                                     scroll_curve=curve)
        ids_doc = [id(e) for e in kept_doc]
        check("有滚动曲线时按文档坐标剔除跨帧正文气泡",
              id(scrolled_text) not in ids_doc)
        check("滚动后无正文的气泡仍保留", id(scrolled_voice) in ids_doc)

        # ROI 原点在预热结束后挪动(实测 (0,60) → (0,302)), 同一个缓存文件里混着
        # 两套局部坐标系。偏移必须**按帧取自记录里的 roi 字段**, 而不是统一用调用方
        # 给的终版偏移 —— 否则预热期帧整体平移 242px, 正文质量门一个都剔不掉。
        cache3 = os.path.join(tmp, "ocr3.jsonl")
        _write_cache(cache3, [
            {"fid": 1, "ts": 1.0, "roi": [0, 60], "lines": [
                {"text": "收到，你说", "bbox": [323, 751, 449, 783],
                 "confidence": 0.95},
            ]},
            {"fid": 9, "ts": 9.0, "roi": [0, 302], "lines": [
                {"text": "收到，你说", "bbox": [323, 751, 449, 783],
                 "confidence": 0.95},
            ]},
            # 旧缓存无 roi 字段: 回退到调用方给的偏移
            {"fid": 10, "ts": 10.0, "lines": [
                {"text": "收到，你说", "bbox": [323, 751, 449, 783],
                 "confidence": 0.95},
            ]},
        ])
        texts3 = sc.scan_text_hits(cache3, roi_offset=(0, 302))
        ys = {round(h.timestamp, 1): h.box[1] for h in texts3}
        check("按帧 roi 换算: 预热期帧用 y+60", ys.get(1.0) == 811.0, str(ys))
        check("按帧 roi 换算: 稳态期帧用 y+302", ys.get(9.0) == 1053.0, str(ys))
        check("无 roi 字段的旧缓存回退到调用方偏移", ys.get(10.0) == 1053.0,
              str(ys))
        # 预热期那只文本气泡(帧坐标 y=795..860)必须被剔除
        warmup_text_bubble = VoiceEvent(
            timestamp=1.0, end_time=1.4, side="发", box=(310, 795, 153, 65),
            stable_timestamp=1.0, stable_box=(310, 795, 153, 65))
        check("预热期文本气泡按各自 roi 换算后被剔除",
              not drop_text_bubbles([warmup_text_bubble], texts3))

    # ---- 尺寸门: 宽**或高**显著小于同批中位数的候选 ----
    # 实测 `自行模拟对话记录4.mp4`: 真语音气泡 h=63~68 且宽度齐平(w=184~185),
    # 混进来的 UI 装饰残片 h=20 而宽度与真气泡一致 —— 只看宽度抓不到, 而它在文档
    # 坐标排序里插到中段, 会吃掉一个配对槽位使其后所有气泡方向整体错一位(4/6)。
    def _ev(h, w=184, side="收", x=115, y=100):
        return VoiceEvent(timestamp=0.0, side=side, box=(x, y, w, h))

    v4 = [_ev(63), _ev(65), _ev(63), _ev(40), _ev(20),
          _ev(68, side="发", x=277), _ev(63)]
    got = [e.box[3] for e in drop_narrow_bubbles(v4)]
    check("尺寸门剔除矮残片(h=20), 保留 h=40 与真气泡",
          got == [63, 65, 63, 40, 68, 63], str(got))
    # 窄贴纸仍按宽度剔除(高度正常)
    check("尺寸门仍按宽度剔除贴纸",
          len(drop_narrow_bubbles([_ev(63) for _ in range(5)]
                                  + [_ev(63, w=70)])) == 5)
    # 无方向的候选不参与统计也不被剔除(与 `_ordered_bubbles` 的 eligible 同口径)
    check("无方向候选不被尺寸门改动",
          len(drop_narrow_bubbles([_ev(47) for _ in range(5)]
                                  + [_ev(26, w=258, side="", x=90)])) == 6)
    # 样本不足时不动(小批次中位数不稳)
    check("样本 <5 时尺寸门不生效", len(drop_narrow_bubbles([_ev(63), _ev(20)])) == 2)

    print("\nALL PASS" if _ok else "\nSOME FAILED")
    sys.exit(0 if _ok else 1)


if __name__ == "__main__":
    main()
