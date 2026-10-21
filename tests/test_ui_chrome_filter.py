# -*- coding: utf-8 -*-
"""单元测试: App界面装饰行过滤(CV卡片区域检测 drop_ui_chrome)。

Unit test for the CV panel detector that drops fixed app furniture (anti-fraud
banner, promo/profile cards, input-box hint) which OCR reads like chat text.

合成帧按 `自行模拟对话记录4.mp4` / `3.mp4` 的实测画面结构搭:
  * 三段式深色照片墙纸 —— 段间亮度台阶 43/36, 都小于 _PANEL_EDGE_MIN(60)。真消息
    行会因这种局部渐变凑出一个"假面板"(横向平坦占比甚至到 1.0);
  * 整宽白卡片 / 顶部竖向渐变横幅 / 底部"灰带+白框"叠层输入条 —— 三种真装饰块,
    分别钉住「同色扩展」「相邻行平滑扩展」「外探窗口要够宽」三条实现细节;
  * 宽白气泡(占 ROI 宽 0.61)与"上边界紧贴卡片的墙纸高面板" —— 两类**实测假阳性**,
    台阶/平坦度全部达标, 分别只被 _PANEL_MIN_FLAT 与 _PANEL_MAX_H 拦住。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                               # noqa: E402

from src.bubble_merger import (_PANEL_EDGE_MIN, _PANEL_EDGE_PROBE,  # noqa: E402
                               _PANEL_MAX_H, _PANEL_MIN_FLAT,
                               _panel_probe, _ui_chrome_on_frame,
                               drop_ui_chrome, is_ui_chrome_line)
from src.data_types import TextLine, TrackedLine                # noqa: E402

_ok = True

W, H = 576, 1100            # video4 实测 ROI 尺寸

# 三段式照片墙纸: 段间台阶 43 / 36, 均 < _PANEL_EDGE_MIN(60)
COL_A, COL_B, COL_C = (83, 93, 115), (126, 135, 152), (90, 100, 120)
SPLIT_1, SPLIT_2 = 400, 800

# 装饰块几何(取自 video4 实测)
CARD_Y1, CARD_Y2 = 304, 424          # 整宽白卡片(对方资料卡)
CARD_BOX = (86, 325, 181, 349)       # 卡内文字「年龄：26岁」
BANNER_H = 129                       # 顶部渐变横幅 y0..128
BANNER_BOX1 = (17, 76, 562, 95)      # 横幅首行(密集长行, 易污染列中位)
BANNER_BOX2 = (17, 96, 166, 116)     # 横幅次行
GRAY_Y1, FIELD_Y1 = 1038, 1050       # 输入条: 灰带 1038..1049 + 白框 1050..底
INPUT_BOX = (71, 1064, 428, 1091)    # 输入框引导「真诚的回复…」
BUBBLE_BOX = (132, 534, 188, 567)    # 左白气泡内文字「在吗」
PLAIN_BOX = (52, 619, 504, 639)      # 直接画在墙纸上的公告行

# 两类实测假阳性(video3/video4 标定): 白气泡与墙纸假面板都能骗过"边界台阶"
WIDE_Y1, WIDE_Y2 = 520, 581          # 宽发送气泡: 占 ROI 宽 0.61
WIDE_X1, WIDE_X2 = 114, 464
WIDE_BOX = (140, 540, 430, 560)      # 气泡内文字(行带占面板高 1/3, 不污染列中位)
TALL_CARD_Y1 = 300                   # 紧贴墙纸 B 段上沿的白卡片 → 假面板量到台阶


def check(name, cond, detail=""):
    global _ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        _ok = False


def wallpaper():
    """三段式深色照片墙纸。"""
    f = np.zeros((H, W, 3), np.uint8)
    f[0:SPLIT_1] = COL_A
    f[SPLIT_1:SPLIT_2] = COL_B
    f[SPLIT_2:H] = COL_C
    return f


def glyph(frame, box, color=(200, 205, 220), step=10, width=4):
    """在 box 内画竖条状"字形"(占空比 width/step ≈ 40%, 接近真实中文行)。"""
    x1, y1, x2, y2 = box
    for x in range(x1, x2, step):
        frame[y1:y2, x:min(x + width, x2)] = color


def scene_card():
    f = wallpaper()
    f[CARD_Y1:CARD_Y2] = 255
    glyph(f, CARD_BOX)
    return f


def scene_banner():
    """顶部竖向渐变横幅: 钉住「相邻行平滑扩展」(与单一参考色比只能框出 27 行)。"""
    f = wallpaper()
    top = np.array([25, 30, 50], float)
    bot = np.array([10, 14, 28], float)
    ramp = np.linspace(0.0, 1.0, BANNER_H)[:, None]
    f[0:BANNER_H] = (top + (bot - top) * ramp).astype(np.uint8)[:, None, :]
    glyph(f, BANNER_BOX1)
    glyph(f, BANNER_BOX2)
    return f


def scene_input():
    """底部叠层输入条: 钉住「外探窗口要够宽」(只探 12 行整段落在灰带里)。"""
    f = wallpaper()
    f[1036:GRAY_Y1] = (157, 148, 149)
    f[GRAY_Y1:FIELD_Y1] = (226, 232, 246)
    f[FIELD_Y1:H] = 255
    glyph(f, (71, 1070, 428, 1086), color=(150, 150, 150))
    return f


def scene_bubble():
    """左白气泡(宽 81px = 0.14 ROI)贴在墙纸 B 段上 —— 真消息。"""
    f = wallpaper()
    f[520:581, 120:201] = 255
    glyph(f, BUBBLE_BOX)
    return f


def scene_plain():
    """直接画在墙纸上的公告行(无卡片背景) —— 本检测器的已知盲区, 必须不滤。"""
    f = wallpaper()
    glyph(f, PLAIN_BOX, color=(240, 240, 240))
    return f


def scene_wide_bubble():
    """假阳性类A: 宽白色发送气泡(占 ROI 宽 0.61)。

    面板正好框住气泡本身, 于是台阶(白 vs 墙纸 129)、矩形内平坦度(0.61)全部达标,
    只有"同色列占比 = 气泡宽占比 0.61 < _PANEL_MIN_FLAT"能拦住它。
    """
    f = wallpaper()
    f[WIDE_Y1:WIDE_Y2, WIDE_X1:WIDE_X2] = 255
    glyph(f, WIDE_BOX, color=(40, 40, 40))
    return f


def scene_tall():
    """假阳性类B: 墙纸假面板的上边界正好紧贴一张白卡片。

    外探窗口落到卡片上, 台阶量到的是"墙纸 vs 卡片"(129 >= _PANEL_EDGE_MIN),
    主判据失效; 只有"面板高 400px > _PANEL_MAX_H × ROI"能拦住它。
    """
    f = wallpaper()
    f[TALL_CARD_Y1:SPLIT_1] = 255
    glyph(f, PLAIN_BOX)
    return f


FRAMES = {
    1: scene_card(), 2: scene_card(),
    3: scene_bubble(),
    4: scene_banner(), 5: scene_banner(),
    6: scene_input(), 7: scene_input(),
    8: scene_plain(),
    9: scene_wide_bubble(),
    10: scene_tall(),
}


def provider(fid):
    return FRAMES.get(fid)


def tracked(text, fid, bbox, fid2=None, bbox2=None):
    """构造跟踪行; 给 fid2 则末帧观测落在 (fid2, bbox2)。"""
    f2 = fid if fid2 is None else fid2
    b2 = bbox if bbox2 is None else bbox2
    obs = [TextLine(text=text, bbox=bbox, frame_id=fid, timestamp=1.0)]
    if (f2, b2) != (fid, bbox):
        obs.append(TextLine(text=text, bbox=b2, frame_id=f2, timestamp=2.0))
    return TrackedLine(line_id=f"L{fid}", text=text, bbox=b2, first_bbox=bbox,
                       confidence=0.99, first_seen_frame=fid, last_seen_frame=f2,
                       first_timestamp=1.0, last_timestamp=2.0, observations=obs)


def main():
    print("=== 0. 合成墙纸的段间台阶确实小于主判据阈值(假面板前提) ===")
    step1 = max(abs(a - b) for a, b in zip(COL_A, COL_B))
    step2 = max(abs(a - b) for a, b in zip(COL_B, COL_C))
    check(f"A→B 台阶 {step1} < _PANEL_EDGE_MIN({_PANEL_EDGE_MIN})",
          step1 < _PANEL_EDGE_MIN)
    check(f"B→C 台阶 {step2} < _PANEL_EDGE_MIN({_PANEL_EDGE_MIN})",
          step2 < _PANEL_EDGE_MIN)

    print("\n=== 1. 真消息: 假面板横向很平坦, 靠「高度上限 + 边界台阶」双否决 ===")
    p = _panel_probe(FRAMES[3], BUBBLE_BOX)
    check("假面板 = 墙纸 B 段 [400,799]",
          (p["pt"], p["pb"]) == (SPLIT_1, SPLIT_2 - 1),
          f"got [{p['pt']},{p['pb']}]")
    check(f"横向平坦占比 {p['frac']:.2f} >= {_PANEL_MIN_FLAT}(气泡行也过得了这条)",
          p["frac"] >= _PANEL_MIN_FLAT)
    check(f"面板高 {p['pb'] - p['pt'] + 1}px > {_PANEL_MAX_H} × ROI = "
          f"{int(_PANEL_MAX_H * H)}px → 判据 1 否决",
          (p["pb"] - p["pt"] + 1) > _PANEL_MAX_H * H)
    check(f"边界台阶 {p['edge']:.0f} < {_PANEL_EDGE_MIN} → 判据 3 也否决",
          p["edge"] < _PANEL_EDGE_MIN)
    check("「在吗」is_ui_chrome_line == False",
          not is_ui_chrome_line(tracked("在吗", 3, BUBBLE_BOX), provider))

    print("\n=== 2. 整宽白卡片 → 滤 ===")
    p = _panel_probe(FRAMES[1], CARD_BOX)
    check(f"面板 = 卡片 [{CARD_Y1},{CARD_Y2 - 1}]",
          (p["pt"], p["pb"]) == (CARD_Y1, CARD_Y2 - 1),
          f"got [{p['pt']},{p['pb']}]")
    check(f"台阶 {p['edge']:.0f} >= {_PANEL_EDGE_MIN}", p["edge"] >= _PANEL_EDGE_MIN)
    check("「年龄：26岁」is_ui_chrome_line == True",
          is_ui_chrome_line(tracked("年龄：26岁", 1, CARD_BOX, 2, CARD_BOX),
                            provider))

    print("\n=== 3. 顶部竖向渐变横幅 → 两行都滤(钉住平滑扩展) ===")
    f4 = FRAMES[4]
    for name, box in (("首行(密集长行)", BANNER_BOX1), ("次行", BANNER_BOX2)):
        p = _panel_probe(f4, box)
        check(f"{name} 面板扩到整条横幅 [0,{BANNER_H - 1}]",
              (p["pt"], p["pb"]) == (0, BANNER_H - 1),
              f"got [{p['pt']},{p['pb']}]")
        check(f"{name} frac={p['frac']:.2f} edge={p['edge']:.0f} "
              f"ratio={p['ratio']:.2f} 全部达标",
              p["frac"] >= _PANEL_MIN_FLAT and p["edge"] >= _PANEL_EDGE_MIN
              and p["ratio"] >= 0.50)
        check(f"{name} is_ui_chrome_line == True",
              is_ui_chrome_line(tracked(name, 4, box, 5, box), provider))

    print("\n=== 4. 底部叠层输入条 → 滤(钉住外探窗口宽度) ===")
    p = _panel_probe(FRAMES[6], INPUT_BOX)
    check(f"面板 = 白色输入框 [{FIELD_Y1},{H - 1}](灰带未被并入)",
          (p["pt"], p["pb"]) == (FIELD_Y1, H - 1), f"got [{p['pt']},{p['pb']}]")
    check(f"外探 {_PANEL_EDGE_PROBE} 行够到灰带上方的深色聊天区: "
          f"edge={p['edge']:.0f} >= {_PANEL_EDGE_MIN}",
          p["edge"] >= _PANEL_EDGE_MIN)
    check("「真诚的回复…」is_ui_chrome_line == True",
          is_ui_chrome_line(tracked("真诚的回复，更能感受到你的真心~",
                                    6, INPUT_BOX, 7, INPUT_BOX), provider))

    print("\n=== 5. 假阳性类A: 宽白色发送气泡 → 只有「同色列占比」拦得住 ===")
    p = _panel_probe(FRAMES[9], WIDE_BOX)
    check(f"面板 = 气泡本身 [{WIDE_Y1},{WIDE_Y2 - 1}]",
          (p["pt"], p["pb"]) == (WIDE_Y1, WIDE_Y2 - 1),
          f"got [{p['pt']},{p['pb']}]")
    check(f"面板高 {p['pb'] - p['pt'] + 1}px <= {_PANEL_MAX_H} × ROI(判据 1 放行)",
          (p["pb"] - p["pt"] + 1) <= _PANEL_MAX_H * H)
    check(f"台阶 {p['edge']:.0f} >= {_PANEL_EDGE_MIN}(判据 3 放行: 白气泡对白卡片)",
          p["edge"] >= _PANEL_EDGE_MIN)
    check(f"矩形内平坦度 {p['ratio']:.2f} >= 0.50(判据 4 放行)",
          p["ratio"] >= 0.50)
    check(f"同色列占比 {p['frac']:.2f} < {_PANEL_MIN_FLAT} → 判据 2 否决"
          f"(= 气泡宽占比 {(WIDE_X2 - WIDE_X1) / W:.2f})",
          p["frac"] < _PANEL_MIN_FLAT)
    check("「上次那份接口文档里，字段说」is_ui_chrome_line == False",
          not is_ui_chrome_line(tracked("上次那份接口文档里，字段说", 9, WIDE_BOX),
                                provider))

    print("\n=== 6. 假阳性类B: 紧贴卡片的墙纸高面板 → 只有「高度上限」拦得住 ===")
    p = _panel_probe(FRAMES[10], PLAIN_BOX)
    check("假面板 = 墙纸 B 段 [400,799](卡片未被并入)",
          (p["pt"], p["pb"]) == (SPLIT_1, SPLIT_2 - 1),
          f"got [{p['pt']},{p['pb']}]")
    check(f"同色列占比 {p['frac']:.2f} >= {_PANEL_MIN_FLAT}(判据 2 放行)",
          p["frac"] >= _PANEL_MIN_FLAT)
    check(f"台阶 {p['edge']:.0f} >= {_PANEL_EDGE_MIN}(判据 3 放行: 外探窗口落在"
          f"紧贴的卡片上)", p["edge"] >= _PANEL_EDGE_MIN)
    check(f"面板高 {p['pb'] - p['pt'] + 1}px > {_PANEL_MAX_H} × ROI = "
          f"{int(_PANEL_MAX_H * H)}px → 判据 1 否决",
          (p["pb"] - p["pt"] + 1) > _PANEL_MAX_H * H)
    check("「收到」is_ui_chrome_line == False",
          not is_ui_chrome_line(tracked("收到", 10, PLAIN_BOX), provider))
    check(f"真装饰块都在上限内(卡片 {CARD_Y2 - CARD_Y1}px / 横幅 {BANNER_H}px / "
          f"输入条 {H - FIELD_Y1}px <= {int(_PANEL_MAX_H * H)}px)",
          max(CARD_Y2 - CARD_Y1, BANNER_H, H - FIELD_Y1) <= _PANEL_MAX_H * H)

    print("\n=== 7. 已知盲区: 直接画在墙纸上的公告行不滤(保守取向) ===")
    check("「亲密度达到100℃…」is_ui_chrome_line == False",
          not is_ui_chrome_line(tracked("亲密度达到100℃，才可以加外部联系方式哦。",
                                        8, PLAIN_BOX), provider))

    print("\n=== 8. 全票一致: 任一候选帧不在面板上即保留 ===")
    check("首帧在气泡(非面板)、末帧滚进卡片 → False",
          not is_ui_chrome_line(
              tracked("收到，你说", 3, BUBBLE_BOX, 1, CARD_BOX), provider))
    check("首帧在卡片、末帧滚出到墙纸 → False(多数票会误删)",
          not is_ui_chrome_line(
              tracked("订单SN0014已确认收款", 1, CARD_BOX, 3, BUBBLE_BOX),
              provider))
    check("只有一个观测帧且判面板 → True(无从复判, 采信该帧)",
          is_ui_chrome_line(tracked("领取", 1, CARD_BOX), provider))

    print("\n=== 9. 豁免与退化输入 ===")
    check("HH:MM 时间条即使落在卡片上也不滤",
          not is_ui_chrome_line(tracked("17:59", 1, CARD_BOX), provider))
    check("空文本不滤", not is_ui_chrome_line(tracked("  ", 1, CARD_BOX), provider))
    check("frame_provider=None 不滤",
          not is_ui_chrome_line(tracked("年龄：26岁", 1, CARD_BOX), None))
    check("chat_enabled=False 不滤",
          not is_ui_chrome_line(tracked("年龄：26岁", 1, CARD_BOX), provider,
                                chat_enabled=False))
    ln = tracked("年龄：26岁", 1, CARD_BOX)
    ln.first_bbox, ln.bbox = None, None
    check("bbox 缺失不滤", not is_ui_chrome_line(ln, provider))
    check("帧太小 _panel_probe 返回 None",
          _panel_probe(np.zeros((8, 8, 3), np.uint8), (1, 1, 5, 5)) is None)
    check("bbox 整体落在帧外(坐标系不匹配) _panel_probe 返回 None",
          _panel_probe(FRAMES[1], (10, H + 20, 200, H + 50)) is None)
    check("单帧判定: 帧号不存在时返回 False",
          not _ui_chrome_on_frame(999, CARD_BOX, provider))

    print("\n=== 10. drop_ui_chrome 端到端: 只删装饰行, 顺序不变 ===")
    lines = [tracked("一切以投资、理财、炒币…均为诈骗，请立即举报", 4, BANNER_BOX1,
                     5, BANNER_BOX1),
             tracked("年龄：26岁", 1, CARD_BOX, 2, CARD_BOX),
             tracked("在吗", 3, BUBBLE_BOX),
             tracked("真诚的回复，更能感受到你的真心~", 6, INPUT_BOX, 7, INPUT_BOX),
             tracked("17:59", 1, CARD_BOX),
             tracked("上次那份接口文档里，字段说", 9, WIDE_BOX),
             tracked("收到", 10, PLAIN_BOX),
             tracked("收到，你说", 3, BUBBLE_BOX)]
    kept, dropped = drop_ui_chrome(lines, provider)
    check(f"剔除 3 条装饰行, got={dropped}", len(dropped) == 3, str(dropped))
    check("保留集顺序 = 真消息(含两类假阳性场景) + 时间条",
          [l.text for l in kept] == ["在吗", "17:59", "上次那份接口文档里，字段说",
                                     "收到", "收到，你说"],
          str([l.text for l in kept]))
    check("frame_provider=None 时原样返回",
          drop_ui_chrome(lines, None)[1] == [])
    check("chat_enabled=False 时原样返回",
          drop_ui_chrome(lines, provider, chat_enabled=False)[1] == [])
    check("空列表不报错", drop_ui_chrome([], provider) == ([], []))

    print("\n" + ("ALL PASS" if _ok else "SOME FAILED"))
    return 0 if _ok else 1


if __name__ == "__main__":
    sys.exit(main())
