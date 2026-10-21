"""
语音气泡检测器测试 (voice bubble detector tests).

用 numpy 合成聊天帧(宽~400), 分别验证:
  (a) 绿色语音气泡+波形(右侧, 我方) 会被检出且 voice_score>=0.6, side=='发';
  (b) 白色语音气泡+波形(左侧, 对方) 会被检出且 voice_score>=0.6, side=='收';
  (c) 绿色文本气泡(右侧) 要么不返回要么 voice_score<0.5;
  (d) 白色文本气泡(左侧) 要么不返回要么 voice_score<0.5;
  (e) 纯背景 -> 返回 [];
  (f) 整幅宽/半屏背景块(伪检) -> 不得返回超过 ROI 宽度上限(MAX_BUBBLE_W_RATIO)的框;
  (g) 上缘被 ROI 裁切的语音气泡(滚动中常见) -> 仍应被检出(side=='收')。

assert 风格; 直接 `python tests/test_audio/test_voice_bubble_detector.py` 运行, 不依赖 pytest。
所有合成帧会用 render_detections 保存标注图到临时目录并打印路径(仅供人工确认, 不算失败)。

运行 / Run:
  python tests/test_audio/test_voice_bubble_detector.py
运行结束打印 "ALL PASS"。
"""

import os
import sys
import tempfile

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.visual.voice_bubble_detector import (
    MAX_BUBBLE_W_RATIO,
    VoiceBubble,
    detect_voice_bubbles,
    render_detections,
)

W, H = 400, 240          # 合成帧尺寸 ~400 宽
BG = (212, 212, 212)     # 中性略深灰背景, 确保白色气泡可被区分

CHAT_GREEN = (50, 175, 70)   # BGR 绿色(我方气泡)
WAVE_DARK = (40, 70, 45)     # 波形竖条颜色(墨色, 明显暗于气泡底)
TEXT_DARK = (35, 35, 38)     # 文字颜色(墨色)

rng = np.random.default_rng(20260831)  # 固定种子 -> 确定性


def _filled_rounded_rect(surface: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                         color, radius: int = 14):
    """在 surface 上绘制实心圆角矩形(用掩码近似)。返回上色后的副本。"""
    hh, ww = surface.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww]
    cx0, cy0, cx1, cy1 = x0 + radius, y0 + radius, x1 - radius, y1 - radius
    inside = True
    # 中段矩形
    mid = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    corner_ok = np.zeros_like(mid, bool)
    # 四个圆角
    corner_ok |= ((yy >= cy0) & (yy <= cy1) & (xx >= x0) & (xx <= x1))
    corner_ok |= ((yy >= y0) & (yy <= y1) & (xx >= cx0) & (xx <= cx1))
    # 四个角圆弧
    r2 = radius * radius
    for cx, cy in ((cx0, cy0), (cx1, cy0), (cx0, cy1), (cx1, cy1)):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        corner_ok |= (d2 <= r2) & (xx >= cx0 - radius) & (xx <= cx1 + radius) \
                     & (yy >= cy0 - radius) & (yy <= cy1 + radius)
    mask = mid & corner_ok
    surface[mask] = color
    return surface


def _draw_waveform(surface, bx: int, by: int, bw: int, bh: int, color) -> None:
    """在气泡内绘"波形": 两条窄竖条, 宽度 2-3px, 高度多变, 底部对齐, 平滑包络。"""
    padx, pady, bar_gap = 10, 8, 3
    inner_x0, inner_x1 = bx + padx, bx + bw - padx
    base_y = by + bh - pady            # 竖条底部对齐基线
    max_h = int(bh * 0.62)
    x = inner_x0
    # 用正弦包络生成若干竖条(高度先升后降), 模拟真实语音波形
    n_bars = max(4, (inner_x1 - inner_x0) // (bar_gap + 3))
    i = 0
    while x < inner_x1 - 1 and i < n_bars:
        env = np.sin(np.pi * i / max(1, n_bars - 1))  # 0..1 先升后降
        bar_h = int(6 + env * max_h)
        bw_ = 2 + (i % 2)              # 2~3 px 宽
        h0 = max(by + pady, base_y - bar_h)
        surface[h0:base_y, x : x + bw_] = color
        x += bw_ + bar_gap
        i += 1


def _draw_text_block(surface, bx: int, by: int, bw: int, bh: int, color) -> None:
    """在气泡内绘"文字": 用随机小矩形拼出若干行汉字状斑块(无实际文字)。"""
    padx, pady = 8, 8
    cx0, cy0 = bx + padx, by + pady
    cy = cy0 + 12                       # 首行基线
    nx = 0
    while cy < by + bh - pady - 6:
        cx = cx0 + 4
        line_len = 5 + rng.integers(3)
        for _ in range(line_len):
            if cx + 16 > cx0 + bw - 8:
                break
            ch = rng.integers(2, 13)    # 每个"字" 由若干小矩形随机拼成
            for _e in range(ch):
                ww = rng.integers(1, 5)
                hh2 = rng.integers(1, 5)
                ox = rng.integers(0, max(1, 16 - ww))
                oy = rng.integers(-18, -1)   # 相对基线向上
                x = cx + ox
                y = cy + min(oy, 0)
                surface[y : y + hh2, x : x + ww] = color
            cx += 18
            nx += 1
        cy += 22


def make_voice_bubble(side: str) -> np.ndarray:
    """合规语音气泡帧: side='发'->绿/右侧, '收'->白/左侧。"""
    img = np.full((H, W, 3), BG, np.uint8)
    bw, bh = 150, 54
    if side == "发":
        bx = W - bw - 24
        color = CHAT_GREEN
    else:
        bx = 24
        color = (255, 255, 255)
    by = (H - bh) // 2
    _filled_rounded_rect(img, bx, by, bx + bw - 1, by + bh - 1, color)
    _draw_waveform(img, bx, by, bw, bh, WAVE_DARK)
    return img


def make_text_bubble(side: str) -> np.ndarray:
    """文本气泡帧: 绿色/右 或 白色/左, 内含文字。"""
    img = np.full((H, W, 3), BG, np.uint8)
    bw, bh = 150, 58
    if side == "发":
        bx = W - bw - 24
        color = CHAT_GREEN
    else:
        bx = 24
        color = (255, 255, 255)
    by = (H - bh) // 2
    _filled_rounded_rect(img, bx, by, bx + bw - 1, by + bh - 1, color)
    _draw_text_block(img, bx, by, bw, bh, TEXT_DARK)
    return img


def make_wide_background() -> np.ndarray:
    """整幅宽/半屏近白背景块(伪检: 应用内横幅或大片墙纸), 不应被当作语音气泡。"""
    img = np.full((H, W, 3), BG, np.uint8)
    _filled_rounded_rect(img, 0, 0, W - 1, 150, (250, 250, 250), radius=6)
    return img


def make_top_cut_voice() -> np.ndarray:
    """上缘被 ROI 裁切的语音气泡(滚动时常见): 顶部被切掉一截, 仍应被检出。"""
    base = make_voice_bubble("收")
    shift = 100
    cut = np.full((H, W, 3), BG, np.uint8)
    cut[: H - shift] = base[shift:]      # 整幅上移, 气泡顶部被裁于 ROI 上缘
    return cut


def main() -> None:
    frames = {
        "voice_green_right": make_voice_bubble("发"),
        "voice_white_left": make_voice_bubble("收"),
        "text_green_right": make_text_bubble("发"),
        "text_white_left": make_text_bubble("收"),
        "plain_background": np.full((H, W, 3), BG, np.uint8),
        "wide_background": make_wide_background(),
        "top_cut_voice": make_top_cut_voice(),
    }

    tmp = tempfile.mkdtemp(prefix="vbd_")
    voice_seen = {"发": False, "收": False}
    n_pass = 0

    for name, frame in frames.items():
        bubbles = detect_voice_bubbles(frame)

        if name == "plain_background":
            assert bubbles == [], f"纯背景应返回 []，实际 {bubbles}"
            print(f"[{name}] 纯背景 -> []  OK")
            n_pass += 1
            continue

        # --- 语音气泡帧: 必须检出语音气泡, voice_score>=0.6, side 正确 ---
        if name.startswith("voice"):
            expect_side = "发" if name.endswith("right") else "收"
            vbs = [b for b in bubbles if b.voice_score >= VOICE_OK]
            assert vbs, f"{name} 语音气泡未被检出/未达阈值: {bubbles}"
            # 取 voice_score 最大的那个作为目标气泡
            top = max(vbs, key=lambda b: b.voice_score)
            assert top.voice_score >= 0.6, f"{name} voice_score 过低: {top.voice_score}"
            assert top.side == expect_side, f"{name} side 错误: {top.side} (期望 {expect_side})"
            voice_seen[expect_side] = True
            print(
                f"[{name}] 检出 voice_score={top.voice_score:.2f} "
                f"side={top.side} box={top.x},{top.y},{top.w},{top.h} OK"
            )
            n_pass += 1

        # --- 文本气泡帧: 要么不返回, 要么 voice_score<0.5 ---
        if name.startswith("text"):
            texts = [b for b in bubbles]
            for b in texts:
                assert b.voice_score < 0.5, (
                    f"{name} 文本气泡 voice_score 过高: {b.voice_score}"
                )
            print(f"[{name}] 文本气泡 -> {len(texts)} 个候选, 均 <0.5 OK")
            n_pass += 1

        # --- 整幅宽/半屏背景块: 不得返回超过宽度上限的框(旧的整片背景伪检) ---
        if name == "wide_background":
            over = [b for b in bubbles if b.w > MAX_BUBBLE_W_RATIO * W]
            assert not over, f"{name} 整幅宽背景被误检为气泡: {over}"
            print(f"[{name}] 整幅宽背景 -> {len(bubbles)} 个候选, 无超宽框 OK")
            n_pass += 1

        # --- 上缘被裁切的语音气泡: 仍须检出(side 由位置判定为 '收') ---
        if name == "top_cut_voice":
            vbs = [b for b in bubbles if b.voice_score >= VOICE_OK]
            assert vbs, f"{name} 上缘裁切气泡丢失: {bubbles}"
            top = max(vbs, key=lambda b: b.voice_score)
            assert top.side == "收", f"{name} side 错误: {top.side} (期望 收)"
            print(
                f"[{name}] 上缘裁切气泡仍检出 voice_score={top.voice_score:.2f} "
                f"side={top.side} box={top.x},{top.y},{top.w},{top.h} OK"
            )
            n_pass += 1

        # --- 渲染标注图, 供人工确认(不参与断言) ---
        annotated = render_detections(frame, bubbles)
        out_path = os.path.join(tmp, f"{name}.png")
        cv2.imwrite(out_path, annotated)
        print(f"    标注图已保存: {out_path}")

    assert voice_seen["发"], "绿色/右侧语音气泡('发')未被成功验证"
    assert voice_seen["收"], "白色/左侧语音气泡('收')未被成功验证"

    print(f"\n通过用例数: {n_pass}")
    print("ALL PASS")


VOICE_OK = 0.6  # 语音气泡阈值(与实现一致)


if __name__ == "__main__":
    main()