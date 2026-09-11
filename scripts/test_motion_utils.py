"""
Phase1 指标可分离性测试 / Phase1 metric separability test.

合成三类帧, 验证两个新指标:
Synthesize three frame types, verify two new metrics:
  1. 静止帧 / static frame
  2. 滚动帧(垂直平移) / scrolled frame (vertical translation)
  3. 滚动拖影帧(单向运动模糊) / motion-smeared frame (directional blur)

预期 / Expectations:
  - 相位相关: 滚动帧 (dx≈0, dy=位移, peak高); 静止帧 (dx≈0, dy≈0, peak高);
    内容替换(滚动+新行) peak低
  - 梯度方向熵: 静止/滚动帧 gde低(各向同性); 拖影帧 gde高(单向集中)
"""
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.motion_utils import (
    phase_correlation_displacement, gradient_direction_entropy
)


def make_chat_frame(width=320, height=400, scroll_offset=0,
                    num_lines=10) -> np.ndarray:
    """合成聊天界面灰度帧 / Synthetic chat frame (grayscale)"""
    frame = np.ones((height, width), dtype=np.uint8) * 240
    line_height = 32
    start_y = 20 + scroll_offset
    for i in range(num_lines):
        y = start_y + i * line_height
        if 0 < y < height - 30:
            cv2.putText(frame, f"Message line {i + 1:02d} content",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    return frame


def make_motion_smear(clear_frame, kernel_size=15) -> np.ndarray:
    """垂直单向运动模糊 / Vertical directional motion blur"""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[:, kernel_size // 2] = 1.0 / kernel_size
    return cv2.filter2D(clear_frame, -1, kernel)


def make_content_replacement(base, shift) -> np.ndarray:
    """滚动位移 + 底部新行(内容替换) / Shift + new bottom rows"""
    h, w = base.shape
    out = np.zeros_like(base)
    out[:h - shift, :] = base[shift:, :]
    for k in range(shift):
        y = h - shift + k
        cv2.putText(out, f"New message {k + 1:02d}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    return out


def main():
    print("=== Phase1: 相位相关位移估计 / Phase correlation ===")
    static = make_chat_frame()
    scrolled_20 = make_chat_frame(scroll_offset=-20)
    scrolled_50 = make_chat_frame(scroll_offset=-50)
    replacement = make_content_replacement(static, 30)

    for name, cur in [("静止/static", static),
                      ("滚动20px/scrolled20", scrolled_20),
                      ("滚动50px/scrolled50", scrolled_50),
                      ("内容替换/content_repl", replacement)]:
        dx, dy, peak = phase_correlation_displacement(static, cur)
        print(f"  {name:22s}: dx={dx:+.1f} dy={dy:+.1f} peak={peak:.3f}")

    print("\n=== Phase1: 梯度方向熵 / Gradient-direction entropy ===")
    smear = make_motion_smear(static)
    cases = [("静止/static", static), ("滚动50px/scrolled50", scrolled_50),
             ("拖影/smear", smear)]
    for name, img in cases:
        gde = gradient_direction_entropy(img)
        print(f"  {name:22s}: gde={gde:.3f}")

    # 断言 / Assertions
    ok = True
    dx, dy, peak_shift = phase_correlation_displacement(static, scrolled_20)
    if not (abs(dx) < 2 and abs(dy - 20) <= 2 and peak_shift > 0.3):
        print(f"  [FAIL] 滚动20px位移估计不准: dy={dy}")
        ok = False
    else:
        print(f"  [PASS] 滚动20px位移估计 dy={dy:.1f} peak={peak_shift:.3f}")

    _, _, peak_repl = phase_correlation_displacement(static, replacement)
    if not (peak_repl < peak_shift):
        print(f"  [FAIL] 内容替换peak应低于纯平移: {peak_repl:.3f} vs {peak_shift:.3f}")
        ok = False
    else:
        print(f"  [PASS] 内容替换peak={peak_repl:.3f} < 纯平移peak={peak_shift:.3f}")

    gde_static = gradient_direction_entropy(static)
    gde_smear = gradient_direction_entropy(smear)
    if not (gde_smear > gde_static + 0.05):
        print(f"  [FAIL] 拖影gde应显著高于静止: {gde_smear:.3f} vs {gde_static:.3f}")
        ok = False
    else:
        print(f"  [PASS] 拖影gde={gde_smear:.3f} > 静止gde={gde_static:.3f}")

    print(f"\n结果: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
