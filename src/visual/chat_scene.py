"""
聊天界面判定 / Chat-scene detection
===================================

用途 / Purpose
-------------
录屏视频里用户会在聊天界面与"非聊天界面"之间来回切换。非聊天界面包括两类:
  1. **全屏查看器**: 点开聊天里的转账回单/截图图片后全屏查看的那几帧;
  2. **独立页面**: 设置页(关于手机)、银行 App、小程序页等。
这些界面的文字会被 OCR 当成聊天消息, 污染正文(如"交易明细查询""设备名称"
"转账回单")。本模块在**帧级**判定当前 ROI 是不是聊天界面, 非聊天帧的 OCR 行
整帧丢弃, 从入口处把非聊天内容挡在正文之外。

算法 / Algorithm (纯启发式, CPU-friendly, 确定性)
-----------------------------------------------
判据分两级, **无法判定时默认放行**(不误杀):

1. **聊天墙纸占比**(正面判据)。聊天界面(本数据集中为浅灰墙纸亮度约 237)的
   背景是 App 墙纸, 消息气泡(浅绿/纯白)悬在墙纸上:

       wallpaper_ratio = 中性色(max-min <= 8) 且亮度在 [230, 244] 的像素占比

   若 wallpaper_ratio >= 阈值(默认 0.35) → 聊天界面(墙纸可见)。

2. **纯平非聊天页**(反面判据, 仅在第 1 步不成立时启用)。观察到的非聊天页面
   都是**纯平的中性浅色页**: 小程序页(浅灰 ~222)、全屏回单/白页(~255)、
   设置/列表页(~242~246)。这些页面 = 中性色占比高 且 中性像素亮度集中在很窄的
   一带(亮度标准差小)。满足则判为非聊天界面。

   若第 1、2 步都不成立(背景是**图片/照片/纹理壁纸** —— 彩色或层次丰富的灰度),
   判据无法可靠区分 → **默认放行**(判为聊天界面), 避免把"图片壁纸上的聊天"误杀。

亮度带之所以收紧到 [230, 244](而非宽松的 [210, 250]), 是因为**"中性浅色"
本身不足以区分聊天与非聊天**: 观测到的非聊天页面同样是中性浅灰, 只是亮度
与聊天墙纸(237)不同(小程序页 ~222 / 回单白页 255 / 列表页 242~246), 会被
宽松带算成"墙纸"而漏判。实测(两个带音频视频, 432x780 ROI):
  - 聊天帧 wallpaper_ratio[230,244] 0.46 ~ 0.83;
  - 小程序页/回单/设置/白页 0.02 ~ 0.15;
  - 阈值 0.35 两侧留有余量。
  - 叠加在聊天上的界面(时间分隔条/长按菜单)墙纸仍可见(>=0.46), 仍判为聊天帧,
    交由下游截图折叠/界面元素过滤处理。
本判据只区分"墙纸/纯平页", 与气泡颜色无关。阈值可在 config.yaml 的
roi.chat_wall_ratio 调整; 若换设备后墙纸明显偏亮/偏暗, 可相应调整本模块的
WALL_MIN_VALUE / WALL_MAX_VALUE。

安全性 / Safety
--------------
判据只用于**丢弃**非聊天帧的行, 不做任何内容改写; 且跟踪侧对同一行有多帧投票,
某行只要在任一聊天帧出现过就能建行, 因此个别帧的误判(边界过渡帧)不会丢内容,
只会丢掉"始终只出现在非聊天界面"的行。检测器不可用时(无 cv2)保守返回聊天
界面(不丢内容)。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:  # 项目已依赖 opencv-python / opencv-python-headless
    import cv2
except Exception:  # pragma: no cover - 不可用时才退化
    cv2 = None

# 判定工作宽度: 与语音气泡检测器一致, 先降采样再统计(兼顾速度)
from src.visual.voice_bubble_detector import _downsample

TARGET_WIDTH = 320          # 判定工作宽度
WALL_NEUTRAL_TOL = 8        # "中性色"判定: 三通道极差 <= 此值(排除彩色内容)
WALL_MIN_VALUE = 230        # 墙纸亮度下限(排除小程序页浅灰 ~222 与深色页面)
WALL_MAX_VALUE = 244        # 墙纸亮度上限(排除近纯白 255 文档/回单底, 及 ~246 列表页)
WALL_MIN_RATIO = 0.35       # 墙纸像素占 ROI 比例 >= 此值 → 聊天界面

# "纯平非聊天页"判据(第 2 级): 中性色主导 + 亮度集中在一条窄带(纯平背景)
PLAIN_NEUTRAL_TOL = 8       # 中性色判定容差(与墙纸同口径) / neutral tolerance
PLAIN_NEUTRAL_MIN_RATIO = 0.65   # 中性色占比 >= 此值才考虑"纯平页" / neutral dominance
PLAIN_BG_BAND = 10          # 亮度窄带半宽(px) / brightness band half-width
PLAIN_FLAT_MIN_RATIO = 0.85 # 窄带内像素占比 >= 此值 → 纯平(非图片) / flatness


def wallpaper_ratio(roi_bgr: np.ndarray) -> float:
    """中性浅色"墙纸"像素占 ROI 的比例。

    墙纸 = 三通道极差 <= WALL_NEUTRAL_TOL 且亮度落在 [WALL_MIN_VALUE,
    WALL_MAX_VALUE] 的像素。近纯白(文档/回单底)、小程序页浅灰与其他亮度/彩色
    页面均不计入。

    调用方需保证 roi_bgr 为可用彩色帧; 不可用时返回 0.0。
    """
    if roi_bgr is None or roi_bgr.size == 0 or roi_bgr.ndim != 3:
        return 0.0
    flat = roi_bgr.reshape(-1, 3).astype(np.int16)
    mn = flat.min(axis=1)
    mx = flat.max(axis=1)
    hit = (mx - mn <= WALL_NEUTRAL_TOL) & (mn >= WALL_MIN_VALUE) & (mx <= WALL_MAX_VALUE)
    return float(np.mean(hit))


def is_plain_page(roi_bgr: np.ndarray) -> bool:
    """是否像"纯平的非聊天页"(文档/回单/设置/小程序浅灰页)。

    判据: 中性色像素占比高(PLAIN_NEUTRAL_MIN_RATIO) 且这些中性像素的亮度集中在
    一条窄带内 —— 即以"亮度中位数 ± PLAIN_BG_BAND"的像素占比 >= PLAIN_FLAT_MIN_RATIO
    表示背景纯平(少量文字/图标不足以拉低该占比)。图片/照片壁纸为彩色分布, 或虽是
    灰度但层次丰富(亮度分散), 均不满足。
    / True if the frame looks like a flat neutral page (non-chat document/settings).
    """
    if roi_bgr is None or roi_bgr.size == 0 or roi_bgr.ndim != 3:
        return False
    flat = roi_bgr.reshape(-1, 3).astype(np.int16)
    mn = flat.min(axis=1)
    mx = flat.max(axis=1)
    neutral = (mx - mn <= PLAIN_NEUTRAL_TOL)
    if float(neutral.mean()) < PLAIN_NEUTRAL_MIN_RATIO:
        return False
    vals = flat[neutral].mean(axis=1)          # 中性像素亮度
    med = float(np.median(vals))
    flat_ratio = float(np.mean(np.abs(vals - med) <= PLAIN_BG_BAND))
    return flat_ratio >= PLAIN_FLAT_MIN_RATIO


def chat_scene_reason(roi_bgr: np.ndarray,
                      wall_min_ratio: float = WALL_MIN_RATIO
                      ) -> Tuple[bool, str]:
    """判定 ROI 帧是否为聊天界面, 返回 (是否聊天, 判据说明)。

    两级判据: 先看聊天墙纸占比, 再看是否为"纯平非聊天页"; 两者都不成立(图片/
    照片壁纸等无法可靠判定)时**默认放行**(判为聊天界面), 不误杀。
    未知/不可用输入同样保守判为聊天界面(不丢内容)。
    / Two-level decision; falls back to "allow" when undecidable (image wallpaper).
    """
    if cv2 is None or roi_bgr is None or roi_bgr.size == 0:
        return True, "unavailable"
    if roi_bgr.ndim != 3 or roi_bgr.shape[2] < 3:
        return True, "unavailable"
    small, _scale = _downsample(roi_bgr)
    ratio = wallpaper_ratio(small)
    if ratio >= wall_min_ratio:
        return True, "wallpaper"
    if is_plain_page(small):
        return False, "plain_page"     # 纯平非聊天页(文档/回单/设置/小程序浅灰)
    return True, "undecided_bg"        # 图片/照片壁纸等无法判定 → 默认放行


def is_chat_scene(roi_bgr: np.ndarray,
                  wall_min_ratio: float = WALL_MIN_RATIO) -> bool:
    """ROI 帧是否为聊天界面(chat_scene_reason 的布尔封装)。"""
    return chat_scene_reason(roi_bgr, wall_min_ratio)[0]
