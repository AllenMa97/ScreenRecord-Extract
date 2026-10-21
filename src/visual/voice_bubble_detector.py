"""
轻量语音气泡检测 (Lightweight voice-bubble detection).

用途 / Purpose
-------------
聊天软件(如微信)中的"语音消息"是**没有文字**的气泡:
  - 绿色气泡, 偏右侧, 我方发出 (green, right side, '发' / outgoing);
  - 白色气泡, 偏左侧, 对方发出 (white, left side, '收' / incoming);
气泡内部是一条"波形图"(一簇高低起伏、底部对齐的细竖条), 以及可选的小时长标注(如 "3″")。
现有 OCR 管线读不到语音气泡内的文字, 故提供本检测器, 在聊天区彩色帧(ROI)中找出
"语音气泡", 得到其出现时刻框与方向('发'/'收'), 作为把音频转写结果配对回聊天记录的
时间锚点。

算法思路 / Algorithm
--------------------
纯启发式, CPU-friendly, 无深度学习依赖, 确定性 deterministic:
  1. 候选分割(三路掩码合并, 去重叠):
     - 背景自适应掩码(**颜色无关, 主力**): 先估计背景(全局中位数 + 逐行中位数,
       行被大面积前景占据时退回全局), 再按"与背景的色差"经 Otsu 自适应阈值切
       前景。不依赖气泡的具体颜色, 故对纯色底与**图片/照片壁纸**、以及任意气泡
       配色(非绿/非白)都成立;
     - 颜色域掩码: 绿色掩码(G 显著大于 R/B) + 白色掩码(近白且低饱和)。它们只
       作为**辅助加分**(颜色与位置一致时略升置信度), 不再是方向或存在性的必要条件。
  2. 在降采样宽度(~320)上对各掩码分别做连通分量, 得到候选气泡矩形;
     用最小尺寸/面积占比/填充率排除大片背景与噪点。
  3. 对候选框内部提取"墨水掩码"(前景内容比气泡底色明显更深), 依据
     波形 vs 文字 的形态差异计算 voice_score:
       * 波形 = 一条横向窄带上细竖条, 竖条底部对齐、列彼此间隔、约占框内少数高度 → 得分高;
       * 文字 = 多行汉字笔画, 竖直展开大、横向笔画多、密度高 → 得分显著低 (<0.5)。
  4. side: 以**气泡水平位置为主判据**——中心 x 在会话区中线左侧 → '收'(对方),
     右侧 → '发'(本方); 颜色(绿/白)只作辅助加分, **不再是必要条件**——非绿色
     气泡也能凭位置被判出发/收。中心落在中线附近死区时才回退用颜色判定。

性能 / Performance: 单帧调用一次, 秒级内即可(主要在 ~320 宽度上运算, box 映射回原图)。
对空白 / 纯背景输入返回 []。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:  # 项目已依赖 opencv-python / opencv-python-headless
    import cv2
except Exception:  # pragma: no cover - 不可用时才退化
    cv2 = None


@dataclass
class VoiceBubble:
    """一个检测到的语音气泡候选。

    side: '发'(本方/气泡靠右) / '收'(对方/气泡靠左) / ''(无法判定)
    confidence: 填充率 + 形态完整度 + 颜色一致性加分 合成的置信度(0-1)
    voice_score: 该气泡像"语音气泡(波形)"的程度(0-1); 文本气泡应显著低
    """

    x: int
    y: int
    w: int
    h: int
    side: str = ""            # '发' / '收' / ''
    confidence: float = 0.0
    voice_score: float = 0.0


# --------------------------------------------------------------------------
# 常量与阈值 (判定依据用中文注释)
# --------------------------------------------------------------------------
TARGET_WIDTH = 320          # 降采样工作宽度; 先在此分辨率做颜色/连通, 再映射回原图
MIN_BUBBLE_W = 50           # 最小气泡宽度(相对工作宽度), 排除噪点  ~ 对应真实聊天气泡
MAX_BOX_AREA_RATIO = 0.62   # 连通域面积不得超过画面该比例, 防止整片背景被当气泡
MIN_FILL_RATIO = 0.30       # mask 在框内最小填充率(气泡应是实心的)
MAX_FILL_RATIO = 0.99       # mask 在框内最大填充率(全满多半是背景/异常)

# 几何比例上限(相对 ROI 宽/高, 与分辨率无关) / Relative geometry caps (ROI-relative)
# 实测标定(真实录屏 432x900 与合成 540x1200 两套 ROI):
#   * 真实语音气泡宽度只占 ROI 宽的 0.16~0.51(带转写文字的宽气泡最高约 0.67),
#     高度只占 ROI 高的 0.02~0.07;
#   * 伪检(应用内横幅、输入框下沿、整片/半屏背景、大块墙纸)宽度占 0.74~1.00,
#     高度可达 0.21~0.48。
# 故取 0.70 作宽度上限(刚好放行 0.67 的带转写语音气泡, 挡掉 ≥0.74 的横幅/背景),
# 取 0.35 作高度上限(远高于单气泡 0.07, 挡掉大块背景)。
# / Measured on real (432x900) and synthetic (540x1200) ROIs. Real bubbles span
#   0.16~0.51 of the ROI width (0.67 with an in-bubble transcript); the spurious
#   detections span 0.74~1.00 (banners / input bar / background slabs). Hence a
#   width cap of 0.70 and a generous height cap of 0.35.
MAX_BUBBLE_W_RATIO = 0.70   # 候选框宽度不得超过 ROI 宽度的该比例, 排除横幅/整片背景
MAX_BUBBLE_H_RATIO = 0.35   # 候选框高度不得超过 ROI 高度的该比例, 排除大块背景

# 颜色辅助阈值 (auxiliary color cues; 不再是方向判定的必要条件)
GREEN_GAP = 45              # 绿色: G 至少高出 R/B 这么多, 才是明显"绿色占优"
GREEN_SAT_MIN = 70          # 绿色最小饱和度(饱和度=最大值-最小值), 排除灰绿背景
WHITE_LOW = 245             # 白色: 各通道都接近 255(近白)
WHITE_SPREAD = 16           # 白色: 通道间最大差(低饱和), 排除带色调的浅色
# 粉/品红气泡(部分聊天主题的"发"方气泡): B 与 R 高、G 低。
# 背景自适应掩码在图片壁纸 + 大块亮色区域时会把顶部气泡与壁纸连成一片, 超过
# `MAX_BUBBLE_H_RATIO` 被整块拒绝(实测漏检 4 只粉色气泡), 故补一路专用颜色掩码。
PINK_MIN_B = 190            # 蓝通道下限
PINK_MIN_R = 120            # 红通道下限
PINK_MAX_G = 140            # 绿通道上限(粉/品红的核心特征)
# 对外接口别名(与既有命名保持兼容) / Public aliases kept for compatibility
VOICE_GREEN_MIN_G_DIFF = GREEN_GAP
VOICE_GREEN_MIN_SAT = GREEN_SAT_MIN
WHITE_MAX_SPREAD = WHITE_SPREAD

# 方向判定: 位置为主, 颜色为辅 / Direction: position first, color auxiliary
MIDLINE_DEAD_ZONE_RATIO = 0.04   # 会话区中线附近死区(占 ROI 宽比例), 此处回退用颜色
COLOR_BONUS = 0.05               # 颜色与位置一致时的置信度加分 / confidence bonus

# 背景自适应前景分割(颜色无关, 对纯色底/图片壁纸都适用) / Background-adaptive
BG_ADAPT_ENABLE = True      # 总开关: 背景自适应前景分割(与气泡颜色无关)
BG_DIFF_FLOOR = 22          # 前景色差下限(保底), 防止 Otsu 阈值取过小
BG_ROW_TOL = 24             # 行背景偏离全局背景的容差, 超则退回全局背景

VOICE_THRESHOLD = 0.58      # voice_score >= 0.58 判定为语音气泡(波形)
TEXT_MAX_SCORE = 0.50       # 文本气泡 voice_score 应 < 0.5
# --------------------------------------------------------------------------


def _downsample(img_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    """等比缩放到 ~TARGET_WIDTH 宽度, 返回 (小图, 缩放比例 scale=W_small/W_orig)。"""
    h, w = img_bgr.shape[:2]
    if w <= TARGET_WIDTH:
        return img_bgr, 1.0
    scale = TARGET_WIDTH / w
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    small = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, scale


def _green_mask(bgr: np.ndarray) -> np.ndarray:
    """绿色寻址掩码: G 显著大于 R/B, 且饱和度高(我方绿色气泡)。返回 uint8 0/255。"""
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    sat = g - np.minimum(b, r)          # 近似饱和度(以 G 为主峰值)
    mask = (g - b > GREEN_GAP) & (g - r > GREEN_GAP) & (sat > GREEN_SAT_MIN)
    return (mask * 255).astype(np.uint8)


def _white_mask(bgr: np.ndarray) -> np.ndarray:
    """白色寻址掩码: 各通道都是近白(>WHITE_LOW)且彼此接近(低饱和, 通道差值小)。"""
    arr = bgr.astype(np.int16)
    mn = arr.min(axis=2)
    mx = arr.max(axis=2)
    near_white = mn > WHITE_LOW            # 各通道均高于阈值(含纯白 255)
    low_sat = (mx - mn) < WHITE_SPREAD     # 通道间最大差小 = 低饱和
    return ((near_white & low_sat) * 255).astype(np.uint8)


def _pink_mask(bgr: np.ndarray) -> np.ndarray:
    """粉/品红掩码: B、R 高而 G 低(如 BGR≈[255,77,181])。

    仅用于补背景自适应掩码在图片壁纸场景下漏检的粉色气泡; 是否成气泡仍由
    几何/填充率/voice_score 统一把关。
    """
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    mask = (b > PINK_MIN_B) & (r > PINK_MIN_R) & (g < PINK_MAX_G)
    mask = (mask * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def _background_estimate(bgr: np.ndarray) -> np.ndarray:
    """估计背景色: 全局中位数 + 逐行中位数(纵向自适应)。

    逐行中位数能适应"图片背景在纵向变化"的情形; 若某行被大面积前景(长气泡)占据,
    其中位数会偏离全局背景, 此时退回全局背景值, 避免把整行误当前景。
    / Per-row median background (vertically adaptive), falling back to the global
    median on rows dominated by foreground. 返回 (H, 3) int16, 每行一个背景色。
    """
    arr = bgr.astype(np.int16)
    global_bg = np.median(arr.reshape(-1, 3), axis=0)          # (3,)
    row_bg = np.median(arr, axis=1)                            # (H, 3)
    dev = np.abs(row_bg - global_bg[None, :]).max(axis=1)      # (H,)
    row_bg[dev > BG_ROW_TOL] = global_bg
    return row_bg.astype(np.int16)


def _background_mask(bgr: np.ndarray) -> np.ndarray:
    """背景自适应前景掩码: 与(逐行)背景的色差超过 Otsu 自适应阈值 → 前景。

    对照片/纹理壁纸鲁棒: 阈值由当前帧的色差直方图经 Otsu 自适应得出, 并加
    BG_DIFF_FLOOR 保底; 不依赖气泡的具体颜色, 故非绿/非白气泡也能被切出。
    / Foreground mask via color distance to the estimated background, thresholded
    by Otsu (auto-adaptive) with a floor; color-agnostic.
    """
    arr = bgr.astype(np.int16)
    bg = _background_estimate(bgr)                             # (H, 3)
    dev = np.abs(arr - bg[:, None, :]).max(axis=2)             # (H, W)
    dev_u8 = np.clip(dev, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(dev_u8, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(max(thr, BG_DIFF_FLOOR))
    mask = ((dev > thr) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)    # 去散噪
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)   # 补内部小洞
    return mask


def _color_side(color_key: str) -> str:
    """由颜色掩码类型得到辅助方向('发'绿 / '收'白 / '' 未知)。"""
    if color_key == "g":
        return "发"
    if color_key == "w":
        return "收"
    return ""


def _resolve_side(cx: float, mid_x: float, dead: float, color_key: str) -> str:
    """方向判定: 位置为主判据, 颜色为辅助。

    中心 x 明显偏左(收)/偏右(发)时直接按位置判定; 仅当落在中线死区内(位置无法
    判定)时, 才用颜色辅助。非绿/非白气泡也能仅凭位置得到方向。
    / Position-primary side; color only breaks ties inside the midline dead zone.
    """
    if cx < mid_x - dead:
        return "收"
    if cx > mid_x + dead:
        return "发"
    return _color_side(color_key)


def _ink_mask(box_bgr: np.ndarray) -> np.ndarray:
    """候选框内的"墨水掩码"= 与框内底色(填充色)明显不同的前景(波形/文字笔画)。

    用**绝对偏差**而非仅"更暗": 气泡底色可能是深色(如深蓝发送气泡), 其上的波形/
    文字为浅色, 只认"更暗"会把这类气泡的 voice_score 判成 0 而整侧漏检。
    / Deviation from the bubble fill color in either polarity, so dark bubbles with
    light ink are not scored as empty.
    """
    gray = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    h, w = gray.shape
    if h < 3 or w < 3:
        return np.zeros((h, w), np.uint8)
    lum = gray
    # 用中值作为底色估计, 更稳健; 前景(波形条/字)与底色差异显著(不论明暗)
    baseline = np.median(lum)
    ink = np.abs(lum - baseline) > 45    # 相对底色偏离 45 个亮度级视为前景
    return (ink * 255).astype(np.uint8)


def _voice_score(ink: np.ndarray) -> float:
    """判断框内内容像"波形语音" 的程度(0-1)。

    波形 vs 文字 的判别特征(以"竖柱保留 + 底部对齐"为核心:
    语音波形 = 底部对齐的细竖条列; 文字虽也能被竖开保留笔画, 但多行文字
    纵向展开大、列底部参差, 会在 align/narrow_band 上显著拉低):
      - 竖柱保留(bars_keep): 细竖条被竖直开运算保留高, 语音≈1、文字偏高。
      - 底部对齐(align): 波形列底沿集中同一基线; 文字多行底部参差。
      - 无横向蔓延(thin_horiz): 波形无横笔画; 文字含大量横向笔画。
      - 纵向展开小(narrow_band): 波形只占窄带; 多行文字占大高度 → 强判别。
      - 稀疏度(sparse_lo): 波形密度极低; 文字笔画填得多 → 弱辅助。
    """
    h, w = ink.shape
    if h < 4 or w < 4:
        return 0.0
    # 归一为 0/1 二值, 保证后续 sum() 得到的是像素个数而非 255 倍个数
    ink = (ink > 0).astype(np.uint8)
    nib = int(ink.sum())
    if nib < 16:
        return 0.0
    total = float(nib)

    # --- 特征 1: 稀疏度 (密度越低越像波形) ---
    density = total / (h * w)
    sparse_lo = 1.0 - min(density / 0.42, 1.0)

    # --- 特征 2+3: 竖直/水平开运算保留率 ---
    v_open = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((9, 1), np.uint8))
    h_open = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 9), np.uint8))
    vert_ret = v_open.sum() / total   # 细竖条被保留的比例
    horiz_ret = h_open.sum() / total  # 横笔画被保留的比例

    bars_keep = min(vert_ret / 0.55, 1.0)          # 竖柱保留→波形加分
    thin_horiz = 1.0 - min(horiz_ret / 0.45, 1.0)  # 横向笔画少→波形加分

    # --- 特征 4: 列底沿对齐度 ---
    colbottoms = []
    for c in range(w):
        col = np.where(ink[:, c] > 0)[0]
        if col.size:
            colbottoms.append(col.max())
    if colbottoms:
        align_std = float(np.std(colbottoms)) / max(h, 1)
        align = max(0.0, 1.0 - align_std * 4.0)
    else:
        align = 0.0

    # --- 特征 5: 纵向展开小(单窄带) ---
    rows = np.where(ink.sum(axis=1) > 0)[0]
    if rows.size:
        span = (rows.max() - rows.min() + 1) / max(h, 1)
        narrow_band = max(0.0, 1.0 - span * 1.6)
    else:
        narrow_band = 0.0

    # 权重: 竖柱保留率是最直观的波形特征, 底部对齐次之; 其余为辅助。
    # 纯文字多行气泡会因纵向展开大(narrow_band低)+横向笔画多(thin_horiz低)
    # 而被压到较低分数; 语音波形则三项皆高 → 区分主要由这两项贡献。
    score = (
        0.45 * bars_keep
        + 0.20 * align
        + 0.15 * thin_horiz
        + 0.12 * narrow_band
        + 0.08 * sparse_lo
    )
    return float(np.clip(score, 0.0, 1.0))


def _bubble_confidence(fill_ratio: float, aspect_ok: bool,
                       color_bonus: float = 0.0) -> float:
    """由 mask 填充率 + 宽高比合理性 + 颜色一致性加分合成置信度(0-1)。"""
    c = 0.35 + 0.45 * fill_ratio
    if aspect_ok:
        c += 0.2
    c += color_bonus
    return float(np.clip(c, 0.0, 1.0))


def _collect_from_mask(
    small: np.ndarray,
    color_mask: np.ndarray,
    color_key: str,
    scale: float,
    mid_x: Optional[float] = None,
    dead_zone: Optional[float] = None,
) -> List[VoiceBubble]:
    """在单一(颜色/背景自适应)掩码上收集候选气泡。

    方向以气泡中心 x 相对会话区中线为准(位置为主), color_key 仅作辅助加分;
    故非绿/非白气泡也能得到方向。mid_x/dead_zone 缺省时按工作宽度自推。
    / Collect candidates from one mask; side is decided by horizontal position.
    """
    out: List[VoiceBubble] = []
    if cv2 is None:
        return out
    H, W = small.shape[:2]
    frame_area = float(H * W)
    if mid_x is None:
        mid_x = W / 2.0
    if dead_zone is None:
        dead_zone = MIDLINE_DEAD_ZONE_RATIO * W
    color_side_aux = _color_side(color_key)
    ncomp1, labels, stats, _ = cv2.connectedComponentsWithStats(
        color_mask.astype(np.uint8), 8
    )
    for i in range(1, ncomp1):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 60:                    # 噪点
            continue
        if w < MIN_BUBBLE_W:             # 太窄, 不是气泡
            continue
        if h < 9:                        # 太扁的框多非普通语音气泡
            continue
        # 几何比例上限: 气泡不可能横跨整幅 ROI, 也不该占掉大半屏高。
        # 注意仅用相对比例(相对工作分辨率 W/H, 与原图同比例), 不写死像素。
        # / ROI-relative caps (scale-invariant). A bubble never spans the full ROI
        #   width nor takes up a large fraction of its height.
        if w > W * MAX_BUBBLE_W_RATIO:   # 太宽 → 横幅/输入框/整片背景
            continue
        if h > H * MAX_BUBBLE_H_RATIO:   # 太高 → 大块背景/墙纸
            continue
        box_area = float(w * h)
        if box_area > frame_area * MAX_BOX_AREA_RATIO:  # 太满→整片背景
            continue
        fill = area / box_area
        if fill < MIN_FILL_RATIO or fill > MAX_FILL_RATIO:
            continue
        aspect_ok = (w / h) > 1.5        # 语音气泡横长>高
        box_small = small[y : y + h, x : x + w]
        ink = _ink_mask(box_small)
        vs = _voice_score(ink)
        cx = x + w / 2.0
        side = _resolve_side(cx, mid_x, dead_zone, color_key) if fill > 0.4 else ""
        # 颜色与位置一致时给置信度加分(仅加分, 不惩罚; 颜色不再是必要条件)
        bonus = COLOR_BONUS if (color_side_aux and color_side_aux == side) else 0.0
        conf = _bubble_confidence(fill, aspect_ok, bonus)
        out.append(
            VoiceBubble(
                x=int(round(x / scale)),
                y=int(round(y / scale)),
                w=int(round(w / scale)),
                h=int(round(h / scale)),
                side=side,
                confidence=float(conf),
                voice_score=vs,
            )
        )
    return out


def detect_voice_bubbles(roi_bgr: np.ndarray) -> List[VoiceBubble]:
    """输入聊天区彩色帧(BGR uint8 HxWx3), 返回检测到的语音气泡列表。

    方向以气泡水平位置(相对会话区中线)为主判据, 颜色仅辅助; 前景分割用颜色无关的
    背景自适应掩码, 对纯色底与图片/照片壁纸、任意气泡配色都成立。空白/纯背景输入
    返回 []。坐标在原 ROI 分辨率。
    """
    if cv2 is None:
        raise RuntimeError("cv2 不可用, 无法执行语音气泡检测")
    if roi_bgr is None or roi_bgr.size == 0:
        return []
    if roi_bgr.ndim != 3 or roi_bgr.shape[2] < 3:
        return []
    small, scale = _downsample(roi_bgr)
    H, W = small.shape[:2]
    mid_x = W / 2.0                            # 会话区中线(工作分辨率)
    dead_zone = MIDLINE_DEAD_ZONE_RATIO * W

    bubbles: List[VoiceBubble] = []
    bubbles += _collect_from_mask(small, _green_mask(small), "g", scale,
                                  mid_x, dead_zone)
    bubbles += _collect_from_mask(small, _white_mask(small), "w", scale,
                                  mid_x, dead_zone)
    bubbles += _collect_from_mask(small, _pink_mask(small), "p", scale,
                                  mid_x, dead_zone)
    # 背景自适应路径: 与气泡颜色无关, 对纯色底与图片壁纸**一律启用**。颜色掩码只能
    # 命中绿/白气泡, 若只靠它们, 换了气泡配色的录屏(如深蓝发送气泡)会整侧漏检。
    # / Always run the color-agnostic bg-adaptive path; the green/white masks alone
    #   would miss bubbles with other palettes (e.g. a dark-blue outgoing bubble).
    if BG_ADAPT_ENABLE:
        bubbles += _collect_from_mask(small, _background_mask(small), "b",
                                      scale, mid_x, dead_zone)
    # 去重叠(同一气泡可能被多掩码命中): 保留 voice_score 更高者
    bubbles.sort(key=lambda b: -b.voice_score)
    keep: List[VoiceBubble] = []
    for b in bubbles:
        if any(_iou(b, k) > 0.5 for k in keep):
            continue
        keep.append(b)
    return keep


def _iou(a: VoiceBubble, b: VoiceBubble) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = a.w * a.h + b.w * b.h - inter
    return inter / ua if ua > 0 else 0.0


def render_detections(roi_bgr: np.ndarray, bubbles: List[VoiceBubble]) -> np.ndarray:
    """在 ROI 上画出各气泡边界框, 标注 side/voice_score, 返回标注图(调试用)。"""
    if cv2 is None:
        return roi_bgr
    img = roi_bgr.copy()
    for b in bubbles:
        side = b.side if b.side else "?"
        color = (0, 220, 0) if b.side == "发" else (0, 220, 220)
        cv2.rectangle(img, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
        label = f"{side} vs={b.voice_score:.2f} cf={b.confidence:.2f}"
        cv2.putText(
            img, label, (b.x, max(0, b.y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1, cv2.LINE_AA,
        )
    return img


if __name__ == "__main__":  # 简单自检(可选)
    demo = np.full((300, 400, 3), 212, np.uint8)
    print(detect_voice_bubbles(demo))  # 纯背景 → []