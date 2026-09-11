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
  1. 颜色分域: 绿色掩码(G 显著大于 R/B)+ 白色掩码(近白且低饱和)。
  2. 在降采样宽度(~320)上对两个掩码分别做连通分量, 得到候选气泡矩形;
     用最小尺寸/面积占比排除大片背景误判。
  3. 对候选框内部提取"墨水掩码"(前景内容比气泡底色明显更深), 依据
     波形 vs 文字 的形态差异计算 voice_score:
       * 波形 = 一条横向窄带上细竖条, 竖条底部对齐、列彼此间隔、约占框内少数高度 → 得分高;
       * 文字 = 多行汉字笔画, 竖直展开大、横向笔画多、密度高 → 得分显著低 (<0.5)。
  4. side: 绿色/偏右 → '发'; 白色/偏左 → '收'。

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

    side: '发'(绿色/右) / '收'(白色/左) / ''(无法判定)
    confidence: 颜色归属 + 形态完整度合成的置信度(0-1)
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

# 颜色判定阈值
GREEN_GAP = 45              # 绿色: G 至少高出 R/B 这么多, 才是明显"绿色占优"
GREEN_SAT_MIN = 70          # 绿色最小饱和度(饱和度=最大值-最小值), 排除灰绿背景
WHITE_LOW = 245             # 白色: 各通道都接近 255(近白)
WHITE_SPREAD = 16           # 白色: 通道间最大差(低饱和), 排除带色调的浅色

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


def _ink_mask(box_bgr: np.ndarray) -> np.ndarray:
    """候选框内的"墨水掩码"= 比框内平均亮度明显更暗的前景(波形/文字笔画)。"""
    gray = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    h, w = gray.shape
    if h < 3 or w < 3:
        return np.zeros((h, w), np.uint8)
    lum = gray
    # 用中值作为底色估计, 更稳健; 前景(波形条/字)通常显著暗于气泡填充色
    baseline = np.median(lum)
    ink = lum < (baseline - 45)          # 相对底色暗 45 个亮度级视为前景
    return (ink * 255).astype(np.uint8)


def _bubble_color_index(color_key: str) -> Tuple[str, float]:
    """由掩码类型得到 side 与基础置信度。"""
    if color_key == "g":
        return "发", 0.95
    if color_key == "w":
        return "收", 0.90
    return "", 0.3


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


def _bubble_confidence(fill_ratio: float, aspect_ok: bool) -> float:
    """由 mask 填充率 + 宽高比合理性合成置信度(0-1)。"""
    c = 0.35 + 0.45 * fill_ratio
    if aspect_ok:
        c += 0.2
    return float(np.clip(c, 0.0, 1.0))


def _collect_from_mask(
    small: np.ndarray,
    color_mask: np.ndarray,
    color_key: str,
    scale: float,
) -> List[VoiceBubble]:
    """在单一颜色掩码上收集候选气泡。"""
    out: List[VoiceBubble] = []
    if cv2 is None:
        return out
    H, W = small.shape[:2]
    frame_area = float(H * W)
    ncomp1, labels, stats, _ = cv2.connectedComponentsWithStats(
        color_mask.astype(np.uint8), 8
    )
    side_base, _ = _bubble_color_index(color_key)
    for i in range(1, ncomp1):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 60:                    # 噪点
            continue
        if w < MIN_BUBBLE_W:             # 太窄, 不是气泡
            continue
        if h < 9 or h > H * 0.55:        # 太扁或太高的框多非普通语音气泡
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
        conf = _bubble_confidence(fill, aspect_ok)
        side = side_base if fill > 0.4 else ""
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

    空白/纯背景输入返回 []。返回的坐标在原 ROI 分辨率。
    """
    if cv2 is None:
        raise RuntimeError("cv2 不可用, 无法执行语音气泡检测")
    if roi_bgr is None or roi_bgr.size == 0:
        return []
    if roi_bgr.ndim != 3 or roi_bgr.shape[2] < 3:
        return []
    small, scale = _downsample(roi_bgr)

    bubbles: List[VoiceBubble] = []
    bubbles += _collect_from_mask(small, _green_mask(small), "g", scale)
    bubbles += _collect_from_mask(small, _white_mask(small), "w", scale)
    # 去重叠(同一气泡可能被双掩码命中): 保留 voice_score 更高者
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