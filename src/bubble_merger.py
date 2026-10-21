"""
气泡内换行合并

纯几何(CV侧)方案, 不依赖LLM: 聊天界面中一条长消息会被自动换行显示成多行,
OCR 返回的也是多行(每行带独立bbox)。组装前必须先按"同一气泡"的几何关系把
这些换行碎片合并为一条完整消息, 否则一条完整消息会被拆成多条"语义不完整"
的结果, 且后续消息的合并/时间/收发归属全被带偏。

核心原则: 同一聊天气泡 = 同一条消息。气泡内所有 OCR 行合并为一条消息,
行与行之间以字面 "\\n" 连接(前端据此还原为气泡内换行)。

三步处理:
  1. 同视觉行水平碎片预合并: OCR 可能把同一视觉行切成左右两个 bbox
     (如 "3.您的支付宝芝麻信用分多少分" + "(请您打"), 先把它们拼回一行,
     收发方向以左侧段为准, 避免右半段被误判为另一条消息。
  2. 同气泡分组: 收发方向一致/任一边缘对齐 + 水平重叠 + 垂直间距在行距
     范围内。间距超过普通行距(_BUBBLE_GAP_MAX)但在段落空行范围内
     (_BUBBLE_GAP_RELAXED)时, 用视频帧的像素验证: 两条文本行之间的竖向
     间隙是否为同一气泡的白色底(而不是两个气泡之间的墙纸), 通过才并入。
  3. 组内整条合并: 一个组(一个气泡)的所有行合并为一条消息; 界面换行截断
     的满行直接粘接(消除拆词重叠), 短行/大间距处保留为消息内换行(字面\\n)。

判定依据(几何 + 轻文本 + 像素, 全部保守, 宁可不合也不错合):
  - 收发方向一致(或任一边缘对齐), 边缘锚点对齐(容差内), 水平有重叠
  - 换行证据: 前一行是"满行"(宽度>=气泡最大宽度的 _FULL_LINE_RATIO)且
    句子未完结 → 被界面换行截断, 与下一行粘接;
  - 拆词重叠消除: 换行处可能把一个词切开, 拼接时去掉跨行重复的最多
    _MAX_OVERLAP 个字符。
"""

import re
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.data_types import TrackedLine, TextLine

# 聊天时间戳行: "23:53"
# 时间戳行作为气泡分隔, 不参与合并
STAMP_RE = re.compile(r"\d{1,2}:\d{2}\s*√?\s*")

_TERMINAL_PUNCT_RE = re.compile(r"[。！？!?…~；;]$")
_BUBBLE_ALIGN_TOL = 40      # 同气泡边缘对齐容差(px)
_BUBBLE_GAP_MAX = 12.0      # 普通行距上限(px): 不超过则无需像素验证
_BUBBLE_GAP_RELAXED = 70.0  # 段内空行最大间距(px): 超过必为两个气泡
_BUBBLE_GAP_OVERLAP_MAX = -8.0   # 允许的bbox重叠(px, 负gap)
_FULL_LINE_RATIO = 0.75     # 满行宽度比例
_MAX_OVERLAP = 4            # 换行拆词最大重叠字符数
_ROW_FRAGMENT_OVERLAP = 0.7  # 同视觉行垂直重叠比例
_ROW_FRAGMENT_GAP_MAX = 40.0  # 同视觉行左右碎片最大水平间隙(px)
_ROW_FRAGMENT_LEFT_MAX = 40.0  # 右碎片左缘最多侵入左碎片右缘左侧的宽度(px), 防止滚动回看中
                               # 两条不同消息的"最新bbox"垂直重叠被误判为同行左右碎片
# 换行尾片段归属判据(见 absorb_wrap_tail_leftovers) / wrap-tail leftover:
_TAIL_MIN_LEN = 2              # 片段最短字数(更短的当噪声, 不归属)
_TAIL_MIN_DIFF = 2             # 片段与父消息的最短长度差(等长=完全重复, 由其它层处理)
_WHITE_MIN = 225            # 气泡白底判定下限: 三通道最小值
_WALLPAPER_RUN = 2          # 连续非白(墙纸)行数下限, 证明两行间存在气泡间隙
# 自适应白底阈值: 固定下限对深色墙纸够用, 但浅色墙纸会把下限越过(本次真实录屏
# 墙纸约237 > 225), 于是墙纸也被当成气泡白底 → 相邻的两条不同消息被误并成一条
# (结果里凭空多出字面 \n)。这里用"气泡白底是文本行bbox内最亮的中性色"这一先验
# 就地定标: 取两条文本行bbox内 min通道 的高分位值(≈气泡白底, 字形只是少数暗像素)
# 减去余量, 与下限取较大者作为实际白底阈值。
_WHITE_ADAPT_MARGIN = 12    # 白底阈值相对本地白底亮度的下调余量(JPEG噪声)
_WHITE_ADAPT_PCT = 90       # 本地白底亮度取文本行bbox内min通道的高分位
# 非白色卡片/气泡填充色匹配容差(各通道差)。必须小于"纯白气泡(255) vs 浅色
# 墙纸(如237)"的色差(≈18), 才能既支持彩色卡片(绿卡间隔=卡片填充色)合并,
# 又不把浅色墙纸误当填充色而错并相邻两条消息。
_BG_TOL = 12

# 收发方向判定(帧内**自适应**气泡色): 固定气泡色在真实录屏里必然失配
# (实测: 墙纸≈(237,237,237)、发出气泡≈(148,234,109)、收到气泡为纯白, 而旧
# 常量写死的是浅绿(237,252,219)), 因此改为每帧就地标定, 不依赖任何绝对色值:
#   1) 墙纸色 = 画面左右各 _SIDE_EDGE 列的中位色(气泡左右都留边距, 边缘必是墙纸);
#   2) "彩色像素" = 与墙纸色差异够大 且 色度(chroma)够高 且 不太暗 → 气泡填充色
#      或头像色斑; "中性像素" = 无彩度 且 亮度超过帧内自适应白底下限(墙纸与本地
#      白底的中点) → 纯白气泡底。绝对阈值会被H.264/JPEG压缩噪声击穿, 必须自适应。
#   3) 定侧三级优先:
#      a. 自身行带(文字上下几px)彩色占比够 → 文字嵌在彩色气泡填充里, 按重心定侧;
#      b. 文本行正上方近距(_SIDE_BAR_MAX_GAP内)有彩色块 → 是本消息的语音条
#         (绿条在转文字块上方; 更远的彩色属于上一条消息的气泡, 会污染重心, 必
#         须排除), 按色块重心定侧;
#      c. 白泡: 自身行带±近邻列内的白块贴哪侧屏幕边 = 哪侧(左右边缘间隙对比)。
#   4) 三级判据都不足 → 返回""交几何回退。**不做**任何"帧级我方色反转"式弱推断:
#      合成视频的语音转写行直接画在墙纸上(无白泡无色块), 弱推断会抢在几何法
#      之前给出随机方向(实测其答案随压缩格式翻转)。
_SIDE_LOOKUP = 70           # 采样带: 文本行上方延伸高度(就近色块搜索上界)
_SIDE_DOWN = 4              # 采样带: 文本行下方延伸高度
_SIDE_EDGE = 8              # 墙纸估计: 画面左右各取多少列
_SIDE_WALL_TOL = 24         # 前景判据: 与墙纸色的合计通道差下限
_SIDE_CHROMA = 28           # 彩色判据: 通道极差(min~max)下限
_SIDE_DARK_MAX = 150        # 彩色判据: 通道最大值下限(排除暗色/字形像素)
_SIDE_NEUTRAL_MIN = 240     # 中性判据: 通道最小值下限(纯白气泡底)
_SIDE_OWN_PAD_Y = 6         # 自身行带: 文本行上下外扩(彩色气泡填充就贴着文字)
_SIDE_OWN_MIN_RATIO = 0.05  # 自身行带彩色占比下限: 达到即按重心定方向(最可信)
_SIDE_BAR_MAX_GAP = 20      # 语音条判据: 语音条底缘距文本行顶的最大间隙(再远就
                            # 是上一条消息的气泡, 会污染重心, 必须排除)
_SIDE_BAR_JOIN_GAP = 6      # 语音条判据: 块内行间彩色中断容忍(JPEG噪声)
_SIDE_BAR_MIN_PIX = 100     # 语音条判据: 色块彩色像素总数下限
_SIDE_MIN_NEUTRAL_RATIO = 0.02  # 中性像素占列窗面积下限: 达到才用白块定侧
_SIDE_NEUTRAL_SPAN = 40     # 白块统计: 文本行左右各取多少px(气泡白底紧贴文字)
_SIDE_NEUTRAL_EDGE_MARGIN = 0.08    # 白块左右边缘间隙差下限(占宽度比例):
                                    # 气泡白底总贴着自己那一侧的屏幕边, 对比
                                    # 白块到左右屏幕边的间隙即可定侧(比"白块
                                    # 重心 vs 帧中线"对满宽气泡稳健得多)。


def classify_side(bbox, roi_width: int, center_margin: int = 20,
                  chat_enabled: bool = True) -> str:
    """
    按气泡水平位置分类收发。

    微信/QQ等聊天界面中: 我方消息气泡靠右, 对方消息气泡靠左。
    依据"左右留白对比": 文本bbox距左边缘更近→左侧气泡→[收];
    距右边缘更近→右侧气泡→[发]; 居中(两侧留白接近)→不标记。

    返回: "发" / "收" / ""
    """
    if not chat_enabled or not bbox or len(bbox) < 4:
        return ""
    if roi_width <= 0:
        return ""
    x1, _, x2, _ = bbox
    left_gap = x1                       # 距左边缘距离
    right_gap = roi_width - x2          # 距右边缘距离
    m = center_margin
    if right_gap + m < left_gap:
        return "发"
    if left_gap + m < right_gap:
        return "收"
    return ""


def _wall_color(frame) -> np.ndarray:
    """估计画面墙纸色: 取左右各 _SIDE_EDGE 列的中位色(逐通道)。

    聊天气泡左右都留边距, 屏幕最左/最右几列在任何一行都落在墙纸上, 不会
    落在气泡或卡片上; 用中值抗JPEG噪声。
    """
    h, w = frame.shape[:2]
    e = max(1, min(_SIDE_EDGE, w // 2))
    band = np.concatenate([frame[:, :e].reshape(-1, 3),
                           frame[:, w - e:].reshape(-1, 3)], axis=0)
    return np.median(band.astype(np.int16), axis=0)


def _foreground_masks(region: np.ndarray, wall: np.ndarray,
                      white_min: float = _SIDE_NEUTRAL_MIN):
    """把区域像素分为"彩色前景"(气泡填充色/头像)与"中性前景"(纯白气泡底)。

    彩色 = 与墙纸差异够大 且 色度够高 且 不太暗; 中性 = 无彩度 且 足够亮。
    中性**不要求**与墙纸差异大: 白底下限(帧内自适应, 见 _pixel_side_on_frame
    的 c 判据)取墙纸与本地白底的中点, 本身已保证与墙纸分离; 若再要求与墙纸
    差异>=24, H.264把白泡压暗到240-243时(与墙纸差<24)白块证据会整个消失。

    white_min 可传帧内自适应的白底下限(见 _pixel_side_on_frame 的 c 判据):
    绝对阈值240恰卡在墙纸(237)与白泡(255)之间, H.264解码噪声会把白泡像素
    拉到240以下使白块证据消失(JPEG存档帧无此问题, 同一帧中性像素可差数百倍)。
    """
    reg = region.astype(np.int16)
    dw = (np.abs(reg[:, :, 0] - wall[0]) + np.abs(reg[:, :, 1] - wall[1])
          + np.abs(reg[:, :, 2] - wall[2]))
    mx = reg.max(axis=2)
    mn = reg.min(axis=2)
    colorful = ((dw >= _SIDE_WALL_TOL) & (mx >= _SIDE_DARK_MAX)
                & ((mx - mn) >= _SIDE_CHROMA))
    neutral = ((mx - mn) < _SIDE_CHROMA) & (mn >= white_min)
    return colorful, neutral


def pixel_classify_side(line: TrackedLine,
                        frame_provider: Callable[[int], Optional[object]]) -> str:
    """按**帧内自适应**气泡色判定收发(右=发 / 左=收)。

    适用场景:
      气泡几乎占据整行宽度、文字靠边放置的聊天界面, 此时文本bbox的左右留白
      无法反映气泡归属侧, 几何法(左右留白对比)失效, 必须看气泡填充色。

    做法(不依赖任何写死的绝对色值, 每帧就地标定):
      a. 自身行带彩色占比够 → 文字嵌在气泡填充色里(绿泡), 按重心定方向;
      b. 文本行正上方近距内有彩色块 → 本消息的语音条, 按色块重心定方向
         (更远的彩色是上一条消息的气泡, 不纳入);
      c. 白泡: 自身行带±近邻列内的白块贴哪侧屏幕边 = 哪侧(左右边缘间隙对比);
      判据不足 → 返回 ""(调用方回退几何法)。
      注意**不做**"帧级我方色反转"之类的弱推断: 合成视频的语音转写行直接画在
      墙纸上(无白泡无色块), 任何弱推断都会抢在几何法之前给出错误方向。

    帧坐标说明:
      bbox为ROI坐标, frame_provider返回的也是ROI裁剪帧, 两者直接对应。
    """
    if frame_provider is None or line is None:
        return ""
    bbox = _pos(line)
    if not bbox or len(bbox) < 4:
        return ""
    # 首次出现帧常常是"刚从ROI底边滚入"的残帧(行被裁半, 自身行带全是墙纸),
    # 像素证据不足。首帧判据不足时回退尝试后续观测帧(行完整可见后判定才可信),
    # 仍无证据才交回几何法。
    candidates = [(line.first_seen_frame, bbox)]
    if line.last_seen_frame != line.first_seen_frame:
        candidates.append((line.last_seen_frame, line.bbox))
    obs = list(getattr(line, "observations", None) or [])
    for o in reversed(obs):
        candidates.append((o.frame_id, o.bbox))
    seen_fids = set()
    for fid, bb in candidates[:8]:
        if fid in seen_fids or not bb or len(bb) < 4:
            continue
        seen_fids.add(fid)
        side = _pixel_side_on_frame(line, fid, bb, frame_provider)
        if side:
            return side
    return ""


def _pixel_side_on_frame(line: TrackedLine, fid: int, bbox,
                         frame_provider) -> str:
    """在指定帧上按自适应气泡色判定收发(单帧, 无回退), 无证据返回 ""。"""
    frame = frame_provider(fid)
    if frame is None or frame.size == 0:
        return ""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    wall = _wall_color(frame)

    def _decide(col) -> str:
        xs = np.nonzero(col)[1]
        if xs.size == 0:
            return ""
        return "发" if xs.mean() > w / 2 else "收"

    # a. 自身行带: 文字嵌在彩色气泡填充里(绿泡内的文本行) / 或嵌在白泡里(c用)
    oytop = max(0, y1 - _SIDE_OWN_PAD_Y)
    oybot = min(h, y2 + _SIDE_OWN_PAD_Y)
    if oybot > oytop:
        own, _ = _foreground_masks(frame[oytop:oybot], wall)
        if own.sum() >= _SIDE_OWN_MIN_RATIO * own.size:
            return _decide(own)
        # c. 白泡: 白块贴哪侧屏幕边 = 哪侧。对比白块左右边缘到屏幕边的间隙,
        #    间隙差够大才表态(2%/98%分位抗零散白点)。白底下限按帧自适应:
        #    取列窗内min通道的90分位≈本地白底亮度, 与墙纸亮度取中点 —— 绝对
        #    阈值240会被H.264解码噪声击穿(白泡像素跌到240以下), 中点阈值对
        #    JPEG/H.264两种压缩都稳。
        cx1 = max(0, x1 - _SIDE_NEUTRAL_SPAN)
        cx2 = min(w, x2 + _SIDE_NEUTRAL_SPAN)
        win = max(1, (cx2 - cx1) * (oybot - oytop))
        if cx2 > cx1:
            win_min = frame[oytop:oybot, cx1:cx2].min(axis=2).astype(np.int16)
            local_white = float(np.percentile(win_min, 90))
            wall_lvl = float(np.mean(wall))
            thr = wall_lvl + max(6.0, (local_white - wall_lvl) * 0.5)
            _, neutral_own = _foreground_masks(frame[oytop:oybot], wall,
                                               white_min=thr)
            if int(neutral_own.sum()) >= _SIDE_MIN_NEUTRAL_RATIO * win:
                nxs = np.nonzero(neutral_own[:, cx1:cx2])[1] + cx1
                if nxs.size:
                    gap_l = float(np.percentile(nxs, 2))
                    gap_r = w - float(np.percentile(nxs, 98))
                    if (abs(gap_l - gap_r) >= _SIDE_NEUTRAL_EDGE_MARGIN * w):
                        return "收" if gap_l < gap_r else "发"

    # b. 就近色块: 文本行正上方近距内的彩色块 = 本消息的语音条(转文字块上方);
    #    更远的彩色属于上一条消息的气泡, 会污染重心, 不纳入。
    ytop = max(0, y1 - _SIDE_LOOKUP)
    ybot = min(h, y2 + _SIDE_DOWN)
    if ybot > ytop:
        colorful, _ = _foreground_masks(frame[ytop:ybot], wall)
        counts = colorful.sum(axis=1)
        # 从文本行顶向上找第一段彩色行: 允许 _SIDE_BAR_MAX_GAP 的空隙起步
        # 索引必须钳在 counts 长度内, 因为 slice 实际高度 ybot-ytop 可能远
        # 小于 _SIDE_LOOKUP(当 y1 靠近画面上沿时); Agent 3 写此段时未钳位,
        # 在小 slice(如 smoke 合成视频 5/34px) 上直接 IndexError。
        # / Clamp i into slice bounds, since the actual slice height ybot-ytop
        # may be much smaller than _SIDE_LOOKUP (when y1 is near the frame top).
        start = None
        gap = 0
        # L238 循环起点原来直接 y1-ytop-1, 但这可能 >= counts.size
        lo = min(y1 - ytop - 1, counts.size - 1)
        if lo < 0:
            lo = 0
        for i in range(lo, -1, -1):
            if counts[i] > 0:
                start = i
                break
            gap += 1
            if gap > _SIDE_BAR_MAX_GAP:
                break
        if start is not None:
            total = 0
            miss = 0
            end = start
            # L249 正向遍历也可能越 count.size 上界(因为 start 是彩色带底,
            # 带顶可能超出 slice 起点), 同样钳到 counts.size。
            for i in range(start - 1, -1, -1):
                if counts[i] > 0:
                    total += int(counts[i])
                    miss = 0
                    end = i
                else:
                    miss += 1
                    if miss > _SIDE_BAR_JOIN_GAP:
                        break
            if total >= _SIDE_BAR_MIN_PIX:
                return _decide(colorful[end:start + 1])
    return ""


def classify_side_line(line: TrackedLine, roi_width: int,
                       center_margin: int = 20, chat_enabled: bool = True,
                       frame_provider: Callable[[int], Optional[object]] = None) -> str:
    """收发判定(像素优先, 几何回退)。

    1. 有帧提供器且能采到气泡填充色 → 按颜色判定(绿=发/白=收);
    2. 无像素证据(非全宽气泡布局/截图内容/无帧) → 回退 classify_side 的
       左右留白几何法。

    返回: "发" / "收" / ""
    """
    if not chat_enabled or line is None:
        return ""
    s = pixel_classify_side(line, frame_provider)
    if s:
        return s
    return classify_side(_pos(line), roi_width, center_margin, True)


# ---------------------------------------------------------------------------
# 居中系统提示过滤 / Centered system-prompt filtering
# ---------------------------------------------------------------------------
# 真实聊天录屏里消息之间不可避免会穿插**系统提示**: 居中、小号、灰色、无气泡,
# 文本形如 "10:23"、"对方撤回了一条消息"、"以下为新消息"、
# "你已添加对方为好友，现在可以开始聊天"。这类行不是聊天消息, 若被当作消息输出
# 会: (a) 污染收发方向判定(居中文本几何法返回空方向, 却仍占一个输出行);
# (b) 占用滚动去重的 seen 索引, 干扰后续正文; (c) 让输出行数虚高。
#
# 判据(保守, 宁可不滤也不错滤):
#   1. 聊天界面开启(否则无收发方向可言, 不做过滤);
#   2. 水平居中(左右留白接近)且几何法判不出方向(classify_side == "", 即不贴左/右);
#   3. 文本命中已知系统模式, 或文本足够短(<= _SYS_MAX_LEN 字)。
# 正常聊天消息都在左/右气泡里(有明确方向), 因此不会被误杀; HH:MM 时间分隔条由
# STAMP_RE 单独处理(消息时间关联需要它们), 不在此丢弃。
# / Real recordings inevitably interleave centered system prompts (small, gray, no
# bubble). They are not chat messages: emitting them would pollute side detection
# and scroll de-duplication and inflate the output line count. Criteria are
# conservative: horizontally centered AND no side by classify_side AND (known
# system pattern OR short text). Normal messages sit in left/right bubbles and
# thus keep an explicit side, so they are never dropped; HH:MM dividers are
# handled by STAMP_RE (needed for message timestamps) and never dropped here.
_SYS_CENTER_TOL = 0.08      # 居中容差(ROI宽度比例, 与时间分隔条同口径)
_SYS_MAX_LEN = 20           # "无方向 + 居中"时的短文本上限(字符数)
# 已知系统提示模式(保守列举, 只收录明确的系统提示措辞)
SYSTEM_PROMPT_RE = re.compile(
    r"撤回"                     # 对方撤回了一条消息 / 你撤回了一条消息
    r"|以下为新消息"            # 以下为新消息
    r"|你已添加"                # 你已添加对方为好友，现在可以开始聊天
    r"|已添加.*好友"
    r"|以上是打招呼"
    r"|开启了朋友验证"
    r"|对方正在输入"
)


def _is_horizontally_centered(bbox, roi_width: int,
                              tol_ratio: float = _SYS_CENTER_TOL) -> bool:
    """文本行是否水平居中(居中容差按 ROI 宽度比例)。roi_width<=0 无从判断 → False。"""
    if not bbox or len(bbox) < 4 or roi_width <= 0:
        return False
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2.0
    return abs(cx - roi_width / 2.0) <= tol_ratio * roi_width


def is_centered_system_prompt(text: str, bbox, roi_width: int,
                              center_margin: int = 20,
                              chat_enabled: bool = True) -> bool:
    """该文本行是否为"居中的系统提示"(不应作为聊天消息输出)。

    判据(全部满足):
      1. 聊天界面开启;
      2. 水平居中且几何法判不出收发方向(classify_side=="", 即不贴左/右);
      3. 文本命中已知系统模式(含"撤回"/"以下为新消息"/"你已添加"等), 或足够短。
    HH:MM 时间分隔条由 STAMP_RE 处理, 这里一律返回 False(不得误丢时间条)。
    """
    if not chat_enabled or roi_width <= 0:
        return False
    t = (text or "").strip()
    if not t or STAMP_RE.fullmatch(t):
        return False
    if classify_side(bbox, roi_width, center_margin, True):
        return False            # 有明确收发方向 → 正常消息, 不滤
    if not _is_horizontally_centered(bbox, roi_width):
        return False
    return bool(SYSTEM_PROMPT_RE.search(t)) or len(t) <= _SYS_MAX_LEN


def drop_system_prompts(lines: List[TrackedLine], roi_width: int,
                        center_margin: int = 20, chat_enabled: bool = True
                        ) -> Tuple[List[TrackedLine], int]:
    """从跟踪行中剔除居中的系统提示文本 / drop centered system prompts.

    返回: (保留的跟踪行, 被剔除数)。无系统提示时原样返回, 零副作用。
    """
    if not lines:
        return list(lines), 0
    kept: List[TrackedLine] = []
    dropped = 0
    for line in lines:
        if is_centered_system_prompt(line.text, _pos(line), roi_width,
                                     center_margin, chat_enabled):
            dropped += 1
            continue
        kept.append(line)
    return kept, dropped


def ends_sentence(text: str) -> bool:
    """以终止标点结尾 → 句子完结"""
    return bool(_TERMINAL_PUNCT_RE.search((text or "").strip()))


def _bubble_anchor(side: str, bbox) -> Tuple[str, int]:
    """气泡对齐锚点: 收(左气泡)锚左边缘x1, 发(右气泡)锚右边缘x2。"""
    x1, _, x2, _ = bbox
    if side == "收":
        return "left", x1
    if side == "发":
        return "right", x2
    return "", 0


def _pos(line: TrackedLine) -> Tuple[int, int, int, int]:
    """排序/几何用bbox: 优先首次出现位置(保持全屏捕获开场的顺序), 无则回退最新位置。"""
    return line.first_bbox or line.bbox


def _merged_line(buf: List[TrackedLine], text: str) -> TrackedLine:
    """用一组行构造合并后的 TrackedLine(bbox取并集)"""
    x1 = min(_pos(l)[0] for l in buf)
    y1 = min(_pos(l)[1] for l in buf)
    x2 = max(_pos(l)[2] for l in buf)
    y2 = max(_pos(l)[3] for l in buf)
    total_len = sum(len(l.text or "") for l in buf) or 1
    conf = sum((l.confidence or 0.0) * len(l.text or "") for l in buf) / total_len
    first, last = buf[0], buf[-1]
    merged_bbox = (x1, y1, x2, y2)
    return TrackedLine(
        line_id=f"merged:{first.line_id}",
        text=text,
        bbox=merged_bbox,
        first_bbox=merged_bbox,
        confidence=conf,
        first_seen_frame=first.first_seen_frame,
        last_seen_frame=last.last_seen_frame,
        first_timestamp=first.first_timestamp,
        last_timestamp=last.last_timestamp,
        observations=[TextLine(text=text, bbox=merged_bbox, confidence=conf,
                               frame_id=first.first_seen_frame,
                               timestamp=first.first_timestamp)],
        is_new=first.is_new,
    )


def _join_with_overlap(prev: str, nxt: str) -> str:
    """拼接两行, 消除换行处被拆词产生的重复字符。

    例: 换行处拆词, 上行末尾"意" + 下行开头"意向" → "意向"(去重一个"意")。
    """
    if not prev or not nxt:
        return prev + nxt
    best = 0
    limit = min(len(prev), len(nxt), _MAX_OVERLAP)
    for k in range(1, limit + 1):
        if prev[-k:] == nxt[:k]:
            best = k
    return prev + nxt[best:]


def _merge_row_fragments(lines: List[TrackedLine]) -> List[TrackedLine]:
    """预合并同一视觉行的左右水平碎片。

    OCR 可能把同一视觉行切成左右两个 bbox(如 "3.您的支付宝芝麻信用分多少分"
    与 "(请您打"), 右半段因靠右会被误判为[发]。这里按"垂直高度重叠比例高 +
    水平相邻且间隙小"把同一视觉行拼回一行, 方向以左侧段为准。
    """
    if len(lines) < 2:
        return list(lines)
    ordered = sorted(lines,
                     key=lambda l: (l.first_timestamp, _pos(l)[1], _pos(l)[0]))
    out: List[TrackedLine] = []
    i, n = 0, len(ordered)
    while i < n:
        cur = ordered[i]
        i += 1
        while i < n:
            nxt = ordered[i]
            if abs(cur.first_timestamp - nxt.first_timestamp) > 0.5:
                break
            # 时间戳行(如"20:15")是聊天时间标记, 不是消息内容的一部分, 绝不
            # 并入文本碎片(否则会拼出"你好呀哥哥20:15"这类污染内容)。
            if (STAMP_RE.fullmatch((nxt.text or "").strip())
                    or STAMP_RE.fullmatch((cur.text or "").strip())):
                break
            pa, pb = _pos(cur), _pos(nxt)
            x1a, y1a, x2a, y2a = pa
            x1b, y1b, x2b, y2b = pb
            h_a, h_b = y2a - y1a, y2b - y1b
            ov = min(y2a, y2b) - max(y1a, y1b)
            # 同一视觉行: 垂直高度重叠比例高
            if h_a <= 0 or h_b <= 0 or ov < _ROW_FRAGMENT_OVERLAP * min(h_a, h_b):
                break
            # nxt 需在 cur 右侧且水平间隙小
            # 右碎片左缘不得大幅侵入左碎片内部: 滚动回看帧中两条不同消息的
            # 跟踪bbox可能垂直重叠, 但其左缘都从气泡左边界开始, 会越过该
            # 上限而被拒绝; 真同行碎片(如 "…芝麻信用分多少分"+"(请您打")
            # 右碎片左缘紧贴左碎片右缘, 不受影响。
            if (x1b - x2a > _ROW_FRAGMENT_GAP_MAX or x2b <= x2a
                    or x2a - x1b > _ROW_FRAGMENT_LEFT_MAX):
                break
            cur = _merged_line(
                [cur, nxt],
                _join_with_overlap((cur.text or "").strip(),
                                   (nxt.text or "").strip()))
            i += 1
        out.append(cur)
    return out


def _local_fill(frame, bbox, w: int, h: int) -> Optional[Tuple[int, int, int]]:
    """估计一条文本行周围(气泡/卡片内边距)的**填充色** = 该行的局部背景色。

    转账/红包等**非白色**卡片(如绿色卡片)的背景是纯色块。旧口径"亮即气泡底"
    会把卡片两行之间的实心色间隔误当成墙纸, 使同一张卡片的多行被拆开。这里从
    bbox 外扩 2~4px 的细环上取众数色: 白泡得白、绿卡得绿, 即该行的填充色。
    """
    x1, y1, x2, y2 = bbox
    patches = []
    for off in (2, 3, 4):
        for y in (y1 - off, y2 + off):
            if 0 <= y < h:
                sx1, sx2 = max(0, x1), min(w - 1, x2)
                if sx2 > sx1:
                    patches.append(frame[y, sx1:sx2 + 1])
        for x in (x1 - off, x2 + off):
            if 0 <= x < w:
                sy1, sy2 = max(0, y1), min(h - 1, y2)
                if sy2 > sy1:
                    patches.append(frame[sy1:sy2 + 1, x])
    if not patches:
        return None
    band = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    q = band.astype(np.int32) >> 3              # 量化到32级, 抗JPEG噪声
    keys = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    uniq, cnt = np.unique(keys, return_counts=True)
    sel = band[keys == uniq[cnt.argmax()]]
    return tuple(int(v) for v in sel.mean(axis=0))


def _color_close(pixel, color: Tuple[int, int, int], tol: int) -> bool:
    """两个BGR像素是否接近(各通道差 <= tol)。"""
    return (abs(int(pixel[0]) - color[0]) <= tol
            and abs(int(pixel[1]) - color[1]) <= tol
            and abs(int(pixel[2]) - color[2]) <= tol)


def _edge_color(frame, y_lo: int, y_hi: int) -> Optional[Tuple[int, int, int]]:
    """取间隙所在行的**画面左右边缘**颜色 = 墙纸色估计。

    聊天气泡左右都留边距, 因此屏幕最左右几列在该行一定是墙纸, 不会落在气泡
    或卡片上。用逐通道中值抗噪。
    """
    h, w = frame.shape[:2]
    y_lo, y_hi = max(0, y_lo), min(h - 1, y_hi)
    if y_hi < y_lo:
        return None
    cols = []
    for x in (1, 2, 3, w - 4, w - 3, w - 2):
        if 0 <= x < w:
            cols.append(frame[y_lo:y_hi + 1, x])
    if not cols:
        return None
    band = np.concatenate([c.reshape(-1, 3) for c in cols], axis=0)
    return tuple(int(v) for v in np.median(band, axis=0))


def _pixel_same_bubble(a: TrackedLine, b: TrackedLine,
                       frame_provider: Callable[[int], Optional[object]]) -> bool:
    """像素验证两条文本行之间的竖向间隙是否为同一气泡/卡片的背景。

    旧实现用"整条间隙的白底占比>=阈值"判定, 但两个气泡之间的墙纸条带往往
    只有2-4px, 而间隙还包含上下两个气泡各约十几像素的白底padding, 占比被
    padding稀释(如36/39=0.92)导致误判为同一气泡。这里逐行扫描: 间隙内只要
    存在连续 _WALLPAPER_RUN 个背景(非气泡填充)像素行, 就证明两行分属两个气泡。

    背景判定(满足任一即视为"同一气泡背景", 不累计墙纸游程):
      1) 亮白: min通道 >= white_min(按两条文本行就地定标, 见 _WHITE_ADAPT_MARGIN);
         —— 覆盖白气泡/浅色背景里的普通换行与大间距段落。
      2) 与该行**填充色**一致, 且**不是墙纸色**: 覆盖转账/红包等**非白色卡片**
         的实心色间隔(如绿色卡片), 其间隔是卡片填充色而非墙纸, 故应并入同一
         条消息。要求"非墙纸"是为了不把"填充色恰好等于墙纸色"的场景(如测试里
         整屏同色)误并: 此时间隔实为墙纸, 两行分属不同气泡。
    """
    frame = frame_provider(b.first_seen_frame)
    if frame is None or frame.size == 0:
        return False
    h, w = frame.shape[:2]
    pa, pb = _pos(a), _pos(b)
    x1a, y1a, x2a, y2a = pa
    x1b, y1b, x2b, y2b = pb
    y_lo, y_hi = y2a + 1, y1b - 1
    if y_hi <= y_lo:
        return False
    lo_x = max(x1a, x1b)          # 两条文本行公共水平范围
    hi_x = min(x2a, x2b)
    if hi_x <= lo_x:
        return False
    white_min = _WHITE_MIN
    fills: List[Tuple[int, int, int]] = []
    for x1, y1, x2, y2 in (pa, pb):
        sx1, sy1 = max(0, x1), max(0, y1)
        sx2, sy2 = min(w - 1, x2), min(h - 1, y2)
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        band = frame[sy1:sy2 + 1, sx1:sx2 + 1]
        lvl = float(np.percentile(band.min(axis=2), _WHITE_ADAPT_PCT))
        white_min = max(white_min, lvl - _WHITE_ADAPT_MARGIN)
        fill = _local_fill(frame, (x1, y1, x2, y2), w, h)
        if fill is not None:
            fills.append(fill)
    wallpaper = _edge_color(frame, y_lo, y_hi)
    # 采样列: 左内侧/中点/右内侧, 任一列见到墙纸条带即可判定分离(气泡小三角
    # 只影响单个边缘列, 多列采样避免被其遮挡)。
    xs: List[int] = []
    for c in (min(x1a, x1b) + 20, (lo_x + hi_x) // 2, max(x2a, x2b) - 20):
        if lo_x <= c <= hi_x:
            xs.append(c)
    if not xs:
        xs = [(lo_x + hi_x) // 2]
    for x in xs:
        if x < 0 or x >= w or y_lo < 0 or y_hi >= h:
            continue
        non_bg_run = 0
        for y in range(y_lo, y_hi + 1):
            px = frame[y, x]
            is_bg = int(px.min()) >= white_min
            if (not is_bg and fills
                    and any(_color_close(px, f, _BG_TOL) for f in fills)):
                # 卡片/气泡填充色, 且不同于墙纸 → 视为同一气泡背景
                is_bg = not (wallpaper is not None
                             and _color_close(px, wallpaper, _BG_TOL))
            if is_bg:
                non_bg_run = 0
            else:
                non_bg_run += 1
                if non_bg_run >= _WALLPAPER_RUN:
                    return False
    return True


def _same_bubble(a: TrackedLine, b: TrackedLine,
                 classify_side: Callable[[TrackedLine], str],
                 frame_provider: Callable[[int], Optional[object]] = None) -> bool:
    """两行是否位于同一个聊天气泡"""
    a_side = classify_side(a)
    b_side = classify_side(b)
    pa, pb = _pos(a), _pos(b)
    x1a, y1a, x2a, y2a = pa
    x1b, y1b, x2b, y2b = pb
    gap = y1b - y2a
    if gap < _BUBBLE_GAP_OVERLAP_MAX:
        return False
    if gap > _BUBBLE_GAP_RELAXED:
        return False
    # 水平无重叠 → 不同气泡(如时间戳、居中系统提示)
    if min(x2a, x2b) - max(x1a, x1b) <= 0:
        return False
    # 收发方向均明确时要求一致
    if a_side and b_side and a_side != b_side:
        return False
    anchor_a, ax_a = _bubble_anchor(a_side, pa)
    anchor_b, ax_b = _bubble_anchor(b_side, pb)
    if anchor_a and anchor_b:
        if not (anchor_a == anchor_b and abs(ax_a - ax_b) <= _BUBBLE_ALIGN_TOL):
            return False
    # 方向未知(居中/未配置ROI宽度): 任一边缘对齐即视为同气泡
    elif not (abs(x1a - x1b) <= _BUBBLE_ALIGN_TOL
              or abs(x2a - x2b) <= _BUBBLE_ALIGN_TOL):
        return False
    # 大间距(段内空行)需要像素验证为同一气泡才合并
    if gap > _BUBBLE_GAP_MAX:
        if frame_provider is None or not _pixel_same_bubble(a, b, frame_provider):
            return False
    return True


def _join_lines(buf: List[TrackedLine], bubble_width: float) -> TrackedLine:
    """把一个气泡内的所有行合并为一条消息。

    满行且句子未完结 → 界面换行截断 → 直接粘接(消除拆词重叠);
    短行 / 大间距 → 发送方自己的换行 → 保留为消息内换行(字面 \\n)。
    """
    if len(buf) == 1:
        return buf[0]
    parts: List[str] = []
    for i, line in enumerate(buf):
        t = (line.text or "").strip()
        if i == 0:
            parts.append(t)
            continue
        prev = buf[i - 1]
        pa, pb = _pos(prev), _pos(line)
        gap = pb[1] - pa[3]
        prev_full = (pa[2] - pa[0]) >= _FULL_LINE_RATIO * bubble_width
        if gap <= _BUBBLE_GAP_MAX and prev_full and not ends_sentence(prev.text):
            parts[-1] = _join_with_overlap(parts[-1], t)
        else:
            parts.append(t)
    text = "\\n".join(parts)
    return _merged_line(buf, text)


def merge_wrapped_lines(tracked_lines: List[TrackedLine],
                        classify_side,
                        frame_provider: Callable[[int], Optional[object]] = None
                        ) -> Tuple[List[TrackedLine], int]:
    """合并同一聊天气泡内的换行碎片为完整消息。

    时间戳行原样保留(后续组装仍需要它们做消息时间关联)。

    参数:
        tracked_lines: 所有跟踪的文本行
        classify_side: 收发分类回调(TrackedLine) -> "发"/"收"/""。
                       像素判定(pixel_classify_side/classify_side_line)依赖
                       frame_provider, 回调需自行持有; 仅几何法时传
                       lambda line: classify_side(_pos(line), ...) 即可。
        frame_provider: 可选, 按帧序号(1基)取ROI帧的回调, 用于像素验证大间距
                        是否同一气泡; None 时大间距保守不合并。

    返回:
        (合并后的跟踪行列表, 合并处数)
    """
    if len(tracked_lines) < 2:
        return list(tracked_lines), 0
    lines = _merge_row_fragments(list(tracked_lines))
    ordered = sorted(lines, key=lambda l: (l.first_timestamp, _pos(l)[1]))

    # 按"同气泡"几何切分分组; 时间戳/空行作为分隔, 不参与合并
    # 分组链要求 first_timestamp 相同(同一帧完整可见): 不同时刻的行其首次位置
    # 来自不同画面, 几何比较无意义(滚动会撕裂同一气泡的行, 宁可不合也不错合)。
    chunks = []          # ("group", [lines]) | ("stamp", line)
    cur: List[TrackedLine] = []
    for line in ordered:
        text = (line.text or "").strip()
        if not text or STAMP_RE.fullmatch(text):
            if cur:
                chunks.append(("group", cur))
                cur = []
            chunks.append(("stamp", line))
            continue
        if (cur and abs(cur[-1].first_timestamp - line.first_timestamp) < 1e-6
                and _same_bubble(cur[-1], line, classify_side, frame_provider)):
            cur.append(line)
        else:
            if cur:
                chunks.append(("group", cur))
            cur = [line]
    if cur:
        chunks.append(("group", cur))

    out: List[TrackedLine] = []
    merged_count = 0
    for kind, payload in chunks:
        if kind == "stamp":
            out.append(payload)
            continue
        group = payload
        if len(group) < 2:
            out.append(group[0])
            continue
        pos_list = [_pos(l) for l in group]
        x1s = [p[0] for p in pos_list]
        x2s = [p[2] for p in pos_list]
        bubble_width = max(x2s) - min(x1s)
        if bubble_width <= 0:
            out.extend(group)
            continue
        # 同一气泡 = 同一条消息: 整组合并(行间以字面\n连接)
        out.append(_join_lines(group, bubble_width))
        merged_count += len(group) - 1
    return out, merged_count


def _is_subrow_of(frag_text: str, parent_text: str) -> bool:
    """片段是否为"多行父消息"中某一子行的子串。

    仅对**多行**父消息(气泡内换行 / 转账卡片多行文字, 合并时以字面 \\n 连接)
    生效。这类元素在滚动/被 ROI 边缘遮挡时可能只露出一行, 被 OCR 单独读成一
    条文本行; 若该文本恰是父消息某子行的子串, 说明它就是同一元素的一部分
    (实测: 转账卡片 "¥200.00已收款\\n微信转账" 在卡片滚到 ROI 边缘时被单独读成
    一条 "¥200.00" 输出行)。

    单行父消息**不做**此判断: 此时"短消息恰好是长消息的子串"更可能是两条独立
    消息(如先发"好的，我提前把文档整理一下"、后发"好的")。

    / A fragment is a sub-row only for multi-row parents (joined by literal \\n):
    when such an element is partially occluded, OCR may read one of its rows as a
    standalone line. Single-row parents are excluded to avoid dropping a genuine
    short message that happens to be a substring of a longer one.
    """
    if "\\n" not in (parent_text or ""):
        return False
    return any(frag_text and frag_text in seg
               for seg in parent_text.split("\\n"))


def absorb_wrap_tail_leftovers(lines: List[TrackedLine], classify_side,
                               ) -> Tuple[List[TrackedLine], int]:
    """把"气泡换行尾片段/多行元素子行"归属回它的父消息(实现上是丢弃该片段)。

    场景(真实录屏常见):
      1) 长消息在气泡内被界面自动换行成多行, 之后滚动把整条消息上移, **父行滚出
         画面而只剩最后一行残留在顶部边缘**。该尾行会被 OCR 重新识别成一条独立的
         文本行; 若直接输出, 同一条消息的尾巴就被重复输出一次(实测: GT
         "行，我下午发，顺便把那两个小的兼容性问题也一起写上" 渲染为两行
         "…兼容性问题也一" / "起写上", 滚动后只剩 "起写上" 可见 → 输出多出
         一条 "+96s [发] ：起写上")。
      2) 多行**卡片**(如转账卡片的金额/状态/"微信转账"三行)滚到 ROI 边缘被遮挡,
         只露出一行时被 OCR 单独读成一条行(如孤立的 "¥200.00"), 与同卡片已在
         别处输出的完整行重复。

    判据(全部满足才吸收, 宁可不吸也不错吸):
      - 片段文本是另一条消息文本的**真后缀**(父消息以它结尾), 或(仅当父消息是
        多行元素时)是父消息某一子行的**子串**, 且明显更短
        (长度差 >= _TAIL_MIN_DIFF)、片段自身 >= _TAIL_MIN_LEN 字;
      - 片段首次出现**不早于**父消息。时间序是关键防误杀判据: 若短消息先出现、
        长消息后出现(先发"晚点说"、后发"行，那晚点说"), 后缀关系成立但时序相反
        → 不吸收;
      - 收发方向一致(任一侧方向未知则不吸收, 保守)。

    参数:
        lines: 已按气泡合并后的跟踪行
        classify_side: 收发分类回调(TrackedLine) -> "发"/"收"/""
    返回:
        (吸收后的跟踪行列表, 被吸收的片段数)
    """
    if len(lines) < 2:
        return list(lines), 0
    ordered = sorted(lines, key=lambda l: (l.first_timestamp, _pos(l)[1]))
    dropped_ids = set()
    for frag in ordered:
        t = (frag.text or "").strip()
        if len(t) < _TAIL_MIN_LEN or id(frag) in dropped_ids:
            continue
        frag_side = classify_side(frag)
        if not frag_side:
            continue
        for parent in ordered:
            if parent is frag or id(parent) in dropped_ids:
                continue
            pt = (parent.text or "").strip()
            if len(pt) - len(t) < _TAIL_MIN_DIFF:
                continue
            if not (pt.endswith(t) or _is_subrow_of(t, pt)):
                continue
            if frag.first_timestamp < parent.first_timestamp:
                continue
            if classify_side(parent) != frag_side:
                continue
            dropped_ids.add(id(frag))
            break
    if not dropped_ids:
        return list(lines), 0
    return [l for l in lines if id(l) not in dropped_ids], len(dropped_ids)
