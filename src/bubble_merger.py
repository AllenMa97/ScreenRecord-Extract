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
_WHITE_MIN = 225            # 气泡白底判定: 三通道最小值
_WALLPAPER_RUN = 2          # 连续非白(墙纸)行数下限, 证明两行间存在气泡间隙

# 气泡填充色判定收发(像素方案): 部分聊天App的气泡几乎占据整行宽度, 文字靠
# 边放置(如"芝麻信用751"文字靠左但其气泡横贯屏幕), 仅靠文本bbox的左右留白
# 对比无法区分收发, 必须采样气泡填充色。该App中:
#   浅绿气泡(237,252,219) = 我方发出; 纯白气泡(255,255,255) = 对方发出;
#   墙纸为深绿(约136,179,132), 与浅绿/纯白色差明显, 不会误采。
_BUBBLE_SEND_COLOR = (237, 252, 219)  # 发: 浅绿气泡
_BUBBLE_SEND_TOL = 14                 # 色差容差(JPEG压缩余量)
_BUBBLE_RECV_WHITE_MIN = 240          # 收: 纯白气泡(三通道均>=此值)
_BUBBLE_SAMPLE_PAD_X = 40             # 采样区水平外扩(覆盖文字bbox外气泡内padding)
_BUBBLE_SAMPLE_PAD_Y = 4              # 采样区垂直外扩
_BUBBLE_SAMPLE_STEP = 3               # 采样步长(px), 兼顾速度与覆盖
_BUBBLE_MIN_FILL_RATIO = 0.25         # 气泡色像素占总采样比例下限(不足则无证据→回退)
_BUBBLE_MIN_COLOR_RATIO = 0.60        # 绿/白占气泡色像素比例阈值


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


def pixel_classify_side(line: TrackedLine,
                        frame_provider: Callable[[int], Optional[object]]) -> str:
    """按气泡填充色判定收发: 绿泡=发, 白泡=收。

    适用场景:
      气泡几乎占据整行宽度、文字靠边放置的聊天界面(如"芝麻信用751"文字靠
      左但其气泡横贯屏幕), 此时文本bbox的左右留白无法反映气泡归属侧, 几何
      法(左右留白对比)失效, 必须采样气泡填充色。

    做法:
      在文本bbox周围(外扩一个padding)隔点采样, 统计浅绿/纯白像素占比:
        - 绿色占比>=阈值 → "发"
        - 白色占比>=阈值 → "收"
        - 无足够气泡色证据(采样区多为墙纸/文字/截图内容) → ""(调用方回退几何法)

    帧坐标说明:
      bbox为ROI坐标, frame_provider返回的也是ROI裁剪帧, 两者直接对应。
    """
    if frame_provider is None or line is None:
        return ""
    bbox = _pos(line)
    if not bbox or len(bbox) < 4:
        return ""
    frame = frame_provider(line.first_seen_frame)
    if frame is None or frame.size == 0:
        return ""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    sx1 = max(0, x1 - _BUBBLE_SAMPLE_PAD_X)
    sy1 = max(0, y1 - _BUBBLE_SAMPLE_PAD_Y)
    sx2 = min(w - 1, x2 + _BUBBLE_SAMPLE_PAD_X)
    sy2 = min(h - 1, y2 + _BUBBLE_SAMPLE_PAD_Y)
    if sx2 <= sx1 or sy2 <= sy1:
        return ""
    sr, sg, sb = _BUBBLE_SEND_COLOR
    tol = _BUBBLE_SEND_TOL
    wmin = _BUBBLE_RECV_WHITE_MIN
    green = white = total = 0
    for y in range(sy1, sy2 + 1, _BUBBLE_SAMPLE_STEP):
        row = frame[y]
        for x in range(sx1, sx2 + 1, _BUBBLE_SAMPLE_STEP):
            b, g, r = int(row[x][0]), int(row[x][1]), int(row[x][2])
            total += 1
            if abs(r - sr) <= tol and abs(g - sg) <= tol and abs(b - sb) <= tol:
                green += 1
            elif r >= wmin and g >= wmin and b >= wmin:
                white += 1
    fill = green + white
    if fill < _BUBBLE_MIN_FILL_RATIO * total:
        return ""
    if green >= _BUBBLE_MIN_COLOR_RATIO * fill:
        return "发"
    if white >= _BUBBLE_MIN_COLOR_RATIO * fill:
        return "收"
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


def _pixel_same_bubble(a: TrackedLine, b: TrackedLine,
                       frame_provider: Callable[[int], Optional[object]]) -> bool:
    """像素验证两条文本行之间的竖向间隙是否为同一气泡的白底。

    旧实现用"整条间隙的白底占比>=阈值"判定, 但两个气泡之间的墙纸条带往往
    只有2-4px, 而间隙还包含上下两个气泡各约十几像素的白底padding, 占比被
    padding稀释(如36/39=0.92)导致误判为同一气泡。这里改为逐行扫描: 间隙内
    只要存在连续 _WALLPAPER_RUN 个非白(墙纸)像素行, 就证明两行分属两个气泡。
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
        non_white_run = 0
        for y in range(y_lo, y_hi + 1):
            if int(frame[y, x].min()) >= _WHITE_MIN:
                non_white_run = 0
            else:
                non_white_run += 1
                if non_white_run >= _WALLPAPER_RUN:
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
