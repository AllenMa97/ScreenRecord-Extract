"""
截图消息过滤模块 / Screenshot Message Filter

聊天中收发双方发送的"其他软件截图"(如借呗/网商银行/网商贷等 App 页面截图)
内部文字会被 OCR 识别成聊天行, 污染聊天正文。本模块在几何合并之后、LLM 增强
之前运行: 用几何特征把截图内部的文字行聚成"截图簇", 判定后整簇折叠为一条
"[图片消息]" 占位, 截图文字不再进入聊天正文。

Screenshots sent inside the chat (e.g. loan-app UI screenshots) get their inner
text OCR'd as chat lines, polluting the transcript. This module runs after
geometric merging and before LLM enhancement: it groups screenshot text rows
into clusters by geometry, and collapses each cluster into a single "[图片消息]"
placeholder so the raw screenshot text never reaches the transcript.

判定依据 / Detection basis (ROI 宽 = 890 时实测):
  - 截图是一张图: 其内部文字行构成窄列(右对齐为主, 簇宽 <= 0.8*ROI宽)
  - 密集多行: 簇内行数 >= 6(真实气泡一条消息一般 1~6 行)
  - 高窄: 簇高 >= 1.2 * 簇宽
  - 左侧/全宽行(x1 < 0.25*ROI宽)作为分隔符打断簇: 避免把上下相邻的
    真实消息并入截图
  - 垂直 gap <= 350px: 截图内偶有大空隙(如按钮区与底部页脚之间)

Known limitation: 6+ 条连续窄右侧真实短消息(无左侧消息间隔)可能被误判,
本视频数据中不存在该模式, 属可接受边界。
"""

import re
from typing import List, Optional, Tuple

from src.data_types import TrackedLine
from src.utils.text_utils import text_similarity

# 配置常量 / Tunable constants
_X_LEFT_RATIO = 0.25    # x1 < 0.25*ROI宽 = 左侧/全宽行, 打断截图簇
_MAX_VGAP = 350         # 截图内允许的最大行垂直间距
_MIN_ROWS = 6           # 截图簇最少行数
_MAX_W_RATIO = 0.8      # 簇宽最多占 ROI 宽度的比例
_MIN_HW_RATIO = 1.15    # 簇高宽比下限(截图高窄特征; 截图底部局部簇实测约1.2)
_PLACEHOLDER_TEXT = "[图片消息]"
# 跨帧残片关联: 同一张截图滚动显示多帧, tracker 会把各帧截图行建成
# first_seen_frame 不同的跟踪行(单帧行数不足 _MIN_ROWS 漏检)。第二阶段把
# 与已检测簇文本高度相似且时间邻近的残片行并入簇, 一起折叠。
_ASSOC_SIM = 0.80       # 归一化后文本相似度下限
_ASSOC_FRAME_GAP = 150  # 与簇首行最大帧差(截图会话跨度, 约5秒@30fps)
# 第三阶段滚动残片传播: 截图滚动露出的相邻区域行与跨帧锚点的最大垂直间距
# (正常截图行距 ~20-60px; 截图下方紧邻的真实消息一般 >100px, 由此区分)
_NEIGH_GAP = 60

_STAMP_RE = re.compile(r"^\d{1,2}:\d{2}\s*√?\s*$")
# 截图文本符号归一: 去空白/装饰符号, 统一括号冒号(OCR对同一截图行的
# 变体: '<信用贷(原借呗)' / '信用贷（原借呗）' / '限时：57折 年利率…①')
_NORM_RE = re.compile(r"[<>\s·√①②③④⑤]+")


def _norm_shot_text(text: str) -> str:
    """规范化截图行文本, 用于跨帧残片与簇内行做相似度关联"""
    t = _NORM_RE.sub("", text or "")
    return (t.replace("（", "(").replace("）", ")")
             .replace("：", ":").replace("，", ","))


def _norm_shot_parts(text: str) -> List[str]:
    """把文本拆成子行并逐行规范化。

    合并块(多帧 OCR 行粘接成 '无卡支付限额①\\n单卡转账限额①')的整行编辑距离
    相似度偏低, 必须按行拆分后逐行与簇内行匹配, 才能命中截图同一行的文本。
    注意: 合并器(_join_lines)以字面 "\\n" 连接行(前端据此还原气泡内换行),
    需先转真实换行再 split; 规范化会把 \\s 整体吞掉, 故先拆行再归一。
    """
    parts: List[str] = []
    for seg in (text or "").replace("\\n", "\n").split("\n"):
        n = _norm_shot_text(seg)
        if n:
            parts.append(n)
    return parts


def _pos(line: TrackedLine) -> Tuple[int, int, int, int]:
    """排序/几何用 bbox: 优先首次出现位置, 无则回退最新位置"""
    return line.first_bbox or line.bbox


def detect_screenshot_clusters(lines: List[TrackedLine],
                               roi_width: int) -> List[List[TrackedLine]]:
    """检测截图内部文字簇。

    返回一个列表, 每个元素是一个截图簇(按垂直顺序排列的 TrackedLine 列表)。
    roi_width <= 0 时不做检测(几何规则依赖屏幕宽度)。
    """
    width = float(roi_width or 0)
    if width <= 0 or len(lines) < _MIN_ROWS:
        return []
    x_left = _X_LEFT_RATIO * width

    # 正文行(跳过空行/时间戳): 时间戳是消息时间标记, 不参与截图判定
    content: List[Tuple[TrackedLine, tuple]] = []
    for line in lines:
        text = (line.text or "").strip()
        if not text or _STAMP_RE.fullmatch(text):
            continue
        content.append((line, _pos(line)))

    # 按首次出现帧分组后再做几何聚簇: 不同帧(滚动后的画面)的截图行/真实消息
    # 在 y 序上会交错, 全局排序会把 run 打断(全宽消息横插其中); 同一帧内的
    # 行才是同屏真实布局, 截图簇的几何特征(窄列/高窄/密集)只在该分组下成立。
    frame_groups: dict = {}
    for line, p in content:
        frame_groups.setdefault(line.first_seen_frame, []).append((line, p))

    runs: List[List[Tuple[TrackedLine, tuple]]] = []
    for lines_in_frame in frame_groups.values():
        lines_in_frame.sort(key=lambda item: (item[1][1], item[1][0]))
        # 右侧窄列 run: 左行打断; 截图内允许 350px 大空隙
        cur: List[Tuple[TrackedLine, tuple]] = []
        for line, p in lines_in_frame:
            if p[0] < x_left:
                if len(cur) >= _MIN_ROWS:
                    runs.append(cur)
                cur = []
                continue
            if cur and p[1] - cur[-1][1][3] <= _MAX_VGAP:
                cur.append((line, p))
            else:
                if len(cur) >= _MIN_ROWS:
                    runs.append(cur)
                cur = [(line, p)]
        if len(cur) >= _MIN_ROWS:
            runs.append(cur)

    clusters: List[List[TrackedLine]] = []
    for run in runs:
        x1 = min(p[0] for _, p in run)
        x2 = max(p[2] for _, p in run)
        y1 = min(p[1] for _, p in run)
        y2 = max(p[3] for _, p in run)
        w, h = x2 - x1, y2 - y1
        if w <= _MAX_W_RATIO * width and h >= _MIN_HW_RATIO * w:
            clusters.append([line for line, _ in run])

    # 第二阶段: 跨帧残片关联。截图滚动显示时 tracker 把同一截图行建成
    # first_seen_frame 不同的多条跟踪行(单帧行数不足 _MIN_ROWS 漏检), 但
    # 残片文本与已检测簇内行高度相似(仅符号/截断差异)且时间邻近。把满足
    # "文本相似度 >= _ASSOC_SIM 且帧差 <= _ASSOC_FRAME_GAP" 的未入簇行
    # 并入对应簇, 一起折叠为 [图片消息]。
    # 残片可能是多行合并块(如 '无卡支付限额①\n单卡转账限额①'), 按子行
    # 逐行匹配, 任一子行命中簇内任一行即视为同屏截图残片。
    if clusters:
        cluster_texts = [[p for l in c for p in _norm_shot_parts(l.text)]
                         for c in clusters]
        cluster_frames = [c[0].first_seen_frame for c in clusters]
        used: set = set()
        for c in clusters:
            used.update(id(line) for line in c)
        for line, p in content:
            if id(line) in used:
                continue
            parts = _norm_shot_parts(line.text or "")
            if not parts:
                continue
            best: Optional[int] = None
            best_gap = 1 << 30
            for ci, texts in enumerate(cluster_texts):
                if not any(text_similarity(part, t) >= _ASSOC_SIM
                           for part in parts for t in texts):
                    continue
                gap = abs(line.first_seen_frame - cluster_frames[ci])
                if gap <= _ASSOC_FRAME_GAP and gap < best_gap:
                    best, best_gap = ci, gap
            if best is not None:
                clusters[best].append(line)
                used.add(id(line))

    # 第三阶段: 截图滚动向下的残片传播。截图向下滚动时, 新露出区域的行
    # (如农行截图底部 '无卡支付限额') 与已并入簇的跨帧行同帧且在其正下方,
    # 文本与簇内行不重叠(同一截图的不同区域)导致文本关联失效。对每个跨帧
    # 锚点行(非首帧并入的行), 收集其同帧正下方 gap <= _NEIGH_GAP 的右侧
    # 窄列行并入同簇, 新并入行再作为锚点继续向下传播(链式)。
    # 只向下传播: 截图上方是聊天界面状态栏/其他消息, 不属于截图; 左侧
    # 真实消息(x1 < x_left)天然排除; gap 收紧到截图行距, 避免把截图下方
    # 相邻的真实消息(如 '凌晨弄的', gap>100px)误并入。
    if clusters:
        in_any: set = set()
        for c in clusters:
            in_any.update(id(line) for line in c)
        for ci, c in enumerate(clusters):
            first_frame = min(l.first_seen_frame for l in c)
            queue = [l for l in c if l.first_seen_frame != first_frame]
            while queue:
                a = queue.pop()
                ap = _pos(a)
                fid = a.first_seen_frame
                for line, p in content:
                    if id(line) in in_any or line.first_seen_frame != fid:
                        continue
                    if p[0] < x_left or p[1] < ap[3]:
                        continue
                    if p[1] - ap[3] > _NEIGH_GAP:
                        continue
                    if not (p[0] <= ap[2] and p[2] >= ap[0]):
                        continue
                    clusters[ci].append(line)
                    in_any.add(id(line))
                    queue.append(line)
    return clusters


def collapse_screenshot_clusters(lines: List[TrackedLine],
                                 clusters: List[List[TrackedLine]],
                                 side_fn) -> List[TrackedLine]:
    """把每个截图簇折叠为一条 "[图片消息]" 占位行。

    side_fn: (bbox) -> "发"/"收"/"", 用于给占位行打收发标记(截图簇按
    几何位置: 右对齐=发, 左对齐=收)。占位行通过 side_override 属性携带
    该方向, 组装阶段优先使用, 避免像素分类对整块截图误判。
    """
    if not clusters:
        return list(lines)
    # 用对象身份建索引: TrackedLine 为 dataclass(eq=True), 不可哈希
    index = {id(line): i for i, line in enumerate(lines)}
    drop: set = set()
    placeholders = {}
    for cluster in clusters:
        if not cluster:
            continue
        idxs = sorted(index[id(line)] for line in cluster)
        # 簇内行按首行索引插入占位; 不做"输入索引连续"要求——merged 列表中
        # 截图行之间可能夹着时间戳/空行(detect 时被过滤), 连续检查会误跳过整簇。
        # 跨帧残片并入簇后, 簇内行散布在不同位置: 占位 bbox/方向取 first_seen
        # 最早的行(截图首次出现位置), 不用整簇并集(跨帧并集会横跨大片区域)。
        anchor = min(cluster, key=lambda l: l.first_seen_frame)
        first = cluster[0]
        bbox = anchor.first_bbox or anchor.bbox
        side = side_fn(bbox) if side_fn else ""
        ph = TrackedLine(
            line_id=f"shot:{first.line_id}",
            text=_PLACEHOLDER_TEXT,
            bbox=bbox,
            first_bbox=bbox,
            confidence=first.confidence if first.confidence else 0.99,
            first_seen_frame=anchor.first_seen_frame,
            last_seen_frame=anchor.last_seen_frame,
            first_timestamp=anchor.first_timestamp,
            last_timestamp=anchor.last_timestamp,
            observations=first.observations,
            is_new=first.is_new,
        )
        if side:
            ph.side_override = side
        drop.update(idxs)
        placeholders[idxs[0]] = ph

    out: List[TrackedLine] = []
    for i, line in enumerate(lines):
        if i in drop:
            if i in placeholders:
                out.append(placeholders[i])
            continue
        out.append(line)
    return out
