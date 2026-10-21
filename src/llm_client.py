"""
LLM客户端(可选)

仅用于最终后处理阶段, 不进入实时流水线。

用途:
  1. OCR错误纠正: 利用跨帧上下文修正OCR识别错误
  2. 结构化输出: 将原始文本行组装为结构化聊天记录或参数表
  3. 缺口标注: 标注内容缺口位置, 辅助人工补查

使用前请在config.yaml中配置:
"""

import bisect
import json
import re
import unicodedata
from typing import List, Optional

from src.audio.ocr_voice_scanner import parse_duration_seconds
from src.bubble_merger import STAMP_RE, classify_side, classify_side_line
from src.data_types import TrackedLine
from src.utils.behavior_logger import BehaviorLogger
from src.utils.text_utils import text_similarity

# 编辑距离截断长度(text_utils.levenshtein_distance 内部截断到200)。
# 长度下界剪枝依赖 ED >= |len1-len2|, 截断后该下界不再成立, 涉及截断的行
# 必须回退全量扫描, 否则剪枝有损、改变判定结果。
_ED_TRUNC = 200

# 短回复(≤2字)气泡的最小尺寸(px)。聊天里的真实短回复(是的/可以/好的/我在)
# 高度约47-59px、宽度约85-179px; App界面里的微型标签/表头(品品28x27、
# 时间33x24、内容54x27、结果32x22)则远小于此, 用尺寸将其挡在正文外。
_SHORT_BUBBLE_MIN_H = 35
_SHORT_BUBBLE_MIN_W = 55

# 界面装饰过滤(降级组装兜底, 与pipeline._clean_ocr_lines口径一致):
# 日期分隔条(如"五月4"/"2026年8月20日"/"08-20")与在线状态是聊天界面的
# 装饰信息, 不是聊天消息。正常流程下pipeline已在上游清洗, 这里兜底
# 漏网之鱼(如断点缓存的旧数据/其他入口), 避免降级输出混入界面元素。
_DATE_DIVIDER_RE = re.compile(
    r"^(?:[0-9一二三四五六七八九十]{1,4}年)?"
    r"[0-9一二三四五六七八九十]{1,3}月[0-9一二三四五六七八九十]{0,3}日?$"
    r"|^\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?$"
)
_ONLINE_STATUS = {"在线", "离线"}

# 尾随聊天时间戳: OCR 常把消息与其下方的小号时间戳识别为一个文本块
# ("口令核实成功20:17"/"可以20:21"), 剥离后才是消息正文; 完整的时间戳行
# 由 STAMP_RE 在更早的分支分离, 不受影响。
_TRAILING_STAMP_RE = re.compile(r"\d{1,2}:\d{2}(?:\.\d+)?\s*√?\s*$")

# LLM输出行前缀: "- 05:55 [发] ：内容" / "- 05:55 ：内容" / "- +12s [收] ：内容"
_LLM_TIME_RE = re.compile(
    r"^(\d{1,2}:\d{2}(?:\.\d+)?|\+\d+s?)\s*(\[[发收]\])?\s*[：:]?\s*(.*)$"
)

# 确定性还原(LLM输出后校验)参数: 内容被权威行包含/为其分段时, 用权威行完整
# 文本替换LLM行。短内容(如"温馨提示"标签)易被长消息误吸, 设最小长度与
# 最小长度占比双重门槛; 完全等于权威行某分段是强证据, 不受占比限制。
_REJOIN_MIN_LEN = 4
_REJOIN_MIN_FRACTION = 0.05

# 同一消息的"OCR变体"判重参数(LLM输出后的滚动去重): 归一化(去标点/空白、统一
# 全半角与小写)后做近似匹配, 并强制"数字锚点"完全一致 —— 否则"您的快递8812"
# 与"您的快递8816"这类仅数字不同的模板近重复消息会被误合并。
# Variant-duplication parameters: normalized near-match plus an identical digit
# anchor, so template-near-duplicates differing only in a digit stay separate.
_VARIANT_SIM = 0.85          # 归一化文本相似度阈值 / normalized similarity threshold
_VARIANT_MIN_LEN = 8         # 前缀包含判重的最短长度(防"好"被长消息吸走) / min length
_VARIANT_MIN_RATIO = 0.35    # 前缀包含判重的最短长度占比 / min length ratio
_VARIANT_PREFIX_SIM = 0.90   # 前缀对齐相似度(容忍截断处的1~2字OCR差异) / prefix sim


def _variant_key(text: str) -> str:
    """归一化判重键: NFKC统一全半角 → 去空白/标点/符号 → 小写(数字原样保留)。"""
    s = unicodedata.normalize("NFKC", text or "")
    s = re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)
    return s.lower()


def _digit_anchor(text: str) -> str:
    """数字锚点: NFKC后所有数字字符的逐位拼接。

    用"拼接"而非"按标点分组"是为了容忍OCR/ASR在标点上的抖动(视觉
    "repro-20261016-0037.md" 与 ASR "repro202610160037MD" 的锚点相同),
    同时仍能区分数字本身不同的模板消息("您的快递8812" vs "您的快递8816")。
    """
    s = unicodedata.normalize("NFKC", text or "")
    return "".join(re.findall(r"\d", s))


def _is_same_message(a: str, b: str) -> bool:
    """两条输出文本是否指向"同一条消息"(OCR变体/滚动重复/滚动截断读法)。"""
    ka, kb = _variant_key(a), _variant_key(b)
    if not ka or not kb:
        return False
    if _digit_anchor(a) != _digit_anchor(b):
        return False
    if ka == kb:
        return True
    if text_similarity(ka, kb) >= _VARIANT_SIM:
        return True
    # 滚动截断读法: 短键 ≈ 长键的前缀(半截行), 长度足够且占比达标。
    # 允许截断处有1~2字OCR差异(如 "...-0042.Ic" 对 "...-0042.Ig 了, ..."),
    # 故用"短键 vs 长键同长前缀"的相似度判定, 而非严格 startswith。
    short, long = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
    if len(short) < _VARIANT_MIN_LEN or len(short) < _VARIANT_MIN_RATIO * len(long):
        return False
    # 后缀方向同理: 滚动**回看**时长消息已完整输出过一次, 之后画面回滚只露出尾部,
    # 尾片段又被读成一条独立行(实测 video4 GT#12: 完整行已在 +29s 输出, 尾片段
    # "明是不是漏了几个？…原始设计稿" 在 +35s 再输出一次)。前缀口径覆盖不到它,
    # 因为它是父消息的**结尾**而非开头。
    # / Suffixes too: after scrolling back only the tail of an already-emitted long
    # message stays visible and is read as a brand-new line.
    return (text_similarity(short, long[:len(short)]) >= _VARIANT_PREFIX_SIM
            or text_similarity(short, long[-len(short):]) >= _VARIANT_PREFIX_SIM)


# 聊天时间戳行(时间分隔条)有效性判定:
# 仅"消息列表区域内、水平居中"的 HH:MM 行才是聊天时间分隔条。据此排除两类
# 伪时间戳(它们同样是 HH:MM 形态, 旧的"只要是 HH:MM 就算聊天时间"会误采):
#   1. 顶部条带(Y < 条带高)内的系统状态栏时钟/标题栏时间;
#   2. 靠边对齐的界面时间值(设置页"已开机时间 49:25"、回单时间、账号/字段值)。
# 微信/QQ 的时间分隔条固定**水平居中**, 因此"居中"是廉价且高精度的判据。
# 容差需收紧: 实测非分隔条的时间值虽非贴边, 也会落在 ROI 中心左右约 50~65px
# 内(如底部悬浮媒体播放条的当前/总时长 "0:21/0:42", 中心偏移 52~54px), 宽松容差
# 会误采。真实分隔条严格居中(偏移 < 10px), 取 0.08 两侧余量充足。
_STAMP_CENTER_TOL = 0.08           # 居中容差(ROI宽度比例)
_STAMP_TOP_STRIP_RATIO = 0.08      # 顶部条带高度占ROI比例(与 pipeline 同口径)
_STAMP_TOP_STRIP_MIN_PX = 120      # 顶部条带最小高度(px)


def _is_short_bubble(bbox) -> bool:
    """短回复气泡尺寸检查: 宽高需达到气泡级, 排除App界面微型标签
    """
    if not bbox or len(bbox) < 4:
        return False
    return ((bbox[2] - bbox[0]) >= _SHORT_BUBBLE_MIN_W
            and (bbox[3] - bbox[1]) >= _SHORT_BUBBLE_MIN_H)


def _stamp_text(s: TrackedLine) -> str:
    """去掉时间戳行中的√标记"""
    return re.sub(r"\s*√\s*", "", (s.text or "").strip()).strip()


def _pos(line: TrackedLine) -> tuple:
    """排序/几何用bbox: 优先首次出现位置(整屏首帧共享同一时间戳, 最新bbox
    会被滚动打乱), 无则回退最新位置。
    """
    return line.first_bbox or line.bbox


def _is_chat_stamp(line: TrackedLine, roi_w: int, roi_h: int) -> bool:
    """HH:MM 行是否为"聊天时间分隔条"(而非状态栏时钟/设置页时长值等)。

    - 位于顶部UI条带(标题栏/状态栏区域)内 → 系统时钟/界面时间 → 非聊天时间;
    - 水平不居中 → 聊天分隔条必居中, 靠边的值(设置页"已开机时间 49:25"等)
      不是聊天时间。
    roi 尺寸未知(<=0)时保持旧行为(不额外过滤), 避免误伤无尺寸信息的调用方。
    """
    if roi_w <= 0 or roi_h <= 0:
        return True
    x1, y1, x2, y2 = _pos(line)
    strip_h = min(_STAMP_TOP_STRIP_RATIO * roi_h, _STAMP_TOP_STRIP_MIN_PX)
    if y1 < strip_h:
        return False
    cx = (x1 + x2) / 2.0
    return abs(cx - roi_w / 2.0) <= _STAMP_CENTER_TOL * roi_w


def _build_stamp_index(tracked_lines: List[TrackedLine],
                       roi_w: int = 0, roi_h: int = 0):
    """收集并排序聊天时间戳行, 供二分最近时间匹配使用

    只采信"消息列表区域内、水平居中"的时间分隔条(_is_chat_stamp), 排除
    系统状态栏时钟与设置页/回单页里同形的 HH:MM 字段值。
    """
    stamps = [
        l for l in tracked_lines
        if (l.text or "").strip() and STAMP_RE.fullmatch((l.text or "").strip())
        and _is_chat_stamp(l, roi_w, roi_h)
    ]
    stamps_sorted = sorted(stamps, key=lambda s: (s.first_timestamp, _pos(s)[1]))
    stamps_ts = [s.first_timestamp for s in stamps_sorted]
    return stamps_sorted, stamps_ts


def _nearest_stamp(stamps_sorted, stamps_ts, line) -> Optional[TrackedLine]:
    """选择与消息视频时刻最接近的时间戳行(同帧用y距离破平)

    二分定位后仅检查: 同帧全部stamp(y破平) + 相邻前后各一个不同时刻。
    与全量扫描语义等价: dt 主导得分, 最近时间戳必在二分邻域内。

    y距离用bbox垂直中点(而非顶边): 多行合并后的消息行(含字面\\n)顶边会
    偏离其自身的时间戳行, 中点更接近。
    """
    tm = line.first_timestamp
    idx = bisect.bisect_left(stamps_ts, tm)
    lo = idx
    while lo > 0 and stamps_ts[lo - 1] == tm:
        lo -= 1
    hi = idx
    while hi < len(stamps_sorted) and stamps_ts[hi] == tm:
        hi += 1
    hi = min(hi + 1, len(stamps_sorted))   # 右侧再取一个不同时刻
    lo = max(lo - 1, 0)                     # 左侧再取一个不同时刻
    pa = _pos(line)
    ya = (pa[1] + pa[3]) / 2.0
    best = None
    best_score = float("inf")
    for i in range(lo, hi):
        s = stamps_sorted[i]
        dt = abs(s.first_timestamp - tm)
        dy = 0.0
        if dt < 1e-6:
            # 同帧出现: 时间戳通常在消息下方, 用垂直中点距离破平
            pb = _pos(s)
            dy = abs(ya - (pb[1] + pb[3]) / 2.0) / 100.0
        score = dt + dy
        if score < best_score:
            best_score = score
            best = s
    return best


class _SeenIndex:
    """长度分桶索引(相似度判定的无损剪枝)

    原理:
      text_similarity = 1 - ED/max_len, 且编辑距离满足 ED >= |len1-len2|,
      故长度差达到 (1-thr)*max_len 的两行相似度必然 <= thr, 可直接从候选
      中剪枝——剪枝无损, 不改变判定结果。

    注意:
      仅用于昂贵的编辑距离(相似度)比较。子串/包含检查走全量扫描(原生
      str.__contains__ 为 C 实现, 快), 不做剪枝以保持语义完全等价。

    复杂度:
      全量两两相似度 O(n²) → 每行仅与长度接近的 O(k) 个候选精确比较。
    """

    def __init__(self) -> None:
        self._buckets: dict = {}   # 长度 -> 该长度的已见文本列表

    def add(self, text: str) -> None:
        """记录一个已见文本"""
        L = len(text)
        self._buckets.setdefault(L, []).append(text)

    def all_texts(self) -> List[str]:
        """所有已见文本(无序)"""
        out: List[str] = []
        for bucket in self._buckets.values():
            if bucket:
                out.extend(bucket)
        return out

    def _range(self, lo: int, hi: int) -> List[str]:
        """返回长度在[lo, hi]内的所有已见文本"""
        out: List[str] = []
        for L in range(max(1, lo), hi + 1):
            bucket = self._buckets.get(L)
            if bucket:
                out.extend(bucket)
        return out

    def similar_candidates(self, text: str, threshold: float) -> List[str]:
        """可能与text相似度>threshold的候选(长度差下界剪枝, 语义无损)

        无损性:
          相似度 1-ED/max_len 且 ED >= |len1-len2|, 由 |L-L2| 可达的剪枝半径
          max_delta = (1-thr)*L/thr 推出(见下)。仅当被剪枝行的长度差足够大、
          相似度必然 <= threshold 时才剪, 判定结果与全量比较完全一致。
          剪枝半径内两行都不触碰截断(<=200)时证明成立; 任一触碰截断的行
          (本行 L>=200 或候选 L2>=200) 长度下界失效, 回退全量包含。
        """
        L = len(text)
        if threshold <= 0.0 or L <= 0:
            return []
        if L >= _ED_TRUNC:
            # 本行被截断: ED下界降为0, 无法按长度剪枝 → 全量
            return self.all_texts()
        # 由 |L-L2| < (1-thr)*max(L,L2) <= (1-thr)*(L+|L-L2|) 解得
        # |L-L2| < (1-thr)*L/thr, 再放宽边界取整
        max_delta = int((1.0 - threshold) * L / threshold) + 2
        out = self._range(L - max_delta, L + max_delta)
        # 候选长度>=200也被截断, 与text的ED下界不再由长度差保证 → 全量包含
        for Lb, bucket in self._buckets.items():
            if Lb >= _ED_TRUNC and bucket:
                out.extend(bucket)
        return out


class LlmClient:
    """LLM后处理客户端"""

    def __init__(self, config: dict, logger: BehaviorLogger):
        llm_cfg = (config or {}).get("llm", {})
        self._enabled: bool = llm_cfg.get("enabled", False)
        self._endpoint: str = llm_cfg.get("api_endpoint", "")
        self._api_key: str = llm_cfg.get("api_key", "")
        self._model: str = llm_cfg.get("model", "deepseek-chat")
        self._timeout: int = llm_cfg.get("timeout", 60)
        self._thinking: bool = llm_cfg.get("thinking", False)

        # 聊天收发标记配置
        chat_cfg = (config or {}).get("chat_format", {})
        self._chat_enabled: bool = chat_cfg.get("enabled", True)
        self._roi_width: int = int(chat_cfg.get("roi_width", 0) or 0)
        self._roi_height: int = 0        # ROI高度(时间戳顶部条带判定用, 流水线注入)
        self._center_margin: int = int(chat_cfg.get("center_margin", 20))
        self._time_fallback: bool = chat_cfg.get("time_fallback", True)
        self._frame_provider = None      # 气泡像素颜色判定用帧提供器(流水线注入)
        self._logger = logger

    def set_roi_width(self, width: int) -> None:
        """设置ROI宽度(收发定位用, 由流水线按实际ROI传入)
        """
        if width and int(width) > 0:
            self._roi_width = int(width)

    def set_roi_height(self, height: int) -> None:
        """设置ROI高度(时间戳顶部条带判定用, 由流水线按实际ROI传入)"""
        if height and int(height) > 0:
            self._roi_height = int(height)

    def set_frame_provider(self, provider) -> None:
        """设置按帧序号(1基)取ROI帧的回调, 供气泡填充色判定收发。"""
        self._frame_provider = provider

    def _classify_side(self, line) -> str:
        """
        按气泡判定收发

        像素优先: 气泡填充色(绿=发/白=收)适用全宽气泡布局; 无像素证据时
        回退 bubble_merger.classify_side 的左右留白几何法。组装与合并共用。
        返回 / Returns: "发" / "收" / ""
        """
        if not self._chat_enabled:
            return ""
        # 截图过滤折叠出的 [图片消息] 占位行: 方向由几何位置确定并提前写入,
        # 像素分类对整块截图区域不可靠, 直接采用该标记。
        override = getattr(line, "side_override", None)
        if override in ("发", "收"):
            return override
        if isinstance(line, TrackedLine):
            return classify_side_line(line, self._roi_width,
                                      self._center_margin, True,
                                      self._frame_provider)
        return classify_side(line, self._roi_width, self._center_margin, True)

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._api_key)

    def enhance_result(self, tracked_lines: List[TrackedLine],
                       gaps: List[dict]) -> dict:
        """
        用LLM增强最终结果

        参数:
            tracked_lines: 所有跟踪的文本行
            gaps: 检测到的内容缺口
        返回:
            增强后的结构化结果

        注意:
            气泡内换行碎片由调用方(ContentAssembler)在进入本方法前, 通过
            bubble_merger.merge_wrapped_lines 完成几何合并; 这里只做增强。
        """
        if not self.enabled:
            return self._assemble_without_llm(tracked_lines, gaps)

        try:
            prompt = self._build_prompt(tracked_lines, gaps)
            response = self._call_api(prompt)
            result = self._parse_response(response, tracked_lines, gaps)
            self._logger.log("llm_client", "enhance",
                             f"LLM增强完成: {len(result.get('sections', []))}个段落")
            return result
        except Exception as e:
            self._logger.log("llm_client", "error",
                             f"LLM增强失败, 降级为基础组装: {e}")
            return self._assemble_without_llm(tracked_lines, gaps)

    def _build_prompt(self, tracked_lines: List[TrackedLine],
                      gaps: List[dict]) -> str:
        """构建LLM提示(聊天记录录屏专用)

        把"散开的碎片"合并为完整的一条条聊天消息: 同一气泡的消息常被
        OCR切成多行/多次观测(换行截断、滚动半屏、气泡过长), 需要按
        时间接近 + 方向一致 + 内容衔接 合并还原。
        """
        stamps_sorted, stamps_ts = _build_stamp_index(
            tracked_lines, self._roi_width, self._roi_height)
        lines_text = []
        for line in sorted(tracked_lines,
                           key=lambda x: (x.first_timestamp, _pos(x)[1])):
            text = (line.text or "").strip()
            if not text:
                continue
            if STAMP_RE.fullmatch(text):
                continue  # 时间戳行不进正文
            # 语音气泡内的时长标签(如 6"…): 不是聊天消息, 不进 prompt,
            # 避免LLM将其原样输出为消息行
            # / Duration label inside a voice bubble: not a chat message,
            # keep it out of the prompt so the LLM cannot echo it back.
            if parse_duration_seconds(text) is not None:
                continue
            side = self._classify_side(line)
            side_txt = {"发": "我方发出", "收": "对方发出"}.get(side, "方向未知")
            st = _nearest_stamp(stamps_sorted, stamps_ts, line)
            tstr = _stamp_text(st) if st is not None else f"+{line.first_timestamp:.0f}s"
            lines_text.append(
                f"[视频t={line.first_timestamp:.1f}s] [聊天时间{tstr}] "
                f"[{side_txt}] “{text}” "
                f"(置信度{line.confidence:.2f}, 观测{line.observation_count()}次)"
            )

        gaps_text = ""
        if gaps:
            gaps_text = "\n检测到内容缺口(视频中这些时段画面跳变/无文本):\n"
            for gap in gaps:
                gaps_text += f"  - 时间{gap.get('timestamp', 0):.1f}s: {gap.get('detail', '')}\n"

        prompt = f"""你是"手机聊天记录录屏"OCR结果的整理助手。输入是聊天界面(微信/QQ类)录屏经OCR识别出的文本行, 已按时间排序。

【输入字段说明】
- [视频t=x.xs]: 该行在视频中的时刻(秒), 用于判断碎片是否紧邻
- [聊天时间xx:xx]: 该行所属消息列表内"时间分隔条"的显示时间(居中时间条), 输出时沿用
- [我方发出/对方发出/方向未知]: 按气泡水平位置判断的收发方向
- (置信度, 观测次数): OCR置信度与多次观测投票

【核心问题】同一条消息常被OCR切成多行/多条, 信息"散开"了。例如:
  "好的，没问题，请您填写以下信息做个简" 与 "单的登记" → 同一条消息的两段
  你需要把它们合并还原成: "好的，没问题，请您填写以下信息做个简单的登记"

【处理规则】
1. 碎片合并: 时间接近、收发方向一致、内容语义衔接的连续碎片合并为一条完整消息;
   判断依据包括行尾话没说完、缺标点、下一段以衔接词开头、位于同一气泡位置等;
   若两条碎片的收发方向标注不一致但语义明显是同一句话的衔接(如括号跨行),
   以语义为主, 取前面碎片的收发方向合并
   注意: 同一气泡内的多行已由几何模块合并为一条输入行, 行与行之间用字面
   \\n 分隔(例如 "好的，没问题…做个简单的登记\\n1.您的年龄" 表示同一气泡
   内的两行, 是同一条消息)。此类输入行是完整消息, 必须整条原样输出,
   禁止拆成多行, 禁止把字面 \\n 改成句号/空格/逗号
2. 每条消息一行, 格式: - 聊天时间 [发] ：完整内容  或  - 聊天时间 [收] ：完整内容
   ([我方发出]→[发], [对方发出]→[收], 聊天时间用行首标注的时间)
   消息内部若含字面 \\n, 原样保留(表示气泡内换行)
3. 滚动回看造成的重复内容只保留一次
4. 明显OCR错字/乱码, 仅在上下文可确定时纠正; 其余严格保留原文
5. 系统通知/无方向文本(如"您已添加了对方"、居中提示)不加收发标记: - 聊天时间 ：内容
6. 独立界面元素不要合并, 各自成行: 昵称与在线状态(如"资金专员★"与"在线")、
   按钮文字、菜单项、页面标题、短提示等不是同一条消息, 禁止拼在一起
7. 严格按原文措辞合并, 不得改写、润色、编造或翻译; 没有碎片可合并的也逐条输出, 不遗漏
8. 聊天界面顶部/中间出现的界面装饰(对方昵称、在线状态如"在线"、日期分隔条
   如"五月4"、系统通知条等)不是聊天消息, 一律丢弃, 禁止输出为消息或并入消息

以下是OCR文本行:
{chr(10).join(lines_text)}
{gaps_text}

请只输出JSON(不要markdown代码块), 结构:
{{
  "sections": [
    {{"timestamp": 0.0, "content": "段落内所有消息行(每行一条, 换行分隔), 格式 - 时间 [发] ：内容", "timestamps": ["05:55"], "note": ""}}
  ],
  "gaps": [
    {{"timestamp": 0.0, "note": "缺口说明"}}
  ],
  "summary": "整体内容摘要"
}}
"timestamp"填该段落第一条消息对应的视频时间(秒); "timestamps"填段落内聊天时间列表。"""
        return prompt

    def _call_api(self, prompt: str) -> dict:
        """调用LLM API"""
        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system",
                 "content": "你是聊天记录录屏OCR整理专家。严格按用户要求输出JSON。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 8192,
        }
        if not self._thinking:
            payload["thinking"] = {"type": "disabled"}
        resp = requests.post(self._endpoint, json=payload, headers=headers,
                             timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _normalize_continuations(sections: List[dict]) -> None:
        """把LLM输出中的裸续行并回上一条消息(以字面\\n连接)
        """
        for sec in sections:
            lines = (sec.get("content") or "").split("\n")
            out = []
            for ln in lines:
                if out and not ln.strip().startswith("- "):
                    out[-1] += "\\n" + ln.strip()
                else:
                    out.append(ln)
            sec["content"] = "\n".join(out)

    @staticmethod
    def _split_llm_line(line: str) -> tuple:
        """解析LLM输出行 -> (head, content)。head形如 "05:55 [发]"。

        行的标准格式 "- 时间 [发] ：内容"。无法解析(无时间前缀/内容为空)时
        返回 ("", 整行), 交由调用方按原文保留。
        """
        s = line.strip()
        if not s.startswith("- "):
            return "", s
        m = _LLM_TIME_RE.match(s[2:])
        if not m or not (m.group(3) or "").strip():
            return "", s
        head = " ".join(x for x in (m.group(1), m.group(2) or "") if x)
        return head, m.group(3).strip()

    def _rejoin_llm_sections(self, sections: List[dict],
                             tracked_lines: List[TrackedLine]) -> None:
        """LLM输出后的确定性校验: 拆行还原 + 滚动去重 + 时间校正(原地修改)。

        背景:
          几何合并(merge_wrapped_lines)已按"同一气泡=同一条消息"把长消息的
          多行合并为含字面 \\n 的完整文本, 这是确定性的权威结果。LLM增强可能
          违反"输入行是完整消息, 必须整条原样输出"的规约, 出现:
            a) 拆行: 在字面 \\n 处把一条完整消息拆成多条;
            b) 重复: 滚动回看使同一消息多次出现, 未按规则只保留一次;
            c) 时间改写: 把多条消息的聊天时间改成同一个值。
          本方法以几何合并结果为权威基准做确定性还原: 内容被权威行包含(或为
          其分段)的LLM行, 用权威行完整文本替换; 同一权威行的多条输出只保留
          一次; 命中的行其聊天时间用权威行对应的真实时间校正。未被权威行覆盖
          的独立行(短回复/系统通知/界面元素等)原样保留。
        """
        # 权威行: 几何合并后的非时间戳正文
        authority = [
            (l.text.strip(), l) for l in tracked_lines
            if (l.text or "").strip()
            and not STAMP_RE.fullmatch((l.text or "").strip())
        ]
        if not authority:
            return

        stamps_sorted, stamps_ts = _build_stamp_index(
            tracked_lines, self._roi_width, self._roi_height)

        def real_time(line: TrackedLine) -> str:
            """权威行对应的真实聊天时间(LLM改写的时间一律以它为准)"""
            st = _nearest_stamp(stamps_sorted, stamps_ts, line)
            return _stamp_text(st) if st is not None else f"+{line.first_timestamp:.0f}s"

        def best_authority(content: str):
            """内容上包含content的最长权威行 -> (text, line) 或 None

            匹配条件(防短词误吸):
              - 内容是被权威行包含的子串, 且长度占比>=_REJOIN_MIN_FRACTION
                (如"温馨提示"4字标签不会被150字长消息吸走);
              - 内容完全等于权威行某个 \\n 分段 → 强证据, 不受占比限制;
              - 内容与某分段互相包含(LLM微调过标点/字数), 同样要求长度占比。
            同时命中多个权威行时, 完全相等分段的权威行优先, 其次取最长。
            """
            c = content.strip()
            if len(c) < _REJOIN_MIN_LEN:
                return None
            best = None            # (text, line)
            best_exact = None
            for t, line in authority:
                min_len = max(_REJOIN_MIN_FRACTION * len(t), _REJOIN_MIN_LEN)
                hit_full = c in t and len(c) >= min_len
                if hit_full and (best is None or len(t) > len(best[0])):
                    best = (t, line)
                for seg in t.split("\\n"):
                    seg = seg.strip()
                    if seg == c:
                        if best_exact is None or len(t) > len(best_exact[0]):
                            best_exact = (t, line)
                        break
                    if (seg and (c in seg or seg in c)
                            and len(c) >= min_len
                            and (best is None or len(t) > len(best[0]))):
                        best = (t, line)
            return best_exact or best

        # 展平为全局行序列(段落边界不影响滚动去重), 便于跨段落判重
        flat = []      # 每项 [head, content, sec_idx]; content=None 表示已丢弃
        for si, sec in enumerate(sections):
            for ln in (sec.get("content") or "").split("\n"):
                head, content = self._split_llm_line(ln)
                flat.append([head, content, si])

        # 已保留的权威行: 归一化近似判重(同一消息的OCR变体/滚动残影/滚动截断读法
        # 只保留一次), 并保留数字锚点约束避免仅数字不同的模板消息被误合并。
        emitted = []   # 每项 [代表文本, flat下标]
        for idx, row in enumerate(flat):
            head, content = row[0], row[1]
            if not content.strip():
                continue
            hit = best_authority(content)
            if hit is None:
                continue
            t, line = hit
            dup = next((e for e in emitted
                        if _is_same_message(e[0], t)), None)
            if dup is not None:
                # 同一消息重复出现: 若本条读法更完整则升级代表文本
                # (时间仍保留首次出现, 即该消息最早被看到的时刻)
                if len(t) > len(dup[0]):
                    dup[0] = t
                    flat[dup[1]][1] = t
                row[1] = None
                continue
            emitted.append([t, idx])
            # 时间以权威行真实时间为准; 收发标记优先用像素判定的权威方向
            # (LLM可能沿用了错误的方向标注), 无像素证据时保留LLM原值
            auth_side = self._classify_side(line)
            parts = head.split()
            mark = parts[-1] if parts and parts[-1] in ("[发]", "[收]") else ""
            if auth_side:
                mark = f"[{auth_side}]"
            row[0] = (real_time(line) + (" " + mark if mark else "")).strip()
            row[1] = t

        out_by_sec = [[] for _ in sections]
        for head, content, si in flat:
            if content is None:
                continue
            out_by_sec[si].append((head, content))
        for si, sec in enumerate(sections):
            sec["content"] = "\n".join(
                f"- {head} ：{content}" if head else content
                for head, content in out_by_sec[si])
        # 全部被去重的段落移除, 避免前端渲染空段落
        sections[:] = [s for s in sections if (s.get("content") or "").strip()]

    def _parse_response(self, response: dict, tracked_lines: List[TrackedLine],
                        gaps: List[dict]) -> dict:
        """解析LLM响应"""
        try:
            content = response["choices"][0]["message"]["content"]
            # 尝试从markdown代码块中提取JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content.strip())
            self._normalize_continuations(result.get("sections", []))
            # 确定性校验: 几何合并结果是权威基准, 把LLM拆散/重复/改时间的行
            # 还原为合并后的完整消息(拆行还原 + 滚动去重 + 时间校正)
            self._rejoin_llm_sections(result.get("sections", []), tracked_lines)
            result["llm_enhanced"] = True
            return result
        except (KeyError, json.JSONDecodeError) as e:
            self._logger.log("llm_client", "parse_error",
                             f"解析LLM响应失败: {e}")
            return self._assemble_without_llm(tracked_lines, gaps)

    def _assemble_without_llm(self, tracked_lines: List[TrackedLine],
                              gaps: List[dict]) -> dict:
        """不使用LLM的基础组装

        处理:
          - 按时间排序
          - 时间戳行(如23:53)从正文分离, 同时作为后续消息的显示时间
          - 滚动回看产生的重复行去重
          - 聊天收发标记: 按bbox水平位置输出 [发]/[收]

        复杂度:
          核心判定(去重/碎片)由"全量两两比较 O(n²)"降为"长度分桶索引剪枝
          O(n·k)": 由编辑距离下界 ED>=|len1-len2| 可知, 长度差达到
          (1-阈值)×max_len 的两行相似度必然不达标, 可无损剪枝。
          时间戳匹配由"全量扫描 O(n×m)"降为"排序+二分 O(n log m)"。
        """
        # 预收集时间戳行(用于消息时间关联), 排序后二分查找最近时间戳
        stamps_sorted, stamps_ts = _build_stamp_index(
            tracked_lines, self._roi_width, self._roi_height)

        def fmt_line(line: TrackedLine, text: str) -> str:
            """生成 "- 时间 [发] ：内容" 格式行"""
            if not self._chat_enabled:
                return text
            side = self._classify_side(line)
            s = _nearest_stamp(stamps_sorted, stamps_ts, line)
            if s is not None:
                tstr = _stamp_text(s)
            elif self._time_fallback:
                tstr = f"+{line.first_timestamp:.0f}s"
            else:
                tstr = ""
            mark = f"[{side}] " if side else ""
            head = f"{tstr} {mark}".strip()
            if head:
                return f"- {head} ：{text}"
            return f"- {text}"

        sections = []
        current_time = None
        current_lines = []
        current_stamps = []
        # 长度分桶索引(已见正文行)
        seen_index = _SeenIndex()

        def is_fragment(candidate: str, kept: str) -> bool:
            """滚动断裂片段判断: 仅当候选行是已保留完整行的子串(短残片)
            注意方向: "kept in candidate"(候选是更长的完整行, 只是包含某个
            已保留的短UI标签, 如4字"温馨提示"标签 vs 42字完整消息)不是残片,
            必须保留, 否则长消息会被短标签误杀丢弃。
            the 4-char label "温馨提示" vs the 42-char full message); that
            """
            if candidate in kept:
                return abs(len(candidate) - len(kept)) >= 2
            return text_similarity(candidate, kept) > 0.90

        def keep_line(line: TrackedLine, text: str) -> bool:
            """滚动残影过滤: 过滤滚动残片/错字行, 保留真实内容
              - 单观测短碎片(≤2字, 如"光公") → 滤, 但"是的/可以/好的/我在"
                这类真实短回复(有明确气泡收发方向 + 高置信) → 保留
              - 单观测低置信(<0.70) → 滤
              - 单观测被已保留完整行包含/高相似 → 滚动断裂片段 → 滤
              - 多观测但平均置信度低(<0.90) → 残影多帧复现 → 滤

            短回复为何跳过碎片检查:
            "可以" 是 "排队时间，你看这样可以吗？" 的子串、"好的" 是
            "好的呢，稍等一下哦，核实一下口令" 的子串, 但它们是独立的
            真实消息。滚动残片要么无清晰气泡方向, 要么与已保留行高度相似
            (会被前面的去重逻辑拦截), 因此带方向+高置信的短行直接保留。
            "可以" is a substring of "排队时间，你看这样可以吗？" and
            "好的" of "好的呢，稍等一下哦，核实一下口令", yet they are
            """
            obs = line.observation_count()
            if obs <= 1:
                if len(text) <= 2:
                    if (line.confidence >= 0.80
                            and self._classify_side(line)
                            and _is_short_bubble(_pos(line))):
                        return True
                    return False
                if line.confidence < 0.70:
                    return False
                # 候选 = 高度相似(无损剪枝) + 包含/被包含(全量扫描, 与旧逻辑一致)
                cands = seen_index.similar_candidates(text, 0.90)
                for seen in seen_index.all_texts():
                    if text in seen or seen in text:
                        cands.append(seen)
                if any(is_fragment(text, kept) for kept in cands):
                    return False
            else:
                if line.confidence < 0.90:
                    return False
            return True

        for line in sorted(tracked_lines, key=lambda x: (x.first_timestamp, _pos(x)[1])):
            text = (line.text or "").strip()
            if not text:
                continue
            # 时间戳分离: "23:53" / "00:00√" → 段落时间标记
            if STAMP_RE.fullmatch(text):
                current_stamps.append(text)
                continue
            # 尾随时间戳剥离(兜底, 与pipeline清洗一致): OCR常把消息与其下方的
            # 小号时间戳识别为一个文本块("口令核实成功20:17"), 剥离后才是正文
            text2 = _TRAILING_STAMP_RE.sub("", text).strip()
            if text2:
                text = text2
            # 语音气泡内的时长标签(如 6"…): 不是聊天消息, 与 pipeline 上游
            # 清洗同口径剔除, 否则会以正文行形态混入聊天面板
            # / Duration label inside a voice bubble (e.g. 6"…): not a chat
            # message; drop here with the same rule as the pipeline upstream.
            if parse_duration_seconds(text) is not None:
                continue
            # 界面装饰过滤(兜底): 日期分隔条(如"五月4")/在线状态是界面信息,
            # 不是聊天消息, 降级路径也一律丢弃(与LLM提示规则8一致)
            if _DATE_DIVIDER_RE.fullmatch(text) or text in _ONLINE_STATUS:
                continue
            # 滚动回看去重: 与已见行高度相似则跳过
            if any(text_similarity(text, seen) > 0.95
                   for seen in seen_index.similar_candidates(text, 0.95)):
                continue
            # 滚动残影过滤(见keep_line)
            if not keep_line(line, text):
                continue
            seen_index.add(text)

            if current_time is None or abs(line.first_timestamp - current_time) > 2.0:
                if current_lines:
                    sections.append({
                        "timestamp": round(current_time, 2) if current_time else 0.0,
                        "content": "\n".join(current_lines),
                        "timestamps": current_stamps,
                        "note": ""
                    })
                current_time = line.first_timestamp
                current_lines = [fmt_line(line, text)]
                current_stamps = []
            else:
                current_lines.append(fmt_line(line, text))

        if current_lines:
            sections.append({
                "timestamp": round(current_time, 2) if current_time else 0.0,
                "content": "\n".join(current_lines),
                "timestamps": current_stamps,
                "note": ""
            })

        return {
            "sections": sections,
            "gaps": [{"timestamp": round(g.get("timestamp", 0), 2),
                      "note": g.get("detail", "")} for g in gaps],
            "summary": f"共{len(tracked_lines)}行文本, {len(sections)}个段落, {len(gaps)}个缺口",
            "llm_enhanced": False
        }
