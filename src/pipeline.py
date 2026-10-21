"""
提取流水线

编排所有模块, 完成从视频到结构化文本的完整处理流程。

流水线流程:
  1. 自适应抽帧 (帧差MAE三态采样)
  2. 拉普拉斯模糊过滤 (相对阈值)
  3. 视觉去重 (行差分剖面 + 条带dHash + 三档置信度)
  4. OCR识别 (PaddleOCR API/本地/Mock)
  5. 文本行跟踪 (空间感知逐行跟踪)
  6. 语义去重 (灰色地带LLM仲裁 + OCR纠错) [方案A]
  7. 阈值校准反馈 (文本→视觉) [方案B]
  8. 时序状态机 (OCR统计特征驱动)
  9. 内容组装 (可选LLM增强)

双模态联动架构:
  视觉模态(OCR前)          文本模态(OCR后)          LLM仲裁(灰色地带)
  三档置信度               空间感知去重              语义判定+OCR纠错
       ↑                        |                       |
       |                        ↓                       ↓
       +--- 方案B: 阈值校准反馈 ←---- 文本判定结果 ←--- LLM判定结果
"""

import hashlib
import json
import os
import queue
import re
import threading
import time

import cv2
import numpy as np
from typing import Optional, List, Tuple

from src.config_loader import Config
from src.frame_sampler import FrameSampler, FrameData
from src.roi_manager import RoiManager
from src.blur_detector import BlurDetector
from src.visual_dedup import VisualDedup, ConfidenceTier
from src.frame_value import FrameValueEstimator
from src.ocr_client import OcrClient
from src.message_bubble_tracker import MessageBubbleTracker
from src.state_machine import StateMachine, SystemState
from src.semantic_dedup import SemanticDedup, SemanticVerdict
from src.content_assembler import ContentAssembler
from src.data_types import TextLine
from src.utils.behavior_logger import BehaviorLogger
from src.utils.image_utils import compute_mae, compute_dhash, hamming_distance
from src.visual.chat_scene import chat_scene_reason

from src.audio.asr_client import AsrClient
from src.audio.transcriptor import transcribe_media, TranscriptResult
from src.audio.voice_bubble_scanner import (VoiceBubbleScanner,
                                            cluster_static_observations,
                                            drop_narrow_bubbles,
                                            drop_text_bubbles,
                                            merge_ocr_and_vision,
                                            probe_media_duration)
from src.audio.audio_loader import has_audio_track
from src.audio.report_merge import (build_scroll_curve, merge_into_result,
                                    scroll_offset_at)
from src.audio.data_types import VoiceEvent
from src.audio.ocr_voice_scanner import OcrVoiceScanner, parse_duration_seconds


# 流水线哨兵
_VERDICT_END = ("__verdict_end__",)   # 采集结束信号
_OCR_END = object()                    # 提交队列结束信号

# 界面元素过滤
# 聊天界面录屏中, 顶部标题栏(对方昵称)、在线状态、日期分隔条(如"五月4")是
# 对话框/手机系统信息, 不属于感兴趣区域内的聊天消息, 必须过滤, 否则会作为
# 消息行进入正文并干扰收发/时间/顺序。手机状态栏/导航栏已由ROI边距剔除。
_UI_STRIP_RATIO = 0.08        # 顶部UI条带占ROI高度比例
_UI_STRIP_MIN_PX = 120        # 顶部UI条带最小高度(px)
_UI_STRIP_CENTER_TOL = 0.05   # 条带内"居中"判定容差(ROI宽度比例)
_UI_SHORT_LEN = 6             # 条带内视为界面元素的短文本上限
_DATE_DIVIDER_RE = re.compile(
    r"^(?:[0-9一二三四五六七八九十]{1,4}年)?"
    r"[0-9一二三四五六七八九十]{1,3}月[0-9一二三四五六七八九十]{0,3}日?$"
    r"|^\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?$"
)
# 日期分隔条词元(整行匹配之外, 用于识别滚动帧中与消息粘成一行的分隔条)
_DATE_DIVIDER_TOKEN_RE = re.compile(
    r"[0-9一二三四五六七八九十]{1,4}月[0-9一二三四五六七八九十]{0,3}日?"
)
# 日期分隔条高度带(ROI内, 标题栏下方固定位置, 滚动时该条常驻顶部)
_UI_DIVIDER_BAND_LO = 15     # 带下界: strip_h 之下偏移
_UI_DIVIDER_BAND_HI = 115    # 带上界: strip_h 之下偏移
# 尾随聊天时间戳: OCR 常把消息与相邻时间戳粘成一行("对啊23:47"/"我没有切换23:40.11"),
# 时间戳是界面标记, 剥离后才是消息正文。
_TRAILING_STAMP_RE = re.compile(r"\d{1,2}:\d{2}(?:\.\d+)?\s*√?\s*$")
# 语音气泡时长标签(如 "5\" ((" / ") 4\"" / "18\" (" / "4° ("):
# 语音消息气泡渲染"时长 + 播放图标", OCR 常把图标读成引号/括号/度号, 得到
# 一串"数字 + 引号/括号"的伪文本。它是界面标记(气泡元信息)而非消息正文。
# 判据: 不含汉字/字母, 含数字, 且全部字符都落在数字/引号/括号集合内。
# Voice-bubble duration badge ("5\" ((" / "18\" (" ...): UI metadata, not content.
_VOICE_BADGE_CHARS = set("0123456789\"'’‘“”″°()（）[]【】{}<> \t")
# 长按菜单 / 气泡操作按钮文案: 界面元素, 不属于消息。
# ALWAYS: 只会出现在菜单里的文案, 直接丢弃;
# AMBIGUOUS: 可能与真实消息同形(如单独一条"删除"), 仅当同一帧出现 >=2 个
# 不同文案时才判定为菜单簇一并丢弃, 避免误杀正文。
# Long-press menu / bubble-action labels: UI chrome, not messages.
_UI_LABEL_ALWAYS = frozenset({"转文字", "听筒播放", "背景播放", "连续朗读",
                              "倍速播放"})
_UI_LABEL_AMBIGUOUS = frozenset({"收藏", "删除", "多选", "引用", "提醒",
                                 "复制", "转发", "翻译", "搜一搜", "朗读"})
# 撤回提示条(如"你撤回了一条消息"/"对方撤回了一条消息"): 系统通知, 不是消息
_RECALL_NOTICE_RE = re.compile(r"^.{0,6}撤回了一条消息$")
# 系统弹窗/通知(如'电量剩余20%是否开启"省电模式"?'): 手机系统 UI 覆盖层, 不是聊天
# 消息。对话框自带的按钮文案(如"开启"/"取消")单独看与正文同形, 故仅当同一帧出现
# 对话框正文时才随簇一并丢弃(与长按菜单簇同一策略, 防误杀正文)。
# System dialog / OS notification banner: OS chrome, not a chat message.
_SYS_DIALOG_RE = re.compile(
    r"(?:电量|电池)(?:剩余|不足|低于)?\s*\d{1,3}\s*%"      # 电量剩余20%
    r"|是否(?:开启|关闭).{0,6}(?:模式|省电|功能)"           # 是否开启"省电模式"?
)
_DIALOG_BUTTONS = frozenset({"开启", "关闭", "取消", "确定", "允许", "拒绝",
                             "稍后", "知道了", "立即开启", "立即关闭"})


def _is_voice_badge(t: str) -> bool:
    """语音气泡时长标签判据 / voice-bubble duration badge test"""
    return (any(ch.isdigit() for ch in t)
            and all(ch in _VOICE_BADGE_CHARS for ch in t))


class ExtractionPipeline:
    """提取流水线(双模态联动)

    参数:
        config: 配置对象
        output_dir: 输出目录覆盖(Web服务按会话隔离用; None=取config.output.dir)
        save_frames: 是否保存诊断帧(覆盖config; None=取config值)
        progress_callback: 进度回调(Web服务实时推送用; None=禁用)
                           事件类型:
                             stage    -> {"type":"stage", "stage", "detail", ...}
                             progress -> {"type":"progress", "frame_id", "total_frames",
                                          "timestamp", "elapsed_sec", "stats"}
                             frame    -> {"type":"frame", "frame_id", "total_frames",
                                          "timestamp"}
                             ocr_frame-> {"type":"ocr_frame", "frame_id", "timestamp",
                                          "lines":[{"text","bbox","confidence"}],
                                          "scroll_frame"}
        live_log: 行为日志实时落盘(Web演示实时日志流用; False=结束统一写盘)
    """

    def __init__(self, config: Config, output_dir: Optional[str] = None,
                 save_frames: Optional[bool] = None,
                 progress_callback=None, live_log: bool = False,
                 cancel_event: Optional[threading.Event] = None):
        self._config = config
        # 取消信号(Web端删除运行中会话时触发): 让流水线尽快安全退出
        # cancel signal (raised when a running session is deleted via web):
        # makes the pipeline exit quickly and safely.
        self._cancel = cancel_event or threading.Event()
        self._output_dir = output_dir or config.output.get("dir", "output")
        self._save_frames: bool = (
            config.output.get("save_frames", False)
            if save_frames is None else bool(save_frames)
        )
        self._progress_cb = progress_callback
        self._logger = BehaviorLogger(
            output_dir=self._output_dir,
            filename=config.output.get("log_file", "behavior_log.jsonl"),
            live_append=live_log
        )

        # 初始化各模块
        self._roi_manager = RoiManager(config.roi, self._logger)
        self._blur_detector = BlurDetector(config.blur, self._logger)
        self._visual_dedup = VisualDedup(config.visual_dedup, self._logger)
        self._ocr_client = OcrClient(config.ocr, self._logger)
        # 滚动帧低置信度过滤阈值(残影识别置信度更低)
        self._scroll_conf: float = float(config.ocr.get("scroll_confidence_threshold", 0.65))
        # OCR行级最低置信度: 真实聊天正文的PP-OCR置信度普遍>=0.95, 而气泡波形
        # 图标、滚动半截行的误识置信度明显更低(实测0.56~0.81)。用统一阈值把这类
        # 噪声挡在文本跟踪之前, 避免其被当作消息输出、或与相邻正文粘成一行。
        # Per-line OCR confidence floor: real chat text scores >=0.95 while waveform
        # icons / half-scrolled lines score far lower; drop them before tracking.
        self._ocr_min_conf: float = float(config.ocr.get("min_confidence", 0.85))
        self._text_tracker = MessageBubbleTracker(config.text_tracker, self._logger)
        self._state_machine = StateMachine(config.state_machine, self._logger)
        self._semantic_dedup = SemanticDedup(config.llm, self._logger)
        self._assembler = ContentAssembler(
            {"llm": config.llm,
             "chat_format": config.output.get("chat_format", {})},
            self._logger, self._output_dir
        )

        self._sampler: Optional[FrameSampler] = None
        self._prev_ocr_lines: Optional[List[TextLine]] = None
        self._start_time: float = 0.0
        self._last_progress_ts: float = 0.0
        self._progress_interval: float = 2.0  # 控制台进度打印间隔(秒)
        # 聊天/非聊天界面分离: 非聊天帧(转账回单/银行页/设置页/长按菜单等)的
        # OCR行整帧丢弃, 从入口挡住非聊天内容污染正文。逐帧判定结果按fid暂存,
        # 供消费线程(_consume_ocr_frame)查用; 采用有界字典防止长视频下无限增长。
        self._chat_scene_filter: bool = bool(config.roi.get("chat_scene_filter", True))
        self._chat_wall_ratio: float = float(config.roi.get("chat_wall_ratio", 0.35))
        self._frame_chat: dict = {}
        self._frame_chat_max: int = 64
        self._stats = {
            "frames_sampled": 0,
            "frames_blur_skipped": 0,
            "frames_cv_dup_skipped": 0,
            "frames_dedup_skipped": 0,
            "frames_ocr_processed": 0,
            "frames_scroll_ignored": 0,
            "frames_non_chat_skipped": 0,
            "frames_scroll_thinned": 0,
            "frames_scroll_suppressed": 0,
            "frames_scroll_sampled": 0,
            "frames_value_break": 0,
            "frames_settle_skip": 0,
            "frames_state_suspended": 0,
            "gaps_detected": 0,
            "gray_zone_count": 0,
            "llm_arbitration_count": 0,
            "llm_duplicates": 0,
            "ocr_corrected_count": 0,
            "ocr_empty_retries": 0,
            "calibration_count": 0,
            "ocr_cache_hit": 0,
            "ocr_content_reused": 0,
            "ocr_calls_made": 0,
        }

        # 断点续传: OCR结果增量落盘
        self._ocr_cache_path: Optional[str] = None
        self._ocr_cache_fh = None
        self._ocr_cache: dict = {}       # fid -> 该帧的缓存文本行
        # fid -> 该帧 OCR 行所属的 ROI 原点(x, y)。自适应 ROI 在预热结束时会把
        # 原点从 margin 基准挪到内容带(实测 (0,60) → (0,302)), 而缓存里的 bbox 是
        # **裁图当时**那一版 ROI 的局部坐标; 用单一"终版 ROI"换算所有帧会整体平移
        # 242px(见 `_roi_origin`)。
        self._frame_roi: dict = {}
        # OCR调用失败(提交/轮询失败, 或重试后仍空)的帧: 其"空结果"不可信, 不落
        # 断点缓存/内容指纹缓存, 避免一次瞬时故障被永久固化(重放命中空缓存 →
        # 整轮视觉通道静默 0 行)。
        self._ocr_failed_fids: set = set()

        # 内容指纹OCR复用: 同内容帧不重复调用OCR(视觉噪声积累的兜底)
        self._content_reuse_max_hd: int = int(config.visual_dedup.get(
            "content_reuse_max_hd", 12))   # 16x16 dHash最大汉明距离上限(0=禁用)
        self._content_lock = threading.Lock()
        self._content_entries: List[Tuple[int, int, float, List[TextLine]]] = []
        #   (hash16, fid, ts, lines): 已OCR过的唯一内容
        self._last_content: Optional[Tuple[int, int, float, List[TextLine]]] = None
        #   最近一次消费帧内容(快路径, 缓解并行滞后)

        # ---- 音频侧(与视觉流水线并行) ----
        self._audio_cfg = config.audio or {}
        self._audio_enabled = bool(self._audio_cfg.get("enabled", True)
                                   and self._audio_cfg.get("base_url"))
        self._audio_thread: Optional[threading.Thread] = None
        self._audio_result: Optional[object] = None      # TranscriptResult
        self._audio_events: List[object] = []            # List[VoiceEvent]
        self._static_obs: List[dict] = []                # 静止帧气泡原始观测
        self._scroll_curve_img: List[tuple] = []         # 图像侧滚动曲线
        self._ocr_text_hits: List[object] = []           # 正文行(质量门用)
        self._ocr_duration_hits: List[object] = []       # 时长标签(反证用)
        self._audio_error: Optional[Exception] = None

        # 抽帧诊断保存
        self._frames_dir: Optional[str] = None

        # 预热/探索阶段状态
        cal = config.calibration
        self._cal_enabled: bool = cal.get("enabled", True)
        self._cal_min_frames: int = cal.get("min_frames", 45)
        self._cal_max_frames: int = cal.get("max_frames", 150)
        self._cal_blur_floor_pct: float = cal.get("blur_floor_pct", 5.0)
        self._cal_dedup_dup_pct: float = cal.get("dedup_dup_pct", 50.0)
        self._cal_dedup_new_pct: float = cal.get("dedup_new_pct", 50.0)
        self._static_floor: float = config.sampling.get("static_floor", 2.0)
        self._warmup_count: int = 0
        self._warmup_done: bool = False
        self._warmup_pairs: List[Tuple[float, float]] = []
        self._warmup_saw_change: bool = False

        # OCR并行流水线结构
        self._ocr_concurrent: int = max(1, config.ocr.get("concurrent", 4))
        # mock模式是否允许写缓存(仅测试用; 默认False防止合成行污染真实缓存)
        self._cache_persist_mock: bool = bool(config.ocr.get("cache_persist_mock", False))
        self._submit_queue: queue.Queue = queue.Queue(
            maxsize=self._ocr_concurrent * 4)          # 待OCR帧(有界, 背压)
        self._verdicts: queue.Queue = queue.Queue()    # 每帧判定(采集→消费)
        self._pending: List[Tuple[str, int, float]] = []   # 在途API job (poller扫描)
        self._pending_lock = threading.Lock()
        self._pending_cond = threading.Condition(self._pending_lock)
        self._results: dict = {}                       # fid → List[TextLine]
        self._result_cond = threading.Condition()
        self._producer_done = threading.Event()
        self._producer: Optional[threading.Thread] = None
        self._workers: List[threading.Thread] = []
        self._poller: Optional[threading.Thread] = None
        self._ocr_expected: int = 0                    # 需要OCR的帧总数(producer统计)
        # 在途job限流令牌(防429限流突刺)
        self._inflight: int = 0
        self._inflight_limit: int = max(1, min(self._ocr_concurrent, 4))
        self._inflight_cond = threading.Condition()
        # 空结果重试: 记住最近提交帧的ROI(重提交用)
        self._pending_rois: dict = {}
        self._pending_rois_max: int = 24
        self._ocr_retries: dict = {}          # fid → 重试次数

        # 像素级强去重: 与上一张送OCR的图比较, 基本重复则跳过
        self._dedup_send_min_mae: float = config.visual_dedup.get(
            "dedup_send_min_mae", 5.0)
        self._last_ocr_gray: Optional[np.ndarray] = None
        # 滚动帧抽稀参数(见producer_loop)
        self._scroll_min_interval: float = float(
            config.sampling.get("scroll_min_interval_seconds", 0.12))
        self._last_ocr_ts: Optional[float] = None
        # settle式快照(见producer_loop): 运动段抑制OCR, 运动结束送静止帧;
        # 长滚动(超过抑制时长)按滚动抽稀粒度抽稀送检——只放行携带滚动中露出
        # 新内容的帧(快速跳变content_change与降速静止帧), 真实滚动位移
        # (scrolling)帧下游禁止新建跟踪行零收益继续抑制; 冗余由模糊/去重/
        # 内容复用兜底。EOF兜底送最后一帧保召回。
        self._settle_suppress_seconds: float = float(
            config.sampling.get("settle_suppress_seconds", 0.4))
        # 帧价值估计(通用送检决策): 运动段内以 新颖度x清晰度 连续分数替代
        # 状态硬分类——高价值帧(弹窗/toast/页面切换/滚动到新屏, 内容替换)
        # 立即打破抑制送检, 低价值帧(位移对齐后无新内容/拖影)继续抑制,
        # 不再依赖 static/scrolling/content_change 状态分类打补丁。
        self._frame_value = FrameValueEstimator(config.sampling, self._logger)
        self._value_enabled: bool = bool(
            config.sampling.get("frame_value_enabled", True))
        self._value_high_threshold: float = float(
            config.sampling.get("value_high_threshold", 0.5))
        # 滚动帧break门槛: 位移累积型内容需滚入0.75屏才视为"新图"; 内容替换
        # (弹窗/toast/切换)不走此门槛, 用value_high_threshold立即送检。
        self._value_scroll_break_novelty: float = float(
            config.sampling.get("value_scroll_break_novelty", 0.75))
        self._value_fallback_interval: float = float(
            config.sampling.get("value_fallback_interval", 0.12))
        self._value_fallback_min_novelty: float = float(
            config.sampling.get("value_fallback_min_novelty", 0.75))
        # 文本重叠反馈: 滚动帧送检后与上一送检帧的OCR行重叠率>=此值判"冗余
        # 滚动"(滚入空白/重复), 位移门槛翻倍; 低重叠=新文本滚入, 门槛重置。
        self._value_fallback_overlap_high: float = float(
            config.sampling.get("value_fallback_overlap_high", 0.6))
        self._value_fallback_novelty_max: float = float(
            config.sampling.get("value_fallback_novelty_max", 1.0))
        # 动态位移门槛(文本反馈自适应): 初始=基线, 冗余滚动时翻倍(封顶max),
        # 新文本时重置回基线。
        self._scroll_fb_novelty: float = self._value_fallback_min_novelty
        # settle门控: 滚动结束的settle静止帧距最后送检<此间隔(秒)时跳过OCR
        # (最后送检帧内容刚覆盖屏幕, settle帧与其大量重复, 如滚动减速停止
        # 阶段; 实测f0156→f0174相距0.34s滚入约0.2屏, 文本重叠率>0.9);
        # 长滚动整段被抑制时距离远(实测f0185→f0255相距1.32s滚入0.69屏),
        # 仍送settle帧兜底召回。0=禁用。
        self._settle_skip_interval: float = float(
            config.sampling.get("settle_skip_interval", 0.5))
        # 滚动长时兜底: 滚动状态连续无送检超过此间隔(秒)时, 强制送检当前
        # 滚动帧。慢滚动位移长期不足位移门槛(实测f0185→f0255段1.3s累计
        # 仅0.69屏<0.75门槛)会导致整段零送检, 全靠settle帧兜底会漏中间
        # 新消息/toast; 兜底帧照常走文本重叠反馈(冗余滚动提高门槛, 新文本
        # 重置), 不破坏内容驱动节奏。0=禁用。
        self._scroll_guard_interval: float = float(
            config.sampling.get("scroll_guard_interval", 0.7))
        # 滚动兜底的内容门槛: 兜底送检仅在当前帧确有新内容(novelty>=此值)时
        # 触发, 防纯时间兜底抢先于自然break并清零累计位移(实测f0132新颖度
        # 0.22被拦, f0156保持0.76屏break; 慢滚动f0223新颖度0.41仍兜底)。
        self._scroll_guard_min_novelty: float = float(
            config.sampling.get("scroll_guard_min_novelty", 0.3))
        # 滚动兜底的速度门槛: 只兜底慢滚动(帧间位移小=缓慢渗出), 快速滚动帧
        # 不兜底, 否则时间兜底推进参考帧会抢先抑制自然break(实测f137 dy=4 /
        # f139 dy=6 不兜底, f0156保持0.76屏break; f0223 dy=2 仍兜底)。
        self._scroll_guard_max_dy: float = float(
            config.sampling.get("scroll_guard_max_dy", 2.0))
        self._settle_motion: bool = False            # 是否处于运动段
        self._settle_motion_start_ts: float = 0.0    # 运动段起始时间
        self._settle_last_fd: Optional[FrameData] = None  # 最近运动帧(EOF兜底)
        self._settle_suppressed_count: int = 0       # 本运动段已抑制帧数
        # 状态恢复时keyframe重置: 消费线程请求, 采集线程用灰度缓存执行
        self._gray_cache: dict = {}
        self._gray_cache_size: int = 64
        self._pending_resets: List[int] = []
        self._reset_lock = threading.Lock()
        self._stats_lock = threading.Lock()

    def run(self, video_path: str) -> dict:
        """
        运行完整流水线

        参数:
            video_path: 视频文件路径
        返回:
            结构化提取结果
        """
        start_time = time.time()
        self._start_time = start_time
        print(f"[进度] 开始处理视频: {video_path}", flush=True)
        self._logger.log("pipeline", "start",
                         f"开始处理视频: {video_path}")
        self._emit("stage", stage="opening", detail="打开视频并检测ROI",
                   total_frames=self._sampler.total_frames if self._sampler else None)

        # 打开视频
        self._sampler = FrameSampler(
            self._config.sampling, (0, 0, 0, 0), self._logger
        )
        if not self._sampler.open(video_path):
            self._logger.log("pipeline", "error",
                             f"无法打开视频: {video_path}")
            return {"error": "无法打开视频 / Cannot open video"}

        # 检测ROI(使用第一帧)
        first_frame = self._sampler.peek_first_frame()
        if first_frame is not None:
            roi_box = self._roi_manager.detect_roi(first_frame)
            self._sampler.set_roi_box(roi_box)
            self._text_tracker.set_roi_dimensions(roi_box[3], roi_box[2])
            self._assembler.set_roi_width(roi_box[2])
            self._assembler.set_roi_height(roi_box[3])
            print(f"[进度] ROI已检测: {roi_box}", flush=True)
            self._logger.log("pipeline", "roi_detected",
                             f"ROI: {roi_box}")
            self._emit("stage", stage="roi", detail=f"ROI: {roi_box}",
                       roi=list(roi_box))
        else:
            self._logger.log("pipeline", "error", "无法读取首帧 / Cannot read first frame")
            return {"error": "无法读取首帧 / Cannot read first frame"}

        # 断点续传: 加载已缓存的OCR结果
        self._init_ocr_cache(video_path)

        # 抽帧诊断保存: 初始化采样帧输出目录
        self._init_frames_dir(video_path)

        # 进入预热/探索阶段: 收集统计量, 校准各模块阈值
        if self._cal_enabled:
            print("[进度] 预热探索阶段开始(收集统计量, 校准阈值)...", flush=True)
            self._logger.log("pipeline", "warmup_start",
                             "预热探索开始 / Warm-up started")
            self._emit("stage", stage="warmup", detail="预热校准(45~150帧密集采样分布校准阈值)")

        # 音频侧: 后台线程并行转写(解码+VAD+ASR, 与视觉OCR重叠进行, 不拖慢墙钟)
        if self._audio_enabled:
            self._start_audio_background(video_path)

        # 主处理循环(并行流水线)
        self._run_pipeline()

        # 被取消(Web 删除会话): 提前退出, 跳过后续的气泡扫描/后处理/音频合并
        # cancelled (session deleted via web): bail early
        if self._cancel.is_set():
            self._logger.log("pipeline", "cancelled", "任务已取消(会话被删除)")
            self._emit("audio_progress", phase="error", done=1, total=1,
                       detail="任务已取消")
            return {"error": "cancelled", "cancelled": True}

        # 音频侧: 气泡扫描(第二遍抽样帧, 使用预热后终版ROI) —— 在视觉循环结束后执行
        if self._audio_enabled:
            try:
                self._scan_voice_bubbles(video_path)
            except Exception as e:
                self._logger.log("pipeline", "warning",
                                 f"语音气泡扫描失败: {e}")
                self._audio_error = e

        # 后处理
        print("[进度] 帧处理完成, 进入后处理(组装/LLM增强)...", flush=True)
        self._logger.log("pipeline", "postprocessing", "开始后处理 / Starting post-processing")
        self._emit("stage", stage="postprocessing", detail="后处理: 文本跟踪收尾/语义去重/内容组装")
        self._assembler.collect_tracked(self._text_tracker)
        # 注入帧提供器: 供气泡合并阶段做像素验证(大间距行距是否同一气泡)
        self._assembler.set_frame_provider(
            lambda fid: self._sampler.read_roi_frame(fid)
            if self._sampler else None
        )
        result = self._assembler.assemble()

        # 音频侧: 等待后台转写线程, 合并转写+配对进最终结果
        if self._audio_enabled:
            try:
                self._merge_audio_into_result(result)
            except Exception as e:
                # 合并异常时绝不用空 stub 覆盖已拿到的转写: 保留完整 transcript,
                # 仅把失败原因写进 audio.reason, 见 M2 修复。
                self._logger.log("pipeline", "warning", f"音频合并失败: {e}")
                if hasattr(self, "_audio_result") and self._audio_result is not None:
                    try:
                        result["audio"] = self._audio_result.to_dict()
                        result["audio"]["reason"] = (
                            (result["audio"].get("reason", "") or "") + f" 合并失败: {e}").strip()
                    except Exception:
                        result["audio"] = {"enabled": True, "reason": f"合并失败: {e}",
                                           "segments": [], "has_audio": False}
                else:
                    result["audio"] = {"enabled": True, "reason": f"合并失败: {e}",
                                       "segments": [], "has_audio": False}

        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        result_path = self._assembler.save(result, video_name=video_stem)

        # 保存行为日志
        log_path = self._logger.save()

        # 关闭断点续传缓存句柄
        if self._ocr_cache_fh is not None:
            self._ocr_cache_fh.flush()
            self._ocr_cache_fh.close()
            self._ocr_cache_fh = None

        # 添加运行统计
        elapsed = time.time() - start_time
        gray_stats = self._visual_dedup.gray_zone_stats
        sem_stats = self._semantic_dedup.stats

        result["run_info"] = {
            "video_path": video_path,
            "elapsed_seconds": round(elapsed, 2),
            "fps": self._sampler.fps,
            "total_video_frames": self._sampler.total_frames,
            "sampled_frames": self._stats["frames_sampled"],
            "blur_skipped": self._stats["frames_blur_skipped"],
            "cv_dup_skipped": self._stats["frames_cv_dup_skipped"],
            "dedup_skipped": self._stats["frames_dedup_skipped"],
            "ocr_processed": self._stats["frames_ocr_processed"],
            "scroll_ignored": self._stats["frames_scroll_ignored"],
            "scroll_thinned": self._stats["frames_scroll_thinned"],
            "scroll_suppressed": self._stats["frames_scroll_suppressed"],
            "scroll_sampled": self._stats["frames_scroll_sampled"],
            "value_break": self._stats["frames_value_break"],
            "ocr_concurrent": self._ocr_concurrent,
            "ocr_cached_frames": self._stats["ocr_cache_hit"],
            "ocr_content_reused": self._stats["ocr_content_reused"],
            "ocr_content_reuse_max_hd": self._content_reuse_max_hd,
            "ocr_calls_made": self._stats["ocr_calls_made"],
            "ocr_cache_path": self._ocr_cache_path,
            "suspended_frames": self._stats["frames_state_suspended"],
            "gaps_detected": self._stats["gaps_detected"],
            "gray_zone_count": self._stats["gray_zone_count"],
            "gray_zone_ratio": gray_stats["gray_zone_ratio"],
            "llm_arbitration_count": self._stats["llm_arbitration_count"],
            "llm_duplicates": sem_stats["llm_duplicates"],
            "ocr_corrected_count": self._stats["ocr_corrected_count"],
            "ocr_empty_retries": self._stats["ocr_empty_retries"],
            "calibration_count": self._stats["calibration_count"],
            "calibrated_dup_threshold": round(self._visual_dedup.current_dup_threshold, 1),
            "calibrated_new_threshold": round(self._visual_dedup.current_new_threshold, 1),
            "warmup_enabled": self._cal_enabled,
            "warmup_frames": self._warmup_count if self._cal_enabled else 0,
            "calibrated_t1": round(self._sampler.t1, 2),
            "calibrated_t2": round(self._sampler.t2, 2),
            "calibrated_blur_floor": round(self._blur_detector.min_absolute, 1),
            "behavior_log_path": log_path,
            "output_path": result_path,
        }

        self._logger.log("pipeline", "complete",
                         f"处理完成: 耗时{elapsed:.1f}s, "
                         f"采样{self._stats['frames_sampled']}帧, "
                         f"OCR{self._stats['frames_ocr_processed']}帧, "
                         f"灰色地带{self._stats['gray_zone_count']}帧, "
                         f"LLM仲裁{self._stats['llm_arbitration_count']}次")

        self._emit("stage", stage="done", detail="提取完成", elapsed=elapsed)
        self._sampler.close()
        return result

    # ============================ 音频侧辅助方法 ============================

    def _build_asr_client(self) -> AsrClient:
        """从配置构建 ASR 客户端。"""
        cfg = self._audio_cfg
        return AsrClient(
            base_url=cfg["base_url"],
            api_key=cfg.get("api_key", ""),
            model=cfg.get("model", "sensevoice-small"),
            max_request_bytes=max(4 * 1024 * 1024, int(
                self._safe_float(cfg, "max_request_bytes", 16 * 1024 * 1024))),
            timeout=self._safe_float(cfg, "timeout", 600.0),
            max_retries=int(cfg.get("max_retries", 3)),
        )

    @staticmethod
    def _safe_float(d: dict, key: str, default: float) -> float:
        try:
            return float(d.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _start_audio_background(self, video_path: str) -> None:
        """后台线程并行做音频转写(解码→VAD→ASR), 与视觉OCR重叠进行。"""
        if self._audio_thread and self._audio_thread.is_alive():
            return
        if not has_audio_track(video_path):
            self._audio_enabled_transcribe = False
            return
        self._audio_enabled_transcribe = True
        client = self._build_asr_client()
        sr = int(self._audio_cfg.get("sample_rate", 16000))
        max_seg = self._safe_float(self._audio_cfg, "max_segment_seconds", 360.0)
        concurrency = max(1, int(self._audio_cfg.get("concurrency", 2)))

        # 先报告"音频处理开始(VAD 阶段)"让前端能立即显示音频进度条
        # Report audio-phase start first so the UI can show the audio bar at once
        self._emit("audio_progress", phase="vad", done=0, total=0,
                   detail="分析语音段")

        def _worker():
            try:
                self._audio_result = transcribe_media(
                    video_path, client, sample_rate=sr,
                    max_segment_seconds=max_seg,
                    max_request_bytes=getattr(client, "max_request_bytes", None),
                    concurrency=concurrency,
                    # 删除会话时取消: 不再调度剩余 ASR 段, 避免占用 ASR/GPU
                    is_cancelled=lambda: self._cancel.is_set(),
                    # VAD 分段后逐段 ASR 的进度(done/total = 已转写段数/总段数)
                    # per-segment ASR progress (done/total = transcribed/total segs)
                    progress=lambda d, t: self._emit(
                        "audio_progress", phase="transcribing",
                        done=int(d), total=int(t),
                        detail=f"语音转文字 {int(d)}/{int(t)}"),
                )
                self._emit("audio_progress", phase="done", done=1, total=1,
                           detail="音频转写完成")
            except Exception as e:  # 后台线程不允许抛出, 收进 _audio_error 避免主线程崩溃
                self._audio_error = e
                self._audio_result = TranscriptResult(
                    source=os.path.basename(video_path),
                    has_audio=True, reason=f"转写异常: {e}")
                self._emit("audio_progress", phase="error", done=1, total=1,
                           detail=f"音频转写失败: {str(e)[:120]}")

        self._audio_thread = threading.Thread(target=_worker, daemon=True)
        self._audio_thread.start()

    def _scan_voice_bubbles(self, video_path: str) -> None:
        """用终版ROI对视频做第二遍抽样, 收集语音气泡出现事件(配对锚点)。

        视觉形状检测 + OCR时长标签信号混合: OCR(主, 高召回)兜底视觉漏检。
        """
        cfg = self._audio_cfg
        if not cfg.get("bubble_enabled", True):
            return
        roi_box = (self._roi_manager.roi_box
                   if getattr(self._roi_manager, "roi_box", None)
                   else None)
        self._audio_events = []
        if not roi_box:
            return
        scanner = VoiceBubbleScanner(
            roi_box=tuple(int(v) for v in roi_box),
            # 默认 0.55 是针对旧渲染(波形条)调的; 新视觉拟真 UI 下真实气泡
            # voice_score 集中在 0.50~0.55(喇叭图标+真实颜色), 再用 0.55 就会
            # 把首条滞后 10s 以上(见 v3_realui vs v3_synth 的首事件时间差)。
            # 降到 0.45 留冗余, 同时保留低置信度 UI chrome 的抑制能力。
            voice_threshold=self._safe_float(cfg, "bubble_threshold", 0.45),
            sample_interval_s=self._safe_float(
                cfg, "bubble_sample_interval_s", 0.2),
        )
        vision_events = scanner.scan(video_path)
        # 静止帧的逐帧原始检测: 供 `cluster_static_observations` 在合并阶段按文档
        # 坐标重建气泡清单(跨帧身份跟踪在滚动式录屏上会错接, 见该函数说明)。
        self._static_obs = list(scanner.static_observations)
        self._scroll_curve_img = list(scanner.scroll_curve)
        # OCR时长标签仅作框内佐证(不在视觉清单外独立计数),
        # 避免全局 mm:ss 把界面时钟/进度误判为语音气泡。
        ocr_hits = []
        text_hits = []
        try:
            ocr_scanner = OcrVoiceScanner(
                frame_width=getattr(self._roi_manager, "_frame_size", (890, 1920))[0])
            # OCR 缓存的 bbox 是 ROI 局部坐标, 视觉气泡框是原始帧坐标; 传入 ROI
            # 左上角偏移, 把标签框换算到原始帧坐标系后再做框内佐证(否则 y 恒定
            # 错开 ROI.y, 时长标签几乎永远匹配不上)。
            roi_offset = (int(roi_box[0]), int(roi_box[1]))
            ocr_hits = ocr_scanner.scan_duration_hits(
                self._ocr_cache_path, roi_offset=roi_offset)
            # 正文行(非时长标签)用于"框内含正文 → 判为文本气泡"的质量门:
            # 检测器阈值被降到 0.45 后, 短文本气泡也会混进语音气泡清单, 按顺序
            # 插入后会把 音频段↔气泡 的位置配对整体错位。
            text_hits = ocr_scanner.scan_text_hits(
                self._ocr_cache_path, roi_offset=roi_offset)
        except Exception as e:
            self._logger.log("pipeline", "warning",
                             f"OCR语音气泡时长标签扫描失败: {e}")
        vision_events = drop_text_bubbles(vision_events, text_hits,
                                         duration_hits=ocr_hits)
        self._ocr_text_hits = list(text_hits)
        self._ocr_duration_hits = list(ocr_hits)
        self._audio_events = merge_ocr_and_vision(
            vision_events, ocr_hits,
            media_duration_s=probe_media_duration(video_path))
        if os.environ.get("DEBUG_VOICE_EVENTS"):
            with open(os.path.join(self._output_dir, "voice_events.json"), "w",
                      encoding="utf-8") as f:
                json.dump([{
                    "first": e.timestamp, "last": e.last_timestamp,
                    "box": e.box, "last_box": e.last_box,
                    "side": e.side, "source": e.source,
                } for e in self._audio_events], f, ensure_ascii=False, indent=1)

    def _merge_audio_into_result(self, result: dict) -> None:
        """等待后台转写线程结束, 把转写结果+气泡配对合并进结果 dict。

        被取消时不长等(2s 内结束), 且不把结果写入(会话已快速删除)。
        """
        if self._audio_thread:
            timeout = 2.0 if self._cancel.is_set() else 1800.0
            self._audio_thread.join(timeout=timeout)
        if self._cancel.is_set():
            return
        transcript = self._audio_result
        if transcript is None:
            transcript = TranscriptResult(
                source="", has_audio=False,
                reason=str(self._audio_error) if self._audio_error else "音频未执行")
        if self._audio_error:
            transcript.reason = (transcript.reason or "") + f" (后台异常: {self._audio_error})"
        # 视觉行锚点: 首见时刻 + 首见纵坐标(换算到原始帧坐标), 用于把语音行按
        # 聊天顺序插入视觉行之间。OCR bbox 为 ROI 局部坐标, 气泡框是原始帧坐标,
        # 必须加 ROI 顶部偏移, 否则两套坐标不在同一基准上。
        # `doc` = 文档纵坐标(屏幕 y + 该时刻累计滚动偏移): 聊天滚动后行会整体上移,
        # 只比较屏幕 y 会把"先出现但已滚到顶部"的行误判成最新消息, 必须消掉滚动量。
        tracked = self._assembler.get_tracked_lines()
        # 滚动偏移曲线取自**逐帧 OCR 原始行**(按文本跨帧配对): 跟踪行在滚动后会
        # 重新建簇(ID 切换), 用它累计出的偏移只有真实值的一半左右。
        # 缓存行的 bbox 是**各自裁图时那一版 ROI** 的局部坐标(预热期与稳态期的 ROI
        # 原点不同), 先按帧还原到原始帧坐标再喂给曲线; 否则 ROI 切换点会凭空多出
        # 一段假位移(实测 242px), 使文档坐标排序整体错乱。
        ocr_frames = []
        for fid, lines in self._ocr_cache.items():
            if not lines:
                continue
            ox, oy = self._roi_origin(fid)
            if ox or oy:
                lines = [TextLine(text=l.text,
                                  bbox=(l.bbox[0] + ox, l.bbox[1] + oy,
                                        l.bbox[2], l.bbox[3]),
                                  confidence=l.confidence,
                                  frame_id=l.frame_id, timestamp=l.timestamp)
                         for l in lines]
            ocr_frames.append((float(lines[0].timestamp), lines))
        ocr_frames.sort(key=lambda x: x[0])
        # 图像侧帧级运动作为 OCR 单行配对不可信时的兜底(见 build_scroll_curve)。
        scroll_curve = build_scroll_curve(
            ocr_frames, image_curve=self._scroll_curve_img or None)
        # 气泡清单(顺序对齐用): 由静止帧原始观测按文档坐标重建, 绕开滚动期的
        # 跨帧身份错接; 再用与事件相同的正文质量门剔除文本气泡。
        order_bubbles: List[object] = []
        try:
            clusters = cluster_static_observations(self._static_obs, scroll_curve)
            order_bubbles = drop_narrow_bubbles(drop_text_bubbles(
                clusters, self._ocr_text_hits,
                duration_hits=self._ocr_duration_hits,
                scroll_curve=scroll_curve))
        except Exception as e:
            self._logger.log("pipeline", "warning",
                             f"静止观测聚类失败: {e}")
        if os.environ.get("DEBUG_ORDER_BUBBLES"):
            with open(os.path.join(self._output_dir, "order_bubbles.json"), "w",
                      encoding="utf-8") as f:
                json.dump([{
                    "t": ev.timestamp,
                    # x 必须落盘: 正文质量门的 `_center_within` 同时判 x/y, 只看
                    # y/w/h 无法复现"框住了正文却没被剔除"的判定(横向不匹配)。
                    "x": ev.box[0] if getattr(ev, "box", None) else None,
                    "y": ev.box[1] if getattr(ev, "box", None) else None,
                    "w": ev.box[2] if getattr(ev, "box", None) else None,
                    "h": ev.box[3] if getattr(ev, "box", None) else None,
                    "side": ev.side,
                    "conf": ev.confidence,
                    "doc": (ev.stable_box[1] + scroll_offset_at(
                        scroll_curve, ev.stable_timestamp))
                    if getattr(ev, "stable_box", None) else None,
                } for ev in order_bubbles], f, ensure_ascii=False, indent=1)
        visual_lines = []
        for ln in tracked:
            box = ln.first_bbox or ln.bbox
            if not box:
                continue
            # 语音气泡内的时长标签(如 6"…)会被 OCR 当正文行识别; 它不是消息,
            # 进入锚点集合会把语音行插到它旁边(实测 doc 2388 造成整体错位)。
            if parse_duration_seconds(ln.text) is not None:
                continue
            t0 = float(ln.first_timestamp)
            # 跟踪行的 bbox 属于它**首见帧**那一版 ROI 的局部坐标(跨 ROI 切换时以
            # 首见为准, 与 `box = first_bbox or bbox` 的取值口径一致)。
            y0 = float(box[1]) + self._roi_origin(
                ln.first_seen_frame if ln.first_bbox
                else ln.last_seen_frame)[1]
            visual_lines.append({"text": ln.text, "t": t0, "y": y0,
                                 "doc": y0 + scroll_offset_at(scroll_curve, t0)})
        if os.environ.get("DEBUG_SCROLL_CURVE"):
            with open(os.path.join(self._output_dir, "scroll_curve.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"curve": scroll_curve,
                           "frame_times": [t for t, _ in ocr_frames],
                           "visual": visual_lines},
                          f, ensure_ascii=False, indent=1)
        merge_into_result(
            result, transcript, self._audio_events,
            max_match_gap_s=self._safe_float(self._audio_cfg, "match_gap_s", 1.5),
            inject_chat=bool(self._audio_cfg.get("inject_chat", True)),
            order_align=str(self._audio_cfg.get("order_align", "auto")),
            visual_lines=visual_lines,
            scroll_curve=scroll_curve,
            order_bubbles=order_bubbles,
        )

    def _init_frames_dir(self, video_path: str) -> None:
        """初始化抽帧保存目录"""
        self._frames_dir = None
        if not self._save_frames:
            return
        try:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            self._frames_dir = os.path.join(
                self._output_dir, "frames", stem
            )
            os.makedirs(self._frames_dir, exist_ok=True)
            self._logger.log("pipeline", "frames_dir",
                             f"抽帧保存目录 / frames dir: {self._frames_dir}")
            print(f"[抽帧] 采样帧将保存到 {self._frames_dir}", flush=True)
        except OSError as e:
            self._logger.log("pipeline", "frames_dir_error",
                             f"抽帧目录创建失败 / mkdir failed: {e}")
            self._frames_dir = None

    def _save_sampled_frame(self, fd: FrameData, verdict: str) -> None:
        """保存采样帧ROI图(诊断用)
        dup/blur帧与上一张OCR帧近乎相同, 不写图像文件(仅写索引), 避免磁盘
        堆积大量重复诊断图。
        """
        if not self._frames_dir:
            return
        try:
            fname = (f"f{fd.frame_id:04d}_t{fd.timestamp:.2f}s_"
                     f"{fd.sampling_state}_{verdict}.jpg")
            if verdict not in ("dup", "blur", "thinned", "suppressed",
                               "settle_skip"):
                ok, buf = cv2.imencode(
                    ".jpg", fd.roi_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if ok:
                    with open(os.path.join(self._frames_dir, fname), "wb") as f:
                        f.write(buf.tobytes())
            with open(os.path.join(self._frames_dir, "frames_index.jsonl"),
                      "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "frame_id": fd.frame_id,
                    "timestamp": round(fd.timestamp, 3),
                    "state": fd.sampling_state,
                    "mae": round(fd.mae_value, 2),
                    "verdict": verdict,
                    "file": fname,
                }, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _init_ocr_cache(self, video_path: str) -> None:
        """初始化断点续传缓存(按视频指纹命名)"""
        if not self._output_dir:
            return
        try:
            cache_dir = os.path.join(self._output_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            st = os.stat(video_path)
            fingerprint = hashlib.sha1(
                f"{os.path.basename(video_path)}|{st.st_size}|{int(st.st_mtime)}"
                .encode()
            ).hexdigest()[:16]
            stem = os.path.splitext(os.path.basename(video_path))[0]
            self._ocr_cache_path = os.path.join(
                cache_dir, f"{stem}_{fingerprint}.jsonl"
            )
            if os.path.exists(self._ocr_cache_path):
                self._load_ocr_cache()
        except OSError as e:
            self._logger.log("pipeline", "cache_init_error",
                             f"缓存初始化失败 / cache init failed: {e}")
            self._ocr_cache_path = None

    def _load_ocr_cache(self) -> None:
        """加载已有OCR缓存(断点续传, 含内容指纹条目)

        逐行容错: 单条损坏只跳过该条, 不影响其余条目(防止一条坏数据
        导致整个缓存被清空后重新用mock数据污染文件)。
        """
        count = 0
        content_count = 0
        try:
            with open(self._ocr_cache_path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        self._logger.log(
                            "pipeline", "cache_load_skip",
                            f"缓存第{lineno}行JSON损坏, 跳过该行 / corrupt line skipped")
                        continue
                    # 该条目的 bbox 所属 ROI 原点(旧缓存无此字段 → 回退当前 ROI)
                    roi = item.get("roi")
                    if roi and len(roi) == 2:
                        try:
                            self._frame_roi[int(item["fid"])] = (
                                int(roi[0]), int(roi[1]))
                        except (KeyError, TypeError, ValueError):
                            pass
                    if item.get("kind") == "content":
                        # 内容指纹条目: 重建内容缓存(供同内容帧复用)
                        try:
                            h = int(item["hash"])
                            lines = [
                                TextLine(
                                    text=t["text"],
                                    bbox=tuple(t["bbox"]),
                                    confidence=t.get("confidence", 1.0),
                                    frame_id=int(item["fid"]),
                                    timestamp=t.get("timestamp", item.get("ts", 0.0)),
                                )
                                for t in item["lines"]
                            ]
                        except (KeyError, TypeError, ValueError):
                            self._logger.log(
                                "pipeline", "cache_load_skip",
                                f"缓存第{lineno}行内容条目损坏, 跳过 / corrupt content entry")
                            continue
                        with self._content_lock:
                            self._content_entries.append(
                                (h, int(item["fid"]), item.get("ts", 0.0), lines)
                            )
                        content_count += 1
                        continue
                    try:
                        fid = int(item["fid"])
                        lines = [
                            TextLine(
                                text=t["text"],
                                bbox=tuple(t["bbox"]),
                                confidence=t.get("confidence", 1.0),
                                frame_id=fid,
                                timestamp=t.get("timestamp", item.get("ts", 0.0)),
                            )
                            for t in item["lines"]
                        ]
                    except (KeyError, TypeError, ValueError):
                        self._logger.log(
                            "pipeline", "cache_load_skip",
                            f"缓存第{lineno}行帧条目损坏, 跳过 / corrupt frame entry")
                        continue
                    self._ocr_cache[fid] = lines
                    count += 1
            if count > 0 or content_count > 0:
                self._logger.log("pipeline", "cache_load",
                                 f"加载OCR缓存{count}帧, 内容指纹{content_count}条 / "
                                 f"Loaded {count} cached frames, {content_count} contents")
                print(f"[断点续传] 加载OCR缓存 {count} 帧, 内容指纹 {content_count} 条, "
                      f"将跳过这些帧的OCR调用", flush=True)
        except Exception as e:
            self._logger.log("pipeline", "cache_load_error",
                             f"缓存加载失败 / cache load failed: {e}")
            self._ocr_cache = {}

    def _roi_origin(self, fid: int) -> Tuple[int, int]:
        """该帧 OCR 行的 bbox 所属 ROI 原点 (x, y); 未记录时回退到当前 ROI。

        为什么不能统一用终版 ROI / Why not one global offset:
          自适应 ROI 在预热结束时把原点从 margin 基准挪到内容带(实测
          `(0,60,576,1100)` → `(0,302,576,808)`), 而 OCR 缓存里的 bbox 是**裁图
          当时**那一版 ROI 的局部坐标 —— 同一个文件里混着两套坐标系。用终版原点
          换算全部帧, 会把预热期帧整体平移 242px: 正文质量门因此一个候选都剔不掉
          (实测 `自行模拟对话记录4.mp4`: 10 个候选气泡全数保留, 其中 3 个是文本
          气泡), 滚动曲线在切换点也凭空多出一段 242px 的假位移, 文档坐标排序随之
          错乱(首屏文本行被排到语音行之后)。
        """
        r = self._frame_roi.get(fid)
        if r:
            return (int(r[0]), int(r[1]))
        box = getattr(self._roi_manager, "roi_box", None)
        return (int(box[0]), int(box[1])) if box else (0, 0)

    def _write_ocr_cache(self, fid: int, ts: float,
                         lines: List[TextLine]) -> None:
        """增量写OCR结果到缓存

        mock模式为测试模式, 合成行(mock_line_xxx)绝不写入缓存, 否则会污染
        真实缓存, 导致后续重放/断点续传拿到垃圾文本。
        """
        if self._ocr_client.mode == "mock" and not self._cache_persist_mock:
            return
        if (any((t.text or "").startswith("mock_line") for t in lines)
                and not self._cache_persist_mock):
            return
        if self._ocr_cache_path is None:
            return
        try:
            if self._ocr_cache_fh is None:
                self._ocr_cache_fh = open(
                    self._ocr_cache_path, "a", encoding="utf-8"
                )
            item = {
                "fid": fid,
                "ts": round(ts, 3),
                # bbox 所属的 ROI 原点: 缓存跨预热期/稳态期混用两套 ROI 局部坐标,
                # 读取方必须按帧还原(见 `_roi_origin`)。
                "roi": list(self._roi_origin(fid)),
                "lines": [{
                    "text": t.text,
                    "bbox": list(t.bbox),
                    "confidence": round(t.confidence, 4),
                    "timestamp": round(t.timestamp, 3),
                } for t in lines],
            }
            self._ocr_cache_fh.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )
            self._ocr_cache_fh.flush()
        except OSError as e:
            self._logger.log("pipeline", "cache_write_error",
                             f"缓存写入失败 / cache write failed: {e}")

    def _write_content_cache(self, h: int, fid: int, ts: float,
                             lines: List[TextLine]) -> None:
        """增量写内容指纹到缓存

        同 _write_ocr_cache: mock模式的合成行绝不持久化。
        """
        if self._ocr_client.mode == "mock" and not self._cache_persist_mock:
            return
        if (any((t.text or "").startswith("mock_line") for t in lines)
                and not self._cache_persist_mock):
            return
        if self._ocr_cache_path is None:
            return
        try:
            if self._ocr_cache_fh is None:
                self._ocr_cache_fh = open(
                    self._ocr_cache_path, "a", encoding="utf-8"
                )
            item = {
                "kind": "content",
                "hash": h,
                "fid": fid,
                "ts": round(ts, 3),
                # 同 _write_ocr_cache: 记录 bbox 所属的 ROI 原点
                "roi": list(self._roi_origin(fid)),
                "lines": [{
                    "text": t.text,
                    "bbox": list(t.bbox),
                    "confidence": round(t.confidence, 4),
                    "timestamp": round(t.timestamp, 3),
                } for t in lines],
            }
            self._ocr_cache_fh.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )
            self._ocr_cache_fh.flush()
        except OSError as e:
            self._logger.log("pipeline", "cache_write_error",
                             f"内容指纹缓存写入失败 / content cache write failed: {e}")

    def _find_content_reuse(self, h16: int) -> Optional[Tuple[int, int, int, List[TextLine]]]:
        """
        在内容缓存中查找hd16不超过容差的条目
        返回(源fid, 源ts, hd16, 行列表); 无匹配返回None
        """
        best_fid, best_ts, best_hd, best_lines = None, None, None, None
        with self._content_lock:
            candidates = list(self._content_entries)
            if self._last_content is not None:
                candidates.append(self._last_content)
            for h, fid, ts, lines in candidates:
                d = hamming_distance(h16, h)
                if best_hd is None or d < best_hd:
                    best_hd, best_fid, best_ts, best_lines = d, fid, ts, lines
        if best_fid is None or best_hd > self._content_reuse_max_hd:
            return None
        return best_fid, best_ts, best_hd, best_lines

    def _update_content_cache(self, h16: int, fid: int, ts: float,
                              lines: List[TextLine]) -> None:
        """消费帧后更新内容指纹缓存(供后续同内容帧复用)
        """
        if self._content_reuse_max_hd <= 0 or h16 is None:
            return
        with self._content_lock:
            for h, _, _, _ in self._content_entries:
                if hamming_distance(h16, h) <= self._content_reuse_max_hd:
                    # 已存在同内容条目, 只刷新快路径
                    self._last_content = (h16, fid, ts, list(lines))
                    return
            self._content_entries.append((h16, fid, ts, list(lines)))
            if len(self._content_entries) > 512:   # 限长, 只保留最近
                self._content_entries = self._content_entries[-256:]
            self._last_content = (h16, fid, ts, list(lines))
        self._write_content_cache(h16, fid, ts, lines)

    def _run_pipeline(self) -> None:
        """
        启动并行流水线并等待全部完成

        并行架构:
          producer(采集线程) → submit_queue → worker池(提交OCR job) → poller(统一轮询)
          producer → verdicts(帧判定) → consumer(主线程, 按帧序消费OCR结果)
        """
        n = self._ocr_concurrent
        if self._ocr_client.mode != "api":
            n = 1  # 本地/mock模式: 本地OCR不可并发, 退化为单worker
        self._workers = [
            threading.Thread(target=self._ocr_worker, daemon=True,
                             name=f"ocr-worker-{i}")
            for i in range(n)
        ]
        for w in self._workers:
            w.start()
        if self._ocr_client.mode == "api":
            self._poller = threading.Thread(target=self._poller_loop,
                                            daemon=True, name="ocr-poller")
            self._poller.start()
        self._producer = threading.Thread(target=self._producer_loop,
                                          daemon=True, name="producer")
        self._producer.start()

        # 主线程按帧序消费
        self._consumer_loop()

        # 取消时不再等待各后台线程(demon; 让它们自然退出, 避免删除会话时卡住)
        # skip joins on cancel: threads are daemon, avoid blocking a deleted session
        if self._cancel.is_set():
            return
        self._producer.join()
        for w in self._workers:
            w.join()
        if self._poller is not None:
            self._poller.join()

    def _is_motion_fd(self, fd: FrameData) -> bool:
        """
        运动段判定

        相位相关启用时双信号融合: pc判"内容在移动"(位移/内容替换), 绝对像素
        地板补pc盲区——慢速滚动(帧间位移<1px但残影使像素漂移)被pc读成static,
        MAE>=static_floor(唯一绝对常数)则不漏。两者OR进运动段, 静止帧(MAE<2
        且pc=static)不受影响。禁用时回退MAE分位数猜状态。
        """
        if self._sampler.pc_enabled:
            return (fd.pc_state != "static"
                    or fd.mae_value >= self._static_floor)
        return fd.mae_value >= self._sampler.t1

    def _is_scroll_fd(self, fd: FrameData) -> bool:
        """
        滚动闸门判定

        只有真实滚动位移(pc_state=scrolling)才触发滚动抽稀/残影过滤;
        内容替换(content_change)是"静止后完整新内容", 立即送OCR不抽稀。
        """
        if self._sampler.pc_enabled:
            return fd.pc_state == "scrolling"
        return fd.mae_value >= self._sampler.t1

    def _value_diag(self, fv: dict) -> dict:
        """
        帧价值诊断透传

        把 estimate() 返回的评分中间量并入决策日志 extra, 一条日志即可还原
        "为何送检/为何抑制"的完整路径(弹窗/toast/切换/新屏排查用)。
        """
        d = fv.get("diag", {})
        return {
            "novelty_src": d.get("novelty_src"),
            "pc_state": d.get("pc_state"),
            "pc_peak": d.get("pc_peak"),
            "pc_dy": d.get("pc_dy"),
            "est_shift": d.get("est_shift"),
            "replace_evidence": d.get("replace_evidence"),
            "novelty_branch": d.get("novelty_branch"),
            "gde": d.get("gde"),
            "short_circuit": d.get("short_circuit"),
            "best_shift": d.get("best_shift"),
            "best_mae": d.get("best_mae"),
            "new_strips": d.get("new_strips"),
            "strip_total": d.get("strip_total"),
        }

    def _producer_loop(self) -> None:
        """采集线程: 读帧→模糊→去重→判定(跳过/OCR)"""
        while True:
            if self._cancel.is_set():
                break
            fd = self._sampler.next_frame()
            if fd is None:
                # 视频在运动段结束: 兜底送最后一帧(保召回, 滚动过滤兜底清理)
                self._flush_eof_settle()
                break
            self._inc_stats("frames_sampled", 1)
            self._print_progress(fd)
            self._emit("frame",
                       frame_id=fd.frame_id,
                       total_frames=self._sampler.total_frames,
                       timestamp=round(fd.timestamp, 2))
            fid, ts = fd.frame_id, fd.timestamp

            # ROI校验(周期性)
            if self._roi_manager.should_recheck(ts):
                self._roi_manager.validate_roi(fd.original_frame, ts)

            # 应用状态恢复的keyframe重置
            self._apply_pending_keyframe_resets()

            # settle式快照(稳态): 滚动/运动动画帧不送OCR, 等运动结束送第一帧
            # 静止帧(内容完整清晰); 超过最长抑制时长(长滚动段)恢复送检保召回。
            # 运动段抑制期间跳过其余质量闸门, 避免滚动垃圾帧污染去重基线。
            is_motion = self._is_motion_fd(fd)
            if self._settle_suppress_seconds > 0 and self._warmup_done:
                if self._settle_motion:
                    self._settle_last_fd = fd
                    if is_motion:
                        if self._value_enabled:
                            # 帧价值估计(通用送检决策, 替代状态闸门+按间隔抽稀):
                            # 每帧评估 价值=新颖度x清晰度(位移对齐后的新条带比例
                            # x GDE清晰度)。高价值帧(弹窗/toast/页面切换/滚动到
                            # 新屏=内容替换)立即打破抑制送检; 兜底: 距上次送检
                            # 超过兜底间隔且确有新内容时强制送检(防新颖度系统性
                            # 低估导致长滚动全抑制, 如20F5协议条款); 其余(滚动
                            # 中间帧位移对齐后无新内容/拖影)继续抑制。滚动、弹窗、
                            # toast、切换在同一把尺子上, 无需为每种运动模式打补丁。
                            fv = self._frame_value.estimate(fd)
                            novelty = float(fv["novelty"])
                            clarity = float(fv["clarity"])
                            value = float(fv["value"])
                            # 滚动帧用更高的break门槛(value_scroll_break_novelty):
                            # 位移累积型内容, 0.5屏在观感上仍"大量重复"; 内容替换
                            # (弹窗/toast/切换, content_change)仍用0.5立即送检。
                            if fv["diag"]["pc_state"] == "scrolling":
                                break_threshold = self._value_scroll_break_novelty
                            else:
                                break_threshold = self._value_high_threshold
                            if value >= break_threshold:
                                # 高价值: 内容替换(弹窗/toast/切换/新屏) → 立即
                                # 打破抑制, 落入下方正常流程(模糊/去重/复用/OCR)
                                self._settle_suppressed_count += 1
                                self._inc_stats("frames_value_break", 1)
                                self._save_sampled_frame(fd, "value_break")
                                self._logger.log(
                                    "pipeline", "settle_value_break",
                                    f"帧{fid}: 高价值打破抑制送检(新颖度="
                                    f"{novelty:.2f}, 清晰, 价值={value:.2f})",
                                    frame_id=fid,
                                    extra={"mae": round(fd.mae_value, 2),
                                           "novelty": round(novelty, 2),
                                           "clarity": clarity,
                                           "value": round(value, 2),
                                           **self._value_diag(fv)})
                                # 不continue: 走正常流程
                                # 决策即推进参考帧: 无论该帧随后是否被下游
                                # 拦截(thinned/去重/复用), 内容已记录, 位移
                                # 应重置, 否则滚动段每帧都持续break。
                                self._frame_value.update_reference(fd.gray_small)
                            elif (self._value_fallback_interval > 0
                                  and self._last_ocr_ts is not None
                                  and ts - self._last_ocr_ts
                                  >= self._value_fallback_interval
                                  and novelty >= self._scroll_fb_novelty
                                  and clarity >= 1.0):
                                # 兜底送检: 长滚动中确有新内容但未达高阈值 →
                                # 保召回(防全抑制丢协议条款类中间内容)
                                self._settle_suppressed_count += 1
                                self._inc_stats("frames_scroll_sampled", 1)
                                self._save_sampled_frame(fd, "value_fallback")
                                self._logger.log(
                                    "pipeline", "settle_value_fallback",
                                    f"帧{fid}: 兜底送检(距上次OCR "
                                    f"{ts - self._last_ocr_ts:.2f}s, 新颖度="
                                    f"{novelty:.2f})",
                                    frame_id=fid,
                                    extra={"mae": round(fd.mae_value, 2),
                                           "novelty": round(novelty, 2),
                                           "clarity": clarity,
                                           **self._value_diag(fv)})
                                # 不continue: 走正常流程
                                # 决策即推进参考帧(同value_break理由)
                                self._frame_value.update_reference(fd.gray_small)
                            elif (self._scroll_guard_interval > 0
                                  and self._last_ocr_ts is not None
                                  and ts - self._last_ocr_ts
                                  >= self._scroll_guard_interval
                                  and fv["diag"]["pc_state"] == "scrolling"
                                  and abs(float(
                                      fv["diag"].get("pc_dy", 0.0)))
                                  <= self._scroll_guard_max_dy
                                  and novelty >= self._scroll_guard_min_novelty
                                  and clarity >= 1.0):
                                # 滚动长时兜底: 慢滚动位移长期不足位移门槛
                                # (实测f0185→f0255段1.3s累计0.69屏<0.75), 若
                                # 不兜底会整段零送检, 全靠settle帧兜底, 中间
                                # 新消息/toast被漏; 滚动持续超时且确有新内容
                                # (novelty>=内容门槛)且确为慢滚动(帧间位移
                                # dy<=速度门槛, 快速滚动帧不兜底以免抢先于
                                # 自然break)时强制送一帧, 该帧仍走文本重叠
                                # 反馈(冗余→提高门槛, 新文本→重置), 不破坏
                                # 内容驱动节奏。内容门槛防止纯时间兜底抢先于
                                # 自然break(实测f0132新颖度0.22被拦, f0137
                                # 新颖度0.31也低于0.35门槛被拦, f0156保持
                                # 0.76屏break; 慢滚动f0223新颖度0.41仍兜底)。
                                self._settle_suppressed_count += 1
                                self._inc_stats("frames_scroll_sampled", 1)
                                self._save_sampled_frame(fd, "scroll_guard")
                                self._logger.log(
                                    "pipeline", "settle_scroll_guard",
                                    f"帧{fid}: 滚动长时兜底送检(距上次OCR "
                                    f"{ts - self._last_ocr_ts:.2f}s"
                                    f">={self._scroll_guard_interval:.2f}s, "
                                    f"新颖度={novelty:.2f})",
                                    frame_id=fid,
                                    extra={"mae": round(fd.mae_value, 2),
                                           "novelty": round(novelty, 2),
                                           "clarity": clarity,
                                           "value": round(value, 2),
                                           **self._value_diag(fv)})
                                # 不continue: 走正常流程(正常流程会更新去重
                                # 基线_last_ocr_gray/_last_ocr_ts并推进内容基线
                                # update_reference清零est_shift)。这是有意为之:
                                # 速度门槛(如上)已保证guard只落在慢滚动(dy<=2),
                                # 快速滚动帧不兜底, 故不会出现"guard先推进参考帧、
                                # 随后自然break被清零位移抑制"的场景(实测f0137
                                # dy=4/f0139 dy=6被拦, f0156保持0.76屏break);
                                # 而guard帧确实送检消费了新内容, 以其为新基线
                                # 重置位移是自洽的。
                            else:
                                # 低价值: 继续抑制(滚动中间帧无新内容/拖影)
                                self._settle_suppressed_count += 1
                                self._inc_stats("frames_scroll_suppressed", 1)
                                self._save_sampled_frame(fd, "suppressed")
                                self._logger.log(
                                    "pipeline", "settle_suppressed",
                                    f"帧{fid}: 运动抑制中(新颖度={novelty:.2f}, "
                                    f"GDE={'清晰' if clarity >= 1.0 else '拖影'}, "
                                    f"价值={value:.2f}<{self._value_high_threshold}), "
                                    f"跳过OCR",
                                    frame_id=fid,
                                    extra={"mae": round(fd.mae_value, 2),
                                           "novelty": round(novelty, 2),
                                           "clarity": clarity,
                                           "value": round(value, 2),
                                           **self._value_diag(fv)})
                                self._verdicts.put((fid, "skip"))
                                self._update_warmup(fd)
                                continue
                        else:
                            # 旧逻辑(对照/回退): 状态闸门 + 按间隔抽稀。
                            if (ts - self._settle_motion_start_ts
                                    <= self._settle_suppress_seconds):
                                # 运动中且未超抑制时长: 抑制
                                self._settle_suppressed_count += 1
                                self._inc_stats("frames_scroll_suppressed", 1)
                                self._save_sampled_frame(fd, "suppressed")
                                self._logger.log(
                                    "pipeline", "settle_suppressed",
                                    f"帧{fid}: 运动抑制中(MAE={fd.mae_value:.1f} "
                                    f">=T1={self._sampler.t1:.1f}), 跳过OCR",
                                    frame_id=fid,
                                    extra={"mae": round(fd.mae_value, 2),
                                           "t1": round(self._sampler.t1, 2)})
                                self._verdicts.put((fid, "skip"))
                                self._update_warmup(fd)
                                continue
                            # 超长运动段(长滚动): 状态闸门+间隔抽稀(旧逻辑)
                            self._settle_last_fd = fd
                            if (self._last_ocr_ts is not None
                                    and ts - self._last_ocr_ts
                                    < self._scroll_min_interval):
                                self._settle_suppressed_count += 1
                                self._inc_stats("frames_scroll_suppressed", 1)
                                self._save_sampled_frame(fd, "suppressed")
                                self._logger.log(
                                    "pipeline", "settle_long_scroll",
                                    f"帧{fid}: 长滚动抽稀(距上次OCR "
                                    f"{ts - self._last_ocr_ts:.2f}s"
                                    f"<{self._scroll_min_interval:.2f}s), 跳过OCR",
                                    frame_id=fid,
                                    extra={"suppressed":
                                           self._settle_suppressed_count})
                                self._verdicts.put((fid, "skip"))
                                self._update_warmup(fd)
                                continue
                            if self._is_scroll_fd(fd):
                                self._settle_suppressed_count += 1
                                self._inc_stats("frames_scroll_suppressed", 1)
                                self._save_sampled_frame(fd, "suppressed")
                                self._logger.log(
                                    "pipeline", "settle_long_scroll",
                                    f"帧{fid}: 长滚动scrolling帧零收益(距上次OCR "
                                    f"{ts - self._last_ocr_ts:.2f}s), 跳过OCR",
                                    frame_id=fid,
                                    extra={"suppressed":
                                           self._settle_suppressed_count})
                                self._verdicts.put((fid, "skip"))
                                self._update_warmup(fd)
                                continue
                            # 距上次OCR足够久且非scrolling: 放行走正常流程
                            self._settle_suppressed_count += 1
                            self._inc_stats("frames_scroll_sampled", 1)
                            self._save_sampled_frame(fd, "long_scroll_sample")
                            self._logger.log(
                                "pipeline", "settle_long_scroll_sample",
                                f"帧{fid}: 长滚动采样送检(距上次OCR "
                                f"{ts - self._last_ocr_ts:.2f}s)",
                                frame_id=fid,
                                extra={"suppressed":
                                       self._settle_suppressed_count})
                        # 不continue: 落入下方正常流程(模糊/去重/内容复用/OCR)
                    else:
                        # 运动结束: 本帧即settled帧, 走正常流程送OCR
                        self._settle_motion = False
                        self._settle_last_fd = None
                        # settle门控: 距最后送检帧太近(滚动减速停止阶段, 画面
                        # 与最后送检帧大量重复, 实测f0156→f0174相距0.34s滚入
                        # 约0.2屏文本重叠>0.9) → 跳过OCR, 冗余交给消费端已有
                        # 语义去重; 距上次送检远(长滚动整段被抑制)仍送settle
                        # 兜底召回。0=禁用。
                        if (self._settle_skip_interval > 0
                                and self._last_ocr_ts is not None
                                and ts - self._last_ocr_ts
                                < self._settle_skip_interval):
                            self._settle_suppressed_count += 1
                            self._inc_stats("frames_settle_skip", 1)
                            self._save_sampled_frame(fd, "settle_skip")
                            self._logger.log(
                                "pipeline", "settle_skip",
                                f"帧{fid}: settle帧与上次送检相距"
                                f"{ts - self._last_ocr_ts:.2f}s"
                                f"(<{self._settle_skip_interval:.2f}s)内容"
                                f"重复, 跳过OCR",
                                frame_id=fid,
                                extra={"since_last_ocr":
                                       round(ts - self._last_ocr_ts, 3),
                                       "interval":
                                       self._settle_skip_interval})
                            self._verdicts.put((fid, "skip"))
                            self._update_warmup(fd)
                            continue
                        self._logger.log(
                            "pipeline", "settle_snapshot",
                            f"帧{fid}: 运动结束, 送settled帧(抑制"
                            f"{self._settle_suppressed_count}帧)",
                            frame_id=fid,
                            extra={"suppressed": self._settle_suppressed_count,
                                   "mae": round(fd.mae_value, 2),
                                   "pc_state": getattr(fd, "pc_state", "static"),
                                   "pc_peak": round(getattr(fd, "pc_peak", 0.0), 3),
                                   "pc_dy": round(getattr(fd, "pc_dy", 0.0), 2)})
                        self._settle_suppressed_count = 0
                elif is_motion:
                    # 运动开始: 抑制本帧, 记录段起始
                    self._settle_motion = True
                    self._settle_motion_start_ts = ts
                    self._settle_last_fd = fd
                    self._settle_suppressed_count = 1
                    self._inc_stats("frames_scroll_suppressed", 1)
                    self._save_sampled_frame(fd, "suppressed")
                    self._logger.log(
                        "pipeline", "settle_suppress_start",
                        f"帧{fid}: 运动开始(MAE={fd.mae_value:.1f}, "
                        f"pc={fd.pc_state}, peak={fd.pc_peak:.2f}, "
                        f"dy={fd.pc_dy:.1f}px), 进入OCR抑制, 等运动结束送settled帧",
                        frame_id=fid,
                        extra={"mae": round(fd.mae_value, 2),
                               "t1": round(self._sampler.t1, 2),
                               "pc_state": getattr(fd, "pc_state", "static"),
                               "pc_peak": round(getattr(fd, "pc_peak", 0.0), 3),
                               "pc_dy": round(getattr(fd, "pc_dy", 0.0), 2)})
                    self._verdicts.put((fid, "skip"))
                    self._update_warmup(fd)
                    continue

            # settle静止尾门控(稳态): 运动段刚结束(settle_skip只拦运动结束的
            # 第一帧)后, 紧接的静止帧距上次送检仍不足settle_skip_interval时,
            # 与刚送检帧几乎相同——滚动减速帧与settle帧只差最后一点滚入量,
            # 实测fb3中f0175距f0170(guard帧, 滚动中途)仅0.09s, 像素去重因
            # 参考帧在滚动中途而失效被误放行, 形成f0174/f0175级近重复; 用与
            # settle_skip相同的"距上次送检时长"门控一并跳过。新内容(弹窗/
            # toast/切换)在首次出现的帧即为content_change/高新颖度, 已走
            # value_break送检, 不受此门控影响(故需not is_motion排除break帧)。
            if (self._settle_skip_interval > 0
                    and self._warmup_done
                    and not is_motion
                    and self._last_ocr_ts is not None
                    and ts - self._last_ocr_ts
                    < self._settle_skip_interval):
                self._settle_suppressed_count += 1
                self._inc_stats("frames_settle_skip", 1)
                self._save_sampled_frame(fd, "settle_skip")
                self._logger.log(
                    "pipeline", "settle_tail_skip",
                    f"帧{fid}: settle静止尾帧距上次送检"
                    f"{ts - self._last_ocr_ts:.2f}s"
                    f"(<{self._settle_skip_interval:.2f}s)内容重复, 跳过OCR",
                    frame_id=fid,
                    extra={"since_last_ocr": round(ts - self._last_ocr_ts, 3),
                           "interval": self._settle_skip_interval})
                self._verdicts.put((fid, "skip"))
                self._update_warmup(fd)
                continue

            # 模糊过滤
            # 仅运动/动画中间帧可能带拖影, 才做模糊检查; settle静止帧(运动
            # 结束后送检)拉普拉斯分数低只代表文本稀疏/页面留白, 不是模糊,
            # 若用预热期密集页面抬高的绝对下限会误杀(实测xxr后期score 875
            # ~4090 vs 预热p5=5093, OCR却全部清晰可读)。
            if is_motion and self._blur_detector.is_blurry(fd.roi_frame):
                self._inc_stats("frames_blur_skipped", 1)
                self._save_sampled_frame(fd, "blur")
                self._verdicts.put((fid, "skip"))
                self._update_warmup(fd)
                continue

            # 滚动帧抽稀(稳态): 滚动动画帧像素差异大但语义常相同(聊天记录上滑/
            # 回弹), 视觉去重无法区分, 逐帧送OCR会大量冗余。预热完成后, 滚动帧
            # (MAE>=T1)距上次OCR帧不足间隔则跳过。静止帧不受影响, 滚动中露出
            # 的新内容仍以≤间隔粒度被采样, 配合文本跟踪跨帧投票保证召回。
            if self._is_scroll_fd(fd) and (self._warmup_done
                                           or fd.pc_state == "scrolling"):
                if (self._last_ocr_ts is not None
                        and ts - self._last_ocr_ts < self._scroll_min_interval):
                    self._inc_stats("frames_scroll_thinned", 1)
                    self._save_sampled_frame(fd, "thinned")
                    self._logger.log("pipeline", "thinned_scroll",
                                     f"帧{fid}: 滚动帧抽稀"
                                     f"(距上次OCR {ts - self._last_ocr_ts:.2f}s"
                                     f"<{self._scroll_min_interval:.2f}s), 跳过OCR",
                                     frame_id=fid)
                    self._verdicts.put((fid, "skip"))
                    self._update_warmup(fd)
                    continue

            # 像素级强去重: 与上一张送OCR的图基本重复则跳过(保底, 不送OCR)
            if self._last_ocr_gray is not None:
                dup_mae = compute_mae(self._last_ocr_gray, fd.gray_small)
                if dup_mae < self._dedup_send_min_mae:
                    self._inc_stats("frames_cv_dup_skipped", 1)
                    self._save_sampled_frame(fd, "dup")
                    self._logger.log("pipeline", "skip_cv_dup",
                                     f"帧{fid}: 与上一张OCR图基本重复"
                                     f"(MAE={dup_mae:.2f}<{self._dedup_send_min_mae}), "
                                     f"跳过OCR",
                                     frame_id=fid)
                    self._verdicts.put((fid, "skip"))
                    self._update_warmup(fd)
                    continue

            # 视觉去重(三档置信度)
            dedup_result, gray = self._visual_dedup.check_with_gray(fd.roi_frame)
            self._cache_gray(fid, gray)

            # 预热观测收集
            if self._cal_enabled and not self._warmup_done:
                self._warmup_pairs.append(
                    (dedup_result.total_diff, fd.mae_value)
                )

            # 高置信度重复: 跳过OCR
            if dedup_result.confidence_tier == ConfidenceTier.HIGH_CONF_DUP:
                self._inc_stats("frames_dedup_skipped", 1)
                self._save_sampled_frame(fd, "dup")
                self._logger.log("pipeline", "skip_duplicate",
                                 f"帧{fid}: 高置信度重复, 跳过OCR",
                                 frame_id=fid)
                self._verdicts.put((fid, "skip"))
                self._update_warmup(fd)
                continue

            # 内容指纹复用: 该帧内容与已OCR过的帧在结构上相同(视觉噪声积累会
            # 使行差分/MAE误判为新内容), 直接重放缓存行, 不再调用OCR。
            content_h16 = None
            if self._content_reuse_max_hd > 0:
                content_h16 = compute_dhash(gray, 16)
                reused = self._find_content_reuse(content_h16)
                if reused is not None:
                    src_fid, src_ts, hd, src_lines = reused
                    # 用当前帧的fid/ts重新盖章
                    stamped = [
                        TextLine(text=t.text, bbox=t.bbox,
                                 confidence=t.confidence,
                                 frame_id=fid, timestamp=ts)
                        for t in src_lines
                    ]
                    # 复用行的 bbox 沿用**源帧**的 ROI 局部坐标(盖章只改 fid/ts),
                    # 故本帧的 ROI 原点记为源帧的, 否则跨预热/稳态复用时错位。
                    self._frame_roi[fid] = self._roi_origin(src_fid)
                    self._store_result(fid, stamped)
                    # 聊天/非聊天判定: 复用路径同样要打标, 否则非聊天帧会漏过门控
                    self._mark_chat_scene(fid, ts, fd.roi_frame)
                    self._verdicts.put(
                        (fid, "ocr", ts, dedup_result, fd.mae_value,
                         content_h16, fd.pc_state)
                    )
                    self._inc_stats("ocr_content_reused", 1)
                    self._save_sampled_frame(fd, "ocr_reused")
                    self._last_ocr_gray = fd.gray_small.copy()
                    self._last_ocr_ts = ts
                    self._frame_value.update_reference(fd.gray_small)
                    self._logger.log(
                        "pipeline", "content_reuse",
                        f"帧{fid}: 内容指纹复用(源帧{src_fid} hd16={hd}/256), "
                        f"跳过OCR调用",
                        frame_id=fid,
                        extra={"src_fid": src_fid, "hd16": hd,
                               "reused_lines": len(stamped)})
                    self._update_warmup(fd)
                    continue

            # 需要OCR: 交给worker池(有界队列背压)
            print(f"[OCR] 帧{fid} ({ts:.1f}s): 正在识别...", flush=True)
            self._save_sampled_frame(fd, "ocr")
            self._last_ocr_gray = fd.gray_small.copy()
            self._last_ocr_ts = ts
            self._frame_value.update_reference(fd.gray_small)
            self._mark_chat_scene(fid, ts, fd.roi_frame)
            # 记录裁图用的 ROI 原点: 该帧 OCR 行的 bbox 是这一版 ROI 的局部坐标
            self._frame_roi[fid] = (int(fd.roi_box[0]), int(fd.roi_box[1]))
            self._submit_queue.put((fd.roi_frame, fid, ts, dedup_result))
            self._ocr_expected += 1
            self._verdicts.put((fid, "ocr", ts, dedup_result, fd.mae_value,
                                content_h16, fd.pc_state))
            self._update_warmup(fd)

        # 生产结束: 通知消费与worker
        self._producer_done.set()
        self._verdicts.put(_VERDICT_END)
        for _ in self._workers:
            self._submit_queue.put(_OCR_END)

    def _flush_eof_settle(self) -> None:
        """
        视频在运动段结束时兜底送最后一帧OCR

        settle抑制期间运动帧不送OCR, 若视频恰在运动段结束(如录制中止于滚动
        中), 最后一帧的内容会丢失; 此处把它按滚动帧提交, 由消费端的滚动
        置信度过滤/碎片规则兜底清理, 尽量保召回。
        """
        self._settle_motion = False
        fd = self._settle_last_fd
        self._settle_last_fd = None
        self._settle_suppressed_count = 0
        if fd is None:
            return
        try:
            dedup_result, gray = self._visual_dedup.check_with_gray(fd.roi_frame)
        except Exception:
            return
        fid, ts = fd.frame_id, fd.timestamp
        print(f"[OCR] 帧{fid} ({ts:.1f}s): 正在识别...", flush=True)
        self._save_sampled_frame(fd, "ocr")
        self._last_ocr_gray = fd.gray_small.copy()
        self._last_ocr_ts = ts
        self._frame_value.update_reference(fd.gray_small)
        self._mark_chat_scene(fid, ts, fd.roi_frame)
        self._frame_roi[fid] = (int(fd.roi_box[0]), int(fd.roi_box[1]))
        self._submit_queue.put((fd.roi_frame, fid, ts, dedup_result))
        self._ocr_expected += 1
        self._verdicts.put((fid, "ocr", ts, dedup_result, fd.mae_value,
                            None, fd.pc_state))
        self._logger.log("pipeline", "settle_eof_flush",
                         f"视频结束于运动段, 兜底送帧{fid} OCR(保召回)",
                         frame_id=fid)

    def _ocr_worker(self) -> None:
        """OCR worker: 提交API任务(或本地/mock直接识别)"""
        while True:
            item = self._submit_queue.get()
            try:
                if item is _OCR_END:
                    break
                roi, fid, ts, _ = item
                try:
                    if fid in self._ocr_cache:
                        # 断点续传: 命中缓存, 跳过OCR调用
                        self._inc_stats("ocr_cache_hit", 1)
                        self._store_result(fid, self._ocr_cache[fid])
                    elif self._ocr_client.mode == "api":
                        self._inc_stats("ocr_calls_made", 1)
                        # 等待在途令牌(防429限流)
                        with self._inflight_cond:
                            while self._inflight >= self._inflight_limit:
                                self._inflight_cond.wait(timeout=0.2)
                            self._inflight += 1
                        try:
                            job_id = self._ocr_client.submit(roi, fid, ts)
                        except Exception:
                            with self._inflight_cond:
                                self._inflight -= 1
                                self._inflight_cond.notify_all()
                            raise
                        self._remember_roi(fid, roi)
                        with self._pending_lock:
                            self._pending.append((job_id, fid, ts))
                        with self._pending_cond:
                            self._pending_cond.notify_all()
                    else:
                        # 本地/mock: 同步识别, 直接写结果
                        self._inc_stats("ocr_calls_made", 1)
                        lines = self._ocr_client.recognize(roi, fid, ts)
                        self._store_result(fid, lines)
                except Exception as e:
                    self._logger.log("pipeline", "ocr_submit_error",
                                     f"帧{fid} OCR失败 / OCR failed: {e}",
                                     frame_id=fid)
                    self._store_result(fid, [], failed=True)
            finally:
                self._submit_queue.task_done()

    def _poller_loop(self) -> None:
        """轮询线程: 统一扫描在途API任务"""
        while True:
            with self._pending_cond:
                while not self._pending and not self._all_ocr_done():
                    self._pending_cond.wait(timeout=0.5)
                if not self._pending and self._all_ocr_done():
                    break
                jobs = list(self._pending)
                self._pending.clear()
            for job_id, fid, ts in jobs:
                try:
                    lines = self._ocr_client.poll_once(job_id, fid, ts)
                except Exception as e:
                    self._logger.log("pipeline", "ocr_poll_error",
                                     f"帧{fid} OCR轮询失败 / poll failed: {e}",
                                     frame_id=fid)
                    lines = []
                if lines is None:
                    with self._pending_lock:
                        self._pending.append((job_id, fid, ts))
                    continue
                if len(lines) > 0:
                    self._store_result(fid, lines)
                    # 释放在途令牌
                    with self._inflight_cond:
                        self._inflight -= 1
                        self._inflight_cond.notify_all()
                    continue
                # 空结果: API返回done但无文本(服务端偶发降级), 重新提交同一图重试
                attempts = self._ocr_retries.get(fid, 0) + 1
                self._ocr_retries[fid] = attempts
                roi = self._pending_rois.get(fid)
                if roi is None or attempts > 2:
                    self._logger.log("pipeline", "ocr_empty_result",
                                     f"帧{fid} OCR返回空(重试{attempts - 1}次仍空), 存空 / "
                                     f"empty after {attempts - 1} retries",
                                     frame_id=fid)
                    self._store_result(fid, [], failed=True)
                    with self._inflight_cond:
                        self._inflight -= 1
                        self._inflight_cond.notify_all()
                    continue
                self._inc_stats("ocr_empty_retries", 1)
                self._logger.log("pipeline", "ocr_empty_retry",
                                 f"帧{fid} OCR空结果, 重试{attempts}/2 / empty, retry {attempts}/2",
                                 frame_id=fid)
                try:
                    self._inc_stats("ocr_calls_made", 1)
                    new_job = self._ocr_client.submit(roi, fid, ts)
                except Exception as e:
                    self._logger.log("pipeline", "ocr_submit_error",
                                     f"帧{fid} OCR重试提交失败 / retry submit failed: {e}",
                                     frame_id=fid)
                    self._store_result(fid, [], failed=True)
                    with self._inflight_cond:
                        self._inflight -= 1
                        self._inflight_cond.notify_all()
                    continue
                with self._pending_lock:
                    self._pending.append((new_job, fid, ts))
                with self._pending_cond:
                    self._pending_cond.notify_all()

    def _all_ocr_done(self) -> bool:
        """所有job已提交且全部处理完"""
        return (self._producer_done.is_set()
                and self._submit_queue.unfinished_tasks == 0)

    def _consumer_loop(self) -> None:
        """消费线程(主线程): 按帧序处理OCR结果"""
        while True:
            if self._cancel.is_set():
                break
            try:
                item = self._verdicts.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is _VERDICT_END:
                break
            fid, kind = item[0], item[1]
            if kind == "skip":
                continue
            _, _, ts, dedup_result, mae, content_h16, pc_state = item
            lines = self._wait_ocr_result(fid)
            self._consume_ocr_frame(fid, ts, dedup_result, lines, mae,
                                    content_h16, pc_state)

    def _wait_ocr_result(self, fid: int) -> List[TextLine]:
        """等待指定帧的OCR结果(阻塞)"""
        with self._result_cond:
            while fid not in self._results:
                self._result_cond.wait(timeout=1.0)
            return self._results.pop(fid)

    def _store_result(self, fid: int, lines: List[TextLine],
                      failed: bool = False) -> None:
        """写入OCR结果并唤醒等待者。

        failed=True 表示该"空结果"来自OCR调用失败而非真实无文本, 标记后消费线程
        不会把它写入断点缓存/内容指纹缓存。
        """
        if failed:
            self._ocr_failed_fids.add(fid)
        with self._result_cond:
            self._results[fid] = list(lines) if lines else []
            self._result_cond.notify_all()

    def _inc_stats(self, key: str, n: int = 1) -> None:
        """线程安全统计计数"""
        with self._stats_lock:
            self._stats[key] += n

    def _emit(self, event_type: str, **kw) -> None:
        """发射进度回调事件(Web服务实时推送)
        回调异常不中断流水线
        """
        cb = self._progress_cb
        if cb is None:
            return
        try:
            cb({"type": event_type, **kw})
        except Exception:
            pass

    def _cache_gray(self, fid: int, gray) -> None:
        """缓存该帧的降采样灰度(限长)"""
        self._gray_cache[fid] = gray
        if len(self._gray_cache) > self._gray_cache_size:
            for k in sorted(self._gray_cache)[
                    :len(self._gray_cache) - self._gray_cache_size]:
                del self._gray_cache[k]

    def _remember_roi(self, fid: int, roi) -> None:
        """记住最近提交帧的ROI(空结果重试用, 限长)"""
        self._pending_rois[fid] = roi
        if len(self._pending_rois) > self._pending_rois_max:
            for k in sorted(self._pending_rois)[
                    :len(self._pending_rois) - self._pending_rois_max]:
                del self._pending_rois[k]

    def _mark_chat_scene(self, fid: int, ts: float, roi_frame) -> None:
        """判定该帧是否为聊天界面并暂存结果(供消费线程查用)。

        非聊天帧(转账回单/银行页/设置页/长按菜单等)整帧丢弃其OCR行, 从入口
        挡住非聊天内容污染正文; 非聊天判定会写行为日志(便于回溯)。
        """
        if not self._chat_scene_filter:
            return
        is_chat, reason = chat_scene_reason(roi_frame, self._chat_wall_ratio)
        self._frame_chat[fid] = is_chat
        if len(self._frame_chat) > self._frame_chat_max:
            for k in sorted(self._frame_chat)[
                    :len(self._frame_chat) - self._frame_chat_max]:
                del self._frame_chat[k]
        if not is_chat:
            self._logger.log(
                "pipeline", "non_chat_frame",
                f"帧{fid}({ts:.1f}s): 非聊天界面({reason}), 丢弃该帧OCR行",
                frame_id=fid, extra={"reason": reason})

    def _apply_pending_keyframe_resets(self) -> None:
        """执行消费线程请求的keyframe重置(用灰度缓存)"""
        with self._reset_lock:
            if not self._pending_resets:
                return
            fids = list(self._pending_resets)
            self._pending_resets.clear()
        for rfid in fids:
            gray = self._gray_cache.get(rfid)
            if gray is not None:
                self._visual_dedup.update_keyframe_with_gray(gray)
                self._logger.log("pipeline", "keyframe_reset",
                                 f"状态恢复: keyframe重置为帧{rfid}",
                                 frame_id=rfid)
            else:
                self._logger.log("pipeline", "keyframe_reset_miss",
                                 f"状态恢复: 帧{rfid}不在灰度缓存, 跳过重置",
                                 frame_id=rfid)

    def _clean_ocr_lines(self, lines: List[TextLine],
                         scroll_frame: bool = False,
                         roi_w: int = 0, roi_h: int = 0) -> List[TextLine]:
        """
        OCR行级清洗(乱码/半截/界面元素过滤)
        保守规则, 宁少删: 只删明显垃圾行与确定的界面元素。
        滚动帧额外过滤: 滚动残影的识别结果置信度普遍更低, 且常出现
        假名/异常字符, 用更高置信度阈值+完整性规则过滤。
        界面元素过滤(需ROI尺寸): 聊天标题栏(对方昵称)、在线状态、日期分隔条
        (如"五月4")是对话框/手机系统的界面信息, 不是聊天消息, 若进入正文会
        被LLM当成消息输出, 污染收发/时间/顺序。过滤仅依赖"位置+长度+是否
        气泡对齐", 不依赖具体文案:
          - 日期分隔: 整行恰为"x月x日"/"2026年8月20日"/"08-20"等 → 丢弃
          - 顶部条带: 位于ROI顶部条带内且非消息气泡对齐(既非贴左缘也非贴右缘)
            且(短文本/含装饰字符/水平居中) → 标题栏/在线状态 → 丢弃
          - 语音气泡时长标签(如 "5\" ((")/长按菜单文案(如"转文字")/撤回提示条
            (如"你撤回了一条消息"): 界面元素/系统通知 → 丢弃
        online status and date separators (e.g. "五月4") are dialog/OS chrome,
          - Date separator: whole line is "x月x日"/"2026年8月20日"/"08-20" → drop
        """
        cleaned = []
        # 顶部UI条带高度: 标题栏+在线状态所在区域(按ROI高度比例, 带最小像素)
        strip_h = (max(_UI_STRIP_RATIO * roi_h, _UI_STRIP_MIN_PX)
                   if roi_h > 0 else 0)
        # 菜单簇判定(帧级): 同一帧出现 >=2 个"可能与正文同形"的菜单文案时,
        # 判定为长按菜单展开 → 这些文案一并丢弃(单条出现时保留, 防误杀正文)。
        ambiguous_hits = sum(1 for ln in lines
                             if (ln.text or "").strip() in _UI_LABEL_AMBIGUOUS)
        menu_cluster = ambiguous_hits >= 2
        # 系统弹窗簇判定(帧级): 同一帧出现系统对话框正文时, 其按钮文案("开启"/
        # "取消"等)也一并丢弃 —— 按钮单独出现时保留, 防误杀正文。
        sys_dialog_cluster = any(_SYS_DIALOG_RE.search((ln.text or "").strip())
                                 for ln in lines)
        for ln in lines:
            t = (ln.text or "").strip()
            if not t:
                continue
            # 时间戳行(如"23:53" / "00:00√")保留, 后续做时间戳分离
            if re.fullmatch(r"\d{1,2}:\d{2}\s*√?\s*", t):
                cleaned.append(ln)
                continue
            # 语音气泡时长标签(如 "5\" ((" / ") 4\"") : 气泡元信息, 丢弃
            if _is_voice_badge(t):
                continue
            # 长按菜单/操作按钮文案 + 撤回提示条: 界面元素/系统通知, 丢弃
            if t in _UI_LABEL_ALWAYS:
                continue
            if menu_cluster and t in _UI_LABEL_AMBIGUOUS:
                continue
            if _RECALL_NOTICE_RE.fullmatch(t):
                continue
            # 系统弹窗正文 + 其按钮文案(同帧成簇): 系统 UI, 丢弃
            if _SYS_DIALOG_RE.search(t):
                continue
            if sys_dialog_cluster and t in _DIALOG_BUTTONS:
                continue
            # 日期分隔条(如"五月4"/"2026年8月20日"/"08-20"): 界面元素, 丢弃
            if _DATE_DIVIDER_RE.fullmatch(t):
                continue
            # 日期分隔条粘合行(滚动帧常见): 分隔条常驻顶部居中, 滚动时OCR会
            # 把它与经过的消息粘成一行("五月4积分成功380"/"再来一单五月4"/
            # "23:五月"), 这类行既不贴左缘也不贴右缘且位于分隔条高度带内 →
            # 整行丢弃(消息正文由静止帧的独立行召回, 丢弃不丢内容)。
            if strip_h > 0 and roi_w > 0:
                x1, y1, x2, y2 = ln.bbox
                div_lo = strip_h + _UI_DIVIDER_BAND_LO
                div_hi = strip_h + _UI_DIVIDER_BAND_HI
                w = float(roi_w)
                if (y1 >= div_lo and y2 <= div_hi
                        and _DATE_DIVIDER_TOKEN_RE.search(t)
                        and not (x1 <= 0.15 * w or x2 >= 0.85 * w)):
                    continue
            # 顶部UI条带: 标题栏(昵称)/在线状态等固定界面元素, 丢弃
            if strip_h > 0 and roi_w > 0:
                x1, y1, x2, y2 = ln.bbox
                if y1 < strip_h and y2 > y1:
                    w = float(roi_w)
                    center_x = (x1 + x2) / 2.0
                    is_centered = (abs(center_x - w / 2.0)
                                   <= _UI_STRIP_CENTER_TOL * w)
                    has_marker = bool(re.search(r"[★☆♥❤]", t))
                    # 消息气泡必贴左/右缘, 条带内既不贴左也不贴右的必是界面
                    # 元素(消息若出现在条带内, 其首次位置在下方已建跟踪, 此处
                    # 仅影响滚动帧投票, 不影响内容召回)。
                    if (not (x1 <= 0.15 * w or x2 >= 0.85 * w)
                            and (len(t) <= _UI_SHORT_LEN
                                 or has_marker or is_centered)):
                        continue
            # 尾随聊天时间戳剥离: OCR 常把消息与相邻时间戳粘成一行
            # ("对啊23:47"/"我没有切换23:40.11"), 时间戳是界面标记, 剥离后
            # 才是消息正文(完整时间戳行在上面已保留, 不受影响)。
            t2 = _TRAILING_STAMP_RE.sub("", t).strip()
            if t2 != t:
                t = t2
                if not t:
                    continue
                ln = TextLine(text=t, bbox=ln.bbox, confidence=ln.confidence)
            # 纯符号/标点
            if re.fullmatch(r"[\W_]+", t):
                continue
            # 单字符残片
            if len(t) <= 1:
                continue
            # 无中文的短残片(如"55")
            if not re.search(r"[\u4e00-\u9fff]", t) and len(t) <= 2:
                continue
            # 纯时间/数字残片(如"23:"), 非完整时间戳
            if re.fullmatch(r"[\d:：.\-]+", t):
                continue
            # 低置信度乱码: 统一最低置信度兜底(见 _ocr_min_conf)
            if ln.confidence < self._ocr_min_conf:
                continue
            if scroll_frame:
                # 滚动帧: 残影识别置信度更低, 用更高阈值
                if ln.confidence < self._scroll_conf:
                    continue
                # 滚动帧: 日文假名(残影错字常见, 如"芯ラ人联尔芯")
                if re.search(r"[\u3040-\u30ff]", t):
                    continue
                # 滚动帧: 短中文碎片(残影半截行)
                if len(t) <= 2 and re.search(r"[\u4e00-\u9fff]", t):
                    continue
            cleaned.append(ln)
        return cleaned

    def _consume_ocr_frame(self, fid: int, ts: float,
                           dedup_result, ocr_lines: List[TextLine],
                           mae: float = 0.0,
                           content_h16: Optional[int] = None,
                           pc_state: str = "") -> None:
        """
        消费一个OCR帧: 文本跟踪→灰色地带LLM→状态机→组装
        在消费线程(主线程)按帧序执行
        """
        self._inc_stats("frames_ocr_processed", 1)

        # 该帧OCR调用是否失败(空结果是故障产物而非真实无文本)。失败的帧一律不写
        # 缓存, 否则重放会命中被固化的空结果, 让视觉通道静默归零。
        ocr_failed = fid in self._ocr_failed_fids
        self._ocr_failed_fids.discard(fid)

        # 断点续传: 新OCR帧增量落盘(保存原始OCR结果, 仲裁在重放时重跑)
        # 滚动帧判定: 帧间变化速度超过静止阈值 → 内容残片/错字, 先过滤再入缓存
        # 相位相关启用时: 只有真实滚动位移(scrolling)才启用滚动过滤; 内容替换
        # (content_change)是静止后的完整新内容, 用宽松过滤保召回。
        if self._sampler.pc_enabled:
            scroll_frame = (pc_state == "scrolling")
        else:
            scroll_frame = (mae >= self._sampler.t1)
        # ROI尺寸供界面元素(标题栏/在线状态/日期分隔)过滤
        roi_box = self._roi_manager.roi_box
        roi_w, roi_h = (roi_box[2], roi_box[3]) if roi_box else (0, 0)
        # 聊天/非聊天界面分离: 非聊天帧(转账回单/银行页/设置页/长按菜单等)的
        # OCR行整帧丢弃, 且不进入跟踪/状态机/组装(避免非聊天界面误触发内容缺口);
        # 仅落一份空OCR缓存, 保证断点重放结果一致。
        if self._chat_scene_filter and not self._frame_chat.pop(fid, True):
            self._inc_stats("frames_non_chat_skipped", 1)
            if fid not in self._ocr_cache and not ocr_failed:
                self._write_ocr_cache(fid, ts, [])
                self._ocr_cache[fid] = []
            print(f"[OCR] 帧{fid}: 非聊天界面, 丢弃{len(ocr_lines)}行", flush=True)
            self._emit("ocr_frame", frame_id=fid, timestamp=round(ts, 2),
                       lines=[], scroll_frame=scroll_frame)
            return
        ocr_lines = self._clean_ocr_lines(
            ocr_lines, scroll_frame=scroll_frame, roi_w=roi_w, roi_h=roi_h)
        if not ocr_failed:
            if fid not in self._ocr_cache:
                self._write_ocr_cache(fid, ts, ocr_lines)
                self._ocr_cache[fid] = list(ocr_lines)

            # 内容指纹缓存: 该帧内容入缓存, 供后续同内容帧复用OCR结果
            self._update_content_cache(content_h16, fid, ts, ocr_lines)

        is_gray_zone = (dedup_result.confidence_tier == ConfidenceTier.GRAY_ZONE)
        if is_gray_zone:
            self._inc_stats("gray_zone_count", 1)

        print(f"[OCR] 帧{fid}: 完成, {len(ocr_lines)}行文本", flush=True)
        self._emit("ocr_frame",
                   frame_id=fid,
                   timestamp=round(ts, 2),
                   lines=[{"text": t.text, "bbox": list(t.bbox),
                           "confidence": round(t.confidence, 4)}
                          for t in ocr_lines],
                   scroll_frame=scroll_frame)

        # 滚动帧: 只匹配不新建, 防滚动残片/错字污染跟踪
        if scroll_frame:
            self._inc_stats("frames_scroll_ignored", 1)

        # 文本行跟踪
        new_lines = self._text_tracker.update(
            fid, ts, ocr_lines, allow_new=not scroll_frame
        )

        # 灰色地带LLM语义判定
        if is_gray_zone and dedup_result.needs_llm_arbitration:
            ocr_lines = self._handle_gray_zone(
                fid, ts, ocr_lines, dedup_result
            )

        # 计算重叠率
        overlap_ratio = 0.0
        if self._prev_ocr_lines is not None and len(self._prev_ocr_lines) > 0:
            overlap_ratio = self._text_tracker.compute_overlap_ratio(
                ocr_lines, self._prev_ocr_lines
            )
        self._prev_ocr_lines = list(ocr_lines)

        # 文本重叠反馈: 滚动帧送检后按与上一送检帧的OCR行重叠率调节动态位移
        # 门槛——冗余滚动(高重叠)翻倍, 新文本滚入(低重叠)重置基线。用真实
        # 文本内容驱动滚动送检节奏, 时间间隔仅保底防抖。
        if (scroll_frame and self._value_fallback_overlap_high > 0
                and self._prev_ocr_lines is not None):
            if overlap_ratio >= self._value_fallback_overlap_high:
                self._scroll_fb_novelty = min(
                    self._value_fallback_novelty_max,
                    self._scroll_fb_novelty * 2)
            else:
                self._scroll_fb_novelty = self._value_fallback_min_novelty
            self._logger.log(
                "pipeline", "scroll_gate_adapt",
                f"帧{fid}: 文本重叠率={overlap_ratio:.2f} → 位移门槛 "
                f"{'翻倍' if overlap_ratio >= self._value_fallback_overlap_high else '重置'}"
                f"={self._scroll_fb_novelty:.2f}",
                frame_id=fid,
                extra={"overlap": round(overlap_ratio, 3),
                       "gate": round(self._scroll_fb_novelty, 3),
                       "baseline": round(self._value_fallback_min_novelty, 3),
                       "max": round(self._value_fallback_novelty_max, 3)})

        # 状态机更新
        action = self._state_machine.update(
            fid, ts, ocr_lines, overlap_ratio
        )

        # 根据状态机动作处理
        if not action.should_process:
            if action.state == SystemState.SUSPENDED:
                self._inc_stats("frames_state_suspended", 1)
            self._logger.log("pipeline", "skip_state",
                             f"帧{fid}: 状态机判定不处理(state={action.state.value})",
                             frame_id=fid)
            return

        # 保存新内容
        self._assembler.add_new_lines(new_lines)

        # 记录内容缺口
        if action.is_gap:
            self._inc_stats("gaps_detected", 1)
            self._assembler.add_gap(ts, action.gap_detail)
            self._logger.log("pipeline", "gap",
                             f"帧{fid}: 内容缺口 - {action.gap_detail}",
                             frame_id=fid)

        # 状态转换时重置去重基线(交给采集线程用缓存灰度执行)
        if action.is_state_change:
            if action.state == SystemState.NORMAL:
                self._request_keyframe_reset(fid)
                self._prev_ocr_lines = None
            self._logger.log("pipeline", "state_change",
                             f"帧{fid}: 状态变更 → {action.state.value}",
                             frame_id=fid)

    def _request_keyframe_reset(self, fid: int) -> None:
        """请求采集线程将keyframe重置为该帧"""
        with self._reset_lock:
            self._pending_resets.append(fid)

    def _print_progress(self, frame_data: FrameData) -> None:
        """按时间间隔打印处理进度"""
        now = time.time()
        if now - self._last_progress_ts < self._progress_interval:
            return
        self._last_progress_ts = now
        with self._stats_lock:
            st = dict(self._stats)
        print(f"[进度] 帧{frame_data.frame_id}/{self._sampler.total_frames} "
              f"({frame_data.timestamp:.1f}s) "
              f"耗时{now - self._start_time:.0f}s "
              f"采样{st['frames_sampled']} "
              f"OCR{st['frames_ocr_processed']} "
              f"模糊跳过{st['frames_blur_skipped']} "
              f"去重跳过{st['frames_dedup_skipped']} "
              f"滚动忽略{st['frames_scroll_ignored']} "
              f"缺口{st['gaps_detected']}", flush=True)
        self._emit("progress",
                   frame_id=frame_data.frame_id,
                   total_frames=self._sampler.total_frames,
                   timestamp=round(frame_data.timestamp, 2),
                   elapsed_sec=round(now - self._start_time, 2),
                   stats=st)

    def _update_warmup(self, frame_data: FrameData) -> None:
        """预热计数, 统计充分后触发校准"""
        if not self._cal_enabled or self._warmup_done:
            return
        # 自适应ROI: 预热期累计行级活动率(内容区=持续变化行带)
        self._roi_manager.feed_frame(frame_data.original_frame)
        self._warmup_count += 1
        if frame_data.mae_value >= self._static_floor:
            self._warmup_saw_change = True
        # 统计充分(帧数达标)且见过变化; 或达到兜底帧数(防全静止视频)
        if (self._warmup_count >= self._cal_min_frames
                and (self._warmup_saw_change
                     or self._warmup_count >= self._cal_max_frames)):
            self._finalize_calibration()

    def _finalize_calibration(self) -> None:
        """结束预热并校准各模块阈值"""
        self._warmup_done = True
        self._sampler.finalize_calibration()
        self._blur_detector.finalize_calibration(self._cal_blur_floor_pct)
        self._visual_dedup.finalize_calibration(
            self._warmup_pairs, self._static_floor,
            dup_pct=self._cal_dedup_dup_pct,
            new_pct=self._cal_dedup_new_pct
        )
        # 自适应ROI: 由预热活动数据求解内容区, 与首帧margin基准不同时更新
        # 下游视口(采样器裁剪/文本跟踪/组装宽度)。
        old_roi = self._roi_manager.roi_box
        new_roi = self._roi_manager.finalize()
        if new_roi != old_roi:
            self._sampler.set_roi_box(new_roi)
            self._text_tracker.set_roi_dimensions(new_roi[3], new_roi[2])
            self._assembler.set_roi_width(new_roi[2])
            self._assembler.set_roi_height(new_roi[3])
            self._logger.log("pipeline", "roi_adaptive",
                             f"自适应ROI生效: {old_roi} -> {new_roi}")
            self._emit("stage", stage="roi",
                       detail=f"自适应ROI: {new_roi}", roi=list(new_roi))
        self._logger.log("pipeline", "calibration_done",
                         f"预热结束: 采样{self._warmup_count}帧, "
                         f"dup={self._visual_dedup.current_dup_threshold:.0f}, "
                         f"new={self._visual_dedup.current_new_threshold:.0f}")
        print(f"[进度] 预热结束(采样{self._warmup_count}帧), 阈值已校准, 进入稳态处理",
              flush=True)
        self._emit("stage", stage="steady",
                   detail=f"稳态处理(预热{self._warmup_count}帧, dup阈值="
                          f"{self._visual_dedup.current_dup_threshold:.0f})")

    def _handle_gray_zone(self, fid: int, ts: float,
                          ocr_lines: List[TextLine],
                          dedup_result) -> List[TextLine]:
        """
        处理灰色地带帧: LLM语义判定 + OCR纠错 + 阈值校准反馈

        方案A: LLM仲裁灰色地带的语义重复判定
        方案B: 文本判定结果反馈给视觉模块校准阈值

        返回可能被纠错后的OCR文本行
        """
        self._inc_stats("llm_arbitration_count", 1)

        # LLM语义判定
        sem_result, corrected_lines = self._semantic_dedup.arbitrate(
            fid, ts, ocr_lines, self._prev_ocr_lines
        )

        # 统计OCR纠错
        if sem_result.llm_called and sem_result.verdict == SemanticVerdict.CORRECTED:
            self._inc_stats("ocr_corrected_count", 1)

        # 方案B: 文本反馈→视觉阈值校准
        if sem_result.text_verdict_for_calibration:
            self._inc_stats("calibration_count", 1)
            self._visual_dedup.calibrate_from_text_feedback(
                sem_result.text_verdict_for_calibration
            )

        # 根据语义判定结果处理
        if sem_result.verdict == SemanticVerdict.DUPLICATE:
            # LLM判定语义重复 → 丢弃当前帧文本
            self._logger.log("pipeline", "semantic_duplicate",
                             f"帧{fid}: LLM判定语义重复, 丢弃当前帧文本. "
                             f"原因: {sem_result.reason}",
                             frame_id=fid,
                             extra={"reason": sem_result.reason,
                                    "confidence": sem_result.confidence})
            # 返回空列表, 使下游不添加新内容
            return []

        self._logger.log("pipeline", "semantic_result",
                         f"帧{fid}: 语义判定={sem_result.verdict.value}, "
                         f"llm_called={sem_result.llm_called}",
                         frame_id=fid,
                         extra={"verdict": sem_result.verdict.value,
                                "llm_called": sem_result.llm_called,
                                "confidence": sem_result.confidence})

        return corrected_lines

    @property
    def stats(self) -> dict:
        return self._stats

    @property
    def logger(self) -> BehaviorLogger:
        return self._logger
