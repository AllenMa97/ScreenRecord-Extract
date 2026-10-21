"""
音频侧数据类型 (Audio-side data types)

承载"音频转文字"流水线各阶段间的数据结构, 与视觉侧在最终报告中按视频时间轴配对各用。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class VoiceUtterance:
    """一段被语音识别出的语音, 带视频时间轴锚点。

    start/end 均为该段在**视频时间轴**上的起止时刻(秒)。ASR 服务端不返回
    时间戳, 因此由客户端 VAD 切段得到, 再喂给 ASR。
    """
    start: float                 # 视频时间轴起点(秒)
    end: float                   # 视频时间轴终点(秒)
    text: str = ""               # ASR 转写文本
    # 非语音事件标记(SenseVoice 会返回如 🎼=音乐 / —=空白 / 笑声标签等)。
    # 若整段判定为纯非语音(如整段音乐), events 非空且 text 为空。
    events: List[str] = field(default_factory=list)
    confidence: float = 1.0      # 预留接口(服务端暂不返回逐字置信度)
    source: str = "asr"          # 来源方式(目前统一 'asr')

    @property
    def mid_time(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class VoiceEvent:
    """视觉侧检测到的"语音气泡"出现/播放事件, 用作与音频段的配对锚点。

    语音气泡内无文字, OCR 读不到, 需轻量视觉检测它会话内容起决定。
    """
    timestamp: float            # 视频时间轴出现时刻(秒)
    end_time: Optional[float] = None   # 气泡停留在屏上最晚时刻(播放结束/滚动离开)
    side: str = ""              # '发' / '收' / '' (绿色气泡=我方发出['发'], 白色气泡=[收])
    confidence: float = 0.0     # 检测置信度
    frame_id: int = 0           # 来源帧
    matched_text: str = ""      # 由音频段按时间就近配对填充的转写文本
    # 检测框(原始帧坐标), 供位置感知去重(区分同侧相邻的不同气泡)补齐
    box: Optional[Tuple[float, float, float, float]] = None   # (x, y, w, h)
    # 来源方式: 'vision'(视觉形状) / 'ocr'(OCR时长标签信号) / 'hybrid'(已合并)
    source: str = "vision"
    # OCR 时长标签(audio_duration)解析出的语音真实播放时长(秒), 供切段精确用。
    # 若为 None, 抽取时回退用 (end_time - timestamp) 气泡停留时长。
    audio_duration: Optional[float] = None
    # 最后一次在屏上的观测(时刻 + 框)。滚动/追加会把气泡整体平移, 首见位置是
    # "中途某一屏"的临时坐标; 只有**稳定停留后**的位置才能与同一屏的文本行比较。
    last_timestamp: Optional[float] = None
    last_box: Optional[Tuple[float, float, float, float]] = None
    # 第一次"屏幕静止"时的观测(时刻 + 框)。静止帧上测得的 y 才与同一时刻的滚动
    # 偏移曲线共基准; 滚动途中的观测会与曲线错位一整步滚动量。
    stable_timestamp: Optional[float] = None
    stable_box: Optional[Tuple[float, float, float, float]] = None


@dataclass
class AudioDecoded:
    """解码后的音频(重采样到目标配置, 单声道)。"""
    samples: "object"          # np.ndarray float64, 长度 = 总样本数
    sample_rate: int           # 实际采样率(ms期望 16000)
    audio_duration: float      # 音频总时长(秒)
    has_audio: bool            # 原文件是否含可解码音轨

    @property
    def duration(self) -> float:
        return self.audio_duration