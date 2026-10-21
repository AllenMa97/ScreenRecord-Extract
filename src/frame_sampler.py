"""
保召回帧采样器

核心思想:
  采样是唯一无法被下游兜底的环节: 采集时漏掉的内容, 下游永远找不回来;
  而采集过多造成的冗余, 可以由OCR后的文本去重/合并轻松清理。

因此:
  - 逐帧采样, 不做任何抽稀
  - 每帧计算"帧间变化速度"v=MAE(上一帧,当前帧) 仅用于状态监测与预热校准

状态监测:
  - T1/T2 从近期v分布的分位数在线估计, 用于标注当前画面状态
  - static / scrolling / rapid 三态仅供日志观察, 不影响采样。
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.image_utils import to_gray, resize_frame, compute_mae, percentile
from src.utils.motion_utils import phase_correlation_displacement
from src.utils.behavior_logger import BehaviorLogger


@dataclass
class FrameData:
    """帧数据容器"""
    frame_id: int                        # 全局帧序号
    timestamp: float                     # 时间戳(秒)
    original_frame: np.ndarray           # 原始帧(全分辨率)
    roi_box: Tuple[int, int, int, int]   # ROI框(x,y,w,h): 剔除状态栏/导航栏后的内容区域
    roi_frame: np.ndarray                # ROI裁剪帧(原始分辨率), 仅对内容区域做OCR
    # ROI = Region of Interest (感兴趣区域): 剔除状态栏/导航栏后真正有内容的屏幕区域
    gray_small: np.ndarray               # 降采样灰度(用于计算)
    mae_value: float = 0.0               # 帧间变化速度(MAE)
    sampling_state: str = "static"       # 采样状态: static/scrolling/rapid
    # 相位相关位移估计(算法级判态)
    pc_dx: float = 0.0                   # 水平位移(px)
    pc_dy: float = 0.0                   # 垂直位移(px)
    pc_peak: float = 0.0                 # 相关峰强度[0,1]: 纯平移高, 内容替换低
    pc_state: str = "static"             # 相位相关状态: static/scrolling/content_change


class FrameSampler:
    """保召回自适应步长采样器"""

    def __init__(self, config: dict, roi_box: Tuple[int, int, int, int],
                 logger: BehaviorLogger):
        # 自校准参数(仅用于状态监测与预热校准)
        self._static_floor: float = config.get("static_floor", 2.0)
        self._adapt_window: int = config.get("adapt_window", 120)
        self._static_percentile: float = config.get("static_percentile", 30.0)
        self._active_percentile: float = config.get("active_percentile", 75.0)
        # 自适应快进采样参数(相对视频帧率推导, 非绝对帧数)
        self._static_confirm_seconds: float = config.get("static_confirm_seconds", 0.7)
        self._max_step_seconds: float = config.get("max_step_seconds", 1.0)
        # T1上限倍率: T1 = min(p30(速度窗口), static_floor * 此值)。
        # 滚动密集视频的速度窗口p30会漂移到滚动MAE区间, 使滚动帧被误标为
        # static, 导致下游滚动质量闸门(置信度过滤/抽稀)失效。锚定在静止地板附近。
        self._t1_cap_ratio: float = config.get("t1_cap_ratio", 3.0)
        # 相位相关判态(算法级, 替代T1/T2分位数猜状态): 滚动是刚体垂直平移,
        # 相位相关直接估计位移(dy)与内容一致性(peak), 物理意义明确, 不再依赖
        # 历史速度分布猜阈值。仅用于状态监测/滚动判定, 不影响采样本身。
        self._pc_enabled: bool = config.get("pc_enabled", False)
        self._peak_motion: float = config.get("peak_motion", 0.3)
        self._dy_scroll_px: float = config.get("dy_scroll_px", 1.0)
        # 启动兜底阈值(窗口未热时使用, 运行后由分位数自适应覆盖)
        self._t1: float = config.get("mae_threshold_static", 3.0)
        self._t2: float = config.get("mae_threshold_active", 20.0)
        self._compute_w: int = config.get("compute_width", 160)
        self._compute_h: int = config.get("compute_height", 120)
        self._roi_box = roi_box
        self._logger = logger

        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._total_frames: int = 0
        self._frame_idx: int = 0
        self._current_state: str = "static"
        self._frame_id_counter: int = 0
        # 采样帧序号 -> 视频帧序号(1基)映射。快进采样会跳帧, 因此跟踪行携带的
        # frame_id(采样序号)与视频帧位置不再一一对应, 随机读帧时必须按此换算,
        # 否则会读到完全不同时刻的画面(气泡像素验证/收发色判定全部失真)。
        self._id_to_video: dict = {}
        # 采样帧序号 -> 该帧裁图实际用的 ROI 盒(x, y, w, h)。自适应 ROI 会在预热
        # 结束时改变(实测 `(0,60,576,1100)` → `(0,302,576,808)`), 而 OCR 行的
        # bbox 是**裁图当时**那一版 ROI 的局部坐标; 随机读帧若一律用终版 ROI 裁,
        # 预热期帧的 bbox 会整体错位 242px(甚至越界), 任何依赖 y 的像素判据失效。
        self._roi_history: dict = {}
        # 帧间变化速度历史(自适应窗口)
        self._velocity_window: deque = deque(maxlen=self._adapt_window)
        self._prev_gray_small: Optional[np.ndarray] = None

        # 最近一次相位相关结果(供_build_frame读取)
        self._pc_last_dx: float = 0.0
        self._pc_last_dy: float = 0.0
        self._pc_last_peak: float = 1.0

        # 自适应快进状态
        self._step: int = 1                  # 当前采样步长(1=逐帧)
        self._static_streak: int = 0         # 连续静止帧数
        self._static_ref_gray: Optional[np.ndarray] = None  # 快进参考帧
        self._static_confirm_frames: int = 20   # 静止确认帧数(open后按fps推导)
        self._max_step: int = 30             # 最大步长(open后按fps推导)

    def open(self, video_path: str) -> bool:
        """打开视频文件"""
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            return False
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 快进参数按fps推导(适配任意帧率视频)
        self._static_confirm_frames = max(2, int(round(self._fps * self._static_confirm_seconds)))
        self._max_step = max(2, int(round(self._fps * self._max_step_seconds)))
        self._logger.log("frame_sampler", "open",
                         f"视频已打开 / Video opened: fps={self._fps}, "
                         f"total_frames={self._total_frames}, "
                         f"static_confirm={self._static_confirm_frames}帧, "
                         f"max_step={self._max_step}帧")
        return True

    def peek_first_frame(self) -> Optional[np.ndarray]:
        """
        读取首帧用于ROI检测, 不消耗帧序号。
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if ret:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_idx = 0
        return frame if ret else None

    def set_roi_box(self, roi_box: Tuple[int, int, int, int]) -> None:
        """设置ROI框"""
        self._roi_box = roi_box

    def read_roi_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        按采样帧序号(1基)随机读取ROI裁剪帧, 供气泡像素验证使用。
        Random-read the ROI-cropped frame by sampled frame id (1-based),
        for bubble pixel verification. 不改变主循环帧计数状态:
        主循环已结束, 此处只seek+read。

        注意: 参数是**采样帧序号**(FrameData.frame_id), 不是视频帧序号。
        快进采样会跳帧, 两者不相等, 必须先经 `_id_to_video` 换算再seek, 否则
        读到的将是另一时刻的画面, 使气泡像素验证/收发色判定失效。

        裁剪用**该帧当时**的 ROI(见 `_roi_history`), 不是终版 ROI: 自适应 ROI
        在预热结束时会挪原点/改高度, 而跟踪行携带的 bbox 是裁图当时那一版 ROI
        的局部坐标, 用终版 ROI 裁会让预热期帧的 bbox 整体错位甚至越界。
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        video_index = self._id_to_video.get(frame_id, frame_id)
        if video_index < 1 or video_index > self._total_frames:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, video_index - 1)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return self._get_roi_frame(frame, self._roi_history.get(frame_id))

    def _compute_roi_gray_small(self, frame: np.ndarray) -> np.ndarray:
        """裁剪ROI并降采样为灰度图"""
        x, y, w, h = self._roi_box
        roi = frame[y:y + h, x:x + w]
        gray = to_gray(roi)
        return resize_frame(gray, self._compute_w, self._compute_h)

    def _get_roi_frame(self, frame: np.ndarray,
                       box: Optional[Tuple[int, int, int, int]] = None
                       ) -> np.ndarray:
        """获取ROI裁剪帧(原始分辨率)。box 缺省用当前 ROI。"""
        x, y, w, h = box or self._roi_box
        return frame[y:y + h, x:x + w].copy()

    @staticmethod
    def _percentile(values: list, pct: float) -> float:
        """线性插值分位数"""
        return percentile(values, pct)

    def _update_thresholds(self) -> None:
        """按近期变化速度分布自适应更新T1/T2"""
        if len(self._velocity_window) < 20:
            return  # 窗口未热, 用启动兜底
        values = sorted(self._velocity_window)
        t1 = self._percentile(values, self._static_percentile)
        t2 = self._percentile(values, self._active_percentile)
        t1 = max(t1, self._static_floor)
        # 上限保护: 滚动密集视频中p30(窗口)远高于静止地板, 若不限制,T1会被
        # 抬进滚动MAE区间, 滚动帧被误标为static, 滚动质量闸门全部失效。
        t1 = min(t1, self._static_floor * self._t1_cap_ratio)
        t2 = max(t2, t1 + 0.5)
        self._t1 = t1
        self._t2 = t2

    def finalize_calibration(self) -> None:
        """
        预热阶段结束时用完整速度窗口校准T1/T2

        由流水线在探索阶段结束后调用
        """
        old_t1, old_t2 = self._t1, self._t2
        self._update_thresholds()
        self._logger.log("frame_sampler", "calibrate",
                         f"预热校准: T1 {old_t1:.1f} → {self._t1:.1f}, "
                         f"T2 {old_t2:.1f} → {self._t2:.1f} (窗口{len(self._velocity_window)}帧)",
                         extra={"old_t1": round(old_t1, 2),
                                "new_t1": round(self._t1, 2),
                                "old_t2": round(old_t2, 2),
                                "new_t2": round(self._t2, 2)})

    def _classify_state(self, velocity: float) -> None:
        """状态分类(仅用于日志/监测, 不影响采样)"""
        if velocity < self._t1:
            self._current_state = "static"
        elif velocity < self._t2:
            self._current_state = "scrolling"
        else:
            self._current_state = "rapid"

    def _classify_pc_state(self, peak: float, dy: float) -> str:
        """
        相位相关状态分类

        peak高 + 位移小 → 静止; peak高 + 位移大 → 滚动; peak低 → 内容替换。
        peak高说明两帧是纯平移关系(滚动/静止), 低说明内容变了(新消息/翻页)。
        """
        if peak >= self._peak_motion:
            return "scrolling" if abs(dy) >= self._dy_scroll_px else "static"
        return "content_change"

    def next_frame(self) -> Optional[FrameData]:
        """
        获取下一帧数据

        自适应步长采样:
          - 逐帧阶段: 每帧解码并与上一帧比MAE; 连续静止超过确认帧数后进入快进。
          - 快进阶段: 跳过step-1帧(仅grab不解码), 解码帧与静止参考帧比较;
            一旦出现变化立即回到逐帧(保召回, 宁杀错勿放过)。
          采样是唯一无法被下游兜底的环节; 冗余交给OCR后的文本去重/合并处理。

        返回None表示视频结束
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        if self._step > 1:
            return self._fast_forward_frame()
        return self._per_frame_frame()

    def _per_frame_frame(self) -> Optional[FrameData]:
        """逐帧采样: 每帧解码, 判定是否进入快进"""
        ret, frame = self._cap.read()
        if not ret:
            self._logger.log("frame_sampler", "eof", "视频结束 / End of video")
            return None
        self._frame_idx += 1
        current_time = self._frame_idx / self._fps

        # 帧间变化速度(与上一视频帧比较)
        gray_small = self._compute_roi_gray_small(frame)
        if self._prev_gray_small is None:
            velocity = 0.0
            pc_dx, pc_dy, pc_peak = 0.0, 0.0, 1.0
        else:
            velocity = compute_mae(self._prev_gray_small, gray_small)
            # 相位相关位移估计: 峰值位置=位移, 峰值强度=内容一致性
            if self._pc_enabled:
                pc_dx, pc_dy, pc_peak = phase_correlation_displacement(
                    self._prev_gray_small, gray_small)
            else:
                pc_dx, pc_dy, pc_peak = 0.0, 0.0, 1.0
        self._prev_gray_small = gray_small
        self._velocity_window.append(velocity)
        self._update_thresholds()
        self._classify_state(velocity)
        self._pc_last_dx, self._pc_last_dy, self._pc_last_peak = (
            pc_dx, pc_dy, pc_peak)

        # 静止计数 → 进入快进
        if velocity < self._t1:
            self._static_streak += 1
        else:
            self._static_streak = 0
        if self._static_streak >= self._static_confirm_frames:
            self._step = 2
            self._static_ref_gray = gray_small
            self._static_streak = 0
            self._logger.log("frame_sampler", "fast_forward_start",
                             f"连续{self._static_confirm_frames}帧静止, 进入快进",
                             frame_id=self._frame_id_counter + 1)

        return self._build_frame(frame, velocity, current_time, gray_small)

    def _fast_forward_frame(self) -> Optional[FrameData]:
        """快进采样: 跳帧解码, 与静止参考帧比较"""
        # 跳过step-1帧(仅grab不解码), 再retrieve()解码缓冲中的最后一帧。
        for _ in range(self._step - 1):
            if not self._cap.grab():
                self._logger.log("frame_sampler", "eof", "视频结束 / End of video")
                return None
        ret, frame = self._cap.retrieve()
        if not ret:
            self._logger.log("frame_sampler", "eof", "视频结束 / End of video")
            return None
        # 以解码器真实位置为准(1基)。手工累加在 grab/retrieve 语义下会漂移:
        # grab()只取帧到内部缓冲、retrieve()解码的是"最后一次grab的那帧",
        # 手工+1会让解码进度与 _frame_idx 每快进一次就差1帧并线性累积
        # (实测+50/+100/+200帧), 导致 _id_to_video 映射失真、read_roi_frame
        # 按错误帧号seek, 读到另一时刻的画面。
        self._frame_idx = int(round(self._cap.get(cv2.CAP_PROP_POS_FRAMES)))
        current_time = self._frame_idx / self._fps

        gray_small = self._compute_roi_gray_small(frame)
        if self._static_ref_gray is None:
            velocity = 0.0
            pc_dx, pc_dy, pc_peak = 0.0, 0.0, 1.0
        else:
            velocity = compute_mae(self._static_ref_gray, gray_small)
            # 相位相关: 与静止参考帧比较, 位移≈0且peak高 → 静止确认;
            # 若内容已替换(peak低)则MAE必超exit_threshold, 由上方快进退出兜底。
            if self._pc_enabled:
                pc_dx, pc_dy, pc_peak = phase_correlation_displacement(
                    self._static_ref_gray, gray_small)
            else:
                pc_dx, pc_dy, pc_peak = 0.0, 0.0, 1.0
        self._pc_last_dx, self._pc_last_dy, self._pc_last_peak = (
            pc_dx, pc_dy, pc_peak)

        # 保召回快进退出: 任何可见变化立即回逐帧。不用自适应T1判定——
        # 静止窗口的T1可能被微动抬高, 导致慢速滚动被误判"仍静止"而继续
        # 快进, 跳过滚动开头的帧(内容丢失)。改用绝对小阈值, 快进只跳过
        # 完全静止的画面。
        exit_threshold = min(self._t1, self._static_floor * 0.25)
        if velocity > exit_threshold:
            # 出现变化: 立即退出快进, 回逐帧(保召回)
            self._step = 1
            self._static_streak = 0
            self._static_ref_gray = None
            self._current_state = "rapid"
            self._logger.log("frame_sampler", "fast_forward_exit",
                             f"检测到变化(MAE={velocity:.2f}>"
                             f"{exit_threshold:.2f}), 回到逐帧",
                             frame_id=self._frame_id_counter + 1)
        else:
            # 仍静止: 渐进加大步长
            self._current_state = "static"
            self._step = min(int(self._step * 1.6), self._max_step)

        # 快进帧不进速度窗口(避免污染T1分布)
        return self._build_frame(frame, velocity, current_time, gray_small)

    def _build_frame(self, frame: np.ndarray, velocity: float,
                     current_time: float,
                     gray_small: np.ndarray) -> FrameData:
        """构造帧数据(公共)"""
        roi_frame = self._get_roi_frame(frame)
        self._frame_id_counter += 1
        self._id_to_video[self._frame_id_counter] = self._frame_idx
        self._roi_history[self._frame_id_counter] = tuple(self._roi_box)

        pc_state = self._classify_pc_state(self._pc_last_peak, self._pc_last_dy)
        self._logger.log("frame_sampler", "sample",
                         f"MAE={velocity:.2f}, state={self._current_state}, "
                         f"pc={pc_state}, T1={self._t1:.1f}, T2={self._t2:.1f}",
                         frame_id=self._frame_id_counter,
                         extra={"timestamp": round(current_time, 3),
                                "mae": round(velocity, 2),
                                "state": self._current_state,
                                "pc_state": pc_state,
                                "pc_dy": round(self._pc_last_dy, 2),
                                "pc_peak": round(self._pc_last_peak, 3)})
        return FrameData(
            frame_id=self._frame_id_counter,
            timestamp=current_time,
            original_frame=frame,
            roi_box=self._roi_box,
            roi_frame=roi_frame,
            gray_small=gray_small,
            mae_value=velocity,
            sampling_state=self._current_state,
            pc_dx=self._pc_last_dx,
            pc_dy=self._pc_last_dy,
            pc_peak=self._pc_last_peak,
            pc_state=pc_state
        )

    def close(self) -> None:
        """关闭视频"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def sampled_count(self) -> int:
        return self._frame_id_counter

    @property
    def t1(self) -> float:
        """自适应静止阈值(监测用)"""
        return self._t1

    @property
    def pc_enabled(self) -> bool:
        """是否启用相位相关判态"""
        return self._pc_enabled

    @property
    def t2(self) -> float:
        """自适应快速阈值(监测用)"""
        return self._t2
