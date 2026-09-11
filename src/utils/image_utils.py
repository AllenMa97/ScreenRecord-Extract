"""
图像工具模块

提供图像预处理、ROI 裁剪、帧差计算、分条、哈希计算等基础操作。

核心算法参考文献:
  - 帧差法 (Frame Differencing): 经典视频摘要方法, 用于运动检测
  - dHash (Difference Hash): 对图像梯度敏感, 适合检测内容位移
  - 行差分剖面 (Row Difference Profile): 利用行级像素差异检测滚动行为
"""

import numpy as np
import cv2
from typing import List, Tuple


def to_gray(frame: np.ndarray) -> np.ndarray:
    """转灰度图"""
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """缩放帧到目标尺寸"""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def crop_roi(frame: np.ndarray, roi_box: Tuple[int, int, int, int]) -> np.ndarray:
    """
    裁剪 ROI 区域。

    ROI (Region of Interest) = 感兴趣区域: 剔除状态栏/导航栏后真正有内容的屏幕区域。

    参数:
        frame: 原始帧
        roi_box: (x, y, w, h) ROI 框
    返回:
        裁剪后的图像(保留原始分辨率)
    """
    x, y, w, h = roi_box
    return frame[y:y + h, x:x + w].copy()


def compute_mae(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    计算两帧之间的平均绝对误差 (MAE)。

    用于帧差法判断画面变化程度。
    """
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    return float(np.mean(np.abs(frame1.astype(np.float64) - frame2.astype(np.float64))))


def split_into_strips(image: np.ndarray, k: int) -> List[np.ndarray]:
    """
    将图像水平分割为 k 个条带。

    用于行条带哈希去重: 不同滚动位置的文本内容会在不同条带中产生差异。
    """
    h = image.shape[0]
    strip_height = h // k
    strips = []
    for i in range(k):
        start = i * strip_height
        end = (i + 1) * strip_height if i < k - 1 else h
        strips.append(image[start:end, :])
    return strips


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """
    计算差值哈希 (dHash)。

    dHash 对水平梯度敏感, 比 pHash 更能检测文本行的垂直位移。

    返回:
        整数形式的哈希值
    """
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    gray = to_gray(resized) if len(resized.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    bits = diff.flatten()
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def hamming_distance(hash1: int, hash2: int) -> int:
    """计算两个哈希值的汉明距离"""
    return bin(hash1 ^ hash2).count("1")


def percentile(values: list, pct: float) -> float:
    """
    线性插值分位数。

    用于自校准: 从观测分布估计阈值, 避免写死绝对阈值。

    参数:
        values: 已排序的观测值列表
        pct: 百分位(0-100)
    """
    n = len(values)
    if n == 0:
        return 0.0
    idx = (n - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def compute_row_diff_profile(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """
    计算行差分剖面。

    对每一行计算像素差的绝对值之和, 返回长度为 H 的一维数组。

    用途:
      - 剖面接近全零 → 画面重复(果冻回弹/往复滑动)
      - 剖面有局部峰值 → 局部内容变化
      - 剖面整体偏移 → 滚动行为
    """
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    diff = np.abs(frame1.astype(np.float64) - frame2.astype(np.float64))
    return diff.sum(axis=1)


def compute_laplacian_variance(gray_image: np.ndarray) -> float:
    """
    计算拉普拉斯方差 (清晰度评分)。

    经典模糊检测方法: 方差越低, 图像越模糊。

    注意:
      与文本密度强相关(高频纹理量), 稀疏页面天然低分——只适合做
      "模糊弱证据", 需与方向熵(GDE)等与内容解耦的特征联合判定。
    """
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return float(laplacian.var())


def compute_gradient_direction_entropy(
        gray_image: np.ndarray, bins: int = 36,
        weak_gradient_ratio: float = 0.1) -> float:
    """
    计算梯度方向熵 (GDE)。

    核心思想:
      运动模糊(滚动拖影)使边缘沿运动方向拉长, 梯度方向趋于一致,
      方向直方图熵显著下降; 而文本密度只影响边缘数量, 不改变方向
      分布, 因此 GDE 与内容密度解耦。

    度量:
      对 Sobel 梯度方向角(-180°..180°)做直方图(默认 36 bin), 计算香农熵。
      清晰文本帧实测 ~3.3, 方向性拖影实测 <2.6, 纯空白=0。

    参考:
      类似思路用于无参考图像质量评价的运动模糊检测。

    参数:
        gray_image: 灰度图
        bins: 方向直方图 bin 数
        weak_gradient_ratio: 弱梯度阈值 = 梯度幅值 p95 × 此比例
    返回:
        梯度方向熵 (0 表示近空白/无有效梯度)
    """
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = max(float(np.percentile(mag, 95)) * weak_gradient_ratio, 1e-6)
    mask = mag > thr
    n = int(mask.sum())
    if n < 100:
        return 0.0
    ang = (np.arctan2(gy, gx) * 180.0 / np.pi)[mask]
    hist, _ = np.histogram(ang, bins=bins, range=(-180.0, 180.0))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log(hist)))


def normalize_for_ocr(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    """
    为 OCR 准备图像: 保持原始分辨率, 仅在过宽时轻度降采样。
    """
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame
