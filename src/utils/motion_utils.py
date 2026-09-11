"""
运动分析工具

提供两个携带语义的结构化特征, 替代"标量+阈值猜语义"的旧做法:

  1. phase_correlation_displacement: 相位相关位移估计。滚动是刚体垂直平移,
     帧间关系用位移量(dx, dy)完整描述; 峰值强度 peak 同时给出"内容一致性"
     (纯平移 vs 内容替换), 不再依赖 MAE 猜状态。

  2. gradient_direction_entropy: 梯度方向熵。单向运动模糊(滚动拖影)的梯度
     方向高度集中(各向异性高), 静止清晰帧的梯度方向分布均匀。熵归一化到
     [0,1]: 0=各向同性(清晰), 1=单向(拖影)。补充拉普拉斯方差的盲区。

参考文献:
  - 梯度方向熵: 运动模糊检测的经典各向异性度量, 见图像质量评估文献。
"""

import numpy as np
import cv2
from typing import Tuple


def phase_correlation_displacement(prev_gray: np.ndarray,
                                   cur_gray: np.ndarray
                                   ) -> Tuple[float, float, float]:
    """
    相位相关位移估计。

    对两帧(同尺寸)灰度图估计刚体平移量。输入应预先降采样并裁剪 ROI。

    参数:
        prev_gray: 前一帧灰度图
        cur_gray: 当前帧灰度图

    返回:
            dx, dy: 水平/垂直位移(px, cur 相对 prev)
            peak_value: 归一化互相关峰值强度[0,1], 越高=越接近纯平移
    """
    if prev_gray.shape != cur_gray.shape:
        cur_gray = cv2.resize(cur_gray, (prev_gray.shape[1], prev_gray.shape[0]))

    f1 = np.fft.fft2(prev_gray.astype(np.float64))
    f2 = np.fft.fft2(cur_gray.astype(np.float64))

    # 归一化互功率谱(相位保留, 幅值归一)
    cross = f1 * np.conj(f2)
    denom = np.abs(cross) + 1e-10
    r = np.fft.ifft2(cross / denom).real

    # 峰值位置 → 位移(折叠负半轴)
    h, w = r.shape
    flat_idx = int(np.argmax(r))
    dy_raw, dx_raw = divmod(flat_idx, w)
    dy = dy_raw if dy_raw <= h // 2 else dy_raw - h
    dx = dx_raw if dx_raw <= w // 2 else dx_raw - w

    peak_value = float(r[dy_raw, dx_raw])
    return float(dx), float(dy), peak_value


def gradient_direction_entropy(gray: np.ndarray, bins: int = 36,
                               min_magnitude: float = 8.0) -> float:
    """
    梯度方向熵。

    统计梯度方向直方图(按幅值加权), 计算归一化各向异性分。

    参数:
        gray: 灰度图
        bins: 方向分箱数
        min_magnitude: 总梯度幅值下限。低于此值视为低纹理页面, 返回 0.0
            (各向同性), 交由拉普拉斯通道判定, 避免熵在噪声下失稳。

    返回:
        gde_score [0,1]: 0=梯度方向各向同性(清晰), 1=单向集中(运动拖影)
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    total_mag = float(mag.sum())

    # 低纹理保护: 无足够梯度时熵无意义, 回退为"各向同性"
    if total_mag < min_magnitude:
        return 0.0

    angle = np.arctan2(gy, gx) + np.pi  # [0, 2pi)
    bin_idx = np.clip((angle / (2.0 * np.pi) * bins).astype(np.int32), 0, bins - 1)
    # 按幅值加权累加(弱梯度噪声权重低)
    hist = np.bincount(bin_idx.ravel(),
                       weights=mag.ravel(), minlength=bins).astype(np.float64)

    p = hist / hist.sum()
    eps = 1e-12
    entropy = -float((p * np.log(p + eps)).sum())
    max_entropy = np.log(bins)
    return float(1.0 - entropy / max_entropy)
