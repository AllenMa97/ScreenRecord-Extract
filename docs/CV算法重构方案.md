# CV侧算法级重构方案 / CV-Side Algorithmic Rework Plan

> 版本 / Version: v0.5 (Phase 1-4 + blur双证据GDE/破锁 / Phases 1-4 + blur GDE dual-cue/unlock implemented)
> 状态 / Status: 已完成 / Completed
> 范围 / Scope: 帧状态分类、视觉去重、运动模糊检测三大模块的算法级改造

***

## 1. 背景与根因分析 / Background & Root-Cause Analysis

### 1.1 现状 / Current state

CV侧当前由四个模块组成，共20+可调超参数：

| 模块 / Module           | 核心方法 / Core method         | 超参数数量 / #params |
| --------------------- | -------------------------- | --------------- |
| `frame_sampler.py`    | MAE标量 + 速度窗口分位数(T1/T2)判态   | 9               |
| `blur_detector.py`    | 拉普拉斯方差 + 相对基线              | 4               |
| `visual_dedup.py`     | 行差分剖面sum + 条带dHash + 三档置信度 | 10              |
| `pipeline.py`(settle) | 时间窗抑制OCR                   | 2               |

上一轮修复（T1上限保护 + settle式快照）解决了两个具体缺陷，但暴露了一个更深层的问题：

### 1.2 根因：用标量猜语义 / Root cause: scalar guessing semantics

**所有超参数都在对抗同一个缺陷——把"运动/内容变化"压成一个标量，再用阈值去猜它的语义。**

1. **MAE（全局平均像素差）** 无法区分"滚动位移"（语义不变，像素全变）和"内容替换"（语义变化）。T1/T2自适应分位数、t1\_cap\_ratio、settle抑制时长……都是在给这个标量打补丁。
2. **行差分剖面** 天然携带"滚动 vs 新消息"的结构信息（滚动是整体平移、新消息是局部峰值），但 `total_diff = profile.sum()` 把剖面形状抹平了。`dup_tight_ratio / new_loose_ratio` 就是在为这个信息损失调参。
3. **条带dHash** 位置绑定：滚动把同一行文本从条带3挪到条带2，就被误判为"新内容"。`strip_distance_threshold / changed_strip_ratio` 都在对抗这个错位。
4. **拉普拉斯方差** 对整体模糊敏感，但对单向运动模糊（滚动拖影）不敏感——滚动中截图虽然"内容糊了"，但梯度方差可能并不低。

**结论：这三个模块需要的不是更好的阈值，而是携带语义的结构化特征。** 方案如下。

***

## 2. 算法改造一：行指纹序列对齐去重 / Rework 1: Row-Fingerprint Sequence Alignment (RFSA)

### 2.1 动机 / Motivation

滚动录屏的本质是**刚体垂直平移**：同一段文本在不同帧只发生 y 方向位移，内容不变。当前条带dHash按固定位置切条带，滚动后同一行落到不同条带，被误判为新内容。

如果把"水平切条带 → 每条带算指纹"看成把一帧编码成一个**行指纹序列**，那么：

* 滚动 = 序列的位移（offset）

* 新消息 = 序列的插入（局部新增）

* 静止 = 序列相同

两帧比较退化为**两个行指纹序列的对齐**（如最长公共子序列/编辑距离）。位移天然免疫，无需任何阈值。

### 2.2 方法 / Method

**Step 1: 行指纹提取 / Row fingerprint extraction**

ROI降采样灰度（160宽）按行切细条带（条带高 ≈ 一个文本行高，配置 `rf_strip_height`，默认16px），每条带计算64位dHash → 整帧得到指纹序列 `S = [f_1, f_2, ..., f_N]`。

**Step 2: 序列对齐 / Sequence alignment**

用带gap罚分的动态规划求两帧指纹序列的全局对齐（对齐即允许位移）：

```
D[i][j] = max(
    D[i-1][j-1] + match(f_i, g_j),   # 对齐(匹配) / match
    D[i-1][j]   + gap_penalty,       # 删除 / delete (只在S中)
    D[i][j-1]   + gap_penalty        # 插入 / insert (只在G中)
)
match(f, g) = 相似度 = 1 - hamming(f,g)/64
```

对齐结果输出：

* `displacement`：序列整体位移（滚动距离的指纹级近似）

* `new_rows`：未对齐的行索引（真正的新内容位置）

* `aligned_ratio`：对齐行占比 → 去重置信度

**Step 3: 三档判定 / Three-tier verdict**

| 条件 / Condition                    | 判定 / Verdict     |
| --------------------------------- | ---------------- |
| `aligned_ratio ≥ dup_ratio`（如0.8） | 重复，跳过OCR         |
| `aligned_ratio ≤ new_ratio`（如0.4） | 新内容，送OCR         |
| 中间 / between                      | 灰色地带，送OCR + 标记滚动 |

**Step 4: 替代关系 / What it replaces**

* 替代 `visual_dedup.py` 中"行差分剖面sum + 条带dHash + changed\_strip\_ratio"这条灰色地带精筛链

* 行差分剖面仍保留做**粗筛**（快速拒绝明显重复），但不再用sum()，而是用剖面形状的离散度

* 删除的超参数：`strip_count, strip_distance_threshold, changed_strip_ratio, dup_tight_ratio, new_loose_ratio`

* 新增超参数：`rf_strip_height, rf_gap_penalty, rf_dup_ratio, rf_new_ratio`（均有语义，非"拍脑袋"）

### 2.3 复杂度 / Complexity

指纹序列长 N ≈ 高/16 ≈ 10\~15。DP O(N²) ≈ 225次汉明距离计算，每次64bit XOR，开销可忽略（<0.1ms/帧）。

### 2.4 边界情况 / Edge cases

* **回弹**（果冻回弹回到旧位置）：序列与上一关键帧相同 → 直接判dup，正是目标

* **滚动中途半截帧**：序列位移到一半 → aligned\_ratio 中等，送OCR但滚动置信度过滤兜底

* **气泡颜色变化但文本相同**：dHash 对内容敏感，气泡背景色不同会拉低 match。缓解：用 `rf_gap_penalty` 容忍，或对每行做"去背景"预处理（后续可加）

***

## 3. 算法改造二：相位相关位移估计 / Rework 2: Phase-Correlation Motion Estimation

### 3.1 动机 / Motivation

T1/T2自适应分位数是"用历史速度分布猜当前状态"，这在滚动密集视频中必然漂移（T1 2.0→23.2）。根本原因是**MAE标量不携带运动方向信息**。

滚动是刚体垂直平移，帧间关系可以用**位移量 (dx, dy)** 完整描述。相位相关（Phase Correlation）直接估计位移，不依赖任何阈值猜状态。

### 3.2 方法 / Method

**Step 1: 计算位移 / Compute displacement**

对两帧降采样灰度 `I_prev, I_cur` 做FFT相位相关：

```
F1 = FFT(I_prev), F2 = FFT(I_cur)
R = F1 * conj(F2) / |F1 * conj(F2)|   # 归一化互功率谱
r = IFFT(R)
(dx, dy) = argmax(r)                   # 峰值位置 = 位移
peak_value = r[dy, dx]                 # 峰值强度 = 内容一致性
```

**Step 2: 峰值强度 = 内容一致性 / Peak = content coherence**

相位相关自带第二个信号：**峰值强度**。位移峰越尖，说明两帧是"纯平移关系"（滚动/静止），越平说明是"内容替换"（新消息/翻页）。这直接替代"用MAE猜滚动"。

| 条件 / Condition                 | 语义 / Meaning         | <br />  | <br />              |
| ------------------------------ | -------------------- | :------ | :------------------ |
| `peak ≥ peak_motion`（如0.5）且 \` | dy                   | < 1px\` | 静止（static）          |
| `peak ≥ peak_motion` 且 \`      | dy                   | ≥ 1px\` | 滚动（scrolling），dy即位移 |
| `peak < peak_motion`           | 内容替换（content change） | <br />  | <br />              |

**Step 3: 替代关系 / What it replaces**

* 替代 `frame_sampler.py` 的 T1/T2 分位数状态分类：`t1 = is_scrolling_by_phase()` 不再是分布猜的，而是运动模型估计的

* settle式快照的"运动段判定"改用 `peak < peak_motion 或 |dy| ≥ 1px`

* 滚动帧抽稀/滚动置信度过滤的触发条件同样替换

* 删除/降级：`static_percentile, active_percentile, t1_cap_ratio`（T1/T2 保留为诊断日志，不再参与判定）

### 3.3 复杂度 / Complexity

降采样 240×180 灰度图，FFT2D ≈ 240×180×(log) 约 0.5ms/帧，可接受。可用 `cv2.dft` 或 numpy FFT。

### 3.4 边界情况 / Edge cases

* **局部滚动（页面内嵌滚动区）**：整体平移假设被破坏，peak 下降 → 落入"内容替换" → 走新内容路径，保守保召回

* **滚动+新消息同时发生**：peak 中低 → 保守送OCR

* **位移超过图像尺寸**：不可能（单帧位移远小于帧高）

***

## 4. 算法改造三：梯度方向熵运动模糊检测 / Rework 3: Gradient-Direction Entropy (GDE) Blur Detection

### 4.1 动机 / Motivation

滚动拖影是**单向运动模糊**：梯度沿运动方向被拉长。静止清晰帧的梯度方向分布均匀（各向异性低），运动模糊帧的梯度方向高度集中（各向异性高）。

拉普拉斯方差对"整体失焦"敏感，但对"单向拖影"不敏感——实测滚动中截图拉普拉斯方差可能 ≥ 815（未被检出）。梯度方向熵直接测量各向异性。

### 4.2 方法 / Method

**Step 1: 梯度方向直方图 / Gradient direction histogram**

对降采样灰度图计算 Sobel 梯度，统计方向角的直方图（36个bin）：

```
gx, gy = Sobel(gray)
angle = atan2(gy, gx)  # 量化到36个bin
hist = 加权直方图(权重 = 梯度幅值)  # 弱梯度噪声权重低
```

**Step 2: 方向熵 / Directional entropy**

```
p_i = hist[i] / sum(hist)
entropy = -Σ p_i * log(p_i)     # 最大值 = log(36)
gde_score = 1 - entropy / log(36)   # 归一化: 0=各向同性(清晰), 1=单向(拖影)
```

**Step 3: 判定 / Verdict**

复用现有 `blur_detector.py` 的相对基线机制：维护清晰帧 GDE 的滑动窗口基线，当前帧 `gde_score` 超过基线×系数 → 运动模糊。`finalize_calibration` 用预热期清晰帧校准基线。

判定为"模糊"的条件改为：

```
is_blurry = (laplacian_score < lap_threshold) OR (gde_score > gde_threshold)
```

### 4.3 替代关系 / What it replaces

* 新增到 `blur_detector.py` 作为第二通道，与拉普拉斯方差 OR 融合

* 新参数：`gde_bins(36), gde_ratio(如1.5倍基线), gde_compute_width`——均有物理意义

* 现有 `window_size, blur_ratio, min_absolute_score` 保留（拉普拉斯通道不变）

### 4.4 边界情况 / Edge cases

* **纯色页面（几乎无梯度）**：hist 接近零 → 熵不稳定。加幅值下限保护：总梯度幅值过低时跳过GDE，回退拉普拉斯

* **文字密集页面**：静止时梯度方向均匀 → 熵高 → 不误判

* **滚动拖影**：垂直拖影 → 方向集中在垂直轴 → 熵低 → 正确检出

***

## 5. 架构与接口设计 / Architecture & Interface

### 5.1 新文件 / New files

```
src/
  row_fingerprint.py     # RFSA 行指纹序列对齐 (新 / new)
src/utils/
  motion_utils.py        # 相位相关位移 + 梯度方向熵 (新 / new)
```

### 5.2 接口设计 / Interfaces

**`src/utils/motion_utils.py`**

```python
def phase_correlation_displacement(prev_gray, cur_gray):
    """返回 (dx, dy, peak_value) / Return displacement & coherence peak"""
    ...

def gradient_direction_entropy(gray, bins=36):
    """返回 0=各向同性 ... 1=单向 / Return directional concentration [0,1]"""
    ...
```

**`src/row_fingerprint.py`**

```python
class RowFingerprintAligner:
    """行指纹序列对齐去重器 / Row-fingerprint sequence alignment deduper"""
    def compare(self, roi_frame) -> RowAlignResult:
        """与上一关键帧比较, 返回位移/新行/对齐率 / Compare vs last keyframe"""
        ...

@dataclass
class RowAlignResult:
    displacement: int      # 序列位移(滚动距离) / Sequence displacement
    new_rows: List[int]    # 未对齐行索引 / Unaligned row indices
    aligned_ratio: float   # 对齐占比 / Aligned ratio
    verdict: str           # "dup" / "new" / "gray"
```

**模块改造 / Module changes**

| 模块 / Module        | 改动 / Change                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `visual_dedup.py`  | 灰色地带精筛改为 RFSA；`check_with_gray` 返回 `RowAlignResult` 并入 `DedupResult` |
| `frame_sampler.py` | 状态分类改用相位相关；T1/T2 降级为诊断日志                                             |
| `blur_detector.py` | 增加 GDE 通道，与拉普拉斯 OR 融合                                                |
| `pipeline.py`      | settle 运动段判定、滚动抽稀/置信度过滤触发条件改用新接口                                     |

### 5.3 兼容与回退 / Compatibility & Rollback

* 所有新算法用配置开关控制：`rf_enabled, pc_enabled, gde_enabled`（默认关闭，灰度切换）

* 配置文件中旧参数保留（`t1_cap_ratio` 等在 pc\_enabled 时忽略，便于A/B对比）

* 每阶段实施后跑冒烟测试 + 真实视频对比，通过才继续下一阶段

***

## 6. 实施计划 / Implementation Plan

| 阶段 / Phase | 内容 / Content                                               | 验证 / Verification                      |
| ---------- | ---------------------------------------------------------- | -------------------------------------- |
| Phase 0    | 本方案文档评审                                                    | 用户确认 ✅                                |
| Phase 1    | `motion_utils.py`: 梯度方向熵 + 相位相关（纯函数，无侵入）                   | 单测: 合成静止/滚动/拖影帧, 指标可分离 ✅             |
| Phase 2    | `row_fingerprint.py`: RFSA 去重器 + 接入 `visual_dedup.py` 灰色地带 | 单测: 回弹/滚动位移/新消息 三态; 冒烟测试 ✅           |
| Phase 3    | `pipeline.py` 接入相位相关判态, 替代 T1/T2 触发链                       | 冒烟测试15/15 + 专项测试PASS ✅; 双信号融合修复(慢速滚动残影) ✅      |
| Phase 4    | 全量回归: 冒烟测试 + htkf/xxr 真实视频 + 残影诊断                           | 见下方 Phase 4 实现记录 ✅ |

### Phase 3 实现记录 / Phase 3 implementation notes

* `frame_sampler.py`: `FrameData` 新增 `pc_dx/pc_dy/pc_peak/pc_state`; 逐帧与快进路径均计算相位相关(快进与静止参考帧比较); `pc_enabled/peak_motion/dy_scroll_px` 可配; 默认开启
* `pipeline.py` 新增两谓词:
  * `_is_motion_fd(fd)` = `pc_state != "static"`(启用时) → settle 运动段抑制判定
  * `_is_scroll_fd(fd)` = `pc_state == "scrolling"`(启用时) → 滚动抽稀/残影过滤闸门; content\_change 不抽稀立即送OCR保召回
* 消费线程 `scroll_frame` 判定改用 `pc_state == "scrolling"`, 内容替换帧用宽松过滤
* `config.yaml` 新增 `pc_enabled: true, peak_motion: 0.3, dy_scroll_px: 1.0`; 关闭即回退 MAE>=T1 旧逻辑

### Phase 4 实现记录 / Phase 4 implementation notes

**双信号运动判定修复（慢速滚动残影）**

* 问题: 相位相关把帧间位移<1px的慢速滚动判为 static(peak 高但 dy 小), 绕过 settle 抑制直接送 OCR, 产生含乱码行(如"三方用尸")
* 修复: `_is_motion_fd` 改为 `pc_state != "static" OR mae >= static_floor` 双信号 OR 融合, pc 判位移/内容替换, MAE 绝对地板补 pc 盲区(慢速滚动的子像素残影), 静止帧不受影响(mae≈0)

**真实视频回归结果 (A/B: 修复前后真实 API)**

| 指标 / Metric                    | htkf.mp4 修复前 | htkf.mp4 修复后 | xxr.mp4 修复前 | xxr.mp4 修复后 |
| ------------------------------- | ----------- | ----------- | ----------- | ----------- |
| 送OCR帧数 / Frames sent to OCR     | 59          | 33 (-44%)   | 52          | 11 (-79%)   |
| 送检帧 motion 占比 / motion ratio  | 含滚动帧       | 0% (全static) | 35%         | 0% (全static) |
| 内容缺口 / gaps                   | 0           | 0           | 0           | 0           |
| 跟踪文本行 / tracked lines        | 152         | 143         | -           | 30(对话完整)   |

* htkf 143 vs 152: 差异来自更彻底的去重(修复前 152 行含滚动残影重复行), 对话流完整
* xxr 全帧 486/603 (80%) 判 scrolling: 视频是逐条消息弹出导致的持续 1-3px 微滚动, 相位相关正确识别并抑制, 送检帧 100% static
* 冒烟测试 15/15 PASS; `analyze_output_fragments.py` 的"59% 截断"判定为误报(聊天行本就不以句号结尾, 非滚动残影)

**GDE 实验反证（不接入 blur 通道）**

* 合成帧上 GDE 可分离(纯平移各向异性低/拖影高), 但真实聊天 UI 上静止清晰帧 GDE=0.120 与拖影帧同量级——文本行水平方向本身各向异性, 方向熵基线被抬到不可分离
* 结论: 放弃 GDE 作为模糊第二通道(避免误杀静止帧); `gradient_direction_entropy` 保留为纯函数供参考; `tmp_gde_measure.py` 记录反证数据后已清理

### 双信号判定核心代码 / Dual-signal predicate

```python
def _is_motion_fd(self, fd: FrameData) -> bool:
    """运动段判定(双信号融合)"""
    if self._sampler.pc_enabled:
        return (fd.pc_state != "static" or fd.mae_value >= self._static_floor)
    return fd.mae_value >= self._sampler.t1
```

### 多关键帧RFSA增强 / Multi-keyframe RFSA (v0.3)

**问题**: RFSA 只与上一关键帧比较, 滚动回看时重现的旧行已在上一关键帧中滚出, 被误判新内容→重复OCR产生噪声行(如 htkf section3 "如禾个人位后有何题")

**方案**: `row_fingerprint.py` 维护最近 N 帧关键帧序列队列(`rf_history_len`, 默认3), `compare` 与所有历史关键帧对齐取最优。整屏回看与更早关键帧全匹配→dup; 真正的新内容与所有历史都不匹配→new。

**验证**:
* 合成盲区测试: 回看整屏旧内容 `dup` ✅(单关键帧为 gray); 全新内容 `new` ✅
* 真实 htkf 全量 API: 33帧/0缺口/148行, 与基准一致(灰色地带未触发, RFSA 不参与, 无副作用)
* **失败实验(v0.3前)**: 把 RFSA 前移到 `high_conf_new` 分支(高diff但匹配历史关键帧→dup) → 真实视频误杀 97% 内容(OCR 33→3帧, 行数 149→39)。根因: 匹配对汉明距离用"中位数×3"相对地板, 内容不同时中位数抬高→所有行都算匹配→恒判dup。**已回退**, 该分支保持直接送OCR
* **v0.3 修复后成功**: 将相对地板改为绝对噪声阈值 `rf_noise_floor=14`(同内容hd≤6/完全不同hd≥16) 后, RFSA 前移为主通道不再误杀, 详见下方"RFSA主通道"实现记录

**已知限制**: 部分行重现(旧行混在新画面)仍会产生 1-2 行 OCR 噪声/视频。修复需更强的行指纹判别力(非空白区指纹/16x16 dHash/行投影), 有误杀风险, 暂缓。

### RFSA主通道 + 绝对噪声地板 + 预热抽稀 (v0.3)

**根因链(OCR MCP逐帧实测实锤)**: htkf 37 张送检帧中 31 张内容重复(组A兑换记录屏12张 + 组E 23:59对话屏19张), 真正有增量的仅 6 张。
1. 预热期 rapid 帧绕过视觉去重: f37-f45 MAE=14-30 全部送检同一画面 9 次
2. 消息弹出使画面平移 1-2px, 行差分对位移敏感 → total_diff 巨大 → 全部误判 `high_conf_new`
3. RFSA 只在灰色地带生效, 从未被触达

**改动 1: 绝对噪声地板 / Absolute noise floor**
* `row_fingerprint.py`: `_noise_floor` 从"匹配对hd中位数×3相对地板"改为绝对阈值(默认14)。旧逻辑缺陷: 两帧内容不同时 hd 普遍大→中位数抬高→所有行都判匹配→aligned_ratio 恒 1.0→恒判 dup(真实帧实测对齐率全为1.0的根因)。实测同内容 hd≤6, 完全不同文本 hd≥16

**改动 2: RFSA 升级为主去重通道 / RFSA as main channel**
* `visual_dedup.py` `check_with_gray`: `high_conf_dup` 之后、`high_conf_new` 之前插入 Step 1.5。RFSA 判 dup → 位移免疫拦截(与任一历史关键帧对齐率≥0.8且无实质新行); 判 new → 更新关键帧并送OCR; 判 gray → 落回行差分阈值与旧条带dHash兜底。旧条带dHash仍作为RFSA禁用/判gray时的兜底

**改动 3: 预热期滚动抽稀 / Warm-up scroll thinning**
* `pipeline.py`: 滚动抽稀扩展到预热期 rapid 帧(`_warmup_done or pc_state=="scrolling"`), 按 `scroll_min_interval_seconds`(0.12s) 抽稀, 解决 f37-f45 同一画面 OCR 9 次

**验证(修复后真实API重跑)**
| 指标 / Metric | 修复前 | 修复后 |
| --- | --- | --- |
| 送OCR帧数 | 33 | **8 (-76%)** |
| 重复帧图片(OCR实测) | 31张/37张 | **3张/8张**(f37/f44/f284 rapid动画中间帧) |
| 去重跳过 | 低 | 65 |
| 内容缺口 | 0 | 0 |
| 跟踪文本行 | 148 | 141 |
| 冒烟测试 | 15/15 | 15/15 |
| xxr回归 | - | OCR 3帧/0缺口 ✅ |

* 剩余 3 张重复帧均为 rapid 动画中间帧(预热期0.12s抽稀边界残留2张 + 弹出动画中RFSA判gray保召回1张), OCR 结果由内容复用机制兜底, 不产生重复文本
* RFSA单测4场景全PASS: 回弹dup / 滚动位移免疫(gray) / 新消息检测(gray+new_rows) / 微平移同屏dup

**已知限制(本轮实测)**: dHash 行指纹对"条带内亚条带位移"(5-15px回弹/微滚动)敏感, 文本行在条带内移动使 hash 完全改变(hd 14-30)→误判new→多一次OCR调用, 由内容复用兜底(性能损失非正确性损失)。整条带位移(滚动)与亚像素位移(静止同屏)均正确免疫

### xxr blur 误杀修复 (v0.4)

**问题反馈**: htkf 结果正常, 但 xxr 提取结果信息量骤减(仅 3 帧 OCR / 27 行文本), 用户疑心"少了太多信息"。

**根因链(行为日志+OCR MCP逐帧实测实锤)**: xxr 稳态 11 个 settle 送检帧全被 blur 过滤。
1. 预热校准把 `min_absolute` 从 6.0 抬升到 5093(预热期 p5 分位), 而拉普拉斯方差与文本密度强相关: 认证记录/留白多的页面天然低分, xxr 后期 settle 帧 score=875~4090 << threshold=5092.99
2. 基线窗口只收"非模糊帧"→ 全滤则基线永久冻结在 5127.70(死锁)
3. 提取 probe 帧 OCR MCP 验证: f0087(40行/99.6%)、f0400(54行/98.5%)、f0588(36行/98.4%) 全部清晰可读 → **blur 误杀实锤**, 且这些是 settle 静止帧(文本稀疏≠模糊)

**修复内容**:
1. `pipeline.py`: blur 检查加 `is_motion` 条件——settle 静止帧(运动结束送检的稳态帧)跳过 blur 过滤。依据: settle 逻辑已保证到达 blur 检查的帧是运动结束后的静止帧, 对它们做 blur 过滤纯属误杀
2. `blur_detector.py`: `finalize_calibration` 不再用预热 p5 抬升 `min_absolute`(只预填基线窗口); `is_blurry` 窗口更新放宽为"分数≥绝对下限即入窗", 基线滑动跟随内容密度, 消除全滤死锁

**验证(修复后真实API重跑)**
| 指标 / Metric | 修复前 | 修复后 |
| --- | --- | --- |
| xxr 送OCR帧数 | 3 | **13** |
| xxr 模糊跳过 | 11 | **0** |
| xxr 跟踪文本行 | 27 | **198** |
| xxr 内容缺口 | 0 | 0 |
| htkf 回归(OCR帧/文本行) | 8/141 | **8/141** ✅ 一致 |
| 冒烟测试 | 15/15 | 15/15 |

* 行为日志佐证: 修复后预热日志为 `绝对下限保持 6.0, 基线预填10帧`(不再抬升), htkf 回归 `total_tracked_lines: 141`
* 拉普拉斯低分 = 文本稀疏/页面留白(settle 静止帧), 不参与 blur 判定; 运动段(滚动拖影)仍由 blur 正常过滤

### blur 泛化性重构: GDE双证据 + 启发式破锁 (v0.5)

**问题**: "修好xxr会不会弄坏htkf?" 的担忧本质是——单拉普拉斯判模糊在算法层面不可靠: 它与文本密度强相关, 基线和阈值都在"猜内容密度"而非"猜模糊", 泛化性依赖运气。三个视频实测证实耦合度:
| 视频 | 拉普拉斯(清晰帧) | 文本密度 |
| --- | --- | --- |
| xxr 后期稀疏页 | 875~4090 | 认证记录/留白多 |
| zy/htkf/xxr 对话页 | 5000~9000 | 长对话密集 |

**GDE(梯度方向熵)新特征**: Sobel 梯度方向角直方图(36bin)的香农熵。运动模糊(拖影)把边缘沿运动方向拉长 → 方向分布收敛 → 熵降; 文本密度只影响边缘数量, 不改变方向分布。实测: 三个视频清晰帧 GDE 稳定 3.25~3.44(与拉普拉斯 875~9000 完全无关), 方向性拖影 <2.6, 高斯模糊51=1.98, 纯空白=0。

**双证据 AND 判定(成本不对称原则)**: `blurry = 拉普拉斯低 AND GDE低`。
- 拉普拉斯=弱证据(感知降分, 对内容密度敏感); GDE=强证据(确认方向性拖影, 与内容密度解耦)
- 单证据一律放行(成本不对称: 宁多一次OCR, 不丢内容): 稀疏页拉普拉斯低但GDE高→放行; 拖影两证据齐→过滤; 高斯模糊(各向同性)靠拉普拉斯极降+GDE拦截
- 放行帧交给下游兜底链: 视觉去重 + OCR置信度(scroll_confidence_threshold) + 内容复用

**启发式破锁(blur死锁三道保险)**:
1. **连续全滤强制重置**: 连续5帧全判模糊 → 用最近分数重建基线(低分位, 反映当前内容密度) + 强制放行一帧(failsafe, 防永久过滤)
2. **窗口过期重置**: 窗口8s未更新(如长时间空白) → 清窗重填, 恢复后立即适应新内容
3. **EWMA平滑**: 重置时若历史不足, 用指数滑动平均作为初始基线

**验证(三个视频真实API回归)**
| 视频 | 改动前(基线) | 改动后 | 一致性 |
| --- | --- | --- | --- |
| zy(新视频, 泛化性实测) | OCR 18/294行/0缺口 | OCR 18/294行/0缺口 | ✅ 完全一致 |
| htkf | OCR 8/141行/0缺口 | OCR 8/141行/0缺口 | ✅ 完全一致 |
| xxr | OCR 13/198行/0缺口 | OCR 13/198行/0缺口 | ✅ 完全一致 |
| 冒烟测试 | 15/15 | 15/15 | ✅ |

* 行为日志佐证: 预热校准日志 `GDE地板 2.60`; 每帧check日志含 `gde=` 与 `lap_low/gde_low` 双证据明细
* 泛化性来源: (1)blur 只对 motion 帧生效(settle 静止帧物理上无运动模糊); (2)GDE 特征与内容密度解耦(绝对下限2.6, 无需自适应校准); (3)AND 双证据+破锁保证"永不永久过滤"

### 组装逻辑复杂度优化: O(n²)→O(n·k) (v0.6)

**问题**: 长视频(SVID_171907, 6314行)后处理耗时 536.6s 总量中近 468s 在 `_assemble_without_llm`。三处 O(n²)/O(n×m) 热点:
1. 滚动回看去重: 每行与全部已见行算编辑距离(文本相似度), 全量两两 O(n²)
2. `keep_line` 残影过滤: 包含检查 + 相似度检查, 同为全量两两
3. `nearest_stamp` 时间戳匹配: 每行全量扫描所有时间戳行 O(n×m)

**修复**:
1. **长度分桶索引 `_SeenIndex`(无损剪枝)**: 相似度 `1-ED/max_len` 且 `ED>=|len1-len2|`, 长度差达到 `(1-阈值)×max_len` 的两行相似度必然不达标 → 每行仅与长度接近的 O(k) 个候选精确比较。剪枝无损: 被剪掉的行在任何阈值下都不可能超过, 判定结果与全量比较逐字节一致。**截断保护**: 编辑距离在 200 字符处截断, 该下界失效 → 长度≥200 的行(本行或候选)回退全量包含, 仍无损
2. **包含检查走全量扫描**: `str.__contains__` 为原生 C 实现, 速度远快于编辑距离 DP, 不做剪枝, 与旧逻辑完全等价
3. **`nearest_stamp` 排序+二分**: 时间戳按 `(first_timestamp, bbox[1])` 排序, `bisect_left` 定位后仅检查"同帧全部(y破平) + 左右相邻各一个不同时刻"。dt 主导得分, 最近时间戳必在二分邻域内, 与全量扫描同结果

**并行化评估(结论: 组装主循环不宜并行)**: 组装是顺序依赖过程(`seen_index` 逐行累积, 后续判定依赖前面所有保留行), 且 Python GIL 下多线程无收益, 多进程要序列化共享状态(IPC 开销吞掉收益)。真正的并行在流水线阶段已经做了(OCR 并发 4 路提交)。优化后组装单线程 1.16s(6314行), 并行化没有必要性。

**验证**:
| 指标 / Metric | 改动前 | 改动后 |
| --- | --- | --- |
| 组装耗时(6314行基准) | 468s | **1.16s (~400x)** |
| 耗时随规模增长 | O(n²) | 近似线性(0.19s/0.65s/1.16s @ 1k/3k/6.3k) |
| 语义等价性 | - | 5组用例逐字段一致(重复/残片/0.95边界/同帧时间戳/长文本截断) |
| 冒烟测试 | 15/15 | **16/16**(新增 `assembly_equiv` 回归用例) |
| 长视频总耗时 | 536.6s | 366.1s(其中后处理≈4s, OCR缓存命中) |

* 等价值回归用例 `tests/smoke_test.py::test_assembly_equivalence`: 用全量比较的旧逻辑作参考实现, 对同一输入断言 sections 逐字段一致, 永久守护本次优化无损性
* 说明: 两次运行跟踪行数 6314 vs 6283 差异来自上游(OCR API 输出方差 + 缓存命中), 组装本身不改变输入行数(statistics.total_tracked_lines = 输入计数), 非组装逻辑问题

### settle 长滚动抽稀送检: gap 内容修复 + 零收益成本消除 (v0.7)

**问题反馈**: 20F5 视频 gap 处真实丢内容——帧531-534(MAE 50-80 快速滚动)被 settle 持续抑制, 协议 ts=30.25("一、在乙")与 ts=32.50("一、在揭作过程中")之间中间条款永久丢失。htkf 则相反: 放行后 20 次长滚动采样全为 scrolling 帧, 下游 `allow_new=False` 全部新增=0, 纯冗余成本。

**根因**:
1. settle 持续抑制: 快速滚动实为"离散跳变"(帧间MAE高), 中间帧本身清晰且携带滚动经过的中间内容, 全抑制则永久丢失
2. 消费端状态差异: `content_change` 按静止帧处理(宽松过滤、允许新建跟踪行), `scrolling` 帧启用滚动残影过滤且禁止新建跟踪行(`allow_new=False`)——scrolling 帧送检零收益(4405 实测 85 次采样中 72 次 scrolling 帧全部新增=0, 仅 13 次 static/content_change 帧产生 130 行新增)

**修复**(`pipeline.py` settle 长滚动分支): 超过 `settle_suppress_seconds`(0.4s) 后按 `scroll_min_interval_seconds`(0.12s) 抽稀放行, 但**只放行非 scrolling 帧**:
1. `pc_state != "scrolling"`(content_change 快速跳变 / static 降速静止帧) → 走正常流程(模糊/去重/内容复用/OCR), 保召回
2. `pc_state == "scrolling"`(真实滚动位移) → 继续抑制, 消除零收益成本
3. 短滚动(≤0.4s) 仍整段抑制、运动结束送 settled 帧; EOF 兜底送最后一帧

**验证(精化后五视频真实API回归)**
| 视频 | 长滚动采样(非scrolling) | scrolling零收益拦截 | OCR帧 | 跟踪行 | 缺口 | 关键验证点 |
| --- | --- | --- | --- | --- | --- | --- |
| htkf | 3 | 大量 | 12 | 147 | 0 | 采样 20→3, 零收益消除 ✅ |
| xxr | 0 | 0 | 13 | 198 | 0 | 与基线一致 ✅ |
| zy | 15 | - | 32 | 333 | 0 | 多捕获39行(效果优先) ✅ |
| 20F5 | 12 | 28 | 99 | 1000 | 1(重叠率标记, 内容完整) | 协议条款"一~六"完整恢复 ✅ |
| 4405 | - | - | 151 | 1280 | 1 | 缺口处"三方支付渠道"说明+链接完整 ✅ |
| 冒烟测试 | - | - | - | - | - | 15/15 PASS ✅ |

* 20F5 剩余 gap=1 为状态机重叠率标记(协议翻页相邻两屏重叠率 0.12<0.15 天然触发), 输出内容已完整, 非实际丢失
* 新增 `frames_scroll_sampled` 统计(采集端实际放行的非 scrolling 长滚动帧数); config `settle_suppress_seconds` 注释同步更新

### 帧价值估计: 统一送检决策替代状态硬分类 (v0.8)

**问题反馈**: settle 长滚动分支(`pc_state==scrolling 抑制 / 非scrolling 放行 + 按间隔抽稀`)仍是"状态硬分类"——每次遇到新场景(弹窗/toast/页面切换/慢滚动)都要补一个状态规则, 反复打补丁且互有 overlap。用户要求**一次性通用改造**: 用连续分数替代状态分类, 不造合成样本, 以现有真实视频作泛化验证集。

**方法**(新增 `src/frame_value.py` FrameValueEstimator, `pipeline.py` settle 运动段内每帧评估):

帧价值 `V = 新颖度 × 清晰度`, 单一连续尺子覆盖滚动/弹窗/toast/切换:
1. **新颖度 = max(内容替换证据, 位移免疫新条带比例)**:
   * 内容替换证据 `replace_evidence = clip(1 - pc_peak/0.5, 0, 1)`: 相位相关峰低 = 两帧非刚体平移 = 弹窗/toast/页面切换/新消息, 无需像素对齐直接强证据
   * 位移比例(滚动态): `novelty = |est_shift|/h`——滚动动画帧像素差异大+运动模糊, 条带对齐后 MAE 仍超阈值(实测 29-59), 逐条带判新必然饱和; 位移比例免疫这一切, 物理意义明确(滚半屏≈50%新内容才值得送检)
   * 条带对齐新颖度(静止/内容替换态): PC dy 引导位移搜索(中心=est_shift, 半宽±8px, clamp=max(60,h))对齐后, 超阈值条带比例——纯滚动对齐后只有滚入/滚出条带算新, 位移免疫
2. **清晰度**: GDE 闸门, 运动拖影帧价值归零
3. **决策即推进参考帧**: break/fallback 决策时立即 `update_reference` 并重置 est_shift, 防滚动段每帧持续 break
4. 高价值(`value ≥ value_high_threshold`)立即打破抑制送检; 兜底(距上次OCR超间隔且新颖度≥门槛且清晰)保召回防长滚动全抑制

**验证(七视频真实API回归, 与 v0.7 基线对比)**
| 视频 | OCR帧(v0.7→v0.8) | 跟踪行(v0.7→v0.8) | 缺口 | 关键验证点 |
| --- | --- | --- | --- | --- |
| 20F5 | 99→229 | 1000→1037 | 1→0 | 协议条款"一~六"完整 + 缺口归零 ✅ |
| htkf | 12→- | 147→125 | 0 | 兑换记录/信誉分/补分对话完整(去重更净) ✅ |
| xxr | 13→- | 198→197 | 0 | 与基线一致 ✅ |
| zy | 32→84 | 333→279 | 0 | 申请→借呗→网商贷→放心借→限额→贴息全流程完整 ✅ |
| 4405 | 151→416 | 1280→1326 | 1→0 | 退费咨询完整 + 缺口归零 ✅ |
| SVID_171653 | -→334 | -→1250 | 0 | 新基线 ✅ |
| SVID_171907 | -→2055 | -→6320 | 0 | 新基线(5.8min/16729帧/24min处理) ✅ |
| 冒烟测试 | - | - | - | 16/16 PASS ✅ |

* **发现的假阳性**: value_break 68 次(20F5)中 56 次为 static 态(高相关峰+低MAE静止帧被评 0.5-1.0 价值), 但全部被内容指纹复用/去重拦截(零 OCR 增量)——冷缓存下会浪费 OCR 调用
* **顶会调研结论**: CVPR/ICCV/ECCV 帧选择工作(HiMu/VSI/Flexible Frame Selection/KeyScore)均面向"给 MLLM 选语义相关代表帧", 目标函数与"OCR 完整提取+位移免疫+统一送检"不同; 录屏分析仅 MobileHCI'18(规则+ROI标注, 正是被替换的硬分类)。**无直接可借鉴方案, 假阳性修复暂缓**——被内容复用兜底, 零 OCR 成本, 效果优先
* config 新增 `frame_value_*` 参数组(见 config.yaml sampling 段); `frame_value_enabled: false` 可整体回退到 v0.7 状态闸门

## 7. 预期收益 / Expected Gains

* **送OCR帧运动中占比**：35% → 目标 <15%（相位相关精准识别滚动，settle只送静止帧）

* **乱码/半截行**：消除"位移误判新内容"类乱码（RFSA免疫位移）

* **超参数**：20+ → \~10个有物理语义的参数，且大部分可自动校准或固定

* **鲁棒性**：不再依赖"速度分布猜状态"，滚动密集/静止/翻页混合视频统一处理

***

## 8. 风险 / Risks

1. **相位相关对非刚体运动（局部滚动区）peak下降** → 落入"内容替换"保守路径，只损失召回优化空间，不丢内容
2. **RFSA 对"同内容不同气泡色"误判** → 留作已知限制，可用后续行级预处理解决
3. **GDE 对极低纹理页面不稳** → 幅值下限保护回退拉普拉斯
4. **重构侵入面大** → 每阶段独立开关、灰度切换、A/B对比，可单阶段回退

