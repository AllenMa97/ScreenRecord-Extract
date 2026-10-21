# 手机录屏内容提取 / Screen Recorder Content Extraction

基于CPU-friendly算法的手机录屏聊天记录/参数提取框架。
A CPU-friendly algorithm framework for extracting chat records
and parameters from phone screen recordings.

**双通道 + 无需训练数据 / Two channels, no training data**:
- **视觉文本通道**: 自适应抽帧 → ROI/模糊/视觉去重 → OCR → 空间感知行跟踪(跨帧投票)
  → 折行合并 → 组装, 输出带 `[发]/[收]` 与时间戳的结构化聊天记录;
  CV+OCR(+可选 LLM 仲裁/纠错), 全部为经典算法, **CPU 可跑、无 GPU 依赖**。
- **音频通道**(§12): 语音消息在屏幕上没有文字, 故用「VAD 切段 → 并发 ASR」拿转写,
  再用轻量 CV 检测出的**语音气泡事件**做时间锚点配对, 逐条注入聊天流水, 并支持
  Web 端逐条审听。
- **Web 可视化**: FastAPI + Vue3, 上传录屏即可看进度、聊天段落与逐条语音。

## 核心算法 / Core Algorithms

### 1. 保召回自适应步长采样 / Recall-First Adaptive-Step Sampling
采样是唯一无法被下游兜底的环节: 采集时漏掉的内容永远找不回, 采集过多
造成的冗余则可在OCR后轻松去重/合并(宁杀错勿放过)。
Sampling is the only stage downstream cannot recover; over-capture is cheap
(dedup/merge after OCR), under-capture is fatal.

为了兼顾召回与CPU, 采用**自适应步长**: / For recall and CPU efficiency, use adaptive step:
- 逐帧阶段: 每帧解码, 与上一帧比MAE; 连续静止超过确认帧数后进入快进。
  Per-frame phase: decode every frame; enter fast-forward after a static streak.
- 快进阶段: 跳过中间帧(仅grab不解码), 解码帧与静止参考帧比较; 一旦出现
  变化立即回到逐帧(变化信号永不放过)。
  Fast-forward: skip intermediate decode; any change snaps back to per-frame.
- 所有节奏参数相对视频帧率推导(静止确认0.7s, 最大步长1s), 不设绝对帧数;
  静止阈值T1由速度分布分位数在线自校准。
  Pacing derives from fps (confirm 0.7s, max step 1s); T1 self-calibrates
  from velocity distribution percentiles.

### 0. 预热/探索阶段 / Warm-up (Exploration) Phase
每次处理视频时, 先进入预热阶段: 收集统计量, 然后从观测分布一次性校准
各模块阈值, 再进入稳态(稳态保留在线漂移)。
Before steady state, the pipeline collects statistics and calibrates each
module's thresholds from observed distributions once (online drift kept after).
- 采样器: T1/T2 用完整速度窗口分位数校准(仅监测用) / Sampler: T1/T2 from velocity percentiles (telemetry)
- 模糊检测: 绝对下限提升到清晰帧Score的p5, 并预填基线 / Blur: floor = p5 of clear scores, baseline prefilled
- 视觉去重: 用"速度"弱标注分桶diff, p50作为dup/new阈值种子 / Dedup: velocity-bucketed diff percentiles seed dup/new thresholds
- 自终止: 帧数达标且见过内容变化才结束(兜底上限防全静止) / Self-terminating: min frames + saw change, capped for all-static video

### 2. 拉普拉斯模糊过滤(相对阈值) / Laplacian Blur Filter (Relative Threshold)
不设绝对阈值, 维护滑动窗口Score基线, 当前帧低于基线30%才判定模糊。
Maintains sliding window baseline; current frame below 30% of baseline is blurry.

### 3. 视觉去重(行差分剖面 + 行条带dHash) / Visual Dedup
**替代全局pHash的关键创新** / Key innovation replacing global pHash:
- 行差分剖面: 逐行计算像素差之和, 检测内容是否变化
- 行条带dHash: 将ROI分成8个水平条带, 每条独立计算dHash
- 解决了pHash在文本密集UI上布局相似导致误丢弃的问题

### 4. 空间感知文本行跟踪 / Spatial-Aware Text Tracking
**替代整帧LCS拼接的关键创新** / Key innovation replacing whole-frame LCS:
- 利用OCR返回的bbox坐标, 逐行跨帧跟踪
- bbox IoU + 文本相似度组合匹配
- 多帧投票纠正OCR错误: **字符级共识**(`utils/text_consensus.py`)而非整行多数表决
  —— 旧做法"错一个字就变成另一张票", 现按位加权投票, 逐位需严格多数才改动
- 利用空间信息判断哪些行是新内容
- **数字锚点守卫 / digit-anchor guard**: 模板消息(`订单SN0010已确认收款` vs
  `订单SN0014已确认收款`)文字部分完全一致、只有数字位不同 → 编辑距离相似度高达
  0.91, 任何相似度判据都分不开。聊天消息高度统一时, 滚动一整条会让**下一条**消息
  IoU≈1 地落进上一条的框, 休眠/宽限行便把它吸收并整行改判, 使后出现的消息继承更早
  的时间戳(输出顺序颠倒)。故再加一道判据: 观测与本行外观库**全部**文本都构成
  "骨架相似度≥0.90 但数字锚点不相容"时判为另一条消息, 拒绝吸收, 让它自建新行
  (`text_tracker.template_digit_guard_sim`, 置 0 关闭)。热行不受限 —— 其数字差异
  更可能是 OCR 噪声, 应留给投票层吸收。
  Template messages differing only in digits score ≈0.91 on edit-distance
  similarity; with uniform message heights a full-line scroll drops the *next*
  message into the previous box at IoU≈1, so a dormant/grace row absorbs it and
  re-votes the whole line — inheriting an earlier timestamp and inverting output
  order. The guard rejects an observation whose digit anchor conflicts with
  *every* text in the row's appearance library while the digit-stripped skeleton
  still matches (hot rows are exempt: their digit noise belongs to the voter).

### 5. OCR驱动的时序状态机 / OCR-Driven State Machine
**无需训练数据** / No training data needed:
- 用OCR返回的文本行数量和密度统计特征驱动状态转换
- NORMAL → SUSPENDED: 文本行数骤降
- SUSPENDED → RESUMING → NORMAL: 文本恢复 + 内容重叠检查
- 内容缺口检测: 相邻帧重叠率过低时标记gap

### 6. 双模态联动(方案A+B) / Dual-Modality Linkage (Plan A+B)
**视觉去重为主, LLM为仲裁和兜底** / Visual dedup primary, LLM as arbitrator and fallback:

- **方案A - 置信度软化**: 视觉去重输出三档置信度
  - HIGH_CONF_DUP: 高置信度重复, 跳过OCR(省算力)
  - HIGH_CONF_NEW: 高置信度新内容, 直接送OCR
  - GRAY_ZONE: 灰色地带, 送OCR + LLM语义仲裁
- **方案B - 在线阈值校准**: 文本判定结果反馈给视觉模块
  - OCR发现文本重复 → 收紧视觉阈值
  - OCR发现文本不同 → 放宽视觉阈值
  - 视觉模态逐渐适应当前视频特征

### 7. 语义去重器(LLM仲裁+OCR纠错) / Semantic Dedup (LLM Arbitration + OCR Correction)
- **灰色地带语义判定**: LLM对比当前帧与历史帧文本的语义重复
  - "今天天气真好" vs "今天天气真不错" → 语义重复
  - "我明天去找你" vs "我后天去找你" → 语义不同
- **OCR错误兜底纠错**:
  - 低置信度行: 在历史文本中查找最相似行替换
  - LLM纠错: 根据上下文推断正确文本
- **降级模式**: LLM不可用时用纯文本相似度判定

### 8. 内容组装(可选LLM增强) / Content Assembly (Optional LLM)
- 基础模式: 按时间顺序组装文本行
- LLM模式: 结构化输出 + 缺口标注

### 8.1 聊天收发标记 / Chat Send-Receive Markers
针对聊天记录类录屏, 无需layout detection即可区分消息收发方向:
For chat-record recordings, send/receive direction is classified without
layout detection (no extra CV model needed):
- 利用OCR行bbox的左右留白对比: 我方消息气泡靠右(距右边缘近)→ `[发]`;
  对方气泡靠左(距左边缘近)→ `[收]`; 居中内容(系统提示/卡片)不标记。
  Compare left vs right margins of each OCR line bbox: right-aligned = my
  message `[发]`; left-aligned = other party `[收]`; centered = unmarked.
- 每行时间戳取自视频时刻最近的时间戳行(如"23:53"), 无时间戳时用
  播放秒数`+Ns`兜底; 输出格式:
  Per-message timestamp is the video-nearest visible stamp; fallback `+Ns`
  (playback seconds) when none. Output format:
  ```
  - 23:46 [发] ：麻烦你了，帮我查一下什么情况
  - 23:46 [收] ：请提供您的会员账号给我，方便为您查询核
  - 23:46 ：五月5
  ```
- 配置见 `output.chat_format` (`enabled` / `center_margin` / `time_fallback`),
  ROI宽度由流水线按首帧检测结果自动注入, 无需手填。
  See `output.chat_format` config; ROI width is injected automatically.

### 8.2 滚动残影过滤 / Scroll-Garbage Filtering
持续上滑滚动时, 滚动帧(帧间MAE≥静止阈值)的OCR结果常含残影/错字/乱码行。
两级过滤 / Two-stage filtering:
- **OCR行级(滚动帧)**: 置信度阈值 0.5 → `ocr.scroll_confidence_threshold`
  (默认0.65), 并过滤含日文假名/短中文碎片的行; 滚动帧结果只匹配不新建,
  完整内容由后续静止帧创建。
  Line-level (scroll frames): confidence floor raised to
  `ocr.scroll_confidence_threshold` (0.65); drop kana/short-Chinese fragments;
  scroll frames are match-only.
- **组装级(所有帧)**: 滚动残影片段多为"单次观测(obs=1)"或"低置信多观测"。
  组合规则过滤(见 `_assemble_without_llm`):
  Assembly-level (all frames): scroll scraps are mostly single-observation
  (obs=1) or low-confidence multi-observation lines; combo rules drop them:
  - 单观测短碎片(≤2字, 如"光公") / short single-obs scraps
  - 单观测低置信(<0.70) / low-confidence single-obs lines
  - 单观测被已保留完整行包含或高度相似(>0.90) → 滚动断裂片段
    single-obs fragments contained in / highly similar to kept full lines
  - 多观测但平均置信度<0.90 → 残影多帧复现 / multi-obs low avg confidence
  高置信单观测错字行(如"二万用尸信言…")残留时, 建议开启LLM纠错。
  Residual high-confidence garbled lines are best handled by LLM correction.

### 8.3 滚动帧抽稀 / Scroll-Frame Thinning
实测发现: 滚动动画帧的像素差异大(MAE 14–31), 行差分/条带dHash/平移对齐等
**视觉方法都无法识别"内容相同"**(滚动残影使像素真正不同), 只能OCR后才知道。
因此视觉去重在快速滚动段会逐帧判NEW→逐帧送OCR, 产生大量冗余调用
(实测268帧OCR中约70%是滚动帧)。
MCP实测(PP-OCRv6): 同一滚动段内f0039/f0041/f0044三帧OCR结果几乎完全相同
(均47行、置信度99%), 证实冗余。
Measured: scroll animation frames differ in pixels (MAE 14–31); visual methods
(row-diff/dHash/shift alignment) cannot detect "same content"; only OCR can.
PP-OCRv6 spot checks: frames f0039/f0041/f0044 in one scroll segment yield
nearly identical OCR (47 lines, ≈99% confidence).

解决方案: 预热(校准)完成后, 滚动帧(MAE≥T1)距上次OCR帧不足
`sampling.scroll_min_interval_seconds`(默认0.12s)则跳过OCR, 静止帧不受影响。
滚动中露出的新内容仍以≤间隔粒度被采样, 配合文本跟踪跨帧投票保证召回。
Fix: after warm-up calibration, scroll frames closer than
`sampling.scroll_min_interval_seconds` (0.12s) to the last OCR frame are
skipped; static frames are unaffected. Recall is kept by interval granularity
+ multi-frame text voting.

实测收益(htkf.mp4): OCR调用 268→120(-55%), 跟踪文本行 133→163(+23%),
内容无丢失, 反而恢复了信誉分页面菜单与关键警告。
Measured: OCR calls 268→120 (-55%), tracked lines 133→163 (+23%), no loss
(recovers account-menu and warning text previously dropped).

### 8.4 内容指纹OCR复用 / Content-Fingerprint OCR Reuse
**问题/Problem**: 录屏经视频编码逐帧压缩, 每帧噪声累加(≈1 MAE/帧)。即使画面
内容完全相同(静止等待/回弹), 相隔数帧后行差分/MAE也远超阈值, 像素级视觉去重
失效——f0304-f0329等多组"内容相同"的帧全部被误判为新内容。
Per-frame compression noise accumulates (≈1 MAE/frame). Even with identical
content (static waits/bounces), row-diff/MAE exceed thresholds a few frames
apart, breaking pixel-level visual dedup.

**方案/Fix**: 帧内容等价判定改用结构指纹——160宽灰度上的16x16 dHash:
- 同一内容: 汉明距离≤8-16; 不同内容: ≥19(实测, 无重叠区间)
- producer每帧计算指纹, 与已OCR内容缓存(限512条+最近帧快路径)比对;
  命中(hd≤`visual_dedup.content_reuse_max_hd`=12)则重放缓存OCR行, 跳过OCR调用
- 指纹条目与帧级缓存一起增量落盘(`output/cache/*.jsonl`), 跨运行复用

**实测(htkf.mp4, mock+真实缓存)**: 全新运行 OCR调用 120→61(-49%),
重放运行 0次调用/120帧全部复用; 复用前后最终输出逐行对比完全一致
(内容保全验证通过, 见 `scripts/recall_check.py` / `scripts/compare_pre_post.py`)。
Measured: fresh run OCR calls 120→61 (-49%); replay run 0 calls, 120 reused;
pre/post-reuse outputs are line-identical (recall verified).

### 9. 并行OCR流水线 / Parallel OCR Pipeline
将串行"采样→OCR→下游"改为生产者-消费者流水线, 只并行网络等待, 不破坏顺序依赖:
Convert the serial "sample→OCR→downstream" flow into a producer-consumer
pipeline. Only the network wait is parallelized; order dependencies are kept.
```
producer(采集线程) ──帧判定──> consumer(主线程, 按帧序消费)
      │ 需要OCR的帧                text_tracker → 灰色地带LLM → 状态机 → 组装
      v
submit_queue ──> worker池(N路, 提交jobId) ──> poller(单线程, 统一轮询) ──> 结果缓冲(按帧号)
```
- 采集(读帧/模糊/去重)有视觉状态依赖, 保持串行, 但毫秒级, 不是瓶颈
- OCR是唯一秒级瓶颈: AI Studio异步Job提交即返回, 可同时N个在途(并发路数 `ocr.concurrent`)
- 下游(text_tracker/状态机)有帧序依赖, 保持按序消费: 即使OCR乱序完成, 结果按frame_id排序后依序处理
- 状态恢复时的keyframe重置: 消费线程请求, 采集线程用灰度缓存执行, 避免跨线程状态竞争
- 本地/mock模式退化为单worker(本地OCR不并发); API模式默认4路并发

### 10. 断点续传(OCR增量落盘) / Checkpoint: Incremental OCR Cache
处理过程中的OCR结果是最贵的产物, 因此增量落盘, 崩溃后可断点续传:
The most expensive product is the OCR result, so it is persisted incrementally.
- consumer每消费一个"新OCR帧", 就把`(fid, ts, lines)`追加写一行到
  `output/cache/{视频名}_{指纹}.jsonl`(指纹=文件名+大小+修改时间)。
  Consumer appends each new OCR frame to a per-video JSONL cache file.
- 重新运行时自动加载缓存, worker对已缓存的帧跳过OCR调用直接重放结果;
  下游(跟踪/状态机/组装)是纯计算, 从首帧重放(毫秒级)。
  On rerun, cached frames skip OCR; downstream is pure compute, replayed fast.
- 缓存保存原始OCR结果, 灰色地带LLM仲裁在重放时重新执行。
  Raw OCR lines are cached; gray-zone arbitration reruns on replay.

### 11. 一头一尾同步OCR + 分片策略(设计方案) / Head+Tail OCR & Sharding (Design)
应对API限流(429)与本地OCR吞吐的策略方案, 尚未实现, 以下为设计。
Strategy design for rate-limited API (429) and local-OCR throughput. **Not yet implemented.**

**动机/Motivation**:
- 429限流: API并发低; 双流(头尾各一条producer)可翻倍在途OCR, 分片再乘k倍。
  Rate limit: two producer streams double in-flight OCR; sharding multiplies by k.
- 本地OCR: 吞吐不再是瓶颈, 但同一内容被多帧反复OCR仍是浪费; 内容指纹是跨时间
  的内容等价判定器, 头尾两侧共享同一缓存 → 任何一侧OCR过, 另一侧自动复用。
  Local OCR removes throughput limits, but repeated OCR of identical content
  remains waste; the shared content fingerprint makes the two sides dedup each other.
- 重复内容判定: 同一屏幕内容被头尾两侧分别观测 → 独立二次观测, 边界区可做
  时序一致性投票(OCR纠错), 这是纯视觉去重做不到的文本层增强。
  Both sides observing the same screen gives independent second observations
  → temporal-consistency voting at the overlap boundary (text-layer enhancement).

**先验/Heuristics (中间区域)**:
1. 时间戳文本锚点: OCR行中的 `HH:MM`(如23:46)是消息绝对时间锚; 头侧报
   "已覆盖到T_h", 尾侧报"已覆盖到T_t"; 单调聊天中 T_t ≤ T_h 即内容流全覆盖
   → 可提前终止(文本层碰头, 比视觉层更可靠)。/ Timestamp-text anchors:
   monotonic chat is fully covered when tail-anchor ≤ head-anchor.
2. 单调滚动先验: 正常聊天按时间单向滚动; 若尾侧时间戳 < 头侧时间戳(乱序),
   说明发生回弹/回翻 → 该区逐帧采样, 由内容指纹兜底去重。/ Monotonicity:
   out-of-order anchors mean bounce/revisit → dense scan, fingerprint dedups.
3. 静止屏先验: 首尾静止屏(长时间MAE≈0)内容不变, 两侧各OCR一次后直接快进。
   / Static bookends: OCR once per side, then fast-forward.
4. 滚动速率估计: 由已扫区估计px/s, 推算中间区覆盖所需时长, 指导提前终止。
   / Scroll-rate estimate guides early termination of the middle.

**架构/Architecture (两阶段落地/Two-phase rollout)**:
- 阶段1 分片并行(低风险): 视频按时间等分k片(边界重叠≈1s), 每片独立跑现有
  正向流水线, 共享内容指纹缓存 → 跨片同屏自动去重; k个进程/实例并行。
  Phase 1 (low risk): shard the video, run the existing forward pipeline per
  shard in parallel, share the content-fingerprint cache for cross-shard dedup.
- 阶段2 片内头尾碰头(需改造): FrameSampler支持反向seek; 头尾两条producer
  共享指纹缓存/帧级缓存; 消费端按fid做合并排序(reorder buffer), 下游
  (tracker/状态机)保持时序不变量; 碰头判据 = 指纹命中对方侧 或 文本锚闭合。
  Phase 2 (needs refactor): backward sampler + two producers + consumer-side
  reorder buffer; meeting = fingerprint hit across sides or anchor closure.

**评估/Evaluation**:
- OCR调用数(已有 `ocr_calls_made` / `ocr_content_reused`) / OCR calls
- 内容保全(已有 `scripts/recall_check.py`) / content recall
- 覆盖曲线与碰头提前终止节省(`scripts/measure_meet_middle.py`): 实测htkf/xxr
  均为纯单调滚动(回访率0%), 碰头终止无法省覆盖, 但双流并行与分片并行收益不变;
  含回翻/重开的录屏(回访>0)才有提前终止收益。
  Coverage curves & early-termination saving: measured htkf/xxr are purely
  monotonic (0% revisit) — head+tail termination saves nothing there, but the
  dual-stream and sharding parallelism still win; revisit-type recordings gain
  early termination.
- 429率 / 墙钟时间 / wall-clock time

### 12. 音频通道(语音消息转写) / Audio Channel (Voice-Message Transcription)
聊天记录里的**语音消息在屏幕上是没有任何文字的**: 只有一个带波形图的气泡和时长
标签, OCR 通道读不到内容。音频通道把视频音轨变成"第几条语音说了什么", 再按时间
锚点插回聊天流水。它与视觉通道**解耦并行**, 不改动既有采样/去重/OCR 流程。
Voice messages carry no on-screen text — only a waveform bubble plus a duration
label. The audio channel turns the track into per-message transcripts and
injects them back by time anchor. It runs decoupled from the visual channel.

```
视频 ─┬─ 视觉通道: 采样 → OCR → 行跟踪 → 组装 ────────────────┐
      │                                                       ├→ report_merge → 报告
      └─ 音频通道: decode → VAD切段 → 并发ASR                 │
                    + 语音气泡扫描(视觉形状) → VoiceEvent ─────┘
```

1. **音轨解码与 VAD 切段 / Decode + VAD**(`src/audio/audio_loader.py`, `vad.py`)
   PyAV 解码音轨 → 16kHz 单声道 float64 PCM。**ASR 服务端不返回时间戳**, 故时间轴
   完全由客户端切段决定: 自适应能量阈值(能量最低 2% 的均值作噪声基底 × 10, 不硬编码
   绝对音量) + 进入语音连续 3 帧越阈去抖 + 退出 hangover 拖尾 + 过滤孤立微段 +
   合并相邻段。numpy 向量化, 确定性。
   The ASR service returns no timestamps, so the timeline comes entirely from
   client-side adaptive-energy VAD (noise floor = trimmed-min 2% × 10), with
   debounce on entry, hangover on exit, micro-segment filtering and merging.

2. **逐段并发 ASR / Concurrent per-segment ASR**(`transcriptor.py`, `asr_client.py`)
   OpenAI 兼容 `POST {base_url}/audio/transcriptions`(默认 `sensevoice-small`),
   `concurrency` 路并发; 超过 `max_segment_seconds`(默认 360s)的段硬切细分 ——
   实测单请求音频 >25MB 会让服务端 CUDA OOM。

3. **轻量语音气泡检测 / Lightweight voice-bubble detection**
   (`src/visual/voice_bubble_detector.py`) 纯启发式、CPU-friendly、无深度学习依赖:
   - **背景自适应掩码(主力, 颜色无关)**: 估计背景(全局中位数 + 逐行中位数)后按色差
     经 Otsu 切前景 → 对图片壁纸与任意气泡配色(非绿/非白)都成立;
   - 颜色域掩码(绿/白)只作**辅助加分**, 不再是存在性或方向的必要条件;
   - 框内提取"墨水掩码", 用**波形 vs 文字**的形态差异算 `voice_score`(波形 = 底部
     对齐的细竖条、横向窄带 → 高分; 汉字多行笔画 → <0.5);
   - **方向以水平位置为主判据**: 气泡中心 x 在会话区中线右侧 → `[发]`, 左侧 → `[收]`;
     仅当落在中线死区时才回退用颜色。另用**宽/高双判据尺寸门**滤掉窄条装饰。

4. **气泡事件归并 / Bubble-event merging**(`voice_bubble_scanner.py`)
   用 cv2 **重开视频做独立的第二遍扫描**(默认每 0.2s 抽样一帧), 跨帧归并出
   `VoiceEvent`(出现时刻 / 时长 / 方向), 作为把音频配对回聊天记录的时间锚点。
   匹配走两级: 静止/小抖动用 IoU≥0.35; 滚动时先做**全局平移估计**(Δy 聚类投票,
   ≥3 票)与**相位相关静止帧判定**, 再走平移兜底(宽高容差 0.25、中心 x 容差 0.15、
   纵向位移上限 2.5 倍气泡高)。位移上限同时挡住"自身气泡已滚出屏幕的陈旧状态"去认领
   屏幕另一端的新气泡(否则新气泡不发事件, 其音频会配到旧气泡的收发方向上)。
   同一采样帧内的多个候选**必然是不同物理气泡**, 不按几何签名合并。

5. **OCR 时长标签双源确认 / Duration labels from the OCR cache**
   (`ocr_voice_scanner.py`)
   不再独立计数 —— 全局 `mm:ss` 正则会把界面时钟/进度数字误判为语音气泡(曾在含实时
   时钟的录屏上错误写出 88 条, 真实仅 25 条); 改为基于流水线**已落盘的 OCR 缓存**
   输出每个候选时长标签的原始观测(时间/文本/框), 只作视觉气泡**框内**的佐证。
   两种真实写法都认: ① `mm:ss`; ② **不带冒号的 `N"`**(实测主流微信渲染就是这种,
   引号还可能被 OCR 读成 `”`/`’`/`′`/`″`, 气泡被屏幕边缘裁切时会读成 `…4"`)。
   领域约束: 秒位 ∈ [1,59] 且总时长 ≤60s(微信单条语音上限), 直接剔除时钟数字 ——
   实测 907 个 `mm:ss` 命中里有 468 个 >60s。

6. **逐条切音与转写 / Per-voice slicing**(`voice_extractor.py`)
   发声窗 = `[气泡timestamp, timestamp + 语音时长]`(时长优先取 OCR 标签; 缺失时
   回退气泡停留时长, 但后者可能含非语音留白)。逐条存 WAV 到 `output/voices/` 并
   逐条转写, 为下游 NLP / 声纹建档提供干净输入。

7. **合并注入与跨路去重 / Merge, inject, cross-path dedup**(`report_merge.py`)
   - `result["audio"]`: 完整逐段转写(视频时间轴锚点);
   - `result["voice_events"]`: 视觉气泡 × 音频段按时间就近配对(`match_gap_s`),
     未配对的(静音/被提前滚走)保留空文本;
   - 注入聊天段落 `- +Ns [语音][发/收] ：text`, **按转写段逐句注入**而非把整簇拼成
     一行 —— 一个气泡的时间窗常横跨相邻若干条消息, 拼接会把多条粘成一条;
   - 注入前做**视觉 × 音频跨路去重**(屏幕上已有同一句时不再注入): 归一化相似度
     ≥0.60 且长度比 ≥0.6, 并附两道保护 —— *共享前缀 + 尾段悬殊* 判为两条不同消息
     (只是开头相同); *数字锚点子序列* 放宽以容忍 ASR 多念/漏念一位数字(真·不同
     数字如 8812/8816 是替换关系, 不放行)。

8. **Web 端逐条审听 / Per-voice playback in the Web UI**
   (`web/backend/services/voice_audio.py`, `web/frontend/src/components/AudioInspector.vue`)
   提取完成后按 `voice_events` 切出逐条 wav, 把相对 URL 回填进
   `voice_events.audio_url`, 经 `GET /api/sessions/{sid}/voices/{name}` 提供,
   前端可直接定位并审听每条语音、逐条核对转写。

**实测 / Measured**
- 语音为主录屏(20 条语音): 收发方向 **20/20** 与 GT 逐条一致;
- 14 条实录版录屏: 音频通道 **6/6 = 100%**, 语音方向 **6/6**; 整体严格一对一
  F1=0.706、方向宏 F1=0.916(`tools/eval_voice_gt.py`, GT 见 `gt/`)。

## 项目结构 / Project Structure

```
├── main.py                        # 主入口(CLI) / Main entry
├── config.example.yaml            # 配置模板 / Config template
├── config.yaml                    # 实际配置(含密钥, 已 gitignore) / Real config (gitignored)
├── requirements.txt               # 依赖 / Dependencies
├── src/
│   ├── pipeline.py                # 并行流水线编排 / Parallel pipeline orchestrator
│   ├── config_loader.py           # 配置加载 / Config loader
│   ├── data_types.py              # 数据类型 / Data types
│   ├── frame_sampler.py           # 自适应抽帧 + 滚动帧抽稀 / Adaptive sampling + scroll thinning
│   ├── frame_value.py             # 帧价值估计(连续分数替代状态硬分类) / Frame-value scoring
│   ├── roi_manager.py             # ROI 管理 / ROI manager
│   ├── blur_detector.py           # 模糊检测 / Blur detection
│   ├── visual_dedup.py            # 视觉去重 / Visual dedup
│   ├── row_fingerprint.py         # RFSA 行指纹序列对齐去重 / Row-fingerprint sequence alignment
│   ├── ocr_client.py              # OCR 客户端(api/local/mock) / OCR client
│   ├── message_bubble_tracker.py  # 空间感知文本行跟踪 + 跨帧投票 / Text-line tracker + voting
│   ├── bubble_merger.py           # 气泡内折行合并/系统提示过滤/残片吸收 / Wrap merging & scrap absorption
│   ├── screenshot_filter.py       # 截图消息整簇折叠 / Screenshot-message cluster folding
│   ├── state_machine.py           # OCR 驱动状态机 / OCR-driven state machine
│   ├── semantic_dedup.py          # 语义去重(LLM 仲裁 + OCR 纠错) / Semantic dedup
│   ├── content_assembler.py       # 内容组装 / Content assembler
│   ├── report_render.py           # md/html 报告渲染 / Report rendering
│   ├── llm_client.py              # LLM 客户端(可选) / LLM client (optional)
│   ├── audio/                     # ★ 音频通道 / Audio channel (§12)
│   │   ├── audio_loader.py        #   音轨解码/切段(PyAV) / Track decode & slicing
│   │   ├── vad.py                 #   自适应能量 VAD / Adaptive energy VAD
│   │   ├── asr_client.py          #   ASR HTTP 客户端 / ASR client
│   │   ├── transcriptor.py        #   逐段并发转写编排 / Concurrent transcription
│   │   ├── voice_bubble_scanner.py#   语音气泡事件归并 / Bubble-event merging
│   │   ├── ocr_voice_scanner.py   #   OCR 时长标签佐证 / Duration-label evidence
│   │   ├── voice_extractor.py     #   逐条切音 + 转写 / Per-voice extraction
│   │   ├── report_merge.py        #   视觉×音频合并注入 / Merge & inject
│   │   └── data_types.py          #   VoiceEvent / VoiceUtterance
│   ├── visual/                    # 纯 CV 视觉算子 / Pure-CV operators
│   │   ├── voice_bubble_detector.py#  轻量语音气泡检测 / Voice-bubble detection
│   │   ├── chat_scene.py          #   聊天界面判定 / Chat-scene detection
│   │   └── scroll_motion.py       #   帧级滚动位移估计 / Scroll-offset estimation
│   └── utils/
│       ├── image_utils.py         # 图像工具 / Image utils
│       ├── motion_utils.py        # 相位相关位移/梯度方向熵 / Phase correlation & GDE
│       ├── text_utils.py          # 文本工具(相似度/IoU/数字锚点) / Text utils (incl. digit anchor)
│       ├── text_consensus.py      # 字符级多观测共识投票 / Char-level consensus voting
│       └── behavior_logger.py     # 行为日志(jsonl) / Behavior logger
├── web/                           # FastAPI + Vue3 可视化 / Web UI
│   ├── run.py                     #   一键启动 / One-command launcher
│   ├── backend/                   #   会话管理 + 提取服务 + 逐条语音审听
│   └── frontend/                  #   Vue3 前端(含 AudioInspector)
├── tools/                         # 评测与运维脚本 / Eval & ops scripts
│   ├── eval_voice_gt.py           #   语音 GT 评测(P/R/F1、收发方向) / Voice-GT eval
│   ├── eval_synthetic.py          #   合成录屏批量评测 / Synthetic batch eval
│   └── eval_ocr_consensus.py      #   OCR 共识投票评测 / Consensus eval
├── scripts/                       # 内容保全/前后对比等辅助脚本 / Recall & diff helpers
├── gt/                            # 人工标注 GT(评测依据) / Hand-labeled ground truth
├── tests/                         # 单元与冒烟测试 / Unit & smoke tests
└── output/                        # 输出目录(已 gitignore) / Output (gitignored)
```

## 安装 / Installation

```bash
pip install -r requirements.txt          # 含音频通道所需的 av(PyAV)
pip install -r web/backend/requirements.txt   # 仅 Web 可视化需要(FastAPI/uvicorn)
```

PaddleOCR(本地模式, 可选) / PaddleOCR (local mode, optional):
```bash
pip install paddlepaddle paddleocr
```

## 使用 / Usage

### 基本用法 / Basic Usage
```bash
python main.py --video recording.mp4
```

### Web 可视化(推荐) / Web UI (recommended)
FastAPI + Vue3, 上传录屏后在浏览器里看进度、聊天段落、以及**逐条语音的审听与转写核对**:
```bash
python web/run.py --port 8000 --host 127.0.0.1
# Windows 也可直接双击 `web服务前后端一键启动脚本.bat`(停止: `web服务一键停止脚本.bat`)
```
打开 `http://127.0.0.1:8000`。首次使用需先构建前端(`web/frontend`: `npm i && npm run build`)。

### 评测 / Evaluation
以人工标注 GT(`gt/`)为准, 输出严格一对一 P/R/F1、收发方向混淆矩阵与漏检/冗余诊断:
```bash
python tools/eval_voice_gt.py <视频名> --gt gt/聊天14条_实录版.md \
    --out-json output/extracted_content_<视频名>.json
python tools/eval_synthetic.py          # 合成录屏批量评测
```

### 冒烟测试 / Smoke Test
```bash
python tests/smoke_test.py
```

### 报告转换 / Report Conversion

将 `output/extracted_content.json` 转为易读的 Markdown 和 HTML 报告:
```bash
python scripts/convert_report.py
# 可选: 指定输入文件/输出目录 / Optional: specify input/output
python scripts/convert_report.py --input output/extracted_content.json --output-dir reports
```
输出: `output/report.md` 和 `output/report.html`

## 配置 / Configuration

编辑 `config.yaml`:

### PaddleOCR API配置 / PaddleOCR API Config

使用PaddleOCR AI Studio异步任务API(URL + bearer TOKEN):
```yaml
ocr:
  mode: "api"              # "api"(AI Studio) / "local" / "mock"(测试用)
  api_endpoint: "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
  api_key: "你的TOKEN"     # AI Studio访问令牌(bearer TOKEN), 自行配置
  model: "PP-OCRv6"
  poll_interval: 5
```

### LLM后处理配置(可选) / LLM Post-processing Config (Optional)
```yaml
llm:
  enabled: true
  api_endpoint: "https://api.deepseek.com/v1/chat/completions"
  api_key: "你的API_KEY"
  model: "deepseek-chat"
```

### 聊天收发标记配置 / Chat Send-Receive Marker Config
```yaml
output:
  chat_format:
    enabled: true        # 是否生成[发]/[收]标记
    roi_width: 0         # 屏幕宽度(0=由流水线按ROI自动传入)
    center_margin: 20    # 居中地带(px), 两侧留白差小于该值则不标记
    time_fallback: true  # 无时间戳时用播放秒数(+Ns)兜底
```

### 音频通道配置 / Audio Channel Config (§12)
```yaml
audio:
  enabled: true
  base_url: "http://your-asr-server:port/v1"   # OpenAI 兼容 ASR 服务
  api_key: "your_api_key_here"
  model: "sensevoice-small"
  sample_rate: 16000
  max_segment_seconds: 360      # 超长段硬切上限(防服务端 OOM)
  concurrency: 2                # ASR 并发路数
  bubble_enabled: true          # 语音气泡视觉检测
  bubble_threshold: 0.55        # voice_score 门限
  bubble_sample_interval_s: 0.2 # 第二遍扫描抽样间隔
  bubble_merge_window_s: 1.2    # 同一气泡跨帧归并窗
  match_gap_s: 1.5              # 视觉气泡 × 音频段 就近配对的最大时间差
  inject_chat: true             # 是否把语音内容注入聊天段落
```

### 文本跟踪配置(节选) / Text-Tracker Config (excerpt)
```yaml
text_tracker:
  iou_threshold: 0.4              # bbox IoU 匹配阈值
  text_sim_threshold: 0.80        # 文本相似度匹配阈值
  iou_strong_threshold: 0.6       # "同一框"判定; 命中后文本门限放宽到下面这项
  text_sim_strong_threshold: 0.50 # 同一框内的放宽文本门限(兜 OCR 噪声)
  dormant_rematch_frames: 10      # 休眠超过该帧数的行需强空间证据才可吸收观测
  template_digit_guard_sim: 0.90  # 数字锚点守卫(§4); 置 0 关闭
  reattach_grace_seconds: 3.0     # 过期行重挂宽限, 消除 ID 切换
  adjacent_gap_ratio: 1.0         # 相邻行排他仲裁; 置 0 关闭
```

## LLM增强说明 / LLM Enhancement

LLM(如Deepseek)仅用于最终后处理阶段, 不进入实时流水线:
LLM (e.g. Deepseek) is only used in final post-processing, not in real-time pipeline:

1. **OCR错误纠正**: 利用跨帧上下文修正OCR识别错误
2. **结构化输出**: 将原始文本行组装为结构化聊天记录
3. **缺口标注**: 标注内容缺口位置

不启用LLM时, 系统使用基础组装模式(按时间排序输出文本行)。
When LLM disabled, system uses basic assembly (chronological text output).

## 算法参考 / Algorithm References

- 帧差法 (Frame Differencing): 经典视频摘要方法
- dHash (Difference Hash): 对图像梯度敏感, 适合检测文本位移
- 拉普拉斯方差: 自动对焦和图像质量评估经典方法
- Video Text Tracking: 使用appearance+geometry+text特征建立帧间对应关系
- Agent-SAMA (arXiv:2505.23596): 移动UI状态机建模
