# 手机录屏内容提取 / Screen Recorder Content Extraction

基于CPU-friendly算法的手机录屏聊天记录/参数提取框架。
A CPU-friendly algorithm framework for extracting chat records
and parameters from phone screen recordings.

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
- 多帧投票纠正OCR错误
- 利用空间信息判断哪些行是新内容

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
实测发现: 滚动动画帧的像素差异大(MAE 14~31), 行差分/条带dHash/平移对齐等
**视觉方法都无法识别"内容相同"**(滚动残影使像素真正不同), 只能OCR后才知道。
因此视觉去重在快速滚动段会逐帧判NEW→逐帧送OCR, 产生大量冗余调用
(实测268帧OCR中约70%是滚动帧)。
MCP实测(PP-OCRv6): 同一滚动段内f0039/f0041/f0044三帧OCR结果几乎完全相同
(均47行、置信度99%), 证实冗余。
Measured: scroll animation frames differ in pixels (MAE 14~31); visual methods
(row-diff/dHash/shift alignment) cannot detect "same content"; only OCR can.
PP-OCRv6 spot checks: frames f0039/f0041/f0044 in one scroll segment yield
nearly identical OCR (47 lines, ~99% confidence).

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
**问题/Problem**: 录屏经视频编码逐帧压缩, 每帧噪声累加(~1 MAE/帧)。即使画面
内容完全相同(静止等待/回弹), 相隔数帧后行差分/MAE也远超阈值, 像素级视觉去重
失效——f0304-f0329等多组"内容相同"的帧全部被误判为新内容。
Per-frame compression noise accumulates (~1 MAE/frame). Even with identical
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
- 阶段1 分片并行(低风险): 视频按时间等分k片(边界重叠~1s), 每片独立跑现有
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

## 项目结构 / Project Structure

```
├── main.py                    # 主入口 / Main entry
├── config.yaml                # 配置文件 / Configuration
├── requirements.txt           # 依赖 / Dependencies
├── src/
│   ├── config_loader.py       # 配置加载 / Config loader
│   ├── data_types.py          # 数据类型 / Data types
│   ├── pipeline.py            # 并行流水线编排 / Parallel pipeline orchestrator
│   ├── frame_sampler.py       # 自适应抽帧 / Adaptive sampling
│   ├── roi_manager.py         # ROI管理 / ROI manager
│   ├── blur_detector.py       # 模糊检测 / Blur detection
│   ├── visual_dedup.py        # 视觉去重 / Visual dedup
│   ├── ocr_client.py          # OCR客户端 / OCR client
│   ├── text_tracker.py        # 文本跟踪 / Text tracker
│   ├── state_machine.py       # 状态机 / State machine
│   ├── semantic_dedup.py      # 语义去重(LLM仲裁+OCR纠错) / Semantic dedup
│   ├── content_assembler.py   # 内容组装 / Content assembler
│   ├── llm_client.py          # LLM客户端(可选) / LLM client (optional)
│   └── utils/
│       ├── image_utils.py     # 图像工具 / Image utils
│       ├── text_utils.py      # 文本工具 / Text utils
│       └── behavior_logger.py # 行为日志 / Behavior logger
├── tests/
│   └── smoke_test.py          # 冒烟测试 / Smoke test
└── output/                    # 输出目录 / Output directory
```

## 安装 / Installation

```bash
pip install -r requirements.txt
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
