<template>
  <div class="status-stepper">
    <!-- 统一进度(主进度条, 前向单调不回跳) / unified forward-only progress -->
    <div class="unified-area">
      <div class="unified-head">
        <span class="unified-title">处理进度</span>
        <span class="unified-pct">{{ unified }}%</span>
      </div>
      <el-progress
        :percentage="unified"
        :stroke-width="15"
        :status="progressStatus"
        text-inside
        color="#4f8ef7"
      />
      <div class="unified-step">
        <el-icon :size="15" class="step-icon"><Promotion /></el-icon>
        <span class="step-text">{{ currentStep }}</span>
        <span v-if="running" class="step-elapsed">已用时 {{ elapsedLabel }}</span>
      </div>
    </div>

    <!-- 流程轴(横向 1→2→1 分合) / horizontal fork-join flow -->
    <div class="flow">
      <div class="flow-node" :class="nodeCls('prep')">
        <div class="node-dot"><el-icon v-if="nodeCls('prep') === 'node-done'"><Check /></el-icon></div>
        <div class="node-title">准备</div>
        <div class="node-desc">读入视频·定位区域</div>
      </div>

      <div class="flow-arrow"><el-icon><ArrowRight /></el-icon></div>

      <!-- 并行分支(视觉+音频同步) / two parallel branches -->
      <div class="flow-col branches">
        <div class="flow-node branch" :class="nodeCls('vision')">
          <div class="node-head">
            <div class="node-dot"><el-icon v-if="nodeCls('vision') === 'node-done'"><Check /></el-icon></div>
            <div>
              <div class="node-title">识别画面</div>
              <div class="node-desc">逐帧检测·OCR·去重</div>
            </div>
          </div>
          <el-progress
            v-if="running"
            :percentage="visionPct"
            :stroke-width="5"
            :show-text="false"
            class="branch-progress"
          />
        </div>
        <div class="flow-node branch" :class="nodeCls('audio')">
          <div class="node-head">
            <div class="node-dot"><el-icon v-if="nodeCls('audio') === 'node-done'"><Check /></el-icon></div>
            <div>
              <div class="node-title">识别音频</div>
              <div class="node-desc">{{ audioBranchDesc }}</div>
            </div>
          </div>
          <el-progress
            v-if="running && hasAudio"
            :percentage="audioPct"
            :stroke-width="5"
            :show-text="false"
            :status="audioStatus"
            class="branch-progress"
          />
        </div>
      </div>

      <div class="flow-arrow"><el-icon><ArrowRight /></el-icon></div>

      <div class="flow-node" :class="nodeCls('merge')">
        <div class="node-dot"><el-icon v-if="nodeCls('merge') === 'node-done'"><Check /></el-icon></div>
        <div class="node-title">整理结果</div>
        <div class="node-desc">合并·配对·组装</div>
      </div>

      <div class="flow-arrow"><el-icon><ArrowRight /></el-icon></div>

      <div class="flow-node" :class="nodeCls('done')">
        <div class="node-dot"><el-icon v-if="nodeCls('done') === 'node-done'"><Check /></el-icon></div>
        <div class="node-title">完成</div>
        <div class="node-desc">结果汇总</div>
      </div>
    </div>

    <!-- 详细进度(视觉/音频拆解, 面向关注内部细节的用户) /
         detailed split (vision/audio) for users who care about internals -->
    <el-collapse
      v-if="running || meta.status === 'done' || meta.status === 'error'"
      v-model="detOpen"
      class="split-collapse"
    >
      <el-collapse-item title="详细进度（视觉 / 音频）" name="det">
        <div class="split-row">
          <span class="split-label">视觉侧 OCR</span>
          <el-progress
            :percentage="visionPct"
            :stroke-width="8"
            :status="done ? 'success' : undefined"
            style="flex: 1"
          />
          <span class="split-sub">
            帧 {{ progress.frame_id ?? '-' }} / {{ progress.total_frames ?? '-' }}
          </span>
        </div>
        <div class="split-row">
          <span class="split-label">音频侧 听写</span>
          <el-progress
            :percentage="audioPct"
            :stroke-width="8"
            :status="audioStatus"
            style="flex: 1"
          />
          <span class="split-sub">{{ audioDetail }}</span>
        </div>
      </el-collapse-item>
    </el-collapse>

    <div v-if="running" class="run-info">
      <el-tag type="warning" effect="dark" size="small">提取中</el-tag>
      <span class="info-item">{{ currentDetail }}</span>
    </div>
    <div v-else-if="meta.status === 'done'" class="run-info">
      <el-tag type="success" effect="dark" size="small">提取完成</el-tag>
      <span class="info-item">{{ meta.summary || '' }}</span>
      <span class="info-item" v-if="meta.run_info">
        采样帧 {{ meta.run_info.sampled_frames }} ·
        OCR处理 {{ meta.run_info.ocr_processed }} ·
        去重跳过 {{ meta.run_info.dedup_skipped }}
      </span>
    </div>
    <div v-else-if="meta.status === 'error'" class="run-info">
      <el-tag type="danger" effect="dark" size="small">提取失败</el-tag>
      <span class="info-item">{{ meta.error || '未知错误' }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { Promotion, Check, ArrowLeft, ArrowRight } from '@element-plus/icons-vue'

const props = defineProps({
  sid: { type: String, default: '' },
  meta: { type: Object, required: true },
})

// 流程轴节点: 准备(0) → 识别画面/识别音频(1, 并行) → 整理结果(2) → 完成(3)
// flow nodes: prep(0) → vision/audio(1, parallel) → merge(2) → done(3)
const FLOW = {
  prep: { idx: 0, title: '准备' },
  vision: { idx: 1, title: '识别画面' },
  audio: { idx: 1, title: '识别音频' },
  merge: { idx: 2, title: '整理结果' },
  done: { idx: 3, title: '完成' },
}

// ------------------------------------------------------------------
// 原始(可回退)百分比 / raw (possibly-regressing) percentages
// ------------------------------------------------------------------
const progress = computed(() => props.meta.progress || {})
const audioProg = computed(() => props.meta.audio_progress || null)
const running = computed(() => props.meta.status === 'extracting')
const done = computed(() => props.meta.status === 'done')
const hasAudio = computed(() => audioProg.value != null)

// 视觉侧原始百分比(帧峰值 / 总帧)
const visionRaw = computed(() => {
  if (done.value) return 100
  const total = progress.value.total_frames || 0
  const maxFid = progress.value.max_frame_id ?? progress.value.frame_id ?? 0
  if (!total) return 0
  return Math.max(0, Math.min(100, Math.round((maxFid / total) * 100)))
})

// 音频侧原始百分比(已转写段数 / 总段数)
const audioRaw = computed(() => {
  if (!hasAudio.value) return null
  if (audioProg.value.phase === 'done') return 100
  if (audioProg.value.phase === 'error') return 100
  const total = audioProg.value.total || 0
  if (!total) return 0
  return Math.min(100, Math.round((audioProg.value.done / total) * 100))
})

// 统一原始百分比: 未开始=0; 准备占 0-6, 之后视觉/音频并行各占一半; 无音轨时视觉单轨
// unified raw: idle=0; prep 0-6, then vision+audio parallel half-half; vision-only if no audio
const unifiedRaw = computed(() => {
  if (done.value) return 100
  if (props.meta.status !== 'extracting') return 0    // 未开始不显示进度
  const v = visionRaw.value
  if (!hasAudio.value) return 6 + v * 0.94
  const a = audioRaw.value ?? 0
  return 6 + (0.5 * v + 0.5 * a) * 0.84
})

// ------------------------------------------------------------------
// 前向单调钳制: 显示值 = max(历史峰值, 新值), 杜绝进度回跳(含 total 被修订)
// forward-only clamp: shown = max(peak, incoming), no backward jump ever
// ------------------------------------------------------------------
const peaks = ref({ unified: 0, vision: 0, audio: 0 })
function clampPeak(key, raw) {
  if (done.value) {
    peaks.value[key] = 100
    return 100
  }
  const r = Math.round(raw)
  if (r > peaks.value[key]) peaks.value[key] = r
  return peaks.value[key]
}

const unified = computed(() => clampPeak('unified', unifiedRaw.value))
const visionPct = computed(() => clampPeak('vision', visionRaw.value))
const audioPct = computed(() =>
  hasAudio.value ? clampPeak('audio', audioRaw.value ?? 0) : 0
)

// 新会话 / 重新开始提取时重置峰值, 避免串到旧进度
// reset peaks on session change / fresh extraction start
watch(
  () => props.sid,
  () => {
    peaks.value = { unified: 0, vision: 0, audio: 0 }
  }
)
watch(
  () => props.meta?.status,
  (ns, os) => {
    if (ns === 'extracting' && os !== 'extracting') {
      peaks.value = { unified: 0, vision: 0, audio: 0 }
    }
  }
)

// ------------------------------------------------------------------
// 流程轴节点状态 / flow node state
// ------------------------------------------------------------------
const flowCurrent = computed(() => {
  const s = props.meta.stage
  if (s === 'opening' || s === 'roi' || s === 'warmup') return 0
  if (s === 'steady') return 1
  if (s === 'postprocessing') return 2
  if (s === 'done') return 3
  return -1
})

// 节点状态: node-done / node-active / node-pending / node-error / node-skipped
function nodeCls(name) {
  const st = props.meta.status
  const idx = FLOW[name].idx
  // 无音轨时(完成态)音频分支置为跳过
  if (name === 'audio' && st === 'done' && !hasAudio.value) return 'node-skipped'
  if (st === 'done') return 'node-done'
  if (st === 'error')
    return idx < flowCurrent.value
      ? 'node-done'
      : idx === flowCurrent.value
        ? 'node-error'
        : 'node-pending'
  if (!running.value) return 'node-pending'          // 尚未开始: 全部待机
  return idx < flowCurrent.value
    ? 'node-done'
    : idx === flowCurrent.value
      ? 'node-active'
      : 'node-pending'
}

// ------------------------------------------------------------------
// 文案 / labels
// ------------------------------------------------------------------
const currentStep = computed(() => {
  const st = props.meta.status
  if (st === 'done') return '整理完成，聊天记录已生成'
  if (st === 'error') return '提取失败，请查看错误信息'
  if (st === 'idle' || st === 'pending') return '尚未开始'
  const s = props.meta.stage
  if (s === 'opening' || s === 'roi') return '准备：读入视频、定位聊天区域'
  if (s === 'warmup') return '准备：校准识别参数'
  if (s === 'postprocessing') return '整理结果：合并画面与语音，生成聊天记录'
  if (hasAudio.value) return '同步处理：识别画面文字 + 识别语音'
  return '识别画面：逐帧提取聊天文字'
})

const audioBranchDesc = computed(() => {
  if (!hasAudio.value) {
    return done.value ? '无音轨' : '等待音轨…'
  }
  return audioDetail.value
})

const audioDetail = computed(() => {
  const ap = audioProg.value
  if (!ap) return done.value ? '无音轨' : '分析中…'
  const d = ap.detail || ''
  if (ap.phase === 'done') return d || '语音转写完成'
  if (ap.phase === 'error') return d || '音频转写失败'
  if (ap.phase === 'vad') return d || '分析语音段…'
  return d || '语音转文字…'
})

const audioStatus = computed(() => {
  if (audioProg.value?.phase === 'error') return 'exception'
  if (audioPct.value >= 100 && hasAudio.value) return 'success'
  return undefined
})

const elapsedLabel = computed(() => {
  const e = progress.value.elapsed_sec
  if (e == null) return '-'
  const s = Math.max(0, Math.round(e))
  const m = Math.floor(s / 60)
  const ss = s % 60
  return m > 0 ? `${m}m${String(ss).padStart(2, '0')}s` : `${ss}s`
})

const currentDetail = computed(() => props.meta.stage_detail || '')
const progressStatus = computed(() => {
  if (props.meta.status === 'error') return 'exception'
  if (props.meta.status === 'done') return 'success'
  if (props.meta.status === 'extracting') return ''
  return 'warning'
})

const detOpen = ref([])
</script>

<style scoped>
.status-stepper {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 18px 8px 10px;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
}

/* 统一进度区 / unified progress area */
.unified-area {
  padding: 0 14px;
}
.unified-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 6px;
}
.unified-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--el-text-color-regular);
}
.unified-pct {
  font-size: 16px;
  font-weight: 700;
  color: var(--el-color-primary);
  font-family: Consolas, monospace;
}
.unified-step {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 8px;
  font-size: 13px;
  color: var(--el-text-color-primary);
}
.step-icon {
  color: var(--el-color-primary);
  flex-shrink: 0;
}
.step-elapsed {
  margin-left: auto;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  white-space: nowrap;
}

/* 流程轴(横向 1→2→1 分合) / horizontal fork-join flow */
.flow {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 6px 12px 4px;
  flex-wrap: nowrap;
  overflow-x: auto;
}
.flow-col {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.flow-arrow {
  display: grid;
  place-items: center;
  color: var(--el-text-color-placeholder);
  flex-shrink: 0;
}
.flow-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 3px;
  min-width: 96px;
  padding: 9px 14px;
  border-radius: 9px;
  border: 1px solid transparent;
  text-align: center;
  transition: all 0.15s ease;
}
.flow-node.branch {
  min-width: 132px;
  padding: 7px 10px;
  align-items: flex-start;
  text-align: left;
}
.node-dot {
  display: grid;
  place-items: center;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 2px solid var(--el-border-color);
  background: var(--el-bg-color);
  font-size: 11px;
  color: #fff;
}
.node-head {
  display: flex;
  align-items: center;
  gap: 8px;
  text-align: left;
}
.node-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--el-text-color-secondary);
}
.node-desc {
  font-size: 11px;
  color: var(--el-text-color-placeholder);
}
.branch-progress {
  width: 100%;
  margin-top: 3px;
}

/* 状态配色 / status colors */
.flow-node.node-done {
  border-color: var(--el-color-success-light-5);
  background: var(--el-color-success-light-9);
}
.node-done .node-dot {
  border-color: var(--el-color-success);
  background: var(--el-color-success);
}
.node-done .node-title {
  color: var(--el-color-success);
}
.flow-node.node-active {
  border-color: var(--el-color-primary);
  background: var(--el-color-primary-light-9);
}
.node-active .node-dot {
  border-color: var(--el-color-primary);
  background: var(--el-color-primary);
}
.node-active .node-title {
  color: var(--el-color-primary);
}
.flow-node.node-pending {
  border-style: dashed;
  border-color: var(--el-border-color-lighter);
  background: var(--el-fill-color-light);
}
.flow-node.node-error {
  border-color: var(--el-color-danger);
  background: var(--el-color-danger-light-9);
}
.node-error .node-title {
  color: var(--el-color-danger);
}
.flow-node.node-skipped {
  border-color: var(--el-border-color-lighter);
  background: var(--el-fill-color-light);
  opacity: 0.65;
}

/* 拆分详情 / split detail */
.split-collapse {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  overflow: hidden;
  margin: 0 10px;
}
.split-collapse :deep(.el-collapse-item__header) {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  padding: 0 12px;
}
.split-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 5px 2px;
}
.split-label {
  width: 78px;
  font-size: 12px;
  color: var(--el-text-color-regular);
  flex-shrink: 0;
}
.split-sub {
  min-width: 120px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  text-align: right;
  flex-shrink: 0;
}

.run-info {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  padding: 0 14px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}
.info-item.stage-detail {
  color: var(--el-text-color-primary);
}
</style>