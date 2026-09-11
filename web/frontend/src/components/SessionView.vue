<template>
  <div v-if="meta" class="session-view">
    <!-- 顶部操作栏 / top action bar -->
    <div class="view-header">
      <div class="header-left">
        <div class="session-name" :title="meta.name">{{ meta.name }}</div>
        <el-tag
          size="small"
          :type="statusMeta(meta.status).type"
          effect="dark"
        >
          {{ statusMeta(meta.status).label }}
        </el-tag>
        <span class="session-sid">{{ sid }}</span>
      </div>
      <div class="header-actions">
        <el-tooltip
          content="运行算法流水线，逐帧抽帧→OCR→去重→汇总（真实OCR服务）"
          placement="top"
        >
          <el-button
            type="primary"
            :icon="VideoPlay"
            :loading="isRunning"
            @click="startExtract"
          >
            {{ isRunning ? '提取中…' : '开始提取' }}
          </el-button>
        </el-tooltip>
      </div>
    </div>

    <!-- 视频信息(播放器 + 元数据) / video info -->
    <VideoInfo :meta="meta" />

    <!-- 算法阶段 Stepper + 进度 / pipeline stepper + progress -->
    <StatusStepper :sid="sid" :meta="meta" />

    <!-- 独立展示区: 帧检视 → 最终结果 → 行为日志(整页滚动) /
         stacked sections: frames → result → logs (page scrolls) -->
    <div class="view-body">
      <section class="view-section">
        <div class="section-head">
          <div class="section-title">
            <span class="section-index">①</span> 帧检视
          </div>
          <span class="section-sub">算法抽帧的每一帧 · 判定结果与 OCR 识别</span>
        </div>
        <FrameInspector
          ref="frameInspector"
          :sid="sid"
          :refresh-tick="refreshTick"
        />
      </section>

      <section class="view-section">
        <div class="section-head">
          <div class="section-title">
            <span class="section-index">②</span> 音频检视
          </div>
          <span class="section-sub">音轨状态 · 语音气泡(视觉侧) · ASR 转写</span>
        </div>
        <AudioInspector
          :sid="sid"
          :refresh-tick="refreshTick"
        />
      </section>

      <section class="view-section">
        <div class="section-head">
          <div class="section-title">
            <span class="section-index">③</span> 最终结果 · 聊天记录
          </div>
          <span class="section-sub">跨帧去重后的聊天内容结构化汇总（提取完成即展示）</span>
        </div>
        <ResultPanel :sid="sid" :status="meta?.status" />
      </section>

      <section class="view-section">
        <div class="section-head">
          <div class="section-title">
            <span class="section-index">④</span> 行为日志
          </div>
          <span class="section-sub">算法各阶段决策的实时记录</span>
        </div>
        <BehaviorLog :sid="sid" :live-logs="liveLogs" :status="meta.status" />
      </section>
    </div>
  </div>

  <div v-else v-loading="true" class="view-loading"></div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { VideoPlay } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import { api, openSessionSocket } from '../api'
import { refreshSessions, store, syncSession } from '../store'
import { statusMeta } from '../utils'

import VideoInfo from './VideoInfo.vue'
import StatusStepper from './StatusStepper.vue'
import FrameInspector from './FrameInspector.vue'
import AudioInspector from './AudioInspector.vue'
import ResultPanel from './ResultPanel.vue'
import BehaviorLog from './BehaviorLog.vue'

const props = defineProps({
  sid: { type: String, required: true },
})

const meta = ref(null)
const liveLogs = ref([])
const refreshTick = ref(0)

const frameInspector = ref(null)

const isRunning = computed(() => meta.value?.status === 'extracting')

// ------------------------------------------------------------------
// 加载与 WebSocket / load & websocket
// ------------------------------------------------------------------
let ws = null
let wsTimer = null
let frameRefreshTimer = null
let pollTimer = null
let disposed = false
let lastNotified = null // 已弹过完成/失败提示的状态 / status already notified

async function loadMeta() {
  try {
    meta.value = await api.getSession(props.sid)
    syncSession(meta.value) // 同步左侧会话列表状态 / keep sidebar status in sync
  } catch (e) {
    ElMessage.error(`加载会话失败：${e.message}`)
  }
}

function connectWs() {
  ws = openSessionSocket(props.sid, {
    snapshot: onSnapshot,
    status: onStatus,
    event: onEvent,
    log: onLog,
    onclose: () => {
      // 页面存活期间一律重连, 避免连接丢失后实时状态推送中断
      // Always reconnect while the view is alive; otherwise realtime
      // pushes are lost and the UI stays stale until manual refresh.
      if (disposed) return
      wsTimer = setTimeout(connectWs, 3000)
    },
  })
}

// 提取结束的统一处理(WS事件/心跳/轮询都可能触发, 去重提示) /
// unified completion handling (WS event / heartbeat / poll; dedup toast)
function handleCompletion(status, errMsg) {
  const first = lastNotified !== status
  lastNotified = status
  loadMeta()
  // 完成/失败即刷新 ①帧检视 ②最终结果, 无需切换会话 /
  // refresh frames & final result immediately, no session switching needed
  refreshTick.value++
  if (!first) return
  refreshSessions()
  if (status === 'done') ElMessage.success('提取完成')
  else ElMessage.error(`提取失败：${errMsg || ''}`)
}

function onSnapshot(msg) {
  applyStatus(msg)
  meta.value = { ...(meta.value || {}), ...applyMetaFields(msg) }
}

function applyStatus(msg) {
  const m = meta.value
  if (!m) return
  if (msg.status) {
    m.status = msg.status
    // 重新进入非终态时, 允许下次完成再次提示 / re-notify on a new run
    if (msg.status !== 'done' && msg.status !== 'error') lastNotified = null
  }
  if (msg.stage) m.stage = msg.stage
  if (msg.stage_detail != null) m.stage_detail = msg.stage_detail
  if (msg.progress) m.progress = { ...m.progress, ...msg.progress }
  if (msg.audio_progress) m.audio_progress = msg.audio_progress
  if (msg.error) m.error = msg.error
  syncSession(m) // 提取过程中实时同步左侧列表 / live-sync sidebar during extraction
}

function applyMetaFields(msg) {
  const out = {}
  if (msg.status) out.status = msg.status
  if (msg.stage) out.stage = msg.stage
  if (msg.stage_detail != null) out.stage_detail = msg.stage_detail
  if (msg.progress) out.progress = msg.progress
  if (msg.audio_progress) out.audio_progress = msg.audio_progress
  return out
}

function onStatus(msg) {
  applyStatus(msg)
  if (msg.status === 'done' || msg.status === 'error') {
    handleCompletion(msg.status, msg.error || msg.stage_detail || '')
  }
}

function onEvent(evt) {
  const m = meta.value
  if (!m) return
  const t = evt.type
  if (t === 'stage') {
    m.stage = evt.stage
    m.stage_detail = evt.detail
  } else if (t === 'frame') {
    // 直接用后端单调进度(重置关键帧会回头重采, 前端不再自算避免回退)
    // use the backend's monotonic percent (keyframe-reset re-walks frames)
    m.progress = {
      frame_id: evt.frame_id,
      max_frame_id: evt.max_frame_id,
      total_frames: evt.total_frames || 0,
      percent: evt.percent ?? 0,
      elapsed_sec: m.progress?.elapsed_sec,
    }
    scheduleFrameRefresh()
  } else if (t === 'ocr_frame') {
    scheduleFrameRefresh()
  } else if (t === 'audio_progress') {
    // 实时音频进度(视觉侧由 frame 事件驱动, 音频侧由本事件驱动)
    // realtime audio progress (vision driven by frame events)
    m.audio_progress = {
      phase: evt.phase,
      done: evt.done || 0,
      total: evt.total || 0,
      detail: evt.detail || '',
    }
    if (evt.phase === 'done' || evt.phase === 'error') {
      scheduleFrameRefresh()
    }
  } else if (t === 'status') {
    applyStatus(evt)
    if (evt.status === 'done' || evt.status === 'error') {
      handleCompletion(evt.status, evt.detail || evt.error || '')
    }
  }
}

function onLog(entry) {
  liveLogs.value.push(entry)
  if (liveLogs.value.length > 3000) {
    liveLogs.value.splice(0, liveLogs.value.length - 3000)
  }
}

// 提取过程中节流刷新帧列表(静默, 不闪加载遮罩; 完成时由 handleCompletion 兜底)
// throttled silent frame-list refresh while extracting; handleCompletion covers done
function scheduleFrameRefresh() {
  if (frameRefreshTimer) return
  frameRefreshTimer = setTimeout(() => {
    frameRefreshTimer = null
    refreshTick.value++
  }, 1500)
}

// ------------------------------------------------------------------
// 兜底轮询: WebSocket 失效时仍能同步最终状态(以服务端为准) /
// fallback polling: final status stays in sync even if WS fails
// ------------------------------------------------------------------
async function pollStatus() {
  if (disposed) return
  let fresh
  try {
    fresh = await api.getSession(props.sid)
  } catch (e) {
    return // 静默, 等下一轮 / silent, retry next tick
  }
  if (!fresh) return
  const prev = meta.value?.status
  if (meta.value) {
    meta.value.status = fresh.status
    meta.value.stage = fresh.stage
    meta.value.stage_detail = fresh.stage_detail
    meta.value.progress = fresh.progress
    meta.value.audio_progress = fresh.audio_progress
    meta.value.error = fresh.error
  } else {
    meta.value = fresh
  }
  syncSession(meta.value)
  if (
    prev &&
    prev !== 'done' &&
    prev !== 'error' &&
    (fresh.status === 'done' || fresh.status === 'error')
  ) {
    handleCompletion(fresh.status, fresh.error || fresh.stage_detail || '')
  }
}

function startPolling() {
  stopPolling()
  pollTimer = setInterval(pollStatus, 5000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

// ------------------------------------------------------------------
// 操作 / actions
// ------------------------------------------------------------------
async function startExtract() {
  try {
    await api.startExtract(props.sid)
    ElMessage.info('已启动提取（真实OCR服务），可在「行为日志」观察实时决策')
    refreshTick.value++ // 清掉旧结果/旧帧, 立即进入新提取状态 / clear stale data
    await loadMeta()
  } catch (e) {
    ElMessage.error(`启动失败：${e.message}`)
  }
}

// ------------------------------------------------------------------
// 生命周期 / lifecycle
// ------------------------------------------------------------------
onMounted(() => {
  loadMeta()
  connectWs()
  startPolling()
})

watch(
  () => props.sid,
  () => {
    liveLogs.value = []
    loadMeta()
    if (ws) ws.close()
    connectWs()
  }
)

onBeforeUnmount(() => {
  disposed = true
  if (ws) ws.close()
  if (wsTimer) clearTimeout(wsTimer)
  if (frameRefreshTimer) clearTimeout(frameRefreshTimer)
  stopPolling()
})
</script>

<style scoped>
.session-view {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.view-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.session-name {
  font-size: 18px;
  font-weight: 700;
  color: var(--el-text-color-primary);
  max-width: 320px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-sid {
  font-size: 12px;
  color: var(--el-text-color-placeholder);
  font-family: Consolas, monospace;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

/* 上下堆叠分区(整页滚动) / stacked sections (page scroll) */
.view-body {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.view-section {
  padding: 16px 18px;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 12px;
}

.section-head {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.section-title {
  font-size: 15px;
  font-weight: 700;
  color: var(--el-text-color-primary);
}

.section-index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--el-color-primary);
  color: #fff;
  font-size: 12px;
  margin-right: 4px;
}

.section-sub {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.view-loading {
  height: 100%;
}
</style>
