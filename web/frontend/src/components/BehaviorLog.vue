<template>
  <div class="behavior-log">
    <!-- 工具栏 / toolbar -->
    <div class="log-toolbar">
      <span class="log-title">
        行为日志
        <span class="log-sub">算法各阶段决策记录 · 实时滚动</span>
      </span>
      <div class="log-actions">
        <el-tooltip
          content="开启：新日志自动滚动到底部；向上滚动查看历史时会自动暂停"
          placement="top"
        >
          <el-switch
            v-model="autoScroll"
            size="small"
            active-text="自动滚动"
            inactive-text="已暂停"
          />
        </el-tooltip>
        <el-tooltip content="回到底部并恢复自动滚动" placement="top">
          <el-button
            size="small"
            type="primary"
            plain
            :icon="Bottom"
            circle
            @click="scrollBottom"
          />
        </el-tooltip>
        <el-tooltip content="重新加载日志" placement="top">
          <el-button size="small" :icon="Refresh" circle @click="reload" />
        </el-tooltip>
      </div>
    </div>

    <div ref="logBox" class="log-box" @scroll="onScroll">
      <div v-loading="loading" class="log-inner">
        <div
          v-for="(e, i) in logs"
          :key="i"
          class="log-entry"
        >
          <span class="log-time">{{ fmtElapsed(e.elapsed_sec) }}</span>
          <el-tag
            size="small"
            :type="moduleType(e.module)"
            effect="plain"
            class="log-module"
          >
            {{ e.module }}
          </el-tag>
          <span class="log-event">{{ e.event }}</span>
          <span class="log-detail">{{ e.detail }}</span>
          <span class="log-fid" v-if="e.frame_id != null">帧#{{ e.frame_id }}</span>
        </div>
        <div v-if="!loading && logs.length === 0" class="log-empty">
          暂无日志（提取过程中会实时生成）
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onBeforeUnmount } from 'vue'
import { Bottom, Refresh } from '@element-plus/icons-vue'
import { api } from '../api'

const props = defineProps({
  sid: { type: String, required: true },
  liveLogs: { type: Array, default: () => [] },
  status: { type: String, default: '' }, // 会话状态: done/error 时自动重载历史
})

const logs = ref([])       // 历史 + 实时合并(去重) / merged history + live (deduped)
const loading = ref(false)
const autoScroll = ref(true)
const logBox = ref(null)

const MODULE_COLORS = {
  frame_sampler: 'primary',
  roi_manager: 'success',
  blur_detector: 'warning',
  visual_dedup: 'info',
  ocr_client: 'success',
  text_tracker: 'warning',
  semantic_dedup: 'danger',
  state_machine: 'primary',
  content_assembler: 'success',
  calibration: 'info',
  pipeline: 'primary',
}

function moduleType(m) {
  return MODULE_COLORS[m] || 'info'
}

function fmtElapsed(sec) {
  if (sec == null) return '-'
  return `${sec.toFixed(1)}s`
}

// 日志条目唯一键(去重用) / dedupe key for a log entry
const keyOf = (e) =>
  `${e.elapsed_sec ?? ''}|${e.module ?? ''}|${e.event ?? ''}|${e.detail ?? ''}|${e.frame_id ?? ''}`

async function loadHistory() {
  loading.value = true
  try {
    const data = await api.logs(props.sid, 0, 800)
    const hist = data.entries || []
    // 历史(文件)为权威来源, 补充 liveLogs 中尚未落盘的条目, 整体去重
    // File history is authoritative; append live entries not yet flushed, dedupe.
    const seen = new Set()
    const merged = []
    for (const e of hist) {
      const k = keyOf(e)
      if (seen.has(k)) continue
      seen.add(k)
      merged.push(e)
    }
    for (const e of props.liveLogs) {
      const k = keyOf(e)
      if (seen.has(k)) continue
      seen.add(k)
      merged.push(e)
    }
    logs.value = merged
  } catch (e) {
    console.error('加载历史日志失败', e)
  } finally {
    loading.value = false
    scrollToBottom(true)
  }
}

function scrollToBottom(force = false) {
  if (!force && !autoScroll.value) return
  nextTick(() => {
    if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight
  })
}

function onScroll() {
  if (!logBox.value) return
  const el = logBox.value
  // 距离底部 <40px 视为"跟随底部"; 否则自动暂停, 方便查看历史
  autoScroll.value = el.scrollHeight - el.scrollTop - el.clientHeight < 40
}

function scrollBottom() {
  autoScroll.value = true
  scrollToBottom(true)
}

function reload() {
  loadHistory()
}

// 实时日志增量追加(去重) / incremental live append (deduped)
watch(
  () => props.liveLogs.length,
  () => {
    const seen = new Set(logs.value.map(keyOf))
    let added = 0
    for (const e of props.liveLogs) {
      const k = keyOf(e)
      if (seen.has(k)) continue
      seen.add(k)
      logs.value.push(e)
      added++
    }
    if (logs.value.length > 4000) {
      logs.value.splice(0, logs.value.length - 4000)
    }
    if (added) scrollToBottom()
  }
)

// 切换会话 / switch session
watch(
  () => props.sid,
  () => loadHistory()
)

// 提取结束(成功/失败)后文件已完整写入, 自动重载补齐(无需手动刷新)
// Reload automatically once extraction ends (file fully written).
let prevStatus = props.status
watch(
  () => props.status,
  (cur) => {
    if ((cur === 'done' || cur === 'error') && cur !== prevStatus) {
      loadHistory()
    }
    prevStatus = cur
  }
)

onBeforeUnmount(() => {
  prevStatus = ''
})

loadHistory()
</script>

<style scoped>
.behavior-log {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.log-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
  flex-wrap: wrap;
  gap: 8px;
}

.log-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.log-sub {
  font-size: 12px;
  font-weight: 400;
  color: var(--el-text-color-secondary);
  margin-left: 8px;
}

.log-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.log-box {
  height: 420px;
  overflow-y: auto;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
  background: #10151f;
  padding: 10px 0;
}

.log-inner {
  min-height: 100%;
}

.log-entry {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 5px 16px;
  font-size: 12px;
  font-family: 'JetBrains Mono', Consolas, 'Courier New', monospace;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  color: #c8d0de;
}

.log-time {
  color: #6b7686;
  flex: 0 0 52px;
  font-variant-numeric: tabular-nums;
}

.log-module {
  flex: 0 0 auto;
  min-width: 74px;
  text-align: center;
  font-size: 11px;
}

.log-event {
  flex: 0 0 84px;
  color: #7fd6ff;
  font-weight: 600;
}

.log-detail {
  flex: 1;
  color: #c8d0de;
  word-break: break-all;
}

.log-fid {
  flex: 0 0 auto;
  color: #8b95a5;
}

.log-empty {
  padding: 40px;
  text-align: center;
  color: #5a6474;
}
</style>
