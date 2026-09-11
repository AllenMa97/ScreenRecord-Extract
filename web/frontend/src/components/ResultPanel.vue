<template>
  <div class="result-panel">
    <div v-if="errorMsg" class="empty-box">{{ errorMsg }}</div>

    <template v-else-if="result">
      <!-- 标题 + 统计胶囊 / title + stat pills -->
      <div class="result-header">
        <div class="header-title">
          <el-icon :size="18"><ChatDotRound /></el-icon>
          聊天记录
        </div>
        <div class="header-stats">
          <el-tag size="small" type="primary" effect="plain">
            文本 {{ statLines }} 行
          </el-tag>
          <el-tag size="small" type="success" effect="plain">
            段落 {{ result.sections_count ?? result.sections?.length ?? 0 }}
          </el-tag>
          <el-tag v-if="voiceStats.voiceEvents>0" size="small" type="primary" effect="plain">
            语音气泡 {{ voiceStats.voiceEvents }}
          </el-tag>
          <el-tag v-if="voiceStats.voiceMatched>0" size="small" type="success" effect="plain">
            已配对 {{ voiceStats.voiceMatched }}
          </el-tag>
          <el-tag size="small" type="warning" effect="plain">
            缺口 {{ result.statistics?.total_gaps ?? result.gaps?.length ?? 0 }}
          </el-tag>
          <el-tag v-if="result.llm_enhanced" size="small" type="info" effect="plain">
            LLM 增强
          </el-tag>
        </div>
      </div>

      <!-- 音频状态卡片（区分无音轨/静音/噪声/已转写，源标注视觉侧 or 音频侧） -->
      <div v-if="audioStatus" class="audio-status-card" :class="`status-${audioStatus.cls}`">
        <el-icon :size="15"><Microphone /></el-icon>
        <span class="audio-status-text">{{ audioStatus.text }}</span>
      </div>

      <!-- 缺口折叠面板 / collapsible gap warnings -->
      <el-collapse v-if="(result.gaps || []).length" v-model="gapOpen" class="gap-collapse">
        <el-collapse-item
          :title="`检测到 ${(result.gaps || []).length} 个内容缺口（页面跳转/快速滚动处，可点击展开查看）`"
          name="gaps"
        >
          <div class="gap-list">
            <div v-for="(g, i) in result.gaps" :key="i" class="gap-item">
              <el-tag size="small" type="warning" effect="plain">
                时间 {{ fmtClock(g.timestamp) }}
              </el-tag>
              <span class="gap-note">{{ g.note }}</span>
            </div>
          </div>
        </el-collapse-item>
      </el-collapse>

      <!-- 聊天记录气泡流 / chat-style message flow -->
      <div class="chat-panel">
        <div class="chat-flow">
          <template v-for="(sec, i) in parsedSections" :key="i">
            <div class="section-gap" v-if="sec.meta">
              <span class="gap-dot"></span>
              <span class="gap-text">{{ fmtClock(sec.meta.timestamp) }} 起</span>
              <span class="gap-dot"></span>
            </div>

            <div
              v-for="(msg, j) in sec.messages"
              :key="`${i}-${j}`"
              class="chat-row"
              :class="msg.side === '发' ? 'row-out' : 'row-in'"
            >
              <div
                class="bubble"
                :class="[
                  msg.side === '发' ? 'bubble-out' : 'bubble-in',
                  msg.isVoice ? 'bubble-voice' : ''
                ]"
              >
                <div class="bubble-time">
                  <span v-if="msg.isVoice" class="voice-marker">🎤 </span>
                  {{ msg.timeLabel }}
                </div>
                <div class="bubble-text">{{ msg.text }}</div>
              </div>
            </div>
          </template>

          <el-empty
            v-if="parsedSections.length === 0"
            description="未提取到聊天内容（视频中未识别到文本，或未使用真实OCR）"
          />
        </div>
      </div>

      <!-- 语音消息气泡完整列表（含未配对）/ full voice bubble list (including unmatched) -->
      <el-collapse v-if="voiceEvents.length > 0" v-model="collapseOpen" class="voice-collapse">
        <el-collapse-item
          :title="`语音气泡（视觉侧检测，共${voiceEvents.length}个，${voiceMatched}个已配对）`"
          name="voice-bubbles"
        >
          <div class="voice-list">
            <div v-for="(ev, i) in voiceEvents" :key="i" class="voice-item">
              <div class="voice-item-header">
                <el-tag size="small" :type="ev.matched ? 'success' : 'warning'" effect="plain">
                  {{ ev.matched ? '已配对' : '未采集' }}
                </el-tag>
                <el-tag size="small" type="info" effect="plain">
                  {{ fmtClock(ev.timestamp) }}
                </el-tag>
                <el-tag size="small" effect="plain">
                  [{{ ev.side }}]
                </el-tag>
              </div>
              <div v-if="ev.text" class="voice-item-text">{{ ev.text }}</div>
              <div v-else class="voice-item-text muted">(视觉侧检测到语音气泡，但对应声音未被录屏采集)</div>
            </div>
          </div>
        </el-collapse-item>
      </el-collapse>

      <!-- 音频转写完整列表（独立于气泡，确保所有音频都可见）/ full audio transcription list -->
      <el-collapse v-if="audioSegments.length > 0 || (result.audio && result.audio.reason)" v-model="collapseOpenAudio" class="audio-collapse">
        <el-collapse-item
          title="音频转写（全部语音段，未经气泡配对）"
          name="audio-transcript"
        >
          <div class="audio-list">
            <div v-if="audioSegments.length === 0" class="audio-empty">
              {{ result.audio?.reason || '有音轨但未采集到有效人声' }}
            </div>
            <div v-for="(seg, i) in audioSegments" :key="i" class="audio-item">
              <span class="audio-time">{{ fmtTime(seg.start) }}–{{ fmtTime(seg.end) }}</span>
              <span class="audio-text">{{ seg.text }}</span>
            </div>
          </div>
        </el-collapse-item>
      </el-collapse>
    </template>

    <div v-else v-loading="loading" class="empty-box">正在加载提取结果…</div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { ChatDotRound, Microphone } from '@element-plus/icons-vue'
import { api } from '../api'
import { fmtClock } from '../utils'

const props = defineProps({
  sid: { type: String, required: true },
  // 状态跳变驱动: 提取中清旧结果, 完成/失败自动加载(避免提取期间反复刷新闪烁)
  // status transitions: clear on extracting, auto-load on done/error
  status: { type: String, default: '' },
})

const result = ref(null)
const loading = ref(false)
const errorMsg = ref('')
const gapOpen = ref([])
const collapseOpen = ref([])
const collapseOpenAudio = ref([])

watch(
  () => props.sid,
  () => load()
)

watch(
  () => props.status,
  (s) => {
    if (s === 'extracting') {
      result.value = null
      errorMsg.value = ''
    } else if (s === 'done' || s === 'error') {
      load()
    }
  }
)

async function load() {
  result.value = null
  errorMsg.value = ''
  loading.value = true
  try {
    result.value = await api.result(props.sid)
  } catch (e) {
    errorMsg.value = '暂无提取结果，请先在左侧选择会话并执行「提取」。'
  } finally {
    loading.value = false
  }
}

// 解析 section.content 多行为结构化消息 / parse multi-line content into messages
function parseLine(line) {
  // 语音消息行: "- +<Ns> [语音] [发/收]：<文本>" → 麦克风样式气泡
  const voice = line.match(/^-\s*\+(\d+)s\s*\[语音\]\s*\[(发|收)\]\s*[:：]\s*(.*)$/)
  if (voice) {
    return {
      timeLabel: `+${voice[1]}s`,
      side: voice[2],
      text: voice[3].trim().replace(/\\n/g, '\n'),
      isVoice: true,
    }
  }
  const m = line.match(/^-\s*(.+?)\s*\[(发|收)\]\s*[:：]\s*(.*)$/)
  if (m) {
    // 后端同一气泡(同一条消息)内的换行以字面 \n 连接, 这里还原为真实换行,
    // 配合 white-space: pre-wrap 在单个气泡内显示多行。
    // Backend joins in-bubble lines with a literal \n; restore it to a real
    // newline so one bubble renders multiple rows (pre-wrap).
    return { timeLabel: m[1].trim(), side: m[2], text: m[3].trim().replace(/\\n/g, '\n') }
  }
  // 无法按气泡解析的行: 作为纯文本段落展示 / fallback plain text
  const t = line.replace(/^-\s*/, '').trim()
  return t ? { timeLabel: '', side: null, text: t } : null
}

const parsedSections = computed(() => {
  const sections = result.value?.sections || []
  const out = []
  for (const sec of sections) {
    const messages = (sec.content || '')
      .split('\n')
      .map((l) => parseLine(l))
      .filter(Boolean)
    if (messages.length) out.push({ messages })
    // 段落起点若与下一段不连续, 插入时间分隔标记
    if (sec.timestamp != null) {
      const last = out[out.length - 1]
      if (last) last.meta = { timestamp: sec.timestamp }
    }
  }
  return out
})

const statLines = computed(() => {
  let n = 0
  for (const s of parsedSections.value) n += s.messages.length
  return n || result.value?.statistics?.total_tracked_lines || 0
})

// 语音气泡完整列表(来自 result.voice_events) / full voice bubble list
const voiceEvents = computed(() => (result.value?.voice_events || []))
const voiceMatched = computed(() =>
  voiceEvents.value.filter((e) => e.matched).length
)
const voiceStats = computed(() => ({
  voiceEvents: voiceEvents.value.length,
  voiceMatched: voiceMatched.value,
}))

// 音频转写段(来自 result.audio.segments) / all audio transcription segments
const audioSegments = computed(() => {
  const segs = (result.value?.audio?.segments) || []
  return segs.filter((s) => (s.text || '').trim())
})

// 音频源 / audio source from backend
const audio = computed(() => result.value?.audio || null)

// 静音阈值: 线性振幅均方根, 低于此值判定为"有音轨但静音"(1414 全 0 时 rms≈0)
// silence threshold: linear RMS below this ⇒ "track present but silent"
const SILENCE_EPS = 1e-3

// 音频状态卡片: 区分 无音轨 / 静音 / 疑似噪声 / 已转写 四态, 源标注视觉侧 or 音频侧
// audio status card: four states, with the static side (visual/audio) attribution
const audioStatus = computed(() => {
  const a = audio.value
  if (!a) return null
  if (!a.has_audio) {
    return { cls: 'gray', text: '无音轨：视频不含可解码音频，仅展示视觉侧内容' }
  }
  const rms = a.rms_level ?? 0
  const hasText = (a.text_segments || 0) > 0
  const vb = voiceEvents.value.length
  if (rms < SILENCE_EPS) {
    return {
      cls: 'yellow',
      text: vb
        ? `视觉侧检测到 ${vb} 个语音气泡，但音频为静音，未采集到对应语音`
        : '有音轨但内容为静音，未采集到任何对话语音',
    }
  }
  if (!hasText) {
    return {
      cls: 'orange',
      text: '有音轨且存在声音，但未识别出有效人声（可能为噪声或纯音乐）',
    }
  }
  return {
    cls: 'green',
    text: `音频转写成功：识别出 ${a.text_segments} 段语音（气泡配对 ${voiceMatched.value}/${vb}）`,
  }
})

// 秒 → 时间串 (mm:ss) / seconds → (mm:ss)
function fmtTime(seconds) {
  if (seconds == null) return '00:00'
  const s = Math.max(0, Math.round(seconds))
  const mm = String(Math.floor(s / 60)).padStart(2, '0')
  const ss = String(s % 60).padStart(2, '0')
  return `${mm}:${ss}`
}

load()
</script>

<style scoped>
.result-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
}

.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  padding: 12px 18px;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
}

.header-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 700;
  color: var(--el-text-color-primary);
}

.header-stats {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

/* 音频状态卡片 / audio status card */
.audio-status-card {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 14px;
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.45;
}
.audio-status-card :deep(.el-icon) {
  flex-shrink: 0;
}
.status-gray {
  background: var(--el-fill-color-light);
  color: var(--el-text-color-secondary);
  border: 1px solid var(--el-border-color-lighter);
}
.status-yellow {
  background: #fdf6ec;
  color: var(--el-color-warning);
  border: 1px solid var(--el-color-warning-light-5);
}
.status-orange {
  background: #fef0e6;
  color: #c4612a;
  border: 1px solid var(--el-color-warning-light-5);
}
.status-green {
  background: #f0f9eb;
  color: var(--el-color-success);
  border: 1px solid var(--el-color-success-light-5);
}

.gap-collapse {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
  overflow: hidden;
}

.gap-collapse :deep(.el-collapse-item__header) {
  font-size: 13px;
  color: var(--el-color-warning);
  padding: 0 14px;
}

.gap-collapse :deep(.el-collapse-item__wrap) {
  max-height: 160px;
  overflow-y: auto;
}

.gap-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 3px 0;
}

.gap-note {
  font-size: 12px;
  color: var(--el-text-color-regular);
}

.chat-panel {
  max-height: 460px;
  overflow-y: auto;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
  background: #f2f4f8;
}

.chat-flow {
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
}

.section-gap {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 18px 0 10px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}

.gap-dot {
  flex: 1;
  height: 1px;
  background: var(--el-border-color);
}

.chat-row {
  display: flex;
  margin-bottom: 10px;
}

.row-out {
  justify-content: flex-end;
}

.row-in {
  justify-content: flex-start;
}

.bubble {
  max-width: 72%;
  padding: 9px 13px;
  border-radius: 10px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
}

.bubble-out {
  background: #95ec69;
  border-top-right-radius: 2px;
}

.bubble-in {
  background: #ffffff;
  border-top-left-radius: 2px;
}

.bubble-time {
  font-size: 11px;
  color: rgba(0, 0, 0, 0.4);
  margin-bottom: 3px;
}

.bubble-out .bubble-time {
  text-align: right;
}

.bubble-text {
  font-size: 14px;
  color: #1f2329;
  line-height: 1.55;
  word-break: break-all;
  white-space: pre-wrap;
}

.empty-box {
  padding: 60px 20px;
  text-align: center;
  color: var(--el-text-color-secondary);
  background: var(--el-bg-color);
  border: 1px dashed var(--el-border-color);
  border-radius: 10px;
}

/* 语音消息气泡轻微区分(麦克风标记 + 淡色描边) /
   subtle voice-bubble distinction (mic marker + tinted border) */
.bubble-voice {
  box-shadow: 0 0 0 1px var(--el-color-primary-light-5);
}
.voice-marker {
  font-size: 11px;
}

/* 语音气泡/音频转写展开列表 */
.voice-collapse,
.audio-collapse {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
  overflow: hidden;
}
.voice-collapse :deep(.el-collapse-item__header),
.audio-collapse :deep(.el-collapse-item__header) {
  font-size: 13px;
  padding: 0 14px;
}
.voice-collapse :deep(.el-collapse-item__wrap),
.audio-collapse :deep(.el-collapse-item__wrap) {
  max-height: 320px;
  overflow-y: auto;
}

.voice-list,
.audio-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 4px 2px;
}
.voice-item,
.audio-item {
  padding: 8px 10px;
  background: var(--el-fill-color-light);
  border-radius: 8px;
}
.voice-item-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
  flex-wrap: wrap;
}
.voice-item-text {
  font-size: 13px;
  color: var(--el-text-color-primary);
  line-height: 1.5;
  word-break: break-word;
}
.voice-item-text.muted {
  color: var(--el-text-color-secondary);
}
.audio-item {
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.audio-time {
  font-size: 12px;
  color: var(--el-color-primary);
  font-family: Consolas, monospace;
  white-space: nowrap;
}
.audio-text {
  font-size: 13px;
  color: var(--el-text-color-primary);
  line-height: 1.5;
  word-break: break-word;
}
.audio-empty {
  font-size: 13px;
  color: var(--el-text-color-secondary);
  padding: 6px 2px;
}
</style>
