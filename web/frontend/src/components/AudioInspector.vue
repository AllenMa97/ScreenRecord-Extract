<template>
  <div class="audio-inspector">
    <!-- 状态卡片 / status card -->
    <div
      v-if="statusCard"
      class="status-card"
      :class="`sc-${statusCard.tone}`"
    >
      <div class="status-card-left">
        <div class="status-title">
          <el-icon :size="15"><Microphone /></el-icon>
          <span>{{ statusCard.title }}</span>
        </div>
        <div class="status-sub" v-if="statusCard.desc">{{ statusCard.desc }}</div>
      </div>
      <div class="status-meta">
        <span
          v-if="audio && audio.duration_s != null"
          class="len-chip"
        >音频长度 {{ fmtClock(audio.duration_s) }}</span>
        <el-tag
          v-if="audio && audio.has_audio === false"
          type="info"
          effect="plain"
          size="small"
        >无音轨</el-tag>
        <el-tag v-else-if="statusCard.tone === 'muted'" type="warning" effect="plain" size="small">静音</el-tag>
        <el-tag v-else-if="statusCard.tone === 'noise'" type="warning" effect="dark" size="small">疑似噪声</el-tag>
        <el-tag v-else type="success" effect="light" size="small">已转写</el-tag>
      </div>
    </div>

    <!-- 语音气泡(视觉侧检测) / voice bubbles (vision-side) -->
    <div v-if="voiceEvents.length" class="block">
      <div class="block-head">
        <span class="block-title">
          语音气泡<span class="weak">（视觉侧+OCR时长标签）</span>
        </span>
        <span class="block-count">
          共 {{ voiceEvents.length }} 个 · {{ matchedCount }} 已配对
        </span>
      </div>
      <div class="voice-list">
        <div v-for="(ev, i) in voiceEvents" :key="i" class="voice-item">
          <div class="voice-item-top">
            <el-tag
              size="small"
              :type="ev.matched ? 'success' : (ev.side === '发' ? 'primary' : 'info')"
              effect="plain"
            >{{ ev.matched ? '已配对' : '未采集' }}</el-tag>
            <el-tag
              v-if="ev.source" size="small" effect="plain"
              :type="ev.source === 'hybrid' ? 'primary' : 'warning'"
            >{{ ev.source === 'hybrid' ? '视觉+OCR' : (ev.source === 'ocr' ? 'OCR信号' : '视觉侧') }}</el-tag>
            <span class="voice-side">[{{ ev.side || '未知' }}]</span>
            <el-tag v-if="ev.audio_duration" size="small" type="info" effect="plain">
              语音 {{ fmtClock(ev.audio_duration) }}
            </el-tag>
            <span class="voice-time" title="该语音气泡在视频中的位置（第几分几秒）">视频 {{ fmtClock(ev.timestamp) }}</span>
            <span class="voice-conf">置信度 {{ (ev.confidence * 100).toFixed(0) }}%</span>
          </div>
          <audio
            v-if="ev.audio_url"
            :src="voiceUrl(ev.audio_url)"
            class="voice-audio"
            controls
            preload="none"
            controlslist="nodownload"
          ></audio>
          <div v-if="ev.text" class="voice-text">{{ ev.text }}</div>
          <div v-else class="voice-text muted">
            (检测到语音气泡，但对应声音未被录屏采集)
          </div>
        </div>
      </div>
    </div>

    <!-- 音频转写段 / audio segments -->
    <div class="block" v-if="hasSegments">
      <div class="block-head">
        <span class="block-title">音频转写<span class="weak">（ASR 时间轴）</span></span>
        <span class="block-count">共 {{ audio.segments.length }} 段</span>
      </div>
      <div class="segment-list">
        <div v-for="(seg, i) in audio.segments" :key="i" class="segment-item">
          <span class="seg-time">视频 {{ fmtClock(seg.start) }} → {{ fmtClock(seg.end) }}</span>
          <span v-if="seg.text" class="seg-text">{{ seg.text }}</span>
          <span v-else class="seg-text muted">{{ seg.events?.length ? `非语音事件：${seg.events.join('、')}` : '(本段无转写文本)' }}</span>
        </div>
      </div>
    </div>

    <!-- 无音频数据 / no audio -->
    <div
      v-if="audioSegments.length === 0 && voiceEvents.length === 0 && audioHasData"
      class="block empty-block"
    >
      <el-empty
        :image-size="60"
        :description="audio?.reason || '无音频转写段，也未检测到语音气泡'"
      />
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { Microphone } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import { api } from '../api'
import { fmtClock } from '../utils'

const props = defineProps({
  sid: { type: String, required: true },
  refreshTick: { type: Number, default: 0 },
})

const result = ref(null)
let failedTick = null

async function load() {
  try {
    result.value = await api.result(props.sid)
    failedTick = null
  } catch (e) {
    failedTick = props.refreshTick // 尚未有结果时, 等下次刷新再试
  }
}

watch(
  () => props.sid,
  () => { result.value = null; load() },
  { immediate: true }
)
watch(
  () => props.refreshTick,
  (t) => { if (t !== failedTick && t > 0) load() }
)

const audio = computed(() => result.value?.audio || null)
const voiceEvents = computed(() => result.value?.voice_events || [])
const audioSegments = computed(() => audio.value?.segments || [])

const hasSegments = computed(() =>
  audioSegments.value.some((s) => (s.text || '').trim())
)
const audioHasData = computed(() => !!audio.value)

const matchedCount = computed(() =>
  voiceEvents.value.filter((e) => e.matched).length
)

// 把后端回填的 audio_url(如 /api/sessions/{sid}/voices/xxx.wav)转成可播放的完整地址
// audio_url is an API path; voiceUrl maps it to a playable URL (browser origin).
function voiceUrl(path) {
  if (!path) return ''
  return path
}

// 音频状态判别(与 ResultPanel 保持一致) / audio status classification
const statusCard = computed(() => {
  const a = audio.value
  if (!a) return null
  // 无音轨: 视频不含可解码音频
  if (a.has_audio === false) {
    return { tone: 'none', title: '无音轨', desc: '视频文件不含可解码的音频流' }
  }
  const rms = a.rms_level ?? 0
  const textSegs = a.text_segments ?? 0
  // 静音: 有音轨但能量≈0
  if (rms < 1e-3) {
    const bubbleNote = voiceEvents.value.length
      ? `，但视觉侧检测到 ${voiceEvents.value.length} 个语音气泡`
      : ''
    return {
      tone: 'muted',
      title: '音频为静音',
      desc: `有音轨但未能解出可听声音（RMS≈0）${bubbleNote}`,
    }
  }
  // 疑似噪声: 有声音但无有效文本段
  if (textSegs === 0) {
    return {
      tone: 'noise',
      title: '检测到音轨但未识别出有效人声',
      desc: '可能为环境噪声或纯音乐、背景音',
    }
  }
  // 已转写
  return {
    tone: 'ok',
    title: `已转写 ${textSegs} 段语音`,
    desc: `全片 RMS=${rms.toFixed(4)} · 气泡配对 ${matchedCount.value}/${voiceEvents.value.length}`,
  }
})
</script>

<style scoped>
.audio-inspector {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 4px 2px;
}

.status-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 16px;
  border-radius: 10px;
  border: 1px solid transparent;
}
.status-card-left { min-width: 0; }
.status-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}
.status-sub {
  margin-top: 3px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  line-height: 1.5;
}
.sc-none { border-color: var(--el-border-color-lighter); background: var(--el-fill-color-light); }
.sc-muted { border-color: var(--el-color-warning); background: var(--el-color-warning-light-9); }
.sc-noise { border-color: var(--el-color-warning); background: var(--el-color-warning-light-9); }
.sc-ok { border-color: var(--el-color-success); background: var(--el-color-success-light-9); }
.status-meta { flex-shrink: 0; display: flex; align-items: center; gap: 8px; }
.len-chip {
  font-size: 12px;
  color: var(--el-text-color-regular);
  background: var(--el-fill-color);
  border-radius: 6px;
  padding: 1px 8px;
  font-family: Consolas, monospace;
}

.block {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
  overflow: hidden;
}
.block-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 14px;
  border-bottom: 1px solid var(--el-border-color-lighter);
  background: var(--el-fill-color-light);
}
.block-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--el-text-color-regular);
}
.block-title .weak { font-weight: 400; color: var(--el-text-color-secondary); }
.block-count { font-size: 12px; color: var(--el-text-color-secondary); }

.voice-list, .segment-list {
  max-height: 320px;
  overflow-y: auto;
}
.voice-item {
  padding: 8px 14px;
  border-bottom: 1px solid var(--el-border-color-lighter);
}
.voice-item:last-child { border-bottom: none; }
.voice-item-top {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.voice-side { font-size: 12px; color: var(--el-text-color-regular); }
.voice-time { font-size: 12px; color: var(--el-text-color-secondary); font-family: Consolas, monospace; }
.voice-conf { font-size: 11px; color: var(--el-text-color-placeholder); margin-left: auto; }
.voice-audio {
  margin-top: 6px;
  width: 100%;
  height: 30px;
}
.voice-text {
  margin-top: 5px;
  font-size: 13px;
  color: var(--el-text-color-primary);
  line-height: 1.5;
  word-break: break-all;
}
.voice-text.muted { color: var(--el-text-color-placeholder); }

.segment-item {
  display: flex;
  align-items: baseline;
  gap: 12px;
  padding: 8px 14px;
  border-bottom: 1px solid var(--el-border-color-lighter);
}
.segment-item:last-child { border-bottom: none; }
.seg-time { font-size: 12px; color: var(--el-text-color-secondary); font-family: Consolas, monospace; flex-shrink: 0; }
.seg-text { font-size: 13px; color: var(--el-text-color-primary); line-height: 1.5; word-break: break-all; }
.seg-text.muted { color: var(--el-text-color-placeholder); }

.empty-block { padding: 4px; }
</style>