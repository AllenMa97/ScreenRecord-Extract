<template>
  <div class="video-info">
    <!-- 视频播放器 / video player -->
    <div class="video-wrap">
      <video
        v-if="videoSrc"
        :src="videoSrc"
        class="video-player"
        controls
        preload="metadata"
      ></video>
      <div v-else class="video-placeholder">视频文件加载失败</div>
    </div>

    <!-- 视频元数据: 两列信息卡, 高度对齐左侧播放器 / metadata: 2-col cards,
         vertically centered to match the player height -->
    <div class="video-meta">
      <div v-for="it in items" :key="it.label" class="meta-item">
        <span class="meta-label">{{ it.label }}</span>
        <span class="meta-value" :title="it.value">{{ it.value }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { api } from '../api'
import { fmtBytes, fmtDuration } from '../utils'

const props = defineProps({
  meta: { type: Object, required: true },
})

const videoSrc = computed(() =>
  props.meta && props.meta.sid ? api.videoUrl(props.meta.sid) : null
)

const items = computed(() => {
  const v = props.meta.video || {}
  const m = props.meta
  return [
    { label: '格式', value: v.format || '-' },
    { label: '时长', value: fmtDuration(v.duration_sec) },
    { label: '文件大小', value: v.size_human || fmtBytes(v.size_bytes) },
    { label: '原始帧数', value: v.total_frames ?? '-' },
    { label: '帧率 FPS', value: v.fps ? v.fps.toFixed(2) : '-' },
    {
      label: '分辨率',
      value: v.width && v.height ? `${v.width} × ${v.height}` : '-',
    },
    { label: '编解码器', value: v.codec || '-' },
    { label: '原始文件名', value: m.name || '-' },
    { label: '会话创建时间', value: m.created_at || '-' },
  ]
})
</script>

<style scoped>
.video-info {
  display: flex;
  gap: 16px;
  align-items: stretch;
}

.video-wrap {
  flex: 0 0 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 150px;
  background: #0f1420;
  border-radius: 10px;
  overflow: hidden;
}

.video-player {
  width: 100%;
  max-height: 260px;
  display: block;
}

.video-placeholder {
  color: #8a93a6;
  font-size: 13px;
}

/* 两列信息卡, 垂直居中对齐播放器高度 / 2-col cards centered to player height */
.video-meta {
  flex: 1;
  min-width: 0;
  align-self: stretch;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px 16px;
  align-content: center;
}

.meta-item {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 10px 12px;
  background: var(--el-fill-color-light);
  border-radius: 8px;
  min-width: 0;
}

.meta-label {
  flex-shrink: 0;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.meta-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--el-text-color-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
