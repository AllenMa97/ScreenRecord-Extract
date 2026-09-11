<template>
  <div class="session-list">
    <div class="list-header">
      <span class="list-title">会话列表</span>
      <el-button
        text
        size="small"
        :icon="Refresh"
        :loading="store.loadingSessions"
        @click="refresh"
      />
    </div>

    <el-scrollbar class="list-scroll" v-loading="store.loadingSessions">
      <div
        v-for="s in store.sessions"
        :key="s.sid"
        class="session-item"
        :class="{ active: s.sid === store.currentSid }"
        @click="selectSession(s.sid)"
      >
        <div class="item-main">
          <div class="item-name" :title="s.name">{{ s.name }}</div>
          <div class="item-meta">
            <span class="item-time">{{ s.created_at }}</span>
            <el-tag
              size="small"
              :type="statusMeta(s.status).type"
              :effect="s.sid === store.currentSid ? 'dark' : 'plain'"
            >
              {{ statusMeta(s.status).label }}
            </el-tag>
          </div>
          <el-progress
            v-if="s.progress && s.progress.percent != null"
            class="item-progress"
            :percentage="s.status === 'done' ? 100 : s.progress.percent"
            :stroke-width="6"
            :show-text="false"
            :status="s.status === 'error' ? 'exception' : ''"
          />
        </div>
        <el-button
          class="item-delete"
          text
          type="danger"
          size="small"
          :icon="Delete"
          @click.stop="onDelete(s)"
        />
      </div>

      <el-empty
        v-if="!store.loadingSessions && store.sessions.length === 0"
        :image-size="60"
        description="暂无会话，请先上传视频"
      />
    </el-scrollbar>
  </div>
</template>

<script setup>
import { Delete, Refresh } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { store, refreshSessions, selectSession } from '../store'
import { api } from '../api'
import { statusMeta } from '../utils'

async function refresh() {
  await refreshSessions()
}

async function onDelete(s) {
  const running = s.status === 'extracting'
  // 提取中的会话: 明确告知删除会中止任务并丢失进度
  // running session: clearly warn that deletion aborts the task and loses progress
  const message = running
    ? `确定删除会话「${s.name}」？\n\n此任务正在提取中，删除将立刻中止处理，已进行的进度、识别结果与日志都会丢失，且不可恢复。`
    : `确定删除会话「${s.name}」？将同时删除其视频与提取结果。`
  try {
    await ElMessageBox.confirm(message, '删除会话', {
      type: 'warning',
      confirmButtonText: '仍要删除',
      cancelButtonText: '取消',
    })
  } catch (e) {
    return // 用户取消 / user cancelled
  }
  try {
    await api.deleteSession(s.sid)
    ElMessage.success('会话已删除')
    if (store.currentSid === s.sid) {
      store.currentSid = null
    }
    await refreshSessions()
  } catch (e) {
    ElMessage.error(`删除失败：${e.message}`)
  }
}
</script>

<style scoped>
.session-list {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 4px 8px;
}

.list-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--el-text-color-secondary);
}

.list-scroll {
  flex: 1;
  min-height: 0;
}

.session-item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  margin-bottom: 8px;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.15s ease;
  background: var(--el-bg-color);
}

.session-item:hover {
  border-color: var(--el-color-primary-light-5);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.session-item.active {
  border-color: var(--el-color-primary);
  background: var(--el-color-primary-light-9);
}

.item-main {
  flex: 1;
  min-width: 0;
}

.item-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--el-text-color-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 4px;
}

.item-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}

.item-time {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.item-progress {
  margin-top: 2px;
}

.item-delete {
  opacity: 0;
  transition: opacity 0.15s ease;
}

.session-item:hover .item-delete {
  opacity: 1;
}
</style>
