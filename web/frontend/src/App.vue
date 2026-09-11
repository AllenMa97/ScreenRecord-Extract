<template>
  <div class="app">
    <header class="app-header">
      <div class="header-brand">
        <el-icon :size="26"><Monitor /></el-icon>
        <div class="brand-text">
          <div class="brand-title">手机录屏内容提取 · 演示平台</div>
          <div class="brand-sub">自适应抽帧 → OCR → 去重 → 聊天记录结构化汇总</div>
        </div>
      </div>
      <div class="header-right">
        <el-tag size="small" type="success" effect="plain" :icon="CircleCheck">
          算法流水线就绪
        </el-tag>
      </div>
    </header>

    <div class="app-body">
      <!-- 左侧: 上传 + 会话列表 / left: upload + sessions -->
      <aside class="app-side">
        <UploadZone />
        <div class="side-divider"></div>
        <SessionList />
      </aside>

      <!-- 右侧: 会话独立展示区 / right: session view -->
      <main class="app-main">
        <SessionView v-if="store.currentSid" :key="store.currentSid" :sid="store.currentSid" />
        <div v-else class="main-empty">
          <el-empty description="请上传录屏视频或选择一个会话开始演示">
            <el-button type="primary" plain @click="demoHint">查看使用说明</el-button>
          </el-empty>
        </div>
      </main>
    </div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { CircleCheck, Monitor } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import { store, refreshSessions } from './store'
import UploadZone from './components/UploadZone.vue'
import SessionList from './components/SessionList.vue'
import SessionView from './components/SessionView.vue'

onMounted(() => {
  refreshSessions()
})

function demoHint() {
  ElMessage.info('1. 上传聊天记录录屏视频 → 2. 点击「开始提取」 → 3. 往下滚动查看 帧检视 / 最终结果(聊天记录) / 行为日志')
}
</script>

<style>
* {
  box-sizing: border-box;
}

html,
body,
#app {
  height: 100%;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
    'Microsoft YaHei', Roboto, Helvetica, Arial, sans-serif;
  background: #eef1f6;
  color: var(--el-text-color-primary);
}

.app {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  height: 56px;
  background: linear-gradient(135deg, #1f2b4d 0%, #2d4379 100%);
  color: #fff;
  flex-shrink: 0;
}

.header-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand-title {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.5px;
}

.brand-sub {
  font-size: 12px;
  opacity: 0.75;
  margin-top: 2px;
}

.app-body {
  flex: 1;
  min-height: 0;
  display: flex;
}

.app-side {
  width: 320px;
  flex-shrink: 0;
  padding: 14px;
  background: var(--el-bg-color);
  border-right: 1px solid var(--el-border-color-lighter);
  display: flex;
  flex-direction: column;
}

.side-divider {
  height: 1px;
  background: var(--el-border-color-lighter);
  margin: 12px 0;
  flex-shrink: 0;
}

.app-main {
  flex: 1;
  min-width: 0;
  padding: 16px 20px;
  overflow: auto;
}

.main-empty {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
