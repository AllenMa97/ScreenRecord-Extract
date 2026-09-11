<template>
  <div class="upload-zone">
    <el-upload
      drag
      :show-file-list="false"
      :http-request="doUpload"
      accept=".mp4,.mov,.avi,.mkv,.webm,.m4v"
      :disabled="uploading"
    >
      <el-icon class="el-icon--upload" :size="48"><UploadFilled /></el-icon>
      <div class="el-upload__text">
        将录屏视频拖到此处，或 <em>点击上传</em>
      </div>
      <template #tip>
        <div class="el-upload__tip">
          支持 MP4 / MOV / AVI / MKV / WEBM / M4V，单个最大 1GB
        </div>
      </template>
    </el-upload>

    <div v-if="uploading" class="upload-progress">
      <div class="progress-row">
        <span class="progress-name">{{ uploadName }}</span>
        <span class="progress-pct">{{ uploadPercent }}%</span>
      </div>
      <el-progress
        :percentage="uploadPercent"
        :stroke-width="12"
        :status="uploadPercent >= 100 ? 'success' : ''"
      />
      <div class="progress-hint">上传完成后自动创建会话</div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api } from '../api'
import { refreshSessions, selectSession } from '../store'

const uploading = ref(false)
const uploadPercent = ref(0)
const uploadName = ref('')

async function doUpload({ file, onError }) {
  uploading.value = true
  uploadPercent.value = 0
  uploadName.value = file.name
  try {
    const session = await api.uploadVideo(file, (ratio) => {
      uploadPercent.value = Math.round(ratio * 100)
    })
    ElMessage.success(`上传完成，会话 ${session.sid.slice(-4)} 已创建`)
    await refreshSessions()
    selectSession(session.sid)
  } catch (e) {
    ElMessage.error(`上传失败：${e.message}`)
    if (onError) onError(e)
  } finally {
    uploading.value = false
  }
}
</script>

<style scoped>
.upload-zone {
  padding: 4px 0 12px;
}

.upload-progress {
  margin-top: 14px;
  padding: 12px;
  border: 1px solid var(--el-border-color-light);
  border-radius: 8px;
  background: var(--el-fill-color-light);
}

.progress-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 13px;
}

.progress-name {
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--el-text-color-primary);
}

.progress-pct {
  color: var(--el-color-primary);
  font-weight: 600;
}

.progress-hint {
  margin-top: 6px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
