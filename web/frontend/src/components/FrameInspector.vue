<template>
  <div class="frame-inspector">
    <!-- 工具栏: 判定筛选 + 分页 + 刷新 / toolbar: verdict filter + pagination + refresh -->
    <div class="toolbar">
      <div class="chips">
        <el-radio-group v-model="group" size="small" @change="reload(1)">
          <el-tooltip
            v-for="(g, key) in GROUPS"
            :key="key"
            :content="g.tip || g.label"
            placement="top"
            :show-after="150"
          >
            <el-radio-button :value="key">
              {{ g.label }}
            </el-radio-button>
          </el-tooltip>
        </el-radio-group>
      </div>
      <div class="toolbar-right">
        <span class="count-label">共 {{ total }} 帧</span>
        <el-button
          size="small"
          :icon="Refresh"
          circle
          :loading="loading"
          @click="reload(page)"
        />
      </div>
    </div>

    <!-- 帧网格 / frame grid -->
    <div v-loading="loading" class="grid-wrap">
      <div class="frame-grid">
        <div
          v-for="f in frames"
          :key="f.frame_id"
          class="frame-card"
          @click="openDetail(f)"
        >
          <div class="thumb">
            <img
              v-if="f.has_image && !failedImgs.has(f.frame_id)"
              :src="imgUrl(f.frame_id)"
              loading="lazy"
              alt="frame"
              @error="failedImgs.add(f.frame_id)"
            />
            <div v-else class="thumb-na">
              <el-icon><Picture /></el-icon>
              <span>图像加载失败</span>
            </div>
            <div class="verdict-tip">
              <el-tooltip
                :content="verdictMeta(f.verdict).tip || verdictMeta(f.verdict).label"
                placement="top"
                :show-after="150"
              >
                <el-tag
                  class="verdict-tag"
                  size="small"
                  :type="verdictMeta(f.verdict).type"
                  effect="dark"
                >
                  {{ verdictMeta(f.verdict).label }}
                </el-tag>
              </el-tooltip>
            </div>
          </div>
          <div class="card-body">
            <div class="card-row">
              <span class="fid">#{{ f.frame_id }}</span>
              <span class="ts">{{ fmtClock(f.timestamp) }}</span>
            </div>
            <div class="card-row" v-if="f.mae != null">
              <span class="lbl">MAE</span>
              <span class="val">{{ f.mae }}</span>
            </div>
            <div class="ocr-preview" v-if="f.ocr">
              <div class="ocr-count">
                <el-icon><Document /></el-icon>
                {{ f.ocr.line_count }} 行文本
                <el-tag v-if="f.ocr.reused" size="small" type="info">缓存复用</el-tag>
              </div>
              <div class="ocr-lines">
                <div v-for="(t, i) in f.ocr.preview" :key="i" class="ocr-line">
                  {{ t }}
                </div>
                <div v-if="f.ocr.line_count > 6" class="ocr-more">
                  +{{ f.ocr.line_count - 6 }} 行更多…
                </div>
              </div>
            </div>
            <div v-else class="ocr-none">无OCR结果</div>
          </div>
        </div>
      </div>

      <el-empty
        v-if="!loading && frames.length === 0"
        description="该筛选分组下没有帧"
      />
    </div>

    <!-- 分页 / pagination -->
    <div class="pager">
      <el-pagination
        v-model:current-page="page"
        v-model:page-size="pageSize"
        :total="total"
        :page-sizes="[12, 24, 48, 96]"
        layout="total, sizes, prev, pager, next, jumper"
        background
        @current-change="reload()"
        @size-change="reload(1)"
      />
    </div>

    <!-- 帧详情对话框: 大图 + OCR 行对照 / detail dialog: large image + OCR lines -->
    <el-dialog
      v-model="dialogVisible"
      :title="dialogTitle"
      width="880px"
      destroy-on-close
    >
      <div v-if="detail" class="detail-body">
        <div class="detail-img">
          <img
            v-if="detail.has_image"
            :src="imgUrl(detail.frame_id)"
            alt="frame-large"
          />
          <div v-else class="thumb-na big">该帧未保存图像（去重/模糊帧不落盘）</div>
        </div>
        <div class="detail-side">
          <div class="detail-meta">
            <el-tag size="small" :type="verdictMeta(detail.verdict).type">
              {{ verdictMeta(detail.verdict).label }}
            </el-tag>
            <span class="meta-item">帧 #{{ detail.frame_id }}</span>
            <span class="meta-item">时间 {{ fmtClock(detail.timestamp) }}</span>
            <span class="meta-item" v-if="detail.state">状态 {{ detail.state }}</span>
            <span class="meta-item" v-if="detail.mae != null">MAE {{ detail.mae }}</span>
          </div>
          <div class="ocr-list">
            <div class="ocr-list-title">
              本帧 OCR 识别结果（{{ ocrLines.length }} 行）
            </div>
            <el-scrollbar class="ocr-list-scroll">
              <div
                v-for="(l, i) in ocrLines"
                :key="i"
                class="ocr-row"
                :class="{ reused: detail.ocr?.reused }"
              >
                <span class="ocr-idx">{{ i + 1 }}</span>
                <span class="ocr-text">{{ l.text }}</span>
                <span
                  class="ocr-conf"
                  :class="confClass(l.confidence)"
                >
                  {{ (l.confidence * 100).toFixed(1) }}%
                </span>
              </div>
              <div v-if="ocrLines.length === 0" class="ocr-empty">
                该帧没有 OCR 文本行
              </div>
            </el-scrollbar>
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue'
import { Document, Picture, Refresh } from '@element-plus/icons-vue'
import { api } from '../api'
import { fmtClock, verdictMeta } from '../utils'

const props = defineProps({
  sid: { type: String, required: true },
  refreshTick: { type: Number, default: 0 }, // 提取中外部刷新信号 / external refresh signal
})

// 筛选分组: 标签 + 悬停解释 / filter groups: label + hover tip
const GROUPS = {
  all: { label: '全部', tip: '查看所有被抽帧的候选帧' },
  ocr: {
    label: '实际OCR',
    tip: '真正完成OCR识别并产生新文本的帧',
  },
  dup: {
    label: '视觉去重',
    tip: '与相邻帧画面几乎相同，被视觉去重跳过的帧',
  },
  blur: {
    label: '模糊',
    tip: '画面模糊（拉普拉斯梯度低于阈值），文字无法辨认被过滤',
  },
  suppressed: {
    label: '抑制',
    tip: '长滚/静止等低价值画面被抑制，不进入OCR',
  },
  value_break: {
    label: '值断裂',
    tip: '文本值突变（时间戳/页码跳变），可能发生页面跳转或快速滚动',
  },
  thinned: {
    label: '滚动稀化',
    tip: '快速滚动中间帧被稀化丢弃，只保留代表性帧',
  },
}

const group = ref('all')
const page = ref(1)
const pageSize = ref(24)
const total = ref(0)
const frames = ref([])
const loading = ref(false)

const dialogVisible = ref(false)
const detail = ref(null)
const detailFid = ref(null)

// 记录图像加载失败的帧(兜底抽帧也可能失败) / frames whose image failed to load
const failedImgs = reactive(new Set())

const imgUrl = (fid) => api.frameImageUrl(props.sid, fid)

const dialogTitle = computed(() => {
  if (!detail.value) return ''
  return `帧 #${detail.value.frame_id} · ${fmtClock(detail.value.timestamp)}`
})

const ocrLines = computed(() => detail.value?.ocr?.lines || [])

watch(
  () => props.sid,
  () => reload(1)
)

// 外部刷新信号(提取中/提取完成): 静默刷新, 不闪加载遮罩; 若已有请求在途则跳过
// external refresh signal: silent reload without overlay; skip if in flight
watch(
  () => props.refreshTick,
  () => {
    if (!loading.value) reload(undefined, true)
  }
)

defineExpose({ reload })

async function reload(targetPage, silent = false) {
  if (!silent) loading.value = true
  try {
    const data = await api.listFrames(
      props.sid,
      targetPage || page.value,
      pageSize.value,
      group.value
    )
    frames.value = data.items || []
    total.value = data.total || 0
    page.value = data.page || 1
  } catch (e) {
    frames.value = []
    total.value = 0
    console.error('加载帧列表失败', e)
  } finally {
    loading.value = false
  }
}

async function openDetail(f) {
  detailFid.value = f.frame_id
  dialogVisible.value = true
  detail.value = null
  try {
    detail.value = await api.frameDetail(props.sid, f.frame_id)
  } catch (e) {
    console.error('加载帧详情失败', e)
  }
}

function confClass(c) {
  if (c >= 0.9) return 'high'
  if (c >= 0.7) return 'mid'
  return 'low'
}

reload(1)
</script>

<style scoped>
.frame-inspector {
  /* 随整页滚动, 不设固定高度 / flows with the page, no fixed height */
}

.toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
  flex-wrap: wrap;
  gap: 10px;
}

.toolbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.count-label {
  font-size: 13px;
  color: var(--el-text-color-secondary);
}

.grid-wrap {
  padding-right: 4px;
}

.frame-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
  gap: 14px;
}

.frame-card {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.15s ease;
  background: var(--el-bg-color);
}

.frame-card:hover {
  border-color: var(--el-color-primary-light-5);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.thumb {
  position: relative;
  aspect-ratio: 9 / 16;
  max-height: 240px;
  background: #0f1420;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.thumb img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.thumb-na {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  color: #8a93a6;
  font-size: 12px;
}

.thumb-na.big {
  min-height: 320px;
  justify-content: center;
}

.verdict-tip {
  /* tooltip 外层做绝对定位, 保证悬停区域可用 /
     absolute position on the tooltip wrapper so hover works */
  position: absolute;
  top: 8px;
  left: 8px;
  z-index: 1;
}

.card-body {
  padding: 10px 12px;
}

.card-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 4px;
}

.fid {
  font-weight: 700;
  font-size: 14px;
  color: var(--el-text-color-primary);
}

.ts {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  font-variant-numeric: tabular-nums;
}

.lbl {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.val {
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  color: var(--el-text-color-primary);
}

.ocr-preview {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--el-border-color-lighter);
}

.ocr-count {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 600;
  color: var(--el-color-primary);
  margin-bottom: 6px;
}

.ocr-lines {
  font-size: 12px;
  color: var(--el-text-color-regular);
}

.ocr-line {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  line-height: 1.6;
}

.ocr-more {
  color: var(--el-text-color-secondary);
}

.ocr-none {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--el-border-color-lighter);
  font-size: 12px;
  color: var(--el-text-color-placeholder);
}

.pager {
  display: flex;
  justify-content: center;
  padding-top: 14px;
}

.detail-body {
  display: flex;
  gap: 18px;
}

.detail-img {
  flex: 0 0 320px;
  background: #0f1420;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  max-height: 520px;
}

.detail-img img {
  width: 100%;
  object-fit: contain;
}

.detail-side {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.detail-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.meta-item {
  font-size: 13px;
  color: var(--el-text-color-regular);
}

.ocr-list {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  overflow: hidden;
}

.ocr-list-title {
  padding: 8px 12px;
  font-size: 13px;
  font-weight: 600;
  background: var(--el-fill-color-light);
  border-bottom: 1px solid var(--el-border-color-lighter);
}

.ocr-list-scroll {
  flex: 1;
  max-height: 440px;
}

.ocr-row {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--el-border-color-extra-light);
}

.ocr-row.reused {
  background: var(--el-fill-color-light);
}

.ocr-idx {
  flex: 0 0 22px;
  font-size: 12px;
  color: var(--el-text-color-placeholder);
  text-align: right;
}

.ocr-text {
  flex: 1;
  font-size: 13px;
  color: var(--el-text-color-primary);
  word-break: break-all;
}

.ocr-conf {
  flex: 0 0 52px;
  text-align: right;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.ocr-conf.high { color: var(--el-color-success); }
.ocr-conf.mid { color: var(--el-color-warning); }
.ocr-conf.low { color: var(--el-color-danger); }

.ocr-empty {
  padding: 30px;
  text-align: center;
  color: var(--el-text-color-placeholder);
  font-size: 13px;
}
</style>
