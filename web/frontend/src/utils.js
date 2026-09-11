// 格式化工具 / formatting helpers

export function fmtBytes(bytes) {
  if (bytes == null || isNaN(bytes)) return '-'
  const units = ['B', 'KB', 'MB', 'GB']
  let i = 0
  let v = Number(bytes)
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v >= 10 ? v.toFixed(1) : v.toFixed(2)} ${units[i]}`
}

export function fmtDuration(sec) {
  if (sec == null || isNaN(sec)) return '-'
  const s = Math.round(Number(sec))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  if (h > 0) return `${h}时${m}分${r}秒`
  if (m > 0) return `${m}分${r}秒`
  return `${r}秒`
}

export function fmtClock(sec) {
  // 秒 -> mm:ss 或 h:mm:ss (帧时间戳展示)
  if (sec == null || isNaN(sec)) return '-'
  const s = Math.round(Number(sec))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = String(s % 60).padStart(2, '0')
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${r}`
  return `${m}:${r}`
}

// 帧判定 -> 标签类型/文案/悬停解释 / verdict -> tag type, label & hover tip
export const VERDICT_META = {
  ocr: {
    type: 'success', label: '实际OCR',
    tip: '该帧完成了OCR识别并产生新的文本行，是真正“读到内容”的帧',
  },
  ocr_reused: {
    type: 'success', label: '缓存复用',
    tip: '该帧画面与内容指纹缓存一致，直接复用之前的OCR结果，未重复调用OCR',
  },
  dup: {
    type: 'info', label: '视觉去重',
    tip: '与相邻帧画面几乎相同（视觉特征重复），被去重跳过，未进入OCR',
  },
  blur: {
    type: 'warning', label: '模糊',
    tip: '画面模糊（拉普拉斯梯度低于阈值），文字无法辨认，已被过滤',
  },
  suppressed: {
    type: 'danger', label: '抑制',
    tip: '长滚/静止等低价值画面被抑制，不进入OCR',
  },
  long_scroll_sample: {
    type: 'danger', label: '长滚采样',
    tip: '长滚动过程中抽取的代表性帧',
  },
  value_break: {
    type: 'warning', label: '值断裂',
    tip: '文本值发生突变（如时间戳/页码跳变），可能发生了页面跳转或快速滚动',
  },
  value_fallback: {
    type: 'warning', label: '值兜底',
    tip: '值断裂后用于兜底重采的帧',
  },
  thinned: {
    type: 'info', label: '滚动稀化',
    tip: '快速滚动过程中的中间帧被稀化丢弃，只保留代表性帧',
  },
}

export function verdictMeta(v) {
  return VERDICT_META[v] || { type: 'info', label: v || '未知', tip: '' }
}

// 会话状态 -> 徽标 / session status -> tag
export const STATUS_META = {
  uploading: { type: 'primary', label: '上传中' },
  ready: { type: 'info', label: '待提取' },
  extracting: { type: 'warning', label: '提取中' },
  done: { type: 'success', label: '已完成' },
  error: { type: 'danger', label: '失败' },
}

export function statusMeta(s) {
  return STATUS_META[s] || { type: 'info', label: s || '-' }
}
