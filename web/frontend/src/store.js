// 全局会话状态 / global session state
import { reactive } from 'vue'
import { api } from './api'

export const store = reactive({
  sessions: [],
  currentSid: null,
  loadingSessions: false,
})

export async function refreshSessions() {
  store.loadingSessions = true
  try {
    const data = await api.listSessions()
    store.sessions = data.sessions || []
  } catch (e) {
    console.error('加载会话列表失败', e)
  } finally {
    store.loadingSessions = false
  }
}

export function selectSession(sid) {
  store.currentSid = sid
}

// 用最新会话信息(详情/WS状态)同步左侧列表, 避免状态停留在旧值(如"提取中")
// Sync the left sidebar entry with fresh session info so statuses never
// stay stale (e.g. stuck on "extracting" after completion).
export function syncSession(meta) {
  if (!meta || !meta.sid) return
  const entry = {
    sid: meta.sid,
    name: meta.name,
    created_at: meta.created_at,
    updated_at: meta.updated_at,
    status: meta.status,
    stage: meta.stage,
    stage_detail: meta.stage_detail,
    progress: meta.progress,
    error: meta.error,
    summary: meta.summary,
  }
  const i = store.sessions.findIndex((s) => s.sid === meta.sid)
  if (i >= 0) {
    store.sessions[i] = { ...store.sessions[i], ...entry }
  } else {
    store.sessions.unshift(entry)
  }
}

export function clearCurrent() {
  store.currentSid = null
}
