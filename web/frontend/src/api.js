// REST 封装 + WebSocket 客户端 / REST wrapper + WebSocket client

const BASE = '/api'

async function request(path, options = {}) {
  const resp = await fetch(BASE + path, options)
  if (!resp.ok) {
    let detail = resp.statusText
    try {
      const data = await resp.json()
      detail = data.detail || JSON.stringify(data)
    } catch (e) { /* keep statusText */ }
    throw new Error(detail)
  }
  return resp.json()
}

export const api = {
  health: () => request('/health'),

  // 会话 / sessions
  listSessions: () => request('/sessions'),
  getSession: (sid) => request(`/sessions/${sid}`),
  videoUrl: (sid) => `${BASE}/sessions/${sid}/video`,
  uploadVideo: (file, onProgress) => new Promise((resolve, reject) => {
    // 用 XHR 获取真实上传进度 / XHR gives real upload progress
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${BASE}/sessions/upload`)
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText))
        } catch (e) {
          reject(new Error('响应解析失败'))
        }
      } else {
        let detail = xhr.responseText
        try {
          detail = JSON.parse(xhr.responseText).detail || detail
        } catch (e) { /* raw text */ }
        reject(new Error(detail))
      }
    }
    xhr.onerror = () => reject(new Error('网络错误，请检查服务是否启动'))
    const fd = new FormData()
    fd.append('file', file)
    xhr.send(fd)
  }),
  // 启动提取: 固定使用真实OCR服务 / always real OCR (AI Studio)
  startExtract: (sid) =>
    request(`/sessions/${sid}/extract`, { method: 'POST' }),
  deleteSession: (sid) =>
    request(`/sessions/${sid}`, { method: 'DELETE' }),

  // 帧检视 / frame inspection
  listFrames: (sid, page, pageSize, group) =>
    request(`/sessions/${sid}/frames?page=${page}&page_size=${pageSize}&group=${group}`),
  frameDetail: (sid, fid) => request(`/sessions/${sid}/frames/${fid}`),
  frameImageUrl: (sid, fid) => `${BASE}/sessions/${sid}/frames/${fid}/image`,

  // 结果与日志 / result & logs
  result: (sid) => request(`/sessions/${sid}/result`),
  logs: (sid, offset, limit) =>
    request(`/sessions/${sid}/logs?offset=${offset}&limit=${limit}`),

  // 逐条语音(审听用) / per-voice wav for review
  voiceUrl: (sid, name) => `${BASE}/sessions/${sid}/voices/${name}`,
}

/**
 * 打开会话实时事件流 / Open the realtime session event stream.
 * handlers 可选键 / optional keys: snapshot / event / log / status / error / onclose / onerror
 */
export function openSessionSocket(sid, handlers) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}${BASE}/sessions/${sid}/events`)
  ws.onmessage = (e) => {
    let msg
    try {
      msg = JSON.parse(e.data)
    } catch (err) {
      return
    }
    const fn = handlers[msg.type]
    if (fn) fn(msg)
  }
  ws.onclose = (e) => handlers.onclose && handlers.onclose(e)
  ws.onerror = (e) => handlers.onerror && handlers.onerror(e)
  return ws
}
