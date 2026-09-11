import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发模式: Vite dev server (5173) 将 /api 代理到 FastAPI (8000)
// Dev mode: proxy /api to the FastAPI backend on 8000.
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        ws: true, // WebSocket 实时事件流 / realtime event stream
      },
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 1500,
  },
})
