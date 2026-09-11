"""
FastAPI 应用入口 / FastAPI application entry.

启动 / Run:
  python web/run.py                    # 便捷启动 / convenience
  uvicorn web.backend.main:app --port 8000
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import sessions, ws
from .config import FRONTEND_DIST

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("web")

app = FastAPI(
    title="手机录屏内容提取 - 演示服务 / Screen-Recorder Extraction Demo",
    version="0.1.0",
)

# 开发期允许跨域(前端Vite dev server); 生产同源无需 / CORS for Vite dev only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router)
app.include_router(ws.router)


@app.get("/api/health")
def health():
    """健康检查 / Health check."""
    return {"ok": True}


# ----------------------------------------------------------------------
# 前端静态资源托管(SPA) / frontend static hosting (SPA)
# ----------------------------------------------------------------------
def _spa_or_404(path: str) -> FileResponse:
    if FRONTEND_DIST.is_dir():
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            # 不缓存入口HTML, 保证每次都能拿到最新构建(避免浏览器旧缓存)
            # Never cache the SPA entry so the browser always gets the
            # latest build (avoids stale-bundle confusion).
            return FileResponse(
                str(index), media_type="text/html",
                headers={"Cache-Control": "no-cache, must-revalidate"},
            )
    return JSONResponse({"detail": "前端未构建: 请先在 web/frontend 执行 "
                                    "npm run build"}, status_code=404)


if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"),
              name="assets")
    # 非API路径统一回退到 index.html (SPA路由)
    # Non-API paths fall back to index.html (SPA routing)
    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        return _spa_or_404(full_path)
else:
    @app.get("/{full_path:path}")
    def spa_fallback_unbuilt(full_path: str):
        return _spa_or_404(full_path)
