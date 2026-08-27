"""FastAPI 应用入口"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.config import get_settings
from app.services.task_manager import TranslationTaskManager

settings = get_settings()

_manager: TranslationTaskManager | None = None

FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


def get_manager() -> TranslationTaskManager:
    """全局任务管理器（供路由依赖注入）"""
    assert _manager is not None, "manager 尚未初始化"
    return _manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager
    _manager = TranslationTaskManager()
    yield
    _manager._executor.shutdown(wait=False)


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="漫画多语言智能翻译系统 API",
    lifespan=lifespan,
)

# CORS（开发环境允许所有来源，生产可收紧）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["system"])
async def health():
    return {"status": "ok", "name": settings.app_name, "version": settings.version}


app.include_router(api_router)

# 静态文件：前端构建产物（若存在），挂载到根路径
if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        f = FRONTEND_DIST / "favicon.svg"
        return FileResponse(f) if f.exists() else FileResponse(FRONTEND_DIST / "index.html")

    # SPA 回退：非 API 路径返回 index.html
    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {"name": settings.app_name, "version": settings.version, "docs": "/docs", "hint": "前端未构建，请先在 frontend 目录执行 npm run build"}
