"""文件服务 API：预览上传原图与结果图"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.services.task_manager import TranslationTaskManager

router = APIRouter(prefix="/api/files", tags=["files"])


def get_manager() -> TranslationTaskManager:
    from app.main import get_manager as _gm

    return _gm()


@router.get("/{filename}")
async def get_file(filename: str, manager: TranslationTaskManager = Depends(get_manager)):
    p = manager.files.resolve(filename)
    if not p:
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(p)
