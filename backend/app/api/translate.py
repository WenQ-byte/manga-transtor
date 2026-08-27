"""翻译 API：创建任务、查询状态、获取结果"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.models.schemas import TranslateResponse, TranslateStatus
from app.services.task_manager import TranslationTaskManager

router = APIRouter(prefix="/api", tags=["translate"])


def get_manager() -> TranslationTaskManager:
    from app.main import get_manager as _gm

    return _gm()


def _validate_upload(file: UploadFile) -> None:
    settings = get_settings()
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名无效")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    allowed = settings.allowed_extensions
    if f".{ext}" not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: .{ext}，仅支持 {allowed}")


@router.post("/translate", response_model=TranslateResponse)
async def translate_image(
    file: UploadFile,
    source_lang: str = "ja",
    target_lang: str = "zh",
    manager: TranslationTaskManager = Depends(get_manager),
):
    _validate_upload(file)
    if source_lang not in ("ja", "en", "zh"):
        raise HTTPException(status_code=400, detail=f"不支持的源语言: {source_lang}")
    if target_lang not in ("ja", "en", "zh"):
        raise HTTPException(status_code=400, detail=f"不支持的目标语言: {target_lang}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")

    settings = get_settings()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过大小限制（{settings.max_upload_mb}MB），请压缩后重试",
        )

    task_id = manager.create_task(source_lang, target_lang, content, file.filename or "image.png")
    return TranslateResponse(task_id=task_id, status="queued", message="翻译任务已创建")


@router.get("/translate/{task_id}/status", response_model=TranslateStatus)
async def get_status(task_id: str, manager: TranslationTaskManager = Depends(get_manager)):
    task = manager.get_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return TranslateStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        error=task.get("error") or None,
        message=task.get("error") or None,
        text_count=task.get("text_count", 0),
        duration_ms=task.get("duration_ms", 0),
    )


@router.get("/translate/{task_id}/result")
async def get_result(task_id: str, manager: TranslationTaskManager = Depends(get_manager)):
    task = manager.get_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["status"] != "completed":
        raise HTTPException(status_code=409, detail="任务尚未完成")
    result_path = task.get("result_path")
    if not result_path:
        raise HTTPException(status_code=404, detail="没有翻译结果")

    p = manager.files.resolve(result_path)
    if not p:
        raise HTTPException(status_code=404, detail="结果文件不存在")
    return FileResponse(p, media_type="image/png", filename=f"translated_{task_id}.png")


@router.delete("/translate/{task_id}")
async def delete_task(task_id: str, manager: TranslationTaskManager = Depends(get_manager)):
    ok = manager.delete_task(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"deleted": True}
