"""翻译 API：创建任务、查询状态、获取结果与批量导出。"""
from __future__ import annotations

import io
import json
import re
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from app.config import get_settings
from app.models.schemas import (
    BatchStatusItem,
    BatchStatusRequest,
    BatchStatusResponse,
    BatchTaskItem,
    BatchTranslateResponse,
    TranslateResponse,
    TranslateStatus,
    SourceLangCode,
    LangCode,
)
from app.services.task_manager import TranslationTaskManager

router = APIRouter(prefix="/api", tags=["translate"])
_DEFAULT_SOURCE_LANG = get_settings().default_source_lang
_DEFAULT_TARGET_LANG = get_settings().default_target_lang


def get_manager() -> TranslationTaskManager:
    from app.main import get_manager as _gm

    return _gm()


def _validate_filename(filename: str | None) -> None:
    settings = get_settings()
    if not filename:
        raise HTTPException(status_code=400, detail="文件名无效")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    allowed = settings.allowed_extensions
    if f".{ext}" not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: .{ext}，仅支持 {allowed}")


def _validate_upload(file: UploadFile) -> None:
    _validate_filename(file.filename)


def _validate_languages(source_lang: str, target_lang: str) -> None:
    if source_lang not in ("auto", "ja", "en", "zh"):
        raise HTTPException(status_code=400, detail=f"不支持的源语言: {source_lang}")
    if target_lang not in ("ja", "en", "zh"):
        raise HTTPException(status_code=400, detail=f"不支持的目标语言: {target_lang}")
    if source_lang != "auto" and source_lang == target_lang:
        raise HTTPException(status_code=400, detail="源语言和目标语言不能相同")


def _validate_content(content: bytes) -> None:
    settings = get_settings()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过大小限制（{settings.max_upload_mb}MB），请压缩后重试",
        )


def _task_filename(task: dict, fallback: str = "未知文件") -> str:
    return str((task.get("meta") or {}).get("filename") or fallback)


def _safe_archive_name(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or "image.png"


def _batch_status_item(task_id: str, task: dict | None, request_index: int) -> BatchStatusItem:
    if task is None:
        return BatchStatusItem(
            task_id=task_id,
            filename="未知文件",
            index=request_index,
            status="failed",
            error="任务不存在",
        )
    meta = task.get("meta") or {}
    return BatchStatusItem(
        task_id=task_id,
        filename=_task_filename(task),
        index=int(meta.get("index") or request_index),
        status=task["status"],
        progress=int(task.get("progress") or 0),
        text_count=int(task.get("text_count") or 0),
        duration_ms=int(task.get("duration_ms") or 0),
        error=task.get("error") or None,
        source_lang=task.get("source_lang") or "auto",
        target_lang=task.get("target_lang") or "zh",
        detected_source_lang=meta.get("detected_source_lang"),
        translation_backends=meta.get("translation_backends") or [],
        translation_failures=meta.get("translation_failures") or [],
        quality_warnings=meta.get("quality_warnings") or [],
        ocr_backend=meta.get("ocr_backend") or "",
        render_font=meta.get("render_font") or "",
        region_diagnostics=meta.get("region_diagnostics") or [],
        performance=meta.get("performance") or {},
    )


@router.post("/translate", response_model=TranslateResponse)
async def translate_image(
    file: UploadFile,
    source_lang: SourceLangCode = Query(_DEFAULT_SOURCE_LANG, description="源语言：auto 自动检测，或 zh、ja、en"),
    target_lang: LangCode = Query(_DEFAULT_TARGET_LANG, description="目标语言：zh、ja、en"),
    manager: TranslationTaskManager = Depends(get_manager),
):
    _validate_upload(file)
    _validate_languages(source_lang, target_lang)

    content = await file.read()
    _validate_content(content)

    task_id = manager.create_task(source_lang, target_lang, content, file.filename or "image.png")
    return TranslateResponse(task_id=task_id, status="queued", message="翻译任务已创建")


@router.post("/translate/batch", response_model=BatchTranslateResponse)
async def translate_batch(
    files: list[UploadFile] = File(..., alias="files[]"),
    source_lang: SourceLangCode = Form(_DEFAULT_SOURCE_LANG, description="源语言：auto 自动检测，或 zh、ja、en"),
    target_lang: LangCode = Form(_DEFAULT_TARGET_LANG, description="目标语言：zh、ja、en"),
    manager: TranslationTaskManager = Depends(get_manager),
):
    """校验全部文件后，为每张图片创建独立的现有单图任务。"""
    settings = get_settings()
    _validate_languages(source_lang, target_lang)
    if not files:
        raise HTTPException(status_code=400, detail="请至少选择一张图片")
    if len(files) > settings.batch_max_files:
        raise HTTPException(
            status_code=400,
            detail=f"批量图片数量超过限制（最多 {settings.batch_max_files} 张）",
        )

    validated: list[tuple[str, bytes]] = []
    total_bytes = 0
    for upload in files:
        _validate_upload(upload)
        content = await upload.read()
        _validate_content(content)
        total_bytes += len(content)
        validated.append((upload.filename or "image.png", content))

    if total_bytes > settings.batch_max_total_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"批量文件总大小超过限制（{settings.batch_max_total_mb}MB）",
        )

    batch_id = uuid.uuid4().hex[:16]
    items = []
    for index, (filename, content) in enumerate(validated, start=1):
        task_id = manager.create_task(
            source_lang,
            target_lang,
            content,
            filename,
            metadata={
                "batch_id": batch_id,
                "filename": filename,
                "index": index,
                "requested_source_lang": source_lang,
                "target_lang": target_lang,
            },
        )
        items.append(BatchTaskItem(task_id=task_id, filename=filename, index=index))
    return BatchTranslateResponse(total=len(items), items=items)


@router.post("/translate/batch/status", response_model=BatchStatusResponse)
async def get_batch_status(
    request: BatchStatusRequest,
    manager: TranslationTaskManager = Depends(get_manager),
):
    items = [
        _batch_status_item(task_id, manager.get_status(task_id), index)
        for index, task_id in enumerate(request.task_ids, start=1)
    ]
    completed = sum(item.status == "completed" for item in items)
    failed = sum(item.status == "failed" for item in items)
    processing = len(items) - completed - failed
    progress = round(sum(item.progress for item in items) / len(items)) if items else 0
    return BatchStatusResponse(
        total=len(items),
        completed=completed,
        processing=processing,
        failed=failed,
        progress=progress,
        items=items,
    )


@router.post("/translate/batch/zip")
async def download_batch_zip(
    request: BatchStatusRequest,
    manager: TranslationTaskManager = Depends(get_manager),
):
    manifest = []
    errors = []
    used_names: dict[str, int] = {}
    archive = io.BytesIO()

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for index, task_id in enumerate(request.task_ids, start=1):
            task = manager.get_status(task_id)
            if task is None:
                manifest.append({
                    "original_filename": "未知文件",
                    "result_filename": None,
                    "task_id": task_id,
                    "status": "failed",
                    "text_count": 0,
                    "duration_ms": 0,
                    "source_lang": None,
                    "target_lang": None,
                    "detected_source_lang": None,
                    "translation_backends": [],
                })
                errors.append(f"未知文件（{task_id}）：任务不存在")
                continue

            original_name = _task_filename(task)
            result_name = None
            error = task.get("error") or ""
            if task["status"] == "completed" and task.get("result_path"):
                result_path = manager.files.resolve(task["result_path"])
                if result_path:
                    safe_name = _safe_archive_name(original_name)
                    stem = Path(safe_name).stem or "image"
                    count = used_names.get(stem, 0) + 1
                    used_names[stem] = count
                    duplicate = f"_{count}" if count > 1 else ""
                    result_name = f"{index:02d}_{stem}{duplicate}_translated.png"
                    bundle.write(result_path, f"translated_images/{result_name}")
                else:
                    error = "结果文件不存在"
            elif task["status"] == "completed":
                error = "任务没有生成结果图片"
            elif not error:
                error = "任务尚未成功完成"

            manifest.append({
                "original_filename": original_name,
                "result_filename": result_name,
                "task_id": task_id,
                "status": task["status"],
                "text_count": int(task.get("text_count") or 0),
                "duration_ms": int(task.get("duration_ms") or 0),
                "source_lang": task.get("source_lang"),
                "target_lang": task.get("target_lang"),
                "detected_source_lang": (task.get("meta") or {}).get("detected_source_lang"),
                "translation_backends": (task.get("meta") or {}).get("translation_backends") or [],
                "translation_failures": (task.get("meta") or {}).get("translation_failures") or [],
            })
            if result_name is None:
                errors.append(f"{_safe_archive_name(original_name)}（{task_id}）：{error}")

        bundle.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        if errors:
            bundle.writestr("errors.txt", ("\n".join(errors) + "\n").encode("utf-8"))

    archive.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="translated_images.zip"'}
    return Response(archive.getvalue(), media_type="application/zip", headers=headers)


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
        source_lang=task.get("source_lang") or "auto",
        target_lang=task.get("target_lang") or "zh",
        detected_source_lang=(task.get("meta") or {}).get("detected_source_lang"),
        detection_confidence=(task.get("meta") or {}).get("detection_confidence"),
        detection_reason=(task.get("meta") or {}).get("detection_reason"),
        translation_backends=(task.get("meta") or {}).get("translation_backends") or [],
        translation_failures=(task.get("meta") or {}).get("translation_failures") or [],
        quality_warnings=(task.get("meta") or {}).get("quality_warnings") or [],
        ocr_backend=(task.get("meta") or {}).get("ocr_backend") or "",
        render_font=(task.get("meta") or {}).get("render_font") or "",
        region_diagnostics=(task.get("meta") or {}).get("region_diagnostics") or [],
        performance=(task.get("meta") or {}).get("performance") or {},
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
