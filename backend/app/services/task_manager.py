"""翻译任务管理器：异步执行流水线，维护进度"""
from __future__ import annotations

import traceback
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from app.models.schemas import LangCode, SourceLangCode
from app.services.pipeline import PIPELINE_STEPS, TranslationPipeline, create_pipeline
from app.storage.database import Database
from app.storage.file_store import FileStore

# 各步骤权重
STEP_WEIGHTS = {
    "detect": 0.15,
    "ocr": 0.25,
    "translate": 0.30,
    "inpaint": 0.15,
    "render": 0.15,
}


class TranslationTaskManager:
    """管理翻译任务的生命周期"""

    def __init__(self, max_workers: int = 1):
        self.db = Database.get_instance()
        self.files = FileStore()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="translate")
        self._pipeline: Optional[TranslationPipeline] = None
        self._pipeline_lock = threading.Lock()

    def _get_pipeline(self) -> TranslationPipeline:
        if self._pipeline is None:
            self._pipeline = create_pipeline()
        return self._pipeline

    def create_task(
        self,
        source_lang: SourceLangCode,
        target_lang: LangCode,
        image_bytes: bytes,
        filename: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        task_id = uuid.uuid4().hex[:16]
        _, abs_path = self.files.save_upload(image_bytes, filename)
        task_meta = dict(metadata or {})
        task_meta.setdefault("requested_source_lang", source_lang)
        task_meta.setdefault("target_lang", target_lang)
        self.db.task_create(
            task_id,
            source_lang,
            target_lang,
            original_path=abs_path,
            meta=task_meta,
        )
        self._executor.submit(self._run, task_id, source_lang, target_lang, abs_path)
        return task_id

    def _run(self, task_id: str, source_lang: SourceLangCode, target_lang: LangCode, image_path: str) -> None:
        step_names = [s for s, _ in PIPELINE_STEPS]

        def progress_cb(step_name: str, progress: int) -> None:
            weight = STEP_WEIGHTS.get(step_name, 0)
            done = sum(
                STEP_WEIGHTS[k]
                for k in STEP_WEIGHTS
                if step_names.index(k) < step_names.index(step_name)
            )
            total = int((done + weight * progress / 100) * 100)
            self.db.task_update(task_id, progress=total, step=step_name)

        self.db.task_update(task_id, status="processing", progress=5)
        try:
            with self._pipeline_lock:
                had_pipeline = getattr(self, "_pipeline", None) is not None
                load_started = time.monotonic()
                pipeline = self._get_pipeline()
                pipeline_load_ms = 0 if had_pipeline else int((time.monotonic() - load_started) * 1000)
                result = pipeline.translate_image(
                    Path(image_path),
                    source_lang,
                    target_lang,
                    progress_cb=progress_cb,
                )
                performance = dict(getattr(result, "performance", {}) or {})
                performance["model_load_ms"] = int(performance.get("model_load_ms", 0)) + pipeline_load_ms
                performance["total_ms"] = int(getattr(result, "duration_ms", 0)) + pipeline_load_ms
                result.performance = performance

            supports_meta = hasattr(self.db, "task_get")
            task = self.db.task_get(task_id) if supports_meta else {}
            task = task or {}
            meta = dict(task.get("meta") or {})
            meta.update({
                "detected_source_lang": getattr(result, "detected_source_lang", source_lang),
                "detection_confidence": getattr(result, "detection_confidence", 1.0),
                "detection_reason": getattr(result, "detection_reason", "显式指定源语言"),
                "ocr_backend": getattr(result, "ocr_backend", ""),
                "translation_backends": sorted(set(getattr(result, "translation_backends", []))),
                "translation_failures": getattr(result, "translation_failures", []),
                "quality_warnings": getattr(result, "quality_warnings", []),
                "translation_skipped": getattr(result, "translation_skipped", False),
                "region_diagnostics": getattr(result, "region_diagnostics", []),
                "render_font": getattr(result, "render_font", ""),
                "performance": getattr(result, "performance", {}),
            })

            image_bytes = getattr(result, "image_bytes", None)
            if not result.regions and not image_bytes:
                self.db.task_update(
                    task_id, status="completed", progress=100, step="render",
                    text_count=0, duration_ms=result.duration_ms,
                    **({"meta": meta} if supports_meta else {}),
                )
                return

            if image_bytes:
                _, result_rel = self.files.save_result(image_bytes, ".png")
                self.db.task_update(
                    task_id, status="completed", progress=100, step="render",
                    result_path=result_rel, text_count=len(result.regions),
                    duration_ms=result.duration_ms,
                    **({"meta": meta} if supports_meta else {}),
                )
            else:
                self.db.task_update(
                    task_id, status="completed", progress=100, step="render",
                    text_count=len(result.regions), duration_ms=result.duration_ms,
                    **({"meta": meta} if supports_meta else {}),
                )
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.db.task_update(task_id, status="failed", error=str(e))

    def get_status(self, task_id: str) -> Optional[dict]:
        return self.db.task_get(task_id)

    def delete_task(self, task_id: str) -> bool:
        task = self.db.task_get(task_id)
        if not task:
            return False
        if task.get("result_path"):
            self.files.delete(task["result_path"])
        if task.get("original_path") and Path(task["original_path"]).exists():
            try:
                Path(task["original_path"]).unlink()
            except OSError:
                pass
        return self.db.task_delete(task_id)
