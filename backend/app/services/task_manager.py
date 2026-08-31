"""翻译任务管理器：异步执行流水线，维护进度"""
from __future__ import annotations

import traceback
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from app.models.schemas import LangCode
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
        source_lang: LangCode,
        target_lang: LangCode,
        image_bytes: bytes,
        filename: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        task_id = uuid.uuid4().hex[:16]
        _, abs_path = self.files.save_upload(image_bytes, filename)
        self.db.task_create(
            task_id,
            source_lang,
            target_lang,
            original_path=abs_path,
            meta=metadata,
        )
        self._executor.submit(self._run, task_id, source_lang, target_lang, abs_path)
        return task_id

    def _run(self, task_id: str, source_lang: LangCode, target_lang: LangCode, image_path: str) -> None:
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
                pipeline = self._get_pipeline()
                result = pipeline.translate_image(
                    Path(image_path),
                    source_lang,
                    target_lang,
                    progress_cb=progress_cb,
                )

            if not result.regions:
                self.db.task_update(
                    task_id, status="completed", progress=100, step="render",
                    text_count=0, duration_ms=result.duration_ms,
                )
                return

            if result.image_bytes:
                _, result_rel = self.files.save_result(result.image_bytes, ".png")
                self.db.task_update(
                    task_id, status="completed", progress=100, step="render",
                    result_path=result_rel, text_count=len(result.regions),
                    duration_ms=result.duration_ms,
                )
            else:
                self.db.task_update(
                    task_id, status="completed", progress=100, step="render",
                    text_count=len(result.regions), duration_ms=result.duration_ms,
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
