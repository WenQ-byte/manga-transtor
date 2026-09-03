"""翻译任务管理器：异步执行流水线，维护进度"""
from __future__ import annotations

import traceback
import threading
import time
import uuid
import io
import pickle
import copy
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from app.models.schemas import LangCode, SourceLangCode
from app.services.pipeline import PIPELINE_STEPS, TextRegion, TranslationPipeline, create_pipeline
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

            if result.regions and getattr(result, "image_bytes", None):
                cleaned = getattr(result, "_cleaned_image_path", None)
                if cleaned and Path(cleaned).exists():
                    state = {"regions": result.regions, "cleaned_image": Path(cleaned).read_bytes()}
                    state_io = io.BytesIO()
                    pickle.dump(state, state_io)
                    state_rel, _ = self.files.save_state(state_io.getvalue())
                    meta["polish_state_path"] = state_rel

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
            polish_style = str(meta.get("polish_style") or "").strip()
            if polish_style:
                try:
                    self.polish_task(task_id, polish_style, str(meta.get("custom_prompt") or ""))
                except Exception as polish_error:  # noqa: BLE001
                    latest = self.db.task_get(task_id) or {}
                    latest_meta = dict(latest.get("meta") or meta)
                    latest_meta["polish_error"] = str(polish_error)
                    self.db.task_update(task_id, meta=latest_meta)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.db.task_update(task_id, status="failed", error=str(e))

    def get_status(self, task_id: str) -> Optional[dict]:
        return self.db.task_get(task_id)

    def erase_task(self, task_id: str, mask_bytes: bytes) -> dict:
        """仅修复用户提交的 mask；不重新检测、识别或翻译。"""
        task = self.db.task_get(task_id)
        if not task or task.get("status") != "completed":
            raise ValueError("只有已完成的翻译任务可以擦除修复")
        state_rel = (task.get("meta") or {}).get("polish_state_path")
        state_path = self.files.resolve_state(state_rel) if state_rel else None
        if not state_path:
            raise ValueError("该任务没有可编辑的图片状态")
        if not mask_bytes:
            raise ValueError("擦除区域不能为空")
        from PIL import Image

        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        base = Image.open(io.BytesIO(state["cleaned_image"])).convert("RGB")
        try:
            mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("擦除 mask 图片无效") from exc
        if mask_image.size != base.size:
            raise ValueError("擦除 mask 尺寸与原图不一致")
        # 以 TextRegion 的缓存 mask 接入现有 inpainter，确保 CV/LaMa 自动选择逻辑不分叉。
        import numpy as np
        mask_array = np.asarray(mask_image, dtype=np.uint8)
        if not mask_array.any():
            raise ValueError("擦除区域不能为空")
        ys, xs = np.where(mask_array > 0)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        manual = TextRegion(
            box=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            source="manual",
            mask={"bbox": (x0, y0, x1, y1), "patch": mask_array[y0:y1, x0:x1]},
        )
        before_image = state["cleaned_image"]
        before_rel, _ = self.files.save_state(before_image)
        temp_base = self.files.save_state(before_image)[1]
        try:
            with self._pipeline_lock:
                pipeline = self._get_pipeline()
                repaired_path = pipeline.inpainter.inpaint(Path(temp_base), [manual])
                repaired_bytes = Path(repaired_path).read_bytes()
                Path(repaired_path).unlink(missing_ok=True)
                render_path = self.files.save_state(repaired_bytes)[1]
                try:
                    image_bytes = pipeline.renderer.render(
                        Path(render_path), state["regions"], target_lang=task.get("target_lang") or "zh"
                    )
                finally:
                    Path(render_path).unlink(missing_ok=True)
            after_rel, _ = self.files.save_state(repaired_bytes)
            new_state_io = io.BytesIO()
            state["cleaned_image"] = repaired_bytes
            pickle.dump(state, new_state_io)
            new_state_rel, _ = self.files.save_state(new_state_io.getvalue())
        except Exception:
            Path(temp_base).unlink(missing_ok=True)
            self.files.delete(before_rel)
            raise
        Path(temp_base).unlink(missing_ok=True)
        operation = {
            "operationId": uuid.uuid4().hex,
            "mask": {"width": base.width, "height": base.height, "bbox": [x0, y0, x1, y1]},
            "beforeImage": before_rel,
            "afterImage": after_rel,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        meta = dict(task.get("meta") or {})
        history = list(meta.get("erase_history") or [])
        history.append(operation)
        dropped_operations = history[:-3]
        history = history[-3:]
        meta["erase_history"] = history
        meta["polish_state_path"] = new_state_rel
        old_result = task.get("result_path")
        try:
            _, new_result = self.files.save_result(image_bytes, ".png")
            self.db.task_update(task_id, result_path=new_result, meta=meta)
        except Exception:
            self.files.delete(after_rel)
            Path(new_state_rel).unlink(missing_ok=True)
            self.files.delete(before_rel)
            raise
        if old_result and old_result != new_result:
            self.files.delete(old_result)
        if state_rel != new_state_rel:
            state_path.unlink(missing_ok=True)
        for dropped in dropped_operations:
            self.files.delete(dropped.get("beforeImage", ""))
            self.files.delete(dropped.get("afterImage", ""))
        return {"task_id": task_id, "status": "completed", "operation": operation}

    def undo_erase_task(self, task_id: str) -> dict:
        """只回退最近一次擦除的底图，保留当前 TextBox 状态。"""
        task = self.db.task_get(task_id)
        if not task or task.get("status") != "completed":
            raise ValueError("只有已完成的翻译任务可以撤销擦除")
        meta = dict(task.get("meta") or {})
        history = list(meta.get("erase_history") or [])
        if not history:
            raise ValueError("没有可撤销的擦除操作")
        operation = history[-1]
        before_path = self.files.resolve_state(operation.get("beforeImage", ""))
        state_rel = meta.get("polish_state_path")
        state_path = self.files.resolve_state(state_rel) if state_rel else None
        if not before_path or not state_path:
            raise ValueError("擦除历史已失效，无法撤销")
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        before_bytes = before_path.read_bytes()
        with self._pipeline_lock:
            pipeline = self._get_pipeline()
            render_path = self.files.save_state(before_bytes)[1]
            try:
                image_bytes = pipeline.renderer.render(Path(render_path), state["regions"], target_lang=task.get("target_lang") or "zh")
            finally:
                Path(render_path).unlink(missing_ok=True)
        state["cleaned_image"] = before_bytes
        state_io = io.BytesIO()
        pickle.dump(state, state_io)
        new_state_rel, _ = self.files.save_state(state_io.getvalue())
        _, new_result = self.files.save_result(image_bytes, ".png")
        history.pop()
        meta["erase_history"] = history
        meta["polish_state_path"] = new_state_rel
        old_result = task.get("result_path")
        self.db.task_update(task_id, result_path=new_result, meta=meta)
        if old_result and old_result != new_result:
            self.files.delete(old_result)
        if state_rel != new_state_rel:
            state_path.unlink(missing_ok=True)
        self.files.delete(operation.get("beforeImage", ""))
        self.files.delete(operation.get("afterImage", ""))
        return {"task_id": task_id, "status": "completed", "undoneOperationId": operation.get("operationId"), "has_erase_history": bool(history)}

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
        state_rel = (task.get("meta") or {}).get("polish_state_path")
        if state_rel:
            state_path = self.files.resolve_state(state_rel)
            if state_path:
                state_path.unlink(missing_ok=True)
        for operation in (task.get("meta") or {}).get("erase_history") or []:
            self.files.delete(operation.get("beforeImage", ""))
            self.files.delete(operation.get("afterImage", ""))
        return self.db.task_delete(task_id)

    def restore_text_region(self, task_id: str, region_index: int) -> dict:
        """恢复选中文本框的原图：只把该文本框区域替换回原图，不移除/重排其它文本框。"""
        task = self.db.task_get(task_id)
        if not task or task.get("status") != "completed":
            raise ValueError("只有已完成的翻译任务可以恢复原图")
        original_path = Path(task.get("original_path") or "")
        if not original_path.is_file():
            raise ValueError("原始图片不存在，无法恢复")
        result_rel = task.get("result_path")
        result_path = self.files.resolve(result_rel) if result_rel else None
        if not result_path or not result_path.is_file():
            raise ValueError("没有可恢复的翻译结果")
        state_rel = (task.get("meta") or {}).get("polish_state_path")
        state_path = self.files.resolve_state(state_rel) if state_rel else None
        if not state_path:
            raise ValueError("该任务没有可编辑的文本区域")
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        regions = state["regions"]
        if not 0 <= region_index < len(regions):
            raise ValueError("文本区域不存在")
        selected = regions[region_index]
        removed_indices = (
            [region_index]
            if selected.group_index is None
            else [index for index, region in enumerate(regions) if region.group_index == selected.group_index]
        )
        removed = [regions[index] for index in removed_indices]
        remaining = [region for index, region in enumerate(regions) if index not in set(removed_indices)]
        from PIL import Image, ImageDraw

        original_image = Image.open(original_path).convert("RGB")
        restored_mask = Image.new("L", original_image.size, 0)
        drawer = ImageDraw.Draw(restored_mask)
        for region in removed:
            bounds = getattr(region, "render_bounds", None) or getattr(region, "bounds", None)
            if bounds:
                x0, y0, x1, y1 = [int(value) for value in bounds]
                drawer.rectangle((x0, y0, x1, y1), fill=255)
            else:
                points = region.box or []
                if points:
                    drawer.polygon([(int(point[0]), int(point[1])) for point in points], fill=255)

        result_image = Image.open(result_path).convert("RGB")
        if result_image.size != original_image.size:
            raise ValueError("结果图片尺寸与原始图片不一致，无法恢复")
        result_image.paste(original_image, mask=restored_mask)
        result_bytes = io.BytesIO()
        result_image.save(result_bytes, format="PNG")

        cleaned_bytes = state.get("cleaned_image")
        if cleaned_bytes:
            try:
                cleaned_image = Image.open(io.BytesIO(cleaned_bytes)).convert("RGB")
            except Exception:
                cleaned_image = None
            if cleaned_image is not None and cleaned_image.size == original_image.size:
                cleaned_image.paste(original_image, mask=restored_mask)
                cleaned_io = io.BytesIO()
                cleaned_image.save(cleaned_io, format="PNG")
                state["cleaned_image"] = cleaned_io.getvalue()

        state["regions"] = remaining
        new_state_io = io.BytesIO()
        pickle.dump(state, new_state_io)
        new_state_rel, _ = self.files.save_state(new_state_io.getvalue())
        _, new_result = self.files.save_result(result_bytes.getvalue(), ".png")
        meta = dict(task.get("meta") or {})
        edit_history = list(meta.get("edit_history") or [])
        edit_history.append({
            "action": "restore_original",
            "region_index": region_index,
            "region_indices": removed_indices,
            "textbox_ids": [getattr(region, "textbox_id", "") for region in removed],
        })
        meta["edit_history"] = edit_history[-100:]
        meta["polish_state_path"] = new_state_rel
        old_result = task.get("result_path")
        self.db.task_update(task_id, result_path=new_result, meta=meta)
        if old_result and old_result != new_result:
            self.files.delete(old_result)
        if state_rel != new_state_rel:
            state_path.unlink(missing_ok=True)
        return {"task_id": task_id, "status": "completed", "restored_region_indices": removed_indices}

    def polish_task(self, task_id: str, style: str, custom_prompt: str = "") -> dict:
        """润色已完成任务并仅在完整成功后替换结果图片。"""
        task = self.db.task_get(task_id)
        if not task or task.get("status") != "completed":
            raise ValueError("只有已完成的翻译任务可以润色")
        state_rel = (task.get("meta") or {}).get("polish_state_path")
        state_path = self.files.resolve_state(state_rel) if state_rel else None
        if not state_path:
            raise ValueError("该任务缺少可润色的排版状态，请重新翻译图片")
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        regions = state["regions"]
        groups: dict[int, list] = {}
        for region in regions:
            key = region.group_index if region.group_index is not None else id(region)
            groups.setdefault(key, []).append(region)
        texts = [
            next((region.group_translated for region in group if region.group_translated.strip()), "")
            or "\n".join(region.translated for region in group if region.translated.strip())
            for group in groups.values()
        ]
        if not any(text.strip() for text in texts):
            raise ValueError("任务没有可润色的译文")
        with self._pipeline_lock:
            pipeline = self._get_pipeline()
            translator = getattr(pipeline, "translator", None)
            from app.services.engines.translator import DeepSeekTranslator
            deepseek = next((item for item in getattr(translator, "_backends", []) if isinstance(item, DeepSeekTranslator)), None)
            deepseek = deepseek or DeepSeekTranslator()
            polished = deepseek.polish_batch(
                texts, task.get("source_lang") or "auto", task.get("target_lang") or "zh", style, custom_prompt
            )
            if len(polished) != len(texts):
                raise ValueError("润色结果数量不一致")
            originals = [region.group_translated or region.translated for region in regions]
            for group, value in zip(groups.values(), polished):
                lines = value.split("\n")
                for index, region in enumerate(group):
                    region.group_translated = value
                    region.translated = lines[index] if index < len(lines) else lines[-1]
            cleaned_path = self.files.save_state(state["cleaned_image"])[1]
            image_bytes = pipeline.renderer.render(Path(cleaned_path), regions, target_lang=task.get("target_lang") or "zh")
            _, new_result = self.files.save_result(image_bytes, ".png")
            Path(cleaned_path).unlink(missing_ok=True)
            new_state_io = io.BytesIO()
            pickle.dump({"regions": regions, "cleaned_image": state["cleaned_image"]}, new_state_io)
            new_state_rel, _ = self.files.save_state(new_state_io.getvalue())
        meta = dict(task.get("meta") or {})
        history = list(meta.get("polish_history") or [])
        history.append({"style": style, "custom_prompt": custom_prompt, "original_translations": originals})
        meta.update({"polish_history": history[-20:], "polish_style": style, "polished": True})
        old_result = task.get("result_path")
        meta["polish_state_path"] = new_state_rel
        self.db.task_update(task_id, result_path=new_result, meta=meta)
        if new_state_rel != state_rel:
            state_path.unlink(missing_ok=True)
        if old_result and old_result != new_result:
            self.files.delete(old_result)
        return {"task_id": task_id, "status": "completed", "style": style}

    def get_edit_regions(self, task_id: str) -> dict:
        task = self.db.task_get(task_id)
        if not task or task.get("status") != "completed":
            raise ValueError("任务尚未完成")
        state_rel = (task.get("meta") or {}).get("polish_state_path")
        state_path = self.files.resolve_state(state_rel) if state_rel else None
        if not state_path:
            raise ValueError("该任务没有可编辑的文本区域")
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        from PIL import Image
        image_size = Image.open(io.BytesIO(state["cleaned_image"])).size
        needs_render_bounds = any(
            (
                (
                    not getattr(region, "layout_bounds_override", False)
                    and not getattr(region, "render_bounds", None)
                )
                or not getattr(region, "render_box", None)
                or not getattr(region, "render_layout", None)
            )
            and bool((getattr(region, "group_translated", "") or getattr(region, "translated", "")).strip())
            for region in state["regions"]
        )
        if needs_render_bounds:
            with self._pipeline_lock:
                pipeline = self._get_pipeline()
                temp_path = self.files.save_state(state["cleaned_image"])[1]
                try:
                    pipeline.renderer.render(
                        Path(temp_path), state["regions"], target_lang=task.get("target_lang") or "zh"
                    )
                    state_io = io.BytesIO()
                    pickle.dump(state, state_io)
                    new_state_rel, _ = self.files.save_state(state_io.getvalue())
                finally:
                    Path(temp_path).unlink(missing_ok=True)
            meta = dict(task.get("meta") or {})
            meta["polish_state_path"] = new_state_rel
            self.db.task_update(task_id, meta=meta)
            if state_rel != new_state_rel:
                state_path.unlink(missing_ok=True)
            state_rel = new_state_rel
            state_path = self.files.resolve_state(state_rel)
        grouped: dict[tuple[str, int], list[tuple[int, object]]] = {}
        for index, region in enumerate(state["regions"]):
            key = ("group", region.group_index) if region.group_index is not None else ("region", index)
            grouped.setdefault(key, []).append((index, region))
        regions = []
        for items in grouped.values():
            indices = [index for index, _ in items]
            members = [region for _, region in items]
            bounds = next((getattr(region, "render_bounds", None) for region in members if getattr(region, "render_bounds", None)), None)
            if not bounds:
                bounds = next((region.group_bounds for region in members if region.group_bounds), None)
            if bounds:
                x0, y0, x1, y1 = bounds
            else:
                all_points = [point for region in members for point in region.box]
                x0 = min(point[0] for point in all_points)
                y0 = min(point[1] for point in all_points)
                x1 = max(point[0] for point in all_points)
                y1 = max(point[1] for point in all_points)
            translated = next((region.group_translated for region in members if region.group_translated.strip()), "")
            if not translated:
                translated = "\n".join(region.translated for region in members if region.translated.strip())
            regions.append({
                "id": getattr(members[0], "textbox_id", "") or f"ocr-{indices[0]}",
                "index": indices[0],
                "region_indices": indices,
                "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                "render_box": (
                    [[members[0].render_box[0], members[0].render_box[1]], [members[0].render_box[2], members[0].render_box[1]], [members[0].render_box[2], members[0].render_box[3]], [members[0].render_box[0], members[0].render_box[3]]]
                    if getattr(members[0], "render_box", None) else None
                ),
                "render_layout": getattr(members[0], "render_layout", None),
                "translated": translated,
                "group_index": members[0].group_index,
                "direction": members[0].direction,
                "font_size": getattr(members[0], "style_font_size", None),
                "default_font_size": getattr(members[0], "render_font_size", None),
                "font_weight": (
                    getattr(members[0], "style_font_weight", None)
                    if getattr(members[0], "style_font_weight", None) is not None
                    else getattr(members[0], "render_font_weight", None)
                ),
                "font_family": (
                    getattr(members[0], "style_font_family", "")
                    if not getattr(members[0], "style_font_family", "")
                    or self._get_pipeline().renderer._resolve_font(getattr(members[0], "style_font_family", ""))
                    else ""
                ),
                "color": list(getattr(members[0], "style_color", None) or []) or None,
                "source": getattr(members[0], "source", "ocr") or "ocr",
            })
        renderer = self._get_pipeline().renderer
        return {"regions": regions, "width": image_size[0], "height": image_size[1], "font_options": renderer.available_fonts(), "has_erase_history": bool((task.get("meta") or {}).get("erase_history"))}

    def get_cleaned_image(self, task_id: str) -> bytes:
        task = self.db.task_get(task_id)
        if not task or task.get("status") != "completed":
            raise ValueError("浠诲姟灏氭湭瀹屾垚")
        state_rel = (task.get("meta") or {}).get("polish_state_path")
        state_path = self.files.resolve_state(state_rel) if state_rel else None
        if not state_path:
            raise ValueError("璇ヤ换鍔℃病鏈夊彲缂栬緫鐨勫浘鐗囨柊鎭?")
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        image = state.get("cleaned_image")
        if not image:
            raise ValueError("娌℃湁鍙敤鐨勬竻鐞嗗悗搴曞浘")
        return image

    def create_text_box(self, task_id: str, x: int, y: int, width: int, height: int, font_size: int = 24, font_family: str = "", font_weight: int = 400, color: str = "", direction: str = "h") -> dict:
        task = self.db.task_get(task_id)
        if not task or task.get("status") != "completed":
            raise ValueError("只有已完成的翻译任务可以添加文本框")
        state_rel = (task.get("meta") or {}).get("polish_state_path")
        state_path = self.files.resolve_state(state_rel) if state_rel else None
        if not state_path:
            raise ValueError("该任务没有可编辑的文本区域")
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        from PIL import Image
        image_width, image_height = Image.open(io.BytesIO(state["cleaned_image"])).size
        if width < 28 or height < 28 or width > image_width or height > image_height:
            raise ValueError("文本框尺寸超出允许范围")
        x = max(0, min(int(x), image_width - int(width)))
        y = max(0, min(int(y), image_height - int(height)))
        if not 8 <= font_size <= 160 or not 100 <= font_weight <= 900:
            raise ValueError("文本样式参数无效")
        if direction not in {"h", "v"}:
            direction = "h"
        renderer = self._get_pipeline().renderer
        if font_family and not renderer._resolve_font(font_family):
            font_family = ""
        color_value = None
        if color:
            import re
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                raise ValueError("文字颜色格式无效")
            color_value = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        index = len(state["regions"])
        textbox_id = f"manual-{uuid.uuid4().hex}"
        region = TextRegion(
            box=[[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            textbox_id=textbox_id, source="manual", group_index=-(index + 1),
            group_bounds=(x, y, x + width, y + height), direction=direction,
            style_font_size=font_size, style_font_family=font_family,
            style_font_weight=font_weight, style_color=color_value, layout_bounds_override=True,
        )
        state["regions"].append(region)
        state_io = io.BytesIO()
        pickle.dump(state, state_io)
        new_state_rel, _ = self.files.save_state(state_io.getvalue())
        meta = dict(task.get("meta") or {})
        history = list(meta.get("edit_history") or [])
        history.append({"action": "create", "textbox_id": textbox_id, "source": "manual", "geometry": [x, y, width, height]})
        meta.update({"edit_history": history[-100:], "polish_state_path": new_state_rel})
        self.db.task_update(task_id, meta=meta)
        if state_rel != new_state_rel:
            state_path.unlink(missing_ok=True)
        return {"id": textbox_id, "index": index, "box": [[x, y], [x + width, y], [x + width, y + height], [x, y + height]], "translated": "", "source": "manual", "group_index": -(index + 1), "direction": direction, "font_size": font_size, "default_font_size": font_size, "font_weight": font_weight, "font_family": font_family, "color": list(color_value) if color_value else None}

    def delete_text_box(self, task_id: str, region_index: int) -> dict:
        task = self.db.task_get(task_id)
        if not task or task.get("status") != "completed":
            raise ValueError("只有已完成的翻译任务可以删除文本框")
        state_rel = (task.get("meta") or {}).get("polish_state_path")
        state_path = self.files.resolve_state(state_rel) if state_rel else None
        if not state_path:
            raise ValueError("该任务没有可编辑的文本区域")
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        regions = state["regions"]
        if not 0 <= region_index < len(regions):
            raise ValueError("文本区域不存在")
        selected = regions[region_index]
        if selected.group_index is None:
            removed_indices = [region_index]
        else:
            removed_indices = [
                index for index, region in enumerate(regions)
                if region.group_index == selected.group_index
            ]
        removed = [regions[index] for index in removed_indices]
        remaining = [region for index, region in enumerate(regions) if index not in set(removed_indices)]
        with self._pipeline_lock:
            pipeline = self._get_pipeline()
            temp_path = self.files.save_state(state["cleaned_image"])[1]
            try:
                image_bytes = pipeline.renderer.render(
                    Path(temp_path), remaining, target_lang=task.get("target_lang") or "zh"
                )
                _, new_result = self.files.save_result(image_bytes, ".png")
                state_io = io.BytesIO()
                pickle.dump({"regions": remaining, "cleaned_image": state["cleaned_image"]}, state_io)
                new_state_rel, _ = self.files.save_state(state_io.getvalue())
            finally:
                Path(temp_path).unlink(missing_ok=True)
        meta = dict(task.get("meta") or {})
        history = list(meta.get("edit_history") or [])
        history.append({
            "action": "delete",
            "textbox_id": getattr(selected, "textbox_id", "") or f"ocr-{region_index}",
            "source": getattr(selected, "source", "ocr") or "ocr",
            "region_indices": removed_indices,
            "translated": next((region.group_translated for region in removed if region.group_translated), "")
                or "\n".join(region.translated for region in removed if region.translated),
            "geometry": list(next((region.group_bounds for region in removed if region.group_bounds), selected.bounds)),
        })
        meta.update({"edit_history": history[-100:], "polish_state_path": new_state_rel})
        old_result = task.get("result_path")
        self.db.task_update(task_id, result_path=new_result, meta=meta)
        if old_result and old_result != new_result:
            self.files.delete(old_result)
        if state_rel != new_state_rel:
            state_path.unlink(missing_ok=True)
        return {
            "task_id": task_id,
            "region_index": region_index,
            "deleted_region_indices": removed_indices,
            "status": "completed",
        }

    def preview_task_region_font(self, task_id: str, region_index: int, font_family: str = "", font_weight: int = 400) -> bytes:
        task = self.db.task_get(task_id)
        if not task or task.get("status") != "completed":
            raise ValueError("只有已完成的翻译任务可以预览字体")
        state_rel = (task.get("meta") or {}).get("polish_state_path")
        state_path = self.files.resolve_state(state_rel) if state_rel else None
        if not state_path:
            raise ValueError("该任务没有可编辑的文本区域")
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        regions = copy.deepcopy(state["regions"])
        if not 0 <= region_index < len(regions):
            raise ValueError("文本区域不存在")
        renderer = self._get_pipeline().renderer
        font_weight = max(100, min(900, int(font_weight or 400)))
        if font_family and not renderer._resolve_font(font_family):
            font_family = ""
        region = regions[region_index]
        group_regions = [item for item in regions if item.group_index == region.group_index] if region.group_index is not None else [region]
        for item in group_regions:
            item.style_font_family = font_family
            item.style_font_weight = font_weight
        with self._pipeline_lock:
            pipeline = self._get_pipeline()
            temp_path = self.files.save_state(state["cleaned_image"])[1]
            try:
                return pipeline.renderer.render(Path(temp_path), regions, target_lang=task.get("target_lang") or "zh")
            finally:
                Path(temp_path).unlink(missing_ok=True)

    def edit_task_region(self, task_id: str, region_index: int, translated: str, font_size: int | None = None, font_family: str = "", color: str = "", font_weight: int | None = None, x: int | None = None, y: int | None = None, width: int | None = None, height: int | None = None, move_only: bool = False) -> dict:
        task = self.db.task_get(task_id)
        if not task or task.get("status") != "completed":
            raise ValueError("只有已完成的翻译任务可以编辑")
        value = translated.strip()
        if not value and not any(value is not None for value in (x, y, width, height)):
            raise ValueError("译文不能为空")
        if font_size is not None and not 8 <= font_size <= 160:
            raise ValueError("字号必须在 8 到 160 之间")
        if font_weight is not None and not 100 <= font_weight <= 900:
            raise ValueError("字体粗细必须在 100 到 900 之间")
        color_value = None
        if color:
            import re
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                raise ValueError("文字颜色格式无效")
            color_value = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        state_rel = (task.get("meta") or {}).get("polish_state_path")
        state_path = self.files.resolve_state(state_rel) if state_rel else None
        if not state_path:
            raise ValueError("该任务没有可编辑的文本区域")
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        from PIL import Image
        image_width, image_height = Image.open(io.BytesIO(state["cleaned_image"])).size
        geometry = None
        if any(value is not None for value in (x, y, width, height)):
            if None in (x, y, width, height) or width <= 0 or height <= 0:
                raise ValueError("文本框位置或尺寸无效")
            if width > image_width or height > image_height:
                raise ValueError("文本框尺寸不能超过图片")
            x = max(0, min(int(x), image_width - int(width)))
            y = max(0, min(int(y), image_height - int(height)))
            geometry = [x, y, x + int(width), y + int(height)]
        regions = state["regions"]
        if not 0 <= region_index < len(regions):
            raise ValueError("文本区域不存在")
        region = regions[region_index]
        group_regions = (
            [item for item in regions if item.group_index == region.group_index]
            if region.group_index is not None else [region]
        )
        previous_geometry = next((list(item.group_bounds) for item in group_regions if item.group_bounds), None)
        if geometry is not None:
            if move_only:
                anchor = next((getattr(item, "render_bounds", None) for item in group_regions if getattr(item, "render_bounds", None)), None) \
                    or next((getattr(item, "group_bounds", None) for item in group_regions if getattr(item, "group_bounds", None)), None)
                sx = sy = 1.0
                anchor_cx = anchor_cy = box_cx = box_cy = 0.0
                if anchor:
                    bw = max(1, int(anchor[2]) - int(anchor[0]))
                    bh = max(1, int(anchor[3]) - int(anchor[1]))
                    new_bw = geometry[2] - geometry[0]
                    new_bh = geometry[3] - geometry[1]
                    if new_bw <= 0 or new_bh <= 0:
                        raise ValueError("文本框尺寸无效")
                    sx = new_bw / bw
                    sy = new_bh / bh
                    anchor_cx = (int(anchor[0]) + int(anchor[2])) / 2
                    anchor_cy = (int(anchor[1]) + int(anchor[3])) / 2
                    box_cx = (geometry[0] + geometry[2]) / 2
                    box_cy = (geometry[1] + geometry[3]) / 2

                def _map_point(x, y):
                    return (int(round(box_cx + (x - anchor_cx) * sx)), int(round(box_cy + (y - anchor_cy) * sy)))

                for item in group_regions:
                    for attr in ("group_bounds", "render_box", "render_bounds"):
                        val = getattr(item, attr, None)
                        if isinstance(val, (tuple, list)) and len(val) == 4:
                            nx0, ny0 = _map_point(int(val[0]), int(val[1]))
                            nx1, ny1 = _map_point(int(val[2]), int(val[3]))
                            setattr(item, attr, (nx0, ny0, nx1, ny1))
                    box = getattr(item, "box", None)
                    if isinstance(box, (list, tuple)) and box and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in box):
                        item.box = [_map_point(int(p[0]), int(p[1])) for p in box]
                    item.layout_bounds_override = True
            else:
                for item in group_regions:
                    item.group_bounds = geometry[:]
                    item.layout_bounds_override = True
        renderer = self._get_pipeline().renderer
        if font_family and not renderer._resolve_font(font_family):
            font_family = ""
        previous = next((item.group_translated for item in group_regions if item.group_translated.strip()), "")
        if not previous:
            previous = "\n".join(item.translated for item in group_regions if item.translated.strip())
        previous_style = {
            "font_size": getattr(group_regions[0], "style_font_size", None),
            "font_family": getattr(group_regions[0], "style_font_family", ""),
            "font_weight": getattr(group_regions[0], "style_font_weight", None),
            "color": list(getattr(group_regions[0], "style_color", None) or []),
        }
        original_font_size = getattr(group_regions[0], "render_font_size", None)
        if not move_only:
            lines = value.splitlines() or [value]
            for index, item in enumerate(group_regions):
                item.group_translated = value
                item.translated = lines[index] if index < len(lines) else lines[-1]
                if font_size is not None:
                    item.style_font_size = font_size
                item.style_font_family = font_family
                if font_weight is not None:
                    item.style_font_weight = font_weight
                item.style_color = color_value
        with self._pipeline_lock:
            pipeline = self._get_pipeline()
            temp_path = self.files.save_state(state["cleaned_image"])[1]
            try:
                image_bytes = pipeline.renderer.render(Path(temp_path), regions, target_lang=task.get("target_lang") or "zh")
                effective_font_size = next(
                    (item._last_render_font_size for item in group_regions if item._last_render_font_size),
                    None,
                )
                too_small = (
                    font_size is not None
                    and original_font_size is not None
                    and font_size < max(8, original_font_size * 0.5)
                )
                too_large = (
                    font_size is not None
                    and original_font_size is not None
                    and font_size > original_font_size * 1.5
                )
                size_rejected = (
                    font_size is not None
                    and (effective_font_size is None or effective_font_size != font_size)
                ) or too_small or too_large
                if size_rejected and effective_font_size is None:
                    for item in group_regions:
                        item.style_font_size = None
                    image_bytes = pipeline.renderer.render(
                        Path(temp_path), regions, target_lang=task.get("target_lang") or "zh"
                    )
                    restored_font_size = next(
                        (item._last_render_font_size for item in group_regions if item._last_render_font_size),
                        original_font_size,
                    )
                    warning = f"字号无法在当前文本框内正常显示，已恢复原始字号 {restored_font_size or '自动'}"
                    saved_font_size = None
                else:
                    restored_font_size = effective_font_size or original_font_size
                    warning = ""
                    saved_font_size = effective_font_size or font_size
                    if effective_font_size:
                        for item in group_regions:
                            item.style_font_size = effective_font_size
                _, new_result = self.files.save_result(image_bytes, ".png")
                new_state_io = io.BytesIO()
                pickle.dump({"regions": regions, "cleaned_image": state["cleaned_image"]}, new_state_io)
                new_state_rel, _ = self.files.save_state(new_state_io.getvalue())
            finally:
                Path(temp_path).unlink(missing_ok=True)
        meta = dict(task.get("meta") or {})
        history = list(meta.get("edit_history") or [])
        history.append({
            "region_index": region_index,
            "region_indices": [index for index, item in enumerate(regions) if any(item is member for member in group_regions)],
            "previous": previous,
            "updated": value,
            "previous_style": previous_style,
            "previous_geometry": previous_geometry,
            "updated_geometry": geometry,
            "updated_style": {"font_size": saved_font_size, "font_family": font_family, "font_weight": font_weight, "color": color},
        })
        meta["edit_history"] = history[-100:]
        old_result = task.get("result_path")
        self.db.task_update(task_id, result_path=new_result, meta={**meta, "polish_state_path": new_state_rel})
        if old_result and old_result != new_result:
            self.files.delete(old_result)
        if state_rel != new_state_rel:
            state_path.unlink(missing_ok=True)
        return {
            "task_id": task_id,
            "region_index": region_index,
            "translated": value,
            "status": "completed",
            "font_size": saved_font_size,
            "default_font_size": restored_font_size,
            "font_family": font_family,
            "font_weight": font_weight if font_weight is not None else (
                getattr(group_regions[0], "style_font_weight", None)
                if getattr(group_regions[0], "style_font_weight", None) is not None
                else getattr(group_regions[0], "render_font_weight", None)
            ),
            "color": color,
            "warning": warning,
            "box": [[geometry[0], geometry[1]], [geometry[2], geometry[1]], [geometry[2], geometry[3]], [geometry[0], geometry[3]]] if geometry else None,
            "render_box": (
                [[group_regions[0].render_box[0], group_regions[0].render_box[1]], [group_regions[0].render_box[2], group_regions[0].render_box[1]], [group_regions[0].render_box[2], group_regions[0].render_box[3]], [group_regions[0].render_box[0], group_regions[0].render_box[3]]]
                if getattr(group_regions[0], "render_box", None) else None
            ),
            "render_layout": getattr(group_regions[0], "render_layout", None),
        }
