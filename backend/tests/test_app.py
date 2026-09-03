"""后端单元测试"""
import os
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# 测试环境
os.environ["MANGA_DATA_DIR"] = tempfile.mkdtemp(prefix="manga_test_")
# 通用流水线测试固定用 PaddleOCR + CV 检测，避免依赖本地 MIT 权重或触发网络下载
os.environ["MANGA_OCR_BACKEND"] = "paddle"
os.environ["MANGA_DETECTOR_BACKEND"] = "cv"
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI
from fastapi.testclient import TestClient

import numpy as np

from app.services.glossary_service import GlossaryService
from app.services.pipeline import create_pipeline


_MIT_MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "mit"


def make_test_image(path: Path, text: str = "Hello World") -> Path:
    img = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([40, 40, 240, 160], outline="black", width=3)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    d.text((70, 80), text, fill="black", font=font)
    img.save(path)
    return path


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.image = make_test_image(Path(self.tmp) / "test.png")

    def test_detect_finds_regions(self):
        from app.services.engines import get_engine

        detector = get_engine("detector")
        regions = detector.detect(self.image)
        self.assertGreater(len(regions), 0)

    def test_pipeline_runs(self):
        from app.services.engines.translator import SmartTranslator

        pipe = create_pipeline()
        with patch.object(SmartTranslator, "_find_available", return_value=None):
            result = pipe.translate_image(self.image, "en", "zh")
        self.assertGreater(len(result.regions), 0)
        self.assertGreater(result.duration_ms, 0)
        self.assertTrue({
            "detection_ms", "ocr_ms", "inpaint_ms", "grouping_ms",
            "translation_ms", "render_ms", "total_ms", "ocr_call_count",
            "translation_request_count",
        }.issubset(result.performance))

    def test_render_preserves_size(self):
        import io

        from app.services.engines import get_engine
        from app.services.engines.translator import SmartTranslator

        pipe = create_pipeline()
        with patch.object(SmartTranslator, "_find_available", return_value=None):
            result = pipe.translate_image(self.image, "en", "zh")
        renderer = get_engine("renderer")
        inpainter = get_engine("inpainter")
        cleaned = inpainter.inpaint(self.image, result.regions)
        out = renderer.render(cleaned, result.regions, target_lang="zh")
        img = Image.open(self.image)
        parsed = Image.open(io.BytesIO(out))
        self.assertEqual(parsed.size, img.size)


class _BatchTestFiles:
    def __init__(self, root: Path):
        self.root = root

    def resolve(self, rel: str):
        candidate = (self.root / rel).resolve()
        if self.root.resolve() not in candidate.parents or not candidate.is_file():
            return None
        return candidate


class _BatchTestManager:
    def __init__(self, root: Path):
        self.tasks = {}
        self.created = []
        self.files = _BatchTestFiles(root)

    def create_task(self, source_lang, target_lang, content, filename, metadata=None):
        task_id = f"task-{len(self.created) + 1}"
        task = {
            "id": task_id,
            "status": "queued",
            "progress": 0,
            "error": "",
            "result_path": "",
            "text_count": 0,
            "duration_ms": 0,
            "meta": metadata or {},
        }
        self.created.append((source_lang, target_lang, content, filename, metadata))
        self.tasks[task_id] = task
        return task_id

    def get_status(self, task_id):
        return self.tasks.get(task_id)

    def delete_task(self, task_id):
        return self.tasks.pop(task_id, None) is not None


class TestBatchTranslateApi(unittest.TestCase):
    """批量任务只编排现有单图任务，并安全汇总和导出结果。"""

    def setUp(self):
        from app.api.translate import get_manager, router

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manager = _BatchTestManager(self.root)
        api = FastAPI()
        api.include_router(router)
        api.dependency_overrides[get_manager] = lambda: self.manager
        self.client = TestClient(api)

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    @staticmethod
    def _files(*names):
        return [("files[]", (name, b"image", "image/png")) for name in names]

    def _settings(self, **overrides):
        values = {
            "allowed_extensions": ".jpg,.jpeg,.png,.webp,.bmp",
            "max_upload_mb": 10,
            "batch_max_files": 100,
            "batch_max_total_mb": 500,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _set_task(self, task_id, filename, **overrides):
        task = {
            "id": task_id,
            "status": "completed",
            "progress": 100,
            "error": "",
            "result_path": "",
            "text_count": 12,
            "duration_ms": 18000,
            "meta": {"filename": filename, "index": len(self.manager.tasks) + 1},
        }
        task.update(overrides)
        self.manager.tasks[task_id] = task
        return task

    def _create_result(self, name, content=b"png-result"):
        path = self.root / name
        path.write_bytes(content)
        return name

    def _download_zip(self, task_ids):
        response = self.client.post("/api/translate/batch/zip", json={"task_ids": task_ids})
        self.assertEqual(response.status_code, 200, response.text)
        return zipfile.ZipFile(io.BytesIO(response.content))

    def test_batch_upload_creates_independent_tasks(self):
        response = self.client.post(
            "/api/translate/batch",
            files=self._files("page-01.png", "page-02.jpg"),
            data={"source_lang": "ja", "target_lang": "zh"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["total"], 2)
        self.assertEqual([item["filename"] for item in body["items"]], ["page-01.png", "page-02.jpg"])
        self.assertEqual([item["index"] for item in body["items"]], [1, 2])
        self.assertEqual(len(self.manager.created), 2)
        self.assertEqual(self.manager.created[0][4]["filename"], "page-01.png")

    def test_default_language_keeps_auto_to_chinese_behavior(self):
        response = self.client.post(
            "/api/translate",
            files={"file": ("page.png", b"image", "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.manager.created[-1][0:2], ("auto", "zh"))

    def test_unsupported_format_rejects_entire_batch(self):
        response = self.client.post(
            "/api/translate/batch",
            files=self._files("ok.png", "bad.txt"),
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.manager.created, [])

    def test_oversized_file_rejects_entire_batch(self):
        settings = self._settings(max_upload_mb=1)
        files = [
            ("files[]", ("ok.png", b"ok", "image/png")),
            ("files[]", ("large.png", b"x" * (1024 * 1024 + 1), "image/png")),
        ]
        with patch("app.api.translate.get_settings", return_value=settings):
            response = self.client.post("/api/translate/batch", files=files)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.manager.created, [])

    def test_too_many_files_are_rejected(self):
        response = self.client.post(
            "/api/translate/batch",
            files=self._files(*[f"page-{i}.png" for i in range(101)]),
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.manager.created, [])

    def test_full_chapter_batch_of_80_files_is_accepted(self):
        response = self.client.post(
            "/api/translate/batch",
            files=self._files(*[f"page-{i:02d}.png" for i in range(1, 81)]),
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["total"], 80)
        self.assertEqual(len(self.manager.created), 80)

    def test_total_size_limit_rejects_entire_batch(self):
        settings = self._settings(max_upload_mb=2, batch_max_total_mb=1)
        files = [
            ("files[]", ("one.png", b"x" * 600000, "image/png")),
            ("files[]", ("two.png", b"y" * 600000, "image/png")),
        ]
        with patch("app.api.translate.get_settings", return_value=settings):
            response = self.client.post("/api/translate/batch", files=files)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.manager.created, [])

    def test_batch_status_calculates_average_progress(self):
        self._set_task("done", "page-01.png")
        self._set_task("work", "page-02.png", status="processing", progress=30)
        response = self.client.post(
            "/api/translate/batch/status", json={"task_ids": ["done", "work"]}
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["progress"], 65)
        self.assertEqual((body["completed"], body["processing"], body["failed"]), (1, 1, 0))

    def test_failed_child_does_not_hide_successful_child(self):
        self._set_task("done", "page-01.png")
        self._set_task("bad", "page-02.png", status="failed", progress=40, error="识别失败")
        response = self.client.post(
            "/api/translate/batch/status", json={"task_ids": ["done", "bad"]}
        )
        body = response.json()
        self.assertEqual((body["completed"], body["failed"]), (1, 1))
        self.assertEqual([item["status"] for item in body["items"]], ["completed", "failed"])

    def test_zip_contains_successful_image(self):
        result = self._create_result("result.png")
        self._set_task("done", "page-01.png", result_path=result)
        with self._download_zip(["done"]) as bundle:
            names = bundle.namelist()
            self.assertIn("translated_images/01_page-01_translated.png", names)

    def test_zip_manifest_contains_task_metadata(self):
        result = self._create_result("result.png")
        self._set_task("done", "page-01.png", result_path=result)
        with self._download_zip(["done"]) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
        self.assertEqual(manifest[0]["original_filename"], "page-01.png")
        self.assertEqual(manifest[0]["task_id"], "done")
        self.assertEqual(manifest[0]["text_count"], 12)
        self.assertEqual(manifest[0]["duration_ms"], 18000)
        self.assertEqual(manifest[0]["result_path"], "translated_images/01_page-01_translated.png")
        self.assertEqual(manifest[0]["error"], "")

    def test_partial_failure_generates_errors_file(self):
        result = self._create_result("result.png")
        self._set_task("done", "page-01.png", result_path=result)
        self._set_task("bad", "page-02.png", status="failed", error="翻译服务不可用")
        with self._download_zip(["done", "bad"]) as bundle:
            errors = bundle.read("errors.txt").decode("utf-8")
            self.assertIn("page-02.png", errors)
            self.assertIn("翻译服务不可用", errors)

    def test_duplicate_names_do_not_overwrite(self):
        first = self._create_result("first.png", b"first")
        second = self._create_result("second.png", b"second")
        self._set_task("one", "page.png", result_path=first)
        self._set_task("two", "page.png", result_path=second)
        with self._download_zip(["one", "two"]) as bundle:
            image_names = [name for name in bundle.namelist() if name.startswith("translated_images/")]
            self.assertEqual(len(image_names), 2)
            self.assertEqual(len(set(image_names)), 2)
            self.assertIn("translated_images/02_page_2_translated.png", image_names)

    def test_path_traversal_filename_is_sanitized(self):
        result = self._create_result("result.png")
        self._set_task("done", "../../secret.png", result_path=result)
        with self._download_zip(["done"]) as bundle:
            image_names = [name for name in bundle.namelist() if name.startswith("translated_images/")]
        self.assertEqual(image_names, ["translated_images/01_secret_translated.png"])
        self.assertTrue(all(".." not in name for name in image_names))

    def test_single_image_endpoint_still_creates_task(self):
        response = self.client.post(
            "/api/translate?source_lang=en&target_lang=zh",
            files={"file": ("single.png", b"image", "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "queued")
        self.assertEqual(self.manager.created[0][3], "single.png")

    def test_single_result_download_uses_safe_original_filename(self):
        result = self._create_result("result.png")
        self._set_task("done", "../../chapter-01.png", result_path=result)
        response = self.client.get("/api/translate/done/result")
        self.assertEqual(response.status_code, 200, response.text)
        disposition = response.headers["content-disposition"]
        self.assertIn("chapter-01_translated.png", disposition)
        self.assertNotIn("..", disposition)

    def test_failed_single_task_can_be_retried_from_stored_original(self):
        original = self._create_result("original.png", b"original-image")
        self._set_task(
            "bad", "page-01.png", status="failed", error="识别失败",
            original_path=original, source_lang="ja", target_lang="zh",
            meta={"filename": "page-01.png", "index": 2, "batch_id": "batch-one"},
        )
        response = self.client.post("/api/translate/bad/retry")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.manager.created[-1][0:4], ("ja", "zh", b"original-image", "page-01.png"))
        self.assertEqual(self.manager.created[-1][4]["retry_of"], "bad")
        self.assertEqual(self.manager.created[-1][4]["index"], 2)

    def test_completed_task_cannot_be_retried(self):
        self._set_task("done", "page.png", status="completed")
        response = self.client.post("/api/translate/done/retry")
        self.assertEqual(response.status_code, 409)


class TestTaskPipelineIsolation(unittest.TestCase):
    """共享模型流水线不能被多个批量子任务同时调用。"""

    def test_default_worker_count_is_serial(self):
        import inspect

        from app.services.task_manager import TranslationTaskManager

        parameter = inspect.signature(TranslationTaskManager).parameters["max_workers"]
        self.assertEqual(parameter.default, 1)

    def test_pipeline_lock_prevents_concurrent_inference(self):
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace

        from app.services.task_manager import TranslationTaskManager

        class FakeDatabase:
            def task_update(self, *args, **kwargs):
                return None

        class FakePipeline:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def translate_image(self, *args, **kwargs):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.03)
                with self.lock:
                    self.active -= 1
                return SimpleNamespace(regions=[], duration_ms=30)

        manager = TranslationTaskManager.__new__(TranslationTaskManager)
        manager.db = FakeDatabase()
        manager._pipeline_lock = threading.Lock()
        pipeline = FakePipeline()
        manager._get_pipeline = lambda: pipeline

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(manager._run, f"task-{index}", "ja", "zh", "unused.png")
                for index in range(2)
            ]
            for future in futures:
                future.result()

        self.assertEqual(pipeline.max_active, 1)


class TestMitEngines(unittest.TestCase):
    """MIT 引擎冒烟测试：权重缺失时跳过（默认 backend/data/models/mit）"""

    @classmethod
    def setUpClass(cls):
        cls.model_dir = _MIT_MODEL_DIR
        if cls.model_dir.exists():
            os.environ["MANGA_MIT_MODEL_DIR"] = str(cls.model_dir)

    def _default_ready(self):
        from app.services.engines.mit import paths

        return paths.model_ready("detection/detect-20241225.ckpt")

    @unittest.skipUnless(
        (_MIT_MODEL_DIR / "detection" / "detect-20241225.ckpt").exists(),
        "detect-20241225.ckpt 缺失，跳过",
    )
    def test_default_detector_loads_and_detects(self):
        from app.services.engines.mit.dbnet import DefaultDetector

        det = DefaultDetector(device="cpu")
        img = np.array(Image.open(self.image).convert("RGB"))
        textlines, raw_mask = det.detect(img, detect_size=768)
        self.assertIsInstance(textlines, list)
        self.assertEqual(raw_mask.ndim, 2)
        self.assertLessEqual(raw_mask.max(), 255)

    @unittest.skipUnless(
        (_MIT_MODEL_DIR / "ocr" / "ocr_ar_48px.ckpt").exists(),
        "ocr_ar_48px.ckpt 缺失，跳过",
    )
    def test_mit48_ocr(self):
        from app.services.engines.mit.ocr_48px import Mit48Ocr
        from app.services.engines.mit.quadrilateral import Quadrilateral

        ocr = Mit48Ocr(device="cpu")
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:] = (255, 255, 255)
        quads = [Quadrilateral(np.array([[50, 50], [150, 50], [150, 66], [50, 66]]), "", 0.9)]
        ocr.recognize(img, quads, prob_threshold=0.2)
        self.assertIsInstance(quads[0].text, str)

    @unittest.skipUnless(
        (_MIT_MODEL_DIR / "ocr" / "ocr_ar_48px.ckpt").exists(),
        "ocr_ar_48px.ckpt 缺失，跳过",
    )
    def test_mit48_large_batch_no_crash(self):
        """大批次束搜索越界回归：finished_batch 提前命中且 best>=beams_k 时不得 IndexError。

        之前用 out_idx[idx*beams_k+best]（每批仅 beams_k 行）索引 25 候选，越界崩 →
        MixedOCR 吞掉异常、整页只得 0 文字（实测一页 36 行 → IndexError: index 90
        is out of bounds for dimension 0 with size 80）。回归用一张多行密集页 fixture，
        断言识别不崩且至少识别出若干行。
        """
        from app.services.engines.mit.ocr_48px import Mit48Ocr
        from app.services.engines.mit.quadrilateral import Quadrilateral

        fixture = Path(__file__).resolve().parent / "fixtures" / "desert_many_lines.jpg"
        if not fixture.exists():
            self.skipTest("desert_many_lines.jpg 未随测试分发，跳过")
        ocr = Mit48Ocr(device="cpu")
        img = np.array(Image.open(fixture).convert("RGB"))
        h, w = img.shape[:2]
        # 生成一批竖排文本行 quads（覆盖整页，乱序行长短不一，能触发 finished 提前命中）
        quads = []
        cols = 28
        for i in range(cols):
            x0 = int(w * (0.03 + 0.92 * i / cols))
            x1 = x0 + int(w * 0.022)
            y1 = int(h * (0.03 + 0.94 * (i % 3) / 3))
            pts = np.array([[x0, 20], [x1, 20], [x1, y1], [x0, y1]], dtype=np.float64)
            quads.append(Quadrilateral(pts, "", 0.9))
        ocr.recognize(img, quads, prob_threshold=0.2)
        filled = sum(1 for q in quads if (q.text or "").strip())
        self.assertGreater(filled, 0)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.image = make_test_image(Path(self.tmp) / "test.png")


class TestBubbleGrouping(unittest.TestCase):
    """气泡分组：同一气泡的 region 合成一组，并按阅读顺序排列"""

    def _make_scene(self):
        img = Image.new("RGB", (400, 260), "white")
        d = ImageDraw.Draw(img)
        # 两个独立气泡
        d.ellipse([20, 20, 190, 130], outline="black", width=3)
        d.ellipse([230, 20, 390, 130], outline="black", width=3)
        d.rectangle([50, 40, 170, 55], fill="black")
        d.rectangle([50, 75, 170, 90], fill="black")
        d.rectangle([260, 40, 370, 55], fill="black")
        d.rectangle([260, 75, 370, 90], fill="black")
        return img

    def test_two_bubbles_grouped(self):
        from app.services.engines.bubble import group_regions_by_bubble
        from app.services.pipeline import TextRegion

        img = self._make_scene()
        arr = np.array(img)
        bgr = arr[:, :, ::-1].copy()
        regions = [
            TextRegion(box=[[50, 40], [170, 40], [170, 55], [50, 55]]),
            TextRegion(box=[[50, 75], [170, 75], [170, 90], [50, 90]]),
            TextRegion(box=[[260, 40], [370, 40], [370, 55], [260, 55]]),
            TextRegion(box=[[260, 75], [370, 75], [370, 90], [260, 90]]),
        ]
        groups = group_regions_by_bubble(bgr, regions, 400, 260)
        self.assertEqual(len(groups), 2)
        for g in groups:
            self.assertEqual(len(g["regions"]), 2)
            for r in g["regions"]:
                self.assertIsNotNone(r.group_index)
                self.assertIsNotNone(r.group_bounds)
        self.assertEqual({g["regions"][0].bounds[0] for g in groups}, {50, 260})


class TestBubbleBlockRender(unittest.TestCase):
    """整块气泡译文排版：横排块/竖排块渲染不报错且保留尺寸"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        cleaned = Path(self.tmp) / "cleaned.png"
        img = Image.new("RGB", (300, 300), "white")
        ImageDraw.Draw(img).rectangle([20, 20, 280, 150], outline="black", width=2)
        img.save(cleaned)
        self.cleaned = cleaned

    def _render(self, regions):
        import io

        from app.services.engines import get_engine

        renderer = get_engine("renderer")
        out = renderer.render(self.cleaned, regions, target_lang="zh")
        parsed = Image.open(io.BytesIO(out))
        self.assertEqual(parsed.size, (300, 300))
        return out

    def test_horizontal_block(self):
        from app.services.pipeline import TextRegion

        r = TextRegion(box=[[40, 40], [260, 40], [260, 70], [40, 70]])
        r.group_index = 0
        r.group_bounds = (20, 20, 280, 150)
        r.group_translated = "第一行\n第二行"
        r.direction = "h"
        self._render([r])

    def test_vertical_block(self):
        from app.services.pipeline import TextRegion

        r = TextRegion(box=[[100, 30], [140, 30], [140, 260], [100, 260]])
        r.group_index = 0
        r.group_bounds = (60, 20, 240, 280)
        r.group_translated = "这是一段比较长的中文译文需要进行竖排分列重排"
        r.direction = "v"
        self._render([r])

    def _render_to_array(self, regions):
        import io

        from app.services.engines import get_engine

        renderer = get_engine("renderer")
        out = renderer.render(self.cleaned, regions, target_lang="zh")
        parsed = Image.open(io.BytesIO(out))
        return np.array(parsed.convert("L"))

    def test_text_never_leaves_bubble(self):
        """掩膜裁剪：气泡掩膜之外的区域应与原图完全一致（文字只画进气泡）"""
        from app.services.engines.bubble import bubble_with_mask
        from app.services.pipeline import TextRegion

        r = TextRegion(box=[[60, 60], [240, 60], [240, 100], [60, 100]])
        r.group_index = 0
        r.group_bounds = (20, 30, 280, 130)
        # 很长的文本，强制营造溢出场景
        r.group_translated = "这句话特别特别长用来测试文字无论如何都不会画出气泡的范围外"
        r.direction = "h"
        gray = self._render_to_array([r])

        cleaned_gray = np.array(Image.open(self.cleaned).convert("L"))
        bgr_img = np.array(Image.open(self.cleaned).convert("RGB"))[:, :, ::-1].copy()
        bb, mask = bubble_with_mask(bgr_img, (60, 60, 240, 100), gray.shape[1], gray.shape[0])
        self.assertIsNotNone(mask)
        inside = mask > 0
        # 掩膜外的像素任何变化都不允许（描边等原图内容保持原样）
        changed = (gray != cleaned_gray) & ~inside
        self.assertFalse(changed.any())

    def test_short_text_single_line(self):
        """宽泡 + 短文本：应单行排版（文本宽高比 > 1.3，而非折成两行）"""
        from app.services.pipeline import TextRegion

        r = TextRegion(box=[[60, 60], [240, 60], [240, 100], [60, 100]])
        r.group_index = 0
        r.group_bounds = (20, 40, 280, 120)
        r.group_translated = "你好世界"
        r.direction = "h"
        gray = self._render_to_array([r])
        # 只取气泡内文本像素
        patch = gray[40:120, 20:280]
        ys, xs = np.where(patch < 200)
        self.assertGreater(xs.size, 0)
        th = ys.max() - ys.min() + 1
        tw = xs.max() - xs.min() + 1
        self.assertGreater(tw / max(1, th), 1.3)


def _import_ok() -> bool:
    import importlib.util

    return importlib.util.find_spec("manga_ocr") is not None


class TestMangaOcrEngine(unittest.TestCase):
    """manga-ocr 引擎冒烟：未安装时跳过"""

    @unittest.skipUnless(_import_ok(), "manga-ocr 未安装，跳过")
    def test_engine_importable(self):
        from app.services.engines.ocr import MangaOCREngine

        self.assertEqual(MangaOCREngine.name, "mangaocr")


class TestGlossary(unittest.TestCase):
    def setUp(self):
        self.service = GlossaryService()

    def test_builtin_seeded(self):
        items = self.service.list_items()
        self.assertGreater(len(items), 0)

    def test_english_builtin_name_mapping(self):
        self.assertEqual(self.service.get_mapping("en")["Ken Takakura"], "高仓健")

    def test_japanese_builtin_name_mapping(self):
        self.assertEqual(self.service.get_mapping("ja")["コテ川"], "古手川")

    def test_japanese_manga_term_mapping(self):
        mapping = self.service.get_mapping("ja")
        self.assertEqual(mapping["ゴッドバレー"], "神之谷")
        self.assertEqual(mapping["うちに寄ってきなよ"], "来我家坐坐吧")

    def test_glossary_distinguishes_target_language(self):
        japanese_id, _ = self.service.create("主角", "主人公", "zh", "", "ja")
        english_id, _ = self.service.create("主角", "protagonist", "zh", "", "en")
        self.assertEqual(self.service.get_mapping("zh", "ja")["主角"], "主人公")
        self.assertEqual(self.service.get_mapping("zh", "en")["主角"], "protagonist")
        self.service.delete(japanese_id)
        self.service.delete(english_id)

    def test_crud(self):
        item_id, msg = self.service.create("テスト", "测试", "ja", "单元测试")
        self.assertGreater(item_id, 0, msg)
        ok, _ = self.service.update(item_id, "テスト", "测试2", "ja", "")
        self.assertTrue(ok)
        self.assertTrue(self.service.delete(item_id))

    def test_duplicate_rejected(self):
        # 相同 source+target+lang 视为重复
        item_id, _ = self.service.create("重複テスト", "重复测试", "ja", "")
        self.assertGreater(item_id, 0)
        item_id2, _ = self.service.create("重複テスト", "重复测试", "ja", "")
        self.assertEqual(item_id2, -1)
        # 清理
        if item_id > 0:
            self.service.delete(item_id)

    def test_import_json(self):
        content = '[{"source":"A","target":"B","lang":"en"},{"source":"C","target":"D","lang":"ja"}]'
        result = self.service.import_json(content)
        self.assertEqual(result["imported"], 2)

    def test_apply_glossary_cjk_suffix(self):
        from app.services.engines.translator import SmartTranslator

        out = SmartTranslator._apply_glossary("坂本くん", {"くん": "君"})
        self.assertEqual(out, "坂本君")

    def test_apply_glossary_cjk_name(self):
        from app.services.engines.translator import SmartTranslator

        out = SmartTranslator._apply_glossary("めいは坂本のこと", {"めい": "芽衣"})
        self.assertEqual(out, "芽衣は坂本のこと")


class TestMocrConfidence(unittest.TestCase):
    def test_mangaocr_sets_confidence_when_text_found(self):
        from app.services.engines.mit.mocr import MangaOcrWrapper
        from app.services.engines.mit.quadrilateral import Quadrilateral

        class _Fake:
            def __call__(self, _img):
                return "テスト"

        w = MangaOcrWrapper.__new__(MangaOcrWrapper)
        w.text_height = 48
        w._mocr = _Fake()
        img = np.ones((80, 40, 3), dtype=np.uint8) * 255
        q = Quadrilateral(np.array([[5, 5], [35, 5], [35, 75], [5, 75]], dtype=float), "", 0.3)
        w.recognize(img, [q])
        self.assertEqual(q.text, "テスト")
        self.assertGreaterEqual(q.prob, 0.85)


class TestMultilingualOcrRouting(unittest.TestCase):
    """三语 OCR 路由和候选比较的无模型回归。"""

    def _region(self):
        from app.services.pipeline import TextRegion

        return TextRegion(box=[[20, 20], [180, 20], [180, 60], [20, 60]])

    def test_language_hints_distinguish_chinese_japanese_and_english(self):
        from app.services.language import detect_language, region_language_hint

        self.assertEqual(detect_language("这是中文对白").language, "zh")
        self.assertEqual(detect_language("これは日本語です").language, "ja")
        self.assertEqual(detect_language("Don't re-enter!").language, "en")
        self.assertEqual(region_language_hint("纯汉字短句").language, "zh")

    def test_explicit_english_uses_paddle_route_only(self):
        from app.services.engines.ocr import LanguageRoutingOCREngine

        class FakePaddle:
            def recognize_regions(self, _path, regions, source_lang):
                regions[0].text = "Don't re-enter!"
                regions[0].confidence = 0.91
                regions[0].source_lang = source_lang
                regions[0].ocr_engine = "paddle"

        router = LanguageRoutingOCREngine.__new__(LanguageRoutingOCREngine)
        router._get_paddle = lambda: FakePaddle()
        router._get_ja = lambda: (_ for _ in ()).throw(AssertionError("英文不应访问日文 OCR"))
        region = self._region()
        router.recognize(Path("unused.png"), [region], "en")
        self.assertEqual(region.source_lang, "en")
        self.assertEqual(region.ocr_engine, "paddle")
        self.assertEqual(region.text, "Don't re-enter!")

    def test_candidate_score_prefers_text_matching_language(self):
        from app.services.engines.ocr import LanguageRoutingOCREngine
        from app.services.language import region_language_hint

        candidates = [
            {"engine": "paddle", "lang": "en", "text": "Hello world", "confidence": 0.72},
            {"engine": "paddle", "lang": "zh", "text": "你好世界", "confidence": 0.70},
        ]
        chosen = LanguageRoutingOCREngine._choose_candidate(candidates, region_language_hint("Hello world"))
        self.assertEqual(chosen["lang"], "en")

    def test_japanese_page_context_resolves_pure_han_region(self):
        from app.services.engines.ocr import LanguageRoutingOCREngine
        from app.services.language import region_language_hint

        candidates = [
            {"engine": "paddle", "lang": "zh", "text": "紀委員♡", "confidence": 0.7461},
            {"engine": "paddle", "lang": "ja", "text": "紀委員♡", "confidence": 0.8310},
        ]
        chosen = LanguageRoutingOCREngine._choose_candidate(
            candidates,
            region_language_hint("2委員♡", "ja"),
            "ja",
        )
        self.assertEqual(chosen["lang"], "ja")

    def test_japanese_page_context_resolves_single_kana_cjk_region(self):
        from app.services.engines.ocr import LanguageRoutingOCREngine
        from app.services.language import region_language_hint

        candidates = [
            {"engine": "paddle", "lang": "zh", "text": "今週を“風紀強化週間”", "confidence": 0.9414},
            {"engine": "paddle", "lang": "ja", "text": "今週を“風紀強化週間”", "confidence": 0.9579},
        ]
        chosen = LanguageRoutingOCREngine._choose_candidate(
            candidates,
            region_language_hint("今週を“風紀強化週間”", "ja"),
            "ja",
        )
        self.assertEqual(chosen["lang"], "ja")

    def test_japanese_page_prefers_kana_candidate_even_from_chinese_ocr(self):
        from app.services.engines.ocr import LanguageRoutingOCREngine
        from app.services.language import region_language_hint

        candidates = [
            {"engine": "paddle", "lang": "zh", "text": "“神” か み", "confidence": 0.9515},
            {"engine": "paddle", "lang": "ja", "text": "神 办 反", "confidence": 0.9996},
        ]
        chosen = LanguageRoutingOCREngine._choose_candidate(
            candidates,
            region_language_hint("ﾞ神ﾞ", "ja"),
            "ja",
        )
        self.assertEqual(chosen["text"], "“神” か み")
        self.assertEqual(chosen["lang"], "ja")

    def test_japanese_page_penalizes_short_latin_gibberish(self):
        from app.services.engines.ocr import LanguageRoutingOCREngine

        candidates = [
            {"engine": "paddle", "lang": "en", "text": "IV IU 1", "confidence": 0.84},
            {"engine": "paddle", "lang": "ja", "text": "ふん…", "confidence": 0.31},
        ]
        chosen = LanguageRoutingOCREngine._choose_candidate(candidates, None, "ja")
        self.assertEqual(chosen["text"], "ふん…")

    def test_japanese_page_preserves_suspicious_fallback_gibberish(self):
        from app.services.pipeline import TextRegion, suspicious_japanese_ocr

        for text in ("IV IU 1", "u ル", '7" π'):
            region = TextRegion(
                box=[[0, 0], [20, 0], [20, 60], [0, 60]],
                text=text,
                ocr_fallback=True,
            )
            self.assertTrue(suspicious_japanese_ocr(region, "ja"), text)

        dialogue = TextRegion(
            box=[[0, 0], [20, 0], [20, 60], [0, 60]],
            text="これで終わり",
            ocr_fallback=True,
        )
        self.assertFalse(suspicious_japanese_ocr(dialogue, "ja"))

    def test_vertical_context_region_expands_short_column_upstream(self):
        from app.services.engines.ocr import LanguageRoutingOCREngine
        from app.services.pipeline import TextRegion

        region = TextRegion(
            box=[[19, 139], [57, 139], [57, 211], [19, 211]], direction="v"
        )
        expanded = LanguageRoutingOCREngine._vertical_context_region(region)
        self.assertLess(expanded.bounds[0], region.bounds[0])
        self.assertLess(expanded.bounds[1], region.bounds[1])
        self.assertGreater(expanded.bounds[3], region.bounds[3])

    def test_english_preprocess_keeps_upscale_and_binary_candidates(self):
        from app.services.engines.ocr import PaddleOCREngine

        engine = PaddleOCREngine.__new__(PaddleOCREngine)
        engine.settings = SimpleNamespace(ocr_en_upscale=2.0, ocr_zh_upscale=1.5)
        variants = engine._preprocess_variants(np.ones((12, 50, 3), dtype=np.uint8) * 230, "en")
        self.assertEqual([name for name, _ in variants], ["original", "gray-contrast", "adaptive-binary"])
        self.assertGreater(variants[0][1].shape[0], 12)

    def test_cross_balloon_english_line_finds_wide_internal_gutter(self):
        from app.services.engines.ocr import _horizontal_ink_split_boxes
        from app.services.pipeline import TextRegion

        image = np.ones((60, 260, 3), dtype=np.uint8) * 255
        image[15:32, 20:72] = 0
        image[15:32, 82:122] = 0
        image[15:32, 164:218] = 0
        region = TextRegion(
            box=[[10, 10], [230, 10], [230, 38], [10, 38]],
            text="HAVE ANY NERD CRAP!",
            direction="h",
            source_lang="en",
        )
        boxes = _horizontal_ink_split_boxes(image, region)
        self.assertEqual(len(boxes), 2)
        self.assertLess(boxes[0][2], boxes[1][0])

    def test_bridge_split_rejects_single_word_side(self):
        from app.services.engines.ocr import LanguageRoutingOCREngine
        from app.services.pipeline import TextRegion

        image = Path(tempfile.gettempdir()) / "manga_bridge_single_word.png"
        pixels = np.ones((60, 260, 3), dtype=np.uint8) * 255
        pixels[15:32, 20:72] = 0
        pixels[15:32, 82:122] = 0
        pixels[15:32, 164:218] = 0
        Image.fromarray(pixels).save(image)
        original = TextRegion(
            box=[[10, 10], [230, 10], [230, 38], [10, 38]],
            text="I'M NOT IN",
            confidence=1.0,
            direction="h",
            source_lang="en",
        )

        class FakePaddle:
            def recognize_regions(self, _path, regions, _lang):
                for region, text in zip(regions, ["I'M", "NOT IN"]):
                    region.text = text
                    region.confidence = 1.0
                    region.ocr_engine = "paddle"

        router = LanguageRoutingOCREngine.__new__(LanguageRoutingOCREngine)
        regions = [original]
        router._split_english_bridge_regions(image, regions, FakePaddle())

        self.assertEqual(regions, [original])

    def test_language_thresholds_are_separate(self):
        from app.services.pipeline import ocr_thresholds

        self.assertEqual(ocr_thresholds("mit48", "ja"), (0.2, 0.0))
        self.assertEqual(ocr_thresholds("paddle", "zh"), (0.55, 0.25))
        self.assertEqual(ocr_thresholds("paddle", "en"), (0.55, 0.25))

    def test_auto_mixed_page_translates_each_bubble_with_its_language(self):
        from app.services.pipeline import TextRegion, TranslationPipeline

        class Detector:
            name = "fake"

            def detect(self, _path):
                return [
                    TextRegion(box=[[20, 20], [90, 20], [90, 60], [20, 60]]),
                    TextRegion(box=[[110, 20], [190, 20], [190, 60], [110, 60]]),
                ]

        class OCR:
            name = "language-router"
            supports_detection = False
            supports_language_routing = True

            def recognize(self, _path, regions, source_lang):
                self.requested = source_lang
                regions[0].text, regions[0].confidence = "你好", 0.9
                regions[0].source_lang, regions[0].ocr_engine = "zh", "paddle"
                regions[1].text, regions[1].confidence = "Hello!", 0.9
                regions[1].source_lang, regions[1].ocr_engine = "en", "paddle"

        class Translator:
            name = "fake-translator"

            def __init__(self):
                self.calls = []
                self.last_failures = []

            def translate_batch(self, texts, source_lang, _target_lang, glossary=None, progress_cb=None):
                self.calls.append((source_lang, list(texts)))
                self.last_backend_names = ["fake"] * len(texts)
                if progress_cb:
                    progress_cb(1.0)
                return [f"译:{text}" for text in texts]

        class Inpainter:
            def inpaint(self, path, _regions):
                return path

        class Renderer:
            def render(self, _path, regions, target_lang):
                for region in regions:
                    region.render_font = "fake.ttf"
                return b"png"

        image = Path(tempfile.gettempdir()) / "manga_auto_mixed.png"
        Image.new("RGB", (220, 100), "white").save(image)
        translator = Translator()
        pipeline = TranslationPipeline(
            detector=Detector(), ocr=OCR(), translator=translator,
            inpainter=Inpainter(), renderer=Renderer(),
        )
        pipeline._bubble_on = False
        pipeline._group_regions = lambda _path, regions, boundary_image_path=None: [
            {"regions": [region], "bbox": region.bounds} for region in regions
        ]
        with patch("app.services.engines.bubble.classify_non_bubble", return_value=False):
            result = pipeline.translate_image(image, "auto", "ja")
        self.assertEqual(pipeline.ocr.requested, "auto")
        self.assertEqual({source for source, _ in translator.calls}, {"zh", "en"})
        self.assertEqual({item["source_lang"] for item in result.region_diagnostics}, {"zh", "en"})
        self.assertTrue(all("bounds" in item and "group_bounds" in item for item in result.region_diagnostics))


class TestPerformanceOptimizations(unittest.TestCase):
    def _paddle_engine(self):
        from app.services.engines.ocr import PaddleOCREngine

        engine = PaddleOCREngine.__new__(PaddleOCREngine)
        engine.settings = SimpleNamespace(
            ocr_en_upscale=2.0,
            ocr_zh_upscale=1.5,
            ocr_candidate_fallback_threshold=0.45,
            paddle_device="cpu",
        )
        engine._ocrs = {}
        engine._device_by_lang = {}
        engine._device_fallback_reasons = []
        engine._requested_device = "cpu"
        engine._preferred_device = "cpu"
        engine._lang_map = {"ja": "japan", "en": "en", "zh": "ch"}
        engine._PaddleOCR = object()
        engine._cached_image_path = None
        engine._cached_image = None
        engine.reset_performance()
        return engine

    def test_paddle_models_are_lazy_and_each_language_loads_once(self):
        import types

        from app.services.engines.ocr import PaddleOCREngine

        created = []

        class FakePaddle:
            def __init__(self, **kwargs):
                created.append(kwargs)

        with (
            patch.object(PaddleOCREngine, "_paddle_cuda_status", return_value=(False, 0)),
            patch.dict(sys.modules, {"paddleocr": types.SimpleNamespace(PaddleOCR=FakePaddle)}),
        ):
            engine = PaddleOCREngine()
            self.assertEqual(created, [])
            first = engine._get_ocr("en")
            second = engine._get_ocr("en")

        self.assertIs(first, second)
        self.assertEqual([item["lang"] for item in created], ["en"])
        self.assertNotIn("japan", engine._ocrs)
        self.assertEqual(engine.last_performance["model_reuse_count"], 1)

    def test_paddle_gpu_device_is_passed_with_required_3x_options(self):
        import types

        from app.services.engines.ocr import PaddleOCREngine

        created = []

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                created.append(kwargs)

        settings = SimpleNamespace(paddle_device="gpu:1")
        with (
            patch("app.services.engines.ocr.get_settings", return_value=settings),
            patch.object(PaddleOCREngine, "_paddle_cuda_status", return_value=(True, 2)),
            patch.dict(sys.modules, {"paddleocr": types.SimpleNamespace(PaddleOCR=FakePaddleOCR)}),
        ):
            engine = PaddleOCREngine()
            engine._get_ocr("en")

        self.assertEqual(created[0]["device"], "gpu:1")
        self.assertFalse(created[0]["enable_mkldnn"])
        self.assertFalse(created[0]["use_doc_orientation_classify"])
        self.assertFalse(created[0]["use_doc_unwarping"])
        self.assertTrue(created[0]["use_textline_orientation"])
        self.assertNotIn("use_gpu", created[0])
        self.assertEqual(engine.last_performance["device"], "gpu:1")

    def test_paddle_cpu_device_is_supported(self):
        import types

        from app.services.engines.ocr import PaddleOCREngine

        created = []

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                created.append(kwargs)

        settings = SimpleNamespace(paddle_device="cpu")
        with (
            patch("app.services.engines.ocr.get_settings", return_value=settings),
            patch.dict(sys.modules, {
                "paddleocr": types.SimpleNamespace(PaddleOCR=FakePaddleOCR),
            }),
        ):
            engine = PaddleOCREngine()
            engine._get_ocr("zh")

        self.assertEqual(created[0]["device"], "cpu")
        self.assertFalse(engine.last_performance["device_fallback"])

    def test_paddle_unavailable_gpu_falls_back_to_cpu(self):
        import types

        from app.services.engines.ocr import PaddleOCREngine

        created = []

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                created.append(kwargs)

        settings = SimpleNamespace(paddle_device="gpu:0")
        with (
            patch("app.services.engines.ocr.get_settings", return_value=settings),
            patch.object(PaddleOCREngine, "_paddle_cuda_status", return_value=(False, 0)),
            patch.dict(sys.modules, {"paddleocr": types.SimpleNamespace(PaddleOCR=FakePaddleOCR)}),
        ):
            engine = PaddleOCREngine()
            first = engine._get_ocr("en")
            second = engine._get_ocr("en")

        self.assertIs(first, second)
        self.assertEqual([item["device"] for item in created], ["cpu"])
        self.assertTrue(engine.last_performance["device_fallback"])
        self.assertIn("未编译 CUDA", engine.last_performance["device_fallback_reason"])
        self.assertEqual(engine.last_performance["model_reuse_count"], 1)

    def test_paddle_gpu_model_load_failure_retries_cpu_once(self):
        import types

        from app.services.engines.ocr import PaddleOCREngine

        created = []

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                created.append(kwargs["device"])
                if kwargs["device"].startswith("gpu"):
                    raise RuntimeError("CUDA out of memory")

        settings = SimpleNamespace(paddle_device="gpu:0")
        with (
            patch("app.services.engines.ocr.get_settings", return_value=settings),
            patch.object(PaddleOCREngine, "_paddle_cuda_status", return_value=(True, 1)),
            patch.dict(sys.modules, {"paddleocr": types.SimpleNamespace(PaddleOCR=FakePaddleOCR)}),
        ):
            engine = PaddleOCREngine()
            first = engine._get_ocr("en")
            second = engine._get_ocr("en")

        self.assertIs(first, second)
        self.assertEqual(created, ["gpu:0", "cpu"])
        self.assertEqual(engine.last_performance["device"], "cpu")
        self.assertTrue(engine.last_performance["device_fallback"])

    def test_paddle_gpu_oom_during_predict_reloads_cpu_and_reuses_it(self):
        engine = self._paddle_engine()

        class FailingGpuOCR:
            def predict(self, _value):
                raise RuntimeError("CUDA out of memory")

        class CpuOCR:
            def __init__(self):
                self.calls = 0

            def predict(self, _value):
                self.calls += 1
                return [{"rec_texts": ["SAFE"], "rec_scores": [0.9]}]

        cpu_ocr = CpuOCR()
        created = []

        def create_cpu(**kwargs):
            created.append(kwargs["device"])
            return cpu_ocr

        gpu_ocr = FailingGpuOCR()
        engine._PaddleOCR = create_cpu
        engine._requested_device = "gpu:0"
        engine._preferred_device = "gpu:0"
        engine._ocrs["en"] = gpu_ocr
        engine._device_by_lang["en"] = "gpu:0"

        first = engine._predict(gpu_ocr, object(), "en", "original")
        second = engine._predict(gpu_ocr, object(), "en", "original")

        self.assertEqual(first, second)
        self.assertEqual(created, ["cpu"])
        self.assertEqual(cpu_ocr.calls, 2)
        self.assertEqual(engine.last_performance["call_count"], 3)
        self.assertEqual(engine.last_performance["device"], "cpu")
        self.assertIn("out of memory", engine.last_performance["device_fallback_reason"])

    def test_pipeline_performance_records_actual_ocr_device(self):
        from app.services.pipeline import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.ocr = SimpleNamespace(last_performance={
            "requested_device": "gpu:0",
            "device": "cpu",
            "device_fallback": True,
            "device_fallback_reason": "当前 PaddlePaddle 未编译 CUDA 支持",
        })
        performance = pipeline._performance_snapshot({}, 10)

        self.assertEqual(performance["ocr_requested_device"], "gpu:0")
        self.assertEqual(performance["ocr_device"], "cpu")
        self.assertTrue(performance["ocr_device_fallback"])
        self.assertIn("CUDA", performance["ocr_device_fallback_reason"])

    def test_high_confidence_original_stops_preprocess_cascade(self):
        from app.services.pipeline import TextRegion

        engine = self._paddle_engine()

        class FakeOCR:
            def __init__(self):
                self.calls = 0

            def predict(self, _value):
                self.calls += 1
                return [{"rec_texts": ["Don't stop!"], "rec_scores": [0.98]}]

        ocr = FakeOCR()
        image = Image.new("RGB", (220, 80), "white")
        region = TextRegion(box=[[20, 20], [200, 20], [200, 60], [20, 60]])
        result = engine._recognize_region_candidates(image, region, ocr, "en")

        self.assertEqual(result["variant"], "original")
        self.assertEqual(ocr.calls, 1)
        self.assertEqual(engine.last_performance["call_count"], 1)

    def test_low_confidence_only_then_runs_contrast_fallback(self):
        from app.services.pipeline import TextRegion

        engine = self._paddle_engine()

        class FakeOCR:
            def __init__(self):
                self.scores = iter([0.20, 0.92])
                self.calls = 0

            def predict(self, _value):
                self.calls += 1
                return [{"rec_texts": ["HELLO WORLD"], "rec_scores": [next(self.scores)]}]

        ocr = FakeOCR()
        image = Image.new("RGB", (220, 80), "white")
        region = TextRegion(box=[[20, 20], [200, 20], [200, 60], [20, 60]])
        result = engine._recognize_region_candidates(image, region, ocr, "en")

        self.assertEqual(result["variant"], "gray-contrast")
        self.assertEqual(ocr.calls, 2)
        self.assertEqual(engine.last_performance["fallback_count"], 1)

    def test_region_recognition_preserves_order_and_coordinates(self):
        from app.services.pipeline import TextRegion

        engine = self._paddle_engine()
        regions = [
            TextRegion(box=[[10, 10], [110, 10], [110, 50], [10, 50]]),
            TextRegion(box=[[130, 10], [230, 10], [230, 50], [130, 50]]),
        ]
        original_bounds = [region.bounds for region in regions]
        original_ids = [id(region) for region in regions]
        engine._get_ocr = lambda _lang: object()
        engine._get_image = lambda _path: Image.new("RGB", (250, 70), "white")
        values = iter(["FIRST LINE", "SECOND LINE"])
        engine._recognize_region_candidates = lambda *_args: {
            "text": (text := next(values)),
            "score": 0.91,
            "variant": "original",
            "candidates": [{
                "engine": "paddle",
                "lang": "en",
                "variant": "original",
                "text": text,
                "confidence": 0.91,
            }],
        }
        engine.recognize_regions(Path("unused.png"), regions, "en")

        self.assertEqual([id(region) for region in regions], original_ids)
        self.assertEqual([region.bounds for region in regions], original_bounds)
        self.assertEqual([region.text for region in regions], ["FIRST LINE", "SECOND LINE"])

    def test_auto_obvious_english_does_not_try_all_paddle_languages(self):
        from app.services.engines.ocr import LanguageRoutingOCREngine
        from app.services.pipeline import TextRegion

        class FakeMit:
            available = True

            def recognize(self, _path, regions, _lang):
                regions[0].text = "HELLO WORLD"
                regions[0].confidence = 0.8

        class FakePaddle:
            available = True
            last_performance = {}

            def __init__(self):
                self.langs = []

            def reset_performance(self):
                return None

            def _get_image(self, _path):
                return Image.new("RGB", (200, 80), "white")

            def _get_ocr(self, lang):
                self.langs.append(lang)
                return object()

            def _recognize_region_candidates(self, _image, _region, _ocr, lang):
                return {
                    "text": "HELLO WORLD",
                    "score": 0.95,
                    "variant": "original",
                    "candidates": [{"engine": "paddle", "lang": lang, "variant": "original", "text": "HELLO WORLD", "confidence": 0.95}],
                }

            def _candidate_sufficient(self, _text, score, _lang):
                return score >= 0.45

        router = LanguageRoutingOCREngine.__new__(LanguageRoutingOCREngine)
        router.settings = SimpleNamespace(auto_source_fallback="ja", ocr_candidate_fallback_threshold=0.45)
        router._mit48 = FakeMit()
        router._ja = None
        router._manga = None
        router._load_errors = []
        paddle = FakePaddle()
        router._paddle = paddle
        region = TextRegion(box=[[10, 10], [180, 10], [180, 50], [10, 50]])
        router.recognize(Path("unused.png"), [region], "auto")

        self.assertEqual(paddle.langs, ["en"])
        self.assertEqual(region.source_lang, "en")

    def test_translation_cache_keeps_language_directions_separate(self):
        from app.services.engines.translator import SmartTranslator

        class FakeBackend:
            name = "fake"
            batch = False
            prompt_glossary = False

            def __init__(self):
                self.calls = 0
                self._request_count = 0

            def translate_one(self, text, source, target):
                self.calls += 1
                self._request_count += 1
                return f"{source}>{target}:{text}"

        backend = FakeBackend()
        smart = SmartTranslator.__new__(SmartTranslator)
        smart.settings = SimpleNamespace(translation_cache_size=32)
        smart._backends = [backend]
        smart._available_by_direction = {("en", "zh"): backend, ("ja", "zh"): backend}
        smart._translation_cache = __import__("collections").OrderedDict()
        smart._last_backend_name = ""

        first = smart.translate_batch(["same"], "en", "zh")
        second = smart.translate_batch(["same"], "en", "zh")
        japanese = smart.translate_batch(["same"], "ja", "zh")

        self.assertEqual(first, second)
        self.assertNotEqual(first, japanese)
        self.assertEqual(backend.calls, 2)
        self.assertEqual(smart.last_performance["request_count"], 1)

    def test_partial_batch_failure_retries_only_failed_item(self):
        from app.services.engines.translator import SmartTranslator

        class FakeBatchBackend:
            name = "fake-batch"
            batch = True
            prompt_glossary = False

            def __init__(self):
                self.single_calls = []
                self._request_count = 0

            def translate_batch(self, texts, *_args, **_kwargs):
                self._request_count += 1
                self.last_failed_indices = [1]
                self.last_batch_failures = ["第二项失败"]
                return ["译一", texts[1], "译三"]

            def translate_one(self, text, *_args):
                self._request_count += 1
                self.single_calls.append(text)
                return "译二"

        backend = FakeBatchBackend()
        smart = SmartTranslator.__new__(SmartTranslator)
        smart.settings = SimpleNamespace(translation_cache_size=32)
        smart._backends = [backend]
        smart._available_by_direction = {("en", "zh"): backend}
        smart._translation_cache = __import__("collections").OrderedDict()
        smart._last_backend_name = ""

        out = smart.translate_batch(["one", "two", "three"], "en", "zh")

        self.assertEqual(out, ["译一", "译二", "译三"])
        self.assertEqual(backend.single_calls, ["two"])
        self.assertEqual(smart.last_performance["request_count"], 2)


class TestMultilingualRenderAndMask(unittest.TestCase):
    def test_english_wrap_preserves_word_boundaries_and_punctuation(self):
        from app.services.engines.renderer import PILRenderer

        renderer = PILRenderer()
        font = renderer._get_font(16, "en")
        lines = renderer._wrap_latin_text("Don't re-enter -- wait!", font, 90)
        self.assertTrue(all(line.strip() for line in lines))
        self.assertTrue(any("Don't" in line for line in lines))
        self.assertTrue(any("!" in line for line in lines))

    def test_renderer_selects_target_language_font_and_records_it(self):
        from app.services.engines.renderer import PILRenderer
        from app.services.pipeline import TextRegion

        renderer = PILRenderer()
        self.assertTrue(renderer._font_paths["zh"])
        self.assertTrue(renderer._font_paths["ja"])
        self.assertTrue(renderer._font_paths["en"])
        region = TextRegion(
            box=[[30, 30], [170, 30], [170, 80], [30, 80]],
            translated="Hello world",
            group_translated="Hello world",
            source_lang="ja",
        )
        source = Path(tempfile.gettempdir()) / "manga_renderer_font_source.png"
        Image.new("RGB", (200, 120), "white").save(source)
        renderer.render(source, [region], target_lang="en")
        self.assertEqual(region.render_font, renderer._font_paths["en"])

    def test_mask_uses_region_language_without_disabling_polygon_fill(self):
        from app.services.engines.mask import build_region_mask
        from app.services.pipeline import TextRegion

        img = np.ones((100, 180, 3), dtype=np.uint8) * 255
        region = TextRegion(
            box=[[20, 30], [160, 30], [160, 65], [20, 65]],
            poly=[[20, 30], [160, 30], [160, 65], [20, 65]],
            source_lang="en",
        )
        result = build_region_mask(img, region)
        self.assertIsNotNone(result)
        self.assertTrue(result[4].any())

    def test_chinese_closing_punctuation_never_starts_a_line(self):
        from app.services.engines.renderer import PILRenderer

        renderer = PILRenderer()
        font = renderer._get_font(20, "zh")
        lines = renderer._protect_cjk_line_boundaries(
            ["UAP的存在", "，并重组为", "太空军！"], font, 200
        )

        self.assertTrue(all(not line.startswith("，") for line in lines))
        self.assertIn("，", "".join(lines))


class TestPunctuationMerge(unittest.TestCase):
    def test_question_mark_merges_into_neighbor(self):
        from app.services.engines.bubble import merge_punctuation_regions
        from app.services.pipeline import TextRegion

        main = TextRegion(
            box=[[100, 20], [140, 20], [140, 200], [100, 200]],
            text="どう思う",
            confidence=0.9,
            direction="v",
        )
        mark = TextRegion(
            box=[[100, 210], [140, 210], [140, 240], [100, 240]],
            text="？",
            confidence=0.9,
            direction="v",
        )
        out = merge_punctuation_regions([main, mark])
        self.assertEqual(len(out), 1)
        self.assertIn("？", out[0].text)


class TestRendererDirection(unittest.TestCase):
    def test_direction_overrides_bubble_shape(self):
        from app.services.engines.renderer import PILRenderer
        from app.services.pipeline import TextRegion

        # 方向优先：竖排文字即使位于矮宽/圆形气泡也保持竖排（用户实测横排为 bug）
        r = TextRegion(box=[[10, 10], [200, 10], [200, 40], [10, 40]], direction="v")
        self.assertTrue(PILRenderer._use_vertical([r], bw=190, bh=30))
        # 形状强竖（高>>宽）同样竖排
        r2 = TextRegion(box=[[10, 10], [40, 10], [40, 200], [10, 200]], direction="v")
        self.assertTrue(PILRenderer._use_vertical([r2], bw=30, bh=190))
        # 横排方向即使在竖长气泡也保持横排
        r3 = TextRegion(box=[[10, 10], [40, 10], [40, 200], [10, 200]], direction="h")
        self.assertFalse(PILRenderer._use_vertical([r3], bw=30, bh=190))

    def test_no_direction_falls_back_to_aspect(self):
        from app.services.engines.renderer import PILRenderer
        from app.services.pipeline import TextRegion

        r = TextRegion(box=[[10, 10], [40, 10], [40, 200], [10, 200]])
        self.assertTrue(PILRenderer._use_vertical([r], bw=30, bh=190))
        self.assertFalse(PILRenderer._use_vertical([r], bw=190, bh=30))


class TestDeepSeekPrompt(unittest.TestCase):
    def test_prompt_contains_glossary_and_no_invent(self):
        from app.services.engines.translator import DeepSeekTranslator

        t = DeepSeekTranslator.__new__(DeepSeekTranslator)
        prompt = t._system_prompt("ja", "zh", glossary={"めい": "芽衣"})
        self.assertIn("芽衣", prompt)
        self.assertIn("めい", prompt)
        self.assertIn("换行", prompt)
        self.assertTrue("脑补" in prompt or "原文没有" in prompt)

    def test_context_prompt_segment_count(self):
        from app.services.engines.translator import DeepSeekTranslator

        t = DeepSeekTranslator.__new__(DeepSeekTranslator)
        prompt = t._context_prompt("ja", "zh", glossary=None, n=5)
        self.assertIn("5段", prompt)
        self.assertIn("<序号>译文</序号>", prompt)
        self.assertIn("跨气泡对白", prompt)
        self.assertIn("不得在后一段重复", prompt)

    def test_parse_segments(self):
        from app.services.engines.translator import DeepSeekTranslator

        parse = DeepSeekTranslator._parse_segments
        # 正常解析
        self.assertEqual(parse("<1>你好</1><2>再见</2>", 2), ["你好", "再见"])
        # 带代码围栏与换行
        self.assertEqual(parse("```xml\n<1>早上好</1>\n<2>晚安</2>\n```", 2), ["早上好", "晚安"])
        # 序号乱序也能按号对齐
        self.assertEqual(parse("<2>乙</2><1>甲</1>", 2), ["甲", "乙"])
        # 缺号 → None
        self.assertIsNone(parse("<1>你好</1>", 2))
        # 多余尾随文本不影响
        self.assertEqual(parse("译文：<1>好</1><2>行</2> 以上", 2), ["好", "行"])

    def test_english_prompt_has_comic_translation_strategy(self):
        from app.services.engines.translator import DeepSeekTranslator

        t = DeepSeekTranslator.__new__(DeepSeekTranslator)
        prompt = t._system_prompt("en", "zh")
        for phrase in ("口语", "俚语", "角色", "禁止脑补", "只输出译文", "谁都好", "同义成分"):
            self.assertIn(phrase, prompt)

    def test_english_full_name_glossary_expands_page_alias(self):
        from app.services.engines.translator import expand_english_glossary_aliases

        texts = [
            "HE LOOKED LIKE KEN TAKAKURA.",
            "WILL I NEVER MEET ANOTHER KEN?",
        ]
        glossary = expand_english_glossary_aliases(texts, {"Ken Takakura": "高仓健"})
        self.assertEqual(glossary["Ken"], "高仓健")
        self.assertNotIn("Takakura", glossary)

    def test_japanese_prompt_does_not_use_english_strategy(self):
        from app.services.engines.translator import DeepSeekTranslator

        t = DeepSeekTranslator.__new__(DeepSeekTranslator)
        prompt = t._system_prompt("ja", "zh")
        self.assertNotIn("英语漫画到中文", prompt)
        self.assertIn("日语", prompt)

    def test_japanese_context_marks_mixed_script_name_without_literal_translation(self):
        from app.services.engines.translator import DeepSeekTranslator

        t = DeepSeekTranslator.__new__(DeepSeekTranslator)
        prompt = t._context_prompt("ja", "zh", None, 2, texts=["さすが", "コテ川だ‼"])
        self.assertIn("コテ川", prompt)
        self.assertIn("不要按字面词义拆译", prompt)
        self.assertIn("惯用中文名", prompt)

    def test_japanese_known_name_glossary_is_applied_after_remote_translation(self):
        from app.services.engines.translator import _apply_glossary_text

        glossary = GlossaryService().get_mapping("ja", "zh")
        self.assertEqual(glossary["コテ川"], "古手川")
        self.assertEqual(_apply_glossary_text("真不愧是コテ川啊！", glossary, "ja"), "真不愧是古手川啊！")

    def test_preserves_japanese_heart_when_translation_returns_placeholder(self):
        from app.services.pipeline import preserve_decorative_symbols

        self.assertEqual(preserve_decorative_symbols("風紀委員♡", "风纪委员□"), "风纪委员♡")
        self.assertEqual(preserve_decorative_symbols("帰ってきた！♡", "回来了！"), "回来了！♡")

    def test_english_context_prompt_keeps_numbered_segments(self):
        from app.services.engines.translator import DeepSeekTranslator

        t = DeepSeekTranslator.__new__(DeepSeekTranslator)
        prompt = t._context_prompt("en", "zh", None, 3)
        self.assertIn("俚语", prompt)
        self.assertIn("不要合并、拆分、遗漏或新增", prompt)

    def test_english_context_prompt_tracks_repeated_character_name(self):
        from app.services.engines.translator import DeepSeekTranslator

        t = DeepSeekTranslator.__new__(DeepSeekTranslator)
        texts = [
            "I LIKE TOUGH GUYS LIKE KEN TAKAKURA!",
            "HE LOOKED LIKE KEN TAKAKURA.",
            "WILL I NEVER MEET ANOTHER KEN?",
        ]
        prompt = t._context_prompt("en", "zh", None, 3, texts=texts)
        self.assertIn("KEN TAKAKURA", prompt)
        self.assertIn("同一专名", prompt)
        self.assertIn("简称", prompt)

    def test_english_uses_separate_model_and_japanese_keeps_original_model(self):
        from app.services.engines.translator import DeepSeekTranslator

        t = DeepSeekTranslator.__new__(DeepSeekTranslator)
        t.settings = SimpleNamespace(
            deepseek_english_model="deepseek-v4-flash",
            deepseek_model="deepseek-v4-flash",
        )
        self.assertEqual(
            t._model_candidates("en", "zh"),
            ["deepseek-v4-flash"],
        )
        self.assertEqual(t._model_candidates("ja", "zh"), ["deepseek-v4-flash"])

    def test_english_sampling_does_not_change_japanese_parameters(self):
        from app.services.engines.translator import DeepSeekTranslator

        english = DeepSeekTranslator._sampling_options("deepseek-v4-flash", "en", "zh")
        japanese = DeepSeekTranslator._sampling_options("deepseek-v4-flash", "ja", "zh")
        self.assertEqual(english["temperature"], 0.7)
        self.assertEqual(english["thinking"], {"type": "disabled"})
        self.assertEqual(japanese["temperature"], 0.3)
        self.assertEqual(japanese["thinking"], {"type": "disabled"})

    def test_english_output_cleanup_does_not_touch_japanese(self):
        from app.services.engines.translator import _normalize_english_translation

        raw = "译文：“高仓健”"
        self.assertEqual(_normalize_english_translation(raw, "en", "zh"), "高仓健")
        self.assertEqual(_normalize_english_translation(raw, "ja", "zh"), raw)

    def test_parse_segments_rejects_duplicate_numbers(self):
        from app.services.engines.translator import DeepSeekTranslator

        self.assertIsNone(DeepSeekTranslator._parse_segments("<1>a</1><1>b</1>", 1))

    def test_context_batch_splits_long_page_without_mixing_other_calls(self):
        from unittest.mock import Mock

        from app.services.engines.translator import DeepSeekTranslator

        t = DeepSeekTranslator.__new__(DeepSeekTranslator)
        t._translate_context = Mock(side_effect=lambda srcs, *_args: [f"译:{item}" for item in srcs])
        texts = ["a" * 3000, "b" * 3000, "c"]
        out = t.translate_batch(texts, "en", "zh")
        self.assertEqual(out[0], "译:" + texts[0])
        self.assertEqual(out[1], "译:" + texts[1])
        self.assertEqual(t._translate_context.call_count, 2)

    def test_context_failure_can_be_handled_by_single_item_fallback(self):
        from unittest.mock import Mock

        from app.services.engines.translator import DeepSeekTranslator, SmartTranslator

        deepseek = DeepSeekTranslator.__new__(DeepSeekTranslator)
        deepseek.batch = True
        deepseek.prompt_glossary = True
        deepseek.name = "deepseek"
        deepseek._translate_context = Mock(side_effect=ValueError("格式错误"))
        deepseek.translate_one = Mock(side_effect=lambda text, *_args, **_kwargs: "单条译文")
        smart = SmartTranslator.__new__(SmartTranslator)
        smart._backends = [deepseek]
        smart._available = deepseek
        smart.last_backend_names = []
        smart.last_failures = []
        smart._last_backend_name = ""
        self.assertEqual(smart.translate_batch(["hello"], "en", "zh"), ["单条译文"])
        deepseek.translate_one.assert_called()


class TestBubbleGroupMergeBalloon(unittest.TestCase):
    """页眉横幅等细长组的并集膨胀不应链式吞并整页气泡（跨页杂志页实测 bug）"""

    def test_wide_strip_does_not_swallow_adjacent_bubbles(self):
        from app.services.engines.bubble import _merge_overlap_groups

        groups = [
            {"bbox": (26, 15, 985, 69), "regions": ["banner"]},
            {"bbox": (6, 38, 110, 276), "regions": ["a1"]},
            {"bbox": (148, 40, 285, 348), "regions": ["b1"]},
            {"bbox": (763, 36, 843, 340), "regions": ["c1"]},
        ]
        out = _merge_overlap_groups(groups)
        self.assertEqual(len(out), 4)

    def test_same_bubble_columns_still_merge(self):
        from app.services.engines.bubble import _merge_overlap_groups

        groups = [
            {"bbox": (764, 309, 829, 796), "regions": ["col1"]},
            {"bbox": (805, 317, 880, 759), "regions": ["col2"]},
        ]
        out = _merge_overlap_groups(groups)
        self.assertEqual(len(out), 1)

    def test_touching_balloons_with_side_by_side_english_stacks_split(self):
        import numpy as np

        from app.services.engines.bubble import _split_side_by_side_text_groups
        from app.services.pipeline import TextRegion

        def region(x0, y0, x1, y1, text):
            return TextRegion(
                box=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                text=text,
                direction="h",
                source_lang="en",
            )

        left = [
            region(206, 992, 268, 1008, "THIS IS"),
            region(199, 1011, 277, 1027, "WHY YOU"),
            region(206, 1028, 269, 1048, "DON'T"),
            region(193, 1042, 278, 1067, "HAVE ANY"),
            region(198, 1071, 278, 1083, "FRIENDS!"),
        ]
        right = [
            region(306, 877, 410, 893, "I'M NOT IN"),
            region(308, 898, 407, 914, "THE MOOD"),
            region(303, 940, 413, 956, "NONSENSE!"),
            region(309, 1004, 406, 1020, "PUSHIN' IT"),
            region(303, 1042, 420, 1067, "NERD CRAP!"),
        ]
        shared_mask = np.full((1168, 802), 255, np.uint8)
        group = {
            "bbox": (180, 850, 440, 1110),
            "regions": left + right,
            "mask": shared_mask,
            "mask_reliable": True,
            "members": [],
        }

        out = _split_side_by_side_text_groups([group], 802, 1168)

        self.assertEqual(len(out), 2)
        self.assertEqual(
            {frozenset(r.text for r in item["regions"]) for item in out},
            {frozenset(r.text for r in left), frozenset(r.text for r in right)},
        )
        self.assertTrue(all(item["mask"] is None for item in out))
        self.assertTrue(all(not item["mask_reliable"] for item in out))

    def test_centered_english_lines_in_one_balloon_stay_together(self):
        from app.services.engines.bubble import _split_side_by_side_text_groups
        from app.services.pipeline import TextRegion

        regions = [
            TextRegion(box=[[300, 20], [420, 20], [420, 40], [300, 40]], text="ONE LINE", direction="h", source_lang="en"),
            TextRegion(box=[[310, 45], [410, 45], [410, 65], [310, 65]], text="SECOND", direction="h", source_lang="en"),
            TextRegion(box=[[320, 70], [400, 70], [400, 90], [320, 90]], text="THIRD", direction="h", source_lang="en"),
            TextRegion(box=[[305, 95], [415, 95], [415, 115], [305, 115]], text="LAST LINE", direction="h", source_lang="en"),
        ]
        group = {"bbox": (280, 0, 440, 135), "regions": regions, "mask": None, "mask_reliable": False}

        out = _split_side_by_side_text_groups([group], 800, 1200)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["regions"], regions)

    def test_conjunction_columns_in_one_balloon_stay_together(self):
        from app.services.engines.bubble import _split_side_by_side_text_groups
        from app.services.pipeline import TextRegion

        def region(x0, y0, x1, y1, text):
            return TextRegion(
                box=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                text=text,
                direction="h",
                source_lang="en",
            )

        regions = [
            region(75, 710, 166, 728, "AND"),
            region(55, 735, 168, 753, "SKINWALKER"),
            region(82, 760, 158, 778, "RANCH!"),
            region(195, 700, 238, 718, "AND"),
            region(188, 725, 285, 743, "STUFF ABOUT"),
            region(190, 750, 292, 768, "LIEUTENANT"),
        ]
        group = {
            "bbox": (45, 680, 305, 800),
            "regions": regions,
            "mask": np.full((1168, 802), 255, np.uint8),
            "mask_reliable": True,
        }

        out = _split_side_by_side_text_groups([group], 802, 1168)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["regions"], regions)

    def test_overlapping_boxes_with_distinct_masks_do_not_merge(self):
        import numpy as np

        from app.services.engines.bubble import _merge_overlap_groups

        left = np.zeros((200, 200), np.uint8)
        right = np.zeros((200, 200), np.uint8)
        left[20:160, 20:90] = 255
        right[30:170, 105:180] = 255
        groups = [
            {"bbox": (20, 20, 125, 160), "regions": ["a"], "mask": left, "mask_reliable": True},
            {"bbox": (75, 30, 180, 170), "regions": ["b"], "mask": right, "mask_reliable": True},
        ]
        out = _merge_overlap_groups(groups)
        self.assertEqual(len(out), 2)

    def test_vertical_cards_with_small_visual_gap_are_not_text_adjacent(self):
        from app.services.engines.bubble import _text_regions_adjacent
        from app.services.pipeline import TextRegion

        upper = TextRegion(
            box=[[1130, 213], [1152, 213], [1152, 350], [1130, 350]],
            text="上方卡片",
            direction="v",
        )
        lower = TextRegion(
            box=[[1126, 411], [1144, 411], [1144, 545], [1126, 545]],
            text="下方卡片",
            direction="v",
        )
        self.assertFalse(_text_regions_adjacent(upper, lower))

    def test_original_separator_vetoes_identical_leaked_masks(self):
        import numpy as np

        from app.services.engines.bubble import _container_merge_score
        from app.services.pipeline import TextRegion

        left = TextRegion(
            box=[[40, 30], [70, 30], [70, 170], [40, 170]],
            text="左侧气泡",
            direction="v",
        )
        right = TextRegion(
            box=[[130, 30], [160, 30], [160, 170], [130, 170]],
            text="右侧气泡",
            direction="v",
        )
        gray = np.full((200, 200), 255, np.uint8)
        gray[20:180, 99:102] = 0
        leaked = np.full((200, 200), 255, np.uint8)
        group = {
            "bbox": (20, 20, 180, 180),
            "regions": [left],
            "members": [((20, 20, 180, 180), leaked, True, left)],
            "mask": leaked,
            "mask_reliable": True,
        }

        score = _container_merge_score(
            (20, 20, 180, 180), leaked, True, right, group, 0.15, gray
        )

        self.assertEqual(score, 0.0)

    def test_same_bubble_without_separator_keeps_reliable_mask_merge(self):
        import numpy as np

        from app.services.engines.bubble import _container_merge_score
        from app.services.pipeline import TextRegion

        left = TextRegion(
            box=[[40, 30], [70, 30], [70, 170], [40, 170]],
            text="同一气泡左列",
            direction="v",
        )
        right = TextRegion(
            box=[[90, 30], [120, 30], [120, 170], [90, 170]],
            text="同一气泡右列",
            direction="v",
        )
        gray = np.full((200, 200), 255, np.uint8)
        shared = np.full((200, 200), 255, np.uint8)
        group = {
            "bbox": (20, 20, 150, 180),
            "regions": [left],
            "members": [((20, 20, 150, 180), shared, True, left)],
            "mask": shared,
            "mask_reliable": True,
        }

        score = _container_merge_score(
            (20, 20, 150, 180), shared, True, right, group, 0.15, gray
        )

        self.assertGreater(score, 0.0)

    def test_touching_neighbor_prevents_false_split_from_far_column(self):
        import numpy as np

        from app.services.engines.bubble import _groups_separated_by_boundary
        from app.services.pipeline import TextRegion

        candidate = TextRegion(
            box=[[70, 30], [90, 30], [90, 170], [70, 170]],
            text="候选列",
            direction="v",
        )
        touching = TextRegion(
            box=[[90, 30], [112, 30], [112, 170], [90, 170]],
            text="贴合列",
            direction="v",
        )
        farther = TextRegion(
            box=[[125, 30], [145, 30], [145, 170], [125, 170]],
            text="较远列",
            direction="v",
        )
        gray = np.full((200, 200), 255, np.uint8)
        gray[20:180, 118:121] = 0

        separated = _groups_separated_by_boundary(
            [candidate], [touching, farther], gray
        )

        self.assertFalse(separated)


class TestBubbleGeometryLeakCap(unittest.TestCase):
    """泛洪掩膜超出组气泡包围盒太多即判泄漏，回退锚定矩形（防整页巨字）"""

    def _make_region(self):
        from app.services.pipeline import TextRegion

        r = TextRegion(box=[[50, 50], [190, 50], [190, 290], [50, 290]], text="x", direction="v")
        r.group_bounds = (40, 40, 200, 300)
        return r

    def test_rejects_mask_beyond_group_bounds(self):
        import numpy as np
        from unittest import mock

        from app.services.engines.renderer import SAFE_EXPAND_RATIO, PILRenderer

        r = self._make_region()
        mask = np.zeros((1000, 1000), np.uint8)
        mask[:150, :] = 255  # 150k px，通过 6× 面积校验但 bbox 远超 group_bounds
        fake = lambda *a, **k: ((0, 0, 1000, 1000), mask)
        with mock.patch("app.services.engines.bubble.bubble_with_mask", fake):
            bb, out = PILRenderer()._bubble_geometry(np.zeros((1000, 1000, 3), np.uint8), [r], 1000, 1000)
        self.assertIsNone(out)
        # 泄漏掩膜被拒后回退安全扩展框：受 SAFE_EXPAND_RATIO 上限约束，绝不用整页掩膜
        cx = (50 + 190) / 2.0
        cap_x1 = min(1000, int(cx + 140 * SAFE_EXPAND_RATIO / 2))
        self.assertLessEqual(bb[2], cap_x1)
        self.assertLess(bb[2], 1000)

    def test_accepts_mask_within_group_bounds(self):
        import numpy as np
        from unittest import mock

        from app.services.engines.renderer import PILRenderer

        r = self._make_region()
        mask = np.zeros((1000, 1000), np.uint8)
        mask[40:300, 40:200] = 255
        fake = lambda *a, **k: ((40, 40, 200, 300), mask)
        with mock.patch("app.services.engines.bubble.bubble_with_mask", fake):
            bb, out = PILRenderer()._bubble_geometry(np.zeros((1000, 1000, 3), np.uint8), [r], 1000, 1000)
        self.assertIsNotNone(out)
        self.assertEqual(bb, (40, 40, 200, 300))

    def test_sparse_multi_region_group_does_not_use_rectangle_fallback(self):
        import numpy as np
        from unittest import mock

        from app.services.engines.renderer import PILRenderer
        from app.services.pipeline import TextRegion

        a = TextRegion(box=[[20, 20], [60, 20], [60, 80], [20, 80]], text="a", direction="v")
        b = TextRegion(box=[[300, 300], [340, 300], [340, 360], [300, 360]], text="b", direction="v")
        a.group_bounds = b.group_bounds = (20, 20, 340, 360)
        fake = lambda *args, **kwargs: ((10, 10, 350, 370), None)
        with mock.patch("app.services.engines.bubble.bubble_with_mask", fake):
            bb, mask = PILRenderer()._bubble_geometry(
                np.zeros((400, 400, 3), np.uint8), [a, b], 400, 400
            )
        self.assertIsNone(bb)
        self.assertIsNone(mask)


class TestChineseBlockLayout(unittest.TestCase):
    def test_chinese_layout_does_not_keep_source_line_breaks(self):
        from app.services.engines.renderer import PILRenderer

        text = PILRenderer._layout_block_text("第一句\n第二句\n『第三句』", "zh")
        self.assertEqual(text, "第一句第二句『第三句』")

    def test_other_languages_keep_line_breaks(self):
        from app.services.engines.renderer import PILRenderer

        self.assertEqual(PILRenderer._layout_block_text("one\ntwo", "en"), "one\ntwo")

    def test_vertical_columns_avoid_splitting_common_words(self):
        from app.services.engines.renderer import PILRenderer

        columns = PILRenderer._split_semantic_columns("已经没有什么祖先的土地了所以必须离开", 6)
        boundaries = {a[-1] + b[0] for a, b in zip(columns, columns[1:])}
        self.assertNotIn("祖先", boundaries)
        self.assertNotIn("土地", boundaries)
        self.assertNotIn("所以", boundaries)
        self.assertNotIn("必须", boundaries)

    def test_vertical_columns_keep_closing_punctuation_with_previous_text(self):
        from app.services.engines.renderer import PILRenderer

        columns = PILRenderer._split_semantic_columns("这是第一句话，但是还没有说完。", 6)
        self.assertTrue(all(not col.startswith(("，", "。")) for col in columns))


class TestTranslationQualityGate(unittest.TestCase):
    def test_flags_free_fallback_and_short_translation(self):
        from app.services.pipeline import assess_translation_quality

        warnings = assess_translation_quality(
            "もう祖先の土地なんてないんだ", "土地", "ja", "zh", "google"
        )
        self.assertTrue(any("回退后端" in item for item in warnings))
        self.assertTrue(any("明显过短" in item for item in warnings))

    def test_flags_japanese_left_in_chinese_translation(self):
        from app.services.pipeline import assess_translation_quality

        warnings = assess_translation_quality("これはテストです", "这是テストです", "ja", "zh", "deepseek")
        self.assertTrue(any("仍含较多日文" in item for item in warnings))

    def test_mit_uses_separate_translation_and_erase_thresholds(self):
        from app.services.pipeline import ocr_thresholds

        self.assertEqual(ocr_thresholds("mit48"), (0.20, 0.0))
        self.assertEqual(ocr_thresholds("paddle"), (0.50, 0.50))

    def test_flags_english_residual_but_not_a_single_name_brand_or_abbreviation(self):
        from app.services.pipeline import assess_translation_quality

        warnings = assess_translation_quality(
            "Hey, I don't know what happened", "Hey, I don't know what happened", "en", "zh", "deepseek"
        )
        self.assertTrue(any("残留大量英文" in item for item in warnings))
        for text in ("Batman", "OpenAI", "NASA"):
            warnings = assess_translation_quality(text, text, "en", "zh", "deepseek")
            self.assertFalse(any("残留大量英文" in item for item in warnings))

    def test_flags_english_number_date_and_unit_omission(self):
        from app.services.pipeline import assess_translation_quality

        warnings = assess_translation_quality(
            "Meet me on 2026/08/31 at 5 kg.", "在那天见。", "en", "zh", "deepseek"
        )
        self.assertTrue(any("数字可能漏译" in item for item in warnings))
        self.assertTrue(any("单位可能漏译" in item for item in warnings))

    def test_english_glossary_is_case_insensitive_and_whole_word_only(self):
        from app.services.engines.translator import SmartTranslator

        self.assertEqual(
            SmartTranslator._apply_glossary("BATMAN and Batmanish", {"Batman": "蝙蝠侠"}, "en"),
            "蝙蝠侠 and Batmanish",
        )

    def test_english_glossary_quality_warning_checks_key_term(self):
        from app.services.pipeline import assess_translation_quality

        warnings = assess_translation_quality(
            "Batman is here", "他来了", "en", "zh", "deepseek", {"Batman": "蝙蝠侠"}
        )
        self.assertTrue(any("词典术语可能遗漏" in item for item in warnings))

    def test_english_alias_quality_warning_uses_full_name_translation(self):
        from app.services.engines.translator import expand_english_glossary_aliases
        from app.services.pipeline import assess_translation_quality

        glossary = expand_english_glossary_aliases(
            ["HE LOOKED LIKE KEN TAKAKURA.", "ANOTHER KEN?"],
            {"Ken Takakura": "高仓健"},
        )
        warnings = assess_translation_quality(
            "ANOTHER KEN?", "另一个肯？", "en", "zh", "deepseek", glossary
        )
        self.assertTrue(any("词典术语可能遗漏:高仓健" in item for item in warnings))

    def test_english_quality_flags_model_explanation_or_markdown(self):
        from app.services.pipeline import assess_translation_quality

        warnings = assess_translation_quality(
            "I am here", "译文：**我在这里**", "en", "zh", "deepseek"
        )
        self.assertTrue(any("解释、Markdown" in item for item in warnings))

    def test_japanese_quality_gate_remains_unchanged(self):
        from app.services.pipeline import assess_translation_quality

        warnings = assess_translation_quality("これはテストです", "这是テストです", "ja", "zh", "deepseek")
        self.assertTrue(any("仍含较多日文" in item for item in warnings))

class TestNonBubbleClassify(unittest.TestCase):
    """非气泡文字（刊头横条/跨页横带）几何判定 → 保留原文不翻译"""

    def test_banner_strip_classified(self):
        from app.services.engines.bubble import _is_strip_bbox

        self.assertTrue(_is_strip_bbox((26, 15, 985, 69), 1921, 1412, "h"))

    def test_legit_vertical_bubble_not_flagged(self):
        from app.services.engines.bubble import _is_strip_bbox

        self.assertFalse(_is_strip_bbox((764, 309, 829, 796), 1921, 1412, "v"))

    def test_legit_horizontal_bubble_not_flagged(self):
        from app.services.engines.bubble import _is_strip_bbox

        self.assertFalse(_is_strip_bbox((300, 400, 600, 520), 1921, 1412, "h"))

    def test_long_english_bubble_line_not_mistaken_for_banner(self):
        from app.services.engines.bubble import classify_non_bubble
        from app.services.pipeline import TextRegion

        region = TextRegion(
            box=[[245, 100], [370, 100], [370, 116], [245, 116]],
            text="WRONG WITH",
            direction="h",
            source_lang="en",
        )
        # 真实失败样例：泛洪回退框宽高比 8.12、高度仅占页面 2.22%。
        with patch("app.services.engines.bubble.detect_bubble", return_value=(202, 95, 413, 121)):
            self.assertFalse(classify_non_bubble(np.zeros((1168, 802, 3), np.uint8), region, 802, 1168, "en"))
            self.assertTrue(classify_non_bubble(np.zeros((1168, 802, 3), np.uint8), region, 802, 1168, "ja"))

    def test_english_page_wide_band_still_classified(self):
        from app.services.engines.bubble import classify_non_bubble
        from app.services.pipeline import TextRegion

        region = TextRegion(
            box=[[40, 20], [700, 20], [700, 50], [40, 50]],
            text="CHAPTER TITLE",
            direction="h",
            source_lang="en",
        )
        with patch("app.services.engines.bubble.detect_bubble", return_value=(30, 15, 760, 55)):
            self.assertTrue(classify_non_bubble(np.zeros((1168, 802, 3), np.uint8), region, 802, 1168, "en"))

    def test_drop_non_bubble_regions(self):
        from app.services.pipeline import TextRegion, drop_non_bubble_regions

        a = TextRegion(box=[[0, 0], [10, 0], [10, 10], [0, 10]], text="a")
        b = TextRegion(box=[[0, 0], [10, 0], [10, 10], [0, 10]], text="b")
        b._no_erase = True
        out = drop_non_bubble_regions([a, b])
        self.assertEqual([r.text for r in out], ["a"])

    def test_preserve_latin_label_on_japanese_page(self):
        from app.services.pipeline import TextRegion, preserve_latin_label

        label = TextRegion(
            box=[[0, 0], [100, 0], [100, 20], [0, 20]],
            text="ASIAN KUNG-FU GENERATION",
        )
        self.assertTrue(preserve_latin_label(label, "ja"))
        self.assertFalse(preserve_latin_label(label, "en"))

    def test_japanese_dialogue_is_not_latin_label(self):
        from app.services.pipeline import TextRegion, preserve_latin_label

        region = TextRegion(box=[[0, 0], [40, 0], [40, 40], [0, 40]], text="メール読まれてよかったね")
        self.assertFalse(preserve_latin_label(region, "ja"))


class TestMixedOcrMixing(unittest.TestCase):
    """manga-ocr 只救空行：mit48 已读出内容（即使低置信）不交给 manga-ocr 覆盖成乱码"""

    def _quad(self, text, prob):
        from types import SimpleNamespace

        return SimpleNamespace(text=text, prob=prob)

    def test_empty_only_sent_to_mangaocr(self):
        from app.services.engines.ocr import _empty_quads_only

        quads = [
            self._quad("ハレンチ撲滅", 0.55),   # mit48 有输出但低置信 → 保留，不再交给 manga-ocr
            self._quad("", 0.0),                # 完全没读出 → 交给 manga-ocr
            self._quad("とします", 0.56),
            self._quad("  ", 0.1),              # 纯空白 → 视为空
        ]
        out = _empty_quads_only(quads)
        self.assertEqual(len(out), 2)
        self.assertTrue(all((q.text or "").strip() == "" for q in out))


class TestVerticalCropOrientation(unittest.TestCase):
    """竖排裁剪对 manga-ocr 应恢复为竖直（窄高）；若保持水平旋转则会读成乱码"""

    def test_mangaocr_vertical_crop_is_tall(self):
        import numpy as np
        import cv2

        from app.services.engines.mit.quadrilateral import Quadrilateral

        pts = np.array([[165, 503], [204, 503], [204, 859], [165, 859]], dtype=np.float64)
        q = Quadrilateral(pts, "", 0.0)
        img = np.zeros((1000, 400, 3), dtype=np.uint8)
        raw = q.get_transformed_region(img, "v", 96)          # 共享函数：会旋转成水平条
        upright = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)    # manga-ocr 路径：转回竖直
        self.assertTrue(raw.shape[1] > raw.shape[0])          # 原始共享结果偏宽（旋转过）
        self.assertTrue(upright.shape[0] > upright.shape[1])  # manga-ocr 用竖直窄高


class TestMaskCoverage(unittest.TestCase):
    """擦除掩膜应覆盖区域内的全部暗像素（防原文灰影残留）"""

    def test_region_mask_covers_dark_strokes(self):
        import numpy as np

        from app.services.engines.mask import build_region_mask
        from app.services.pipeline import TextRegion

        # 竖排一列假字（黑字），笔画紧贴 box 边缘——pad 缩进会把外缘笔画整块丢掉
        img = np.full((300, 120, 3), 255, dtype=np.uint8)
        for y in range(30, 270, 30):
            img[y:y + 18, 42:82] = 0  # 笔画从 box 左缘 x=42 铺到右缘 x=82
        box = [[40, 24], [84, 24], [84, 274], [40, 274]]
        poly = [[46, 28], [78, 28], [78, 270], [46, 270]]
        r = TextRegion(box=box, text="仮", poly=poly)
        res = build_region_mask(img, r, pad=2)
        self.assertIsNotNone(res)
        x0, y0, x1, y1, patch = res
        # 以完整 box 为评估区（放大 2px 边带），掩膜应包住所有暗像素，不残留
        cbx0, cby0, cbx1, cby1 = 38, 22, 86, 276
        cover_region = img[cby0:cby1, cbx0:cbx1]
        dark = (cover_region.mean(axis=2) < 120)
        if not dark.any():
            self.skipTest("无暗像素可测")
        # 把 mask 对齐到同一个全局坐标
        mask_full = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        mask_full[y0:y1, x0:x1] = patch
        sub = mask_full[cby0:cby1, cbx0:cbx1] > 0
        cov = float((sub & dark).sum()) / float(dark.sum())
        self.assertGreaterEqual(cov, 0.95, f"掩膜只覆盖了 {cov:.2%} 暗像素，会残留原文")

    def test_english_polygon_expands_instead_of_leaving_zero_border(self):
        from app.services.engines.mask import build_region_mask
        from app.services.pipeline import TextRegion

        img = np.full((80, 180, 3), 255, dtype=np.uint8)
        img[24:44, 30:150] = 0
        region = TextRegion(
            box=[[30, 24], [150, 24], [150, 44], [30, 44]],
            poly=[[30, 24], [150, 24], [150, 44], [30, 44]],
            text="WIDE ENGLISH!",
            direction="h",
            source_lang="en",
        )

        x0, y0, x1, y1, patch = build_region_mask(img, region, pad=2)
        full = np.zeros(img.shape[:2], dtype=np.uint8)
        full[y0:y1, x0:x1] = patch
        dark = img.mean(axis=2) < 120

        self.assertLess(x0, 30)
        self.assertLess(y0, 24)
        self.assertTrue((full[dark] > 0).all())


class TestBackgroundComplexity(unittest.TestCase):
    def test_black_text_on_white_is_not_mistaken_for_texture(self):
        import numpy as np

        from app.services.engines.mask import _complex_background

        patch = np.full((180, 32), 255, np.uint8)
        for y in range(8, 172, 22):
            patch[y:y + 12, 7:25] = 0

        self.assertFalse(_complex_background(patch))

    def test_dense_screentone_is_complex_background(self):
        import numpy as np

        from app.services.engines.mask import _complex_background

        yy, xx = np.indices((80, 80))
        patch = np.where((xx + yy) % 4 < 2, 245, 100).astype(np.uint8)

        self.assertTrue(_complex_background(patch))

    def test_vertical_furigana_margin_only_expands_sideways(self):
        import numpy as np

        from app.services.engines.mask import _add_furigana_margin

        image = np.full((100, 100, 3), 255, np.uint8)
        candidate = np.full((40, 20), 255, np.uint8)
        x0, y0, x1, y1, _ = _add_furigana_margin(
            image, 40, 30, 60, 70, candidate, "v"
        )

        self.assertLess(x0, 40)
        self.assertGreater(x1, 60)
        self.assertEqual((y0, y1), (30, 70))


class TestCVInpainter(unittest.TestCase):
    def test_row_fill_uses_local_bright_background(self):
        import numpy as np

        from app.services.engines.inpainter import CVInpainter

        img = np.full((1, 30, 3), 255, dtype=np.uint8)
        img[:, :10] = 30
        img[:, 20:] = 220
        mask = np.zeros((1, 30), dtype=np.uint8)
        mask[0, 3:5] = 255
        mask[0, 24:26] = 255

        CVInpainter()._fill_with_row_bg(img, mask)

        self.assertLess(img[0, 3].mean(), 80)
        self.assertGreater(img[0, 24].mean(), 180)

    def test_second_pass_removes_gray_residual_inside_known_mask(self):
        import numpy as np

        from app.services.engines.inpainter import CVInpainter
        from app.services.pipeline import TextRegion

        img = np.full((50, 50, 3), 255, np.uint8)
        img[20:23, 20:30] = 150
        patch = np.zeros((24, 24), np.uint8)
        patch[8:13, 8:20] = 255
        region = TextRegion(box=[[10, 10], [34, 10], [34, 34], [10, 34]], text="字")
        region.mask = {"bbox": (10, 10, 34, 34), "patch": patch}
        cleaned = CVInpainter()._second_pass_residual(img, [region])
        self.assertGreater(float(cleaned[20:23, 20:30].mean()), 200.0)

    def test_white_bubble_is_selected_for_local_background_fill(self):
        import numpy as np

        from app.services.engines.inpainter import CVInpainter
        from app.services.pipeline import TextRegion

        img = np.full((60, 60, 3), 255, np.uint8)
        img[20:40, 28:32] = 0
        patch = np.zeros((30, 20), np.uint8)
        patch[5:25, 7:13] = 255
        region = TextRegion(box=[[20, 15], [40, 15], [40, 45], [20, 45]], text="字")
        region.mask = {"bbox": (20, 15, 40, 45), "patch": patch}
        flat = CVInpainter()._flat_background_mask(img, [region])
        self.assertGreater(int((flat > 0).sum()), 0)

    def test_flat_fill_excludes_detector_mask_outside_text_polygon(self):
        import numpy as np

        from app.services.engines.inpainter import CVInpainter
        from app.services.pipeline import TextRegion

        img = np.full((60, 60, 3), 255, np.uint8)
        patch = np.zeros((30, 30), np.uint8)
        patch[3:27, 3:27] = 255
        region = TextRegion(
            box=[[22, 22], [38, 22], [38, 38], [22, 38]],
            poly=[[22, 22], [38, 22], [38, 38], [22, 38]],
            text="字",
        )
        region.mask = {"bbox": (15, 15, 45, 45), "patch": patch}

        flat = CVInpainter()._flat_background_mask(img, [region])

        self.assertEqual(int(flat[18, 18]), 0)
        self.assertEqual(int(flat[30, 30]), 255)


class TestLaMaInpainter(unittest.TestCase):
    def test_crop_boxes_merge_nearby_text_and_separate_distant_bubbles(self):
        import numpy as np

        from app.services.engines.lama import LaMaInpainter

        mask = np.zeros((300, 400), np.uint8)
        mask[40:70, 40:70] = 255
        mask[80:110, 45:75] = 255
        mask[210:240, 310:340] = 255

        boxes = LaMaInpainter._mask_crop_boxes(mask, merge_gap=10, context=8)

        self.assertEqual(len(boxes), 2)
        self.assertTrue(any(x0 <= 40 and y0 <= 40 and x1 >= 75 and y1 >= 110 for x0, y0, x1, y1 in boxes))

    def test_crop_inference_does_not_downscale_sparse_full_page(self):
        import numpy as np

        from app.services.engines.lama import LaMaInpainter

        engine = LaMaInpainter.__new__(LaMaInpainter)
        seen = []

        def fake_infer(image, mask):
            seen.append(image.shape[:2])
            out = image.copy()
            out[mask > 0] = 255
            return out

        engine._infer = fake_infer
        image = np.zeros((1200, 1600, 3), np.uint8)
        mask = np.zeros((1200, 1600), np.uint8)
        mask[100:180, 100:180] = 255
        mask[900:980, 1300:1380] = 255

        result = engine._infer_by_crops(image, mask)

        self.assertEqual(len(seen), 2)
        self.assertTrue(all(h < 300 and w < 300 for h, w in seen))
        self.assertTrue((result[mask > 0] == 255).all())

    def test_lama_runs_residual_cleanup_after_neural_inference(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import numpy as np
        from PIL import Image

        from app.services.engines.lama import LaMaInpainter
        from app.services.pipeline import TextRegion

        engine = LaMaInpainter.__new__(LaMaInpainter)
        engine.model = object()
        engine._infer_by_crops = lambda image, mask: image.copy()
        engine._save_temp = lambda arr: Path("cleaned.png")
        region = TextRegion(box=[[5, 5], [15, 5], [15, 15], [5, 15]], text="字")
        region.mask = {"bbox": (5, 5, 15, 15), "patch": np.full((10, 10), 255, np.uint8)}
        image = np.full((20, 20, 3), 255, np.uint8)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "app.services.engines.inpainter.CVInpainter._second_pass_residual",
            return_value=image,
        ) as cleanup:
            source = Path(tmp) / "source.png"
            Image.fromarray(image).save(source)
            engine.inpaint(source, [region])

        cleanup.assert_called_once()


class TestVerticalColumnOrder(unittest.TestCase):
    """竖排整块应按原文行序分配列，不按字数重切打散句读"""

    def test_columns_preserve_line_order(self):
        from app.services.engines.renderer import PILRenderer

        r = PILRenderer.__new__(PILRenderer)
        lines = ["因为你总是那样想", "所以没人能理解你", "一旦认定了", "就是这样", "从小时候起"]
        # 足够大的可用高度，让每行都能独立成列（列数=行数）
        cols = r._balance_columns(lines, sum(len(l) for l in lines), avail_h=10000, char_h=30)
        self.assertIsNotNone(cols)
        self.assertEqual(cols, lines, "竖排列应保持原文行序，不应按字数切碎")


class TestRendererSafetyBounds(unittest.TestCase):
    """渲染异常输入必须有限完成，且单气泡失败不能影响其他气泡。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.source = Path(self.tmp.name) / "renderer_safety.png"
        Image.new("RGB", (240, 180), "white").save(self.source)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _region(index, box, text, direction="h"):
        from app.services.pipeline import TextRegion

        x0, y0, x1, y1 = box
        region = TextRegion(
            box=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            translated=text,
            group_translated=text,
            direction=direction,
        )
        region.group_index = index
        region.group_bounds = box
        return region

    def _render_size(self, regions, target_lang="zh"):
        from app.services.engines.renderer import PILRenderer

        out = PILRenderer().render(self.source, regions, target_lang=target_lang)
        return Image.open(io.BytesIO(out)).size

    def test_empty_translation_finishes_and_keeps_image_size(self):
        region = self._region(0, (30, 30, 210, 140), "")
        self.assertEqual(self._render_size([region]), (240, 180))

    def test_long_chinese_translation_finishes(self):
        region = self._region(0, (30, 30, 210, 140), "这是一段超长中文译文" * 800)
        self.assertEqual(self._render_size([region]), (240, 180))

    def test_long_english_word_finishes(self):
        region = self._region(0, (30, 30, 210, 140), "A" * 5000)
        self.assertEqual(self._render_size([region], target_lang="en"), (240, 180))

    def test_long_english_word_wrap_keeps_all_characters(self):
        from app.services.engines.renderer import PILRenderer

        renderer = PILRenderer()
        renderer._active_target_lang = "en"
        font = renderer._get_font(16, "en")
        text = "A" * 5000
        lines = renderer._wrap_latin_text(text, font, 1)
        self.assertEqual("".join(lines), text)

    def test_long_english_word_in_subcharacter_container_finishes(self):
        region = self._region(0, (100, 10, 102, 170), "A" * 5000)
        self.assertEqual(self._render_size([region], target_lang="en"), (240, 180))

    def test_narrow_vertical_bubble_finishes(self):
        region = self._region(0, (100, 10, 112, 170), "竖排译文" * 300, direction="v")
        self.assertEqual(self._render_size([region]), (240, 180))

    def test_narrow_vertical_bubble_retries_below_page_minimum_font(self):
        from app.services.engines.renderer import PILRenderer

        renderer = PILRenderer()
        renderer._active_target_lang = "zh"
        overlay = Image.new("RGBA", (38, 99), (0, 0, 0, 0))
        renderer._render_vertical_bubble_block(
            ImageDraw.Draw(overlay), "RN对战烟火", 0, 0, 38, 99, 12
        )
        self.assertGreater(np.count_nonzero(np.array(overlay.getchannel("A"))), 0)

    def test_extreme_aspect_bubble_finishes(self):
        region = self._region(0, (5, 80, 235, 88), "横向极端气泡译文" * 100)
        self.assertEqual(self._render_size([region]), (240, 180))

    def test_narrow_horizontal_bubble_font_search_is_bounded(self):
        from app.services.engines.renderer import PILRenderer

        region = self._region(6, (80, 40, 151, 127), "窄气泡十字文本")
        renderer = PILRenderer()
        with patch.object(renderer, "_get_font", wraps=renderer._get_font) as get_font:
            out = renderer.render(self.source, [region], target_lang="zh")
        self.assertEqual(Image.open(io.BytesIO(out)).size, (240, 180))
        self.assertLessEqual(get_font.call_count, 2)

    def test_narrow_horizontal_multiline_font_search_is_bounded(self):
        from app.services.engines.renderer import PILRenderer

        renderer = PILRenderer()
        with patch.object(renderer, "_get_font", side_effect=AssertionError("估算过程不应加载字体")):
            size = renderer._estimate_multiline_font("什么？！不！等等……", 54, 66, 73, 10)
        self.assertGreaterEqual(size, 10)
        self.assertLessEqual(size, 73)

    def test_manual_horizontal_font_size_is_not_auto_shrunk(self):
        from app.services.engines.renderer import PILRenderer

        region = self._region(6, (30, 45, 210, 125), "手动字号预览")
        region.style_font_size = 25
        renderer = PILRenderer()
        renderer.render(self.source, [region], target_lang="zh")
        self.assertEqual(region._last_render_font_size, 25)

    def test_adjacent_two_text_boxes_are_split_after_container_merge(self):
        from app.services.engines.bubble import _split_spatially_separated_text_groups

        left = self._region(0, (20, 45, 45, 75), "left")
        right = self._region(1, (100, 50, 125, 80), "right")
        group = {"regions": [left, right], "bbox": (10, 35, 135, 90), "members": []}
        split = _split_spatially_separated_text_groups([group], 240, 180)
        self.assertEqual(len(split), 2)
        self.assertEqual({item["regions"][0].translated for item in split}, {"left", "right"})

    def test_distant_vertical_text_boxes_are_split_after_container_merge(self):
        from app.services.engines.bubble import _split_spatially_separated_text_groups

        left = self._region(0, (20, 30, 40, 120), "left", direction="v")
        right = self._region(1, (140, 35, 160, 125), "right", direction="v")
        group = {"regions": [left, right], "bbox": (10, 20, 170, 135), "members": []}
        split = _split_spatially_separated_text_groups([group], 240, 180)
        self.assertEqual(len(split), 2)

    def test_nearby_vertical_columns_remain_in_one_group(self):
        from app.services.engines.bubble import _split_spatially_separated_text_groups

        left = self._region(0, (70, 30, 90, 120), "left", direction="v")
        right = self._region(1, (100, 35, 120, 125), "right", direction="v")
        group = {"regions": [left, right], "bbox": (60, 20, 130, 135), "members": []}
        split = _split_spatially_separated_text_groups([group], 240, 180)
        self.assertEqual(len(split), 1)

    def test_stacked_vertical_bubbles_are_split(self):
        from app.services.engines.bubble import _split_spatially_separated_text_groups

        upper = self._region(0, (90, 25, 125, 75), "upper", direction="v")
        lower = self._region(1, (88, 105, 128, 165), "lower", direction="v")
        group = {"regions": [upper, lower], "bbox": (75, 15, 140, 175), "members": []}
        split = _split_spatially_separated_text_groups([group], 240, 180)
        self.assertEqual(len(split), 2)

    def test_vertical_layout_has_fixed_search_bound(self):
        from app.services.engines.renderer import MAX_VERTICAL_FONT_TRIES, PILRenderer

        renderer = PILRenderer()
        calls = 0
        original = renderer._balance_columns

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        with patch.object(renderer, "_balance_columns", side_effect=counted):
            renderer._vertical_layout(["竖排文本" * 300], avail_w=10000, avail_h=20, min_font_size=1)
        self.assertLessEqual(calls, MAX_VERTICAL_FONT_TRIES + 1)

    def test_one_bubble_failure_does_not_block_other_bubbles(self):
        from app.services.engines.renderer import PILRenderer

        first = self._region(0, (20, 20, 100, 100), "first")
        second = self._region(1, (130, 20, 220, 100), "second")
        renderer = PILRenderer()
        original = renderer._render_horizontal_bubble
        calls = 0

        def flaky(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("模拟单气泡布局失败")
            return original(*args, **kwargs)

        with patch.object(renderer, "_render_horizontal_bubble", side_effect=flaky):
            out = renderer.render(self.source, [first, second], target_lang="en")
        self.assertEqual(calls, 2)
        self.assertEqual(Image.open(io.BytesIO(out)).size, (240, 180))

    def test_ten_bubbles_with_long_translation_finish(self):
        regions = []
        for index in range(10):
            x0 = 5 + (index % 5) * 47
            y0 = 5 + (index // 5) * 85
            box = (x0, y0, x0 + 40, y0 + 70)
            for _ in range(6):
                regions.append(self._region(index, box, "中文长译文" * 160))
        self.assertEqual(self._render_size(regions), (240, 180))


class TestThreeLanguageTranslation(unittest.TestCase):
    def test_detects_chinese_japanese_and_english(self):
        from app.services.language import detect_language

        self.assertEqual(detect_language("你好，今天一起回家吧").language, "zh")
        self.assertEqual(detect_language("今日は一緒に帰ろう").language, "ja")
        self.assertEqual(detect_language("Let's go home together!").language, "en")

    def test_detection_handles_mixed_empty_and_numeric_text(self):
        from app.services.language import detect_language

        self.assertEqual(detect_language("東京へGO!", fallback="zh").language, "ja")
        self.assertEqual(detect_language("", fallback="ja").language, "ja")
        self.assertEqual(detect_language("2026?!", fallback="zh").language, "zh")
        self.assertEqual(detect_language("2026?!", fallback="zh").confidence, 0.0)

    def test_all_six_directions_have_dynamic_prompts(self):
        from app.services.engines.translator import build_manga_prompt

        directions = (("zh", "ja"), ("zh", "en"), ("ja", "zh"), ("ja", "en"), ("en", "zh"), ("en", "ja"))
        for source, target in directions:
            prompt = build_manga_prompt(source, target)
            self.assertIn("只输出译文", prompt)
            self.assertIn("原文没有的信息", prompt)
        self.assertIn("自然的漫画日语", build_manga_prompt("zh", "ja"))
        self.assertIn("漫画英语", build_manga_prompt("zh", "en"))

    def test_provider_language_maps(self):
        from app.services.language import provider_language

        self.assertEqual(provider_language("google", "zh"), "zh-CN")
        self.assertEqual(provider_language("mymemory", "ja"), "ja")
        self.assertEqual(provider_language("deepl", "en"), "EN")

    def test_language_validation_accepts_six_directions_and_auto(self):
        from app.api.translate import _validate_languages

        for source, target in (("zh", "ja"), ("zh", "en"), ("ja", "zh"), ("ja", "en"), ("en", "zh"), ("en", "ja"), ("auto", "zh")):
            _validate_languages(source, target)

    def test_language_validation_rejects_same_language(self):
        from fastapi import HTTPException
        from app.api.translate import _validate_languages

        for language in ("zh", "ja", "en"):
            with self.assertRaises(HTTPException):
                _validate_languages(language, language)

    def test_target_language_quality_checks(self):
        from app.services.pipeline import assess_translation_quality

        english = assess_translation_quality("你好世界", "你好世界", "zh", "en", "deepseek")
        japanese = assess_translation_quality("你好世界", "你好世界", "zh", "ja", "deepseek")
        self.assertTrue(any("中日文字符" in item for item in english))
        self.assertTrue(any("只有汉字" in item for item in japanese))

    def test_context_prompt_keeps_numbered_output_for_reverse_direction(self):
        from app.services.engines.translator import build_manga_prompt

        prompt = build_manga_prompt("zh", "en", context_count=4)
        self.assertIn("<序号>译文</序号>", prompt)
        self.assertIn("不要合并、拆分、遗漏或新增", prompt)

    def test_english_target_forces_horizontal_rendering(self):
        from app.services.engines.renderer import PILRenderer
        from app.services.pipeline import TextRegion

        source = Path(tempfile.gettempdir()) / "manga_renderer_language_source.png"
        Image.new("RGB", (180, 220), "white").save(source)
        region = TextRegion(
            box=[[60, 20], [120, 20], [120, 200], [60, 200]],
            text="你好",
            translated="A very long English sentence",
            group_translated="A very long English sentence",
            direction="v",
        )
        renderer = PILRenderer()
        with patch.object(renderer, "_render_vertical_bubble_block") as vertical:
            renderer.render(source, [region], target_lang="en")
        vertical.assert_not_called()


if __name__ == "__main__":
    unittest.main()
