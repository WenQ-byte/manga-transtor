"""后端单元测试"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

# 测试环境
os.environ["MANGA_DATA_DIR"] = tempfile.mkdtemp(prefix="manga_test_")
# 通用流水线测试固定用 PaddleOCR + CV 检测，避免依赖本地 MIT 权重或触发网络下载
os.environ["MANGA_OCR_BACKEND"] = "paddle"
os.environ["MANGA_DETECTOR_BACKEND"] = "cv"
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont

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
        pipe = create_pipeline()
        result = pipe.translate_image(self.image, "en", "zh")
        self.assertGreater(len(result.regions), 0)
        self.assertGreater(result.duration_ms, 0)

    def test_render_preserves_size(self):
        import io

        from app.services.engines import get_engine

        pipe = create_pipeline()
        result = pipe.translate_image(self.image, "en", "zh")
        renderer = get_engine("renderer")
        inpainter = get_engine("inpainter")
        cleaned = inpainter.inpaint(self.image, result.regions)
        out = renderer.render(cleaned, result.regions, target_lang="zh")
        img = Image.open(self.image)
        parsed = Image.open(io.BytesIO(out))
        self.assertEqual(parsed.size, img.size)


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

        from app.services.engines.renderer import PILRenderer

        r = self._make_region()
        mask = np.zeros((1000, 1000), np.uint8)
        mask[:150, :] = 255  # 150k px，通过 6× 面积校验但 bbox 远超 group_bounds
        fake = lambda *a, **k: ((0, 0, 1000, 1000), mask)
        with mock.patch("app.services.engines.bubble.bubble_with_mask", fake):
            bb, out = PILRenderer()._bubble_geometry(np.zeros((1000, 1000, 3), np.uint8), [r], 1000, 1000)
        self.assertIsNone(out)
        self.assertLessEqual(bb[2], 190 + int(140 * 0.35) + 1)

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

    def test_drop_non_bubble_regions(self):
        from app.services.pipeline import TextRegion, drop_non_bubble_regions

        a = TextRegion(box=[[0, 0], [10, 0], [10, 10], [0, 10]], text="a")
        b = TextRegion(box=[[0, 0], [10, 0], [10, 10], [0, 10]], text="b")
        b._no_erase = True
        out = drop_non_bubble_regions([a, b])
        self.assertEqual([r.text for r in out], ["a"])


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


if __name__ == "__main__":
    unittest.main()
