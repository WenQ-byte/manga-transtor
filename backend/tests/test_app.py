"""后端单元测试"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

# 测试环境
os.environ["MANGA_DATA_DIR"] = tempfile.mkdtemp(prefix="manga_test_")
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont

from app.services.glossary_service import GlossaryService
from app.services.pipeline import create_pipeline


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


if __name__ == "__main__":
    unittest.main()
