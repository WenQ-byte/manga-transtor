"""OCR 引擎：优先使用真实 OCR（PaddleOCR），未安装时降级为 demo 模式"""
from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.services.engines.base import BaseOCR
from app.services.pipeline import TextRegion


class PaddleOCREngine(BaseOCR):
    """基于 PaddleOCR 的真实 OCR（需要安装 paddleocr + paddlepaddle）

    PaddleOCR 自带文本检测+识别，直接对整图处理，效果远优于启发式检测。
    """

    name = "paddle"

    supports_detection = True

    def __init__(self):
        self.settings = get_settings()
        self._ocrs: dict[str, object] = {}
        self._load_error: str = ""
        self._load()

    def _load(self):
        try:
            from paddleocr import PaddleOCR

            lang_map = {"ja": "japan", "en": "en", "zh": "ch"}
            self._lang_map = lang_map
            self._PaddleOCR = PaddleOCR
            # 预加载默认语言（日语优先）
            default = self._lang_for("ja")
            self._ocrs[default] = self._create_ocr(default)
        except ImportError:
            self._load_error = "paddleocr 未安装，OCR 使用 demo 模式"
        except Exception as e:  # noqa: BLE001
            self._load_error = f"PaddleOCR 加载失败: {e}"

    def _create_ocr(self, lang: str):
        """创建 PaddleOCR 实例

        关键：
        - 禁用文档方向矫正与文档矫正（会干扰漫画竖排文字检测）
        - 保留文本行方向分类（横排文字识别需要）
        """
        return self._PaddleOCR(
            lang=lang,
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )

    def _lang_for(self, source_lang: str) -> str:
        return self._lang_map.get(source_lang, "japan")

    def _get_ocr(self, source_lang: str) -> object | None:
        lang = self._lang_for(source_lang)
        if lang not in self._ocrs:
            try:
                self._ocrs[lang] = self._create_ocr(lang)
            except Exception as e:  # noqa: BLE001
                self._load_error = f"PaddleOCR({lang}) 加载失败: {e}"
                return None
        return self._ocrs[lang]

    @property
    def available(self) -> bool:
        return bool(self._ocrs)

    def recognize(
        self,
        image_path: Path,
        regions: list[TextRegion],
        source_lang: str = "ja",
    ) -> None:
        if not self.available:
            for r in regions:
                r.text = ""
                r.confidence = 0.0
            return

        ocr = self._get_ocr(source_lang)
        if ocr is None:
            for r in regions:
                r.text = ""
                r.confidence = 0.0
            return

        try:
            # 双次检测：原图横排 + 旋转90度竖排（日/中文漫画竖排常见）
            detections = self._detect_all(image_path, ocr, source_lang)
        except Exception:
            detections = []

        # 若检测到结果，重建 region 内容
        if detections:
            self._apply_detections(regions, detections)
            return

        # 否则对现有区域逐个裁剪识别
        self._recognize_by_regions(image_path, regions, ocr)

    def _detect_all(self, image_path: Path, ocr, source_lang: str) -> list[dict]:
        """双次检测：原图横排 + 旋转 90° 竖排（合并去重）"""
        detections: list[dict] = []

        # 1. 横排检测
        try:
            result = ocr.predict(str(image_path))
            detections += self._extract_detections(result)
        except Exception:
            pass

        # 2. 竖排检测：旋转 90°（日语/中文漫画竖排常见，英语跳过）
        if source_lang in ("ja", "zh"):
            try:
                from PIL import Image
                import numpy as np

                img = Image.open(image_path)
                W, H = img.size
                # 逆时针旋转 90°，竖排文字变为横排
                rot = img.rotate(90, expand=True)
                rot_arr = np.array(rot.convert("RGB"))
                rot_result = ocr.predict(rot_arr)
                rot_dets = self._extract_detections(rot_result)
                # 坐标映射回原图：旋转图(x',y') -> 原图(W-1-y', x')
                for det in rot_dets:
                    if det["box"]:
                        det["box"] = [
                            [W - 1 - p[1], p[0]] for p in det["box"]
                        ]
                    det["vertical"] = True
                detections += rot_dets
            except Exception:
                pass

        return self._deduplicate(detections)

    def _deduplicate(self, detections: list[dict]) -> list[dict]:
        """去重：竖排检测优先，去除被竖排框覆盖的横排误检"""
        vertical = [d for d in detections if d.get("vertical")]
        horizontal = [d for d in detections if not d.get("vertical")]

        if not vertical:
            return detections

        result = list(vertical)
        for det in horizontal:
            if not det.get("box"):
                result.append(det)
                continue
            covered = False
            for v in vertical:
                if not v.get("box"):
                    continue
                if self._iou(det["box"], v["box"]) > 0.3:
                    covered = True
                    break
            if not covered:
                result.append(det)
        return result

    @staticmethod
    def _box_bounds(box) -> tuple[float, float, float, float]:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return min(xs), min(ys), max(xs), max(ys)

    def _iou(self, box_a, box_b) -> float:
        """计算两个框的 IoU"""
        ax0, ay0, ax1, ay1 = self._box_bounds(box_a)
        bx0, by0, bx1, by1 = self._box_bounds(box_b)
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        inter_w, inter_h = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = (ax1 - ax0) * (ay1 - ay0)
        area_b = (bx1 - bx0) * (by1 - by0)
        return inter / (area_a + area_b - inter)

    def _extract_detections(self, result) -> list[dict]:
        """从 PaddleOCR 3.x 的 predict 结果中提取检测框与文本"""
        detections = []
        for res in result or []:
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
            polys = res.get("rec_polys") or res.get("dt_polys") or []
            for i, text in enumerate(texts):
                if not text or not str(text).strip():
                    continue
                box = None
                if polys and i < len(polys):
                    poly = polys[i]
                    if poly is not None and len(poly) >= 4:
                        pts = [[float(p[0]), float(p[1])] for p in poly[:4]]
                        box = pts
                detections.append(
                    {
                        "text": str(text).strip(),
                        "score": float(scores[i]) if scores and i < len(scores) else 0.0,
                        "box": box,
                    }
                )
        return detections

    def _apply_detections(self, regions: list[TextRegion], detections: list[dict]) -> None:
        """将 PaddleOCR 检测结果映射到现有 regions，未匹配的追加新 region"""
        # 若 region 数量为 0，直接用检测结果生成 region
        if not regions:
            for det in detections:
                box = det["box"] or [[0, 0], [1, 0], [1, 1], [0, 1]]
                regions.append(
                    TextRegion(box=box, text=det["text"], confidence=det["score"])
                )
            return

        # 按中心点匹配：为每个检测文本找到最近的 region
        matched = [False] * len(regions)
        for det in detections:
            if det["box"]:
                cx = sum(p[0] for p in det["box"]) / 4
                cy = sum(p[1] for p in det["box"]) / 4
                best_idx = -1
                best_dist = float("inf")
                for i, region in enumerate(regions):
                    if matched[i]:
                        continue
                    x0, y0, x1, y1 = region.bounds
                    rcx, rcy = (x0 + x1) / 2, (y0 + y1) / 2
                    dist = (cx - rcx) ** 2 + (cy - rcy) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                if best_idx >= 0:
                    matched[best_idx] = True
                    regions[best_idx].text = det["text"]
                    regions[best_idx].confidence = det["score"]
                    if det["box"]:
                        regions[best_idx].box = det["box"]
                else:
                    regions.append(
                        TextRegion(box=det["box"], text=det["text"], confidence=det["score"])
                    )
            else:
                # 无坐标，按顺序分配
                for i, region in enumerate(regions):
                    if not matched[i]:
                        matched[i] = True
                        region.text = det["text"]
                        region.confidence = det["score"]
                        break

    def _recognize_by_regions(self, image_path: Path, regions: list[TextRegion], ocr) -> None:
        """逐区域裁剪识别（PaddleOCR 未检测到时兜底）"""
        import numpy as np
        from PIL import Image

        img = Image.open(image_path)
        for region in regions:
            x0, y0, x1, y1 = region.bounds
            pad = 6
            crop = img.crop(
                (
                    max(0, x0 - pad),
                    max(0, y0 - pad),
                    min(img.width, x1 + pad),
                    min(img.height, y1 + pad),
                )
            )
            arr = np.array(crop.convert("RGB"))
            try:
                result = ocr.predict(arr)
                texts = []
                for res in result or []:
                    for t in res.get("rec_texts") or []:
                        if t and str(t).strip():
                            texts.append(str(t).strip())
                region.text = " ".join(texts).strip()
                region.confidence = 0.9
            except Exception:
                region.text = ""
                region.confidence = 0.0


class DemoOCREngine(BaseOCR):
    """Demo 模式 OCR：从区域图像中提取"文字"（模拟）

    在没有模型时，无法真实识别文字内容。为保证全流程可演示，
    此引擎基于区域非白像素密度判定是否为文字区，并填充样例对话文本，
    使翻译/渲染环节产生可见效果。接入真实 OCR 后替换为实际识别文本。
    """

    name = "demo"

    SAMPLE_JA = ["こんにちは！", "すごいね！", "行こう！", "どうしたの？", "待って！", "大丈夫？"]
    SAMPLE_EN = ["Hello there!", "That's amazing!", "Let's go!", "What happened?", "Wait!", "Are you okay?"]

    def recognize(
        self,
        image_path: Path,
        regions: list[TextRegion],
        source_lang: str = "ja",
    ) -> None:
        from PIL import Image

        img = Image.open(image_path)
        sample = self.SAMPLE_EN if source_lang == "en" else self.SAMPLE_JA
        for i, region in enumerate(regions):
            x0, y0, x1, y1 = region.bounds
            crop = img.crop((max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)))
            gray = crop.convert("L")
            px = gray.load()
            w, h = crop.size
            dark = 0
            samples = 0
            if w > 0 and h > 0:
                step = max(1, min(w, h) // 16)
                for yy in range(0, h, step):
                    for xx in range(0, w, step):
                        samples += 1
                        if px[xx, yy] < 200:
                            dark += 1
            density = dark / max(1, samples)
            if density > 0.01:
                region.text = sample[i % len(sample)]
                region.confidence = min(0.95, 0.3 + density)
            else:
                region.text = ""
                region.confidence = 0.0


def create_ocr_engine() -> BaseOCR:
    """根据配置返回 OCR 引擎。demo 模式始终可用；real 模式尝试 PaddleOCR"""
    settings = get_settings()
    if settings.pipeline_mode == "real":
        engine = PaddleOCREngine()
        if engine.available:
            return engine
    # 默认也尝试真实 OCR（若已安装且可用），否则 demo
    try:
        engine = PaddleOCREngine()
        if engine.available:
            return engine
    except Exception:  # noqa: BLE001
        pass
    return DemoOCREngine()
