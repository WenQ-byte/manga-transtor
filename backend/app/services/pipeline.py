"""翻译流水线服务：文本检测 → OCR → 翻译 → 图像修复 → 渲染"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.models.schemas import LangCode
from app.services.glossary_service import GlossaryService

# 流水线步骤定义（顺序）
PIPELINE_STEPS = [
    ("detect", "检测文本区域"),
    ("ocr", "识别文字"),
    ("translate", "翻译"),
    ("inpaint", "修复图像"),
    ("render", "渲染译文"),
]

ProgressCallback = Callable[[str, int], None]


@dataclass
class TextRegion:
    """检测到的文本区域"""

    box: list[list[int]]  # 四个角点 [[x,y],...]
    text: str = ""  # OCR结果
    translated: str = ""  # 翻译结果
    confidence: float = 0.0
    poly: Optional[list[list[float]]] = None  # OCR原始检测多边形点的完整列表（可选）
    mask: Optional[object] = None  # 预计算的笔画掩膜（numpy数组，由 mask.build_full_mask 填写）
    direction: Optional[str] = None  # 横/竖排: "h" | "v"（MIT 检测器给出）
    fg_color: Optional[tuple[int, int, int]] = None  # 字符前景色（MIT 48px OCR 预测）
    bg_color: Optional[tuple[int, int, int]] = None  # 字符背景色（MIT 48px OCR 预测）

    # 内部：MIT 检测器附加的 Quadrilateral、OCR 附加的内部状态
    _quad: Optional[object] = field(default=None, repr=False, compare=False)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


@dataclass
class PipelineResult:
    """流水线执行结果"""

    regions: list[TextRegion] = field(default_factory=list)
    duration_ms: int = 0
    image_bytes: Optional[bytes] = None


class TranslationPipeline:
    """翻译流水线编排器，各步骤通过引擎实现，可插拔"""

    def __init__(self, *, detector=None, ocr=None, translator=None, inpainter=None, renderer=None):
        from app.config import get_settings
        from app.services.engines import get_engine
        self.detector = detector or get_engine("detector")
        self.ocr = ocr or get_engine("ocr")
        self.translator = translator or get_engine("translator")
        self.inpainter = inpainter or get_engine("inpainter")
        self.renderer = renderer or get_engine("renderer")
        self.glossary = GlossaryService()
        from app.services.engines.bubble import create_bubble_filter

        self.bubble_filter = create_bubble_filter()
        bf = get_settings().bubble_filter
        if bf == "off":
            self._bubble_on = False
        elif bf == "on":
            self._bubble_on = True
        else:  # auto
            self._bubble_on = getattr(self.detector, "name", "") != "manga"

    def _report(self, cb: Optional[ProgressCallback], step_index: int, progress: int) -> None:
        if cb:
            cb(PIPELINE_STEPS[step_index][0], progress)

    def translate_image(
        self,
        image_path: Path,
        source_lang: LangCode,
        target_lang: LangCode,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> PipelineResult:
        start = time.monotonic()

        # 1. 检测（OCR 引擎自带检测时，直接由 OCR 完成检测+识别）
        if getattr(self.ocr, "supports_detection", False):
            self._report(progress_cb, 0, 100)
            regions: list[TextRegion] = []
            self._report(progress_cb, 1, 10)
            self.ocr.recognize(image_path, regions, source_lang)
            self._report(progress_cb, 1, 100)
            if not regions:
                return PipelineResult(regions=[], duration_ms=int((time.monotonic() - start) * 1000))
        else:
            self._report(progress_cb, 0, 10)
            regions = self.detector.detect(image_path)
            self._report(progress_cb, 0, 100)

            if not regions:
                return PipelineResult(regions=[], duration_ms=int((time.monotonic() - start) * 1000))

            # 2. OCR
            self._report(progress_cb, 1, 10)
            self.ocr.recognize(image_path, regions, source_lang)
            self._report(progress_cb, 1, 100)

        # 1.4 过滤低置信度识别与空文本（噪声）
        regions = [r for r in regions if r.confidence >= 0.5 and r.text.strip()]

        # 1.5 过滤气泡外文字（丢弃涂鸦/噪声），无有效气泡时保留原样
        # MIT 检测引擎自带漫画文本先验，且白占比启发式对彩色/深色气泡误删，跳过
        bf = getattr(self, "_bubble_on", True)
        if bf:
            regions = self.bubble_filter.filter(image_path, regions)
        if not regions:
            return PipelineResult(regions=[], duration_ms=int((time.monotonic() - start) * 1000))

        # 3. 翻译（应用专有名词词典）
        glossary = self.glossary.get_mapping(source_lang)
        self._report(progress_cb, 2, 5)
        texts = [r.text for r in regions]
        translated = self.translator.translate_batch(
            texts, source_lang, target_lang, glossary=glossary, progress_cb=lambda p: self._report(progress_cb, 2, 5 + int(p * 0.95))
        )
        for region, text in zip(regions, translated):
            region.translated = text
        self._report(progress_cb, 2, 100)

        # 4. 图像修复（擦除原文）
        self._report(progress_cb, 3, 10)
        cleaned = self.inpainter.inpaint(image_path, regions)
        self._report(progress_cb, 3, 100)

        # 5. 渲染译文
        self._report(progress_cb, 4, 10)
        result_bytes = self.renderer.render(cleaned, regions, target_lang=target_lang)
        self._report(progress_cb, 4, 100)

        duration = int((time.monotonic() - start) * 1000)
        return PipelineResult(regions=regions, duration_ms=duration, image_bytes=result_bytes)


def create_pipeline() -> TranslationPipeline:
    return TranslationPipeline()
