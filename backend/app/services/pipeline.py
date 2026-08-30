"""翻译流水线服务：文本检测 → OCR → 翻译 → 图像修复 → 渲染"""
from __future__ import annotations

import re
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
    ("inpaint", "修复图像"),
    ("translate", "翻译"),
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

    # 气泡级分组（pipeline 按气泡分组翻译后回填）
    group_index: Optional[int] = None  # 同一气泡的 region 共享
    group_bounds: Optional[tuple[int, int, int, int]] = None  # 气泡包围盒 (x0,y0,x1,y1)
    group_translated: str = ""  # 整块气泡译文（renderer 优先用此排版）
    group_mask: Optional[object] = field(default=None, repr=False, compare=False)  # 分组时确认的容器掩膜
    group_mask_reliable: bool = field(default=False, repr=False, compare=False)
    translation_backend: str = ""
    quality_warnings: list[str] = field(default_factory=list)

    # 内部：MIT 检测器附加的 Quadrilateral、OCR 附加的内部状态
    _quad: Optional[object] = field(default=None, repr=False, compare=False)
    _no_erase: bool = field(default=False, repr=False, compare=False)  # mit_ignore_bubble 判定时跳过擦除

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def drop_non_bubble_regions(regions: list[TextRegion]) -> list[TextRegion]:
    """移出非气泡文字（刊头/拟声词，标 _no_erase）：原文保留，不翻译不渲染"""
    return [r for r in regions if not getattr(r, "_no_erase", False)]


def preserve_latin_label(region: TextRegion, source_lang: LangCode) -> bool:
    """日语漫画中的高比例拉丁文本通常是歌名、艺名或装饰字，默认保留原文。"""
    lang = getattr(source_lang, "value", source_lang)
    if lang != "ja":
        return False
    chars = [c for c in (region.text or "") if not c.isspace()]
    latin = [c for c in chars if c.isascii() and (c.isalpha() or c.isdigit() or c in "-'&")]
    return len(latin) >= 4 and len(latin) / max(1, len(chars)) >= 0.7


@dataclass
class PipelineResult:
    """流水线执行结果"""

    regions: list[TextRegion] = field(default_factory=list)
    duration_ms: int = 0
    image_bytes: Optional[bytes] = None
    ocr_backend: str = ""
    translation_backends: list[str] = field(default_factory=list)
    quality_warnings: list[str] = field(default_factory=list)


_JA_RE = re.compile(r"[\u3040-\u30ff]")
_MEANINGFUL_RE = re.compile(r"[\w\u3040-\u30ff\u3400-\u9fff]")


def assess_translation_quality(source: str, translated: str, source_lang: LangCode, target_lang: LangCode, backend: str):
    """轻量质量门控：只标记可客观判断的漏译、原文直出和低质量回退。"""
    warnings: list[str] = []
    src = "".join(_MEANINGFUL_RE.findall(source or ""))
    dst = "".join(_MEANINGFUL_RE.findall(translated or ""))
    if backend in {"google", "mymemory", "glossary", "original"}:
        warnings.append(f"翻译使用回退后端:{backend}")
    if src and (not dst or len(dst) < max(2, int(len(src) * 0.25))):
        warnings.append("译文明显过短，可能漏译")
    if source_lang == "ja" and target_lang == "zh":
        ja_count = len(_JA_RE.findall(translated or ""))
        if ja_count >= 2 and ja_count / max(1, len(dst)) > 0.15:
            warnings.append("译文仍含较多日文")
    source_numbers = set(re.findall(r"\d+", source or ""))
    translated_numbers = set(re.findall(r"\d+", translated or ""))
    if source_numbers - translated_numbers:
        warnings.append("数字可能漏译")
    return warnings


def ocr_thresholds(ocr_name: str) -> tuple[float, float]:
    """MIT 检测框可靠但字符置信度偏保守；翻译与擦除使用不同阈值。"""
    if "mit48" in (ocr_name or "").lower():
        return 0.20, 0.0
    return 0.50, 0.50


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

    def _group_regions(self, image_path: Path, regions: list[TextRegion]) -> list[dict]:
        """按气泡泛洪填充把 region 分组（组内按阅读顺序），并回填 group_index/group_bounds"""
        from PIL import Image

        import numpy as np

        from app.services.engines.bubble import group_regions_by_bubble

        img = np.array(Image.open(image_path).convert("RGB"))
        bgr = img[:, :, ::-1].copy()
        h, w = bgr.shape[:2]
        return group_regions_by_bubble(bgr, regions, w, h)

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

        # 1.4 翻译与擦除阈值解耦：MIT 检测框/逐像素 mask 很可靠，但字符概率常仅 0.2~0.6；
        # 低概率行仍需擦除，并保留可读文本参与整块翻译。Paddle/CV 继续使用保守阈值。
        translate_threshold, erase_threshold = ocr_thresholds(getattr(self.ocr, "name", ""))
        text_regions = [
            r for r in regions if r.confidence >= translate_threshold and r.text.strip()
        ]
        text_region_ids = {id(r) for r in text_regions}
        erase_only_regions = [
            r
            for r in regions
            if id(r) not in text_region_ids
            and r.confidence >= erase_threshold
            and (r.mask is not None or not r.text.strip())
        ]

        from app.services.engines.bubble import merge_punctuation_regions

        text_regions = merge_punctuation_regions(text_regions)

        # 1.5 过滤气泡外文字（丢弃涂鸦/噪声），无有效气泡时保留原样
        # MIT 检测引擎自带漫画文本先验，且白占比启发式对彩色/深色气泡误删，跳过
        bf = getattr(self, "_bubble_on", True)
        if bf:
            text_regions = self.bubble_filter.filter(image_path, text_regions)
        if not text_regions and not erase_only_regions:
            return PipelineResult(regions=[], duration_ms=int((time.monotonic() - start) * 1000))

        # 1.6 非气泡文字（页眉横条/拟声词，泛洪几何判定）保留原文：不擦除、不翻译
        from PIL import Image

        import numpy as np

        from app.services.engines.bubble import classify_non_bubble

        img0 = np.array(Image.open(image_path).convert("RGB"))
        bgr0 = img0[:, :, ::-1].copy()
        h0, w0 = bgr0.shape[:2]
        for r in text_regions + erase_only_regions:
            if preserve_latin_label(r, source_lang) or classify_non_bubble(bgr0, r, w0, h0):
                r._no_erase = True
        text_regions = drop_non_bubble_regions(text_regions)
        erase_only_regions = drop_non_bubble_regions(erase_only_regions)
        erase_regions = text_regions + erase_only_regions
        if not erase_regions:
            return PipelineResult(regions=[], duration_ms=int((time.monotonic() - start) * 1000))

        # 3. 图像修复（擦除原文）—— 提前到翻译前，使气泡分组可在干净图上进行
        self._report(progress_cb, 2, 10)
        cleaned = self.inpainter.inpaint(image_path, erase_regions)
        self._report(progress_cb, 2, 100)

        # 4. 按气泡分组（干净图上泛洪，笔画已擦除 → 分组可靠）→ 整块翻译
        glossary = self.glossary.get_mapping(source_lang)
        self._report(progress_cb, 3, 5)
        groups = self._group_regions(cleaned, text_regions)
        group_texts = ["\n".join(r.text for r in g["regions"] if r.text.strip()) for g in groups]
        translated_blocks = self.translator.translate_batch(
            group_texts,
            source_lang,
            target_lang,
            glossary=glossary,
            progress_cb=lambda p: self._report(progress_cb, 3, 5 + int(p * 0.95)),
        )
        backend_names = list(getattr(self.translator, "last_backend_names", []))
        if len(backend_names) != len(groups):
            fallback_name = getattr(self.translator, "name", type(self.translator).__name__)
            backend_names = [fallback_name] * len(groups)
        page_warnings: list[str] = []
        for group_idx, (g, block) in enumerate(zip(groups, translated_blocks)):
            block = block or ""
            if not block.strip():
                # 翻译失败/空结果：回退原文整块，避免文字消失
                block = "\n".join(r.text for r in g["regions"])
            source_block = group_texts[group_idx]
            backend_name = backend_names[group_idx]
            warnings = assess_translation_quality(source_block, block, source_lang, target_lang, backend_name)
            page_warnings.extend(f"气泡{group_idx + 1}:{warning}" for warning in warnings)
            lines = block.split("\n")
            last = lines[-1] if lines else ""
            for i, r in enumerate(g["regions"]):
                r.group_translated = block
                r.translated = lines[i] if i < len(lines) else last
                r.translation_backend = backend_name
                r.quality_warnings = list(warnings)
        self._report(progress_cb, 3, 100)

        # 5. 渲染译文
        self._report(progress_cb, 4, 10)
        result_bytes = self.renderer.render(cleaned, text_regions, target_lang=target_lang)
        self._report(progress_cb, 4, 100)

        duration = int((time.monotonic() - start) * 1000)
        print(
            f"[pipeline] ocr={getattr(self.ocr, 'name', type(self.ocr).__name__)} "
            f"translate={','.join(sorted(set(backend_names)))} warnings={len(page_warnings)}"
        )
        if page_warnings:
            print("[quality] " + " | ".join(page_warnings))
        return PipelineResult(
            regions=text_regions,
            duration_ms=duration,
            image_bytes=result_bytes,
            ocr_backend=getattr(self.ocr, "name", type(self.ocr).__name__),
            translation_backends=backend_names,
            quality_warnings=page_warnings,
        )


def create_pipeline() -> TranslationPipeline:
    return TranslationPipeline()
