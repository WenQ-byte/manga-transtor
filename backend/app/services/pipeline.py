"""翻译流水线服务：文本检测 → OCR → 翻译 → 图像修复 → 渲染"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app.models.schemas import LangCode, SourceLangCode
from app.services.language import LANGUAGE_NAMES, LanguageDetection, detect_language
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
    # 语言路由诊断：保留最终采用的引擎/语言和候选回退信息，便于排查中英文效果。
    source_lang: str = ""
    ocr_engine: str = ""
    ocr_fallback: bool = False
    ocr_fallback_reason: str = ""
    ocr_candidates: list[dict] = field(default_factory=list, repr=False, compare=False)
    ocr_route_reason: str = ""
    ocr_attempted_models: list[str] = field(default_factory=list)
    ocr_preprocess_variants: list[str] = field(default_factory=list)
    render_font: str = ""

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
    detected_source_lang: str = ""
    detection_confidence: float = 1.0
    detection_reason: str = "显式指定源语言"
    translation_failures: list[str] = field(default_factory=list)
    translation_skipped: bool = False
    region_diagnostics: list[dict] = field(default_factory=list)
    render_font: str = ""
    performance: dict = field(default_factory=dict)


_JA_RE = re.compile(r"[\u3040-\u30ff]")
_MEANINGFUL_RE = re.compile(r"[\w\u3040-\u30ff\u3400-\u9fff]")
_ENGLISH_WORD_RE = re.compile(r"\b[A-Za-z]{2,}(?:['’][A-Za-z]+)?\b")
_ENGLISH_UNIT_RE = re.compile(
    r"(?<!\w)(?:\d+(?:[.,]\d+)?)\s*(%|％|kg|g|mg|km|m|cm|mm|lb|lbs|oz|ft|in|mph|°c|°f|usd|dollars?|bucks?)(?![A-Za-z])",
    re.IGNORECASE,
)
_ENGLISH_UNIT_VARIANTS = {
    "%": ("%", "％", "百分之"),
    "％": ("%", "％", "百分之"),
    "kg": ("kg", "千克", "公斤"),
    "g": ("g", "克"),
    "mg": ("mg", "毫克"),
    "km": ("km", "千米", "公里"),
    "m": ("m", "米"),
    "cm": ("cm", "厘米"),
    "mm": ("mm", "毫米"),
    "lb": ("lb", "lbs", "磅"),
    "lbs": ("lb", "lbs", "磅"),
    "oz": ("oz", "盎司"),
    "ft": ("ft", "英尺"),
    "in": ("in", "英寸"),
    "mph": ("mph", "英里/小时"),
    "°c": ("°c", "℃", "摄氏度"),
    "°f": ("°f", "℉", "华氏度"),
    "usd": ("usd", "美元"),
    "dollar": ("dollar", "dollars", "美元"),
    "dollars": ("dollar", "dollars", "美元"),
    "buck": ("buck", "bucks", "美元"),
    "bucks": ("buck", "bucks", "美元"),
}


def _language_value(lang):
    return getattr(lang, "value", lang)


def _looks_like_name_or_abbreviation(words: list[str]) -> bool:
    if not words:
        return False
    if all(word.isupper() or len(word) <= 3 for word in words):
        return True
    return all(word[:1].isupper() and word[1:].islower() for word in words)


def preserve_decorative_symbols(source: str, translated: str) -> str:
    result = translated or ""
    for symbol in ("♡", "♥"):
        missing = (source or "").count(symbol) - result.count(symbol)
        while missing > 0 and "□" in result:
            result = result.replace("□", symbol, 1)
            missing -= 1
        if missing > 0:
            result += symbol * missing
    return result


def assess_translation_quality(
    source: str,
    translated: str,
    source_lang: LangCode,
    target_lang: LangCode,
    backend: str,
    glossary: Optional[dict[str, str]] = None,
):
    """轻量质量门控：只标记可客观判断的漏译、原文直出和低质量回退。"""
    direction = f"{LANGUAGE_NAMES.get(_language_value(source_lang), source_lang)}→{LANGUAGE_NAMES.get(_language_value(target_lang), target_lang)}"
    warnings: list[str] = []
    src = "".join(_MEANINGFUL_RE.findall(source or ""))
    dst = "".join(_MEANINGFUL_RE.findall(translated or ""))
    if backend in {"google", "mymemory", "glossary", "original"}:
        warnings.append(f"翻译使用回退后端:{backend}")
    if src and (not dst or len(dst) < max(2, int(len(src) * 0.25))):
        warnings.append("译文明显过短，可能漏译")
    if _language_value(source_lang) == "ja" and _language_value(target_lang) == "zh":
        ja_count = len(_JA_RE.findall(translated or ""))
        if ja_count >= 2 and ja_count / max(1, len(dst)) > 0.15:
            warnings.append("译文仍含较多日文")
    source_numbers = set(re.findall(r"\d+", source or ""))
    translated_numbers = set(re.findall(r"\d+", translated or ""))
    if source_numbers - translated_numbers:
        warnings.append("数字可能漏译")
    protected_symbols = set(re.findall(r"[%％$€£¥￥℃℉+#@=]", source or ""))
    if protected_symbols - set(translated or ""):
        warnings.append("关键符号可能遗漏")

    if _language_value(source_lang) == "en" and _language_value(target_lang) == "zh":
        source_words = _ENGLISH_WORD_RE.findall(source or "")
        translated_words = _ENGLISH_WORD_RE.findall(translated or "")
        normalized_source = re.sub(r"[^a-z0-9]+", "", (source or "").lower())
        normalized_translated = re.sub(r"[^a-z0-9]+", "", (translated or "").lower())
        if translated_words and len(translated_words) >= 3:
            if not _looks_like_name_or_abbreviation(translated_words):
                warnings.append("译文疑似残留大量英文")
        if (
            len(source_words) >= 3
            and normalized_source
            and normalized_source == normalized_translated
            and not _looks_like_name_or_abbreviation(source_words)
        ):
            warnings.append("译文与原文高度重复")
        if translated and re.search(
            r"```|\*\*|\[[^\]]+\]\([^)]*\)|(?:^|\n)\s*(?:译文|翻译|translation|answer|解释|分析|说明)\s*[:：]",
            translated,
            re.IGNORECASE,
        ):
            warnings.append("译文疑似包含解释、Markdown 或提示内容")
        for match in _ENGLISH_UNIT_RE.finditer(source or ""):
            unit = match.group(1).lower()
            variants = _ENGLISH_UNIT_VARIANTS.get(unit, (unit,))
            translated_lower = (translated or "").lower()
            if not any(variant.lower() in translated_lower for variant in variants):
                warnings.append("单位可能漏译")
                break
        if glossary:
            translated_folded = (translated or "").casefold()
            for term, target in glossary.items():
                if not term or not target:
                    continue
                if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", source or "", re.IGNORECASE):
                    if target.casefold() not in translated_folded:
                        warnings.append(f"词典术语可能遗漏:{target}")
    elif glossary:
        translated_folded = (translated or "").casefold()
        for term, expected in glossary.items():
            if not term or not expected:
                continue
            if _language_value(source_lang) == "en":
                present = re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", source or "", re.IGNORECASE)
            else:
                present = term in (source or "")
            if present and expected.casefold() not in translated_folded:
                warnings.append(f"词典术语可能遗漏:{expected}")
    normalized_source = re.sub(r"\s+", "", source or "").casefold()
    normalized_target = re.sub(r"\s+", "", translated or "").casefold()
    if len(normalized_source) >= 4 and normalized_source == normalized_target:
        warnings.append("译文与原文高度重复")
    target = _language_value(target_lang)
    if target == "en":
        latin_count = len(re.findall(r"[A-Za-z]", translated or ""))
        cjk_count = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", translated or ""))
        if cjk_count >= 3 and cjk_count > latin_count:
            warnings.append("英文译文仍含较多中日文字符")
    elif target == "ja":
        kana_count = len(_JA_RE.findall(translated or ""))
        han_count = len(re.findall(r"[\u3400-\u9fff]", translated or ""))
        if han_count >= 4 and kana_count == 0:
            warnings.append("日文译文疑似只有汉字，可能与中文混淆")
    if translated and re.search(
        r"```|\*\*|\[[^\]]+\]\([^)]*\)|(?:^|\n)\s*(?:译文|翻译|translation|answer|解释|分析|说明)\s*[:：]",
        translated,
        re.IGNORECASE,
    ):
        warnings.append("译文疑似包含解释、Markdown 或提示内容")
    return [f"[{direction}]{warning}" for warning in dict.fromkeys(warnings)]


def ocr_thresholds(ocr_name: str, source_lang: str | None = None) -> tuple[float, float]:
    """按识别器和语言分离翻译/擦除阈值；MIT 日文保持原有低阈值。"""
    from app.config import get_settings

    settings = get_settings()
    lang = _language_value(source_lang or "")
    if "mit48" in (ocr_name or "").lower() and lang not in {"zh", "en"}:
        return float(settings.ocr_ja_translate_threshold), float(settings.ocr_ja_erase_threshold)
    values = {
        "zh": (settings.ocr_zh_translate_threshold, settings.ocr_zh_erase_threshold),
        "en": (settings.ocr_en_translate_threshold, settings.ocr_en_erase_threshold),
    }
    if lang in values:
        return tuple(float(v) for v in values[lang])
    return 0.50, 0.50


def region_diagnostics(regions: list[TextRegion]) -> list[dict]:
    """返回可写入任务 meta 的轻量区域诊断，不包含图像或完整原文。"""
    return [
        {
            "index": index,
            "source_lang": r.source_lang or "",
            "ocr_engine": r.ocr_engine or "",
            "confidence": round(float(r.confidence or 0.0), 4),
            "ocr_fallback": bool(r.ocr_fallback),
            "fallback_reason": r.ocr_fallback_reason or "",
            "text": (r.text or "")[:160],
            "candidate_count": len(r.ocr_candidates or []),
            "route_reason": r.ocr_route_reason or "",
            "attempted_models": list(r.ocr_attempted_models or []),
            "preprocess_variants": list(r.ocr_preprocess_variants or []),
            "bounds": list(r.bounds),
            "group_index": r.group_index,
            "group_bounds": list(r.group_bounds) if r.group_bounds else None,
        }
        for index, r in enumerate(regions, start=1)
    ]


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

    def _ensure_ocr_metadata(self, regions: list[TextRegion], fallback_lang: str) -> None:
        engine = getattr(self.ocr, "name", type(self.ocr).__name__)
        for region in regions:
            if not region.source_lang or region.source_lang == "auto":
                region.source_lang = fallback_lang
            if not region.ocr_engine:
                region.ocr_engine = engine

    def _performance_snapshot(self, stage_ms, total_ms, translation_perf=None) -> dict:
        ocr_perf = dict(getattr(self.ocr, "last_performance", {}) or {})
        translation_perf = dict(translation_perf or {})
        measured = sum(int(stage_ms.get(name, 0)) for name in (
            "detection_ms", "ocr_ms", "inpaint_ms", "grouping_ms", "translation_ms", "render_ms"
        ))
        return {
            **{name: int(value) for name, value in stage_ms.items()},
            "model_load_ms": int(ocr_perf.get("model_load_ms", 0)),
            "ocr_inference_ms": int(ocr_perf.get("inference_ms", 0)),
            "ocr_call_count": int(ocr_perf.get("call_count", 0)),
            "ocr_models": list(ocr_perf.get("models", [])),
            "ocr_preprocess_variants": list(ocr_perf.get("variants", [])),
            "ocr_model_reuse_count": int(ocr_perf.get("model_reuse_count", 0)),
            "ocr_fallback_count": int(ocr_perf.get("fallback_count", 0)),
            "ocr_requested_device": str(ocr_perf.get("requested_device", "")),
            "ocr_device": str(ocr_perf.get("device", "")),
            "ocr_device_fallback": bool(ocr_perf.get("device_fallback", False)),
            "ocr_device_fallback_reason": str(ocr_perf.get("device_fallback_reason", "")),
            "translation_request_count": int(translation_perf.get("request_count", 0)),
            "translation_cache_hits": int(translation_perf.get("cache_hits", 0)),
            "translation_fallback": bool(translation_perf.get("fallback", False)),
            "translation_backends_attempted": list(translation_perf.get("backends_attempted", [])),
            "other_ms": max(0, int(total_ms) - measured),
            "total_ms": int(total_ms),
        }

    def _group_regions(
        self,
        image_path: Path,
        regions: list[TextRegion],
        boundary_image_path: Path | None = None,
    ) -> list[dict]:
        """按气泡泛洪填充把 region 分组（组内按阅读顺序），并回填 group_index/group_bounds"""
        from PIL import Image

        import numpy as np

        from app.services.engines.bubble import group_regions_by_bubble

        img = np.array(Image.open(image_path).convert("RGB"))
        bgr = img[:, :, ::-1].copy()
        h, w = bgr.shape[:2]
        boundary_bgr = None
        if boundary_image_path is not None:
            boundary = np.array(Image.open(boundary_image_path).convert("RGB"))
            if boundary.shape[:2] == img.shape[:2]:
                boundary_bgr = boundary[:, :, ::-1].copy()
        return group_regions_by_bubble(
            bgr,
            regions,
            w,
            h,
            boundary_bgr=boundary_bgr,
        )

    def translate_image(
        self,
        image_path: Path,
        source_lang: SourceLangCode,
        target_lang: LangCode,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> PipelineResult:
        from app.config import get_settings

        start = time.monotonic()
        stage_ms = {
            "detection_ms": 0,
            "ocr_ms": 0,
            "inpaint_ms": 0,
            "grouping_ms": 0,
            "translation_ms": 0,
            "render_ms": 0,
        }
        translation_perf = {
            "request_count": 0,
            "cache_hits": 0,
            "fallback": False,
            "backends_attempted": [],
        }
        settings = get_settings()

        def stage_start(name: str) -> float:
            started = time.monotonic()
            print(f"[pipeline] {name} 开始", flush=True)
            return started

        def stage_end(name: str, key: str, started: float) -> None:
            elapsed = int((time.monotonic() - started) * 1000)
            stage_ms[key] += elapsed
            print(f"[pipeline] {name} 结束：{elapsed}ms", flush=True)

        requested_source = _language_value(source_lang)
        # 新路由器需要收到 auto 才能做区域级候选比较；旧版单语言 OCR 仍使用稳定回退。
        if requested_source == "auto" and getattr(self.ocr, "supports_language_routing", False):
            ocr_source = "auto"
        else:
            ocr_source = settings.auto_source_fallback if requested_source == "auto" else requested_source

        def empty_result(actual: str | None = None, confidence: float | None = None, reason: str | None = None):
            if requested_source == "auto" and actual is None:
                empty_detection = detect_language("", fallback=settings.auto_source_fallback)
                actual = empty_detection.language
                confidence = empty_detection.confidence
                reason = empty_detection.reason
            total_ms = int((time.monotonic() - start) * 1000)
            return PipelineResult(
                regions=[],
                duration_ms=total_ms,
                ocr_backend=getattr(self.ocr, "name", type(self.ocr).__name__),
                detected_source_lang=actual or requested_source,
                detection_confidence=1.0 if confidence is None else confidence,
                detection_reason=reason or "显式指定源语言",
                performance=self._performance_snapshot(stage_ms, total_ms, translation_perf),
            )

        # 1. 检测（OCR 引擎自带检测时，直接由 OCR 完成检测+识别）
        if getattr(self.ocr, "supports_detection", False):
            self._report(progress_cb, 0, 100)
            regions: list[TextRegion] = []
            self._report(progress_cb, 1, 10)
            phase_started = stage_start("ocr")
            self.ocr.recognize(image_path, regions, ocr_source)
            stage_end("ocr", "ocr_ms", phase_started)
            self._ensure_ocr_metadata(regions, ocr_source)
            self._report(progress_cb, 1, 100)
            if not regions:
                return empty_result()
        else:
            self._report(progress_cb, 0, 10)
            phase_started = stage_start("detect")
            regions = self.detector.detect(image_path)
            stage_end("detect", "detection_ms", phase_started)
            self._report(progress_cb, 0, 100)

            if not regions:
                return empty_result()

            # 2. OCR
            self._report(progress_cb, 1, 10)
            phase_started = stage_start("ocr")
            self.ocr.recognize(image_path, regions, ocr_source)
            stage_end("ocr", "ocr_ms", phase_started)
            self._ensure_ocr_metadata(regions, ocr_source)
            self._report(progress_cb, 1, 100)

        if requested_source == "auto":
            # 路由器已经给出区域级语言；以置信度加权多数作为页面翻译方向，
            # 同时保留每个区域的 source_lang，避免整页一次判断吞掉混合语言信息。
            nonempty = [r for r in regions if r.text.strip()]
            weighted: dict[str, float] = {}
            for region in nonempty:
                lang = region.source_lang or detect_language(region.text, settings.auto_source_fallback).language
                weighted[lang] = weighted.get(lang, 0.0) + max(0.05, float(region.confidence or 0.0))
            if weighted:
                actual_source = max(weighted, key=weighted.get)
                detection = LanguageDetection(
                    actual_source,
                    max(0.45, min(0.99, weighted[actual_source] / max(0.1, sum(weighted.values())) + 0.35)),
                    f"区域级 OCR 路由：{actual_source}（{len(nonempty)} 个区域）",
                )
            else:
                detection = detect_language("", fallback=settings.auto_source_fallback)
                actual_source = detection.language
        else:
            actual_source = requested_source
            detection = None

        detection_confidence = detection.confidence if detection else 1.0
        detection_reason = detection.reason if detection else "显式指定源语言"
        target_value = _language_value(target_lang)
        auto_has_other_language = requested_source == "auto" and any(
            r.text.strip() and r.source_lang and r.source_lang != target_value for r in regions
        )
        if actual_source == target_value and not auto_has_other_language:
            from PIL import Image
            import io

            output = io.BytesIO()
            Image.open(image_path).convert("RGB").save(output, format="PNG")
            total_ms = int((time.monotonic() - start) * 1000)
            return PipelineResult(
                regions=[],
                duration_ms=total_ms,
                image_bytes=output.getvalue(),
                ocr_backend=getattr(self.ocr, "name", type(self.ocr).__name__),
                quality_warnings=[f"[{LANGUAGE_NAMES.get(actual_source)}→{LANGUAGE_NAMES.get(target_lang)}]源语言与目标语言相同，已保留原图"],
                detected_source_lang=actual_source,
                detection_confidence=detection_confidence,
                detection_reason=detection_reason,
                translation_skipped=True,
                performance=self._performance_snapshot(stage_ms, total_ms, translation_perf),
            )

        # 1.4 翻译与擦除阈值解耦：MIT 检测框/逐像素 mask 很可靠，但字符概率常仅 0.2~0.6；
        # 低概率行仍需擦除，并保留可读文本参与整块翻译。Paddle/CV 继续使用保守阈值。
        text_regions = [
            r for r in regions
            if not (requested_source == "auto" and r.source_lang == target_value)
            and r.confidence >= ocr_thresholds(
                r.ocr_engine or getattr(self.ocr, "name", ""),
                r.source_lang or actual_source,
            )[0]
            and r.text.strip()
        ]
        text_region_ids = {id(r) for r in text_regions}
        erase_only_regions = [
            r
            for r in regions
            if id(r) not in text_region_ids
            and not (requested_source == "auto" and r.source_lang == target_value)
            and r.confidence >= ocr_thresholds(
                r.ocr_engine or getattr(self.ocr, "name", ""),
                r.source_lang or actual_source,
            )[1]
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
            return empty_result(actual_source, detection_confidence, detection_reason)

        # 1.6 非气泡文字（页眉横条/拟声词，泛洪几何判定）保留原文：不擦除、不翻译
        from PIL import Image

        import numpy as np

        from app.services.engines.bubble import classify_non_bubble

        img0 = np.array(Image.open(image_path).convert("RGB"))
        bgr0 = img0[:, :, ::-1].copy()
        h0, w0 = bgr0.shape[:2]
        for r in text_regions + erase_only_regions:
            region_source = r.source_lang or actual_source
            if preserve_latin_label(r, region_source) or classify_non_bubble(
                bgr0, r, w0, h0, source_lang=region_source
            ):
                r._no_erase = True
        text_regions = drop_non_bubble_regions(text_regions)
        erase_only_regions = drop_non_bubble_regions(erase_only_regions)
        erase_regions = text_regions + erase_only_regions
        if not erase_regions:
            return empty_result(actual_source, detection_confidence, detection_reason)

        # 3. 图像修复（擦除原文）—— 提前到翻译前，使气泡分组可在干净图上进行
        self._report(progress_cb, 2, 10)
        phase_started = stage_start("inpaint")
        cleaned = self.inpainter.inpaint(image_path, erase_regions)
        stage_end("inpaint", "inpaint_ms", phase_started)
        self._report(progress_cb, 2, 100)

        # 4. 按气泡分组（干净图上泛洪，笔画已擦除 → 分组可靠）→ 整块翻译
        self._report(progress_cb, 3, 5)
        phase_started = stage_start("bubble grouping")
        groups = self._group_regions(
            cleaned,
            text_regions,
            boundary_image_path=image_path,
        )
        stage_end("bubble grouping", "grouping_ms", phase_started)
        group_texts = ["\n".join(r.text for r in g["regions"] if r.text.strip()) for g in groups]
        group_languages: list[str] = []
        for group in groups:
            weights: dict[str, float] = {}
            for region in group["regions"]:
                lang = region.source_lang or actual_source
                weights[lang] = weights.get(lang, 0.0) + max(0.05, float(region.confidence or 0.0))
            group_languages.append(max(weights, key=weights.get) if weights else actual_source)

        translated_blocks = [""] * len(groups)
        backend_names = [""] * len(groups)
        group_glossaries: list[dict[str, str]] = [{} for _ in groups]
        translation_failures: list[str] = []
        language_buckets: dict[str, list[int]] = {}
        for index, language in enumerate(group_languages):
            # auto 混排中与目标语言相同的气泡无需擦除/回写；保留原文最安全。
            if requested_source == "auto" and language == target_value:
                translated_blocks[index] = group_texts[index]
                backend_names[index] = "original"
                continue
            language_buckets.setdefault(language, []).append(index)

        bucket_items = list(language_buckets.items())
        phase_started = stage_start("translate")
        for bucket_pos, (language, indices) in enumerate(bucket_items):
            texts = [group_texts[index] for index in indices]
            glossary = self.glossary.get_mapping(language, target_value)
            if language == "en" and target_value == "zh":
                from app.services.engines.translator import expand_english_glossary_aliases

                glossary = expand_english_glossary_aliases(texts, glossary)
            for index in indices:
                group_glossaries[index] = glossary
            start_progress = bucket_pos / max(1, len(bucket_items))
            span = 1 / max(1, len(bucket_items))
            outputs = self.translator.translate_batch(
                texts,
                language,
                target_lang,
                glossary=glossary,
                progress_cb=lambda p, start_progress=start_progress, span=span: self._report(
                    progress_cb, 3, 5 + int((start_progress + p * span) * 0.95)
                ),
            )
            bucket_perf = dict(getattr(self.translator, "last_performance", {}) or {})
            translation_perf["request_count"] += int(bucket_perf.get("request_count", 0))
            translation_perf["cache_hits"] += int(bucket_perf.get("cache_hits", 0))
            translation_perf["fallback"] = bool(translation_perf["fallback"] or bucket_perf.get("fallback", False))
            for attempted in bucket_perf.get("backends_attempted", []):
                if attempted not in translation_perf["backends_attempted"]:
                    translation_perf["backends_attempted"].append(attempted)
            names = list(getattr(self.translator, "last_backend_names", []))
            fallback_name = getattr(self.translator, "name", type(self.translator).__name__)
            if len(names) != len(indices):
                names = [fallback_name] * len(indices)
            for local_index, group_index in enumerate(indices):
                translated_blocks[group_index] = outputs[local_index] if local_index < len(outputs) else ""
                backend_names[group_index] = names[local_index]
            translation_failures.extend(getattr(self.translator, "last_failures", []))
        page_warnings: list[str] = []
        for group_idx, (g, block) in enumerate(zip(groups, translated_blocks)):
            block = block or ""
            if not block.strip():
                # 翻译失败/空结果：回退原文整块，避免文字消失
                block = "\n".join(r.text for r in g["regions"])
            source_block = group_texts[group_idx]
            block = preserve_decorative_symbols(source_block, block)
            backend_name = backend_names[group_idx]
            warnings = assess_translation_quality(
                source_block,
                block,
                group_languages[group_idx],
                target_lang,
                backend_name,
                glossary=group_glossaries[group_idx],
            )
            page_warnings.extend(f"气泡{group_idx + 1}:{warning}" for warning in warnings)
            lines = block.split("\n")
            last = lines[-1] if lines else ""
            for i, r in enumerate(g["regions"]):
                r.group_translated = block
                r.translated = lines[i] if i < len(lines) else last
                r.translation_backend = backend_name
                r.quality_warnings = list(warnings)
        stage_end("translate", "translation_ms", phase_started)
        self._report(progress_cb, 3, 100)

        # 5. 渲染译文
        self._report(progress_cb, 4, 10)
        phase_started = stage_start("render")
        result_bytes = self.renderer.render(cleaned, text_regions, target_lang=target_lang)
        stage_end("render", "render_ms", phase_started)
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
            detected_source_lang=actual_source,
            detection_confidence=detection_confidence,
            detection_reason=detection_reason,
            translation_failures=translation_failures,
            region_diagnostics=region_diagnostics(regions),
            render_font=next((r.render_font for r in text_regions if r.render_font), ""),
            performance=self._performance_snapshot(stage_ms, duration, translation_perf),
        )


def create_pipeline() -> TranslationPipeline:
    return TranslationPipeline()
