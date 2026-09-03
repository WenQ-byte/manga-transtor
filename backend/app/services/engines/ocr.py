"""OCR 引擎：优先使用真实 OCR（PaddleOCR），未安装时降级为 demo 模式"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import get_settings
from app.services.engines.base import BaseOCR
from app.services.language import region_language_hint
from app.services.pipeline import TextRegion


_PADDLE_DLL_HANDLES = []


def _has_japanese_script(text: str) -> bool:
    return any("\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff" for ch in (text or ""))


def _looks_like_japanese_ocr_garbage(text: str, confidence: float, page_hint: str) -> bool:
    """日文页面中，MIT48 输出非日文字符时判定为疑似误识别，交给 manga-ocr 复核。"""
    value = (text or "").strip()
    if page_hint != "ja" or not value or _has_japanese_script(value):
        return False
    letters = re.findall(r"[A-Za-z]+", value)
    # 清晰、较长的英文单词可能是漫画中的英文标签，不要误送日文模型。
    if any(len(word) >= 4 for word in letters) and confidence >= 0.70:
        return False
    return confidence < 0.70 or not letters or all(len(word) <= 3 for word in letters)


def _register_paddle_gpu_dll_directories() -> None:
    if os.name != "nt" or _PADDLE_DLL_HANDLES:
        return
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    candidates = [
        nvidia_root / "cu13" / "bin" / "x86_64",
        nvidia_root / "cudnn" / "bin",
    ]
    existing = [path for path in candidates if path.is_dir()]
    if not existing:
        return
    current_path = os.environ.get("PATH", "")
    for path in existing:
        value = str(path)
        if hasattr(os, "add_dll_directory"):
            _PADDLE_DLL_HANDLES.append(os.add_dll_directory(value))
        if value.lower() not in current_path.lower():
            current_path = value + os.pathsep + current_path
    os.environ["PATH"] = current_path


def _horizontal_ink_split_boxes(image: np.ndarray, region: TextRegion) -> list[tuple[int, int, int, int]]:
    """按异常宽的竖向空白带拆分跨气泡英文行；普通词间空格不拆。"""
    if getattr(region, "direction", None) not in {None, "h"}:
        return []
    x0, y0, x1, y1 = region.bounds
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
    width, height = x1 - x0, y1 - y0
    text = (region.text or "").strip()
    if width < 80 or height < 8 or width / max(1, height) < 6.0 or text.count(" ") < 2:
        return []
    crop = image[y0:y1, x0:x1]
    if crop.ndim == 3:
        gray = np.round(crop[..., :3].mean(axis=2)).astype(np.uint8)
    else:
        gray = crop.astype(np.uint8)
    active = (gray < 170).sum(axis=0) >= max(1, int(round(height * 0.08)))
    if not active.any():
        return []
    # 去掉单列噪声，但不填词间空格。
    active = np.convolve(active.astype(np.uint8), np.ones(3, np.uint8), mode="same") >= 2
    inactive = ~active
    starts = np.flatnonzero(inactive & np.r_[True, ~inactive[:-1]])
    ends = np.flatnonzero(inactive & np.r_[~inactive[1:], True]) + 1
    margin = max(4, int(round(height * 0.45)))
    gaps = [
        (start, end)
        for start, end in zip(starts, ends)
        if start >= margin and end <= width - margin
    ]
    if not gaps:
        return []
    gap_start, gap_end = max(gaps, key=lambda item: item[1] - item[0])
    min_gap = max(8, int(round(height * 0.55)))
    if gap_end - gap_start < min_gap:
        return []
    left_active = np.flatnonzero(active[:gap_start])
    right_active = np.flatnonzero(active[gap_end:])
    if left_active.size == 0 or right_active.size == 0:
        return []
    left_width = int(left_active[-1] - left_active[0] + 1)
    right_width = int(right_active[-1] - right_active[0] + 1)
    if min(left_width, right_width) < max(16, int(round(height * 1.4))):
        return []
    pad = min(3, max(1, (gap_end - gap_start) // 4))
    return [
        (int(x0), int(y0), int(min(x1, x0 + gap_start + pad)), int(y1)),
        (int(max(x0, x0 + gap_end - pad)), int(y0), int(x1), int(y1)),
    ]


def _english_word_count(text: str) -> int:
    """统计可独立复识别的英文词，避免把普通词间空格当作跨气泡间隙。"""
    import re

    return len(re.findall(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*", text or ""))


class PaddleOCREngine(BaseOCR):
    """基于 PaddleOCR 的真实 OCR（需要安装 paddleocr + paddlepaddle）

    PaddleOCR 自带文本检测+识别，直接对整图处理，效果远优于启发式检测。
    """

    name = "paddle"

    supports_detection = True

    def __init__(self):
        self.settings = get_settings()
        self._ocrs: dict[str, object] = {}
        self._device_by_lang: dict[str, str] = {}
        self._device_fallback_reasons: list[str] = []
        self._load_error: str = ""
        self._PaddleOCR = None
        self._requested_device = self._normalize_device(
            getattr(self.settings, "paddle_device", "gpu:0")
        )
        self._preferred_device = "cpu"
        self._lang_map = {"ja": "japan", "en": "en", "zh": "ch"}
        self._cached_image_path: Path | None = None
        self._cached_image: Image.Image | None = None
        self.reset_performance()
        self._load()

    def reset_performance(self) -> None:
        self.last_performance = {
            "model_load_ms": 0,
            "inference_ms": 0,
            "call_count": 0,
            "models": [],
            "variants": [],
            "model_reuse_count": 0,
            "fallback_count": 0,
            "requested_device": getattr(self, "_requested_device", "gpu:0"),
            "device": self._device_summary(),
            "device_fallback": bool(getattr(self, "_device_fallback_reasons", [])),
            "device_fallback_reason": "; ".join(getattr(self, "_device_fallback_reasons", [])),
        }

    @staticmethod
    def _normalize_device(value: str) -> str:
        requested = (value or "gpu:0").strip().lower()
        if requested in {"gpu", "cuda"}:
            return "gpu:0"
        if requested.startswith("cuda:"):
            requested = "gpu:" + requested.split(":", 1)[1]
        if requested == "cpu":
            return requested
        if requested.startswith("gpu:") and requested.split(":", 1)[1].isdigit():
            return requested
        return "cpu"

    def _device_summary(self) -> str:
        devices = list(dict.fromkeys(getattr(self, "_device_by_lang", {}).values()))
        if devices:
            return ",".join(devices)
        if getattr(self, "_PaddleOCR", None) is None:
            return "unavailable"
        return getattr(self, "_preferred_device", "cpu")

    def _sync_device_performance(self) -> None:
        if not hasattr(self, "last_performance"):
            return
        self.last_performance["requested_device"] = self._requested_device
        self.last_performance["device"] = self._device_summary()
        self.last_performance["device_fallback"] = bool(self._device_fallback_reasons)
        self.last_performance["device_fallback_reason"] = "; ".join(
            self._device_fallback_reasons
        )

    def _record_device_fallback(self, reason: str) -> None:
        value = " ".join((reason or "未知原因").split())[:300]
        if value not in self._device_fallback_reasons:
            self._device_fallback_reasons.append(value)
            print(f"[paddle-ocr] GPU 回退 CPU: {value}")
        self._sync_device_performance()

    @staticmethod
    def _paddle_cuda_status() -> tuple[bool, int]:
        import paddle

        compiled = bool(paddle.device.is_compiled_with_cuda())
        count = int(paddle.device.cuda.device_count()) if compiled else 0
        return compiled, count

    def _resolve_preferred_device(self) -> str:
        if self._requested_device == "cpu":
            return "cpu"
        try:
            compiled, device_count = self._paddle_cuda_status()
            if not compiled:
                self._record_device_fallback("当前 PaddlePaddle 未编译 CUDA 支持")
                return "cpu"
            device_id = int(self._requested_device.split(":", 1)[1])
            if device_id >= device_count:
                self._record_device_fallback(
                    f"配置设备 {self._requested_device} 不可用，可用 GPU 数量为 {device_count}"
                )
                return "cpu"
            return self._requested_device
        except Exception as exc:  # noqa: BLE001
            self._record_device_fallback(f"GPU 探测失败: {type(exc).__name__}: {exc}")
            return "cpu"

    def _load(self):
        _register_paddle_gpu_dll_directories()
        try:
            from paddleocr import PaddleOCR

            self._PaddleOCR = PaddleOCR
        except ImportError:
            self._load_error = "paddleocr 未安装，OCR 使用 demo 模式"
        except Exception as e:  # noqa: BLE001
            self._load_error = f"PaddleOCR 加载失败: {e}"
        if self._PaddleOCR is not None:
            self._preferred_device = self._resolve_preferred_device()
        self._sync_device_performance()

    def _create_ocr(self, lang: str, device: str):
        """创建 PaddleOCR 实例

        关键：
        - 禁用文档方向矫正与文档矫正（会干扰漫画竖排文字检测）
        - 保留文本行方向分类（横排文字识别需要）
        """
        return self._PaddleOCR(
            lang=lang,
            device=device,
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )

    def _lang_for(self, source_lang: str) -> str:
        return self._lang_map.get(source_lang, "japan")

    def _get_ocr(self, source_lang: str) -> object | None:
        lang = self._lang_for(source_lang)
        if lang in self._ocrs:
            self.last_performance["model_reuse_count"] += 1
            if lang not in self.last_performance["models"]:
                self.last_performance["models"].append(lang)
            return self._ocrs[lang]
        if self._PaddleOCR is not None:
            device = self._preferred_device
            started = time.monotonic()
            try:
                instance = self._create_ocr(lang, device)
            except Exception as e:  # noqa: BLE001
                if device.startswith("gpu"):
                    self._record_device_fallback(
                        f"PaddleOCR({lang}) 在 {device} 初始化失败: {type(e).__name__}: {e}"
                    )
                    self._preferred_device = "cpu"
                    device = "cpu"
                    try:
                        instance = self._create_ocr(lang, device)
                    except Exception as cpu_exc:  # noqa: BLE001
                        self._load_error = f"PaddleOCR({lang}) CPU 加载失败: {cpu_exc}"
                        self.last_performance["model_load_ms"] += int(
                            (time.monotonic() - started) * 1000
                        )
                        self._sync_device_performance()
                        return None
                else:
                    self._load_error = f"PaddleOCR({lang}) 加载失败: {e}"
                    self.last_performance["model_load_ms"] += int(
                        (time.monotonic() - started) * 1000
                    )
                    self._sync_device_performance()
                    return None
            self._ocrs[lang] = instance
            self._device_by_lang[lang] = device
            self.last_performance["model_load_ms"] += int((time.monotonic() - started) * 1000)
            if lang not in self.last_performance["models"]:
                self.last_performance["models"].append(lang)
            self._sync_device_performance()
        return self._ocrs[lang]

    @property
    def available(self) -> bool:
        return self._PaddleOCR is not None

    def _get_image(self, image_path: Path) -> Image.Image:
        resolved = image_path.resolve()
        if self._cached_image_path != resolved or self._cached_image is None:
            with Image.open(image_path) as source:
                self._cached_image = source.convert("RGB").copy()
            self._cached_image_path = resolved
        return self._cached_image

    @staticmethod
    def _is_gpu_runtime_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(token in message for token in (
            "cuda", "cudnn", "gpu", "out of memory", "resource exhausted",
            "memory allocation", "not compiled with cuda",
        ))

    def _switch_language_to_cpu(self, model_lang: str, exc: Exception):
        self._record_device_fallback(
            f"PaddleOCR({model_lang}) GPU 推理失败: {type(exc).__name__}: {exc}"
        )
        started = time.monotonic()
        try:
            cpu_ocr = self._create_ocr(model_lang, "cpu")
        except Exception as cpu_exc:  # noqa: BLE001
            self._load_error = f"PaddleOCR({model_lang}) CPU 回退加载失败: {cpu_exc}"
            self.last_performance["model_load_ms"] += int(
                (time.monotonic() - started) * 1000
            )
            self._sync_device_performance()
            return None
        self.last_performance["model_load_ms"] += int((time.monotonic() - started) * 1000)
        self._ocrs[model_lang] = cpu_ocr
        self._device_by_lang[model_lang] = "cpu"
        self._preferred_device = "cpu"
        self._sync_device_performance()
        return cpu_ocr

    def _predict(self, ocr, value, model_lang: str, variant: str):
        ocr = self._ocrs.get(model_lang, ocr)
        started = time.monotonic()
        self.last_performance["call_count"] += 1
        if model_lang not in self.last_performance["models"]:
            self.last_performance["models"].append(model_lang)
        label = f"{model_lang}:{variant}"
        if label not in self.last_performance["variants"]:
            self.last_performance["variants"].append(label)
        try:
            result = ocr.predict(value)
        except Exception as exc:
            self.last_performance["inference_ms"] += int((time.monotonic() - started) * 1000)
            device = self._device_by_lang.get(model_lang, self._preferred_device)
            if device.startswith("gpu") and self._is_gpu_runtime_error(exc):
                cpu_ocr = self._switch_language_to_cpu(model_lang, exc)
                if cpu_ocr is not None:
                    retry_started = time.monotonic()
                    self.last_performance["call_count"] += 1
                    try:
                        return cpu_ocr.predict(value)
                    finally:
                        self.last_performance["inference_ms"] += int(
                            (time.monotonic() - retry_started) * 1000
                        )
            raise
        self.last_performance["inference_ms"] += int((time.monotonic() - started) * 1000)
        return result

    def recognize(
        self,
        image_path: Path,
        regions: list[TextRegion],
        source_lang: str = "ja",
    ) -> None:
        self.reset_performance()
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
            self._apply_detections(regions, detections, source_lang)
            return

        # 否则对现有区域逐个裁剪识别
        self._recognize_by_regions(image_path, regions, ocr, source_lang)

    def recognize_regions(self, image_path: Path, regions: list[TextRegion], source_lang: str) -> None:
        """按已有检测框识别，并对中英文使用区域级温和预处理候选。

        语言路由器调用此接口，避免为了换识别语言重新复制整条流水线，也避免
        Paddle 的整页检测结果把不同语言区域错误映射到彼此的框。
        """
        ocr = self._get_ocr(source_lang)
        if ocr is None:
            for region in regions:
                region.text = ""
                region.confidence = 0.0
                region.source_lang = source_lang
                region.ocr_engine = self.name
            return
        self.recognize_region_crops(image_path, regions, source_lang, ocr=ocr)

    def recognize_region_crops(self, image_path: Path, regions: list[TextRegion], source_lang: str, ocr=None) -> None:
        """只识别给定裁剪；供整页未匹配区域和英文桥接拆分复识别使用。"""
        ocr = ocr or self._get_ocr(source_lang)
        if ocr is None:
            return
        img = self._get_image(image_path)
        for region in regions:
            result = self._recognize_region_candidates(img, region, ocr, source_lang)
            self._apply_region_result(region, result, source_lang, f"显式源语言:{source_lang}")

    def _apply_region_result(self, region, result, source_lang, reason) -> None:
        region.text = result["text"]
        region.confidence = result["score"]
        region.source_lang = source_lang
        region.ocr_engine = self.name
        region.ocr_candidates = result["candidates"]
        region.ocr_attempted_models = [f"paddle:{source_lang}"]
        region.ocr_preprocess_variants = [
            candidate.get("variant", "") for candidate in result["candidates"] if candidate.get("variant")
        ]
        region.ocr_route_reason = reason
        variant = result.get("variant", "original")
        region.ocr_fallback = variant not in {"original", "full-page"} or region.confidence < self.settings.ocr_candidate_fallback_threshold
        if variant not in {"original", "full-page"}:
            region.ocr_fallback_reason = f"采用预处理候选:{variant}"
        elif region.confidence < self.settings.ocr_candidate_fallback_threshold:
            region.ocr_fallback_reason = "候选置信度低于回退阈值"

    def _recognize_region_candidates(self, image, region, ocr, source_lang: str) -> dict:
        x0, y0, x1, y1 = region.bounds
        pad = max(3, int(round(min(max(1, x1 - x0), max(1, y1 - y0)) * 0.12)))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(image.width, x1 + pad)
        y1 = min(image.height, y1 + pad)
        crop = image.crop((x0, y0, x1, y1))
        arr = np.asarray(crop)
        work = self._scaled_region(arr, source_lang)
        gray = None
        enhanced = None
        candidates = []
        attempted = 0
        variant_names = ["original", "gray-contrast"]
        if source_lang == "en":
            variant_names.append("adaptive-binary")
        model_lang = self._lang_for(source_lang)
        for variant_name in variant_names:
            attempted += 1
            if variant_name == "original":
                variant = work
            else:
                import cv2

                if gray is None:
                    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
                if enhanced is None:
                    enhanced = cv2.convertScaleAbs(
                        gray, alpha=1.22 if source_lang == "en" else 1.10, beta=0
                    )
                if variant_name == "gray-contrast":
                    variant = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                else:
                    threshold = cv2.adaptiveThreshold(
                        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 21, 7,
                    )
                    variant = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
            try:
                detections = self._extract_detections(
                    self._predict(ocr, variant, model_lang, variant_name)
                )
            except Exception:
                detections = []
            if not detections:
                continue
            text = self._normalize_text(" ".join(d["text"] for d in detections), source_lang)
            score = max(float(d.get("score", 0.0)) for d in detections)
            if text:
                candidates.append({"engine": self.name, "lang": source_lang, "variant": variant_name, "text": text, "confidence": score})
                if self._candidate_sufficient(text, score, source_lang):
                    break
        if not candidates:
            return {"text": "", "score": 0.0, "variant": "", "candidates": []}
        if attempted > 1:
            self.last_performance["fallback_count"] += 1
        candidates.sort(
            key=lambda item: (
                self._candidate_rank(item["text"], item["confidence"], source_lang),
                len(item["text"]),
            ),
            reverse=True,
        )
        best = candidates[0]
        return {"text": best["text"], "score": float(best["confidence"]), "variant": best["variant"], "candidates": candidates}

    def _candidate_sufficient(self, text: str, score: float, source_lang: str) -> bool:
        if not text.strip() or score < float(self.settings.ocr_candidate_fallback_threshold):
            return False
        hint = region_language_hint(text, source_lang)
        return hint.language == source_lang or hint.confidence < 0.60

    @staticmethod
    def _candidate_rank(text: str, score: float, source_lang: str) -> float:
        hint = region_language_hint(text, source_lang)
        if hint.language == source_lang:
            return float(score) + 0.10 * hint.confidence
        if hint.confidence >= 0.60:
            return float(score) - 0.10
        return float(score)

    def _scaled_region(self, arr: np.ndarray, source_lang: str) -> np.ndarray:
        import cv2

        h, w = arr.shape[:2]
        if source_lang == "en":
            scale = 3.0 if min(h, w) < 18 else float(self.settings.ocr_en_upscale)
        elif source_lang == "zh":
            scale = float(self.settings.ocr_zh_upscale)
        else:
            scale = 1.0
        scale = max(1.0, min(3.0, scale))
        if scale <= 1.01:
            return arr
        return cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def _preprocess_variants(self, arr: np.ndarray, source_lang: str) -> list[tuple[str, np.ndarray]]:
        """返回原图、放大增强图和必要的二值候选；中文不使用激进锐化。"""
        import cv2

        work = self._scaled_region(arr, source_lang)
        variants: list[tuple[str, np.ndarray]] = [("original", work)]
        gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        # 轻度对比度增强，保留描边字体和低对比度细笔画的灰阶信息。
        enhanced = cv2.convertScaleAbs(gray, alpha=1.22 if source_lang == "en" else 1.10, beta=0)
        variants.append(("gray-contrast", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)))
        if source_lang == "en":
            threshold = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7
            )
            variants.append(("adaptive-binary", cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)))
        return variants

    @staticmethod
    def _normalize_text(text: str, source_lang: str) -> str:
        value = " ".join((text or "").replace("\u00a0", " ").split())
        if source_lang == "en":
            # 保留英文单词间空格、大小写、撇号/连字符/长横线和标点。
            value = value.replace(" ’", "’").replace("' ", "'")
        elif source_lang == "zh":
            # 只删除 CJK 字符之间的 OCR 偶发空格，不动全角标点和数字间空格。
            import re

            value = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", value)
        return value.strip()

    def _detect_all(self, image_path: Path, ocr, source_lang: str) -> list[dict]:
        """双次检测：原图横排 + 旋转 90° 竖排（合并去重）"""
        detections: list[dict] = []

        # 1. 横排检测
        try:
            result = self._predict(ocr, str(image_path), self._lang_for(source_lang), "full-page")
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
                rot_result = self._predict(ocr, rot_arr, self._lang_for(source_lang), "full-page-rotated")
                rot_dets = self._extract_detections(rot_result)
                # 坐标映射回原图：旋转图(x',y') -> 原图(W-1-y', x')
                for det in rot_dets:
                    if det["box"]:
                        det["box"] = [
                            [W - 1 - p[1], p[0]] for p in det["box"]
                        ]
                    if det.get("poly"):
                        det["poly"] = [
                            [W - 1 - p[1], p[0]] for p in det["poly"]
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
                poly_pts = None
                if polys and i < len(polys):
                    poly = polys[i]
                    if poly is not None and len(poly) >= 4:
                        # 保留完整多边形点（可能 bbox 不止4点）
                        poly_pts = [[float(p[0]), float(p[1])] for p in poly]
                        box = [[float(p[0]), float(p[1])] for p in poly[:4]]
                detections.append(
                    {
                        "text": str(text).strip(),
                        "score": float(scores[i]) if scores and i < len(scores) else 0.0,
                        "box": box,
                        "poly": poly_pts,
                    }
                )
        return detections

    def _apply_detections(self, regions: list[TextRegion], detections: list[dict], source_lang: str = "ja") -> None:
        """将 PaddleOCR 检测结果映射到现有 regions，未匹配的追加新 region"""
        # 若 region 数量为 0，直接用检测结果生成 region
        if not regions:
            for det in detections:
                box = det["box"] or [[0, 0], [1, 0], [1, 1], [0, 1]]
                regions.append(
                    TextRegion(box=box, text=det["text"], confidence=det["score"], poly=det.get("poly"), source_lang=source_lang, ocr_engine=self.name)
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
                    regions[best_idx].source_lang = source_lang
                    regions[best_idx].ocr_engine = self.name
                    if det["box"]:
                        regions[best_idx].box = det["box"]
                    if det.get("poly"):
                        regions[best_idx].poly = det["poly"]
                else:
                    regions.append(
                        TextRegion(box=det["box"], text=det["text"], confidence=det["score"], poly=det.get("poly"), source_lang=source_lang, ocr_engine=self.name)
                    )
            else:
                # 无坐标，按顺序分配
                for i, region in enumerate(regions):
                    if not matched[i]:
                        matched[i] = True
                        region.text = det["text"]
                        region.confidence = det["score"]
                        region.source_lang = source_lang
                        region.ocr_engine = self.name
                        break

    def _recognize_by_regions(self, image_path: Path, regions: list[TextRegion], ocr, source_lang: str = "ja") -> None:
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
                result = self._predict(ocr, arr, self._lang_for(source_lang), "region-original")
                texts = []
                for res in result or []:
                    for t in res.get("rec_texts") or []:
                        if t and str(t).strip():
                            texts.append(str(t).strip())
                region.text = self._normalize_text(" ".join(texts), source_lang)
                region.confidence = 0.9
                region.source_lang = source_lang
                region.ocr_engine = self.name
            except Exception:
                region.text = ""
                region.confidence = 0.0
                region.source_lang = source_lang
                region.ocr_engine = self.name


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
    # 默认也尝试真�?OCR（若已安装且可用），否则 demo
    try:
        engine = PaddleOCREngine()
        if engine.available:
            return engine
    except Exception:  # noqa: BLE001
        pass
    return DemoOCREngine()


class MIT48OCREngine(BaseOCR):
    """移植自 manga-image-translator 的 48px 自训 OCR（ConvNeXt + RoFormer + beam search）

    不自带检测（supports_detection=False），由 detector 提供 textline 四边形。
    """

    name = "mit48"

    supports_detection = False

    def __init__(self):
        self.settings = get_settings()
        self._impl = None
        self._load_error = ""
        self._load()

    @property
    def available(self) -> bool:
        return self._impl is not None

    def _load(self):
        try:
            from app.services.engines.mit.config import ocr_params, resolve_device
            from app.services.engines.mit.ocr_48px import Mit48Ocr

            p = ocr_params()
            self._impl = Mit48Ocr(device=resolve_device(self.settings.mit_device))
            self._prob = p.prob
        except Exception as e:  # noqa: BLE001
            self._load_error = f"mit48 OCR 加载失败: {e}"
            self._impl = None

    def recognize(
        self,
        image_path: Path,
        regions: list[TextRegion],
        source_lang: str = "ja",
    ) -> None:
        if self._impl is None:
            for r in regions:
                r.text = ""
                r.confidence = 0.0
            return
        try:
            from app.services.engines.mit.quadrilateral import Quadrilateral

            img = np.array(Image.open(image_path).convert("RGB"))
            quads = []
            for r in regions:
                q = r._quad
                if q is None:
                    q = Quadrilateral(np.asarray(r.box, dtype=float).reshape(4, 2), r.text, r.confidence)
                    r._quad = q
                quads.append(q)
            self._impl.recognize(img, quads, prob_threshold=getattr(self, "_prob", 0.2))
            for r, q in zip(regions, quads):
                pts = np.asarray(q.pts).round().astype(int)
                r.box = [[int(pts[i][0]), int(pts[i][1])] for i in range(4)]
                r.poly = [[float(pts[i][0]), float(pts[i][1])] for i in range(4)]
                r.direction = q.direction
                r.text = q.text or ""
                r.confidence = float(q.prob)
                r.source_lang = "ja"
                r.ocr_engine = "mit48"
                if q.text:
                    r.fg_color = (int(q.fg_r), int(q.fg_g), int(q.fg_b))
                    r.bg_color = (int(q.bg_r), int(q.bg_g), int(q.bg_b))
        except Exception as e:  # noqa: BLE001
            print(f"[mit48] OCR 推理失败: {e}")
            for r in regions:
                r.confidence = 0.0


def create_ocr_engine_router() -> BaseOCR:
    """按 ocr_backend 配置路由

    优先尝试配置的后端；mit 系列缺失（如混合缺 manga-ocr）时回退 mit48，
    最后回退 PaddleOCR/demo。
    """
    settings = get_settings()
    backend = settings.ocr_backend
    # 默认 MIT 配置由统一路由器管理：ja 保留 MIT48+manga-ocr，zh/en 明确走 Paddle。
    # 这样不改 MIT 核心模型，也不会让中英文经过 manga-ocr。
    if backend in {"mit48", "mangaocr", "mit48+mangaocr"}:
        return LanguageRoutingOCREngine()
    table = {
        "mit48": MIT48OCREngine,
        "mangaocr": MangaOCREngine,
        "mit48+mangaocr": MixedOCREngine,
    }
    candidates = [backend]
    if backend in table and backend != "mit48":
        candidates.append("mit48")
    for name in candidates:
        if name not in table:
            continue
        try:
            engine = table[name]()
            if engine.available:
                return engine
            print(f"[ocr] {name} 不可用: {engine._load_error}")
        except Exception as e:  # noqa: BLE001
            print(f"[ocr] {name} 加载异常: {e}")
    return create_ocr_engine()


def _empty_quads_only(quads: list) -> list:
    """只把 mit48 完全没读出的行（空文本）交给 manga-ocr；mit48 有输出（即使低置信）直接保留

    背景：mit48 对风格化/小字常能读对内容但置信度低（0.45~0.6）；manga-ocr 无真实置信度
    （强行抬到 0.85）且对这类低置信行反而会幻觉成含拉丁字母的乱码（NS/SM/纽哈梅/古哈米）。
    若按「prob < 阈值」把低置信行也交给 manga-ocr，会把已读对的内容覆盖成乱码。
    故改为仅当 mit48 完全没读出（空文本）时才用 manga-ocr 补识别。
    """
    return [q for q in quads if not (q.text or "").strip()]


def _quads_from_regions(regions: list[TextRegion]) -> list:
    """为 region 建立/复用 MIT Quadrilateral"""
    from app.services.engines.mit.quadrilateral import Quadrilateral

    quads = []
    for r in regions:
        q = r._quad
        if q is None:
            q = Quadrilateral(np.asarray(r.box, dtype=float).reshape(4, 2), r.text, r.confidence)
            r._quad = q
        quads.append(q)
    return quads


class MangaOCREngine(BaseOCR):
    """manga-ocr（kha-white/manga-ocr-base）识别引擎（日漫风格化字体召回强，无置信度）"""

    name = "mangaocr"

    supports_detection = False

    def __init__(self):
        self.settings = get_settings()
        self._impl = None
        self._load_error = ""
        self._load()

    @property
    def available(self) -> bool:
        return self._impl is not None and self._impl.available

    def _load(self):
        from app.services.engines.mit.mocr import MangaOcrWrapper

        self._impl = MangaOcrWrapper()
        if not self._impl.available:
            self._load_error = self._impl._error

    def recognize(
        self,
        image_path: Path,
        regions: list[TextRegion],
        source_lang: str = "ja",
    ) -> None:
        if self._impl is None:
            for r in regions:
                r.confidence = 0.0
            return
        try:
            img = np.array(Image.open(image_path).convert("RGB"))
            quads = _quads_from_regions(regions)
            self._impl.recognize(img, quads)
            for r, q in zip(regions, quads):
                pts = np.asarray(q.pts).round().astype(int)
                r.box = [[int(pts[i][0]), int(pts[i][1])] for i in range(4)]
                r.poly = [[float(pts[i][0]), float(pts[i][1])] for i in range(4)]
                r.direction = q.direction
                r.text = q.text or ""
                r.confidence = float(q.prob)
                r.source_lang = "ja"
                r.ocr_engine = "mangaocr"
                r.ocr_fallback = True
                r.ocr_fallback_reason = "日文区域由 manga-ocr 识别"
        except Exception as e:  # noqa: BLE001
            print(f"[mangaocr] OCR 推理失败: {e}")
            for r in regions:
                r.confidence = 0.0


class LanguageRoutingOCREngine(BaseOCR):
    """按源语言/区域路由 MIT48、manga-ocr 与 PaddleOCR。

    该类只做识别器编排，检测、擦除、翻译和渲染仍走原有流水线。auto 模式
    绝不把未知区域直接交给 manga-ocr：先用 MIT48/Paddle 候选确认日文，
    manga-ocr 只救已确认日文且 MIT48 为空的区域。
    """

    name = "language-router"
    supports_detection = False
    supports_language_routing = True

    def __init__(self):
        self.settings = get_settings()
        self._mit48 = None
        self._ja = None
        self._paddle = None
        self._manga = None
        self._load_errors: list[str] = []
        self._route_model_load_ms = 0
        self._native_call_count = 0
        self._native_inference_ms = 0
        self.last_performance: dict = {}

    def reset_performance(self) -> None:
        self._route_model_load_ms = 0
        self._native_call_count = 0
        self._native_inference_ms = 0
        self.last_performance = {}
        paddle = getattr(self, "_paddle", None)
        if paddle is not None and hasattr(paddle, "reset_performance"):
            paddle.reset_performance()

    def _finish_performance(self, regions) -> None:
        paddle_perf = dict(getattr(getattr(self, "_paddle", None), "last_performance", {}) or {})
        models = list(paddle_perf.get("models", []))
        for region in regions:
            for model in region.ocr_attempted_models:
                if model not in models:
                    models.append(model)
        self.last_performance = {
            "model_load_ms": self._route_model_load_ms + int(paddle_perf.get("model_load_ms", 0)),
            "inference_ms": self._native_inference_ms + int(paddle_perf.get("inference_ms", 0)),
            "call_count": self._native_call_count + int(paddle_perf.get("call_count", 0)),
            "models": models,
            "variants": list(paddle_perf.get("variants", [])),
            "model_reuse_count": int(paddle_perf.get("model_reuse_count", 0)),
            "fallback_count": sum(1 for region in regions if region.ocr_fallback),
            "requested_device": paddle_perf.get("requested_device", ""),
            "device": paddle_perf.get("device", ""),
            "device_fallback": bool(paddle_perf.get("device_fallback", False)),
            "device_fallback_reason": paddle_perf.get("device_fallback_reason", ""),
        }

    @property
    def available(self) -> bool:
        return True

    def _get_paddle(self):
        if self._paddle is None:
            try:
                started = time.monotonic()
                self._paddle = PaddleOCREngine()
                self._route_model_load_ms += int((time.monotonic() - started) * 1000)
            except Exception as exc:  # noqa: BLE001
                self._load_errors.append(f"paddle: {exc}")
        return self._paddle if self._paddle and self._paddle.available else None

    def _get_mit48(self):
        if self._mit48 is None:
            try:
                started = time.monotonic()
                self._mit48 = MIT48OCREngine()
                self._route_model_load_ms += int((time.monotonic() - started) * 1000)
            except Exception as exc:  # noqa: BLE001
                self._load_errors.append(f"mit48: {exc}")
        return self._mit48 if self._mit48 and self._mit48.available else None

    def _get_ja(self):
        if self._ja is None:
            backend = self.settings.ocr_backend
            try:
                started = time.monotonic()
                if backend == "mit48+mangaocr":
                    self._ja = MixedOCREngine()
                    self._route_model_load_ms += int((time.monotonic() - started) * 1000)
                elif backend == "mangaocr":
                    self._ja = MangaOCREngine()
                    self._route_model_load_ms += int((time.monotonic() - started) * 1000)
                else:
                    self._ja = self._get_mit48()
            except Exception as exc:  # noqa: BLE001
                self._load_errors.append(f"ja: {exc}")
                # manga-ocr 缺失时仍保留 MIT48 日文路径。
                self._ja = self._get_mit48()
        return self._ja if self._ja and getattr(self._ja, "available", True) else None

    def _get_manga(self):
        if self._manga is None:
            try:
                self._manga = MangaOCREngine()
            except Exception as exc:  # noqa: BLE001
                self._load_errors.append(f"mangaocr: {exc}")
        return self._manga if self._manga and self._manga.available else None

    def recognize(self, image_path: Path, regions: list[TextRegion], source_lang: str = "auto") -> None:
        self.reset_performance()
        source_lang = source_lang or "auto"
        if source_lang == "ja":
            engine = self._get_ja() or self._get_paddle()
            if engine is None:
                self._clear(regions, source_lang)
                self._finish_performance(regions)
                return
            self._native_call_count += 1
            started = time.monotonic()
            engine.recognize(image_path, regions, "ja")
            self._native_inference_ms += int((time.monotonic() - started) * 1000)
            for region in regions:
                if not region.source_lang:
                    region.source_lang = "ja"
                if not region.ocr_engine:
                    region.ocr_engine = getattr(engine, "name", "ja")
                region.ocr_attempted_models = [getattr(engine, "name", "ja")]
                region.ocr_route_reason = "显式日文，使用日文漫画 OCR"
            self._finish_performance(regions)
            return
        if source_lang in {"zh", "en"}:
            paddle = self._get_paddle()
            if paddle is not None:
                if hasattr(paddle, "recognize_regions"):
                    paddle.recognize_regions(image_path, regions, source_lang)
                else:
                    paddle.recognize(image_path, regions, source_lang)
                if source_lang == "en":
                    self._split_english_bridge_regions(image_path, regions, paddle)
                self._finish_performance(regions)
                return
            self._clear(regions, source_lang)
            self._finish_performance(regions)
            return
        self._recognize_auto(image_path, regions)
        paddle = self._get_paddle()
        if paddle is not None:
            self._split_english_bridge_regions(image_path, regions, paddle)
        self._finish_performance(regions)

    @staticmethod
    def _clear(regions, source_lang):
        for region in regions:
            region.text = ""
            region.confidence = 0.0
            region.source_lang = source_lang
            region.ocr_engine = "unavailable"

    def _split_english_bridge_regions(self, image_path: Path, regions: list[TextRegion], paddle) -> None:
        """把同一 CTD 框中横跨两个气泡的英文文本栈拆开并分别复识别。"""
        if not image_path.is_file():
            return
        if hasattr(paddle, "_get_image"):
            image = np.asarray(paddle._get_image(image_path))
        else:
            with Image.open(image_path) as source:
                image = np.asarray(source.convert("RGB"))
        output: list[TextRegion] = []
        for region in regions:
            if region.source_lang != "en":
                output.append(region)
                continue
            boxes = _horizontal_ink_split_boxes(image, region)
            if len(boxes) != 2:
                output.append(region)
                continue
            children = [
                TextRegion(
                    box=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    poly=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    direction="h",
                    source_lang="en",
                )
                for x0, y0, x1, y1 in boxes
            ]
            if hasattr(paddle, "recognize_region_crops"):
                paddle.recognize_region_crops(image_path, children, "en")
            else:
                paddle.recognize_regions(image_path, children, "en")
            if not all(
                child.text.strip()
                and child.confidence >= 0.45
                and _english_word_count(child.text) >= 2
                for child in children
            ):
                output.append(region)
                continue
            for child in children:
                child.ocr_fallback = True
                child.ocr_fallback_reason = "拆分跨气泡英文桥接行后复识别"
            output.extend(children)
        regions[:] = output

    def _recognize_auto(self, image_path: Path, regions: list[TextRegion]) -> None:
        mit = self._get_mit48()
        if mit is not None:
            self._native_call_count += 1
            started = time.monotonic()
            mit.recognize(image_path, regions, "ja")
            self._native_inference_ms += int((time.monotonic() - started) * 1000)
        mit_states = {
            id(region): ((region.text or "").strip(), float(region.confidence or 0.0))
            for region in regions
        }
        paddle = self._get_paddle()
        image = paddle._get_image(image_path) if paddle is not None else None
        strong_hints = []
        for item in regions:
            value = (item.text or "").strip()
            if not value:
                continue
            hint = region_language_hint(value, self.settings.auto_source_fallback)
            if hint.confidence >= 0.65:
                strong_hints.append(hint.language)
        page_hint = ""
        if strong_hints:
            counts = {lang: strong_hints.count(lang) for lang in {"zh", "ja", "en"}}
            winner = max(counts, key=counts.get)
            if counts[winner] >= 3 and counts[winner] / len(strong_hints) >= 0.80:
                page_hint = winner
        for region in regions:
            mit_text, mit_conf = mit_states[id(region)]
            candidates = []
            context_region = None
            attempted_models = ["mit48"] if mit is not None else []
            if mit_text:
                candidates.append({
                    "engine": "mit48",
                    "lang": "ja" if region_language_hint(mit_text, "ja").language == "ja" else "",
                    "variant": "mit48",
                    "text": mit_text,
                    "confidence": mit_conf,
                })
            # 有假名的 MIT 结果直接采用，避免中文/英文被日漫模型覆盖。
            mit_hint = region_language_hint(mit_text, "ja") if mit_text else None
            if mit_hint and mit_hint.language == "ja" and mit_hint.confidence >= 0.55:
                region.source_lang = "ja"
                region.ocr_engine = "mit48"
                region.ocr_candidates = candidates
                region.ocr_attempted_models = attempted_models
                region.ocr_preprocess_variants = ["mit48"]
                region.ocr_route_reason = mit_hint.reason
                continue

            if mit_text and not any(
                ch.isalpha() or "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff"
                for ch in mit_text
            ):
                region.source_lang = self.settings.auto_source_fallback
                region.ocr_engine = "mit48"
                region.ocr_candidates = candidates
                region.ocr_attempted_models = attempted_models
                region.ocr_preprocess_variants = ["mit48"]
                region.ocr_route_reason = "纯数字或符号，保留首次 OCR 结果"
                continue

            if paddle is not None and image is not None:
                if mit_hint and mit_hint.language == "en" and mit_hint.confidence >= 0.55:
                    langs = ("en",)
                    route_reason = mit_hint.reason
                elif mit_hint and mit_hint.language == "zh" and "常见中文词" in mit_hint.reason:
                    langs = ("zh",)
                    route_reason = mit_hint.reason
                elif mit_hint and mit_hint.language == "zh" and page_hint == "zh":
                    langs = ("zh",)
                    route_reason = "纯汉字区域采用高置信中文页面共识"
                elif mit_hint and mit_hint.language == "zh" and page_hint == "ja":
                    langs = ("ja", "zh")
                    route_reason = "纯汉字区域采用高置信日文页面共识比较候选"
                elif mit_hint and mit_hint.language == "zh":
                    langs = ("zh", "ja")
                    route_reason = "纯汉字区域，比较中文与日文候选"
                else:
                    ordered = [
                        page_hint,
                        self.settings.auto_source_fallback,
                        mit_hint.language if mit_hint else "",
                        "zh",
                        "en",
                        "ja",
                    ]
                    langs = tuple(dict.fromkeys(lang for lang in ordered if lang in {"zh", "ja", "en"}))
                    route_reason = (
                        f"语言特征不足，优先采用高置信页面共识:{page_hint}"
                        if page_hint else "语言特征不足，按置信度级联候选"
                    )
                for lang in langs:
                    ocr = paddle._get_ocr(lang)
                    if ocr is None:
                        continue
                    attempted_models.append(f"paddle:{lang}")
                    result = paddle._recognize_region_candidates(image, region, ocr, lang)
                    candidates.extend(result.get("candidates", []))
                    if len(langs) != 2 or set(langs) != {"zh", "ja"}:
                        if result.get("text") and paddle._candidate_sufficient(
                            result["text"], float(result.get("score", 0.0)), lang
                        ):
                            break
                x0, y0, x1, y1 = region.bounds
                if (
                    page_hint == "ja"
                    and getattr(region, "direction", None) == "v"
                    and mit_conf < self.settings.ocr_candidate_fallback_threshold
                    and y1 - y0 <= max(1, x1 - x0) * 3.2
                ):
                    context_region = self._vertical_context_region(region)
                    ja_ocr = paddle._get_ocr("ja")
                    if ja_ocr is not None:
                        attempted_models.append("paddle:ja-context")
                        context_result = paddle._recognize_region_candidates(
                            image, context_region, ja_ocr, "ja"
                        )
                        for candidate in context_result.get("candidates", []):
                            candidate = dict(candidate)
                            candidate["variant"] = f"vertical-context:{candidate.get('variant', 'original')}"
                            candidates.append(candidate)
            elif paddle is None:
                route_reason = "PaddleOCR 不可用，保留日文模型候选"

            chosen = self._choose_candidate(candidates, mit_hint, page_hint)
            if chosen is None:
                self._clear([region], self.settings.auto_source_fallback)
                region.ocr_candidates = candidates
                region.ocr_attempted_models = attempted_models
                region.ocr_route_reason = route_reason
                continue
            region.text = chosen["text"]
            region.confidence = float(chosen["confidence"])
            region.source_lang = chosen["lang"] or self.settings.auto_source_fallback
            region.ocr_engine = chosen["engine"]
            if context_region is not None and str(chosen.get("variant", "")).startswith("vertical-context:"):
                region.box = context_region.box
                region.poly = context_region.poly
                region.mask = None
            low_confidence = float(chosen["confidence"]) < self.settings.ocr_candidate_fallback_threshold
            region.ocr_fallback = chosen["engine"] != "mit48" or low_confidence
            if chosen["engine"] != "mit48":
                region.ocr_fallback_reason = "auto 区域级语言候选比较"
            elif low_confidence:
                region.ocr_fallback_reason = "候选置信度低于回退阈值"
            region.ocr_candidates = candidates
            region.ocr_attempted_models = attempted_models
            region.ocr_preprocess_variants = list(dict.fromkeys(
                candidate.get("variant", "") for candidate in candidates if candidate.get("variant")
            ))
            region.ocr_route_reason = route_reason

            # 只对已经从 Paddle 候选确认的日文空结果启用 manga-ocr。
            needs_manga_review = (
                region.source_lang == "ja"
                and _looks_like_japanese_ocr_garbage(region.text, float(region.confidence or 0.0), page_hint)
            )
            if region.source_lang == "ja" and ((not region.text.strip() and mit_text == "") or needs_manga_review):
                manga = self._get_manga()
                if manga is not None:
                    previous_state = {
                        name: getattr(region, name, None)
                        for name in ("box", "poly", "direction", "text", "confidence", "source_lang", "ocr_engine")
                    }
                    manga.recognize(image_path, [region], "ja")
                    if needs_manga_review and region.text.strip() and not _has_japanese_script(region.text):
                        for name, old_value in previous_state.items():
                            setattr(region, name, old_value)
                    else:
                        region.ocr_fallback = True
                        region.ocr_engine = "mangaocr"
                        region.ocr_fallback_reason = (
                            "MIT48 疑似字符集误识别，manga-ocr 复核"
                            if needs_manga_review else "MIT48 空结果，manga-ocr 救空"
                        )

    @staticmethod
    def _vertical_context_region(region):
        x0, y0, x1, y1 = region.bounds
        width = max(1, x1 - x0)
        left = max(0, x0 - int(round(width * 0.24)))
        right = x1 + int(round(width * 0.08))
        top = max(0, y0 - int(round(width * 1.16)))
        bottom = y1 + max(2, int(round(width * 0.10)))
        box = [[left, top], [right, top], [right, bottom], [left, bottom]]
        return TextRegion(box=box, poly=[[float(x), float(y)] for x, y in box], direction="v")

    @staticmethod
    def _choose_candidate(candidates, hint, page_hint=""):
        usable = [c for c in candidates if (c.get("text") or "").strip()]
        if not usable:
            return None
        ranked = []
        for candidate in usable:
            detection = region_language_hint(candidate["text"], candidate["lang"])
            score = float(candidate.get("confidence", 0.0))
            meaningful = [
                ch for ch in candidate["text"]
                if ch.isalpha() or "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff"
            ]
            cjk_only = bool(meaningful) and all(
                "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff"
                for ch in meaningful
            )
            page_resolves_han = cjk_only and page_hint in {"zh", "ja"}
            resolved = dict(candidate)
            resolved_lang = candidate.get("lang") or page_hint
            if detection.confidence >= 0.55 and detection.language == page_hint:
                resolved_lang = detection.language
            elif page_resolves_han:
                resolved_lang = page_hint
            resolved["lang"] = resolved_lang

            if detection.language == resolved_lang:
                score += 0.22 * detection.confidence
            elif detection.confidence >= 0.55:
                score -= 0.16
            if hint and hint.language == resolved_lang and not page_resolves_han:
                score += 0.08
            if page_resolves_han and resolved_lang == page_hint:
                score += 0.06
            if page_hint == "ja" and detection.language == "ja" and detection.confidence >= 0.55:
                score += 0.22
            if page_hint == "ja" and detection.language == "en":
                latin_words = re.findall(r"[A-Za-z]+", candidate["text"])
                if latin_words and not any(len(word) >= 3 for word in latin_words):
                    score -= 0.55
            ranked.append((score, resolved))
        ranked.sort(key=lambda item: (item[0], len(item[1].get("text", ""))), reverse=True)
        return ranked[0][1]


class MixedOCREngine(BaseOCR):
    """mit48 + manga-ocr 混合：先跑 48px，概率低于阈值（或未识别出）的行改用 manga-ocr 补识别"""

    name = "mit48+mangaocr"

    supports_detection = False

    def __init__(self):
        self.settings = get_settings()
        self._mit48 = None
        self._mocr = None
        self._load_error = ""
        self._load()

    @property
    def available(self) -> bool:
        return self._mit48 is not None and self._mocr is not None and self._mocr.available

    def _load(self):
        from app.services.engines.mit.config import ocr_params, resolve_device
        from app.services.engines.mit.mocr import MangaOcrWrapper
        from app.services.engines.mit.ocr_48px import Mit48Ocr

        p = ocr_params()
        self._mit48 = Mit48Ocr(device=resolve_device(self.settings.mit_device))
        self._mocr = MangaOcrWrapper()
        self._prob = p.prob
        if not self._mocr.available:
            raise RuntimeError(self._mocr._error)

    def recognize(
        self,
        image_path: Path,
        regions: list[TextRegion],
        source_lang: str = "ja",
    ) -> None:
        if self._mit48 is None:
            for r in regions:
                r.confidence = 0.0
            return
        try:
            img = np.array(Image.open(image_path).convert("RGB"))
            quads = _quads_from_regions(regions)
            self._mit48.recognize(img, quads, prob_threshold=getattr(self, "_prob", 0.2))
            # 仅把 mit48 完全没读出的行交给 manga-ocr；mit48 有输出（即使低置信）保留，防被幻觉乱码覆盖
            low = _empty_quads_only(quads)
            fallback_ids = {id(q) for q in low}
            if low:
                self._mocr.recognize(img, low)
            for r, q in zip(regions, quads):
                pts = np.asarray(q.pts).round().astype(int)
                r.box = [[int(pts[i][0]), int(pts[i][1])] for i in range(4)]
                r.poly = [[float(pts[i][0]), float(pts[i][1])] for i in range(4)]
                r.direction = q.direction
                r.text = q.text or ""
                r.confidence = float(q.prob)
                r.source_lang = "ja"
                if id(q) in fallback_ids:
                    r.ocr_engine = "mangaocr"
                    r.ocr_fallback = True
                    r.ocr_fallback_reason = "MIT48 空结果，manga-ocr 救空"
                else:
                    r.ocr_engine = "mit48"
                if q.text:
                    r.fg_color = (int(q.fg_r), int(q.fg_g), int(q.fg_b))
                    r.bg_color = (int(q.bg_r), int(q.bg_g), int(q.bg_b))
        except Exception as e:  # noqa: BLE001
            print(f"[mit48+mangaocr] OCR 推理失败: {e}")
            for r in regions:
                r.confidence = 0.0
