"""三语语言定义、后端映射与轻量自动识别。"""
from __future__ import annotations

import re
from dataclasses import dataclass

LANGUAGE_NAMES = {"zh": "中文", "ja": "日语", "en": "英语"}
SUPPORTED_LANGUAGES = tuple(LANGUAGE_NAMES)
SUPPORTED_SOURCE_LANGUAGES = ("auto", *SUPPORTED_LANGUAGES)

PROVIDER_LANGUAGE_MAP = {
    "google": {"zh": "zh-CN", "ja": "ja", "en": "en"},
    "mymemory": {"zh": "zh-CN", "ja": "ja", "en": "en"},
    "deepl": {"zh": "ZH", "ja": "JA", "en": "EN"},
}

_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HAN_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class LanguageDetection:
    language: str
    confidence: float
    reason: str


def provider_language(provider: str, language: str) -> str:
    """返回后端语言代码，不支持时明确报错。"""
    mapping = PROVIDER_LANGUAGE_MAP.get(provider)
    if mapping is None or language not in mapping:
        raise ValueError(f"{provider} 不支持语言代码: {language}")
    return mapping[language]


def detect_language(text: str, fallback: str = "ja") -> LanguageDetection:
    """按整页 OCR 文本识别中日英；低信息文本使用稳定回退。"""
    value = text or ""
    kana = len(_KANA_RE.findall(value))
    han = len(_HAN_RE.findall(value))
    latin = len(_LATIN_RE.findall(value))
    meaningful = kana + han + latin
    if meaningful == 0:
        reason = "文本为空" if not value.strip() else "仅含数字或符号"
        return LanguageDetection(fallback, 0.0, f"{reason}，使用默认语言")
    if kana:
        confidence = min(0.99, 0.75 + kana / max(8, meaningful))
        return LanguageDetection("ja", confidence, "检测到平假名或片假名")
    if latin >= max(2, han * 2):
        confidence = min(0.98, 0.65 + latin / max(20, meaningful * 2))
        return LanguageDetection("en", confidence, "拉丁字母占主导")
    if han:
        confidence = 0.78 if han >= 2 else 0.55
        return LanguageDetection("zh", confidence, "检测到汉字且不含假名")
    if latin:
        return LanguageDetection("en", 0.55, "检测到少量拉丁字母")
    return LanguageDetection(fallback, 0.0, "语言特征不足，使用默认语言")
