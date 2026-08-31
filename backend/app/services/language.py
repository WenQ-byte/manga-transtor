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

_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
_KATAKANA_RE = re.compile(r"[\u30a0-\u30ff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HAN_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_MEANINGFUL_RE = re.compile(r"[A-Za-z\u3040-\u30ff\u3400-\u9fff]")
_CJK_PUNCT_RE = re.compile(r"[，。！？、；：‘’“”《》「」『』（）【】]" )

# 这些词只作为短文本的弱特征，不能凌驾于假名/拉丁字母比例之上。
_ZH_HINTS = frozenset("的了是不我你他她它们这那有在和就都一个没有什么可以吗吧啊呢").union(
    {"我们", "你们", "他们", "然后", "因为", "所以", "现在", "真的", "不是"}
)
_JA_HINTS = frozenset({"です", "ます", "でした", "ください", "ない", "する", "した", "こと", "もの", "から", "まで"})


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
    """按 OCR 文本识别中日英；汉字本身不再被当作日文。"""
    value = text or ""
    kana = len(_KANA_RE.findall(value))
    han = len(_HAN_RE.findall(value))
    latin = len(_LATIN_RE.findall(value))
    meaningful = kana + han + latin
    if meaningful == 0:
        reason = "文本为空" if not value.strip() else "仅含数字或符号"
        return LanguageDetection(fallback, 0.0, f"{reason}，使用默认语言")
    kana_ratio = kana / max(1, meaningful)
    latin_ratio = latin / max(1, meaningful)
    if kana and (kana_ratio >= 0.12 or kana >= 2):
        hint = "平假名" if _HIRAGANA_RE.search(value) else "片假名"
        confidence = min(0.99, 0.72 + kana_ratio * 0.25)
        return LanguageDetection("ja", confidence, f"检测到{hint}或假名-汉字混合文本")
    if latin >= max(2, han * 2) or latin_ratio >= 0.55:
        confidence = min(0.98, 0.62 + latin_ratio * 0.3)
        return LanguageDetection("en", confidence, "拉丁字母占主导")
    if han:
        compact = re.sub(r"\s+", "", value)
        zh_hint = sum(1 for word in _ZH_HINTS if word in compact)
        ja_hint = sum(1 for word in _JA_HINTS if word in compact)
        if ja_hint > zh_hint and len(compact) >= 2:
            return LanguageDetection("ja", 0.58, "汉字短文本包含日语词尾特征")
        confidence = min(0.9, 0.62 + min(han, 8) * 0.03 + zh_hint * 0.04)
        reason = "包含常见中文词特征" if zh_hint else "检测到汉字且无假名，按中文处理"
        if _CJK_PUNCT_RE.search(value):
            reason += "（含中文全角标点）"
        return LanguageDetection("zh", confidence, reason)
    if latin:
        return LanguageDetection("en", 0.55, "检测到少量拉丁字母")
    return LanguageDetection(fallback, 0.0, "语言特征不足，使用默认语言")


def region_language_hint(text: str, fallback: str = "zh") -> LanguageDetection:
    """区域级语言判断。

    与整页判断相比，这里对数字、符号和极短文本降低置信度，供 OCR 路由器
    结合各识别器的分数作候选比较；不会仅凭一个汉字把区域送入 manga-ocr。
    """
    value = (text or "").strip()
    if not value:
        return LanguageDetection(fallback, 0.0, "区域 OCR 为空")
    detection = detect_language(value, fallback=fallback)
    meaningful = len(_MEANINGFUL_RE.findall(value))
    if meaningful <= 1 or not re.search(r"[A-Za-z\u3040-\u30ff\u3400-\u9fff]", value):
        return LanguageDetection(detection.language, min(detection.confidence, 0.45), "区域信息不足，结合 OCR 置信度")
    return detection
