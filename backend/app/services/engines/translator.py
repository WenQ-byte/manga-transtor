"""翻译引擎：支持多种后端 + 自动降级回退链

后端优先级（可配置 translator_backend）：配置后端 → 其他已配置高质量后端 → 免费后端。

任意后端失败时自动回退到下一个可用后端，全部失败则返回原文
（专有名词词典替换始终生效）。
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Optional

import httpx

from app.config import get_settings
from app.models.schemas import LangCode
from app.services.engines.base import BaseTranslator
from app.services.language import LANGUAGE_NAMES, provider_language

# 文本中常见的拟声词/符号，保留不翻译
KEEP_PATTERN = re.compile(r"^[\s~！!？?…☆★♪♫◎○●▲△□■]+$")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_ENGLISH_ENTITY_AFTER_RE = re.compile(
    r"(?=(?i:\b(?:like|called|named|meet|met|another)\s+)"
    r"([A-Z][A-Z'-]{1,}(?:\s+[A-Z][A-Z'-]{1,}){0,3})\b)"
)
_ENGLISH_ENTITY_TITLE_RE = re.compile(
    r"\b([A-Z][a-z]+(?:[-'][A-Za-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Za-z]+)?)+)\b"
)
_JAPANESE_MIXED_NAME_RE = re.compile(r"[ァ-ヶー]{1,8}[\u3400-\u9fff]{1,3}")
_ENGLISH_ENTITY_STOP_WORDS = {
    "A",
    "AN",
    "ANOTHER",
    "HE",
    "HER",
    "HIM",
    "HIS",
    "I",
    "IT",
    "LIKE",
    "SHE",
    "THAT",
    "THE",
    "THEY",
    "THIS",
    "WE",
    "WHAT",
    "WHO",
    "YOU",
}


ENGLISH_ZH_PROMPT = (
    "你正在进行英语漫画到中文的翻译。对白优先使用自然、简洁、符合中文漫画阅读习惯的表达，"
    "不要把英语口语机械翻成书面语。结合上下文正确处理俚语、缩写、双关、反讽、否定、时态、"
    "条件句、反问句和省略，保留角色的年龄、身份、关系、说话风格、语气和情绪。"
    "数字、日期、单位、角色名、作品名和其他专有名词不得无故遗漏或改写；英文人名、地名、组织名"
    "优先遵循词典，未收录的名称在上下文中保持一致。旁白、对白、标题和拟声词使用合适的中文表达。"
    "英语习语必须按语境意译：somebody, anybody 这类递进表达应译成“谁都好”，不要逐词并列；"
    "make me sick 等情绪表达应按人物态度译成自然中文；damn 等强化词要保留情绪强度，但不要无故加重粗俗程度。"
    "避免“说到底、终究”这类同义成分重复堆叠，同一句只保留最自然的一种表达。"
    "不要补充背景设定、人物心理或原文没有的信息，禁止脑补。严格保持原文段落数量、行数和换行结构，"
    "不要为了排版在句子中间随意断行。只输出译文，不要解释、分析、引号、Markdown 或提示语。"
)

_TARGET_STYLE = {
    "zh": "使用自然、简洁、符合中文漫画阅读习惯的表达，避免机械书面腔",
    "ja": "使用自然的漫画日语，正确使用平假名、片假名、汉字、长音、促音和人物口吻",
    "en": "使用自然、简洁的漫画英语和地道口语，保留人物身份、年龄、情绪与关系",
}

_DIRECTION_STYLE = {
    ("en", "zh"): (
        "英语习语必须按语境意译，somebody、anybody 等递进表达不要逐词并列；"
        "make me sick 等情绪表达按人物态度译成自然中文；damn 等强化词保留情绪强度但不无故加重粗俗程度；"
        "正确处理否定、时态、条件句、反问和省略，避免同义成分重复堆叠"
    ),
    ("ja", "zh"): "日语语气词、敬语、称呼后缀和省略表达按人物关系转成自然中文，不套用英语口语规则",
    ("zh", "ja"): "根据人物关系选择自然的敬体或常体，不机械保留中文语序",
    ("zh", "en"): "根据人物身份选择自然的英语口语和缩略形式，不机械保留中文语序",
    ("ja", "en"): "把日语敬语、语气词和省略关系转成自然英语对白",
    ("en", "ja"): "把英语俚语、缩写和反讽转成自然日语对白，并保持人物关系",
}


def build_manga_prompt(source, target, *, context_count: int = 0) -> str:
    """按翻译方向生成统一漫画提示词。"""
    source_value = getattr(source, "value", source)
    target_value = getattr(target, "value", target)
    if source_value == "en" and target_value == "zh" and not context_count:
        return ENGLISH_ZH_PROMPT
    prefix = "你是专业的漫画翻译。"
    if context_count:
        prefix += (
            f"下面是同一页漫画中按阅读顺序排列的{context_count}段文字，每段使用<序号>...</序号>包裹。"
            "必须结合整页上下文保持称呼、语气和专有名词一致。相邻编号可能是同一句跨气泡对白，"
            "应先按完整句意理解，再让每个编号只承载对应原文片段的译文；各段连读必须自然，"
            "不得在后一段重复前一段已经表达的主语、谓语或结论。"
        )
    prompt = (
        f"{prefix}把{LANGUAGE_NAMES[source_value]}翻译成{LANGUAGE_NAMES[target_value]}。"
        f"{_TARGET_STYLE[target_value]}；根据语境处理口语、俚语、缩写、反讽、双关和省略表达。"
        "保持原文语义、语气、句子顺序与人物说话风格，不擅自补充背景、心理或原文没有的信息。"
        "人名、拟声词、专有名词、数字、日期、单位和符号不得无故遗漏或错误改写。"
        "严格保持每段内部的原文换行结构，不合并、拆分或新增内容，不为排版在句中随意断行。"
    )
    direction_style = _DIRECTION_STYLE.get((source_value, target_value), "")
    if direction_style:
        prompt += direction_style + "。"
    if context_count:
        prompt += (
            f"只输出{context_count}段译文，格式严格为<序号>译文</序号>，编号一一对应，"
            "不要合并、拆分、遗漏或新增，不得重复或改变编号；不要解释、注释、引号、Markdown 或提示语。"
        )
    else:
        prompt += "只输出译文，不要解释、注释、引号、Markdown 或提示语。"
    return prompt


def _is_english_to_chinese(source, target) -> bool:
    return (
        getattr(source, "value", source) == "en"
        and getattr(target, "value", target) == "zh"
    )


def _is_japanese_to_chinese(source, target) -> bool:
    return (
        getattr(source, "value", source) == "ja"
        and getattr(target, "value", target) == "zh"
    )


def _english_entity_hints(texts) -> list[str]:
    content = " ".join(str(text or "") for text in texts)
    found: list[str] = []
    seen: set[str] = set()
    for pattern in (_ENGLISH_ENTITY_AFTER_RE, _ENGLISH_ENTITY_TITLE_RE):
        for match in pattern.finditer(content):
            value = re.sub(r"\s+", " ", match.group(1)).strip(".,!?;:")
            words = value.split()
            if len(words) < 2 or len(words) > 4:
                continue
            if words[0].upper() in _ENGLISH_ENTITY_STOP_WORDS:
                continue
            if words[-1].upper() in _ENGLISH_ENTITY_STOP_WORDS:
                continue
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                found.append(value)
    return found[:12]


def _append_english_entity_hints(prompt: str, texts) -> str:
    hints = _english_entity_hints(texts)
    if not hints:
        return prompt
    names = "、".join(hints)
    return (
        f"{prompt}本页疑似专名原文：{names}。同一专名在所有段落中必须使用同一个中文译名，"
        "即使原文只写了简称，也不得漏译、擅自改名或前后不一致。"
    )


def _append_japanese_entity_hints(prompt: str, texts) -> str:
    content = " ".join(str(text or "") for text in texts)
    names = list(dict.fromkeys(_JAPANESE_MIXED_NAME_RE.findall(content)))[:12]
    if not names:
        return prompt
    return (
        f"{prompt}本页疑似日文人名或简称：{'、'.join(names)}。"
        "片假名与汉字混写的名称不要按字面词义拆译；能从作品或上下文确定惯用中文名时使用惯用译名，"
        "无法确定时保留原文并在本页保持一致，不要臆造带贬义的中文名字。"
    )


def _append_japanese_manga_term_hints(prompt: str, texts) -> str:
    content = " ".join(str(text or "") for text in texts)
    mappings = []
    for source, target in (
        ("ゴッドバレー", "神之谷"),
        ("ゴッド・バレー", "神之谷"),
        ("小僧共", "小鬼们"),
        ("うちに寄ってきなよ", "来我家坐坐吧"),
        ("鼻緒", "木屐带"),
    ):
        if source in content:
            mappings.append(f"{source}→{target}")
    if not mappings:
        return prompt
    return (
        f"{prompt}本页出现的固定日语漫画词组必须整体理解并统一翻译：{'、'.join(mappings)}。"
        "不得按片假名音节、汉字单字或词组中间位置拆开，不能遗漏短语中的方向、归属和邀请语气。"
    )


def _normalize_english_translation(text: str, source, target) -> str:
    if not _is_english_to_chinese(source, target):
        return text
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:text|markdown|中文)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = re.sub(r"^(?:译文|翻译)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    if len(cleaned) >= 2 and cleaned[0] in "\"“‘" and cleaned[-1] in "\"”’":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def expand_english_glossary_aliases(texts, glossary) -> dict[str, str]:
    expanded = dict(glossary or {})
    if not expanded:
        return expanded
    content = "\n".join(str(text or "") for text in texts)
    alias_stop_words = {
        "big",
        "black",
        "little",
        "new",
        "old",
        "the",
        "white",
        "young",
    }
    for term, target in list(expanded.items()):
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", str(term))
        if len(words) < 2:
            continue
        without_full_name = re.sub(
            r"(?<!\w)" + re.escape(str(term)) + r"(?!\w)",
            " ",
            content,
            flags=re.IGNORECASE,
        )
        for alias in (words[0], words[-1]):
            if len(alias) < 3 or alias.casefold() in alias_stop_words:
                continue
            if re.search(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
                without_full_name,
                re.IGNORECASE,
            ):
                expanded.setdefault(alias, target)
    return expanded


def _apply_glossary_text(text: str, glossary, source_lang=None) -> str:
    if not glossary:
        return text
    lang = getattr(source_lang, "value", source_lang)
    items = sorted(
        ((str(k), str(v)) for k, v in glossary.items() if k),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
    for key, value in items:
        if _CJK_RE.search(key):
            text = text.replace(key, value)
            continue
        pattern = r"(?<!\w)" + re.escape(key) + r"(?!\w)"
        flags = re.IGNORECASE if lang == "en" else 0
        text = re.sub(pattern, lambda _match: value, text, flags=flags)
    return text


class BaseRemoteTranslator(BaseTranslator):
    """远程翻译基类：提供批量翻译与回退链"""

    _client: Optional[httpx.Client] = None

    def get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        return self._client

    def _request(self, method: str, url: str, **kwargs):
        self._request_count = int(getattr(self, "_request_count", 0)) + 1
        return self.get_client().request(method, url, **kwargs)

    def translate_batch(self, texts, source_lang, target_lang, glossary=None, progress_cb=None):
        out = []
        total = len(texts)
        for i, t in enumerate(texts):
            src = _apply_glossary_text(t, glossary, source_lang)
            out.append(self.translate_one(src, source_lang, target_lang))
            if progress_cb and total:
                progress_cb((i + 1) / total)
        return out

    def translate_one(self, text: str, source: LangCode, target: LangCode) -> str:
        raise NotImplementedError


class GoogleTranslator(BaseRemoteTranslator):
    """使用 translate.googleapis.com 的免费接口

    采用 `dict-chrome-ex` 客户端标识，比 `gtx` 更稳定（后者常被限流）。
    """

    name = "google"

    # 客户端标识轮换，主用 dict-chrome-ex（实测更稳定）
    _CLIENTS = ["dict-chrome-ex", "gtx"]

    def __init__(self):
        self.settings = get_settings()

    def translate_one(self, text: str, source: LangCode, target: LangCode) -> str:
        if not text.strip() or KEEP_PATTERN.match(text):
            return text
        base_params = {
            "sl": provider_language("google", source),
            "tl": provider_language("google", target),
            "dt": "t",
            "q": text,
        }
        last_error = None
        for client in self._CLIENTS:
            params = {**base_params, "client": client}
            try:
                resp = self._request(
                    "GET",
                    self.settings.google_translate_base, params=params
                )
                if resp.status_code == 429:
                    last_error = "rate limited"
                    continue
                resp.raise_for_status()
                data = resp.json()
                sentences = []
                for seg in data[0]:
                    if seg and seg[0]:
                        sentences.append(seg[0])
                result = "".join(sentences).strip()
                if result:
                    return result
            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                continue
        return ""


class MyMemoryTranslator(BaseRemoteTranslator):
    """MyMemory 免费翻译接口（备用，英语质量较好）"""

    name = "mymemory"
    url = "https://api.mymemory.translated.net/get"

    def __init__(self):
        self.settings = get_settings()

    def translate_one(self, text: str, source: LangCode, target: LangCode) -> str:
        if not text.strip() or KEEP_PATTERN.match(text):
            return text
        langpair = f"{provider_language('mymemory', source)}|{provider_language('mymemory', target)}"
        try:
            resp = self._request("GET", self.url, params={"q": text, "langpair": langpair})
            resp.raise_for_status()
            data = resp.json()
            result = data.get("responseData", {}).get("translatedText", "")
            # MyMemory 失败时返回 QUERY LENGTH LIMIT 等
            if result and "QUERY LENGTH" not in result and result != text:
                return result
            return ""
        except Exception:
            return ""


class DeepLTranslator(BaseRemoteTranslator):
    """DeepL API 翻译（需要 MANGA_DEEPL_AUTH_KEY）"""

    name = "deepl"
    endpoint = "https://api-free.deepl.com/v2/translate"
    batch = True

    def __init__(self):
        self.settings = get_settings()

    def translate_one(self, text: str, source: LangCode, target: LangCode) -> str:
        if not text.strip() or KEEP_PATTERN.match(text):
            return text
        target_deepl = provider_language("deepl", target)
        payload = {"text": [text], "target_lang": target_deepl}
        if source != target:
            payload["source_lang"] = provider_language("deepl", source)
        try:
            resp = self._request("POST",
                self.endpoint,
                json=payload,
                headers={"Authorization": f"DeepL-Auth-Key {self.settings.deepl_auth_key}"},
                timeout=25,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("translations"):
                return data["translations"][0]["text"]
            return ""
        except Exception:
            return ""

    def translate_batch(self, texts, source: LangCode, target: LangCode, glossary=None, progress_cb=None):
        """一次 API 调用批量翻译（DeepL 支持数组输入），空文本/拟声词保留原样"""
        result = list(texts)
        indices: list[int] = []
        payload_texts: list[str] = []
        for i, t in enumerate(texts):
            if not t.strip() or KEEP_PATTERN.match(t):
                continue
            indices.append(i)
            payload_texts.append(t)
        if not payload_texts:
            self.last_failed_indices = []
            self.last_batch_failures = []
            return result
        self.last_batch_failures = []
        target_deepl = provider_language("deepl", target)
        payload = {"text": payload_texts, "target_lang": target_deepl}
        if source != target:
            payload["source_lang"] = provider_language("deepl", source)
        try:
            resp = self._request("POST",
                self.endpoint,
                json=payload,
                headers={"Authorization": f"DeepL-Auth-Key {self.settings.deepl_auth_key}"},
                timeout=25,
            )
            resp.raise_for_status()
            data = resp.json()
            succeeded = set()
            for j, item in enumerate(data.get("translations") or []):
                if j < len(indices) and item.get("text"):
                    result[indices[j]] = item["text"]
                    succeeded.add(indices[j])
            self.last_failed_indices = [index for index in indices if index not in succeeded]
        except Exception as exc:
            self.last_failed_indices = list(indices)
            self.last_batch_failures.append(str(exc))
        return result


class DeepSeekTranslator(BaseRemoteTranslator):
    """DeepSeek API 翻译（需要 MANGA_DEEPSEEK_API_KEY）

    使用 deepseek-chat（或 deepseek-v3），对漫画口语/网络用语理解力优于 DeepL。
    页级上下文批量：整页全部文字合并成一次请求，结合语境翻译更地道。
    """

    name = "deepseek"
    prompt_glossary = True
    batch = True
    CONTEXT_MAX_SEGMENTS = 24
    CONTEXT_MAX_CHARS = 6000

    def __init__(self):
        self.settings = get_settings()

    def _model_candidates(self, source: LangCode, target: LangCode) -> list[str]:
        primary = (
            self.settings.deepseek_english_model
            if _is_english_to_chinese(source, target)
            else self.settings.deepseek_model
        )
        models = [primary or self.settings.deepseek_model]
        if self.settings.deepseek_model and self.settings.deepseek_model not in models:
            models.append(self.settings.deepseek_model)
        return models

    @staticmethod
    def _sampling_options(model: str, source: LangCode, target: LangCode) -> dict:
        options = {"temperature": 0.7 if _is_english_to_chinese(source, target) else 0.3}
        if model.startswith("deepseek-v4"):
            options["thinking"] = {"type": "disabled"}
        return options

    def _system_prompt(self, source: LangCode, target: LangCode, glossary=None, texts=None) -> str:
        prompt = build_manga_prompt(source, target)
        if glossary:
            mapping = "、".join(f"{k}→{v}" for k, v in glossary.items() if k and v)
            if mapping:
                prompt += f"人名对照：{mapping}"
        if _is_english_to_chinese(source, target):
            prompt = _append_english_entity_hints(prompt, texts or [])
        if _is_japanese_to_chinese(source, target):
            prompt = _append_japanese_entity_hints(prompt, texts or [])
            prompt = _append_japanese_manga_term_hints(prompt, texts or [])
        return prompt

    def _context_prompt(self, source: LangCode, target: LangCode, glossary, n: int, texts=None) -> str:
        """页级批量系统提示：整页 n 段文字一次翻译，结合上下文"""
        prompt = build_manga_prompt(source, target, context_count=n)
        if glossary:
            mapping = "、".join(f"{k}→{v}" for k, v in glossary.items() if k and v)
            if mapping:
                prompt += f"人名对照：{mapping}"
        if _is_english_to_chinese(source, target):
            prompt = _append_english_entity_hints(prompt, texts or [])
        if _is_japanese_to_chinese(source, target):
            prompt = _append_japanese_entity_hints(prompt, texts or [])
            prompt = _append_japanese_manga_term_hints(prompt, texts or [])
        return prompt

    @staticmethod
    def _parse_segments(content: str, n: int):
        """解析 <i>...</i> 编号段；序号缺失/重复/多余时返回 None"""
        content = re.sub(r"```[a-zA-Z]*", "", content).strip()
        pairs = re.findall(r"<(\d+)>\s*(.*?)\s*</\1>", content, re.S)
        seg: dict[int, str] = {}
        for num, txt in pairs:
            key = int(num)
            if key in seg:
                return None
            seg[key] = txt.strip()
        if sorted(seg.keys()) != list(range(1, n + 1)):
            return None
        return [seg[i] for i in range(1, n + 1)]

    def _translate_context(self, srcs, source: LangCode, target: LangCode, glossary=None):
        """整页 n 段一次请求，返回与 srcs 等长的译文列表；解析失败抛异常"""
        base = self.settings.deepseek_base_url or "https://api.deepseek.com"
        numbered = "".join(f"<{i + 1}>{t}</{i + 1}>" for i, t in enumerate(srcs))
        last_error = None
        for model in self._model_candidates(source, target):
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._context_prompt(
                                source, target, glossary, len(srcs), texts=srcs
                            ),
                        },
                        {"role": "user", "content": numbered},
                    ],
                    **self._sampling_options(model, source, target),
                }
                resp = self._request("POST",
                    f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"},
                    json=payload,
                    timeout=90,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                segs = self._parse_segments(content, len(srcs))
                if segs is None:
                    raise ValueError(f"段数不匹配: 期望 {len(srcs)} 段")
                return [_normalize_english_translation(text, source, target) for text in segs]
            except Exception as exc:
                last_error = exc
        raise last_error or RuntimeError("DeepSeek 请求失败")

    def translate_batch(self, texts, source_lang, target_lang, glossary=None, progress_cb=None):
        """页级上下文批量：按安全大小分块；只把失败分块交给 SmartTranslator 回退。"""
        idx_map = [i for i, t in enumerate(texts) if t.strip() and not KEEP_PATTERN.match(t)]
        if not idx_map:
            self.last_failed_indices = []
            self.last_batch_failures = []
            return list(texts)
        srcs = [texts[i] for i in idx_map]
        chunks: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for text in srcs:
            item_chars = len(text) + 16
            if current and (
                len(current) >= self.CONTEXT_MAX_SEGMENTS
                or current_chars + item_chars > self.CONTEXT_MAX_CHARS
            ):
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(text)
            current_chars += item_chars
        if current:
            chunks.append(current)

        out = list(texts)
        self.last_failed_indices = []
        self.last_batch_failures = []
        source_offset = 0
        for index, chunk in enumerate(chunks):
            chunk_indices = idx_map[source_offset:source_offset + len(chunk)]
            source_offset += len(chunk)
            try:
                translated = self._translate_context(chunk, source_lang, target_lang, glossary)
                if len(translated) != len(chunk):
                    raise ValueError(f"段数不匹配: 期望 {len(chunk)} 段")
                for original_index, value in zip(chunk_indices, translated):
                    if value.strip():
                        out[original_index] = value
                    else:
                        self.last_failed_indices.append(original_index)
            except Exception as exc:
                self.last_failed_indices.extend(chunk_indices)
                self.last_batch_failures.append(str(exc))
            if progress_cb:
                progress_cb((index + 1) / len(chunks))
        return out

    def translate_one(self, text: str, source: LangCode, target: LangCode, glossary=None) -> str:
        if not text.strip() or KEEP_PATTERN.match(text):
            return text
        base = self.settings.deepseek_base_url or "https://api.deepseek.com"
        for model in self._model_candidates(source, target):
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._system_prompt(
                                source, target, glossary=glossary, texts=[text]
                            ),
                        },
                        {"role": "user", "content": text},
                    ],
                    **self._sampling_options(model, source, target),
                }
                resp = self._request("POST",
                    f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"},
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                return _normalize_english_translation(
                    resp.json()["choices"][0]["message"]["content"], source, target
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[deepseek] 翻译失败: {exc}")
        return ""

    def polish_batch(self, texts, source: LangCode, target: LangCode, style: str, custom_prompt: str = "") -> list[str]:
        """使用现有 DeepSeek 接口润色已完成译文，不重新执行翻译/OCR。"""
        if not texts:
            return []
        style_text = custom_prompt.strip() if style == "custom" else {
            "natural": "自然：清晰流畅，符合中文漫画对白习惯",
            "colloquial": "口语化：像角色真实说话，轻松自然",
            "passionate": "热血：增强战斗感和情绪张力，但不夸大原意",
            "funny": "搞笑：保留原意，适度体现漫画喜感",
            "formal": "正式：表达克制、准确、规范，但仍适合漫画对白",
        }.get(style, "自然：清晰流畅，符合中文漫画对白习惯")
        if not style_text:
            raise ValueError("自定义润色提示词不能为空")
        prompt = (
            "你是专业的中文漫画对白润色编辑。下面是已经翻译完成的中文译文，按阅读顺序使用<序号>...</序号>包裹。"
            f"润色风格：{style_text}。保持原意、人物关系、语气和情绪，不添加原文不存在的信息；"
            "符合漫画对白表达，尽量简洁，避免明显增加文本长度。不要重新翻译，不要解释，不要添加引号、Markdown或提示语。"
            f"只返回{len(texts)}段润色后的文本，严格使用<序号>润色文本</序号>格式，不得遗漏、合并、拆分或改变编号。"
        )
        numbered = "".join(f"<{i + 1}>{text}</{i + 1}>" for i, text in enumerate(texts))
        base = self.settings.deepseek_base_url or "https://api.deepseek.com"
        last_error = None
        for model in self._model_candidates(source, target):
            try:
                resp = self._request(
                    "POST", f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"},
                    json={"model": model, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": numbered}], "temperature": 0.4},
                    timeout=90,
                )
                resp.raise_for_status()
                result = self._parse_segments(resp.json()["choices"][0]["message"]["content"], len(texts))
                if result is None or any(not item.strip() for item in result):
                    raise ValueError("润色结果段数不一致或包含空文本")
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise last_error or RuntimeError("DeepSeek 润色请求失败")


class OpenAITranslator(BaseRemoteTranslator):
    """OpenAI 兼容接口翻译（需要 MANGA_OPENAI_API_KEY）"""

    name = "openai"
    prompt_glossary = True

    def __init__(self):
        self.settings = get_settings()

    def _system_prompt(self, source: LangCode, target: LangCode, glossary=None, texts=None) -> str:
        prompt = build_manga_prompt(source, target)
        if glossary:
            mapping = "、".join(f"{k}→{v}" for k, v in glossary.items() if k and v)
            if mapping:
                prompt += f"人名对照：{mapping}"
        if _is_english_to_chinese(source, target):
            prompt = _append_english_entity_hints(prompt, texts or [])
        if _is_japanese_to_chinese(source, target):
            prompt = _append_japanese_entity_hints(prompt, texts or [])
        return prompt

    def translate_one(self, text: str, source: LangCode, target: LangCode, glossary=None) -> str:
        if not text.strip() or KEEP_PATTERN.match(text):
            return text
        try:
            resp = self._request("POST",
                f"{self.settings.openai_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json={
                    "model": self.settings.openai_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._system_prompt(
                                source, target, glossary=glossary, texts=[text]
                            ),
                        },
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.3,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return _normalize_english_translation(
                resp.json()["choices"][0]["message"]["content"], source, target
            )
        except Exception:
            return ""


class SmartTranslator(BaseTranslator):
    """智能翻译器：按配置选择后端，失败时自动回退

    回退链顺序：配置后端 → 其他已配置高质量后端 → google → mymemory → 原文
    """

    name = "smart"

    def __init__(self):
        self.settings = get_settings()
        self._backends: list[BaseRemoteTranslator] = []
        self._available_by_direction: dict[tuple[str, str], BaseRemoteTranslator] = {}
        self._translation_cache = OrderedDict()
        self.last_backend_names: list[str] = []
        self.last_failures: list[str] = []
        self.last_performance: dict = {}
        self._last_backend_name = ""
        self._attempted_current: list[str] = []
        self._build_chain()

    def _build_chain(self) -> None:
        s = self.settings
        preferred = s.translator_backend

        configured = {
            "deepseek": (bool(s.deepseek_api_key), DeepSeekTranslator),
            "openai": (bool(s.openai_api_key), OpenAITranslator),
            "deepl": (bool(s.deepl_auth_key), DeepLTranslator),
        }
        order = [preferred] + [name for name in ("deepseek", "openai", "deepl") if name != preferred]
        for name in order:
            item = configured.get(name)
            if item and item[0]:
                self._backends.append(item[1]())
        self._backends.extend([GoogleTranslator(), MyMemoryTranslator()])

    @staticmethod
    def _apply_glossary(text: str, glossary, source_lang=None) -> str:
        """应用词典：CJK 最长匹配；拉丁词仍用整词边界"""
        return _apply_glossary_text(text, glossary, source_lang)

    def _ensure_runtime_state(self) -> None:
        if not hasattr(self, "settings"):
            self.settings = get_settings()
        if not hasattr(self, "_translation_cache"):
            self._translation_cache = OrderedDict()
        if not hasattr(self, "_available_by_direction"):
            self._available_by_direction = {}
            legacy_backend = getattr(self, "_available", None)
            if legacy_backend is not None:
                self._available_by_direction[("en", "zh")] = legacy_backend

    @staticmethod
    def _glossary_fingerprint(glossary) -> tuple:
        return tuple(sorted((str(key), str(value)) for key, value in (glossary or {}).items()))

    def _cache_key(self, texts, source, target, glossary, backend) -> tuple:
        return (
            str(source), str(target), tuple(str(text) for text in texts),
            self._glossary_fingerprint(glossary),
        )

    def _cache_get(self, key):
        value = self._translation_cache.get(key)
        if value is None:
            return None
        self._translation_cache.move_to_end(key)
        return list(value[0]), list(value[1])

    def _cache_put(self, key, out, names) -> None:
        limit = max(0, int(getattr(self.settings, "translation_cache_size", 512)))
        if limit <= 0 or any(name in {"original", "glossary"} for name in names):
            return
        self._translation_cache[key] = (tuple(out), tuple(names))
        self._translation_cache.move_to_end(key)
        while len(self._translation_cache) > limit:
            self._translation_cache.popitem(last=False)

    def _request_total(self) -> int:
        return sum(int(getattr(backend, "_request_count", 0)) for backend in self._backends)

    def _attempt(self, backend) -> None:
        name = getattr(backend, "name", type(backend).__name__)
        if name not in self._attempted_current:
            self._attempted_current.append(name)

    def _finish_performance(self, request_before, cache_hits=0) -> None:
        primary_name = self._attempted_current[0] if self._attempted_current else ""
        self.last_performance = {
            "request_count": max(0, self._request_total() - request_before),
            "cache_hits": int(cache_hits),
            "fallback": bool(
                self.last_failures
                or len(self._attempted_current) > 1
                or any(name not in {primary_name, "keep"} for name in self.last_backend_names)
            ),
            "backends_attempted": list(self._attempted_current),
        }

    def translate_batch(self, texts, source_lang, target_lang, glossary=None, progress_cb=None):
        self._ensure_runtime_state()
        self.last_backend_names = []
        self.last_failures = []
        self._attempted_current = []
        request_before = self._request_total()
        # 找出第一个可用的后端（发送探针，结果缓存）
        direction = (str(source_lang), str(target_lang))
        legacy_backend = getattr(self, "_available", None)
        if legacy_backend is not None and direction not in self._available_by_direction:
            self._available_by_direction[direction] = legacy_backend
        if direction not in self._available_by_direction:
            available = self._find_available(source_lang, target_lang)
            self._available_by_direction[direction] = available
        backend = self._available_by_direction.get(direction)
        if backend is None:
            # 全部失败：仅应用词典
            self.last_backend_names = ["glossary"] * len(texts)
            out = self._apply_glossary_only(texts, glossary, source_lang)
            self._finish_performance(request_before)
            return out

        use_prompt = getattr(backend, "prompt_glossary", False)
        srcs = list(texts) if use_prompt else [
            self._apply_glossary(t, glossary, source_lang) for t in texts
        ]
        cache_key = self._cache_key(texts, source_lang, target_lang, glossary, backend)
        cached = self._cache_get(cache_key)
        if cached is not None:
            out, self.last_backend_names = cached
            if progress_cb:
                progress_cb(1.0)
            self._finish_performance(request_before, cache_hits=len(out))
            return out

        # 支持批量 API 的后端（DeepL 数组 / DeepSeek 页级上下文）一次请求完成
        if getattr(backend, "batch", False):
            self._attempt(backend)
            try:
                out = backend.translate_batch(
                    srcs,
                    source_lang,
                    target_lang,
                    glossary=glossary if use_prompt else None,
                    progress_cb=progress_cb,
                )
                for failure in getattr(backend, "last_batch_failures", []) or []:
                    self.last_failures.append(f"{getattr(backend, 'name', 'unknown')}: {failure}")
                if out and len(out) == len(srcs):
                    out = [
                        _normalize_english_translation(item, source_lang, target_lang)
                        for item in out
                    ]
                    if glossary:
                        out = [self._apply_glossary(item, glossary, source_lang) for item in out]
                    backend_name = getattr(backend, "name", type(backend).__name__)
                    self.last_backend_names = [backend_name] * len(srcs)
                    failed_indices = sorted(set(getattr(backend, "last_failed_indices", []) or []))
                    for failed_index in failed_indices:
                        if not (0 <= failed_index < len(srcs)):
                            continue
                        result = self._translate_with_fallback(
                            srcs[failed_index], source_lang, target_lang, backend,
                            glossary=glossary if use_prompt else None,
                        )
                        out[failed_index] = result
                        self.last_backend_names[failed_index] = self._last_backend_name or "original"
                    if progress_cb:
                        progress_cb(1.0)
                    self._cache_put(cache_key, out, self.last_backend_names)
                    self._finish_performance(request_before)
                    return out
            except Exception as e:  # noqa: BLE001
                self.last_failures.append(f"{getattr(backend, 'name', 'unknown')}: {e}")
                print(f"[translator] 批量翻译失败，逐条回退: {e}")
            # 批量失败 → 落到下面的逐条回退链

        # 其余后端逐条翻译，保留错误时的回退
        out = []
        total = len(srcs)
        for i, src in enumerate(srcs):
            result = self._translate_with_fallback(
                src, source_lang, target_lang, backend, glossary=glossary if use_prompt else None
            )
            if glossary:
                result = self._apply_glossary(result, glossary, source_lang)
            out.append(result)
            self.last_backend_names.append(self._last_backend_name or "original")
            if progress_cb and total:
                progress_cb((i + 1) / total)
        # 整批全部失败（输出等于原文）→ 清除缓存的后端，下次任务重新探测
        if srcs and all(o == s for o, s in zip(out, srcs)):
            self._available_by_direction[direction] = None
        else:
            successful = next((name for name in self.last_backend_names if name not in {"original", "keep"}), "")
            replacement = next((item for item in self._backends if getattr(item, "name", "") == successful), None)
            if replacement is not None:
                self._available_by_direction[direction] = replacement
        self._cache_put(cache_key, out, self.last_backend_names)
        self._finish_performance(request_before)
        return out

    def _translate_with_fallback(self, text: str, source, target, primary, glossary=None) -> str:
        ordered = [primary] + [b for b in self._backends if b is not primary]
        for backend in ordered:
            if not text.strip():
                self._last_backend_name = "keep"
                return text
            try:
                self._attempt(backend)
                if getattr(backend, "prompt_glossary", False):
                    result = backend.translate_one(text, source, target, glossary=glossary)
                else:
                    src = self._apply_glossary(text, glossary, source) if glossary else text
                    result = backend.translate_one(src, source, target)
                result = _normalize_english_translation(result, source, target)
                if result:
                    self._last_backend_name = getattr(backend, "name", type(backend).__name__)
                    return result
            except Exception as e:
                self.last_failures.append(f"{getattr(backend, 'name', 'unknown')}: {e}")
                continue
        self._last_backend_name = "original"
        return text

    def _find_available(self, source, target):
        for backend in self._backends:
            try:
                self._attempt(backend)
                probe = {"ja": "こんにちは", "en": "hello", "zh": "你好"}.get(source, "hello")
                result = backend.translate_one(probe, source, target)
                if result:
                    return backend
            except Exception:
                continue
        return None

    def _apply_glossary_only(self, texts, glossary, source_lang=None):
        return [self._apply_glossary(t, glossary, source_lang) for t in texts]


def create_translator() -> BaseTranslator:
    return SmartTranslator()
