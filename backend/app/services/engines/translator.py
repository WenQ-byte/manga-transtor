"""翻译引擎：支持多种后端 + 自动降级回退链

后端优先级（可配置 translator_backend）：
  1. deepseek  - 深度求索 API（漫画口语最自然，需要 MANGA_DEEPSEEK_API_KEY）
  2. google    - translate.googleapis.com（免费，质量一般）
  3. mymemory  - api.mymemory.translated.net（备用免费接口）
  4. deepl     - 需要 MANGA_DEEPL_AUTH_KEY（口语翻译较差）
  5. openai    - 需要 MANGA_OPENAI_API_KEY

任意后端失败时自动回退到下一个可用后端，全部失败则返回原文
（专有名词词典替换始终生效）。
"""
from __future__ import annotations

import re
from typing import Optional

import httpx

from app.config import get_settings
from app.models.schemas import LangCode
from app.services.engines.base import BaseTranslator

LANG_MAP = {
    "ja": "ja",
    "en": "en",
    "zh": "zh-CN",
}

# 文本中常见的拟声词/符号，保留不翻译
KEEP_PATTERN = re.compile(r"^[\s~！!？?…☆★♪♫◎○●▲△□■]+$")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


class BaseRemoteTranslator(BaseTranslator):
    """远程翻译基类：提供批量翻译与回退链"""

    _client: Optional[httpx.Client] = None

    def get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        return self._client

    def translate_batch(self, texts, source_lang, target_lang, glossary=None, progress_cb=None):
        out = []
        total = len(texts)
        for i, t in enumerate(texts):
            src = t
            if glossary:
                for k, v in glossary.items():
                    if k and k in src:
                        src = src.replace(k, v)
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
            "sl": LANG_MAP.get(source, source),
            "tl": LANG_MAP.get(target, target),
            "dt": "t",
            "q": text,
        }
        last_error = None
        for client in self._CLIENTS:
            params = {**base_params, "client": client}
            try:
                resp = self.get_client().get(
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
        langpair = f"{LANG_MAP.get(source, source)}|{LANG_MAP.get(target, target)}"
        try:
            resp = self.get_client().get(self.url, params={"q": text, "langpair": langpair})
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
        target_deepl = "ZH" if target == "zh" else target.upper()
        payload = {"text": [text], "target_lang": target_deepl}
        if source != target:
            payload["source_lang"] = source.upper()
        try:
            resp = httpx.post(
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

    def translate_batch(self, texts, source: LangCode, target: LangCode):
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
            return result
        target_deepl = "ZH" if target == "zh" else target.upper()
        payload = {"text": payload_texts, "target_lang": target_deepl}
        if source != target:
            payload["source_lang"] = source.upper()
        try:
            resp = httpx.post(
                self.endpoint,
                json=payload,
                headers={"Authorization": f"DeepL-Auth-Key {self.settings.deepl_auth_key}"},
                timeout=25,
            )
            resp.raise_for_status()
            data = resp.json()
            for j, item in enumerate(data.get("translations") or []):
                if j < len(indices) and item.get("text"):
                    result[indices[j]] = item["text"]
        except Exception:
            pass
        return result


class DeepSeekTranslator(BaseRemoteTranslator):
    """DeepSeek API 翻译（需要 MANGA_DEEPSEEK_API_KEY）

    使用 deepseek-chat（或 deepseek-v3），对漫画口语/网络用语理解力优于 DeepL。
    """

    name = "deepseek"
    prompt_glossary = True

    def __init__(self):
        self.settings = get_settings()

    def _system_prompt(self, source: LangCode, target: LangCode, glossary=None) -> str:
        lang_names = {"ja": "日语", "en": "英语", "zh": "中文"}
        prompt = (
            f"你是专业的漫画翻译。将以下{lang_names.get(source, source)}文本翻译成{lang_names.get(target, target)}。"
            "要求：1) 自然流畅，符合漫画对话口吻（口语化、接地气）；"
            "2) 保留人名、专有名词的统一译法；"
            "3) 语气词、感叹词用中文常用表达（如「嗯」「诶」「诶嘿」）；"
            "4) 原文有几行，译文也要几行，用换行对齐；"
            "5) 不要添加原文没有的内容，看不懂就按字面直译，禁止脑补；"
            "6) 只输出译文本身，不要任何解释或引号。"
        )
        if glossary:
            mapping = "、".join(f"{k}→{v}" for k, v in glossary.items() if k and v)
            if mapping:
                prompt += f"人名对照：{mapping}"
        return prompt

    def translate_one(self, text: str, source: LangCode, target: LangCode, glossary=None) -> str:
        if not text.strip() or KEEP_PATTERN.match(text):
            return text
        base = self.settings.deepseek_base_url or "https://api.deepseek.com"
        model = self.settings.deepseek_model or "deepseek-chat"
        try:
            resp = httpx.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self._system_prompt(source, target, glossary=glossary)},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.3,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""


class OpenAITranslator(BaseRemoteTranslator):
    """OpenAI 兼容接口翻译（需要 MANGA_OPENAI_API_KEY）"""

    name = "openai"
    prompt_glossary = True

    def __init__(self):
        self.settings = get_settings()

    def _system_prompt(self, source: LangCode, target: LangCode, glossary=None) -> str:
        lang_names = {"ja": "日语", "en": "英语", "zh": "中文"}
        prompt = (
            f"你是专业的漫画翻译。将以下{lang_names.get(source, source)}文本翻译成{lang_names.get(target, target)}。"
            "要求：1) 自然流畅，符合漫画对话口吻；2) 保留人名、专有名词的统一译法；"
            "3) 原文有几行，译文也要几行，用换行对齐；"
            "4) 不要添加原文没有的内容，看不懂就按字面直译，禁止脑补；"
            "5) 只输出译文本身，不要任何解释或引号。"
        )
        if glossary:
            mapping = "、".join(f"{k}→{v}" for k, v in glossary.items() if k and v)
            if mapping:
                prompt += f"人名对照：{mapping}"
        return prompt

    def translate_one(self, text: str, source: LangCode, target: LangCode, glossary=None) -> str:
        if not text.strip() or KEEP_PATTERN.match(text):
            return text
        try:
            resp = httpx.post(
                f"{self.settings.openai_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json={
                    "model": self.settings.openai_model,
                    "messages": [
                        {"role": "system", "content": self._system_prompt(source, target, glossary=glossary)},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.3,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""


class SmartTranslator(BaseTranslator):
    """智能翻译器：按配置选择后端，失败时自动回退

    回退链顺序：配置后端 → google → mymemory → 原文
    """

    name = "smart"

    def __init__(self):
        self.settings = get_settings()
        self._backends: list[BaseRemoteTranslator] = []
        self._available: Optional[BaseRemoteTranslator] = None
        self._build_chain()

    def _build_chain(self) -> None:
        s = self.settings
        preferred = s.translator_backend

        # 配置的后端优先
        if preferred == "deepseek" and s.deepseek_api_key:
            self._backends.append(DeepSeekTranslator())
        elif preferred == "deepl" and s.deepl_auth_key:
            self._backends.append(DeepLTranslator())
        elif preferred == "openai" and s.openai_api_key:
            self._backends.append(OpenAITranslator())

        # 免费后端（google 为主，mymemory 备选）
        self._backends.append(GoogleTranslator())
        self._backends.append(MyMemoryTranslator())

        # 有 key 的付费后端作为补充
        if preferred != "deepseek" and s.deepseek_api_key:
            self._backends.append(DeepSeekTranslator())
        if preferred != "deepl" and s.deepl_auth_key:
            self._backends.append(DeepLTranslator())
        if preferred != "openai" and s.openai_api_key:
            self._backends.append(OpenAITranslator())

    @staticmethod
    def _apply_glossary(text: str, glossary) -> str:
        """应用词典：CJK 最长匹配；拉丁词仍用整词边界"""
        if not glossary:
            return text
        items = sorted(((k, v) for k, v in glossary.items() if k), key=lambda kv: len(kv[0]), reverse=True)
        for k, v in items:
            if _CJK_RE.search(k):
                text = text.replace(k, v)
            else:
                text = re.sub(r"(?<!\w)" + re.escape(k) + r"(?!\w)", v, text)
        return text

    def translate_batch(self, texts, source_lang, target_lang, glossary=None, progress_cb=None):
        # 找出第一个可用的后端（发送探针，结果缓存）
        if self._available is None:
            self._available = self._find_available(source_lang, target_lang)
        backend = self._available
        if backend is None:
            # 全部失败：仅应用词典
            return self._apply_glossary_only(texts, glossary)

        use_prompt = getattr(backend, "prompt_glossary", False)
        srcs = list(texts) if use_prompt else [self._apply_glossary(t, glossary) for t in texts]

        # 支持批量 API 的后端（DeepL）一次性发送，减少请求数
        if getattr(backend, "batch", False):
            out = backend.translate_batch(srcs, source_lang, target_lang)
            if progress_cb:
                progress_cb(1.0)
            return out

        # 其余后端逐条翻译，保留错误时的回退
        out = []
        total = len(srcs)
        for i, src in enumerate(srcs):
            result = self._translate_with_fallback(
                src, source_lang, target_lang, backend, glossary=glossary if use_prompt else None
            )
            out.append(result)
            if progress_cb and total:
                progress_cb((i + 1) / total)
        return out

    def _translate_with_fallback(self, text: str, source, target, primary, glossary=None) -> str:
        ordered = [primary] + [b for b in self._backends if b is not primary]
        for backend in ordered:
            if not text.strip():
                return text
            try:
                if getattr(backend, "prompt_glossary", False):
                    result = backend.translate_one(text, source, target, glossary=glossary)
                else:
                    src = self._apply_glossary(text, glossary) if glossary else text
                    result = backend.translate_one(src, source, target)
                if result:
                    return result
            except Exception:
                continue
        return text

    def _find_available(self, source, target):
        for backend in self._backends:
            try:
                probe = "hello" if source != "zh" else "你好"
                result = backend.translate_one(probe, source, target)
                if result:
                    return backend
            except Exception:
                continue
        return None

    def _apply_glossary_only(self, texts, glossary):
        return [self._apply_glossary(t, glossary) for t in texts]


def create_translator() -> BaseTranslator:
    return SmartTranslator()
