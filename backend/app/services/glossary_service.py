"""专有名词（词典）服务"""
from __future__ import annotations

import json
from typing import Optional

from app.models.schemas import GlossaryItem
from app.storage.database import Database

# 内置词库：常见动漫专有名词（日语→中文）
BUILTIN_GLOSSARY = [
    {"source": "ナルト", "target": "鸣人", "lang": "ja", "note": "火影忍者主角"},
    {"source": "サスケ", "target": "佐助", "lang": "ja", "note": "火影忍者角色"},
    {"source": "ワンピース", "target": "海贼王", "lang": "ja", "note": "航海王"},
    {"source": "ルフィ", "target": "路飞", "lang": "ja", "note": "海贼王主角"},
    {"source": "ドラゴンボール", "target": "龙珠", "lang": "ja", "note": "七龙珠"},
    {"source": "ポケモン", "target": "宝可梦", "lang": "ja", "note": "精灵宝可梦"},
    {"source": "ガンダム", "target": "高达", "lang": "ja", "note": "机动战士"},
    {"source": "センセイ", "target": "老师", "lang": "ja", "note": "称呼"},
    {"source": "くん", "target": "君", "lang": "ja", "note": "称呼后缀"},
    {"source": "ちゃん", "target": "酱", "lang": "ja", "note": "称呼后缀"},
]

BUILTIN_ENGLISH_GLOSSARY = {
    "Ken Takakura": "高仓健",
}


class GlossaryService:
    """专有名词管理，含内置词库 + 用户自定义"""

    def __init__(self):
        self.db = Database.get_instance()

    def _ensure_builtin(self) -> None:
        """首次启动时导入内置词库"""
        existing = self.db.glossary_list()
        if existing:
            return
        for item in BUILTIN_GLOSSARY:
            self.db.glossary_create(
                item["source"], item["target"], item["lang"], item.get("note", ""), "zh"
            )

    def list_items(self, lang: Optional[str] = None, search: str = "") -> list[dict]:
        self._ensure_builtin()
        return self.db.glossary_list(lang=lang, search=search)

    def create(
        self, source: str, target: str, lang: str, note: str = "", target_lang: str = "zh"
    ) -> tuple[int, str]:
        """返回 (id, 消息)。id=-1 表示重复"""
        self._ensure_builtin()
        item_id = self.db.glossary_create(source, target, lang, note, target_lang)
        if item_id == -1:
            return -1, "该词条已存在"
        return item_id, "已添加"

    def update(
        self, item_id: int, source: str, target: str, lang: str, note: str = "", target_lang: str = "zh"
    ) -> tuple[bool, str]:
        ok = self.db.glossary_update(item_id, source, target, lang, note, target_lang)
        return ok, "已更新" if ok else "更新失败或词条冲突"

    def delete(self, item_id: int) -> bool:
        return self.db.glossary_delete(item_id)

    def get_mapping(self, lang: str, target_lang: str = "zh") -> dict[str, str]:
        """返回 {source: target} 映射（用户词库 + 内置）"""
        self._ensure_builtin()
        mapping = self.db.glossary_get_mapping(lang, target_lang)
        if lang == "en" and target_lang == "zh":
            return {**BUILTIN_ENGLISH_GLOSSARY, **mapping}
        return mapping

    def import_json(self, content: str) -> dict:
        """从 JSON 导入词条：[{source,target,lang,note}, ...]"""
        errors: list[str] = []
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return {"imported": 0, "skipped": 0, "errors": [f"JSON 格式错误: {e}"]}

        if not isinstance(data, list):
            return {"imported": 0, "skipped": 0, "errors": ["JSON 必须为数组"]}

        valid_items = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"第{i+1}项不是对象")
                continue
            src = str(item.get("source", "")).strip()
            tgt = str(item.get("target", "")).strip()
            if not src or not tgt:
                errors.append(f"第{i+1}项缺少 source 或 target")
                continue
            valid_items.append(
                {
                    "source": src,
                    "target": tgt,
                    "lang": str(item.get("lang", "ja")),
                    "target_lang": str(item.get("target_lang", "zh")),
                    "note": str(item.get("note", "")),
                }
            )

        imported, skipped = self.db.glossary_import(valid_items)
        return {"imported": imported, "skipped": skipped, "errors": errors}
