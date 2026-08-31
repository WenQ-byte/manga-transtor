"""SQLite 数据访问层"""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Optional

from app.config import get_settings


class Database:
    """SQLite 数据库封装，线程安全"""

    _instance: Optional["Database"] = None
    _lock = threading.Lock()

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._init_schema()

    @classmethod
    def get_instance(cls) -> "Database":
        with cls._lock:
            if cls._instance is None:
                settings = get_settings()
                cls._instance = cls(str(settings.db_path))
            return cls._instance

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS glossary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                lang TEXT NOT NULL DEFAULT 'ja',
                target_lang TEXT NOT NULL DEFAULT 'zh',
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                step TEXT NOT NULL DEFAULT 'queued',
                error TEXT NOT NULL DEFAULT '',
                result_path TEXT NOT NULL DEFAULT '',
                original_path TEXT NOT NULL DEFAULT '',
                text_count INTEGER NOT NULL DEFAULT 0,
                duration_ms INTEGER NOT NULL DEFAULT 0,
                meta TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(glossary)").fetchall()}
        if "target_lang" not in columns:
            conn.execute("ALTER TABLE glossary ADD COLUMN target_lang TEXT NOT NULL DEFAULT 'zh'")
        conn.execute("DROP INDEX IF EXISTS idx_glossary_unique")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_glossary_unique "
            "ON glossary (source, target, lang, target_lang)"
        )
        conn.commit()

    # ---- Glossary ----
    def glossary_create(
        self, source: str, target: str, lang: str, note: str, target_lang: str = "zh"
    ) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO glossary (source, target, lang, note, target_lang) VALUES (?,?,?,?,?)",
                (source, target, lang, note, target_lang),
            )
            conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return -1

    def glossary_list(self, lang: Optional[str] = None, search: str = "") -> list[dict[str, Any]]:
        conn = self._connect()
        sql = "SELECT * FROM glossary WHERE 1=1"
        params: list = []
        if lang:
            sql += " AND lang=?"
            params.append(lang)
        if search:
            sql += " AND (source LIKE ? OR target LIKE ?)"
            params += [f"%{search}%", f"%{search}%"]
        sql += " ORDER BY id DESC"
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def glossary_update(
        self, item_id: int, source: str, target: str, lang: str, note: str, target_lang: str = "zh"
    ) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE glossary SET source=?, target=?, lang=?, note=?, target_lang=?, updated_at=datetime('now') WHERE id=?",
                (source, target, lang, note, target_lang, item_id),
            )
            conn.commit()
            return cur.rowcount > 0
        except sqlite3.IntegrityError:
            return False

    def glossary_delete(self, item_id: int) -> bool:
        conn = self._connect()
        cur = conn.execute("DELETE FROM glossary WHERE id=?", (item_id,))
        conn.commit()
        return cur.rowcount > 0

    def glossary_get_mapping(self, lang: str, target_lang: str = "zh") -> dict[str, str]:
        """返回 {source: target} 映射，用于翻译时替换专有名词"""
        conn = self._connect()
        rows = conn.execute(
            "SELECT source, target FROM glossary WHERE lang=? AND target_lang=?",
            (lang, target_lang),
        ).fetchall()
        return {r["source"]: r["target"] for r in rows}

    def glossary_import(self, items: list[dict[str, Any]]) -> tuple[int, int]:
        imported = 0
        skipped = 0
        for it in items:
            rid = self.glossary_create(
                str(it.get("source", "")).strip(),
                str(it.get("target", "")).strip(),
                str(it.get("lang", "ja")),
                str(it.get("note", "")),
                str(it.get("target_lang", "zh")),
            )
            if rid == -1:
                skipped += 1
            else:
                imported += 1
        return imported, skipped

    # ---- Tasks ----
    def task_create(
        self,
        task_id: str,
        source_lang: str,
        target_lang: str,
        original_path: str = "",
        meta: Optional[dict] = None,
    ) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT INTO tasks (id, source_lang, target_lang, status, original_path, meta) VALUES (?,?,?,?,?,?)",
            (task_id, source_lang, target_lang, "queued", original_path, json.dumps(meta or {})),
        )
        conn.commit()

    def task_update(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        step: Optional[str] = None,
        error: Optional[str] = None,
        result_path: Optional[str] = None,
        text_count: Optional[int] = None,
        duration_ms: Optional[int] = None,
        meta: Optional[dict] = None,
    ) -> None:
        conn = self._connect()
        fields: list[str] = []
        params: list = []
        for col, val in [
            ("status", status),
            ("progress", progress),
            ("step", step),
            ("error", error),
            ("result_path", result_path),
            ("text_count", text_count),
            ("duration_ms", duration_ms),
            ("meta", json.dumps(meta, ensure_ascii=False) if meta is not None else None),
        ]:
            if val is not None:
                fields.append(f"{col}=?")
                params.append(val)
        if not fields:
            return
        fields.append("updated_at=datetime('now')")
        params.append(task_id)
        conn.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE id=?", params)
        conn.commit()

    def task_get(self, task_id: str) -> Optional[dict[str, Any]]:
        conn = self._connect()
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["meta"] = json.loads(d.get("meta") or "{}")
        return d

    def task_delete(self, task_id: str) -> bool:
        conn = self._connect()
        cur = conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        conn.commit()
        return cur.rowcount > 0
