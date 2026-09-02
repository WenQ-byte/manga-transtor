"""文件存储服务：上传图片与结果图片管理"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from app.config import get_settings


class FileStore:
    """管理上传/结果图片的本地存储"""

    def __init__(self):
        self.settings = get_settings()

    @staticmethod
    def _safe_filename(filename: str) -> str:
        """提取安全的文件名（只保留基本名+扩展名）"""
        name = Path(filename).name
        name = name.replace(" ", "_")
        # 防止路径穿越
        return Path(name).name

    def save_upload(self, content: bytes, original_name: str) -> tuple[str, str]:
        """保存上传文件，返回 (存储相对路径, 磁盘绝对路径)"""
        ext = Path(self._safe_filename(original_name)).suffix.lower()
        if not ext:
            ext = ".png"
        rel = f"{uuid.uuid4().hex}{ext}"
        abs_path = self.settings.upload_path / rel
        abs_path.write_bytes(content)
        return rel, str(abs_path)

    def save_result(self, content: bytes, ext: str = ".png") -> tuple[str, str]:
        """保存结果图片，返回 (存储相对路径, 磁盘绝对路径)"""
        rel = f"{uuid.uuid4().hex}{ext}"
        abs_path = self.settings.result_path / rel
        abs_path.write_bytes(content)
        return rel, str(abs_path)

    def save_state(self, content: bytes) -> tuple[str, str]:
        """保存任务内部状态，路径位于上传目录且不直接暴露给下载接口。"""
        rel = f".state-{uuid.uuid4().hex}.bin"
        path = self.settings.upload_path / rel
        path.write_bytes(content)
        return rel, str(path)

    def resolve_state(self, rel: str) -> Optional[Path]:
        path = (self.settings.upload_path / Path(rel).name).resolve()
        root = self.settings.upload_path.resolve()
        return path if path.is_file() and root in path.parents and path.name.startswith(".state-") else None

    def resolve(self, rel: str) -> Optional[Path]:
        """根据相对路径解析磁盘路径，防止路径穿越"""
        p = (self.settings.result_path / rel).resolve()
        result_root = self.settings.result_path.resolve()
        upload_root = self.settings.upload_path.resolve()
        if p.is_file() and (result_root in p.parents or upload_root in p.parents):
            return p
        return None

    def delete(self, rel: str) -> None:
        p = self.resolve(rel)
        if p:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def result_url(self, rel: str) -> str:
        return f"/api/files/{rel}"
