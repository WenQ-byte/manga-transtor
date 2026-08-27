"""引擎抽象基类"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.models.schemas import LangCode
from app.services.pipeline import TextRegion


class BaseDetector(ABC):
    """文本检测引擎：定位文字区域"""

    name = "detector"

    @abstractmethod
    def detect(self, image_path: Path) -> list[TextRegion]: ...


class BaseOCR(ABC):
    """OCR 引擎：识别文字"""

    name = "ocr"

    # 是否自带文本检测能力（如 PaddleOCR 检测+识别一体化）
    supports_detection: bool = False

    @abstractmethod
    def recognize(
        self,
        image_path: Path,
        regions: list[TextRegion],
        source_lang: str = "ja",
    ) -> None:
        """填充每个 region 的 text 字段；若自带检测且 regions 为空，需自行创建 region"""


class BaseTranslator(ABC):
    """翻译引擎"""

    name = "translator"

    @abstractmethod
    def translate_batch(
        self,
        texts: list[str],
        source_lang: LangCode,
        target_lang: LangCode,
        glossary: dict[str, str] | None = None,
        progress_cb=None,
    ) -> list[str]: ...


class BaseInpainter(ABC):
    """图像修复引擎：擦除原文字"""

    name = "inpainter"

    @abstractmethod
    def inpaint(self, image_path: Path, regions: list[TextRegion]) -> Path:
        """返回清理后的图像路径"""


class BaseRenderer(ABC):
    """渲染引擎：将译文排版回图像"""

    name = "renderer"

    @abstractmethod
    def render(self, cleaned_image_path: Path, regions: list[TextRegion], target_lang: LangCode) -> bytes:
        """返回渲染后的图像字节"""
