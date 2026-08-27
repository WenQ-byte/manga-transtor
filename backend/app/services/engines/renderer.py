"""渲染引擎：将译文排版回图像，保持原气泡位置与尺寸"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.models.schemas import LangCode
from app.services.engines.base import BaseRenderer
from app.services.pipeline import TextRegion

# 中文字体候选（Windows / Linux）
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",  # 黑体
    "C:/Windows/Fonts/msjh.ttc",  # 微软正黑（繁中）
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


class PILRenderer(BaseRenderer):
    """基于 PIL 的排版渲染：在修复后的图像上绘制译文"""

    name = "pil"

    def __init__(self):
        self._font_cache: dict[int, ImageFont.FreeTypeFont] = {}
        self._font_path = self._find_font()

    def _find_font(self) -> str | None:
        for p in FONT_CANDIDATES:
            if Path(p).exists():
                return p
        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        if size in self._font_cache:
            return self._font_cache[size]
        if self._font_path:
            font = ImageFont.truetype(self._font_path, size)
        else:
            font = ImageFont.load_default()
        self._font_cache[size] = font
        return font

    def render(self, cleaned_image_path: Path, regions: list[TextRegion], target_lang: LangCode) -> bytes:
        img = Image.open(cleaned_image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for region in regions:
            text = region.translated
            if not text or not text.strip():
                continue
            x0, y0, x1, y1 = region.bounds
            box_w = x1 - x0
            box_h = y1 - y0
            if box_w <= 0 or box_h <= 0:
                continue

            # 根据区域大小估算字号：宽度/行文字数
            font_size = self._estimate_font_size(text, box_w, box_h)

            lines = self._wrap_text(text, font_size, box_w)
            # 文本高度超出区域时缩小字号
            while len(lines) * font_size * 1.3 > box_h and font_size > 8:
                font_size -= 2
                lines = self._wrap_text(text, font_size, box_w)

            font = self._get_font(font_size)
            line_spacing = int(font_size * 1.35)
            total_h = len(lines) * line_spacing
            ty = y0 + (box_h - total_h) // 2

            for line in lines:
                line_w = draw.textlength(line, font=font)
                tx = x0 + (box_w - line_w) // 2
                # 绘制文字阴影/描边提高可读性
                self._draw_text_with_border(draw, tx, ty, line, font, fill=(0, 0, 0), border=2)
                ty += line_spacing

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _estimate_font_size(self, text: str, box_w: int, box_h: int) -> int:
        max_len = max(1, len(text))
        size_by_width = box_w / max_len * 1.8
        size_by_height = box_h / max(1, min(3, len(text))) * 0.9
        return int(max(8, min(size_by_width, size_by_height, box_h * 0.8)))

    def _wrap_text(self, text: str, font_size: int, max_width: int) -> list[str]:
        """按宽度换行"""
        font = self._get_font(font_size)
        lines = []
        current = ""
        for ch in text:
            test = current + ch
            if draw_textlength(font, test) > max_width and current:
                lines.append(current)
                current = ch
            else:
                current = test
        if current:
            lines.append(current)
        if not lines:
            lines = [text]
        return lines

    def _draw_text_with_border(self, draw, x, y, text, font, fill, border=2):
        for dx in range(-border, border + 1):
            for dy in range(-border, border + 1):
                if dx * dx + dy * dy <= border * border:
                    draw.text((x + dx, y + dy), text, font=font, fill=(255, 255, 255))
        draw.text((x, y), text, font=font, fill=fill)


def draw_textlength(font, text: str) -> float:
    """兼容性封装：临时ImageDraw用于测量"""
    from PIL import Image as _Img

    _tmp = _Img.new("RGB", (1, 1))
    _d = ImageDraw.Draw(_tmp)
    return _d.textlength(text, font=font)


def create_renderer() -> BaseRenderer:
    return PILRenderer()
