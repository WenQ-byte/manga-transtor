"""渲染引擎：将译文排版回图像，保持原气泡位置与尺寸

字号方案（参考 manga-image-translator，精确匹配原文字）：
  - 原文字字号 ≈ 检测框的高度（横排）/ 宽度（竖排）
  - 用 PIL textbbox/multiline_textbbox 精确测量（与渲染一致，含描边）
  - 二分查找能完整放进检测框的最大字号，译文不超出原文字区域
"""
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

# 竖排字距系数
VERTICAL_CHAR_RATIO = 1.1
# 描边宽度系数（相对字号）
STROKE_RATIO = 1 / 14
# 横排行距（相对字号）
LINE_SPACING_RATIO = 0.1


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
        size = max(1, int(size))
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
        img_w, img_h = img.size
        min_font_size = max(1, round((img_w + img_h) / 200))

        # 统一字号基准：取横排检测框高度的中位数，消除检测框高度波动
        unified_h = self._unified_height(regions)

        for region in regions:
            text = (region.translated or "").strip()
            if not text:
                continue
            x0, y0, x1, y1 = region.bounds
            box_w = x1 - x0
            box_h = y1 - y0
            if box_w <= 0 or box_h <= 0:
                continue

            if box_h > box_w * 1.5:
                self._render_vertical(draw, text, x0, y0, box_w, box_h, min_font_size)
            else:
                self._render_horizontal(draw, text, x0, y0, box_w, box_h, unified_h, min_font_size)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _unified_height(self, regions: list[TextRegion]) -> int | None:
        """返回横排检测框高度的中位数（用于统一译文字号），无横排时返回 None"""
        heights = []
        for region in regions:
            x0, y0, x1, y1 = region.bounds
            w, h = x1 - x0, y1 - y0
            if w > 0 and h > 0 and h <= w * 1.5:
                heights.append(h)
        if not heights:
            return None
        heights.sort()
        return heights[len(heights) // 2]

    def _render_horizontal(self, draw, text, x0, y0, box_w, box_h, unified_h, min_font_size):
        """横排渲染：字号基准 = 统一高度（中位数）或检测框高，二分查找最大适配字号"""
        base_h = unified_h if unified_h is not None else box_h
        base_font_size = max(min_font_size, int(base_h))
        font_size = self._find_max_font_size(draw, text, box_w, box_h, base_font_size, min_font_size)

        font = self._get_font(font_size)
        sw = max(1, int(font_size * STROKE_RATIO))
        spacing = max(1, int(font_size * LINE_SPACING_RATIO))
        lines = self._wrap_text(text, font, box_w)
        joined = "\n".join(lines)

        bbox = draw.multiline_textbbox(
            (0, 0), joined, font=font, stroke_width=sw, spacing=spacing, align="center"
        )
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        cx = x0 + box_w / 2
        cy = y0 + box_h / 2
        tx = cx - tw / 2 - bbox[0]
        ty = cy - th / 2 - bbox[1]

        draw.multiline_text(
            (tx, ty),
            joined,
            font=font,
            fill=(0, 0, 0),
            stroke_width=sw,
            stroke_fill=(255, 255, 255),
            spacing=spacing,
            align="center",
        )

    def _render_vertical(self, draw, text, x0, y0, box_w, box_h, min_font_size):
        """竖排渲染：字号基准 = 检测框宽，逐字垂直排列"""
        base_font_size = max(min_font_size, int(box_w))
        font_size = base_font_size
        while font_size > min_font_size:
            font = self._get_font(font_size)
            char_h = int(font_size * VERTICAL_CHAR_RATIO)
            if len(text) * char_h <= box_h:
                break
            font_size = max(min_font_size, font_size - 1)

        font = self._get_font(font_size)
        sw = max(1, int(font_size * STROKE_RATIO))
        char_h = int(font_size * VERTICAL_CHAR_RATIO)
        total_h = len(text) * char_h
        cx = x0 + box_w / 2
        cy = y0 + box_h / 2
        ty = cy - total_h / 2

        for ch in text:
            tx = cx - font_size / 2
            draw.text(
                (tx, ty),
                ch,
                font=font,
                fill=(0, 0, 0),
                stroke_width=sw,
                stroke_fill=(255, 255, 255),
            )
            ty += char_h

    def _find_max_font_size(self, draw, text, box_w, box_h, max_size, min_size):
        """二分查找 [min_size, max_size] 内能完整放进 (box_w, box_h) 的最大字号"""
        lo, hi = min_size, max_size
        best = min_size
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._fits(draw, text, mid, box_w, box_h):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _fits(self, draw, text, font_size, box_w, box_h) -> bool:
        """判断字号 font_size 的译文能否完整放进检测框（含描边+行距）"""
        font = self._get_font(font_size)
        sw = max(1, int(font_size * STROKE_RATIO))
        spacing = max(1, int(font_size * LINE_SPACING_RATIO))
        lines = self._wrap_text(text, font, box_w)
        if not lines:
            return True
        joined = "\n".join(lines)
        bbox = draw.multiline_textbbox(
            (0, 0), joined, font=font, stroke_width=sw, spacing=spacing, align="center"
        )
        return (bbox[2] - bbox[0]) <= box_w and (bbox[3] - bbox[1]) <= box_h

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        """按宽度逐字符换行（适合中文/日文）"""
        lines = []
        current = ""
        for ch in text:
            test = current + ch
            if _text_length(font, test) > max_width and current:
                lines.append(current)
                current = ch
            else:
                current = test
        if current:
            lines.append(current)
        if not lines:
            lines = [text]
        return lines


def _text_length(font: ImageFont.FreeTypeFont, text: str) -> float:
    """测量文本宽度"""
    try:
        return font.getlength(text)
    except AttributeError:
        tmp = Image.new("RGB", (1, 1))
        d = ImageDraw.Draw(tmp)
        return d.textlength(text, font=font)


def create_renderer() -> BaseRenderer:
    return PILRenderer()
