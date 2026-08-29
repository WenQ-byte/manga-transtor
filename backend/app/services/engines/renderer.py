"""渲染引擎：将译文排版回图像，保持原气泡位置与尺寸

字号方案：
  - 气泡检测：在修复后的图像上对每个文本框做泛洪填充，找到所属气泡的真实边界
    （修复后气泡内部为纯色，泛洪填充可靠）
  - 同一气泡内的多行文本合并渲染，字号统一
  - 用 PIL multiline_textbbox 精确测量 + 二分查找能放进气泡的最大字号
    （留少量边距），译文既填满气泡又不超出
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
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

# 竖排字距系数（字号 * 该系数 = 相邻字符垂直间距）
VERTICAL_CHAR_RATIO = 1.15
# 竖排列宽占列间距比例（留出描边与间隔，防相邻列文字重叠）
VERTICAL_COL_USE_RATIO = 0.82
# 描边宽度系数（相对字号）
STROKE_RATIO = 1 / 14
# 横排行距（相对字号，中文行距需偏大避免拥挤）
LINE_SPACING_RATIO = 0.3

# 气泡内边距比例（相对气泡宽/高，用于留白控制）
PAD_RATIO = 0.12
# 单行最大字号上限（相对气泡高度）
MAX_FONT_RATIO = 0.85
# 气泡泛洪填充颜色容差
FLOOD_TOL = 40
# 气泡相对文本框的最大放大倍数（防止无边框气泡泄漏到整张图）
BUBBLE_GROW_RATIO = 6.0

try:
    import cv2
except ImportError:  # noqa: BLE001
    cv2 = None


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

        bgr = np.array(img)[:, :, ::-1].copy()

        groups = self._group_by_bubble(bgr, regions, img_w, img_h)

        for bubble, group_regions in groups:
            bx0, by0, bx1, by1 = bubble
            bw, bh = bx1 - bx0, by1 - by0
            if bw <= 0 or bh <= 0:
                continue
            if bh > bw * 1.5:
                self._render_vertical_bubble(draw, group_regions, bx0, by0, bw, bh, min_font_size)
            else:
                self._render_horizontal_bubble(draw, group_regions, bx0, by0, bw, bh, min_font_size)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _group_by_bubble(self, bgr, regions, img_w, img_h):
        """将同一气泡内的 region 分组，返回 [(bubble_bbox, [regions])]

        用重叠率（交集 / 较小框面积）而非 IoU：同一气泡内相邻竖排列
        的检测框常部分重叠但 IoU 偏低，重叠率能正确合并它们。
        """
        groups: list[list] = []
        for region in regions:
            text = (region.translated or "").strip()
            if not text:
                continue
            bb = self._detect_bubble(bgr, region.bounds, img_w, img_h)
            best_idx, best_ov = -1, 0.0
            for i, g in enumerate(groups):
                ov = self._overlap_ratio(bb, g[0])
                if ov > best_ov:
                    best_ov, best_idx = ov, i
            if best_idx >= 0 and best_ov > 0.15:
                groups[best_idx][0] = self._union(bb, groups[best_idx][0])
                groups[best_idx][1].append(region)
            else:
                groups.append([bb, [region]])
        return groups

    @staticmethod
    def _overlap_ratio(a, b) -> float:
        """交集面积 / 较小框面积，衡量两框重叠程度（对相邻列更敏感）"""
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        iw = max(0, min(ax1, bx1) - max(ax0, bx0))
        ih = max(0, min(ay1, by1) - max(ay0, by0))
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (ax1 - ax0) * (ay1 - ay0)
        area_b = (bx1 - bx0) * (by1 - by0)
        return inter / max(1, min(area_a, area_b))

    def _detect_bubble(self, bgr, bounds, img_w, img_h):
        """通过泛洪填充找到文本框所属气泡的真实边界"""
        x0, y0, x1, y1 = [int(v) for v in bounds]
        x0 = max(0, min(x0, img_w - 1))
        x1 = max(0, min(x1, img_w - 1))
        y0 = max(0, min(y0, img_h - 1))
        y1 = max(0, min(y1, img_h - 1))
        if x1 <= x0 or y1 <= y0:
            return (x0, y0, x1, y1)

        max_bw = max((x1 - x0) * BUBBLE_GROW_RATIO, img_w * 0.85)
        max_bh = max((y1 - y0) * BUBBLE_GROW_RATIO, img_h * 0.85)

        if cv2 is not None:
            h, w = bgr.shape[:2]
            tol = FLOOD_TOL
            flags = 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
            # 多个候选种子点，取最大合法填充（避免种子落在残留笔画上）
            seeds = [
                ((x0 + x1) // 2, (y0 + y1) // 2),
                (x0 + 2, y0 + 2),
                (x1 - 2, y0 + 2),
                (x0 + 2, y1 - 2),
                (x1 - 2, y1 - 2),
            ]
            best = None
            best_area = -1
            for seed in seeds:
                sx, sy = seed
                if not (0 <= sx < w and 0 <= sy < h):
                    continue
                try:
                    mask = np.zeros((h + 2, w + 2), np.uint8)
                    cv2.floodFill(bgr, mask, seed, 0, (tol, tol, tol), (tol, tol, tol), flags)
                except Exception:  # noqa: BLE001
                    continue
                filled = mask[1:-1, 1:-1]
                ys, xs = np.where(filled > 0)
                if xs.size == 0:
                    continue
                bx0 = int(xs.min())
                by0 = int(ys.min())
                bx1 = int(xs.max()) + 1
                by1 = int(ys.max()) + 1
                if (bx1 - bx0) > max_bw or (by1 - by0) > max_bh:
                    continue
                area = (bx1 - bx0) * (by1 - by0)
                if area > best_area:
                    best_area = area
                    best = (bx0, by0, bx1, by1)
            if best is not None:
                # 退化保护：检测到的气泡面积小于文本框本身 → 不可信，用兜底扩展
                if (best[2] - best[0]) * (best[3] - best[1]) < (x1 - x0) * (y1 - y0) * 0.5:
                    return self._fallback_box(x0, y0, x1, y1, img_w, img_h)
                return best

        # 兜底：文本框向外扩展
        return self._fallback_box(x0, y0, x1, y1, img_w, img_h)

    @staticmethod
    def _fallback_box(x0, y0, x1, y1, img_w, img_h):
        pad_x = int((x1 - x0) * 0.35)
        pad_y = int((y1 - y0) * 0.35)
        return (
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(img_w, x1 + pad_x),
            min(img_h, y1 + pad_y),
        )

    @staticmethod
    def _iou(a, b) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (ax1 - ax0) * (ay1 - ay0)
        area_b = (bx1 - bx0) * (by1 - by0)
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _union(a, b):
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    def _render_horizontal_bubble(self, draw, regions, bx0, by0, bw, bh, min_font_size):
        """横排气泡：多行合并，统一字号填满气泡"""
        lines = sorted(regions, key=lambda r: (r.bounds[1], r.bounds[0]))
        text = "\n".join((r.translated or "").strip() for r in lines)
        avail_w = bw * (1 - 2 * PAD_RATIO)
        avail_h = bh * (1 - 2 * PAD_RATIO)
        max_font = max(min_font_size, int(bh * MAX_FONT_RATIO))
        font_size = self._find_max_font_in(draw, text, avail_w, avail_h, max_font, min_font_size)

        font = self._get_font(font_size)
        sw = max(1, int(font_size * STROKE_RATIO))
        spacing = max(1, int(font_size * LINE_SPACING_RATIO))
        wrapped = self._wrap_paragraph(text, font, avail_w)
        joined = "\n".join(wrapped)
        bbox = draw.multiline_textbbox(
            (0, 0), joined, font=font, stroke_width=sw, spacing=spacing, align="center"
        )
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        cx = bx0 + bw / 2
        cy = by0 + bh / 2
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

    def _render_vertical_bubble(self, draw, regions, bx0, by0, bw, bh, min_font_size):
        """竖排气泡：按列从右到左排列，字号统一，整组对称居中

        - 列间距 = 可用宽 / 列数，字号 ≤ 列间距 * VERTICAL_COL_USE_RATIO（防列间重叠）
        - 字号同时受高度约束：最长列字符数 * 字距 ≤ 可用高
        - 整组水平居中（左右对称边距），每列垂直居中
        """
        columns = sorted(regions, key=lambda r: -r.bounds[0])
        texts = [t for t in ((r.translated or "").strip() for r in columns) if t]
        if not texts:
            return
        n = len(texts)
        pad_x = PAD_RATIO * bw
        pad_y = PAD_RATIO * bh
        avail_w = bw - 2 * pad_x
        avail_h = bh - 2 * pad_y
        if avail_w <= 0 or avail_h <= 0:
            return
        col_gap = avail_w / n
        longest = max(len(t) for t in texts)
        # 字号：受列间距（防重叠）与高度共同约束
        font_size = int(min(col_gap * VERTICAL_COL_USE_RATIO, avail_h / max(1, longest * VERTICAL_CHAR_RATIO)))
        font_size = max(1, font_size)

        font = self._get_font(font_size)
        sw = max(1, int(font_size * STROKE_RATIO))
        char_h = int(font_size * VERTICAL_CHAR_RATIO)
        # 整组水平居中：从右到左排列，第一列中心在右侧
        right_pad = pad_x + col_gap / 2
        x_center = bx0 + bw - right_pad
        for t in texts:
            total_h = len(t) * char_h
            ty = by0 + avail_h / 2 - total_h / 2 + pad_y
            tx = x_center - font_size / 2
            for ch in t:
                draw.text(
                    (tx, ty),
                    ch,
                    font=font,
                    fill=(0, 0, 0),
                    stroke_width=sw,
                    stroke_fill=(255, 255, 255),
                )
                ty += char_h
            x_center -= col_gap

    def _find_max_font_in(self, draw, text, max_w, max_h, max_size, min_size):
        """二分查找 [min_size, max_size] 内能放进 (max_w, max_h) 的最大字号"""
        lo, hi = min_size, max_size
        best = min_size
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._fits_in(draw, text, mid, max_w, max_h):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _fits_in(self, draw, text, font_size, max_w, max_h) -> bool:
        font = self._get_font(font_size)
        sw = max(1, int(font_size * STROKE_RATIO))
        spacing = max(1, int(font_size * LINE_SPACING_RATIO))
        wrapped = self._wrap_paragraph(text, font, max_w)
        joined = "\n".join(wrapped)
        bbox = draw.multiline_textbbox(
            (0, 0), joined, font=font, stroke_width=sw, spacing=spacing, align="center"
        )
        return (bbox[2] - bbox[0]) <= max_w and (bbox[3] - bbox[1]) <= max_h

    def _wrap_paragraph(self, text, font, max_width) -> list[str]:
        result = []
        for line in text.split("\n"):
            result.extend(self._wrap_text(line, font, max_width))
        if not result:
            result = [""]
        return result

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
