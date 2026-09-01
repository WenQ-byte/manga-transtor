"""渲染引擎：将译文排版回图像，保持原气泡位置与尺寸

字号方案：
  - 气泡检测：在修复后的图像上对每个文本框做泛洪填充，找到所属气泡的真实边界
    （修复后气泡内部为纯色，泛洪填充可靠）
  - 同一气泡内的多行文本合并渲染，字号统一（单行优先、避免孤字折行）
  - 文本绘制到透明 overlay，再用气泡泛洪掩膜裁剪合成——文字永不出气泡框
"""
from __future__ import annotations

import io
import logging
import math
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from app.config import get_settings
from app.models.schemas import LangCode
from app.services.engines.base import BaseRenderer
from app.services.pipeline import TextRegion

# 中文字体候选（Windows / Linux）
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
    "C:/Windows/Fonts/YuGothR.ttc",  # 游ゴシック
    "C:/Windows/Fonts/msgothic.ttc",  # MS Gothic
    "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑 Bold（无常规字体时回退）
    "C:/Windows/Fonts/simhei.ttf",  # 黑体
    "C:/Windows/Fonts/msjh.ttc",  # 微软正黑（繁中）
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]

FONT_CANDIDATES_BY_LANG = {
    "zh": [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msjh.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ],
    "ja": [
        "C:/Windows/Fonts/YuGothR.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ],
    "en": [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
}

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
# 有限安全扩展框参数（气泡掩膜失败时的二级回退）
# 扩展上限（相对锚点宽高，约 1.5~2 倍）
SAFE_EXPAND_RATIO = 1.8
# 单边单步扩展像素数
SAFE_EXPAND_STEP = 3
# 边缘像素判定阈值（Sobel 梯度幅度，低于该值视为平坦）
SAFE_EDGE_THRESH = 25.0
# 候选带内边缘像素占比上限（超过即视为触及轮廓/分镜线/人物纹理，停止该边）
SAFE_EDGE_RATIO_MAX = 0.12

# 单个气泡的排版必须是有界计算。漫画对话的正常译文远小于这些上限；超限时保留前缀并
# 记录诊断，避免异常服务响应或错误 OCR 让整张图片停在渲染阶段。
MAX_RENDER_TEXT_CHARS = 1200
MAX_RENDER_GROUP_PIXELS = 24_000_000
MAX_RENDER_FONT_SIZE = 256
MAX_VERTICAL_FONT_TRIES = 64
MAX_HORIZONTAL_FONT_TRIES = 16

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:  # noqa: BLE001
    cv2 = None


class PILRenderer(BaseRenderer):
    """基于 PIL 的排版渲染：在修复后的图像上绘制译文"""

    name = "pil"

    def __init__(self):
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
        self._glyph_cache: dict[str, set[int] | None] = {}
        self._font_paths = {lang: self._find_font(lang) for lang in FONT_CANDIDATES_BY_LANG}
        self._font_path = self._font_paths.get("zh") or self._find_font("zh")
        self._active_target_lang = "zh"
        self._active_font_path = self._font_path
        self.last_font_path = self._font_path or ""
        s = get_settings()
        self.pad_ratio = max(0.02, float(s.render_padding))
        self.vertical_min_ratio = max(1.0, float(s.render_vertical_min_ratio))
        self.render_diagnostics = bool(getattr(s, "render_diagnostics", False))
        self._render_gradient = None
        self._render_gradient_source = None

    def _diagnose(self, message: str, *args) -> None:
        if self.render_diagnostics:
            logger.info("[render] " + message, *args)

    @staticmethod
    def _bounded_text(text: str | None) -> tuple[str, bool]:
        """清理不可绘制字符并限制单个气泡的排版规模。"""
        value = unicodedata.normalize("NFC", (text or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n"))
        value = "".join(
            ch for ch in value
            if ch in {"\n", "\t"} or (not unicodedata.category(ch).startswith("C") and not 0xD800 <= ord(ch) <= 0xDFFF)
        )
        if len(value) <= MAX_RENDER_TEXT_CHARS:
            return value, False
        return value[: MAX_RENDER_TEXT_CHARS - 1] + "…", True

    def _find_font(self, target_lang: str = "zh") -> str | None:
        candidates = FONT_CANDIDATES_BY_LANG.get(target_lang, FONT_CANDIDATES)
        for p in candidates:
            if Path(p).exists():
                return p
        return None

    def _get_font(self, size: int, target_lang: str | None = None) -> ImageFont.FreeTypeFont:
        target_lang = target_lang or self._active_target_lang
        size = max(1, int(size))
        path = self._active_font_path if target_lang == self._active_target_lang else self._font_paths.get(target_lang)
        path = path or self._font_path
        key = (path or target_lang, size)
        if key in self._font_cache:
            return self._font_cache[key]
        if path:
            font = ImageFont.truetype(path, size)
        else:
            font = ImageFont.load_default()
        self._font_cache[key] = font
        return font

    def _select_font_for_text(self, target_lang: str, text: str) -> str | None:
        """选择实际存在且覆盖页面字符的字体；无法读取 cmap 时退回首个可用字体。"""
        existing = [path for path in FONT_CANDIDATES_BY_LANG.get(target_lang, FONT_CANDIDATES) if Path(path).exists()]
        for path in existing:
            glyphs = self._font_glyphs(path)
            if glyphs is None or all(ch.isspace() or ord(ch) in glyphs for ch in text):
                return path
        return existing[0] if existing else self._font_path

    def _font_glyphs(self, path: str) -> set[int] | None:
        if path in self._glyph_cache:
            return self._glyph_cache[path]
        glyphs: set[int] = set()
        try:
            from fontTools.ttLib import TTCollection, TTFont

            if Path(path).suffix.lower() == ".ttc":
                collection = TTCollection(path, lazy=True)
                fonts = collection.fonts
            else:
                collection = None
                fonts = [TTFont(path, lazy=True)]
            for font in fonts:
                for table in font["cmap"].tables:
                    glyphs.update(table.cmap)
                font.close()
            if collection is not None:
                collection.close()
            self._glyph_cache[path] = glyphs
        except Exception:  # noqa: BLE001
            self._glyph_cache[path] = None
        return self._glyph_cache[path]

    def _line_spacing(self) -> float:
        settings = get_settings()
        return {
            "zh": float(settings.render_zh_line_spacing),
            "ja": float(settings.render_ja_line_spacing),
            "en": float(settings.render_en_line_spacing),
        }.get(self._active_target_lang, LINE_SPACING_RATIO)

    def render(self, cleaned_image_path: Path, regions: list[TextRegion], target_lang: LangCode) -> bytes:
        render_started = time.monotonic()
        logger.info("[render] 进入渲染：区域数=%s，目标语言=%s", len(regions), getattr(target_lang, "value", target_lang))
        img = Image.open(cleaned_image_path).convert("RGB")
        img_w, img_h = img.size
        target_value = getattr(target_lang, "value", target_lang)
        self._active_target_lang = target_value if target_value in FONT_CANDIDATES_BY_LANG else "zh"
        render_text = "\n".join(
            (region.group_translated or region.translated or "") for region in regions
        )
        render_text, _ = self._bounded_text(render_text)
        self._active_font_path = self._select_font_for_text(self._active_target_lang, render_text)
        logger.info("[render] 字体准备完成：文本长度=%s", len(render_text))
        self.last_font_path = self._active_font_path or ""
        for region in regions:
            region.render_font = self.last_font_path
        min_font_size = min(MAX_RENDER_FONT_SIZE, max(1, round((img_w + img_h) / 200)))

        bgr = np.array(img)[:, :, ::-1].copy()
        self._render_gradient = None
        self._render_gradient_source = id(bgr)

        grouping_started = time.monotonic()
        logger.info("[render] 气泡分组开始：区域数=%s", len(regions))
        groups = self._group_by_bubble(bgr, regions, img_w, img_h)
        logger.info(
            "[render] 气泡分组结束：气泡数=%s，耗时=%.1fms",
            len(groups), (time.monotonic() - grouping_started) * 1000,
        )

        # 每组使用独立 overlay，先按自身容器裁剪再合成，避免失败组污染其他组的覆盖率统计。
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

        for fallback_index, (_bubble, group_regions) in enumerate(groups):
            group_index = next((r.group_index for r in group_regions if r.group_index is not None), fallback_index)
            group_bounds = next((r.group_bounds for r in group_regions if r.group_bounds), None)
            self._active_group_context = (group_index, group_bounds, target_value)
            group_text = self._group_block_text(group_regions)
            raw_text_length = len(group_text or "\n".join((r.translated or "") for r in group_regions))
            logger.info(
                "[render] 气泡开始：group_index=%s，区域数=%s，bbox=%s，文本长度=%s，目标语言=%s",
                group_index, len(group_regions), group_bounds, raw_text_length, target_value,
            )
            self._diagnose("气泡%s开始渲染：bbox=%s，区域数=%s，文本长度=%s，目标语言=%s", group_index, group_bounds, len(group_regions), raw_text_length, target_value)
            if group_text:
                group_text, was_truncated = self._bounded_text(group_text)
                if was_truncated:
                    self._diagnose("气泡%s译文过长，已截断：文本长度=%s，目标语言=%s", group_index, len(self._group_block_text(group_regions) or ""), target_value)
            geometry_started = time.monotonic()
            logger.info("[render] 气泡几何开始：group_index=%s，bbox=%s", group_index, group_bounds)
            bb, mask = self._safe_bubble_geometry(bgr, group_regions, img_w, img_h)
            geometry_ms = (time.monotonic() - geometry_started) * 1000
            logger.info("[render] 气泡几何结束：group_index=%s，耗时=%.1fms", group_index, geometry_ms)
            self._diagnose("气泡%s几何耗时：%.1fms", group_index, geometry_ms)
            if bb is None:
                logger.warning("[render] 气泡跳过：group_index=%s，原因=没有可用几何区域，bbox=%s", group_index, group_bounds)
                continue
            bx0, by0, bx1, by1 = bb
            bw, bh = bx1 - bx0, by1 - by0
            logger.info("[render] 气泡几何结果：group_index=%s，bbox=%s，尺寸=%sx%s，掩膜=%s", group_index, bb, bw, bh, mask is not None)
            self._diagnose("气泡%s几何完成：bbox=%s，掩膜=%s，尺寸=%sx%s", group_index, bb, mask is not None, bw, bh)
            if bw <= 0 or bh <= 0 or bw * bh > MAX_RENDER_GROUP_PIXELS:
                logger.warning("[render] 气泡%s尺寸异常，已隔离：bbox=%s，尺寸=%sx%s，目标语言=%s", group_index, bb, bw, bh, target_value)
                continue
            block = group_text
            block = self._layout_block_text(block, target_lang)
            fill, stroke = self._text_colors(group_regions)
            group_overlay = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
            odraw = ImageDraw.Draw(group_overlay)
            layout_started = time.monotonic()
            logger.info("[render] 气泡布局开始：group_index=%s，方向=%s，尺寸=%sx%s，文本长度=%s", group_index, "竖排" if self._use_vertical(group_regions, bw, bh) and target_value != "en" else "横排", bw, bh, len(block or ""))
            self._render_group_layout(
                odraw, group_regions, block, bw, bh, min_font_size, target_value, fill, stroke
            )
            logger.info("[render] 气泡布局结束：group_index=%s，耗时=%.1fms", group_index, (time.monotonic() - layout_started) * 1000)
            self._diagnose("气泡%s布局耗时：%.1fms", group_index, (time.monotonic() - layout_started) * 1000)

            mask_started = time.monotonic()
            group_alpha = np.array(group_overlay.getchannel("A"))
            drawn = group_alpha > 0
            mask_crop = None
            if not drawn.any():
                logger.warning("[render] 气泡跳过：group_index=%s，原因=没有绘制像素", group_index)
                continue
            group_clip = np.full((bh, bw), 255, np.uint8)

            if mask is not None:
                # 可靠容器是硬边界；覆盖不足说明排版/几何不可信，宁可跳过也不放行到人物背景。
                mask_crop = mask[by0:by1, bx0:bx1]
                if mask_crop.shape != group_alpha.shape:
                    logger.warning("[render] 气泡跳过：group_index=%s，原因=掩膜尺寸不匹配，掩膜=%s，绘制=%s", group_index, mask_crop.shape, group_alpha.shape)
                    continue
                coverage = float((drawn & (mask_crop > 0)).sum()) / float(drawn.sum())
                if coverage < 0.5:
                    logger.warning("[render] 气泡跳过：group_index=%s，原因=掩膜覆盖率过低，覆盖率=%.3f", group_index, coverage)
                    continue
                group_clip = mask_crop
            logger.info("[render] 气泡掩膜处理完成：group_index=%s，耗时=%.1fms，掩膜=%s，覆盖率=%s", group_index, (time.monotonic() - mask_started) * 1000, mask is not None, "无" if mask is None else f"{coverage:.3f}")
            composite_started = time.monotonic()
            keep = np.minimum(group_alpha, group_clip).astype(np.uint8)
            group_overlay.putalpha(Image.fromarray(keep, "L"))
            overlay.alpha_composite(group_overlay, dest=(bx0, by0))
            logger.info("[render] 气泡 alpha 合成完成：group_index=%s，耗时=%.1fms", group_index, (time.monotonic() - composite_started) * 1000)
            logger.info("[render] 气泡完成：group_index=%s", group_index)
            self._diagnose("气泡%s布局完成：绘制像素=%s", group_index, int(drawn.sum()))

        self._render_gradient = None
        self._render_gradient_source = None
        base = img.convert("RGBA")
        merged = Image.alpha_composite(base, overlay).convert("RGB")

        buf = io.BytesIO()
        merged.save(buf, format="PNG")
        self._diagnose("整页渲染完成：图片尺寸=%sx%s，气泡数=%s，总耗时=%.1fms", img_w, img_h, len(groups), (time.monotonic() - render_started) * 1000)
        return buf.getvalue()

    def _render_group_layout(self, draw, group_regions, block, bw, bh, min_font_size, target_value, fill, stroke):
        """隔离单个气泡的布局失败，不能让一个坏译文阻塞整页。"""
        group_index, group_bounds, language = getattr(self, "_active_group_context", (-1, None, target_value))
        text = block or "\n".join((r.translated or "") for r in group_regions)
        try:
            if target_value != "en" and self._use_vertical(group_regions, bw, bh):
                if block:
                    self._render_vertical_bubble_block(
                        draw, block, 0, 0, bw, bh, min_font_size, fill=fill, stroke=stroke
                    )
                else:
                    self._render_vertical_bubble(
                        draw, group_regions, 0, 0, bw, bh, min_font_size, fill=fill, stroke=stroke
                    )
            else:
                self._render_horizontal_bubble(
                    draw, group_regions, 0, 0, bw, bh, min_font_size, block=block, fill=fill, stroke=stroke
                )
        except (ValueError, OverflowError, OSError, MemoryError, TypeError, RuntimeError) as exc:
            logger.warning(
                "[render] 气泡%s布局失败，已隔离：bbox=%s，文本长度=%s，目标语言=%s，原因=%s",
                group_index, group_bounds, len(text), language, exc,
            )

    def _safe_bubble_geometry(self, bgr, group_regions, img_w, img_h):
        group_index, group_bounds, language = getattr(self, "_active_group_context", (-1, None, ""))
        text = self._group_block_text(group_regions) or "\n".join((r.translated or "") for r in group_regions)
        try:
            return self._bubble_geometry(bgr, group_regions, img_w, img_h)
        except (ValueError, OverflowError, OSError, MemoryError, TypeError) as exc:
            logger.warning(
                "[render] 气泡%s几何计算失败，已隔离：bbox=%s，文本长度=%s，目标语言=%s，原因=%s",
                group_index, group_bounds, len(text), language, exc,
            )
            return None, None

    def _bubble_geometry(self, bgr, group_regions: list[TextRegion], img_w, img_h):
        """在修复后图像上重推该组的气泡：返回 (bbox, mask|None)

        回退分级（紧致文本框是最后兜底，不是掩膜失败时的默认方案）：
          一级  可靠气泡掩膜（泛洪通过全部可信度校验）→ (bbox, mask) 按气泡形状裁剪
          二级  有限安全扩展框（从文本框锚点向四周渐进扩展，边缘/纹理检查阻止
                穿过气泡轮廓/分镜线/人物高纹理区）→ (box, None) 按矩形裁剪
          三级  紧致文本框（锚点本身，必然在气泡内）→ (tight, None)
          四级  全部失败 → None（渲染端跳过该气泡）
        """
        from app.services.engines.bubble import bubble_with_mask

        if not group_regions or img_w <= 0 or img_h <= 0:
            return None, None
        x0 = max(0, min(img_w, min(r.bounds[0] for r in group_regions)))
        y0 = max(0, min(img_h, min(r.bounds[1] for r in group_regions)))
        x1 = max(0, min(img_w, max(r.bounds[2] for r in group_regions)))
        y1 = max(0, min(img_h, max(r.bounds[3] for r in group_regions)))
        if x1 <= x0 or y1 <= y0:
            return None, None

        def tight():
            return (
                max(0, x0 + 1),
                max(0, y0 + 1),
                min(img_w, x1 - 1),
                min(img_h, y1 - 1),
            )

        # 优先复用分组阶段在同一张修复图上确认的容器，避免渲染阶段重新泛洪得到不同区域。
        stored_mask = next(
            (
                r.group_mask
                for r in group_regions
                if getattr(r, "group_mask_reliable", False) and r.group_mask is not None
            ),
            None,
        )
        if stored_mask is not None and bool(stored_mask.any()):
            ys, xs = np.where(stored_mask > 0)
            stored_bb = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            if self._mask_reliable(
                stored_bb, stored_mask, group_regions, (x0, y0, x1, y1), img_w, img_h
            ):
                return stored_bb, stored_mask

        # 一级：可靠气泡掩膜
        bb, mask = bubble_with_mask(bgr, (x0, y0, x1, y1), img_w, img_h)
        if mask is not None and mask.any() and self._mask_reliable(
            bb, mask, group_regions, (x0, y0, x1, y1), img_w, img_h
        ):
            return bb, mask

        # 多 region 且没有共同可靠容器时，稀疏大包围盒通常是多个卡片被误并。
        # 这种情况不得进入矩形扩展回退，否则译文会覆盖人物或背景。
        if len(group_regions) > 1:
            region_area = sum(
                max(1, (r.bounds[2] - r.bounds[0]) * (r.bounds[3] - r.bounds[1]))
                for r in group_regions
            )
            union_area = max(1, (x1 - x0) * (y1 - y0))
            if union_area > region_area * 3.0:
                return None, None

        # 二级：有限安全扩展框（锚点外扩，受边缘/纹理约束）
        safe = self._safe_expand_box(bgr, x0, y0, x1, y1, img_w, img_h)
        if safe is not None:
            return safe, None

        # 三级：紧致文本框兜底
        t = tight()
        if t[2] - t[0] >= 6 and t[3] - t[1] >= 6:
            return t, None
        # 四级：跳过该气泡
        return None, None

    @staticmethod
    def _mask_reliable(bb, mask, group_regions, tight_box, img_w, img_h) -> bool:
        """气泡掩膜可信度校验：覆盖全部擦除笔画、面积不超限、不越分组包围盒"""
        x0, y0, x1, y1 = tight_box
        tb_area = max(1, (x1 - x0) * (y1 - y0))
        if int((mask > 0).sum()) > tb_area * 6:
            return False
        # 绝对上限：泛洪结果不得远超分组已知的气泡包围盒（防大组泛洪泄漏到整页背景）
        gb = next((r.group_bounds for r in group_regions if r.group_bounds), None)
        if gb is not None:
            margin = 0.15 * max(gb[2] - gb[0], gb[3] - gb[1])
            if (
                bb[0] < gb[0] - margin
                or bb[1] < gb[1] - margin
                or bb[2] > gb[2] + margin
                or bb[3] > gb[3] + margin
            ):
                return False
        for r in group_regions:
            m = r.mask
            if not m or "patch" not in m or "bbox" not in m:
                continue
            px0, py0, px1, py1 = m["bbox"]
            patch = m["patch"] > 0
            if not patch.any():
                continue
            sub = mask[py0:py1, px0:px1]
            if sub.shape != patch.shape:
                return False
            covered = float((sub > 0)[patch].sum()) / float(patch.sum())
            if covered < 0.85:
                return False
        return True

    def _safe_expand_box(self, bgr, x0, y0, x1, y1, img_w, img_h):
        """从紧致文本框锚点向四周渐进扩展，受边缘/纹理检查约束，返回最大安全排版框。

        锚点（原始文字区域）必然在气泡内。逐边逐条带外扩：候选带内边缘像素占比低
        （气泡内部平坦）即接受；一旦触及气泡轮廓/分镜线/人物高纹理（边缘密度升高）
        即停该边。扩展上限为锚点宽高的 SAFE_EXPAND_RATIO 倍。无法安全扩展时返回
        None（上层退回紧致文本框兜底）。
        """
        import cv2

        if cv2 is None:
            return None
        bw = x1 - x0
        bh = y1 - y0
        if bw <= 0 or bh <= 0:
            return None

        # 同一页的多个气泡共用同一份梯度图，避免每个气泡重复对整页执行 Sobel。
        if self._render_gradient_source == id(bgr) and self._render_gradient is not None:
            grad = self._render_gradient
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad = cv2.magnitude(gx, gy)
            self._render_gradient = grad
            self._render_gradient_source = id(bgr)

        # 锚点内部基线：气泡内部平坦区域边缘密度的参考
        inner = grad[y0:y1, x0:x1]
        base = float(inner.mean()) if inner.size else 0.0
        edge_thr = max(SAFE_EDGE_THRESH, base * 1.5 + 8)

        # 上限框：以锚点中心外扩到 ratio 倍
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        cap_w = max(1.0, bw * SAFE_EXPAND_RATIO)
        cap_h = max(1.0, bh * SAFE_EXPAND_RATIO)
        cap_x0 = max(0, int(cx - cap_w / 2))
        cap_x1 = min(img_w, int(cx + cap_w / 2))
        cap_y0 = max(0, int(cy - cap_h / 2))
        cap_y1 = min(img_h, int(cy + cap_h / 2))

        def band_safe(band) -> bool:
            if band.size == 0:
                return True
            return float((band > edge_thr).mean()) <= SAFE_EDGE_RATIO_MAX

        bx0, by0, bx1, by1 = x0, y0, x1, y1
        step = max(2, SAFE_EXPAND_STEP)
        changed = True
        guard = 0
        max_steps = min(2000, max(1, math.ceil((cap_w + cap_h) / step) + 4))
        while changed and guard < max_steps:
            changed = False
            guard += 1
            if by0 - step >= cap_y0 and band_safe(grad[by0 - step:by0, bx0:bx1]):
                by0 -= step
                changed = True
            if by1 + step <= cap_y1 and band_safe(grad[by1:by1 + step, bx0:bx1]):
                by1 += step
                changed = True
            if bx0 - step >= cap_x0 and band_safe(grad[by0:by1, bx0 - step:bx0]):
                bx0 -= step
                changed = True
            if bx1 + step <= cap_x1 and band_safe(grad[by0:by1, bx1:bx1 + step]):
                bx1 += step
                changed = True

        # 只要任一侧发生过有效扩展就用扩展框（含单侧扩展，如竖排气泡只往右有空间）；
        # 完全没扩出去（四周全是轮廓/人物纹理）才退回紧致框兜底。
        if bx0 == x0 and by0 == y0 and bx1 == x1 and by1 == y1:
            return None
        return int(bx0), int(by0), int(bx1), int(by1)

    @staticmethod
    def _use_vertical(group_regions, bw, bh, min_ratio=None) -> bool:
        dirs = [r.direction for r in group_regions if r.direction]
        v = dirs.count("v")
        h = dirs.count("h")
        # 方向优先：检测器给出的文字方向最可靠（圆形气泡竖排文字不能靠形状判定）
        if v > h:
            return True
        if h > v:
            return False
        # 无方向信息时按形状：高明显大于宽 → 竖排
        ratio = min_ratio if min_ratio is not None else 1.2
        return bh > bw * ratio

    @staticmethod
    def _text_colors(group_regions) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        fill = (0, 0, 0)
        stroke = (255, 255, 255)
        fgs = [r.fg_color for r in group_regions if r.fg_color]
        bgs = [r.bg_color for r in group_regions if r.bg_color]
        if fgs:
            fg = fgs[0]
            lum = 0.299 * fg[0] + 0.587 * fg[1] + 0.114 * fg[2]
            if lum < 180:
                fill = fg
        if bgs:
            bg = bgs[0]
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            if lum > 160:
                stroke = bg
        return fill, stroke

    @staticmethod
    def _group_block_text(group_regions: list[TextRegion]) -> str | None:
        """返回气泡分组对应的整块译文（pipeline 按气泡整块翻译的结果）"""
        for r in group_regions:
            if r.group_translated and r.group_translated.strip():
                return r.group_translated
        return None

    @staticmethod
    def _layout_block_text(text: str | None, target_lang) -> str | None:
        """中文排版不继承日文机械换行；换行只服务翻译上下文，最终按容器重新分列。"""
        if not text:
            return text
        lang = getattr(target_lang, "value", target_lang)
        if lang not in {"zh", "zh-cn", "zh-CN"}:
            return text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "".join(lines)

    def _group_by_bubble(self, bgr, regions, img_w, img_h):
        """将同一气泡内的 region 分组，返回 [(bubble_bbox, [regions])]

        pipeline 已按气泡分组（region.group_index 非空）时直接复用分组，
        气泡框取 group_bounds；否则用泛洪填充推气泡后按重叠率合并。
        """
        if any(r.group_index is not None for r in regions):
            groups_by_idx: dict[int, list] = {}
            for region in regions:
                if getattr(region, "_no_erase", False):
                    continue
                text = (region.group_translated or "").strip() or (region.translated or "").strip()
                if not text:
                    continue
                gi = region.group_index if region.group_index is not None else -1
                groups_by_idx.setdefault(gi, []).append(region)
            result = []
            for gi, gs in groups_by_idx.items():
                bounds_list = [r.group_bounds for r in gs if r.group_bounds] or [r.bounds for r in gs]
                bx0 = min(b[0] for b in bounds_list)
                by0 = min(b[1] for b in bounds_list)
                bx1 = max(b[2] for b in bounds_list)
                by1 = max(b[3] for b in bounds_list)
                result.append(((bx0, by0, bx1, by1), gs))
            return result

        groups: list[list] = []
        for region in regions:
            if getattr(region, "_no_erase", False):
                continue
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

    def _render_horizontal_bubble(
        self, draw, regions, bx0, by0, bw, bh, min_font_size, block=None, fill=(0, 0, 0), stroke=(255, 255, 255)
    ):
        """横排气泡：整块文本（或各 region 合并）排版，单行优先、避免孤字折行、填满气泡"""
        lines = sorted(regions, key=lambda r: (r.bounds[1], r.bounds[0]))
        text = block if block is not None else "\n".join((r.translated or "").strip() for r in lines)
        text, _ = self._bounded_text(text)
        if not text.strip():
            return
        pad = self.pad_ratio
        avail_w = bw * (1 - 2 * pad)
        avail_h = bh * (1 - 2 * pad)
        if avail_w <= 1 or avail_h <= 1:
            return
        max_font = min(MAX_RENDER_FONT_SIZE, max(1, int(bh * MAX_FONT_RATIO)))
        local_min_font = min(max_font, max(1, min_font_size))
        group_index = getattr(self, "_active_group_context", (-1, None, ""))[0]
        logger.info("[render] 横排字号选择开始：group_index=%s，字号范围=%s-%s", group_index, local_min_font, max_font)
        font_size = self._select_horizontal_font(draw, text, avail_w, avail_h, max_font, local_min_font)
        logger.info("[render] 横排字号选择结束：group_index=%s，字号=%s", group_index, font_size)

        font = self._get_font(font_size)
        sw = max(1, int(font_size * STROKE_RATIO))
        spacing = max(1, int(font_size * self._line_spacing()))
        logger.info("[render] 横排换行开始：group_index=%s，最大宽度=%.1f", group_index, avail_w)
        wrapped = self._wrap_paragraph(text, font, avail_w)
        logger.info("[render] 横排换行结束：group_index=%s，行数=%s", group_index, len(wrapped))
        joined = "\n".join(wrapped)
        logger.info("[render] 横排尺寸计算开始：group_index=%s", group_index)
        tw, th = self._multiline_size(font, wrapped, sw, spacing)
        logger.info("[render] 横排尺寸计算结束：group_index=%s", group_index)

        cx = bx0 + bw / 2
        cy = by0 + bh / 2
        tx = cx - tw / 2 + sw
        ty = cy - th / 2 + sw

        logger.info("[render] 横排 PIL 绘制开始：group_index=%s", group_index)
        draw.multiline_text(
            (tx, ty),
            joined,
            font=font,
            fill=fill,
            stroke_width=sw,
            stroke_fill=stroke,
            spacing=spacing,
            align="center",
        )
        logger.info("[render] 横排 PIL 绘制结束：group_index=%s", group_index)

    def _select_horizontal_font(self, draw, text, avail_w, avail_h, max_size, min_size) -> int:
        """在有限搜索范围内选择横排字号。"""
        return self._select_horizontal_font_bounded(draw, text, avail_w, avail_h, max_size, min_size)

    def _select_horizontal_font_bounded(self, draw, text, avail_w, avail_h, max_size, min_size) -> int:
        """只测量最小字号一次，其余字号用线性尺度估算，避免反复创建 FreeType 字体。"""
        max_size = min(MAX_RENDER_FONT_SIZE, max(min_size, int(max_size)))
        min_size = max(1, min(int(min_size), max_size))
        group_index = getattr(self, "_active_group_context", (-1, None, ""))[0]
        started = time.monotonic()
        base_font = self._get_font(min_size)
        base_stroke = max(1, int(min_size * STROKE_RATIO))
        base_width = _text_length(base_font, text) + base_stroke * 2
        base_height = min_size * 1.4 + base_stroke * 2
        if base_width <= avail_w and base_height <= avail_h:
            width_scale = avail_w / max(1.0, base_width)
            height_scale = avail_h / max(1.0, base_height)
            selected = min(max_size, max(min_size, int(min_size * min(width_scale, height_scale))))
            logger.info(
                "[render] 横排单行字号估算结束：group_index=%s，字号=%s，耗时=%.1fms",
                group_index, selected, (time.monotonic() - started) * 1000,
            )
            return selected
        selected = self._estimate_multiline_font(text, avail_w, avail_h, max_size, min_size)
        logger.info(
            "[render] 横排多行字号估算结束：group_index=%s，字号=%s，耗时=%.1fms",
            group_index, selected, (time.monotonic() - started) * 1000,
        )
        return selected

    def _estimate_multiline_font(self, text, avail_w, avail_h, max_size, min_size) -> int:
        """用字符宽度单位估算多行字号；搜索过程不加载候选字体。"""
        paragraphs = text.split("\n") or [""]

        def units(ch: str) -> float:
            if ch.isspace():
                return 0.35
            if ch.isascii():
                return 0.58 if ch.isalnum() else 0.5
            return 1.0

        paragraph_units = [sum(units(ch) for ch in paragraph) for paragraph in paragraphs]
        max_unit = max((units(ch) for ch in text if ch != "\n"), default=1.0)
        lo, hi, best = min_size, max_size, min_size
        attempts = 0
        while lo <= hi and attempts < MAX_HORIZONTAL_FONT_TRIES:
            attempts += 1
            size = (lo + hi) // 2
            capacity = avail_w / max(1.0, size)
            if max_unit > capacity:
                fits = False
            else:
                line_count = sum(max(1, math.ceil(value / max(0.1, capacity))) for value in paragraph_units)
                stroke = max(1, int(size * STROKE_RATIO))
                spacing = max(1, int(size * self._line_spacing()))
                height = line_count * size * 1.4 + max(0, line_count - 1) * spacing + stroke * 2
                fits = height <= avail_h
            if fits:
                best = size
                lo = size + 1
            else:
                hi = size - 1
        return best

    def _render_vertical_bubble(
        self, draw, regions, bx0, by0, bw, bh, min_font_size, fill=(0, 0, 0), stroke=(255, 255, 255)
    ):
        """竖排气泡（每 region 即一列）：整组对称居中，每列垂直居中"""
        columns = sorted(regions, key=lambda r: -r.bounds[0])
        fallback_text, _ = self._bounded_text("\n".join((r.translated or "").strip() for r in columns))
        texts = [t for t in fallback_text.split("\n") if t]
        if not texts:
            return
        n = len(texts)
        pad = self.pad_ratio
        pad_x = pad * bw
        pad_y = pad * bh
        avail_w = bw - 2 * pad_x
        avail_h = bh - 2 * pad_y
        if avail_w <= 0 or avail_h <= 0:
            return
        col_gap = avail_w / n
        longest = max(len(t) for t in texts)
        font_size = int(min(col_gap * VERTICAL_COL_USE_RATIO, avail_h / max(1, longest * VERTICAL_CHAR_RATIO)))
        font_size = max(1, font_size)

        font = self._get_font(font_size)
        sw = max(1, int(font_size * STROKE_RATIO))
        char_h = int(font_size * VERTICAL_CHAR_RATIO)
        # 整组对称居中（从右到左依次排布）
        total_w = n * col_gap
        left = bx0 + (bw - total_w) / 2
        for i, t in enumerate(texts):
            total_h = len(t) * char_h
            ty = by0 + (bh - total_h) / 2
            # 日文原稿的竖排列序从右向左；group_regions 已按右到左整理，绘制时
            # 也必须把第一个源列放在最右侧，避免整句译文列序反转。
            tx = left + col_gap * (n - i - 0.5) - font_size / 2
            for ch in t:
                draw.text(
                    (tx, ty),
                    ch,
                    font=font,
                    fill=fill,
                    stroke_width=sw,
                    stroke_fill=stroke,
                )
                ty += char_h

    def _render_vertical_bubble_block(
        self, draw, text, bx0, by0, bw, bh, min_font_size, fill=(0, 0, 0), stroke=(255, 255, 255)
    ):
        """竖排气泡整块重排：整块译文按气泡框拆分为多列，整组对称居中、顶部对齐

        - 列用均衡切分：各列长度相似（target = ceil(总字数/列数)），整行优先入列
        - 字号上界 = min(可用宽*比率, 最长行高度约束)，向下搜索至「列数×列距 ≤ 可用宽」
        - 列组水平居中，各列顶部对齐
        """
        lines = [ln for ln in text.split("\n") if ln]
        if not lines:
            return
        pad = self.pad_ratio
        pad_x = pad * bw
        pad_y = pad * bh
        avail_w = bw - 2 * pad_x
        avail_h = bh - 2 * pad_y
        if avail_w <= 0 or avail_h <= 0:
            return

        font_size, columns, char_h = self._vertical_layout(lines, avail_w, avail_h, min_font_size)
        if font_size <= 0 or not columns:
            return
        font = self._get_font(font_size)
        sw = max(1, int(font_size * STROKE_RATIO))

        n = len(columns)
        col_gap = avail_w / max(1, n)
        total_w = n * col_gap
        left = bx0 + (bw - total_w) / 2
        max_col = max(len(c) for c in columns)
        group_top = by0 + (bh - max_col * char_h) / 2
        for col_idx, col in enumerate(columns):
            # columns 保持源文本的右到左顺序，因此第 0 列位于最右侧。
            cx = left + col_gap * (n - col_idx - 0.5)
            ty = group_top
            tx = cx - font_size / 2
            for ch in col:
                draw.text(
                    (tx, ty),
                    ch,
                    font=font,
                    fill=fill,
                    stroke_width=sw,
                    stroke_fill=stroke,
                )
                ty += char_h

    def _vertical_layout(self, lines, avail_w, avail_h, min_font_size) -> tuple[int, list[str], int]:
        """在有限候选字号内计算竖排分列布局。"""
        return self._vertical_layout_bounded(lines, avail_w, avail_h, min_font_size)

    def _vertical_layout_bounded(self, lines, avail_w, avail_h, min_font_size) -> tuple[int, list[str], int]:
        """有限候选字号的竖排布局，任何输入都在固定次数内结束。"""
        if not lines or avail_w <= 0 or avail_h <= 0:
            return 0, [], 0
        total = sum(len(line) for line in lines)
        upper = min(MAX_RENDER_FONT_SIZE, max(1, int(avail_w * VERTICAL_COL_USE_RATIO)))
        lower = min(upper, max(1, int(min_font_size)))
        step = max(1, math.ceil((upper - lower + 1) / MAX_VERTICAL_FONT_TRIES))
        candidates = list(range(upper, lower - 1, -step))
        if candidates[-1] != lower:
            candidates.append(lower)
        for font in candidates:
            char_h = max(1, int(font * VERTICAL_CHAR_RATIO))
            cols = self._balance_columns(lines, total, avail_h, char_h)
            if not cols:
                continue
            max_col_len = max(len(col) for col in cols)
            if max_col_len * char_h <= avail_h and font <= (avail_w / len(cols)) * VERTICAL_COL_USE_RATIO:
                return font, cols, char_h
        return 0, [], 0

    def _balance_columns(self, lines, total, avail_h, char_h) -> list[str] | None:
        """按原文行序而非字数重切：整行作为一列，保持句读顺序不被跨列打散

        行为：行数不超可用列数 → 每行独立成列（顺序即原文顺序）；
        行数超可用列数 → 才把相邻行合并进同一列（从上到下依次排），仍不跨句重切。
        返回列列表；若任意列超出可用高度返回 None。
        """
        avail_cols = max(1, int(avail_h // char_h))
        if not lines:
            return None
        # 优先整行独立成列：每行长度不超列容量即可
        if len(lines) <= avail_cols and all(len(ln) <= avail_cols for ln in lines):
            return list(lines)
        # 行数超列数：按顺序把相邻行塞进同一列，直到放满一列再开新列
        cols: list[str] = []
        cur = ""
        for raw in lines:
            ln = raw
            if not ln:
                continue
            # 长句按中文词语/标点边界均衡分列，避免“祖|先”“所|以”等机械断词。
            chunks = self._split_semantic_columns(ln, avail_cols)
            while len(chunks) > 1:
                if cur:
                    cols.append(cur)
                    cur = ""
                cols.append(chunks.pop(0))
            ln = chunks[0] if chunks else ""
            if cur and len(cur) + len(ln) > avail_cols:
                cols.append(cur)
                cur = ""
            cur += ln
        if cur:
            cols.append(cur)
        return cols or None

    @staticmethod
    def _split_semantic_columns(text: str, capacity: int) -> list[str]:
        """在容量约束内均衡切列，优先标点边界并避免拆开常见中文双字/连接词。"""
        if capacity <= 0 or len(text) <= capacity:
            return [text]
        no_split = {
            "祖先", "土地", "所以", "但是", "已经", "这个", "没有", "就是", "属于",
            "必须", "自己", "别人", "时候", "起来", "理解", "敌人", "文化", "大家",
            "想法", "白人", "金钱", "保护", "一直", "这么", "地方", "有钱",
        }
        closing = set("，。！？；：、…」』）】》,.!?;:")
        opening = set("「『（【《")
        remaining = text
        result: list[str] = []
        while len(remaining) > capacity:
            columns_left = max(2, (len(remaining) + capacity - 1) // capacity)
            target = min(capacity, max(1, (len(remaining) + columns_left - 1) // columns_left))
            lo = max(1, target - 3)
            hi = min(capacity, target + 3, len(remaining) - 1)
            best_cut, best_score = target, float("-inf")
            for cut in range(lo, hi + 1):
                left, right = remaining[:cut], remaining[cut:]
                score = -abs(cut - target)
                if left[-1] in closing:
                    score += 8
                if right[0] in closing or left[-1] in opening:
                    score -= 12
                if left[-1] + right[0] in no_split:
                    score -= 16
                if score > best_score:
                    best_cut, best_score = cut, score
            result.append(remaining[:best_cut])
            remaining = remaining[best_cut:]
        if remaining:
            result.append(remaining)
        return result

    @staticmethod
    def _multiline_size(font: ImageFont.FreeTypeFont, lines: list[str], stroke_width: int, spacing: int) -> tuple[float, float]:
        """用字体度量计算多行尺寸，避免 Pillow 对整段文本做原生 bbox 测量。"""
        safe_lines = lines or [""]
        width = max((_text_length(font, line) for line in safe_lines), default=0.0) + stroke_width * 2
        try:
            ascent, descent = font.getmetrics()
            line_height = max(1, ascent + descent)
        except (AttributeError, OSError, ValueError):
            line_height = max(1, int(getattr(font, "size", 1) * 1.4))
        height = line_height * len(safe_lines) + spacing * max(0, len(safe_lines) - 1) + stroke_width * 2
        return width, height

    def _wrap_paragraph(self, text, font, max_width) -> list[str]:
        result = []
        for line in text.split("\n"):
            result.extend(self._wrap_text(line, font, max_width))
        if not result:
            result = [""]
        return result

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        """按宽度均衡换行（适合中文/日文）：每行长度相近，末行不过短"""
        if " " in text and any(ch.isascii() and ch.isalpha() for ch in text):
            return self._wrap_latin_text(text, font, max_width)
        # 先宽贪心求最少行数
        greedy = self._greedy_wrap(text, font, max_width)
        if len(greedy) <= 1:
            return greedy
        total = sum(_text_length(font, ln) for ln in greedy)
        target = total / len(greedy)
        # 按目标宽度断行：达到目标宽度即换行，超限则硬断
        lines = []
        current = ""
        current_width = 0.0
        for ch in text:
            char_width = _text_length(font, ch)
            if current and current_width >= target:
                lines.append(current)
                current = ""
                current_width = 0.0
            current += ch
            current_width += char_width
            if current_width > max_width and len(current) > 1:
                lines.append(current[:-1])
                current = current[-1]
                current_width = char_width
        if current:
            lines.append(current)
        if not lines:
            lines = [text]
        return self._protect_cjk_line_boundaries(lines, font, max_width)

    @staticmethod
    def _protect_cjk_line_boundaries(lines, font, max_width):
        """闭合标点不置于行首，开标点不留在行尾。"""
        lines = list(lines)
        closing = set("，。！？；：、）》】」』〕〉…％%")
        opening = set("（《【「『〔〈")
        for index in range(1, len(lines)):
            while lines[index] and lines[index][0] in closing and lines[index - 1]:
                mark = lines[index][0]
                if _text_length(font, lines[index - 1] + mark) <= max_width:
                    lines[index - 1] += mark
                    lines[index] = lines[index][1:]
                elif len(lines[index - 1]) > 1:
                    lines[index] = lines[index - 1][-1] + lines[index]
                    lines[index - 1] = lines[index - 1][:-1]
                else:
                    break
            while lines[index - 1] and lines[index - 1][-1] in opening:
                lines[index] = lines[index - 1][-1] + lines[index]
                lines[index - 1] = lines[index - 1][:-1]
        return [line for line in lines if line]

    @staticmethod
    def _wrap_latin_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        """按有限步长拆分拉丁文本，确保窄容器也会持续收敛。"""
        return PILRenderer._wrap_latin_text_bounded(text, font, max_width)

    @staticmethod
    def _wrap_latin_text_bounded(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        """按单词换行；超长单词按小窗口测量拆分，保证每轮至少消费一个字符。"""
        import re

        width = max(1, int(max_width))
        lines: list[str] = []
        current = ""
        probe_width = max(1.0, _text_length(font, "M"))
        estimate = max(1, int(width / probe_width))

        def split_token(token: str) -> list[str]:
            chunks: list[str] = []
            remaining = token
            while remaining:
                limit = min(len(remaining), max(1, estimate * 2 + 4))
                lo, hi, best = 1, limit, 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if _text_length(font, remaining[:mid]) <= width:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                # 字符本身宽于容器时仍强制前进，不能原地循环。
                chunks.append(remaining[:best])
                remaining = remaining[best:]
            return chunks

        for token in re.findall(r"\s+|\S+", text):
            if token.isspace():
                if current:
                    current += token
                continue
            if current and _text_length(font, current + token) > width:
                lines.append(current.rstrip())
                current = ""
            if _text_length(font, token) <= width:
                current += token
                continue
            pieces = split_token(token)
            lines.extend(piece for piece in pieces[:-1] if piece)
            current = pieces[-1] if pieces else ""
        if current.rstrip():
            lines.append(current.rstrip())
        return lines or [text]

    @staticmethod
    def _greedy_wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        """宽贪心换行：每行尽量填满"""
        lines = []
        current = ""
        current_width = 0.0
        for ch in text:
            char_width = _text_length(font, ch)
            if current and current_width + char_width > max_width:
                lines.append(current)
                current = ch
                current_width = char_width
            else:
                current += ch
                current_width += char_width
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
