"""渲染引擎：将译文排版回图像，保持原气泡位置与尺寸

字号方案：
  - 气泡检测：在修复后的图像上对每个文本框做泛洪填充，找到所属气泡的真实边界
    （修复后气泡内部为纯色，泛洪填充可靠）
  - 同一气泡内的多行文本合并渲染，字号统一（单行优先、避免孤字折行）
  - 文本绘制到透明 overlay，再用气泡泛洪掩膜裁剪合成——文字永不出气泡框
"""
from __future__ import annotations

import io
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
        s = get_settings()
        self.pad_ratio = max(0.02, float(s.render_padding))
        self.vertical_min_ratio = max(1.0, float(s.render_vertical_min_ratio))

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
        img_w, img_h = img.size
        min_font_size = max(1, round((img_w + img_h) / 200))

        bgr = np.array(img)[:, :, ::-1].copy()

        groups = self._group_by_bubble(bgr, regions, img_w, img_h)

        # 每组使用独立 overlay，先按自身容器裁剪再合成，避免失败组污染其他组的覆盖率统计。
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

        for _bubble, group_regions in groups:
            bb, mask = self._bubble_geometry(bgr, group_regions, img_w, img_h)
            if bb is None:
                continue
            bx0, by0, bx1, by1 = bb
            bw, bh = bx1 - bx0, by1 - by0
            if bw <= 0 or bh <= 0:
                continue
            block = self._group_block_text(group_regions)
            block = self._layout_block_text(block, target_lang)
            fill, stroke = self._text_colors(group_regions)
            group_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            odraw = ImageDraw.Draw(group_overlay)
            target_value = getattr(target_lang, "value", target_lang)
            if target_value != "en" and self._use_vertical(group_regions, bw, bh):
                if block:
                    self._render_vertical_bubble_block(
                        odraw, block, bx0, by0, bw, bh, min_font_size, fill=fill, stroke=stroke
                    )
                else:
                    self._render_vertical_bubble(
                        odraw, group_regions, bx0, by0, bw, bh, min_font_size, fill=fill, stroke=stroke
                    )
            else:
                self._render_horizontal_bubble(
                    odraw, group_regions, bx0, by0, bw, bh, min_font_size, block=block, fill=fill, stroke=stroke
                )

            group_alpha = np.array(group_overlay.getchannel("A"))
            drawn = group_alpha > 0
            if not drawn.any():
                continue
            group_clip = np.zeros((img_h, img_w), np.uint8)

            if mask is not None:
                # 可靠容器是硬边界；覆盖不足说明排版/几何不可信，宁可跳过也不放行到人物背景。
                coverage = float((drawn & (mask > 0)).sum()) / float(drawn.sum())
                if coverage < 0.5:
                    continue
                group_clip = mask
            else:
                cx0, cy0 = max(0, bx0), max(0, by0)
                cx1, cy1 = min(img_w, bx1), min(img_h, by1)
                if cx1 > cx0 and cy1 > cy0:
                    group_clip[cy0:cy1, cx0:cx1] = 255
            keep = np.minimum(group_alpha, group_clip).astype(np.uint8)
            group_overlay.putalpha(Image.fromarray(keep, "L"))
            overlay = Image.alpha_composite(overlay, group_overlay)

        base = img.convert("RGBA")
        merged = Image.alpha_composite(base, overlay).convert("RGB")

        buf = io.BytesIO()
        merged.save(buf, format="PNG")
        return buf.getvalue()

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

        x0 = min(r.bounds[0] for r in group_regions)
        y0 = min(r.bounds[1] for r in group_regions)
        x1 = max(r.bounds[2] for r in group_regions)
        y1 = max(r.bounds[3] for r in group_regions)

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

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)

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
        while changed and guard < 2000:
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
        if not text.strip():
            return
        pad = self.pad_ratio
        avail_w = bw * (1 - 2 * pad)
        avail_h = bh * (1 - 2 * pad)
        max_font = max(min_font_size, int(bh * MAX_FONT_RATIO))
        font_size = self._select_horizontal_font(draw, text, avail_w, avail_h, max_font, min_font_size)

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
            fill=fill,
            stroke_width=sw,
            stroke_fill=stroke,
            spacing=spacing,
            align="center",
        )

    def _select_horizontal_font(self, draw, text, avail_w, avail_h, max_size, min_size) -> int:
        """横排字号：优先单行，否则取能放下的最大字号并规避孤字最后一行"""
        # 单行优先：某字号整段一行放得下 → 直接单行（不折出怪行）
        for font in range(max_size, min_size - 1, -1):
            if self._fits_oneline(draw, text, font, avail_w, avail_h):
                return font
        # 多行：取最大适应字号，并尝试避免最后一行孤儿
        for font in range(max_size, min_size - 1, -1):
            if self._fits_in(draw, text, font, avail_w, avail_h):
                return self._refine_last_line(draw, text, font, avail_w, avail_h, min_size)
        return min_size

    def _fits_oneline(self, draw, text, font_size, max_w, max_h) -> bool:
        font = self._get_font(font_size)
        sw = max(1, int(font_size * STROKE_RATIO))
        w = _text_length(font, text) + sw * 2
        h = font_size * 1.4 + sw * 2
        return w <= max_w and h <= max_h

    def _refine_last_line(self, draw, text, font_size, avail_w, avail_h, min_size) -> int:
        """若多行最后一行过短（孤字），尝试更小字号减少行数"""
        font = self._get_font(font_size)
        wrapped = self._wrap_paragraph(text, font, avail_w)
        if len(wrapped) < 2:
            return font_size
        last = wrapped[-1].strip()
        if not last:
            return font_size
        max_len = max(len(l.strip()) for l in wrapped)
        if max_len <= 0 or len(last) / max_len >= 0.25:
            return font_size
        for trial in range(font_size - 1, max(min_size, int(font_size * 0.8)) - 1, -1):
            tw = self._wrap_paragraph(text, self._get_font(trial), avail_w)
            if len(tw) < len(wrapped) and self._fits_in(draw, text, trial, avail_w, avail_h):
                return trial
            if len(tw) < len(wrapped):
                break
        return font_size

    def _render_vertical_bubble(
        self, draw, regions, bx0, by0, bw, bh, min_font_size, fill=(0, 0, 0), stroke=(255, 255, 255)
    ):
        """竖排气泡（每 region 即一列）：整组对称居中，每列垂直居中"""
        columns = sorted(regions, key=lambda r: -r.bounds[0])
        texts = [t for t in ((r.translated or "").strip() for r in columns) if t]
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
        """竖排布局：返回 (font_size, balanced_columns, char_h)；无解时 (0, [], 0)

        字号上界由「列宽」约束（均衡切列会把长行拆入多列，高度约束在平衡后按列校验）。
        """
        if not lines:
            return 0, [], 0
        total = sum(len(ln) for ln in lines)
        upper = int(avail_w * VERTICAL_COL_USE_RATIO)
        upper = max(min_font_size, upper)
        font = upper
        while font >= min_font_size:
            char_h = max(1, int(font * VERTICAL_CHAR_RATIO))
            cols = self._balance_columns(lines, total, avail_h, char_h)
            if cols is None:
                font -= 1
                continue
            # 每列高度校验 + 列数宽度校验
            max_col_len = max(len(c) for c in cols)
            if max_col_len * char_h <= avail_h and font <= (avail_w / len(cols)) * VERTICAL_COL_USE_RATIO:
                return font, cols, char_h
            font -= 1
        # 兜底：最小字号下能放下的均衡列
        char_h = max(1, int(min_font_size * VERTICAL_CHAR_RATIO))
        cols = self._balance_columns(lines, total, avail_h, char_h)
        if cols and max(len(c) for c in cols) * char_h <= avail_h:
            return min_font_size, cols, char_h
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
        for ch in text:
            if current and _text_length(font, current) >= target:
                lines.append(current)
                current = ""
            current += ch
            if _text_length(font, current) > max_width and len(current) > 1:
                lines.append(current[:-1])
                current = current[-1]
        if current:
            lines.append(current)
        if not lines:
            lines = [text]
        return lines

    @staticmethod
    def _wrap_latin_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        """英文优先在词间换行，单个超长词才安全拆分。"""
        lines: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if current and _text_length(font, candidate) > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
            while current and _text_length(font, current) > max_width:
                cut = len(current) - 1
                while cut > 1 and _text_length(font, current[:cut]) > max_width:
                    cut -= 1
                lines.append(current[:cut])
                current = current[cut:]
        if current:
            lines.append(current)
        return lines or [text]

    @staticmethod
    def _greedy_wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        """宽贪心换行：每行尽量填满"""
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
