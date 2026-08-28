"""气泡过滤：只保留位于气泡内的文字区域，丢弃气泡外的涂鸦/噪声

原理：漫画气泡内部通常是纯色区域（白/浅色），被黑色描边包围。
从文本框外侧采样周围底色并做泛洪填充：
- 文字在气泡内 → 填充被描边限制在气泡内，面积适中且覆盖文本框 → 保留
- 涂鸦在页面上（无气泡）→ 填充泄漏到整页背景，面积巨大 → 丢弃
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.services.pipeline import TextRegion

# 泛洪填充颜色容差
FLOOD_TOL = 40
# 填充面积超过整图该比例视为泄漏（非气泡）
MAX_FILL_RATIO = 0.5

try:
    import cv2
except ImportError:  # noqa: BLE001
    cv2 = None


class BubbleFilter:
    """基于背景色泛洪填充的气泡内外判别"""

    name = "bubble"

    def filter(self, image_path: Path, regions: list[TextRegion]) -> list[TextRegion]:
        if not regions or cv2 is None:
            return regions
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:  # noqa: BLE001
            return regions
        bgr = np.array(img)[:, :, ::-1].copy()
        img_h, img_w = bgr.shape[:2]

        kept = []
        for region in regions:
            if self._in_bubble(bgr, region.bounds, img_w, img_h):
                kept.append(region)

        # 至少保留一个气泡内文字才启用过滤，避免整页文字被误删
        return kept if kept else regions

    def _in_bubble(self, bgr, bounds, img_w, img_h) -> bool:
        x0, y0, x1, y1 = [int(v) for v in bounds]
        pad = 5
        sx0, sy0 = max(0, x0 - pad), max(0, y0 - pad)
        sx1, sy1 = min(img_w, x1 + pad), min(img_h, y1 + pad)
        if sx1 <= sx0 or sy1 <= sy0:
            return True

        h, w = bgr.shape[:2]
        tol = FLOOD_TOL
        flags = 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        # 候选种子：外扩框四角 + 四边中点（都在文本框外，属于周围底色）
        seeds = [
            (sx0, sy0),
            (sx1 - 1, sy0),
            (sx0, sy1 - 1),
            (sx1 - 1, sy1 - 1),
            ((sx0 + sx1) // 2, sy0),
            ((sx0 + sx1) // 2, sy1 - 1),
            (sx0, (sy0 + sy1) // 2),
            (sx1 - 1, (sy0 + sy1) // 2),
        ]
        img_area = img_w * img_h
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
            fx0 = int(xs.min())
            fy0 = int(ys.min())
            fx1 = int(xs.max()) + 1
            fy1 = int(ys.max()) + 1
            contains = fx0 <= x0 and fy0 <= y0 and fx1 >= x1 and fy1 >= y1
            bounded = (fx1 - fx0) * (fy1 - fy0) < img_area * MAX_FILL_RATIO
            if contains and bounded:
                return True
        return False


def create_bubble_filter() -> BubbleFilter:
    return BubbleFilter()
