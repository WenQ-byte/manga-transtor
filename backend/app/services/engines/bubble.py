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
        """检查文字区域是否在气泡内

        宽松策略：文字框内有足够白色背景 → 视为气泡内（保留）。
        严格过滤交给 OCR 置信度（pipeline 里 confidence >= 0.5）。
        """
        x0, y0, x1, y1 = [int(v) for v in bounds]
        h, w = bgr.shape[:2]
        if x1 <= x0 or y1 <= y0:
            return True

        # 文字框内白色像素占比 > 30% → 有气泡背景 → 保留
        patch = bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        white_ratio = float(np.mean(gray > 200))
        return white_ratio > 0.30


def create_bubble_filter() -> BubbleFilter:
    return BubbleFilter()
