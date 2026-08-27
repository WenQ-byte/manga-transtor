"""图像修复引擎：擦除检测区域的文字

方案：漫画气泡内通常为纯色（白/浅色），文字为深色笔画。
采样气泡背景色，精确检测文字笔画（与背景差异大的像素），
用背景色填充，避免 cv2.inpaint 混合周围像素产生的污渍。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.engines.base import BaseInpainter
from app.services.pipeline import TextRegion


class CVInpainter(BaseInpainter):
    """基于背景色填充的文字擦除（无模型，轻量）"""

    name = "cv"

    def inpaint(self, image_path: Path, regions: list[TextRegion]) -> Path:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img).astype(np.int16)
        gray = np.array(img.convert("L")).astype(np.int16)
        h, w = gray.shape

        try:
            import cv2
        except ImportError:
            cv2 = None

        result = arr.copy()

        for region in regions:
            x0, y0, x1, y1 = region.bounds
            # 向内收缩一点，避免包含气泡边框
            pad = 3
            x0 = max(0, x0 + pad)
            y0 = max(0, y0 + pad)
            x1 = min(w, x1 - pad)
            y1 = min(h, y1 - pad)
            if x1 <= x0 or y1 <= y0:
                continue

            # 背景色 = 区域边缘采样（气泡内部颜色）
            bg = self._sample_edge_color(arr, x0, y0, x1, y1)
            bg_gray = int(np.mean(bg))

            # 文字掩膜 = 与背景灰度差异大的像素
            region_gray = gray[y0:y1, x0:x1]
            diff = np.abs(region_gray - bg_gray)
            text_mask = diff > 45

            if text_mask.size and text_mask.sum() > 0:
                if cv2 is not None:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    text_mask = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                # 用背景色填充文字
                result[y0:y1, x0:x1][text_mask] = bg

        return self._save_temp(result.astype(np.uint8))

    def _sample_edge_color(self, arr, x0, y0, x1, y1):
        """采样区域边缘的背景色（中位数）"""
        h, w = arr.shape[:2]
        samples = []
        if y0 > 0:
            samples.append(arr[y0 : y0 + 2, x0:x1].reshape(-1, 3))
        if y1 < h:
            samples.append(arr[y1 - 2 : y1, x0:x1].reshape(-1, 3))
        if x0 > 0:
            samples.append(arr[y0:y1, x0 : x0 + 2].reshape(-1, 3))
        if x1 < w:
            samples.append(arr[y0:y1, x1 - 2 : x1].reshape(-1, 3))
        if samples:
            all_px = np.concatenate(samples, axis=0)
            return np.median(all_px, axis=0)
        # 无边缘可采样，默认白色
        return np.array([255, 255, 255], dtype=np.int16)

    def _save_temp(self, arr: np.ndarray) -> Path:
        tmp = tempfile.mkdtemp(prefix="manga_inpaint_")
        path = Path(tmp) / "cleaned.png"
        Image.fromarray(arr).save(path)
        return path


def create_inpainter() -> BaseInpainter:
    return CVInpainter()
