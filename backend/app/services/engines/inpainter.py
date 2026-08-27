"""图像修复引擎：擦除检测区域的文字"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.engines.base import BaseInpainter
from app.services.pipeline import TextRegion


class CVInpainter(BaseInpainter):
    """基于 OpenCV 图像修复擦除文字（无模型，轻量）

    对每个文本区域生成掩膜，使用邻域背景色填充。
    """

    name = "cv"

    def inpaint(self, image_path: Path, regions: list[TextRegion]) -> Path:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)

        try:
            import cv2
        except ImportError:
            # 无 OpenCV：用背景色填充
            for region in regions:
                x0, y0, x1, y1 = region.bounds
                # 取区域边缘的采样色作为背景
                bg = self._sample_background(arr, x0, y0, x1, y1)
                arr[y0:y1, x0:x1] = bg
            return self._save_temp(arr)

        mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        # 膨胀掩膜以覆盖文字边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        for region in regions:
            x0, y0, x1, y1 = region.bounds
            x0 = max(0, x0 - 2)
            y0 = max(0, y0 - 2)
            x1 = min(arr.shape[1], x1 + 2)
            y1 = min(arr.shape[0], y1 + 2)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 255
        mask = cv2.dilate(mask, kernel, iterations=1)

        try:
            result = cv2.inpaint(arr, mask, 3, cv2.INPAINT_TELEA)
        except Exception:
            result = arr.copy()
            for region in regions:
                x0, y0, x1, y1 = region.bounds
                bg = self._sample_background(arr, x0, y0, x1, y1)
                result[y0:y1, x0:x1] = bg
        return self._save_temp(result)

    def _sample_background(self, arr, x0, y0, x1, y1):
        """采样区域边缘的背景色"""
        h, w = arr.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        samples = []
        if y0 > 0:
            samples.append(arr[y0 - 1, x0:x1].reshape(-1, 3))
        if y1 < h:
            samples.append(arr[y1, x0:x1].reshape(-1, 3))
        if x0 > 0:
            samples.append(arr[y0:y1, x0].reshape(-1, 3))
        if x1 < w:
            samples.append(arr[y0:y1, x1 - 1].reshape(-1, 3))
        if samples:
            all_px = np.concatenate(samples)
            return tuple(int(v) for v in np.median(all_px, axis=0))
        return (255, 255, 255)

    def _save_temp(self, arr: np.ndarray) -> Path:
        tmp = tempfile.mkdtemp(prefix="manga_inpaint_")
        path = Path(tmp) / "cleaned.png"
        Image.fromarray(arr).save(path)
        return path


def create_inpainter() -> BaseInpainter:
    return CVInpainter()
