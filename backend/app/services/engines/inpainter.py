"""图像修复引擎：擦除检测区域的文字

CV 无模型方案（方案A）：精确笔画掩膜（Otsu+poly）+ 局部行背景填充 + 边界 TELEA 羽化。
优于旧版：不再是全区域单一中位数色平涂，且边界无硬接缝。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.engines.base import BaseInpainter
from app.services.engines.mask import build_full_mask
from app.services.pipeline import TextRegion


class CVInpainter(BaseInpainter):
    """无模型修复：精确掩膜 + 行背景重建 + TELEA 边界羽化"""

    name = "cv"

    def inpaint(self, image_path: Path, regions: list[TextRegion]) -> Path:
        import cv2

        img = np.array(Image.open(image_path).convert("RGB")).astype(np.uint8)
        # 生成整图掩膜（缓存到 region.mask）
        mask = build_full_mask(img, regions)
        if not mask.any():
            return self._save_temp(img)

        result = img.copy()
        # 1. 局部行背景填充掩膜内部（保留气泡渐变）
        body = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        self._fill_with_row_bg(result, body)
        # 2. 边界环 TELEA 羽化（补抗锯齿残影，消除硬边）
        ring = cv2.bitwise_and(mask, cv2.bitwise_not(body))
        if ring.any():
            repaired = cv2.inpaint(result, ring, 2, cv2.INPAINT_TELEA)
            result[ring > 0] = repaired[ring > 0]

        return self._save_temp(result)

    def _fill_with_row_bg(self, img: np.ndarray, body_mask: np.ndarray) -> None:
        """对 body_mask 像素按行重采样背景（左右条带中位数）"""
        rows = np.where(body_mask.any(axis=1))[0]
        if rows.size == 0:
            return
        edge = 4  # 边缘采样带宽
        for y in rows:
            xs = body_mask[y]
            if not xs.any():
                continue
            indices = np.flatnonzero(xs)
            starts = np.r_[0, np.flatnonzero(np.diff(indices) > 1) + 1]
            ends = np.r_[starts[1:] - 1, len(indices) - 1]
            for start, end in zip(starts, ends):
                x0, x1 = int(indices[start]), int(indices[end])
                left = img[y, max(0, x0 - edge):x0].reshape(-1, 3)
                right = img[y, x1 + 1:min(img.shape[1], x1 + 1 + edge)].reshape(-1, 3)
                samples = [part for part in (left, right) if part.size]
                if not samples:
                    continue
                color = np.median(np.concatenate(samples, axis=0), axis=0).astype(np.uint8)
                img[y, x0:x1 + 1] = color

    def _save_temp(self, arr: np.ndarray) -> Path:
        tmp = tempfile.mkdtemp(prefix="manga_inpaint_")
        path = Path(tmp) / "cleaned.png"
        Image.fromarray(arr).save(path)
        return path


def create_inpainter() -> BaseInpainter:
    """按配置选择修复引擎：lama（神经网络）优先，失败回退 cv（无模型）"""
    from app.config import get_settings

    if get_settings().inpainter_backend == "lama":
        try:
            from app.services.engines.lama import create_lama_inpainter

            engine = create_lama_inpainter()
            if engine is not None:
                return engine
        except Exception:  # noqa: BLE001
            pass
    return CVInpainter()
