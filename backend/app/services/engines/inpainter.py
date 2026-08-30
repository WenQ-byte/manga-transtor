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
        # 纯色气泡用局部背景直接重建，避免 TELEA 从边缘把灰色笔画带回；
        # 网点/复杂纹理仍保留完整 TELEA 修复。
        flat_mask = self._flat_background_mask(img, regions)
        body = cv2.erode(flat_mask, np.ones((3, 3), np.uint8), iterations=1)
        self._fill_with_row_bg(result, body, prefer_bright=True)
        repair_mask = cv2.bitwise_and(mask, cv2.bitwise_not(body))
        if repair_mask.any():
            repaired = cv2.inpaint(result, repair_mask, 3, cv2.INPAINT_TELEA)
            result[repair_mask > 0] = repaired[repair_mask > 0]

        result = self._second_pass_residual(result, regions)
        return self._save_temp(result)

    def _flat_background_mask(self, img: np.ndarray, regions: list[TextRegion]) -> np.ndarray:
        """筛出边带灰度稳定的纯色文字区域，供局部背景填充。"""
        import cv2

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        flat = np.zeros((h, w), np.uint8)
        for region in regions:
            cached = region.mask
            if not isinstance(cached, dict) or "bbox" not in cached or "patch" not in cached:
                continue
            x0, y0, x1, y1 = [int(v) for v in cached["bbox"]]
            patch = np.asarray(cached["patch"], dtype=np.uint8)
            if patch.shape != (y1 - y0, x1 - x0) or not patch.any():
                continue
            margin = 6
            ex0, ey0 = max(0, x0 - margin), max(0, y0 - margin)
            ex1, ey1 = min(w, x1 + margin), min(h, y1 + margin)
            local_mask = np.zeros((ey1 - ey0, ex1 - ex0), np.uint8)
            local_mask[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = patch
            outer = cv2.dilate(local_mask, np.ones((9, 9), np.uint8), iterations=1) > 0
            inner = cv2.dilate(local_mask, np.ones((3, 3), np.uint8), iterations=1) > 0
            ring = outer & ~inner
            samples = gray[ey0:ey1, ex0:ex1][ring]
            if samples.size < 12:
                continue
            source_patch = gray[y0:y1, x0:x1]
            white_ratio = float((source_patch > 245).mean()) if source_patch.size else 0.0
            q25, q50, q75 = np.percentile(samples, [25, 50, 75])
            bright_ratio = float((samples > 220).mean())
            light_flat = (
                white_ratio >= 0.35
                and q50 >= 220
                and (q75 - q25 <= 65 or bright_ratio >= 0.72)
            )
            if light_flat:
                points = region.poly or region.box
                safe = np.zeros_like(patch)
                if points:
                    pts = np.asarray(
                        [[int(p[0]) - x0, int(p[1]) - y0] for p in points],
                        dtype=np.int32,
                    )
                    cv2.fillPoly(safe, [pts.reshape(-1, 1, 2)], 255)
                    safe = cv2.erode(safe, np.ones((3, 3), np.uint8), iterations=1)
                    safe = cv2.bitwise_and(safe, patch)
                flat[y0:y1, x0:x1] = np.maximum(flat[y0:y1, x0:x1], safe)
        return flat

    def _second_pass_residual(self, img: np.ndarray, regions: list[TextRegion]) -> np.ndarray:
        """在已知文字区域内检测与局部背景反差明显的残字，再做一次保守修复。"""
        import cv2

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        residual = np.zeros((h, w), np.uint8)
        for region in regions:
            cached = region.mask
            if not isinstance(cached, dict) or "bbox" not in cached or "patch" not in cached:
                continue
            x0, y0, x1, y1 = [int(v) for v in cached["bbox"]]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            patch = np.asarray(cached["patch"], dtype=np.uint8)
            if patch.shape != (y1 - y0, x1 - x0) or not patch.any():
                continue
            core = cv2.dilate(patch, np.ones((3, 3), np.uint8), iterations=1) > 0
            outer = cv2.dilate(patch, np.ones((9, 9), np.uint8), iterations=1) > 0
            inner = cv2.dilate(patch, np.ones((5, 5), np.uint8), iterations=1) > 0
            ring = outer & ~inner
            local_gray = gray[y0:y1, x0:x1]
            samples = local_gray[ring]
            if samples.size < 12:
                continue
            q25, q75 = np.percentile(samples, [25, 75])
            # 网点/复杂纹理的边带跨度大，二次处理容易误擦背景，直接跳过。
            if q75 - q25 > 55:
                continue
            if q75 >= 190:
                bg = float(q75)
            elif q25 <= 65:
                bg = float(q25)
            else:
                bg = float(np.median(samples))
            threshold = max(20.0, float(np.std(samples)) * 1.5)
            candidate = core & (np.abs(local_gray.astype(np.float32) - bg) > threshold)
            if candidate.any():
                residual[y0:y1, x0:x1][candidate] = 255
        if not residual.any():
            return img
        residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        residual = cv2.dilate(residual, np.ones((3, 3), np.uint8), iterations=1)
        return cv2.inpaint(img, residual, 3, cv2.INPAINT_TELEA)

    def _fill_with_row_bg(self, img: np.ndarray, body_mask: np.ndarray, prefer_bright: bool = False) -> None:
        """对 body_mask 像素按行重采样背景（左右条带中位数）"""
        rows = np.where(body_mask.any(axis=1))[0]
        if rows.size == 0:
            return
        edge = 12 if prefer_bright else 4
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
                src = np.concatenate(samples, axis=0)
                # 白色/浅色气泡中，邻近文字笔画可能占据一侧采样带；优先用亮像素
                # 的中位数，避免把相邻黑字或整页远处纹理涂回擦除区。
                bright = src[np.mean(src, axis=1) > 180]
                chosen = bright if prefer_bright and len(bright) else (bright if len(bright) >= 2 else src)
                color = np.median(chosen, axis=0).astype(np.uint8)
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
