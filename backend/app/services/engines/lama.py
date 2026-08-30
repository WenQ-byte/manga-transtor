"""LaMa 神经修复引擎（方案B）：用大 LaMa（FFC）生成重建文字笔画覆盖的背景

- 依赖 torch（CPU 推理）
- 权重：lama_large_512px.ckpt（默认搜索：配置 > 项目 backend/data/models/ > 本机 manga-image-translator 路径）
- 掩膜复用 mask.py 的精确笔画掩膜（缓存到 region.mask）
- 无 torch / 无权重时 create_inpainter 回退到 CVInpainter
推理流程对齐 manga-image-translator 的 LamaMPEInpainter._infer：
阈值化 mask → 缩放到 ≤lama_inpaint_size → pad 到 8 的倍数 →
img*(1-mask) 输入 → sigmoid 输出 *255 → 按原始 mask 与原图混合。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import get_settings
from app.services.engines.base import BaseInpainter
from app.services.engines.mask import build_full_mask
from app.services.pipeline import TextRegion


class LaMaInpainter(BaseInpainter):
    """LaMa (big) 神经修复，CPU 推理"""

    name = "lama"

    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self._load_error = ""
        self._load()

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def load_error(self) -> str:
        return self._load_error

    def _resolve_model_path(self) -> Path | None:
        candidates = []
        if self.settings.lama_model_path:
            candidates.append(Path(self.settings.lama_model_path))
        # 项目内 models 目录
        from app.config import BASE_DIR

        candidates.append(BASE_DIR / "data" / "models" / "lama_large_512px.ckpt")
        # 本机 manga-image-translator 已下载权重（开发环境便捷回退）
        candidates.append(
            Path(r"D:\Develop\code\Python\manga-image-translator\models\inpainting\lama_large_512px.ckpt")
        )
        for c in candidates:
            if c and c.exists():
                return c
        return None

    def _load(self):
        model_path = self._resolve_model_path()
        if model_path is None:
            self._load_error = "未找到 lama_large_512px.ckpt 权重"
            return
        try:
            import torch

            from app.services.engines.lama_model import load_lama_large

            device = "cpu"
            want = (self.settings.inpaint_device or "cpu").lower()
            if want.startswith("cuda") and torch.cuda.is_available():
                device = want
            self.model = load_lama_large(str(model_path), device=device)
            self._model_path = model_path
        except ImportError as e:
            self._load_error = f"torch 未安装，无法使用 LaMa 修复: {e}"
            self.model = None
        except Exception as e:  # noqa: BLE001
            self._load_error = f"LaMa 模型加载失败: {e}"
            self.model = None

    def inpaint(self, image_path: Path, regions: list[TextRegion]) -> Path:
        import cv2

        img = np.array(Image.open(image_path).convert("RGB")).astype(np.uint8)
        mask = build_full_mask(img, regions)
        if not mask.any():
            return self._save_temp(img)

        if not self.available:
            # 理论上 create_inpainter 不会返回不可用的 lama，这里兜底
            return self._save_temp(img)

        from app.services.engines.inpainter import CVInpainter

        cv_inpainter = CVInpainter()
        result = img.copy()
        flat_mask = cv_inpainter._flat_background_mask(img, regions)
        flat_body = cv2.erode(flat_mask, np.ones((3, 3), np.uint8), iterations=1)
        cv_inpainter._fill_with_row_bg(result, flat_body, prefer_bright=True)
        neural_mask = cv2.bitwise_and(mask, cv2.bitwise_not(flat_body))
        if neural_mask.any():
            result = self._infer_by_crops(result, neural_mask)
        result = cv_inpainter._second_pass_residual(result, regions)
        return self._save_temp(result)

    def _infer_by_crops(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = image.copy()
        boxes = self._mask_crop_boxes(mask)
        if not boxes:
            return result

        page_area = image.shape[0] * image.shape[1]
        crop_area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes)
        if crop_area >= page_area * 0.85:
            return self._infer(image, mask)

        for x0, y0, x1, y1 in boxes:
            crop_mask = mask[y0:y1, x0:x1]
            if not crop_mask.any():
                continue
            crop = result[y0:y1, x0:x1]
            result[y0:y1, x0:x1] = self._infer(crop, crop_mask)
        return result

    @staticmethod
    def _mask_crop_boxes(
        mask: np.ndarray,
        merge_gap: int = 14,
        context: int = 40,
    ) -> list[tuple[int, int, int, int]]:
        import cv2

        binary = (mask > 0).astype(np.uint8)
        if not binary.any():
            return []

        kernel_size = merge_gap * 2 + 1
        merged = cv2.dilate(
            binary,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        height, width = binary.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []
        for label in range(1, count):
            x, y, w, h, _ = stats[label]
            x0 = max(0, int(x) - context)
            y0 = max(0, int(y) - context)
            x1 = min(width, int(x + w) + context)
            y1 = min(height, int(y + h) + context)
            boxes.append((x0, y0, x1, y1))
        return boxes

    def _infer(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import cv2
        import torch

        size = int(self.settings.lama_inpaint_size or 1024)
        img_original = image.copy()
        mask_original = mask.copy()
        mask_original = (mask_original >= 127).astype(np.float32)[:, :, None]

        height, width = image.shape[:2]
        if max(image.shape[0], image.shape[1]) > size:
            image = self._resize_keep_aspect(image, size)
            mask = self._resize_keep_aspect(mask, size)

        # pad 到 8 的倍数
        pad = 8
        h, w = image.shape[:2]
        new_h = h if h % pad == 0 else h + (pad - h % pad)
        new_w = w if w % pad == 0 else w + (pad - w % pad)
        if new_h != h or new_w != w:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        img_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0
        mask_t = (mask_t >= 0.5).float()

        with torch.no_grad():
            img_t = img_t * (1 - mask_t)
            out_t = self.model(img_t, mask_t)

        out = (out_t.to(torch.float32).cpu().squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        if new_h != height or new_w != width:
            out = cv2.resize(out, (width, height), interpolation=cv2.INTER_LINEAR)

        # 按原始 mask 混合：仅掩膜内用修复结果，其余保留原图
        ans = out.astype(np.float32) * mask_original + img_original.astype(np.float32) * (1 - mask_original)
        return ans.astype(np.uint8)

    @staticmethod
    def _resize_keep_aspect(img: np.ndarray, size: int) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        if max(h, w) <= size:
            return img
        scale = size / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _save_temp(self, arr: np.ndarray) -> Path:
        tmp = tempfile.mkdtemp(prefix="manga_inpaint_")
        path = Path(tmp) / "cleaned.png"
        Image.fromarray(arr).save(path)
        return path


def create_lama_inpainter() -> LaMaInpainter | None:
    """可用时返回 LaMaInpainter，否则 None（让工厂回退 CV）"""
    engine = LaMaInpainter()
    if engine.available:
        return engine
    return None
