"""文本检测引擎: CV（OpenCV 启发式）+ Manga（移植自 manga-image-translator 的 DBNet/ctd）"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.services.engines.base import BaseDetector
from app.services.pipeline import TextRegion

try:
    import cv2
except ImportError:  # noqa: BLE001
    cv2 = None


class CVDetector(BaseDetector):
    """基于 OpenCV MSER + 自适应阈值轮廓检测的文字区域检测

    使用多策略检测候选文字区域，再合并相邻区域。
    轻量无需下载模型，适合气泡文字检测。
    """

    name = "cv"

    def __init__(self):
        try:
            import cv2  # noqa: F401
            self._cv2 = __import__("cv2")
        except ImportError:
            self._cv2 = None
        self._mser = None
        self._init_mser()

    def _init_mser(self):
        if self._cv2 is not None:
            try:
                # OpenCV 5.x 使用位置参数
                self._mser = self._cv2.MSER_create(3, 80, 40000, 0.3, 0.3)
            except TypeError:
                try:
                    self._mser = self._cv2.MSER_create()
                except Exception:
                    self._mser = None
            except Exception:
                self._mser = None

    def detect(self, image_path: Path) -> list[TextRegion]:
        img = Image.open(image_path)
        if self._cv2 is None:
            return self._detect_fallback(img)

        cv2 = self._cv2
        gray = np.array(img.convert("L"))
        h, w = gray.shape
        boxes: list[tuple[int, int, int, int]] = []

        # 策略1：MSER
        if self._mser is not None:
            try:
                mser_regions, _ = self._mser.detectRegions(gray)
                for region in mser_regions:
                    x, y, bw, bh = cv2.boundingRect(region.reshape(-1, 1, 2))
                    if self._valid_box(x, y, bw, bh, w, h):
                        boxes.append((x, y, bw, bh))
            except Exception:
                pass

        # 策略2：自适应阈值 + 形态学 + 轮廓
        try:
            # 增强对比度（CLAHE）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            # 二值化（文字为深色时）
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
            )
            # 形态学：水平方向膨胀连接文字笔画
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_h)
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, bw, bh = cv2.boundingRect(c)
                if self._valid_box(x, y, bw, bh, w, h):
                    boxes.append((x, y, bw, bh))
        except Exception:
            pass

        # 合并重叠框
        merged = self._merge_boxes(boxes)
        # 过滤过大的合并框（覆盖整个画面的垃圾框）
        merged = [b for b in merged if b[2] * b[3] < w * h * 0.9]

        regions = []
        for (x, y, bw, bh) in merged:
            box = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
            regions.append(TextRegion(box=box, confidence=0.8))
        return regions

    def _valid_box(self, x, y, bw, bh, img_w, img_h) -> bool:
        if bw < 10 or bh < 8 or bw > img_w * 0.9 or bh > img_h * 0.5:
            return False
        ratio = bw / bh
        if ratio > 15 or ratio < 0.3:
            return False
        return True

    def _merge_boxes(self, boxes: list, gap_x=30, gap_y=10) -> list:
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        merged = []
        for b in boxes:
            merged_into = False
            for i, m in enumerate(merged):
                x1, y1, w1, h1 = m
                x2, y2, w2, h2 = b
                inter_y = min(y1 + h1, y2 + h2) - max(y1, y2)
                min_h = min(h1, h2)
                v_overlap = inter_y / min_h if min_h > 0 else 0
                h_gap = max(x1, x2) - min(x1 + w1, x2 + w2)
                if v_overlap > 0.4 and h_gap < gap_x:
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1 + w1, x2 + w2) - nx
                    nh = max(y1 + h1, y2 + h2) - ny
                    merged[i] = (nx, ny, nw, nh)
                    merged_into = True
                    break
            if not merged_into:
                merged.append(tuple(b))
        return merged

    def _detect_fallback(self, img: Image.Image) -> list[TextRegion]:
        """无 OpenCV 时的简易检测：将图片分为 3x3 网格，返回非空区域"""
        w, h = img.size
        regions = []
        for gy in range(3):
            for gx in range(3):
                x0 = gx * w // 3
                y0 = gy * h // 3
                x1 = (gx + 1) * w // 3
                y1 = (gy + 1) * h // 3
                box = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                regions.append(TextRegion(box=box, confidence=0.3))
        return regions


class MangaDetector(BaseDetector):
    """移植自 manga-image-translator 的漫画文本检测器

    支持 default（DBNet+ResNet34）与 ctd（ComicTextDetector）。
    输出 textline 4 点多边形 + 逐像素文本掩膜（存入 region.mask）。
    """

    name = "manga"

    def __init__(self):
        from app.services.engines.mit.config import detector_params

        params = detector_params()
        self._params = params
        if params.detector == "ctd":
            from app.services.engines.mit.ctd import CTDDetector

            self._impl = CTDDetector(device=params.device)
        else:
            from app.services.engines.mit.dbnet import DefaultDetector

            self._impl = DefaultDetector(device=params.device)

    @property
    def available(self) -> bool:
        return True

    def detect(self, image_path: Path) -> list[TextRegion]:
        img = np.array(Image.open(image_path).convert("RGB"))
        p = self._params
        textlines, raw_mask = self._impl.detect(
            img,
            detect_size=p.detect_size,
            text_threshold=p.text_threshold,
            box_threshold=p.box_threshold,
            unclip_ratio=p.unclip_ratio,
        )

        # 掩膜缩到原图尺寸（MIT 输出常为原图 2 倍）
        full_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if raw_mask.size:
            full_mask = np.asarray(
                cv2.resize(raw_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            ).astype(np.uint8)

        regions: list[TextRegion] = []
        for quad in textlines:
            pts = np.asarray(quad.pts).round().astype(int)
            box = [[int(pts[i][0]), int(pts[i][1])] for i in range(4)]
            region = TextRegion(
                box=box,
                text="",
                confidence=float(quad.prob),
                poly=[[float(pts[i][0]), float(pts[i][1])] for i in range(4)],
                direction=quad.direction,
            )
            region._quad = quad
            region.mask = self._mask_patch(img, full_mask, region)
            regions.append(region)
        return regions

    @staticmethod
    def _mask_patch(image: np.ndarray, full_mask: np.ndarray, region: TextRegion, pad: int = 2):
        """从神经文本掩膜中裁剪某 region 的 0/255 笔画 patch（兼容 build_full_mask 契约）

        若启用 mit_ignore_bubble，且该区域判定为非气泡（拟声词/图片文字），返回 None 不生成掩膜（原文保留不擦除）。
        """
        import cv2

        from app.config import get_settings

        ignore_bubble = get_settings().mit_ignore_bubble
        x0, y0, x1, y1 = region.bounds
        h, w = full_mask.shape[:2]
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        patch = full_mask[y0:y1, x0:x1].astype(np.uint8)
        if not patch.any():
            return None
        if ignore_bubble >= 1:
            from app.services.engines.mit.bubble import is_ignore

            region_img = image[y0:y1, x0:x1]
            if is_ignore(region_img, ignore_bubble):
                return None
        patch = (patch > 127).astype(np.uint8) * 255
        # 覆盖抗锯齿边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        patch = cv2.dilate(patch, kernel, iterations=1)
        return {"bbox": (x0, y0, x1, y1), "patch": patch}


def create_detector_engine() -> BaseDetector:
    """按配置选择检测器：manga（需 torch）优先，失败回退 cv"""
    from app.config import get_settings

    settings = get_settings()
    mit_backends = ("mit48", "mangaocr", "mit48+mangaocr")
    backend = settings.detector_backend or (
        "manga" if settings.ocr_backend in mit_backends else "cv"
    )
    if backend == "manga":
        try:
            return MangaDetector()
        except Exception as e:  # noqa: BLE001
            print(f"[manga] 检测器加载失败，回退 CV: {e}")
    return CVDetector()
