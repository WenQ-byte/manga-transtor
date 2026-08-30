"""MangaOcr（kha-white/manga-ocr-base，HF）识别包装

manga-ocr 对日文漫画风格化字体/手写体召回更强，但无置信度输出。
依赖：pip install manga-ocr（拉取 transformers 与 HF 权重 ~600MB，首次运行联网下载）。
"""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image

from .quadrilateral import Quadrilateral


class MangaOcrWrapper:
    """懒加载 MangaOcr；模型缺失/未安装时 available=False"""

    def __init__(self, text_height: int = 96):
        self.text_height = text_height
        self._mocr = None
        self._error = ""
        try:
            from manga_ocr import MangaOcr  # noqa: F401

            self._mocr = MangaOcr()
        except Exception as e:  # noqa: BLE001
            self._error = f"manga-ocr 加载失败: {e}"
            self._mocr = None

    @property
    def available(self) -> bool:
        return self._mocr is not None

    def recognize(self, image: np.ndarray, textlines: List[Quadrilateral]) -> None:
        if self._mocr is None:
            return
        import cv2

        for q in textlines:
            try:
                crop = q.get_transformed_region(image, q.direction, self.text_height)
                if crop.size == 0:
                    continue
                # get_transformed_region 会把竖排旋转 90° CCW 供水平训练模型；manga-ocr 本身支持任意朝向，
                # 竖排应恢复为竖直裁剪（旋转回去再读），否则竖排文字会被读成横排列乱码（NS/SM/纽哈梅等）。
                if q.direction == "v":
                    crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                txt = self._mocr(Image.fromarray(crop))
                if txt:
                    q.text = txt
                    q.prob = max(float(q.prob), 0.85)
            except Exception:  # noqa: BLE001
                continue