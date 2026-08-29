"""图像缩放工具（resize_aspect_ratio），对齐 MIT detection/default_utils/imgproc.py"""
from __future__ import annotations

import cv2
import numpy as np


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape
    target_size = mag_ratio * square_size
    ratio = target_size / max(height, width)
    target_h, target_w = int(round(height * ratio)), int(round(width * ratio))
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    MULT = 256
    target_h32, target_w32 = target_h, target_w
    pad_h, pad_w = 0, 0
    if target_h % MULT != 0:
        pad_h = MULT - target_h % MULT
        target_h32 = target_h + pad_h
    if target_w % MULT != 0:
        pad_w = MULT - target_w % MULT
        target_w32 = target_w + pad_w
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
    resized[0:target_h, 0:target_w, :] = proc
    return resized, ratio, (int(target_w / 2), int(target_h / 2)), pad_w, pad_h