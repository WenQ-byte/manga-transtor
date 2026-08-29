"""气泡/拟声词判别（is_ignore），对齐 MIT utils/bubble.py"""
from __future__ import annotations

import cv2
import numpy as np


def check_color(image):
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])[..., np.newaxis]
    color_distance = np.sum((image - gray) ** 2, axis=-1)
    n = np.sum(color_distance > 100)
    return n > 10


def is_ignore(region_img, ignore_bubble=0):
    """根据文本块四周 2px 边缘黑白像素占比判断是否为正常气泡区，是则无需翻译"""
    if ignore_bubble < 1 or ignore_bubble > 50:
        return False
    _, binary_raw_mask = cv2.threshold(region_img, 127, 255, cv2.THRESH_BINARY)
    height, width = binary_raw_mask.shape[:2]
    total = 0
    val0 = 0
    s = binary_raw_mask
    val0 += int((s[0:2, 0:width] == 0).sum())
    total += s[0:2, 0:width].size
    val0 += int((s[height - 2 : height, 0:width] == 0).sum())
    total += s[height - 2 : height, 0:width].size
    val0 += int((s[2 : height - 2, 0:2] == 0).sum())
    total += s[2 : height - 2, 0:2].size
    val0 += int((s[2 : height - 2, width - 2 : width] == 0).sum())
    total += s[2 : height - 2, width - 2 : width].size
    if total == 0:
        return False
    ratio = round(val0 / total, 6) * 100
    if ignore_bubble <= ratio <= (100 - ignore_bubble):
        return True
    if check_color(region_img):
        return True
    return False