"""与 MIT utils/generic2.py 相同的基础工具函数"""
from __future__ import annotations

import unicodedata
from typing import List

import cv2
import numpy as np


def color_difference(rgb1: List, rgb2: List) -> float:
    color1 = np.array(rgb1, dtype=np.uint8).reshape(1, 1, 3)
    color2 = np.array(rgb2, dtype=np.uint8).reshape(1, 1, 3)
    diff = cv2.cvtColor(color1, cv2.COLOR_RGB2LAB).astype(np.float32) - cv2.cvtColor(
        color2, cv2.COLOR_RGB2LAB
    ).astype(np.float32)
    diff[..., 0] *= 0.392
    diff = np.linalg.norm(diff, axis=2)
    return diff.item()


def is_punctuation(ch):
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def is_whitespace(ch):
    if ch in (" ", "\t", "\n", "\r") or ord(ch) == 0:
        return True
    return unicodedata.category(ch) == "Zs"


def is_control(ch):
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch) in ("Cc", "Cf")


def is_valuable_char(ch):
    return not is_punctuation(ch) and not is_control(ch) and not is_whitespace(ch) and not ch.isdigit()


def is_valuable_text(text):
    return any(is_valuable_char(ch) for ch in text)


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    return 0


def is_right_to_left_char(ch):
    return (
        "\u0600" <= ch <= "\u06FF"
        or "\u0750" <= ch <= "\u077F"
        or "\u08A0" <= ch <= "\u08FF"
        or "\uFB50" <= ch <= "\uFDFF"
        or "\uFE70" <= ch <= "\uFEFF"
        or "\U00010E60" <= ch <= "\U00010E7F"
        or "\U0001EE00" <= ch <= "\U0001EEFF"
    )