"""坐标调整工具，对齐 MIT detection/default_utils/craft_utils.py 的 adjustResultCoordinates"""
from __future__ import annotations

import numpy as np


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= ratio_w * ratio_net, ratio_h * ratio_net
    return polys