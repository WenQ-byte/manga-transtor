"""精确笔画掩膜生成：供修复引擎（CV 填充 / LaMa 神经修复）复用

关键差异（相比旧版固定灰度阈值）：
- 用 OCR 检测多边形（region.poly）限制检索范围
- Otsu 自适应阈值 + 边缘背景色对比，双通道提取笔画
- 剔除气泡边框组件
- 形态学闭运算 + 膨胀，覆盖抗锯齿边缘
结果缓存到 region.mask（dict: bbox + patch），多引擎共享避免重复计算
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.services.pipeline import TextRegion


def build_full_mask(img: np.ndarray, regions: list[TextRegion], pad: int = 2) -> np.ndarray:
    """整图 0/255 笔画掩膜（LaMa 用），缓存 region.mask；已缓存（如 MIT 检测器预填充）的直接复用"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        res = region_patch(img, region, pad=pad)
        if res is None:
            continue
        x0, y0, x1, y1, patch = res
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], patch)
        region.mask = {"bbox": (x0, y0, x1, y1), "patch": patch}
    return mask


def region_patch(img: np.ndarray, region: TextRegion, pad: int = 2) -> Optional[tuple[int, int, int, int, np.ndarray]]:
    """获取缓存/计算某 region 的掩膜 patch；返回 (x0,y0,x1,y1,mask patch 0/255)，无效时 None"""
    cached = region.mask
    if cached and "patch" in cached:
        x0, y0, x1, y1 = cached["bbox"]
        return x0, y0, x1, y1, cached["patch"]  # type: ignore
    return build_region_mask(img, region, pad=pad)


def build_region_mask(img: np.ndarray, region: TextRegion, pad: int = 2) -> Optional[tuple[int, int, int, int, np.ndarray]]:
    """单 region 的笔画掩膜 patch，坐标为 bbox（缩进 pad 像素）

    img: RGB np.ndarray (H,W,3)
    返回 (x0,y0,x1,y1,patch 0/255 uint8) 或 None
    """
    import cv2

    h, w = img.shape[:2]
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # bbox：优先用 poly（OCR 原始多边形），否则用 box
    pts = region.poly if region.poly else region.box
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))

    # 缩进，避免覆盖气泡边框
    x0 = max(0, x0 + pad)
    y0 = max(0, y0 + pad)
    x1 = min(w, x1 - pad)
    y1 = min(h, y1 - pad)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None

    patch_gray = gray[y0:y1, x0:x1]

    # poly 裁剪：仅在多边形内部寻找笔画（避免 pad 区域内混入边框/画面）
    poly_mask = _poly_mask(region, x0, y0, x1, y1, pad)

    cand = _stroke_candidates(patch_gray)
    if cand is None or not cand.any():
        return None

    if poly_mask is not None:
        cand = np.where(poly_mask, cand, 0).astype(np.uint8)

    # 仅 box 兜底（无 poly）时剔除疑似气泡边框的大连通组件
    if poly_mask is None:
        cand = _drop_border_components(cand)
    if not cand.any():
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 膨胀覆盖抗锯齿边缘
    cand = cv2.dilate(cand, kernel, iterations=1)
    return (x0, y0, x1, y1, cand)


def _poly_mask(region: TextRegion, x0: int, y0: int, x1: int, y1: int, pad: int) -> Optional[np.ndarray]:
    """region.poly 相对 bbox 的填充掩膜；无 poly 时返回 None（不裁剪）"""
    if not region.poly or len(region.poly) < 3:
        return None
    try:
        import cv2

        pts = np.array(
            [[int(p[0]) - x0, int(p[1]) - y0] for p in region.poly],
            dtype=np.int32,
        )
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
        return mask.astype(bool)
    except Exception:  # noqa: BLE001
        return None


def _stroke_candidates(patch_gray: np.ndarray) -> Optional[np.ndarray]:
    """笔画候选：Otsu 阈值 + 边缘背景色差异，取并集"""
    import cv2

    h, w = patch_gray.shape
    if h < 4 or w < 4:
        return None

    # 边缘背景灰度（中位数）
    border = np.concatenate(
        [
            patch_gray[:2].reshape(-1),
            patch_gray[-2:].reshape(-1),
            patch_gray[:, :2].reshape(-1),
            patch_gray[:, -2:].reshape(-1),
        ]
    )
    bg = float(np.median(border))

    # Otsu 自适应阈值（暗笔画）
    _, dark = cv2.threshold(
        patch_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # 与背景灰度差异大的像素（含亮笔画，例如深底白字）
    diff = np.abs(patch_gray.astype(np.int16) - bg)
    cand = np.maximum(dark, (diff > 45).astype(np.uint8) * 255)
    return cand


def _drop_border_components(cand: np.ndarray) -> np.ndarray:
    """去除疑似气泡边框的连通组件：同时跨越图幅两组对边的超大组件"""
    import cv2

    try:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    except Exception:  # noqa: BLE001
        return cand
    if num <= 1:
        return cand
    h, w = cand.shape
    total = h * w
    remove = np.zeros(num, dtype=bool)
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        if area < total * 0.25:
            continue
        spans_w = x <= 2 and x + cw >= w - 2
        spans_h = y <= 2 and y + ch >= h - 2
        # 只有像边框（环状）那种同时贴近边缘的超大面积件才剔除
        if spans_w or spans_h:
            remove[i] = True
    if remove.any():
        cand[remove[labels]] = 0
    return cand
