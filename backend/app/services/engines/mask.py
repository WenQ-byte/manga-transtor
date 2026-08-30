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
    """整图 0/255 文本掩膜（LaMa/CV 修复用），缓存结果到 region.mask

    默认整块擦除：优先文本多边形填充（poly），无 poly 才用 Otsu 笔画；结果覆盖
    MIT 检测器预填充的紧致笔画掩膜（后者偏紧会残留）。显式标记 _no_erase 的 region
    （mit_ignore_bubble 判定为拟声词等非气泡）跳过不擦除。
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        if getattr(region, "_no_erase", False):
            continue
        # MIT 检测器已经提供逐像素文本掩膜，优先复用它；只有没有缓存时才
        # 依据 OCR 多边形/笔画候选重新计算，避免重算掩膜漏掉风格化字或注音。
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
    """单 region 的掩膜 patch，坐标为 bbox（缩进 pad 像素）

    img: RGB np.ndarray (H,W,3)
    返回 (x0,y0,x1,y1,patch 0/255 uint8) 或 None

    策略：有 poly 时整块填充文本多边形并膨胀（文本区域整体擦除，零残留，
    背景由修复引擎重建）；无 poly 时用 Otsu 笔画候选（保守，仅亮/暗笔画）。
    """
    import cv2

    h, w = img.shape[:2]

    # bbox：以 box 与 poly 的并集为界（poly 常偏紧包不住笔画外缘，box 更完整）
    pts = region.poly if region.poly else region.box
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    if region.box:
        bxs = [p[0] for p in region.box]
        bys = [p[1] for p in region.box]
        x0 = min(x0, int(min(bxs)))
        x1 = max(x1, int(max(bxs)))
        y0 = min(y0, int(min(bys)))
        y1 = max(y1, int(max(bys)))

    # 缩进，避免覆盖气泡边框
    x0 = max(0, x0 + pad)
    y0 = max(0, y0 + pad)
    x1 = min(w, x1 - pad)
    y1 = min(h, y1 - pad)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None

    poly_filled = _fill_poly(region, x0, y0, x1, y1)
    if poly_filled is not None:
        # 整块擦除：文本多边形内部全部是掩膜；再并上 Otsu 笔画候选（覆盖 poly 外缘漏出的笔画）
        # 并放大膨胀补防抗锯齿边缘，杜绝原文灰影残留。
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        patch_gray = img_gray[y0:y1, x0:x1]
        strokes = _stroke_candidates(patch_gray)
        cand = poly_filled
        if strokes is not None and strokes.any():
            cand = np.maximum(cand, strokes)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cand = cv2.dilate(cand, kernel, iterations=1)
        # 把 pad 缩进丢掉的边带补回掩膜（覆盖到 box/poly 边界），防边缘笔画残留
        if pad > 0:
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(w, x1 + pad)
            y1 = min(h, y1 + pad)
            padded = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            padded[pad:pad + cand.shape[0], pad:pad + cand.shape[1]] = cand
            cand = padded
        # 注音扩展：竖排汉字旁常印小号假名（furigana），检测器往往漏检，若不擦除会原文灰影残留。
        # 沿主字 bbox 上下/左右各扩 FURIGANA_MARGIN，并仅拾取带内的小连通字形成分（避免误伤大结构）。
        x0, y0, x1, y1, cand = _add_furigana_margin(img, x0, y0, x1, y1, cand)
        return (x0, y0, x1, y1, cand)

    # 无 poly 兜底：Otsu 笔画候选（仅 box），并剔除疑似气泡边框的大连通组件
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    patch_gray = gray[y0:y1, x0:x1]
    cand = _stroke_candidates(patch_gray)
    if cand is None or not cand.any():
        return None
    cand = _drop_border_components(cand)
    if not cand.any():
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel, iterations=1)
    cand = cv2.dilate(cand, kernel, iterations=1)
    return (x0, y0, x1, y1, cand)


def _fill_poly(region: TextRegion, x0: int, y0: int, x1: int, y1: int) -> Optional[np.ndarray]:
    """region.poly 相对 bbox 的整块填充掩膜；无 poly 或无交集时返回 None"""
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
        if not mask.any():
            return None
        return mask
    except Exception:  # noqa: BLE001
        return None


# 注音（furigana）扩展带：主字 bbox 外扩该距离，拾取带内小字形成分
FURIGANA_MARGIN = 16
# 注音字形尺寸上限（px）：高 ≤ 该值、宽 ≤ 该值且面积 ≤ 该上限
_FURIGANA_MAX_H = 16
_FURIGANA_MAX_W = 18
_FURIGANA_MAX_AREA = 260


def _add_furigana_margin(img, x0, y0, x1, y1, cand):
    """沿主字 bbox 外扩 FURIGANA_MARGIN，把带内小连通字形成分（注音）并入掩膜。

    只拾取尺寸/面积达标的紧凑成分，不误伤旁侧大结构（面板/网点/人物）。
    返回扩大的 (x0,y0,x1,y1,cand)；无注音时保持原 bbox。
    """
    import cv2

    h, w = img.shape[:2]
    nx0 = max(0, x0 - FURIGANA_MARGIN)
    ny0 = max(0, y0 - FURIGANA_MARGIN)
    nx1 = min(w, x1 + FURIGANA_MARGIN)
    ny1 = min(h, y1 + FURIGANA_MARGIN)
    if nx0 >= x0 and ny0 >= y0 and nx1 <= x1 and ny1 <= y1:
        return x0, y0, x1, y1, cand
    expand_w = nx1 - nx0
    expand_h = ny1 - ny0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    band = (gray[ny0:ny1, nx0:nx1] < 140).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(band, 8)
    new_cand = np.zeros((expand_h, expand_w), dtype=np.uint8)
    # 主字已有的掩膜拷入新 bbox 左上(相对偏移)
    ox, oy = x0 - nx0, y0 - ny0
    new_cand[oy:oy + cand.shape[0], ox:ox + cand.shape[1]] = cand
    for i in range(1, num):
        bx, by, bw, bh, area = stats[i]
        if (
            FURIGANA_MARGIN - 1 <= bh <= _FURIGANA_MAX_H
            and 1 <= bw <= _FURIGANA_MAX_W
            and area <= _FURIGANA_MAX_AREA
        ):
            new_cand[by:by + bh, bx:bx + bw] = 255
    return nx0, ny0, nx1, ny1, new_cand


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
