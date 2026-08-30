"""气泡过滤：只保留位于气泡内的文字区域，丢弃气泡外的涂鸦/噪声

原理：漫画气泡内部通常是纯色区域（白/浅色），被黑色描边包围。
从文本框外侧采样周围底色并做泛洪填充：
- 文字在气泡内 → 填充被描边限制在气泡内，面积适中且覆盖文本框 → 保留
- 涂鸦在页面上（无气泡）→ 填充泄漏到整页背景，面积巨大 → 丢弃
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.services.pipeline import TextRegion

# 泛洪填充颜色容差
FLOOD_TOL = 40
# 填充面积超过整图该比例视为泄漏（非气泡）
MAX_FILL_RATIO = 0.5
# 气泡相对文本框的最大放大倍数（防止无边框气泡泄漏到整张图）
BUBBLE_GROW_RATIO = 6.0

try:
    import cv2
except ImportError:  # noqa: BLE001
    cv2 = None


class BubbleFilter:
    """基于背景色泛洪填充的气泡内外判别"""

    name = "bubble"

    def filter(self, image_path: Path, regions: list[TextRegion]) -> list[TextRegion]:
        if not regions or cv2 is None:
            return regions
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:  # noqa: BLE001
            return regions
        bgr = np.array(img)[:, :, ::-1].copy()
        img_h, img_w = bgr.shape[:2]

        kept = []
        for region in regions:
            if self._in_bubble(bgr, region.bounds, img_w, img_h):
                kept.append(region)

        # 至少保留一个气泡内文字才启用过滤，避免整页文字被误删
        return kept if kept else regions

    def _in_bubble(self, bgr, bounds, img_w, img_h) -> bool:
        """检查文字区域是否在气泡内

        宽松策略：文字框内有足够白色背景 → 视为气泡内（保留）。
        严格过滤交给 OCR 置信度（pipeline 里 confidence >= 0.5）。
        """
        x0, y0, x1, y1 = [int(v) for v in bounds]
        h, w = bgr.shape[:2]
        if x1 <= x0 or y1 <= y0:
            return True

        # 文字框内白色像素占比 > 30% → 有气泡背景 → 保留
        patch = bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        white_ratio = float(np.mean(gray > 200))
        return white_ratio > 0.30


def create_bubble_filter() -> BubbleFilter:
    return BubbleFilter()


def detect_bubble(bgr, bounds, img_w, img_h, flood_tol=FLOOD_TOL, grow_ratio=BUBBLE_GROW_RATIO):
    """通过泛洪填充找到文本框所属气泡的真实边界（与 renderer 同算法）"""
    bb, _ = bubble_with_mask(bgr, bounds, img_w, img_h, flood_tol=flood_tol, grow_ratio=grow_ratio)
    return bb


# 非气泡文字（刊头/拟声词/贴纸字）几何判据：泛洪 bbox 为细长横条或跨页宽横带
STRIP_RATIO = 8.0
STRIP_H_RATIO = 0.04
BAND_W_RATIO = 0.5
BAND_H_RATIO = 0.05


def _is_strip_bbox(bb, img_w, img_h, direction) -> bool:
    """泛洪 bbox 是否为非气泡形态：细长横条（w/h≥8 且高≤4%页高）或跨页宽横带"""
    bw = bb[2] - bb[0]
    bh = bb[3] - bb[1]
    if bw <= 0 or bh <= 0:
        return False
    if bw / bh >= STRIP_RATIO and bh <= STRIP_H_RATIO * img_h:
        return True
    if direction == "h" and bw >= BAND_W_RATIO * img_w and bh <= BAND_H_RATIO * img_h:
        return True
    return False


def classify_non_bubble(bgr, region, img_w, img_h) -> bool:
    """判定 region 是否为非气泡文字（刊头/拟声词等）：泛洪 bbox 命中细横条/跨页横带"""
    bb = detect_bubble(bgr, region.bounds, img_w, img_h)
    return _is_strip_bbox(bb, img_w, img_h, getattr(region, "direction", None))


def bubble_with_mask(bgr, bounds, img_w, img_h, flood_tol=FLOOD_TOL, grow_ratio=BUBBLE_GROW_RATIO):
    """泛洪推气泡：返回 (bbox, mask)

    mask 为气泡形状的整图 0/255 uint8（可作绘制裁剪）；漏尾/无边框时退回矩形框并返回 mask=None。
    """
    x0, y0, x1, y1 = [int(v) for v in bounds]
    x0 = max(0, min(x0, img_w - 1))
    x1 = max(0, min(x1, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))
    y1 = max(0, min(y1, img_h - 1))
    if x1 <= x0 or y1 <= y0:
        return _fallback_box(x0, y0, x1, y1, img_w, img_h), None

    max_bw = max((x1 - x0) * grow_ratio, img_w * 0.85)
    max_bh = max((y1 - y0) * grow_ratio, img_h * 0.85)

    if cv2 is not None:
        h, w = bgr.shape[:2]
        flags = 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        seeds = _bright_seeds(bgr, x0, y0, x1, y1, w, h)
        best = None
        best_mask = None
        best_area = -1
        for seed in seeds:
            sx, sy = seed
            if not (0 <= sx < w and 0 <= sy < h):
                continue
            try:
                mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(bgr, mask, seed, 0, (flood_tol, flood_tol, flood_tol), (flood_tol, flood_tol, flood_tol), flags)
            except Exception:  # noqa: BLE001
                continue
            filled = mask[1:-1, 1:-1]
            ys, xs = np.where(filled > 0)
            if xs.size == 0:
                continue
            bx0 = int(xs.min())
            by0 = int(ys.min())
            bx1 = int(xs.max()) + 1
            by1 = int(ys.max()) + 1
            if (bx1 - bx0) > max_bw or (by1 - by0) > max_bh:
                continue
            area = (bx1 - bx0) * (by1 - by0)
            if area > best_area:
                best_area = area
                best = (bx0, by0, bx1, by1)
                # 注意 uint8 溢出：255*255 会回绕成 1，必须先转 bool 再放大到 0/255
                best_mask = (filled > 0).astype(np.uint8) * 255
        if best is not None:
            area = (best[2] - best[0]) * (best[3] - best[1])
            # 填充过小（没填到文本区）或过大（泄漏到面板间隙/相邻气泡）都不可信
            if area < (x1 - x0) * (y1 - y0) * 0.5:
                return _fallback_box(x0, y0, x1, y1, img_w, img_h), None
            if best_mask is not None and int((best_mask > 0).sum()) > (x1 - x0) * (y1 - y0) * 8:
                return _fallback_box(x0, y0, x1, y1, img_w, img_h), None
            return best, best_mask

    return _fallback_box(x0, y0, x1, y1, img_w, img_h), None


def _fallback_box(x0, y0, x1, y1, img_w, img_h, pad_xr=0.35, pad_yr=0.35):
    pad_x = int((x1 - x0) * pad_xr)
    pad_y = int((y1 - y0) * pad_yr)
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(img_w, x1 + pad_x),
        min(img_h, y1 + pad_y),
    )


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _overlap_ratio(a, b) -> float:
    """交集面积 / 较小框面积"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / max(1, min(area_a, area_b))


def group_regions_by_bubble(bgr, regions: list[TextRegion], img_w, img_h, overlap=0.15) -> list[dict]:
    """把同气泡的 region 分到一组，组内按阅读顺序排序（横排 y→x，竖排 右→左）

    返回 [{bbox, regions}]；同时回填 region.group_bounds（组包围盒）。
    """
    groups: list[dict] = []
    for region in regions:
        bb = detect_bubble(bgr, region.bounds, img_w, img_h)
        best_idx, best_ov = -1, 0.0
        for i, g in enumerate(groups):
            ov = _overlap_ratio(bb, g["bbox"])
            if ov > best_ov:
                best_ov, best_idx = ov, i
        if best_idx >= 0 and best_ov > overlap and _balloon_ok(bb, groups[best_idx]["bbox"]):
            groups[best_idx]["bbox"] = _union(bb, groups[best_idx]["bbox"])
            groups[best_idx]["regions"].append(region)
        else:
            groups.append({"bbox": bb, "regions": [region]})

    groups = _merge_overlap_groups(groups, overlap)

    out = []
    for g in groups:
        regions = g["regions"]
        dirs = [r.direction for r in regions if r.direction]
        vertical = (dirs.count("v") > dirs.count("h")) if dirs else (g["bbox"][3] - g["bbox"][1] > (g["bbox"][2] - g["bbox"][0]))
        if vertical:
            regions_sorted = sorted(
                regions,
                key=lambda r: (-((r.bounds[0] + r.bounds[2]) / 2), r.bounds[1]),
            )
        else:
            regions_sorted = sorted(
                regions,
                key=lambda r: (r.bounds[1], r.bounds[0]),
            )
        bbox = g["bbox"]
        for i, r in enumerate(regions_sorted):
            r.group_index = len(out)
            r.group_bounds = bbox
        out.append({"bbox": bbox, "regions": regions_sorted})
    return out


_PUNCT = set("。、，,！!？?…〜~ー―─-「」『』（）()♪☆★・･ ")


def merge_punctuation_regions(regions: list[TextRegion]) -> list[TextRegion]:
    """把 1～2 字纯标点行并入最近邻文本行"""
    if len(regions) < 2:
        return regions

    def is_frag(r: TextRegion) -> bool:
        t = (r.text or "").strip()
        return bool(t) and len(t) <= 2 and all(c in _PUNCT for c in t)

    def center(r: TextRegion):
        a, b, c, d = r.bounds
        return ((a + c) / 2, (b + d) / 2)

    frags = [r for r in regions if is_frag(r)]
    mains = [r for r in regions if not is_frag(r)]
    if not frags or not mains:
        return regions

    dropped = set()
    for frag in frags:
        fx, fy = center(frag)
        best, best_d = None, 1e18
        for m in mains:
            mx, my = center(m)
            dist = (fx - mx) ** 2 + (fy - my) ** 2
            if dist < best_d:
                best_d, best = dist, m
        if best is None:
            continue
        ft = (frag.text or "").strip()
        vertical = (best.direction or frag.direction) == "v"
        bx, by = center(best)
        if (fy >= by) if vertical else (fx >= bx):
            best.text = (best.text or "") + ft
        else:
            best.text = ft + (best.text or "")
        x0, y0, x1, y1 = best.bounds
        fx0, fy0, fx1, fy1 = frag.bounds
        nx0, ny0 = min(x0, fx0), min(y0, fy0)
        nx1, ny1 = max(x1, fx1), max(y1, fy1)
        best.box = [[nx0, ny0], [nx1, ny0], [nx1, ny1], [nx0, ny1]]
        if best.poly or frag.poly:
            best.poly = list(best.poly or best.box) + list(frag.poly or frag.box)
        dropped.add(id(frag))
    return [r for r in regions if id(r) not in dropped]


def _bright_seeds(bgr, x0, y0, x1, y1, w, h):
    """文本框附近的亮像素（气泡底）作泛洪种子，避开黑字中心"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    raw = [
        ((x0 + x1) // 2, (y0 + y1) // 2),
        (x0 + 2, y0 + 2),
        (x1 - 2, y0 + 2),
        (x0 + 2, y1 - 2),
        (x1 - 2, y1 - 2),
    ]
    seeds = []
    seen = set()
    for sx, sy in raw:
        px, py = _nearest_bright(gray, sx, sy, w, h)
        key = (px, py)
        if key not in seen:
            seen.add(key)
            seeds.append((px, py))
    return seeds


def _nearest_bright(gray, sx, sy, w, h, radius=12, thresh=200):
    sx = int(np.clip(sx, 0, w - 1))
    sy = int(np.clip(sy, 0, h - 1))
    if int(gray[sy, sx]) > thresh:
        return sx, sy
    y0, y1 = max(0, sy - radius), min(h, sy + radius + 1)
    x0, x1 = max(0, sx - radius), min(w, sx + radius + 1)
    patch = gray[y0:y1, x0:x1]
    ys, xs = np.where(patch > thresh)
    if xs.size == 0:
        return sx, sy
    d = (xs + x0 - sx) ** 2 + (ys + y0 - sy) ** 2
    i = int(np.argmin(d))
    return int(xs[i] + x0), int(ys[i] + y0)


def _contains(a, b) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _proximity_overlap(a, b) -> bool:
    """两框各自按短边 20% 外扩后是否重叠/包含（同气泡列间、行间的近邻兜底）"""
    def expand(box):
        x0, y0, x1, y1 = box
        pad = 0.2 * max(1, min(x1 - x0, y1 - y0))
        return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    ea, eb = expand(a), expand(b)
    return _overlap_ratio(ea, eb) > 0.15 or _contains(ea, eb) or _contains(eb, ea)


# 并集膨胀上限：合并后包围盒面积超过两框面积和该倍数说明是"桥接"（如页眉横条吞并远处气泡）
MERGE_BALLOON_RATIO = 1.6


def _balloon_ok(a, b) -> bool:
    """合并防链式吞并：合法同气泡合并的并集 ≈ 两框之和，桥接合并的并集含大片空白会暴涨"""
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    u = _union(a, b)
    ua = max(1, (u[2] - u[0]) * (u[3] - u[1]))
    return ua <= MERGE_BALLOON_RATIO * (aa + ab)


def _merge_overlap_groups(groups, overlap=0.15):
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(groups):
            j = i + 1
            while j < len(groups):
                a, b = groups[i]["bbox"], groups[j]["bbox"]
                if _balloon_ok(a, b) and (
                    _overlap_ratio(a, b) > overlap
                    or _contains(a, b)
                    or _contains(b, a)
                    or _proximity_overlap(a, b)
                ):
                    groups[i]["bbox"] = _union(a, b)
                    groups[i]["regions"].extend(groups[j]["regions"])
                    groups.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    return groups
