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


def group_regions_by_bubble(
    bgr,
    regions: list[TextRegion],
    img_w,
    img_h,
    overlap=0.15,
    boundary_bgr=None,
) -> list[dict]:
    """把同气泡的 region 分到一组，组内按阅读顺序排序（横排 y→x，竖排 右→左）

    返回 [{bbox, regions}]；同时回填 region.group_bounds（组包围盒）。
    """
    boundary_gray = None
    if boundary_bgr is not None and cv2 is not None:
        boundary_gray = (
            cv2.cvtColor(boundary_bgr, cv2.COLOR_BGR2GRAY)
            if boundary_bgr.ndim == 3
            else boundary_bgr
        )
    groups: list[dict] = []
    for region in regions:
        bb, container_mask = bubble_with_mask(bgr, region.bounds, img_w, img_h)
        mask_reliable = container_mask is not None and bool(container_mask.any())
        best_idx, best_ov = -1, 0.0
        for i, g in enumerate(groups):
            score = _container_merge_score(
                bb,
                container_mask,
                mask_reliable,
                region,
                g,
                overlap,
                boundary_gray,
            )
            if score > best_ov:
                best_ov, best_idx = score, i
        if best_idx >= 0:
            group = groups[best_idx]
            group["bbox"] = _union(bb, group["bbox"])
            group["regions"].append(region)
            group["members"].append((bb, container_mask, mask_reliable, region))
            if mask_reliable:
                if group["mask"] is None:
                    group["mask"] = container_mask
                    group["mask_reliable"] = True
                else:
                    group["mask"] = np.maximum(group["mask"], container_mask)
        else:
            groups.append(
                {
                    "bbox": bb,
                    "regions": [region],
                    "mask": container_mask if mask_reliable else None,
                    "mask_reliable": mask_reliable,
                    "members": [(bb, container_mask, mask_reliable, region)],
                }
            )

    groups = _merge_overlap_groups(groups, overlap, boundary_gray=boundary_gray)

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
            r.group_mask = g.get("mask")
            r.group_mask_reliable = bool(g.get("mask_reliable"))
        out.append(
            {
                "bbox": bbox,
                "regions": regions_sorted,
                "mask": g.get("mask"),
                "mask_reliable": bool(g.get("mask_reliable")),
            }
        )
    return out


def _mask_overlap_ratio(a, b) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    aa = a > 0
    bb = b > 0
    denom = min(int(aa.sum()), int(bb.sum()))
    if denom <= 0:
        return 0.0
    return float((aa & bb).sum()) / float(denom)


def _container_merge_score(
    bb,
    mask,
    mask_reliable,
    region,
    group,
    overlap,
    boundary_gray=None,
) -> float:
    """容器合并评分：可靠掩膜是硬边界，回退框只允许紧凑近邻合并。"""
    group_mask = group.get("mask")
    group_mask_reliable = bool(group.get("mask_reliable")) and group_mask is not None
    if boundary_gray is not None and _groups_separated_by_boundary(
        [region], group.get("regions", []), boundary_gray
    ):
        return 0.0
    if mask_reliable and group_mask_reliable:
        mask_ov = _mask_overlap_ratio(mask, group_mask)
        if mask_ov < 0.55:
            return 0.0
        return 2.0 + mask_ov

    gb = group["bbox"]
    box_ov = _overlap_ratio(bb, gb)
    if not _balloon_ok(bb, gb):
        return 0.0
    if not _compact_group_ok(group["members"], (bb, mask, mask_reliable, region)):
        return 0.0
    if not _region_adjacent_to_group(region, group["regions"]):
        return 0.0
    if box_ov >= max(0.45, overlap * 2.5):
        return 1.0 + box_ov
    if _contains(bb, gb) or _contains(gb, bb):
        return 1.0
    if _directions_compatible(region, group["regions"]) and _proximity_overlap(bb, gb):
        return 0.5
    return 0.0


def _directions_compatible(region, group_regions) -> bool:
    direction = getattr(region, "direction", None)
    known = [getattr(r, "direction", None) for r in group_regions]
    known = [d for d in known if d]
    return not direction or not known or direction == max(set(known), key=known.count)


def _axis_overlap(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    return inter / max(1, min(a1 - a0, b1 - b0))


def _axis_gap(a0, a1, b0, b1) -> float:
    return max(0, max(a0, b0) - min(a1, b1))


def _text_regions_adjacent(a, b) -> bool:
    """按原始文字框判断是否为同一排/列，避免外扩框在相邻卡片间搭桥。"""
    if not hasattr(a, "bounds") or not hasattr(b, "bounds"):
        return True
    ax0, ay0, ax1, ay1 = a.bounds
    bx0, by0, bx1, by1 = b.bounds
    aw, ah = max(1, ax1 - ax0), max(1, ay1 - ay0)
    bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)
    direction = getattr(a, "direction", None) or getattr(b, "direction", None)
    if direction == "v":
        same_rows = _axis_overlap(ay0, ay1, by0, by1) >= 0.45
        nearby_columns = _axis_gap(ax0, ax1, bx0, bx1) <= 1.5 * max(aw, bw)
        same_column = _axis_overlap(ax0, ax1, bx0, bx1) >= 0.5
        nearby_parts = _axis_gap(ay0, ay1, by0, by1) <= 0.25 * max(ah, bh)
        return (same_rows and nearby_columns) or (same_column and nearby_parts)
    same_columns = _axis_overlap(ax0, ax1, bx0, bx1) >= 0.45
    nearby_rows = _axis_gap(ay0, ay1, by0, by1) <= 1.5 * max(ah, bh)
    same_row = _axis_overlap(ay0, ay1, by0, by1) >= 0.5
    nearby_parts = _axis_gap(ax0, ax1, bx0, bx1) <= 0.25 * max(aw, bw)
    return (same_columns and nearby_rows) or (same_row and nearby_parts)


def _region_adjacent_to_group(region, group_regions) -> bool:
    return _directions_compatible(region, group_regions) and any(
        _text_regions_adjacent(region, other) for other in group_regions
    )


def _groups_text_compatible(a_regions, b_regions) -> bool:
    real_a = [r for r in a_regions if hasattr(r, "bounds")]
    real_b = [r for r in b_regions if hasattr(r, "bounds")]
    if not real_a or not real_b:
        return True
    return any(
        _directions_compatible(a, real_b) and _text_regions_adjacent(a, b)
        for a in real_a
        for b in real_b
    )


def _groups_separated_by_boundary(a_regions, b_regions, gray) -> bool:
    """原图中若每一对可比较文字框之间都有长轮廓，则它们属于不同气泡。"""
    comparable = []
    for a in a_regions:
        for b in b_regions:
            separated = _region_boundary_separator(a, b, gray)
            if separated is not None:
                comparable.append(separated)
    return bool(comparable) and all(comparable)


def _region_boundary_separator(a, b, gray):
    """检查两个文字框间隙中的长黑线；无法形成可靠轴向走廊时返回 None。"""
    if not hasattr(a, "bounds") or not hasattr(b, "bounds"):
        return None
    ax0, ay0, ax1, ay1 = a.bounds
    bx0, by0, bx1, by1 = b.bounds
    ah, bh = max(1, ay1 - ay0), max(1, by1 - by0)
    aw, bw = max(1, ax1 - ax0), max(1, bx1 - bx0)

    if ax1 <= bx0 or bx1 <= ax0:
        left, right = ((ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1))
        if bx1 <= ax0:
            left, right = right, left
        gx0, gx1 = left[2], right[0]
        gy0, gy1 = max(left[1], right[1]), min(left[3], right[3])
        overlap_h = gy1 - gy0
        if overlap_h < max(8, int(0.25 * min(ah, bh))):
            return None
        if gx1 - gx0 < 3:
            return False
        return _long_barrier_in_gap(gray, gx0, gy0, gx1, gy1, vertical=True)

    if ay1 <= by0 or by1 <= ay0:
        upper, lower = ((ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1))
        if by1 <= ay0:
            upper, lower = lower, upper
        gy0, gy1 = upper[3], lower[1]
        gx0, gx1 = max(upper[0], lower[0]), min(upper[2], lower[2])
        overlap_w = gx1 - gx0
        if overlap_w < max(8, int(0.25 * min(aw, bw))):
            return None
        if gy1 - gy0 < 3:
            return False
        return _long_barrier_in_gap(gray, gx0, gy0, gx1, gy1, vertical=False)
    return False if _text_regions_adjacent(a, b) else None


def _long_barrier_in_gap(gray, x0, y0, x1, y1, vertical: bool) -> bool:
    if cv2 is None:
        return False
    h, w = gray.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    patch = gray[y0:y1, x0:x1]
    dark = (patch < 150).astype(np.uint8) * 255
    kernel = np.ones((5, 3) if vertical else (3, 5), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    required = (patch.shape[0] if vertical else patch.shape[1]) * 0.45
    for label in range(1, count):
        _, _, cw, ch, area = stats[label]
        span = ch if vertical else cw
        thickness = cw if vertical else ch
        if span >= max(6, required) and area >= span and thickness <= max(12, span * 0.45):
            return True
    return False


def _compact_group_ok(members, candidate) -> bool:
    """拒绝由窄小区域桥接出的稀疏大组。"""
    boxes = [m[0] for m in members] + [candidate[0]]
    union = boxes[0]
    for box in boxes[1:]:
        union = _union(union, box)
    union_area = max(1, (union[2] - union[0]) * (union[3] - union[1]))
    member_area = sum(max(1, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes)
    return union_area <= 2.2 * member_area


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


def _merge_overlap_groups(groups, overlap=0.15, boundary_gray=None):
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(groups):
            j = i + 1
            while j < len(groups):
                a, b = groups[i]["bbox"], groups[j]["bbox"]
                ma, mb = groups[i].get("mask"), groups[j].get("mask")
                ra = bool(groups[i].get("mask_reliable")) and ma is not None
                rb = bool(groups[j].get("mask_reliable")) and mb is not None
                same_mask = ra and rb and _mask_overlap_ratio(ma, mb) >= 0.55
                separated = boundary_gray is not None and _groups_separated_by_boundary(
                    groups[i].get("regions", []), groups[j].get("regions", []), boundary_gray
                )
                fallback_ok = (
                    not (ra and rb)
                    and _balloon_ok(a, b)
                    and _groups_text_compatible(groups[i]["regions"], groups[j]["regions"])
                    and _compact_group_ok(
                        groups[i].get("members") or [(a, ma, ra, None)],
                        (b, mb, rb, None),
                    )
                    and (
                        _overlap_ratio(a, b) > max(0.45, overlap * 2.5)
                        or _contains(a, b)
                        or _contains(b, a)
                        or _proximity_overlap(a, b)
                    )
                )
                if not separated and (same_mask or fallback_ok):
                    groups[i]["bbox"] = _union(a, b)
                    groups[i]["regions"].extend(groups[j]["regions"])
                    groups[i].setdefault("members", []).extend(groups[j].get("members", []))
                    if mb is not None:
                        groups[i]["mask"] = mb if ma is None else np.maximum(ma, mb)
                        groups[i]["mask_reliable"] = ra or rb
                    groups.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    return groups
