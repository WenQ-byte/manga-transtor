"""Quadrilateral / BBox / sort_pnts，对齐 MIT utils/generic.py（仅保留推理所需）"""
from __future__ import annotations

import functools
from functools import cached_property
from typing import List

import cv2
import numpy as np
from shapely.geometry import MultiPoint, Polygon

from .generic2 import dist, rect_distance


class BBox:
    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        text: str,
        prob: float,
        fg_r: int = 0,
        fg_g: int = 0,
        fg_b: int = 0,
        bg_r: int = 0,
        bg_g: int = 0,
        bg_b: int = 0,
    ):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.prob = prob
        self.fg_r = fg_r
        self.fg_g = fg_g
        self.fg_b = fg_b
        self.bg_r = bg_r
        self.bg_g = bg_g
        self.bg_b = bg_b

    def width(self):
        return self.w

    def height(self):
        return self.h

    def to_points(self):
        tl = np.array([self.x, self.y])
        tr = np.array([self.x + self.w, self.y])
        br = np.array([self.x + self.w, self.y + self.h])
        bl = np.array([self.x, self.y + self.h])
        return tl, tr, br, bl

    @property
    def xywh(self):
        return np.array([self.x, self.y, self.w, self.h], dtype=np.int32)


def sort_pnts(pts: np.ndarray):
    if isinstance(pts, List):
        pts = np.array(pts)
    assert isinstance(pts, np.ndarray) and pts.shape == (4, 2)
    pairwise_vec = (pts[:, None] - pts[None]).reshape((16, -1))
    pairwise_vec_norm = np.linalg.norm(pairwise_vec, axis=1)
    long_side_ids = np.argsort(pairwise_vec_norm)[[8, 10]]
    long_side_vecs = pairwise_vec[long_side_ids]
    inner_prod = (long_side_vecs[0] * long_side_vecs[1]).sum()
    if inner_prod < 0:
        long_side_vecs[0] = -long_side_vecs[0]
    struc_vec = np.abs(long_side_vecs.mean(axis=0))
    is_vertical = struc_vec[0] <= struc_vec[1]

    if is_vertical:
        pts = pts[np.argsort(pts[:, 1])]
        pts = pts[[*np.argsort(pts[:2, 0]), *np.argsort(pts[2:, 0])[::-1] + 2]]
        return pts, is_vertical
    else:
        pts = pts[np.argsort(pts[:, 0])]
        pts_sorted = np.zeros_like(pts)
        pts_sorted[[0, 3]] = sorted(pts[[0, 1]], key=lambda x: x[1])
        pts_sorted[[1, 2]] = sorted(pts[[2, 3]], key=lambda x: x[1])
        return pts_sorted, is_vertical


def distance_point_point(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


def distance_point_lineseg(p: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    x, y = p[0], p[1]
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1
    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:
        param = dot / len_sq
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    return np.sqrt((x - xx) ** 2 + (y - yy) ** 2)


class Quadrilateral:
    def __init__(
        self,
        pts,
        text: str,
        prob: float,
        fg_r: int = 0,
        fg_g: int = 0,
        fg_b: int = 0,
        bg_r: int = 0,
        bg_g: int = 0,
        bg_b: int = 0,
    ):
        self.pts, is_vertical = sort_pnts(np.asarray(pts).astype(float))
        self.direction = "v" if is_vertical else "h"
        self.text = text
        self.prob = prob
        self.fg_r = fg_r
        self.fg_g = fg_g
        self.fg_b = fg_b
        self.bg_r = bg_r
        self.bg_g = bg_g
        self.bg_b = bg_b
        self.assigned_direction = None
        self.textlines: List = []

    @property
    def fg_colors(self):
        return np.array([self.fg_r, self.fg_g, self.fg_b])

    @property
    def bg_colors(self):
        return np.array([self.bg_r, self.bg_g, self.bg_b])

    @cached_property
    def structure(self):
        p1 = ((self.pts[0] + self.pts[1]) / 2).astype(int)
        p2 = ((self.pts[2] + self.pts[3]) / 2).astype(int)
        p3 = ((self.pts[1] + self.pts[2]) / 2).astype(int)
        p4 = ((self.pts[3] + self.pts[0]) / 2).astype(int)
        return [p1, p2, p3, p4]

    @cached_property
    def valid(self) -> bool:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        unit_vector_1 = v1 / np.linalg.norm(v1)
        unit_vector_2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product) * 180 / np.pi
        return abs(angle - 90) < 10

    @cached_property
    def aspect_ratio(self) -> float:
        """hor/ver"""
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        return np.linalg.norm(v2) / np.linalg.norm(v1)

    @cached_property
    def font_size(self) -> float:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        return min(np.linalg.norm(v2), np.linalg.norm(v1))

    def width(self) -> int:
        return self.aabb.w

    def height(self) -> int:
        return self.aabb.h

    @cached_property
    def xyxy(self):
        return self.aabb.x, self.aabb.y, self.aabb.x + self.aabb.w, self.aabb.y + self.aabb.h

    def clip(self, width, height):
        self.pts[:, 0] = np.clip(np.round(self.pts[:, 0]), 0, width)
        self.pts[:, 1] = np.clip(np.round(self.pts[:, 1]), 0, height)

    @cached_property
    def aabb(self) -> BBox:
        max_coord = np.max(self.pts, axis=0)
        min_coord = np.min(self.pts, axis=0)
        return BBox(
            min_coord[0], min_coord[1], max_coord[0] - min_coord[0], max_coord[1] - min_coord[1],
            self.text, self.prob, self.fg_r, self.fg_g, self.fg_b, self.bg_r, self.bg_g, self.bg_b,
        )

    def get_transformed_region(self, img, direction, textheight) -> np.ndarray:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v_vec = l1b - l1a
        h_vec = l2b - l2a
        ratio = np.linalg.norm(v_vec) / np.linalg.norm(h_vec)

        src_pts = self.pts.astype(np.int64).copy()
        im_h, im_w = img.shape[:2]
        x1, y1 = src_pts[:, 0].min(), src_pts[:, 1].min()
        x2, y2 = src_pts[:, 0].max(), src_pts[:, 1].max()
        x1 = np.clip(x1, 0, im_w)
        y1 = np.clip(y1, 0, im_h)
        x2 = np.clip(x2, 0, im_w)
        y2 = np.clip(y2, 0, im_h)
        img_croped = img[y1:y2, x1:x2]
        if img_croped.size == 0:
            return np.zeros((max(int(textheight), 2), max(int(textheight), 2), 3), dtype=np.uint8)

        src_pts[:, 0] -= x1
        src_pts[:, 1] -= y1
        self.assigned_direction = direction
        if direction == "h":
            h = max(int(textheight), 2)
            w = max(int(round(textheight / ratio)), 2)
        else:
            w = max(int(textheight), 2)
            h = max(int(round(textheight * ratio)), 2)
        dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        region = cv2.warpPerspective(img_croped, M, (w, h))
        if direction == "v":
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return region

    @cached_property
    def is_axis_aligned(self) -> bool:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        e1 = np.array([0, 1])
        e2 = np.array([1, 0])
        unit_vector_1 = v1 / np.linalg.norm(v1)
        return abs(np.dot(unit_vector_1, e1)) < 1e-2 or abs(np.dot(unit_vector_1, e2)) < 1e-2

    @cached_property
    def is_approximate_axis_aligned(self) -> bool:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        e1 = np.array([0, 1])
        e2 = np.array([1, 0])
        u1 = v1 / np.linalg.norm(v1)
        u2 = v2 / np.linalg.norm(v2)
        return (
            abs(np.dot(u1, e1)) < 0.05
            or abs(np.dot(u1, e2)) < 0.05
            or abs(np.dot(u2, e1)) < 0.05
            or abs(np.dot(u2, e2)) < 0.05
        )

    @cached_property
    def cosangle(self) -> float:
        [l1a, l1b, _, _] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        e2 = np.array([1, 0])
        unit_vector_1 = v1 / np.linalg.norm(v1)
        return np.dot(unit_vector_1, e2)

    @cached_property
    def angle(self) -> float:
        return np.fmod(np.arccos(self.cosangle) + np.pi, np.pi)

    @cached_property
    def centroid(self) -> np.ndarray:
        return np.average(self.pts, axis=0)

    def distance_to_point(self, p: np.ndarray) -> float:
        d = 1.0e20
        for i in range(4):
            d = min(d, distance_point_point(p, self.pts[i]))
            d = min(d, distance_point_lineseg(p, self.pts[i], self.pts[(i + 1) % 4]))
        return d

    @cached_property
    def polygon(self) -> Polygon:
        return MultiPoint([tuple(self.pts[i]) for i in range(4)]).convex_hull

    @cached_property
    def area(self) -> float:
        return self.polygon.area

    def poly_distance(self, other) -> float:
        return self.polygon.distance(other.polygon)

    def distance(self, other, rho=0.5) -> float:
        return self.distance_impl(other, rho)

    def distance_impl(self, other, rho=0.5) -> float:
        if (
            self.assigned_direction is None
            or other.assigned_direction is None
            or self.assigned_direction != other.assigned_direction
        ):
            return self.poly_distance(other)
        pattern = "h_left" if self.assigned_direction == "h" else "v_top"
        fs = max(self.font_size, other.font_size)
        if self.assigned_direction == "h":
            poly1 = MultiPoint([self.pts[0], self.pts[3], other.pts[0], other.pts[3]]).convex_hull
            poly2 = MultiPoint([self.pts[2], self.pts[1], other.pts[2], other.pts[1]]).convex_hull
            poly3 = MultiPoint(
                [self.structure[0], self.structure[1], other.structure[0], other.structure[1]]
            ).convex_hull
            dist1 = poly1.area / fs
            dist2 = poly2.area / fs
            dist3 = poly3.area / fs
            if dist1 < fs * rho:
                pattern = "h_left"
            if dist2 < fs * rho and dist2 < dist1:
                pattern = "h_right"
            if dist3 < fs * rho and dist3 < dist1 and dist3 < dist2:
                pattern = "h_middle"
            if pattern == "h_left":
                return dist(self.pts[0][0], self.pts[0][1], other.pts[0][0], other.pts[0][1])
            elif pattern == "h_right":
                return dist(self.pts[1][0], self.pts[1][1], other.pts[1][0], other.pts[1][1])
            else:
                return dist(self.structure[0][0], self.structure[0][1], other.structure[0][0], other.structure[0][1])
        else:
            poly1 = MultiPoint([self.pts[0], self.pts[1], other.pts[0], other.pts[1]]).convex_hull
            poly2 = MultiPoint([self.pts[2], self.pts[3], other.pts[2], other.pts[3]]).convex_hull
            dist1 = poly1.area / fs
            dist2 = poly2.area / fs
            if dist1 < fs * rho:
                pattern = "v_top"
            if dist2 < fs * rho and dist2 < dist1:
                pattern = "v_bottom"
            if pattern == "v_top":
                return dist(self.pts[0][0], self.pts[0][1], other.pts[0][0], other.pts[0][1])
            else:
                return dist(self.pts[2][0], self.pts[2][1], other.pts[2][0], other.pts[2][1])

    def copy(self, new_pts: np.ndarray):
        return Quadrilateral(new_pts, self.text, self.prob, *self.fg_colors, *self.bg_colors)


def quadrilateral_can_merge_region(
    a: Quadrilateral,
    b: Quadrilateral,
    ratio=1.9,
    discard_connection_gap=2,
    char_gap_tolerance=0.6,
    char_gap_tolerance2=1.5,
    font_size_ratio_tol=1.5,
    aspect_ratio_tol=2,
) -> bool:
    b1 = a.aabb
    b2 = b.aabb
    char_size = min(a.font_size, b.font_size)
    x1, y1, w1, h1 = b1.x, b1.y, b1.w, b1.h
    x2, y2, w2, h2 = b2.x, b2.y, b2.w, b2.h
    p1 = Polygon(a.pts)
    p2 = Polygon(b.pts)
    dist = p1.distance(p2)
    if dist > discard_connection_gap * char_size:
        return False
    if max(a.font_size, b.font_size) / char_size > font_size_ratio_tol:
        return False
    if a.aspect_ratio > aspect_ratio_tol and b.aspect_ratio < 1.0 / aspect_ratio_tol:
        return False
    if b.aspect_ratio > aspect_ratio_tol and a.aspect_ratio < 1.0 / aspect_ratio_tol:
        return False
    a_aa = a.is_approximate_axis_aligned
    b_aa = b.is_approximate_axis_aligned
    if a_aa and b_aa:
        if dist < char_size * char_gap_tolerance:
            if abs(x1 + w1 // 2 - (x2 + w2 // 2)) < char_gap_tolerance2:
                return True
            if w1 > h1 * ratio and h2 > w2 * ratio:
                return False
            if w2 > h2 * ratio and h1 > w1 * ratio:
                return False
            if w1 > h1 * ratio or w2 > h2 * ratio:  # h
                return abs(x1 - x2) < char_size * char_gap_tolerance2 or abs(
                    x1 + w1 - (x2 + w2)
                ) < char_size * char_gap_tolerance2
            if h1 > w1 * ratio or h2 > w2 * ratio:  # v
                return abs(y1 - y2) < char_size * char_gap_tolerance2 or abs(
                    y1 + h1 - (y2 + h2)
                ) < char_size * char_gap_tolerance2
            return False
        return False
    if abs(a.angle - b.angle) < 15 * np.pi / 180:
        fs_a = a.font_size
        fs_b = b.font_size
        fs = min(fs_a, fs_b)
        if a.poly_distance(b) > fs * char_gap_tolerance2:
            return False
        if abs(fs_a - fs_b) / fs > 0.25:
            return False
        return True
    return False


def quadrilateral_can_merge_region_coarse(
    a: Quadrilateral, b: Quadrilateral, discard_connection_gap=2, font_size_ratio_tol=0.7
) -> bool:
    if a.assigned_direction != b.assigned_direction:
        return False
    if abs(a.angle - b.angle) > 15 * np.pi / 180:
        return False
    fs_a = a.font_size
    fs_b = b.font_size
    if abs(fs_a - fs_b) / min(fs_a, fs_b) > font_size_ratio_tol:
        return False
    dist = a.poly_distance(b)
    if dist > discard_connection_gap * max(fs_a, fs_b):
        return False
    return True