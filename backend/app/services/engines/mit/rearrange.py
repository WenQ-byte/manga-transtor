"""超长图重排（det_rearrange_forward + square_pad_resize），对齐 MIT utils/generic.py"""
from __future__ import annotations

from typing import Callable

import cv2
import einops
import numpy as np


def square_pad_resize(img: np.ndarray, tgt_size: int):
    h, w = img.shape[:2]
    pad_h, pad_w = 0, 0
    if w < h:
        pad_w = h - w
        w += pad_w
    elif h < w:
        pad_h = w - h
        h += pad_h
    pad_size = tgt_size - h
    if pad_size > 0:
        pad_h += pad_size
        pad_w += pad_size
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)
    down_scale_ratio = tgt_size / img.shape[0]
    assert down_scale_ratio <= 1
    if down_scale_ratio < 1:
        img = cv2.resize(img, (tgt_size, tgt_size), interpolation=cv2.INTER_LINEAR)
    return img, down_scale_ratio, pad_h, pad_w


def det_rearrange_forward(
    img: np.ndarray,
    dbnet_batch_forward: Callable[[np.ndarray, str], tuple[np.ndarray, np.ndarray]],
    tgt_size: int = 1280,
    max_batch_size: int = 4,
    device="cpu",
    verbose=False,
):
    """条件满足时把超长图重排为方形批次送入网络；否则返回 (None, None)"""
    h, w = img.shape[:2]
    transpose = False
    if h < w:
        transpose = True
        h, w = img.shape[1], img.shape[0]
    asp_ratio = h / w
    down_scale_ratio = h / tgt_size
    require_rearrange = down_scale_ratio > 2.5 and asp_ratio > 3
    if not require_rearrange:
        return None, None

    if transpose:
        img = einops.rearrange(img, "h w c -> w h c")

    pw_num = max(int(np.floor(2 * tgt_size / w)), 2)
    patch_size = ph = pw_num * w
    ph_num = int(np.ceil(h / ph))
    ph_step = int((h - ph) / (ph_num - 1)) if ph_num > 1 else 0
    rel_step_list = []
    patch_list = []
    for ii in range(ph_num):
        t = ii * ph_step
        b = t + ph
        rel_step_list.append(t / h)
        patch_list.append(img[t:b])

    p_num = int(np.ceil(ph_num / pw_num))
    pad_num = p_num * pw_num - ph_num
    for ii in range(pad_num):
        patch_list.append(np.zeros_like(patch_list[0]))

    patch_lst = np.array(patch_list)
    if transpose:
        patch_lst = einops.rearrange(
            patch_lst, "(p_num pw_num) ph pw c -> p_num (pw_num pw) ph c", p_num=p_num
        )
    else:
        patch_lst = einops.rearrange(
            patch_lst, "(p_num pw_num) ph pw c -> p_num ph (pw_num pw) c", p_num=p_num
        )

    batches = [[]]
    for patch in patch_lst:
        if len(batches[-1]) >= max_batch_size:
            batches.append([])
        p, _, pad_h, pad_w = square_pad_resize(patch, tgt_size=tgt_size)
        assert pad_h == pad_w
        batches[-1].append(p)
    pad_size = pad_h

    def _unrearrange(patch_torch_lst, transpose, channel=1, pad_num=0):
        psize = _h = int(patch_torch_lst[0].shape[-1])
        _step = int(ph_step * psize / patch_size)
        _pw = int(psize / pw_num)
        _h = int(_pw / w * h)
        tgtmap = np.zeros((channel, _h, _pw), dtype=np.float32)
        num_patches = len(patch_torch_lst) * pw_num - pad_num
        for ii, p in enumerate(patch_torch_lst):
            if transpose:
                p = einops.rearrange(p, "c h w -> c w h")
            for jj in range(pw_num):
                pidx = ii * pw_num + jj
                rel_t = rel_step_list[pidx]
                t = int(round(rel_t * _h))
                b = min(t + psize, _h)
                l = jj * _pw
                r = l + _pw
                tgtmap[..., t:b, :] += p[..., : b - t, l:r]
                if pidx > 0:
                    interleave = psize - _step
                    tgtmap[..., t : t + interleave, :] /= 2.0
                if pidx >= num_patches - 1:
                    break
        if transpose:
            tgtmap = einops.rearrange(tgtmap, "c h w -> c w h")
        return tgtmap[None, ...]

    db_lst, mask_lst = [], []
    for batch in batches:
        batch = np.array(batch)
        db, mask = dbnet_batch_forward(batch, device=device)
        for d, m in zip(db, mask):
            if pad_size > 0:
                paddb = int(db.shape[-1] / tgt_size * pad_size)
                padmsk = int(mask.shape[-1] / tgt_size * pad_size)
                d = d[..., :-paddb, :-paddb]
                m = m[..., :-padmsk, :-padmsk]
            db_lst.append(d)
            mask_lst.append(m)

    db = _unrearrange(db_lst, transpose, channel=2, pad_num=pad_num)
    mask = _unrearrange(mask_lst, transpose, channel=1, pad_num=pad_num)
    return db, mask