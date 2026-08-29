"""ComicTextDetector（ctd）检测器，移植自 MIT detection/ctd.py

网络 = YOLOv5-s backbone + UNet 文本掩膜头 + DBNet 文本行头。
GPU 用 torch 权重 comictextdetector.pt；CPU 用 ONNX + cv2.dnn。
YOLO 块/语言检测输出已弃用（照搬 MIT 行为，从 lines_map 提文本行），仅用 mask + lines。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import einops
import numpy as np
import torch

from .ctd_utils.basemodel import TextDetBase, TextDetBaseDNN
from .ctd_utils.utils.db_utils import SegDetectorRepresenter
from .ctd_utils.utils.imgproc_utils import letterbox
from .ctd_utils.textmask import refine_mask
from .paths import ensure_downloaded
from .quadrilateral import Quadrilateral
from .rearrange import det_rearrange_forward


class CTDDetector:
    """ComicTextDetector（同步推理）"""

    _PT_REL = "detection/comictextdetector.pt"
    _ONNX_REL = "detection/comictextdetector.pt.onnx"

    def __init__(self, device: str = "cpu", input_size=1024, half=False, nms_thresh=0.35, conf_thresh=0.4):
        self.device = device
        self.use_gpu = device.startswith("cuda") or device == "mps" or device == "xpu"
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.half = half
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)
        if self.use_gpu:
            self.model = TextDetBase(str(ensure_downloaded(self._PT_REL)), device=device, act="leaky")
            self.model.to(device)
            self.backend = "torch"
        else:
            model_path = ensure_downloaded(self._ONNX_REL)
            self.model = TextDetBaseDNN(input_size[0], str(model_path))
            self.backend = "opencv"

    def det_batch_forward_ctd(self, batch: np.ndarray, device: str = None) -> tuple[np.ndarray, np.ndarray]:
        if self.backend == "torch":
            batch = einops.rearrange(batch.astype(np.float32) / 255.0, "n h w c -> n c h w")
            batch = torch.from_numpy(batch).to(self.device)
            if self.half:
                batch = batch.half()
            with torch.no_grad():
                _, mask, lines = self.model(batch)
            mask = mask.detach().cpu().numpy()
            lines = lines.detach().cpu().numpy()
        else:
            mask_lst, line_lst = [], []
            for b in batch:
                _, mask, lines = self.model(b)
                if mask.shape[1] == 2:  # some opencv versions swap outputs
                    mask, lines = lines, mask
                mask_lst.append(mask)
                line_lst.append(lines)
            lines = np.concatenate(line_lst, 0)
            mask = np.concatenate(mask_lst, 0)
        return lines, mask

    def detect(
        self,
        image: np.ndarray,
        detect_size: int = 1280,
        text_threshold: float = 0.7,
        box_threshold: float = 0.7,
        unclip_ratio: float = 2.2,
    ) -> tuple[list[Quadrilateral], np.ndarray]:
        im_h, im_w = image.shape[:2]
        lines_map, mask = det_rearrange_forward(
            image, self.det_batch_forward_ctd, self.input_size[0], 4, device=self.device, verbose=False
        )
        if lines_map is None:
            img_in, dw, dh = self._preprocess_img(image)
            with torch.no_grad():
                _, mask, lines_map = self.model(img_in)
            if self.backend == "opencv":
                if mask.shape[1] == 2:  # some opencv versions swap outputs
                    mask, lines_map = lines_map, mask
            mask = mask.squeeze()
            mask = mask[..., : mask.shape[0] - dh, : mask.shape[1] - dw]
            lines_map = lines_map[..., : lines_map.shape[2] - dh, : lines_map.shape[3] - dw]

        mask = self._postprocess_mask(mask)
        lines, scores = self.seg_rep(None, lines_map, height=im_h, width=im_w)
        box_thresh = 0.6
        idx = np.where(np.asarray(scores[0]) > box_thresh)
        lines = np.asarray(lines[0])[idx]
        scores = np.asarray(scores[0])[idx]

        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

        textlines = []
        for pts, score in zip(lines, scores):
            try:
                textlines.append(Quadrilateral(pts.astype(int), "", score))
            except Exception:  # noqa: BLE001
                pass
        textlines = [q for q in textlines if q.area > 16]

        try:
            mask_refined = refine_mask(image, mask, textlines, refine_mode=None)
        except Exception:  # noqa: BLE001
            mask_refined = mask
        return textlines, mask_refined

    def _preprocess_img(self, img):
        img_in, _, (dw, dh) = letterbox(img, new_shape=self.input_size, auto=False, stride=64)
        if self.backend == "torch":
            img_in = img_in.transpose((2, 0, 1))[::-1]
            img_in = np.ascontiguousarray(img_in)
            img_in = torch.from_numpy(np.array([img_in])).float() / 255.0
            img_in = img_in.to(self.device)
            if self.half:
                img_in = img_in.half()
        return img_in, dw, dh

    @staticmethod
    def _postprocess_mask(img):
        if torch.is_tensor(img):
            img = img.squeeze().detach().cpu().numpy()
        else:
            img = np.squeeze(img)
        return (img * 255).astype(np.uint8)