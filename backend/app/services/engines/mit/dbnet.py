"""DBNet 检测模型（TextDetection + DBHead）与推理，移植自 MIT detection/default.py 及 default_utils

- 网络结构对齐 DBNet_resnet34.py，backbone 用自实现 resnet34（避免 torchvision 版本依赖）
- 输出文本行 4 点多边形 textlines + 逐像素文本掩膜 raw_mask
"""
from __future__ import annotations

from pathlib import Path

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn

from .craft_utils import adjustResultCoordinates
from .dbnet_utils import SegDetectorRepresenter
from .imgproc import resize_aspect_ratio
from .paths import ensure_downloaded
from .quadrilateral import Quadrilateral
from .rearrange import det_rearrange_forward
from .resnet34 import resnet34


class DBHead(nn.Module):
    def __init__(self, in_channels, out_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 4, 2, 1),
        )
        self.binarize.apply(self.weights_init)
        self.thresh = self._init_thresh(in_channels)
        self.thresh.apply(self.weights_init)

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        if self.training:
            binary_maps = self.step_function(shrink_maps.sigmoid(), threshold_maps)
            return torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        return torch.cat((shrink_maps, threshold_maps), dim=1)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        return nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class _double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.down = nn.AvgPool2d(2, stride=2) if stride > 1 else None
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        return self.conv(x)


class _double_conv_up(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TextDetection(nn.Module):
    """对齐 MIT DBNet_resnet34.TextDetection"""

    def __init__(self):
        super().__init__()
        self.backbone = resnet34()
        self.conv_db = DBHead(64, 0)
        self.conv_mask = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.down_conv1 = _double_conv(0, 512, 512, 2)
        self.down_conv2 = _double_conv(0, 512, 512, 2)
        self.down_conv3 = _double_conv(0, 512, 512, 2)
        self.upconv1 = _double_conv_up(0, 512, 256)
        self.upconv2 = _double_conv_up(256, 512, 256)
        self.upconv3 = _double_conv_up(256, 512, 256)
        self.upconv4 = _double_conv_up(256, 512, 256)
        self.upconv5 = _double_conv_up(256, 256, 128)
        self.upconv6 = _double_conv_up(128, 128, 64)
        self.upconv7 = _double_conv_up(64, 64, 64)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        h4 = self.backbone.layer1(x)
        h8 = self.backbone.layer2(h4)
        h16 = self.backbone.layer3(h8)
        h32 = self.backbone.layer4(h16)
        h64 = self.down_conv1(h32)
        h128 = self.down_conv2(h64)
        h256 = self.down_conv3(h128)
        up256 = self.upconv1(h256)
        up128 = self.upconv2(torch.cat([up256, h128], dim=1))
        up64 = self.upconv3(torch.cat([up128, h64], dim=1))
        up32 = self.upconv4(torch.cat([up64, h32], dim=1))
        up16 = self.upconv5(torch.cat([up32, h16], dim=1))
        up8 = self.upconv6(torch.cat([up16, h8], dim=1))
        up4 = self.upconv7(torch.cat([up8, h4], dim=1))
        return self.conv_db(up8), self.conv_mask(up4)


class DefaultDetector:
    """MIT 默认检测器（DBNet + ResNet34），同步推理接口"""

    _REL_PATH = "detection/detect-20241225.ckpt"

    def __init__(self, device: str = "cpu"):
        self.model_path = Path(ensure_downloaded(self._REL_PATH))
        self.device = device
        self.model = None
        self._load()

    def _load(self):
        model = TextDetection()
        sd = torch.load(self.model_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        model.load_state_dict(sd)
        model.eval()
        if self.device == "cpu":
            model = model.cpu()
        else:
            model = model.to(self.device)
        self.model = model

    def _batch_forward(self, batch: np.ndarray, device: str = None):
        if isinstance(batch, list):
            batch = np.array(batch)
        batch = einops.rearrange(batch.astype(np.float32) / 127.5 - 1.0, "n h w c -> n c h w")
        batch = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            db, mask = self.model(batch)
            db = db.sigmoid().cpu().numpy()
            mask = mask.cpu().numpy()
        return db, mask

    def detect(
        self,
        image: np.ndarray,
        detect_size: int = 1280,
        text_threshold: float = 0.7,
        box_threshold: float = 0.7,
        unclip_ratio: float = 2.2,
    ) -> tuple[list[Quadrilateral], np.ndarray]:
        db, mask = det_rearrange_forward(
            image, self._batch_forward, detect_size, 4, device=self.device, verbose=False
        )
        if db is None:
            img_resized, target_ratio, _, pad_w, pad_h = resize_aspect_ratio(
                cv2.bilateralFilter(image, 17, 80, 80), detect_size, cv2.INTER_LINEAR, mag_ratio=1
            )
            img_resized_h, img_resized_w = img_resized.shape[:2]
            ratio_h = ratio_w = 1 / target_ratio
            db, mask = self._batch_forward([img_resized])
        else:
            img_resized_h, img_resized_w = image.shape[:2]
            ratio_w = ratio_h = 1
            pad_h = pad_w = 0

        mask = mask[0, 0, :, :]
        det = SegDetectorRepresenter(text_threshold, box_threshold, unclip_ratio=unclip_ratio)
        boxes, scores = det({"shape": [(img_resized_h, img_resized_w)]}, db)
        boxes, scores = boxes[0], scores[0]
        if boxes.size == 0:
            polys = []
        else:
            idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
            polys, _ = boxes[idx], scores[idx]
            polys = polys.astype(np.float64)
            polys = adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=1)
            polys = polys.astype(np.int64)

        textlines = [Quadrilateral(pts.astype(int), "", score) for pts, score in zip(polys, scores)]
        textlines = [q for q in textlines if q.area > 16]

        mask_resized = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        if pad_h > 0:
            mask_resized = mask_resized[:-pad_h, :]
        elif pad_w > 0:
            mask_resized = mask_resized[:, :-pad_w]
        raw_mask = np.clip(mask_resized * 255, 0, 255).astype(np.uint8)
        return textlines, raw_mask


def load_detector(device: str = "cpu") -> DefaultDetector:
    return DefaultDetector(device=device)