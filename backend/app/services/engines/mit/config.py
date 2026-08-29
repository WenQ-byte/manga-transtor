"""MIT 引擎内部参数（对齐 manga-image-translator 的 DetectorConfig/OcrConfig 推理子集）"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings


@dataclass
class DetectorParams:
    detector: str = "default"  # default | ctd
    device: str = "cpu"
    detect_size: int = 1280
    text_threshold: float = 0.7
    box_threshold: float = 0.7
    unclip_ratio: float = 2.2


@dataclass
class OcrParams:
    prob: float = 0.2
    ignore_bubble: int = 0
    text_height: int = 48
    max_chunk_size: int = 16
    upscale_min_font: int = 16

@dataclass
class MixedOcrParams:
    mix_threshold: float = 0.7


def resolve_device(device: str) -> str:
    """解析 MANGA_MIT_DEVICE：auto 时自动探测 cuda/mps/cpu"""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detector_params() -> DetectorParams:
    s = get_settings()
    return DetectorParams(
        detector=s.mit_detector,
        device=resolve_device(s.mit_device),
        detect_size=s.mit_detect_size,
        text_threshold=s.mit_text_threshold,
        box_threshold=s.mit_box_threshold,
        unclip_ratio=s.mit_unclip_ratio,
    )


def ocr_params() -> OcrParams:
    s = get_settings()
    return OcrParams(
        prob=s.mit_ocr_prob,
        ignore_bubble=s.mit_ignore_bubble,
        upscale_min_font=s.mit_ocr_upscale,
    )


def mixed_ocr_params() -> MixedOcrParams:
    s = get_settings()
    return MixedOcrParams(mix_threshold=s.mit_ocr_mix_threshold)