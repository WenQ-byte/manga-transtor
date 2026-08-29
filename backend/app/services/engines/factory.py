"""引擎工厂：按需实例化可插拔引擎"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings


@lru_cache(maxsize=16)
def get_engine(engine_type: str):
    """根据配置返回引擎实例，类型: detector|ocr|translator|inpainter|renderer"""
    settings = get_settings()

    if engine_type == "detector":
        from app.services.engines.detector import create_detector_engine

        return create_detector_engine()

    if engine_type == "ocr":
        from app.services.engines.ocr import create_ocr_engine_router

        return create_ocr_engine_router()

    if engine_type == "translator":
        from app.services.engines.translator import create_translator

        return create_translator()

    if engine_type == "inpainter":
        from app.services.engines.inpainter import create_inpainter

        return create_inpainter()

    if engine_type == "renderer":
        from app.services.engines.renderer import create_renderer

        return create_renderer()

    raise ValueError(f"未知引擎类型: {engine_type}")
