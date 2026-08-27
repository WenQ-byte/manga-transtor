"""AI 引擎包：可插拔的检测/OCR/翻译/修复/渲染引擎"""
from app.services.engines.factory import get_engine

__all__ = ["get_engine"]
