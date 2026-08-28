"""应用配置模块"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置，可通过环境变量覆盖"""

    app_name: str = "漫画多语言智能翻译系统"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000

    # 存储
    data_dir: str = str(BASE_DIR / "data")
    upload_dir: str = ""
    result_dir: str = ""

    # 文件限制
    max_upload_mb: int = 10
    allowed_extensions: str = ".jpg,.jpeg,.png,.webp,.bmp"

    # 流水线引擎选择: demo | real
    pipeline_mode: str = "real"
    # 翻译语言
    default_target_lang: str = "CHS"
    # OCR 支持语言
    ocr_langs: str = "ja,en,ch"

    # 翻译API（可选）
    translator_backend: str = "google"  # google | deepseek | deepl | openai
    deepl_auth_key: str = ""
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    google_translate_base: str = "https://translate.googleapis.com/translate_a/single"

    # 图像修复引擎: cv（无模型，轻量） | lama（神经网络，需 torch + 权重）
    inpainter_backend: str = "cv"
    # LaMa 权重路径（lama_large_512px.ckpt）；空则自动搜索 项目data/models/ 与本机 manga-image-translator 路径
    lama_model_path: str = ""
    # LaMa 推理缩放上限（最长边像素；CPU 建议 1024，越大越慢）
    lama_inpaint_size: int = 1024
    # 修复推理设备: cpu | cuda（cuda 不可用时自动回退 cpu）
    inpaint_device: str = "cpu"

    class Config:
        env_prefix = "MANGA_"
        env_file = ".env"
        extra = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base = Path(self.data_dir)
        self.upload_dir = self.upload_dir or str(base / "uploads")
        self.result_dir = self.result_dir or str(base / "results")

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def result_path(self) -> Path:
        p = Path(self.result_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / "manga_translator.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
