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
    batch_max_files: int = 100
    batch_max_total_mb: int = 500
    allowed_extensions: str = ".jpg,.jpeg,.png,.webp,.bmp"

    # 流水线引擎选择: demo | real
    pipeline_mode: str = "real"
    # 翻译语言
    default_source_lang: str = "auto"
    default_target_lang: str = "zh"
    auto_source_fallback: str = "ja"
    # OCR 支持语言
    ocr_langs: str = "ja,en,ch"
    # OCR 引擎（默认混合最准）: mit48+mangaocr（默认，48px + manga-ocr 补识别） | mit48 | mangaocr | paddle（回退）
    ocr_backend: str = "mit48+mangaocr"
    # 检测引擎: cv（OpenCV 启发式） | paddle（PaddleOCR 自带检测） | manga（MIT DBNet/ctd）
    # 空字符串 = 自动：OCR 用 mit 系列时选 manga，否则 cv
    detector_backend: str = ""
    # MIT 检测器类型（默认 ctd，复杂漫画召回更强）: ctd（ComicTextDetector） | default（DBNet+ResNet34）
    mit_detector: str = "ctd"
    # MIT 模型权重目录（默认项目 data/models/mit，可指向已下载的 MIT 仓库 models/）
    mit_model_dir: str = ""
    # 额外回退目录（可指向本机 manga-image-translator 的 models/），mit_model_dir 缺失时使用
    mit_fallback_dir: str = ""
    # MIT 推理设备: cpu | cuda | mps | auto（auto 会自动探测，缺 torch 时回退）
    mit_device: str = "auto"
    # MIT 检测/识别参数
    mit_detect_size: int = 1280
    mit_text_threshold: float = 0.7
    mit_box_threshold: float = 0.7
    mit_unclip_ratio: float = 2.2
    mit_ocr_prob: float = 0.2
    # 识别优化：字号小于该值(px)的文本行先放大 2x 再识别（增强小字/风格化字体召回）
    mit_ocr_upscale: int = 16
    # 混合模式 mit48+mangaocr 的切换阈值：48px 概率低于该值时改用 manga-ocr 补识别
    mit_ocr_mix_threshold: float = 0.7
    # 气泡过滤: auto（MIT 检测时关闭白占比启发式） | on | off
    bubble_filter: str = "auto"
    # MIT 掩膜级气泡去噪（1-50，数值越小越激进；0=关闭）
    mit_ignore_bubble: int = 0

    # 翻译API（可选）
    translator_backend: str = "google"  # google | deepseek | deepl | openai
    deepl_auth_key: str = ""
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_english_model: str = "deepseek-v4-flash"
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

    # 气泡级渲染参数
    # 气泡内边距比例（相对气泡宽/高，文本留白）
    render_padding: float = 0.12
    # 竖排判定：气泡高宽比需超过该值且方向以竖排为主时才竖排
    render_vertical_min_ratio: float = 1.2

    class Config:
        env_prefix = "MANGA_"
        env_file = ".env"
        extra = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base = Path(self.data_dir)
        self.upload_dir = self.upload_dir or str(base / "uploads")
        self.result_dir = self.result_dir or str(base / "results")
        self.mit_model_dir = self.mit_model_dir or str(base / "models" / "mit")

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
