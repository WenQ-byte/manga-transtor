"""数据模型：请求/响应 schema"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

LangCode = Literal["ja", "en", "zh"]


class TranslateRequest(BaseModel):
    """翻译任务请求"""

    source_lang: LangCode = Field(..., description="源语言")
    target_lang: LangCode = Field("zh", description="目标语言")
    # 可选：字体、渲染方向等高级参数（P1）

    model_config = {"json_schema_extra": {"example": {"source_lang": "ja", "target_lang": "zh"}}}


class TranslateResponse(BaseModel):
    """创建翻译任务响应"""

    task_id: str
    status: str = "queued"
    message: str = ""


class BatchTaskItem(BaseModel):
    """批量任务中的单图任务。"""

    task_id: str
    filename: str
    index: int
    status: Literal["queued", "processing", "completed", "failed"] = "queued"


class BatchTranslateResponse(BaseModel):
    """批量创建任务响应。"""

    total: int
    items: list[BatchTaskItem]


class BatchStatusRequest(BaseModel):
    """批量状态或导出请求。"""

    task_ids: list[str] = Field(..., min_length=1)


class BatchStatusItem(BaseModel):
    """批量状态中的单图进度。"""

    task_id: str
    filename: str
    index: int
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int = 0
    text_count: int = 0
    duration_ms: int = 0
    error: Optional[str] = None


class BatchStatusResponse(BaseModel):
    """批量任务汇总状态。"""

    total: int
    completed: int
    processing: int
    failed: int
    progress: int
    items: list[BatchStatusItem]


class PipelineStep(BaseModel):
    """流水线步骤进度"""

    name: str
    label: str
    progress: int  # 0-100
    done: bool


class TranslateStatus(BaseModel):
    """翻译任务状态"""

    task_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int = 0
    step: Optional[PipelineStep] = None
    error: Optional[str] = None
    message: Optional[str] = None
    text_count: int = 0
    duration_ms: int = 0


class TranslateResult(BaseModel):
    """翻译完成结果元数据"""

    task_id: str
    status: str = "completed"
    result_url: str
    original_url: str
    source_lang: str
    target_lang: str
    text_count: int = 0
    duration_ms: int = 0


class GlossaryItem(BaseModel):
    """专有名词条目"""

    id: Optional[int] = None
    source: str = Field(..., min_length=1, max_length=100, description="源词")
    target: str = Field(..., min_length=1, max_length=100, description="目标词")
    lang: str = Field("ja", description="源词语言")
    note: str = Field("", max_length=200, description="备注")

    model_config = {
        "json_schema_extra": {
            "example": {"source": "ナルト", "target": "鸣人", "lang": "ja", "note": "火影忍者主角"}
        }
    }


class GlossaryList(BaseModel):
    items: list[GlossaryItem]
    total: int


class GlossaryImportResult(BaseModel):
    imported: int
    skipped: int
    errors: list[str] = []
