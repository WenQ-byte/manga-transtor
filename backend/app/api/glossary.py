"""专有名词（词典）API"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import GlossaryImportResult, GlossaryItem, GlossaryList
from app.services.glossary_service import GlossaryService

router = APIRouter(prefix="/api/glossary", tags=["glossary"])


def _get_service() -> GlossaryService:
    return GlossaryService()


@router.get("", response_model=GlossaryList)
async def list_glossary(lang: str = "", search: str = "", q: str = ""):
    service = _get_service()
    search_term = search or q or ""
    items = service.list_items(lang=lang or None, search=search_term)
    return GlossaryList(items=[GlossaryItem(**i) for i in items], total=len(items))


@router.post("", response_model=GlossaryItem)
async def create_glossary(item: GlossaryItem):
    if item.lang == item.target_lang:
        raise HTTPException(status_code=400, detail="词典源语言和目标语言不能相同")
    service = _get_service()
    item_id, msg = service.create(
        item.source, item.target, item.lang, item.note, item.target_lang
    )
    if item_id == -1:
        raise HTTPException(status_code=409, detail=msg)
    return GlossaryItem(id=item_id, **item.model_dump(exclude={"id"}))


@router.put("/{item_id}", response_model=GlossaryItem)
async def update_glossary(item_id: int, item: GlossaryItem):
    if item.lang == item.target_lang:
        raise HTTPException(status_code=400, detail="词典源语言和目标语言不能相同")
    service = _get_service()
    ok, msg = service.update(
        item_id, item.source, item.target, item.lang, item.note, item.target_lang
    )
    if not ok:
        raise HTTPException(status_code=404, detail=msg)
    return GlossaryItem(id=item_id, **item.model_dump(exclude={"id"}))


@router.delete("/{item_id}")
async def delete_glossary(item_id: int):
    service = _get_service()
    if not service.delete(item_id):
        raise HTTPException(status_code=404, detail="词条不存在")
    return {"deleted": True}


@router.post("/import", response_model=GlossaryImportResult)
async def import_glossary(request: Request):
    service = _get_service()
    body = await request.body()
    result = service.import_json(body.decode("utf-8", errors="replace"))
    return GlossaryImportResult(**result)
