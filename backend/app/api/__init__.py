"""API 路由汇总"""
from fastapi import APIRouter

from app.api import files, glossary, translate

api_router = APIRouter()
api_router.include_router(translate.router)
api_router.include_router(glossary.router)
api_router.include_router(files.router)
