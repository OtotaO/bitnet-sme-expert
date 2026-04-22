"""API endpoint router exports."""

from fastapi import APIRouter

from app.api.endpoints.core import router as core_router
from app.api.endpoints.fine_tuning import router as fine_tuning_router

api_router = APIRouter()
api_router.include_router(core_router)
api_router.include_router(fine_tuning_router)

__all__ = ["api_router", "core_router", "fine_tuning_router"]
