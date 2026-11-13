from typing import Annotated

from fastapi import APIRouter, Depends

from ai_service.core.config import Settings, get_settings
from ai_service.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    return HealthResponse(status="UP", service=settings.app_name)
