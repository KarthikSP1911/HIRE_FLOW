from fastapi import FastAPI

from ai_service.api.router import api_router
from ai_service.api.routes import health
from ai_service.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(health.router)
    app.include_router(api_router)
    return app


app = create_app()
