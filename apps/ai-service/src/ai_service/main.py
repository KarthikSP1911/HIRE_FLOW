import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_service.api.router import api_router
from ai_service.api.routes import health, process, ui
from ai_service.core.config import get_settings
from ai_service.labelling.port import DecisionError
from ai_service.stt.transcriber import TranscriptionError


def _problem(status: int, title: str, detail: str) -> JSONResponse:
    return JSONResponse(
        {"type": "about:blank", "title": title, "status": status, "detail": detail},
        status_code=status,
        media_type="application/problem+json",
    )


def create_app() -> FastAPI:
    settings = get_settings()
    logging.getLogger("ai_service").setLevel(settings.log_level)
    logging.basicConfig()
    app = FastAPI(title=settings.app_name)
    app.include_router(health.router)
    app.include_router(ui.router)
    app.include_router(process.router)
    app.include_router(api_router)

    @app.exception_handler(DecisionError)
    async def decision_error(_: Request, exc: DecisionError) -> JSONResponse:
        return _problem(502, "Decision model unavailable", str(exc))

    @app.exception_handler(TranscriptionError)
    async def transcription_error(_: Request, exc: TranscriptionError) -> JSONResponse:
        return _problem(422, "Transcription failed", str(exc))

    return app


app = create_app()
