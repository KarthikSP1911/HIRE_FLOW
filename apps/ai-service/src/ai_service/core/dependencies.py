from functools import lru_cache
from typing import Annotated

import httpx
from fastapi import Depends

from ai_service.core.config import Settings, get_settings
from ai_service.labelling.factory import build_labeller
from ai_service.services.pipeline import ProcessingPipeline
from ai_service.services.sessions import SessionStore
from ai_service.services.verifier import Verifier
from ai_service.stt.transcriber import FasterWhisperTranscriber, Transcriber


@lru_cache
def get_transcriber() -> Transcriber:
    settings = get_settings()
    return FasterWhisperTranscriber(
        settings.whisper_model,
        settings.whisper_device,
        settings.whisper_beam_size,
    )


@lru_cache
def get_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=get_settings().decision_timeout_seconds)


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore()


def get_pipeline(
    settings: Annotated[Settings, Depends(get_settings)],
    transcriber: Annotated[Transcriber, Depends(get_transcriber)],
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
) -> ProcessingPipeline:
    return ProcessingPipeline(
        transcriber,
        build_labeller(settings, client),
        Verifier(settings.confidence_threshold, settings.stt_low_confidence),
        settings.context_sentences,
    )
