from functools import lru_cache
from typing import Annotated

import httpx
from fastapi import Depends

from ai_service.assessment.catalog import Catalog, load_catalog
from ai_service.assessment.judgements import Judge
from ai_service.assessment.matching import TokenSetMatcher
from ai_service.core.config import Settings, get_settings
from ai_service.labelling.factory import build_labeller
from ai_service.rating.ollama import OllamaRater
from ai_service.services.assessor import ThreadAssessor
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
def get_ollama_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=get_settings().ollama_timeout_seconds)


@lru_cache
def get_catalog() -> Catalog:
    settings = get_settings()
    return load_catalog(settings.topics_path, settings.question_bank_path)


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore()


def get_pipeline(
    settings: Annotated[Settings, Depends(get_settings)],
    transcriber: Annotated[Transcriber, Depends(get_transcriber)],
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    ollama_client: Annotated[httpx.AsyncClient, Depends(get_ollama_client)],
    catalog: Annotated[Catalog, Depends(get_catalog)],
) -> ProcessingPipeline:
    labeller = build_labeller(settings, client)
    judge = Judge(
        labeller,
        catalog.topics,
        settings.topic_confidence_threshold,
        settings.confidence_threshold,
    )
    assessor = ThreadAssessor(
        judge,
        TokenSetMatcher(catalog.bank, settings.bank_match_threshold),
        OllamaRater(ollama_client, settings.ollama_url, settings.ollama_model),
        settings.rating_prompt_mode,
    )
    return ProcessingPipeline(
        transcriber,
        labeller,
        Verifier(settings.confidence_threshold, settings.stt_low_confidence),
        settings.context_sentences,
        assessor,
    )
