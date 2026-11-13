import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, TypeAdapter, ValidationError

from ai_service.core.config import Settings, get_settings
from ai_service.core.dependencies import get_pipeline, get_session_store
from ai_service.schemas.transcript import ProcessResponse, SpeakerChange
from ai_service.services.pipeline import ProcessingPipeline
from ai_service.services.sessions import SessionNotFoundError, SessionStore

router = APIRouter(tags=["process"])

_LOG = TypeAdapter(list[SpeakerChange])
_SUFFIXES = {"webm": ".webm", "ogg": ".ogg", "mp4": ".mp4", "wav": ".wav", "mpeg": ".mp3"}


class SessionCreated(BaseModel):
    id: str


def _suffix(content_type: str) -> str:
    subtype = content_type.split(";")[0].split("/")[-1]
    return _SUFFIXES.get(subtype, ".bin")


async def _read_upload(
    audio: UploadFile, speaker_log: str, settings: Settings
) -> tuple[bytes, str, list[SpeakerChange]]:
    """Validate the upload at the boundary: type, size and speaker log."""
    content_type = audio.content_type or ""
    if not content_type.startswith(("audio/", "video/webm")):
        raise HTTPException(status_code=415, detail="audio upload required")
    try:
        log = _LOG.validate_python(json.loads(speaker_log))
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail="invalid speaker_log") from exc

    limit = settings.max_audio_mb * 1024 * 1024
    data = await audio.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="audio too large")
    return data, _suffix(content_type), log


@router.post("/process", response_model=ProcessResponse)
async def process(
    audio: Annotated[UploadFile, File()],
    speaker_log: Annotated[str, Form()],
    pipeline: Annotated[ProcessingPipeline, Depends(get_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProcessResponse:
    data, suffix, log = await _read_upload(audio, speaker_log, settings)
    return await pipeline.run(data, suffix, log)


@router.post("/sessions", status_code=201, response_model=SessionCreated)
def create_session(
    response: Response, store: Annotated[SessionStore, Depends(get_session_store)]
) -> SessionCreated:
    session_id = store.create()
    response.headers["Location"] = f"/sessions/{session_id}"
    return SessionCreated(id=session_id)


@router.post("/sessions/{session_id}/chunks", response_model=ProcessResponse)
async def add_chunk(
    session_id: str,
    audio: Annotated[UploadFile, File()],
    speaker_log: Annotated[str, Form()],
    offset: Annotated[float, Form(ge=0)],
    pipeline: Annotated[ProcessingPipeline, Depends(get_pipeline)],
    store: Annotated[SessionStore, Depends(get_session_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProcessResponse:
    try:
        session = store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc
    data, suffix, log = await _read_upload(audio, speaker_log, settings)
    return await pipeline.add_chunk(session, data, suffix, offset, log)


@router.delete("/sessions/{session_id}", status_code=204)
def end_session(
    session_id: str, store: Annotated[SessionStore, Depends(get_session_store)]
) -> None:
    store.delete(session_id)
