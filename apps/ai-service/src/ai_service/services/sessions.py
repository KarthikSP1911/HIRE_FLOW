"""In-memory live sessions. Nothing is persisted; a session is dropped when it ends."""

import asyncio
import secrets
from dataclasses import dataclass, field

from ai_service.assessment.cache import AssessmentCache
from ai_service.schemas.transcript import Segment, Sentence, SpeakerChange

MAX_SESSIONS = 20


@dataclass
class Session:
    segments: list[Segment] = field(default_factory=list)
    log: list[SpeakerChange] = field(default_factory=list)
    language: str = "und"
    language_probability: float = 0.0
    # Labelled sentences keyed by (id, text): finished sentences are labelled once,
    # while a sentence that grew or changed in a later chunk is labelled again.
    labelled: dict[tuple[str, str], Sentence] = field(default_factory=dict)
    assessment: AssessmentCache = field(default_factory=AssessmentCache)
    # One chunk at a time per session, so chunk order is preserved.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionNotFoundError(LookupError):
    """No live session has this id."""


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> str:
        if len(self._sessions) >= MAX_SESSIONS:
            # Drop the oldest so abandoned sessions cannot pile up in memory.
            self._sessions.pop(next(iter(self._sessions)))
        session_id = secrets.token_urlsafe(12)
        self._sessions[session_id] = Session()
        return session_id

    def get(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
