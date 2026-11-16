from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qa_cases import BANK, TOPICS, FakeChoicePort, ScriptedRater, raw_assessment

from ai_service.assessment.judgements import Judge
from ai_service.assessment.matching import TokenSetMatcher
from ai_service.core.dependencies import get_pipeline, get_session_store
from ai_service.main import create_app
from ai_service.rating.consensus import PromptMode
from ai_service.rating.port import PromptStyle
from ai_service.schemas.transcript import Segment, Sentence, Transcription, Word
from ai_service.services.assessor import ThreadAssessor
from ai_service.services.pipeline import ProcessingPipeline
from ai_service.services.sessions import SessionStore
from ai_service.services.verifier import Verifier

QUESTION = "Explain how binary search works."
ANSWER = "You halve the list each time."


class ScriptedTranscriber:
    """One sentence per chunk, then silence: the final chunk carries no new speech."""

    def __init__(self) -> None:
        self._texts = iter([QUESTION, ANSWER])

    def transcribe(self, audio: bytes, suffix: str) -> Transcription:
        text = next(self._texts, None)
        if text is None:
            return Transcription(language="en", language_probability=0.9, segments=[])
        word = Word(text=text, start=1, end=2, probability=0.9)
        return Transcription(
            language="en",
            language_probability=0.9,
            segments=[Segment(start=1, end=2, text=text, words=[word])],
        )


class ScriptedLabeller:
    name = "fake"

    def start_conversation(self, context_size: int):  # type: ignore[no-untyped-def]
        return self

    async def label(self, targets: list[Sentence]) -> list[dict[str, object]]:
        return [
            {
                "sentence_id": s.id,
                "label": "question" if s.text == QUESTION else "answer",
                "confidence": 0.9,
            }
            for s in targets
        ]


@pytest.fixture
def client() -> Iterator[TestClient]:
    rater = ScriptedRater({style: raw_assessment(["t2.s1"]) for style in PromptStyle})
    assessor = ThreadAssessor(
        Judge(FakeChoicePort(), TOPICS, 0.6, 0.6),
        TokenSetMatcher(BANK, 0.5),
        rater,
        PromptMode.SINGLE,
    )
    pipeline = ProcessingPipeline(
        ScriptedTranscriber(), ScriptedLabeller(), Verifier(0.6, 0.5), 4, assessor
    )
    store = SessionStore()
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_session_store] = lambda: store
    yield TestClient(app)


def send(client: TestClient, session: str, offset: float, final: bool) -> dict[str, Any]:
    log = '[{"speaker": "interviewer", "at": 0}, {"speaker": "candidate", "at": 10}]'
    response = client.post(
        f"/sessions/{session}/chunks",
        files={"audio": ("s", b"x", "audio/webm")},
        data={"speaker_log": log, "offset": str(offset), "final": str(final).lower()},
    )
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


def test_the_last_answer_is_rated_only_once_the_client_says_final(client: TestClient) -> None:
    session = client.post("/sessions").json()["id"]

    send(client, session, 0, final=False)
    live = send(client, session, 10, final=False)
    done = send(client, session, 20, final=True)

    assert [t["answer_ids"] for t in live["threads"]] == [["t2.s1"]]
    assert live["threads"][0]["rating"] is None  # last question stays open
    assert done["threads"][0]["rating"]["correctness"] == "right"
    assert done["threads"][0]["topic"] == "algorithms"
