from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ai_service.core.dependencies import get_pipeline, get_session_store
from ai_service.main import create_app
from ai_service.schemas.transcript import Segment, Sentence, Transcription, Word
from ai_service.services.pipeline import ProcessingPipeline
from ai_service.services.sessions import SessionStore
from ai_service.services.verifier import Verifier


class ChunkTranscriber:
    """Each call returns the next scripted sentence, with chunk-relative times."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = iter(texts)

    def transcribe(self, audio: bytes, suffix: str) -> Transcription:
        text = next(self._texts)
        words = [Word(text=text, start=1, end=2, probability=0.9)]
        return Transcription(
            language="en",
            language_probability=0.9,
            segments=[Segment(start=1, end=2, text=text, words=words)],
        )


class CountingLabeller:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def decide(self, sentence: Sentence, context: object) -> dict[str, object]:
        self.calls.append(sentence.id)
        return {"sentence_id": sentence.id, "label": "other", "confidence": 0.9}


@pytest.fixture
def labeller() -> CountingLabeller:
    return CountingLabeller()


@pytest.fixture
def client(labeller: CountingLabeller) -> Iterator[TestClient]:
    app = create_app()
    pipeline = ProcessingPipeline(
        ChunkTranscriber(["One.", "Two."]), labeller, Verifier(0.6, 0.5), 4
    )
    store = SessionStore()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_session_store] = lambda: store
    yield TestClient(app)


def send(client: TestClient, session: str, offset: float, speaker: str) -> dict[str, Any]:
    log = f'[{{"speaker": "{speaker}", "at": 0}}]'
    response = client.post(
        f"/sessions/{session}/chunks",
        files={"audio": ("s", b"x", "audio/webm")},
        data={"speaker_log": log, "offset": str(offset)},
    )
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


def test_chunks_accumulate_on_one_timeline_and_label_each_sentence_once(
    client: TestClient, labeller: CountingLabeller
) -> None:
    session = client.post("/sessions").json()["id"]

    first = send(client, session, 0, "interviewer")
    second = send(client, session, 10, "interviewer")

    sentences = second["turns"][0]["sentences"]
    assert len(first["turns"][0]["sentences"]) == 1
    assert [s["text"] for s in sentences] == ["One.", "Two."]
    assert sentences[1]["start"] == 11  # chunk time 1 + offset 10
    assert len(second["turns"]) == 1  # same speaker continues one turn
    assert labeller.calls == ["t1.s1", "t1.s2"]  # t1.s1 was not labelled again


def test_unknown_session_is_404(client: TestClient) -> None:
    response = client.post(
        "/sessions/nope/chunks",
        files={"audio": ("s", b"x", "audio/webm")},
        data={"speaker_log": "[]", "offset": "0"},
    )
    assert response.status_code == 404


def test_ending_a_session_returns_204(client: TestClient) -> None:
    session = client.post("/sessions").json()["id"]
    assert client.delete(f"/sessions/{session}").status_code == 204
