import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ai_service.core.dependencies import get_pipeline
from ai_service.main import create_app
from ai_service.schemas.transcript import Segment, Transcription, Word
from ai_service.services.pipeline import ProcessingPipeline
from ai_service.services.verifier import Verifier


class FakeTranscriber:
    def transcribe(self, audio: bytes, suffix: str) -> Transcription:
        words = [Word(text="Hello.", start=0, end=1, probability=0.9)]
        return Transcription(
            language="en",
            language_probability=0.9,
            segments=[Segment(start=0, end=1, text="Hello.", words=words)],
        )


class FakeLabeller:
    name = "fake"

    async def decide(self, sentence, context):  # type: ignore[no-untyped-def]
        return {"sentence_id": sentence.id, "label": "other", "confidence": 0.9}


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_pipeline] = lambda: ProcessingPipeline(
        FakeTranscriber(), FakeLabeller(), Verifier(0.6, 0.5), 4
    )
    yield TestClient(app)


def post(client: TestClient, content_type: str = "audio/webm", log: str | None = None):  # type: ignore[no-untyped-def]
    log = log if log is not None else json.dumps([{"speaker": "interviewer", "at": 0}])
    return client.post(
        "/process", files={"audio": ("r", b"data", content_type)}, data={"speaker_log": log}
    )


def test_process_returns_labelled_turns(client: TestClient) -> None:
    body = post(client).json()
    sentence = body["turns"][0]["sentences"][0]
    assert body["language"] == "en" and body["labeller"] == "fake"
    assert sentence["id"] == "t1.s1" and sentence["label"] == "other"


def test_non_audio_upload_is_rejected(client: TestClient) -> None:
    assert post(client, "text/plain").status_code == 415


def test_bad_speaker_log_is_rejected(client: TestClient) -> None:
    assert post(client, log="not json").status_code == 422


def test_index_and_health(client: TestClient) -> None:
    assert "Voice tagging" in client.get("/").text
    assert client.get("/health").json()["status"] == "UP"
