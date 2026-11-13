import asyncio
import json

import httpx

from ai_service.core.config import Settings
from ai_service.labelling.factory import build_labeller
from ai_service.schemas.transcript import Segment, Speaker, SpeakerChange, Transcription, Word
from ai_service.services.pipeline import ProcessingPipeline
from ai_service.services.verifier import Verifier

LOG = [SpeakerChange(speaker=Speaker.INTERVIEWER, at=0)]


class FakeTranscriber:
    def transcribe(self, audio: bytes, suffix: str) -> Transcription:
        words = [Word(text="Why?", start=0, end=1, probability=0.9)]
        return Transcription(
            language="en",
            language_probability=0.99,
            segments=[Segment(start=0, end=1, text="Why?", words=words)],
        )


def handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["questions"]["sentence_type"]["type"] == "choice"
    # Assumed response shape; see README "Not verified".
    return httpx.Response(
        200,
        json={
            "answers": {
                "sentence_type": {"choice": "question", "probabilities": {"question": 0.93}}
            }
        },
    )


def run(labeller_name: str) -> str:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(labeller=labeller_name)
    labeller = build_labeller(settings, client)
    pipeline = ProcessingPipeline(FakeTranscriber(), labeller, Verifier(0.6, 0.5), 4)
    result = asyncio.run(pipeline.run(b"x", ".webm", LOG))
    sentence = result.turns[0].sentences[0]
    assert sentence.label == "question" and not sentence.needs_review
    return result.labeller


def test_strands_is_default() -> None:
    assert run("strands") == "strands"


def test_laya_is_selected_by_config_with_same_pipeline() -> None:
    assert run("laya") == "laya"


def test_unreachable_decision_server_keeps_sentences_flagged() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(down))
    labeller = build_labeller(Settings(), client)
    pipeline = ProcessingPipeline(FakeTranscriber(), labeller, Verifier(0.6, 0.5), 4)
    sentence = asyncio.run(pipeline.run(b"x", ".webm", LOG)).turns[0].sentences[0]
    assert sentence.text == "Why?" and sentence.label is None and sentence.needs_review
