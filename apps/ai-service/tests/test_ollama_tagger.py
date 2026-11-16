import asyncio
import json
import math
import re
from collections.abc import Callable, Mapping

import httpx
import pytest

from ai_service.labelling.confidence import UNMEASURED
from ai_service.labelling.ollama import OllamaTagger
from ai_service.labelling.port import TaggingError
from ai_service.schemas.transcript import (
    Segment,
    Sentence,
    Speaker,
    SpeakerChange,
    Transcription,
    Word,
)
from ai_service.services.pipeline import MISSING_LABEL, ProcessingPipeline
from ai_service.services.sessions import Session
from ai_service.services.verifier import Verifier

LOG = [
    SpeakerChange(speaker=Speaker.INTERVIEWER, at=0),
    SpeakerChange(speaker=Speaker.CANDIDATE, at=2),
]
# Pre-schema alternatives where the value starts: the chosen value is by far the most likely.
SURE = {"question": math.log(0.9), "answer": math.log(0.05), "What": math.log(0.05)}


class FakeTranscriber:
    """An interviewer question followed by a candidate answer."""

    def transcribe(self, audio: bytes, suffix: str) -> Transcription:
        def segment(start: float, text: str) -> Segment:
            word = Word(text=text, start=start, end=start + 1, probability=0.9)
            return Segment(start=start, end=start + 1, text=text, words=[word])

        return Transcription(
            language="en",
            language_probability=0.99,
            segments=[segment(0, "Why use a hash map?"), segment(2, "Lookups are fast.")],
        )


def reply(
    content: Mapping[str, str], alternatives: Mapping[str, Mapping[str, float]] | None = None
) -> httpx.Response:
    """An Ollama /api/chat body with logprobs. Each value is one token, as real models emit.

    `alternatives` maps a value to the alternatives reported at its token; by default the
    chosen value gets probability 0.9.
    """
    text = json.dumps(content)
    tokens: list[dict[str, object]] = []
    position = 0
    for match in re.finditer(r'"[^"]+": "(?P<value>[^"]*)"', text):
        value = match["value"]
        tokens.append({"token": text[position : match.start("value")], "top_logprobs": []})
        alts = (alternatives or {}).get(value, {value: math.log(0.9), "x": math.log(0.1)})
        top = [{"token": t, "logprob": lp} for t, lp in alts.items()]
        tokens.append({"token": value, "logprob": 0.0, "top_logprobs": top})
        position = match.end("value")
    tokens.append({"token": text[position:], "top_logprobs": []})
    return httpx.Response(200, json={"message": {"content": text}, "logprobs": tokens})


def tagger(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaTagger:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaTagger(client, "http://ollama:11434/", "qwen3.5:4b")


def pipeline(handler: Callable[[httpx.Request], httpx.Response]) -> ProcessingPipeline:
    return ProcessingPipeline(FakeTranscriber(), tagger(handler), Verifier(0.6, 0.5), 4)


def asked_ids(request: httpx.Request) -> list[str]:
    return list(json.loads(request.content)["format"]["required"])


def test_one_call_labels_every_new_sentence_with_a_compact_schema() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return reply({"1": "question", "2": "answer"}, {"question": SURE})

    result = asyncio.run(pipeline(handler).run(b"x", ".webm", LOG))

    assert len(seen) == 1
    body = seen[0]
    assert body["format"]["required"] == ["1", "2"]  # type: ignore[index]
    assert body["logprobs"] is True and body["options"] == {"temperature": 0}
    sentences = [s for turn in result.turns for s in turn.sentences]
    assert [s.label for s in sentences] == ["question", "answer"]
    assert sentences[0].label_confidence == pytest.approx(0.9 / 0.95)
    assert not any(s.needs_review for s in sentences)
    assert result.labeller == "ollama:qwen3.5:4b"


def test_cached_sentences_are_not_sent_again() -> None:
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(asked_ids(request))
        return reply(dict.fromkeys(calls[-1], "other"))

    subject = pipeline(handler)
    session = Session()
    asyncio.run(subject.add_chunk(session, b"x", ".webm", 0.0, LOG))
    # Same sentences again: everything is already labelled.
    session.segments.clear()
    asyncio.run(subject.add_chunk(session, b"x", ".webm", 0.0, LOG))

    assert len(calls) == 1


def test_missing_and_invented_ids_are_flagged_and_retried_next_chunk() -> None:
    # Key "9" matches no sentence; on the retry only the second sentence is asked, as "1".
    replies = iter([{"1": "question", "9": "answer"}, {"1": "answer"}])

    subject = pipeline(lambda request: reply(next(replies)))
    session = Session()
    first = asyncio.run(subject.add_chunk(session, b"x", ".webm", 0.0, LOG))
    answer = first.turns[1].sentences[0]
    assert answer.needs_review and answer.review_reason == MISSING_LABEL and answer.label is None

    # No new audio: only the missing sentence is asked again.
    session.segments.clear()
    second = asyncio.run(subject.add_chunk(session, b"x", ".webm", 0.0, LOG))
    assert second.turns[1].sentences[0].label == "answer"


def test_an_unsure_model_is_flagged_for_review() -> None:
    torn = {"question": math.log(0.4), "follow": math.log(0.35), "other": math.log(0.25)}

    def handler(request: httpx.Request) -> httpx.Response:
        return reply({"1": "question", "2": "answer"}, {"question": torn})

    result = asyncio.run(pipeline(handler).run(b"x", ".webm", LOG))

    question = result.turns[0].sentences[0]
    assert question.label == "question" and question.label_confidence == pytest.approx(0.4)
    assert question.needs_review and "low label confidence" in (question.review_reason or "")


def test_unreachable_ollama_keeps_every_sentence_flagged() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    result = asyncio.run(pipeline(down).run(b"x", ".webm", LOG))

    sentences = [s for turn in result.turns for s in turn.sentences]
    assert all(s.needs_review and s.label is None for s in sentences)
    assert "tagging model unavailable" in (sentences[0].review_reason or "")


def test_a_server_without_logprobs_is_a_typed_error() -> None:
    def old_server(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": '{"choice": "a"}'}})

    with pytest.raises(TaggingError):
        asyncio.run(tagger(old_server).choose("s", "topic", "Which?", {"a": "x"}))


def test_non_json_content_raises_a_typed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "sure, here you go"}})

    with pytest.raises(TaggingError):
        asyncio.run(tagger(handler).choose("s", "topic", "Which?", {"a": "x"}))


def test_choose_offers_exactly_the_criteria_and_measures_confidence() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        alts = {"leading": math.log(0.6), "neutral": math.log(0.2), "Yes": math.log(0.2)}
        return reply({"choice": "leading"}, {"leading": alts})

    criteria = {"leading": "hints at the answer", "neutral": "open question"}
    raw = asyncio.run(tagger(handler).choose("Q: isn't it O(1)?", "leading", "Leading?", criteria))

    assert raw["choice"] == "leading"
    assert raw["confidence"] == pytest.approx(0.75)  # 0.6 / (0.6 + 0.2): "Yes" is not allowed
    choice = seen[0]["format"]["properties"]["choice"]  # type: ignore[index]
    assert choice["enum"] == ["leading", "neutral"]
    prompt = seen[0]["messages"][0]["content"]  # type: ignore[index]
    assert "- neutral: open question" in prompt and "isn't it O(1)?" in prompt


def test_a_choice_the_model_barely_considered_is_unmeasured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # The schema forced "neutral", but it is not among the reported alternatives.
        return reply({"choice": "neutral"}, {"neutral": {"Yes": math.log(0.9)}})

    raw = asyncio.run(
        tagger(handler).choose("s", "leading", "Leading?", {"leading": "", "neutral": ""})
    )

    assert raw["confidence"] == UNMEASURED


def _sentence(sid: str, text: str) -> Sentence:
    return Sentence(
        id=sid, speaker=Speaker.INTERVIEWER, start=0, end=1, text=text, transcription_confidence=0.9
    )


def test_each_batch_continues_one_conversation_replaying_replies_exactly() -> None:
    seen: list[list[dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["messages"])
        return reply(dict.fromkeys(asked_ids(request), "question"))

    conversation = tagger(handler).start_conversation(4)
    asyncio.run(conversation.label([_sentence("t1.s1", "Why?")]))
    asyncio.run(conversation.label([_sentence("t2.s1", "How?")]))

    first, second = seen
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[:2] == first  # the earlier prompt is an exact prefix
    assert second[2]["content"] == '{"1": "question"}'  # the reply as generated
    assert second[3]["content"] == "[1] interviewer: How?"  # only the new sentence


def test_a_long_conversation_restarts_and_restates_earlier_sentences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ai_service.labelling.ollama.MAX_HISTORY_CHARS", 10)
    seen: list[list[dict[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["messages"])
        return reply(dict.fromkeys(asked_ids(request), "question"))

    conversation = tagger(handler).start_conversation(4)
    asyncio.run(conversation.label([_sentence("t1.s1", "Why is the sky blue?")]))
    asyncio.run(conversation.label([_sentence("t2.s1", "How?")]))

    restarted = seen[1]
    assert [m["role"] for m in restarted] == ["system", "user"]
    assert "Earlier:\ninterviewer: Why is the sky blue?" in restarted[1]["content"]
