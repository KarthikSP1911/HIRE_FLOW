"""While recording only labels and threads are built; judging and rating wait for the end."""

import asyncio

from qa_cases import BANK, TOPICS, FakeChoicePort, ScriptedRater, raw_assessment

from ai_service.assessment.judgements import Judge
from ai_service.assessment.matching import TokenSetMatcher
from ai_service.rating.consensus import PromptMode
from ai_service.rating.port import PromptStyle, RatingRequest, RawRating
from ai_service.schemas.transcript import (
    Segment,
    Sentence,
    Speaker,
    SpeakerChange,
    Transcription,
    Word,
)
from ai_service.services.assessor import ThreadAssessor
from ai_service.services.pipeline import ProcessingPipeline
from ai_service.services.sessions import Session
from ai_service.services.verifier import Verifier

QUESTION = "Explain how binary search works."
ANSWER = "You halve the list each time."
NEXT_QUESTION = "What is a deadlock?"
LOG = [
    SpeakerChange(speaker=Speaker.INTERVIEWER, at=0),
    SpeakerChange(speaker=Speaker.CANDIDATE, at=10),
    SpeakerChange(speaker=Speaker.INTERVIEWER, at=20),
]


class ChunkTranscriber:
    """Each chunk carries the next (start, text), or silence once the script runs out."""

    def __init__(self, script: list[tuple[float, str]]) -> None:
        self._script = iter(script)

    def transcribe(self, audio: bytes, suffix: str) -> Transcription:
        line = next(self._script, None)
        if line is None:
            return Transcription(language="en", language_probability=0.9, segments=[])
        start, text = line
        word = Word(text=text, start=start, end=start + 1, probability=0.9)
        return Transcription(
            language="en",
            language_probability=0.9,
            segments=[Segment(start=start, end=start + 1, text=text, words=[word])],
        )


class Labeller:
    name = "fake"

    def start_conversation(self, context_size: int):  # type: ignore[no-untyped-def]
        return self

    async def label(self, targets: list[Sentence]) -> list[dict[str, object]]:
        return [
            {
                "sentence_id": s.id,
                "label": "answer" if s.speaker is Speaker.CANDIDATE else "question",
                "confidence": 0.9,
            }
            for s in targets
        ]


class GatedRater(ScriptedRater):
    """Rating blocks until the test opens the gate, like a slow model on CPU."""

    def __init__(self) -> None:
        super().__init__({style: raw_assessment(["t2.s1"]) for style in PromptStyle})
        self.gate = asyncio.Event()

    async def rate(self, request: RatingRequest, style: PromptStyle) -> RawRating:
        await self.gate.wait()
        return await super().rate(request, style)


def build(rater: GatedRater, port: FakeChoicePort | None = None) -> ProcessingPipeline:
    assessor = ThreadAssessor(
        Judge(port or FakeChoicePort(), TOPICS, 0.6, 0.6),
        TokenSetMatcher(BANK, 0.5),
        rater,
        PromptMode.SINGLE,
    )
    # Start times are chunk-relative; the pipeline adds each chunk's offset.
    script = [(1, QUESTION), (1, ANSWER), (1, NEXT_QUESTION)]
    return ProcessingPipeline(ChunkTranscriber(script), Labeller(), Verifier(0.6, 0.5), 4, assessor)


def test_while_recording_no_model_judges_or_rates_anything() -> None:
    async def go() -> None:
        rater = GatedRater()  # gate stays shut: any rating call would hang the test
        port = FakeChoicePort()
        pipeline = build(rater, port)
        session = Session()
        for offset in (0.0, 10.0):
            await pipeline.add_chunk(session, b"x", ".webm", offset, LOG)

        # The next question closes the first thread; the reply does not wait for any model.
        reply = await asyncio.wait_for(
            pipeline.add_chunk(session, b"x", ".webm", 20.0, LOG), timeout=2
        )
        first = reply.threads[0]
        assert first.closed and first.status == "answered"
        assert first.rating is None and first.topic is None
        assert port.calls == [] and rater.requests == [] and rater.drafted == []

    asyncio.run(go())


def test_the_final_chunk_waits_until_everything_is_rated() -> None:
    async def go() -> None:
        rater = GatedRater()
        rater.gate.set()
        pipeline = build(rater)
        session = Session()
        for offset in (0.0, 10.0):
            await pipeline.add_chunk(session, b"x", ".webm", offset, LOG)

        reply = await pipeline.add_chunk(session, b"x", ".webm", 20.0, LOG, final=True)

        assert reply.threads[0].rating is not None

    asyncio.run(go())
