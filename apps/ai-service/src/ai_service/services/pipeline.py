"""Audio in, labelled turns out: transcribe -> turns/sentences -> label -> verify."""

import asyncio
import logging
import time

from ai_service.labelling.port import DecisionError, DecisionPort
from ai_service.schemas.transcript import (
    ProcessResponse,
    Segment,
    Sentence,
    SpeakerChange,
    Transcription,
    Turn,
    Word,
)
from ai_service.services.assessor import ThreadAssessor
from ai_service.services.sessions import Session
from ai_service.services.turns import build_turns
from ai_service.services.verifier import Verifier
from ai_service.stt.transcriber import Transcriber

logger = logging.getLogger(__name__)


def _shift(segments: list[Segment], offset: float) -> list[Segment]:
    """Move chunk-relative times onto the recording's timeline."""
    return [
        Segment(
            start=s.start + offset,
            end=s.end + offset,
            text=s.text,
            words=[
                Word(
                    text=w.text,
                    start=w.start + offset,
                    end=w.end + offset,
                    probability=w.probability,
                )
                for w in s.words
            ],
        )
        for s in segments
    ]


class ProcessingPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        labeller: DecisionPort,
        verifier: Verifier,
        context_size: int,
        assessor: ThreadAssessor | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._labeller = labeller
        self._verifier = verifier
        self._context_size = context_size
        self._assessor = assessor

    async def run(self, audio: bytes, suffix: str, log: list[SpeakerChange]) -> ProcessResponse:
        """One-shot: a whole recording in a single request."""
        return await self.add_chunk(Session(), audio, suffix, 0.0, log, final=True)

    async def add_chunk(
        self,
        session: Session,
        audio: bytes,
        suffix: str,
        offset: float,
        log: list[SpeakerChange],
        final: bool = False,
    ) -> ProcessResponse:
        """Live: add the next slice of audio and return everything labelled so far.

        `final` says no more audio follows, so the last question's answers can be rated.
        """
        async with session.lock:
            # Whisper is CPU-bound, so keep it off the event loop.
            started = time.perf_counter()
            transcription = await asyncio.to_thread(self._transcriber.transcribe, audio, suffix)
            transcribed = time.perf_counter()
            self._absorb(session, transcription, offset, log)
            response = await self._label(session)
            labelled = time.perf_counter()
            if self._assessor is not None:
                sentences = [s for turn in response.turns for s in turn.sentences]
                threads = await self._assessor.assess(session.assessment, sentences, final)
                response = response.model_copy(update={"threads": threads})
            # Timings only, never content (candidate data is PII).
            logger.info(
                "chunk timing: transcribe=%.1fs label=%.1fs assess=%.1fs",
                transcribed - started,
                labelled - transcribed,
                time.perf_counter() - labelled,
            )
            return response

    @staticmethod
    def _absorb(
        session: Session, transcription: Transcription, offset: float, log: list[SpeakerChange]
    ) -> None:
        session.segments.extend(_shift(transcription.segments, offset))
        session.log = log
        if (
            transcription.segments
            and transcription.language_probability >= session.language_probability
        ):
            session.language = transcription.language
            session.language_probability = transcription.language_probability

    async def _label(self, session: Session) -> ProcessResponse:
        turns = build_turns(session.segments, session.log)
        sentences = [s for turn in turns for s in turn.sentences]
        known_ids = {s.id for s in sentences}
        checked: dict[str, Sentence] = {}
        for index, sentence in enumerate(sentences):
            cached = session.labelled.get((sentence.id, sentence.text))
            if cached is not None:
                checked[sentence.id] = cached
                continue
            context = sentences[max(0, index - self._context_size) : index]
            try:
                raw = await self._labeller.decide(sentence, context)
            except DecisionError as exc:
                # Keep every sentence visible; flag it (and retry next chunk) instead of failing.
                checked[sentence.id] = sentence.model_copy(
                    update={
                        "labeller": self._labeller.name,
                        "needs_review": True,
                        "review_reason": str(exc),
                    }
                )
                continue
            result = self._verifier.apply(sentence, raw, known_ids, self._labeller.name)
            session.labelled[(sentence.id, sentence.text)] = result
            checked[sentence.id] = result

        return ProcessResponse(
            language=session.language,
            language_probability=session.language_probability,
            labeller=self._labeller.name,
            turns=[
                Turn(
                    **turn.model_dump(exclude={"sentences"}),
                    sentences=[checked[s.id] for s in turn.sentences],
                )
                for turn in turns
            ],
        )
