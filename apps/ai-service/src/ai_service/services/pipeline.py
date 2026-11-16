"""Audio in, labelled turns out: transcribe -> turns/sentences -> tag (one batch) -> verify."""

import asyncio
import logging
import time

from ai_service.labelling.port import RawDecision, SentenceTagger, TaggingError
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

MISSING_LABEL = "tagging model returned no label"


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
        labeller: SentenceTagger,
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

        `final` says no more audio follows. Until then a reply carries labels and question
        threads only, so it keeps up with speech on CPU; topics, attempt/leading checks and
        ratings are model findings the guide shows after the interview, so they run on the
        final chunk.
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
                threads = (
                    await self._assessor.assess(session.assessment, sentences, final=True)
                    if final
                    else self._assessor.pair_only(sentences)
                )
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
        checked: dict[str, Sentence] = {}
        targets: list[Sentence] = []
        for sentence in sentences:
            cached = session.labelled.get((sentence.id, sentence.text))
            if cached is not None:
                checked[sentence.id] = cached
            else:
                targets.append(sentence)

        if targets:
            checked.update(await self._label_batch(session, targets, sentences))

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

    async def _label_batch(
        self,
        session: Session,
        targets: list[Sentence],
        sentences: list[Sentence],
    ) -> dict[str, Sentence]:
        """Label all new sentences in one model call. Flagged results are retried next chunk."""
        if session.conversation is None:
            session.conversation = self._labeller.start_conversation(self._context_size)
        try:
            raw_labels = await session.conversation.label(targets)
        except TaggingError as exc:
            # Keep every sentence visible; flag it instead of failing the request.
            return {s.id: self._flagged(s, str(exc)) for s in targets}

        by_id: dict[str, RawDecision] = {}
        for item in raw_labels:
            sentence_id = item.get("sentence_id")
            if isinstance(sentence_id, str):
                by_id.setdefault(sentence_id, item)

        known_ids = {s.id for s in sentences}
        results: dict[str, Sentence] = {}
        for sentence in targets:
            raw = by_id.get(sentence.id)
            if raw is None:
                results[sentence.id] = self._flagged(sentence, MISSING_LABEL)
                continue
            result = self._verifier.apply(sentence, raw, known_ids, self._labeller.name)
            session.labelled[(sentence.id, sentence.text)] = result
            results[sentence.id] = result
        return results

    def _flagged(self, sentence: Sentence, reason: str) -> Sentence:
        return sentence.model_copy(
            update={"labeller": self._labeller.name, "needs_review": True, "review_reason": reason}
        )
