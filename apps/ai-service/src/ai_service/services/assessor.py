"""Question threads: pair, tag the topic, then rate the answers of finished threads."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from ai_service.assessment.cache import AssessmentCache, ThreadKey
from ai_service.assessment.judgements import LEADING, NO_ATTEMPT, NO_TOPIC, Judge, Judgement
from ai_service.assessment.matching import QuestionMatcher
from ai_service.assessment.pairing import pair_threads
from ai_service.assessment.rating_verifier import check_rating
from ai_service.rating.consensus import DISAGREE_REASON, PromptMode, agree
from ai_service.rating.port import (
    AnswerRater,
    AnswerSentence,
    PromptStyle,
    RatingError,
    RatingRequest,
)
from ai_service.schemas.assessment import (
    AnswerStatus,
    QuestionThread,
    Rating,
    Rubric,
    RubricSource,
)
from ai_service.schemas.transcript import Sentence

JOIN = "; "


def _flag(thread: QuestionThread, reason: str) -> None:
    thread.needs_review = True
    thread.review_reason = (
        f"{thread.review_reason}{JOIN}{reason}" if thread.review_reason else reason
    )


class ThreadAssessor:
    def __init__(
        self, judge: Judge, matcher: QuestionMatcher, rater: AnswerRater, mode: PromptMode
    ) -> None:
        self._judge = judge
        self._matcher = matcher
        self._rater = rater
        self._styles = [PromptStyle.CONCEPT_FIRST]
        if mode is PromptMode.DUAL:
            self._styles.append(PromptStyle.TERMINOLOGY_AWARE)

    async def assess(
        self, cache: AssessmentCache, sentences: Sequence[Sentence], final: bool
    ) -> list[QuestionThread]:
        by_id = {s.id: s for s in sentences}
        return [
            await self._assess_thread(cache, thread, by_id)
            for thread in pair_threads(sentences, final)
        ]

    async def _assess_thread(
        self, cache: AssessmentCache, thread: QuestionThread, by_id: dict[str, Sentence]
    ) -> QuestionThread:
        question = by_id[thread.question_id]
        key = self._thread_key(thread, by_id)
        cached = cache.threads.get(key)
        if cached is not None and thread.closed:
            return cached.model_copy(deep=True)

        retry = await self._tag_topic(cache, thread, question)
        if thread.closed and thread.status is AnswerStatus.ANSWERED:
            retry = await self._rate(cache, thread, question, by_id) or retry
        # Open threads change as more audio arrives, and unreachable models are retried later.
        if thread.closed and not retry:
            cache.threads[key] = thread.model_copy(deep=True)
        return thread

    @staticmethod
    def _thread_key(thread: QuestionThread, by_id: dict[str, Sentence]) -> ThreadKey:
        members = [
            *thread.answer_ids,
            *thread.follow_up_ids,
            *thread.clarification_ids,
            *thread.self_answer_ids,
        ]
        return (
            thread.question_id,
            by_id[thread.question_id].text,
            tuple((i, by_id[i].text) for i in members),
        )

    async def _judged(
        self,
        cache: AssessmentCache,
        kind: str,
        question: Sentence,
        text: str,
        ask: Callable[[], Awaitable[Judgement]],
    ) -> Judgement:
        key = (kind, question.id, text)
        if key in cache.judgements:
            return cache.judgements[key]
        judgement = await ask()
        if not judgement.unavailable:
            cache.judgements[key] = judgement
        return judgement

    async def _tag_topic(
        self, cache: AssessmentCache, thread: QuestionThread, question: Sentence
    ) -> bool:
        """Returns True if the topic model was unreachable and should be retried."""
        judgement = await self._judged(
            cache, "topic", question, question.text, lambda: self._judge.topic(question.text)
        )
        if judgement.choice not in (None, NO_TOPIC):
            thread.topic = judgement.choice
        thread.topic_confidence = judgement.confidence
        if judgement.problem:
            _flag(thread, f"topic: {judgement.problem}")
        return judgement.unavailable

    async def _rate(
        self,
        cache: AssessmentCache,
        thread: QuestionThread,
        question: Sentence,
        by_id: dict[str, Sentence],
    ) -> bool:
        """Returns True if a model was unreachable and the thread should be retried."""
        answers = [AnswerSentence(i, by_id[i].text) for i in thread.answer_ids]
        answer_text = " ".join(a.text for a in answers)

        attempt = await self._judged(
            cache,
            "attempt",
            question,
            answer_text,
            lambda: self._judge.attempt(question.text, answer_text),
        )
        if attempt.problem:
            _flag(thread, f"attempt: {attempt.problem}")
        if attempt.choice == NO_ATTEMPT:
            # Declining to answer is not a wrong answer, so it is never rated.
            thread.status = AnswerStatus.NO_ATTEMPT
            return attempt.unavailable

        leading = await self._judged(
            cache, "leading", question, question.text, lambda: self._judge.leading(question.text)
        )
        thread.primed = leading.choice == LEADING
        if leading.problem:
            _flag(thread, f"leading: {leading.problem}")

        retry = attempt.unavailable or leading.unavailable
        rubric = await self._rubric(cache, thread, question)
        if rubric is None:
            return True
        thread.rubric = rubric
        return await self._rate_answers(thread, question, rubric, answers, set(by_id)) or retry

    async def _rubric(
        self, cache: AssessmentCache, thread: QuestionThread, question: Sentence
    ) -> Rubric | None:
        entry = self._matcher.match(question.text)
        if entry is not None:
            return Rubric(
                points=list(entry.expected_points), source=RubricSource.BANK, bank_id=entry.id
            )
        points = cache.drafts.get(question.text)
        if points is None:
            try:
                points = await self._rater.draft_points(question.text)
            except RatingError as exc:
                _flag(thread, f"expected points unavailable ({exc})")
                return None
            cache.drafts[question.text] = points
        return Rubric(points=points, source=RubricSource.GENERATED)

    async def _rate_answers(
        self,
        thread: QuestionThread,
        question: Sentence,
        rubric: Rubric,
        answers: list[AnswerSentence],
        known_ids: set[str],
    ) -> bool:
        request = RatingRequest(question.text, rubric.points, answers)
        ratings: list[Rating] = []
        # Both prompts go out together: on CPU, decoding is memory-bound, so a server that
        # batches two requests finishes both in far less than twice the time of one.
        outcomes = await asyncio.gather(
            *(self._rater.rate(request, style) for style in self._styles), return_exceptions=True
        )
        for raw in outcomes:
            if isinstance(raw, RatingError):
                _flag(thread, f"rating unavailable ({raw})")
                return True
            if isinstance(raw, BaseException):
                raise raw
            rating, problems = check_rating(
                raw, set(thread.answer_ids), known_ids, len(rubric.points)
            )
            for problem in problems:
                _flag(thread, problem)
            if rating is None:
                return False  # rejected output is flagged, not retried: it would repeat
            ratings.append(rating)

        if len(ratings) > 1 and not agree(ratings[0], ratings[1]):
            _flag(thread, DISAGREE_REASON)
        thread.rating = ratings[0]
        return False
