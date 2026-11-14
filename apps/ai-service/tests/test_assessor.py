"""The three scripted cases through the whole assessment: pair, tag, rate, verify."""

import asyncio

import pytest
from qa_cases import (
    BANK,
    DONT_KNOW,
    JARGON_SHALLOW,
    PLAIN_CORRECT,
    TOPICS,
    FakeChoicePort,
    QaCase,
    ScriptedRater,
    answer_ids,
    build_sentences,
    raw_assessment,
)

from ai_service.assessment.cache import AssessmentCache
from ai_service.assessment.judgements import Judge
from ai_service.assessment.matching import TokenSetMatcher
from ai_service.rating.consensus import DISAGREE_REASON, PromptMode
from ai_service.rating.port import PromptStyle, RatingError, RatingRequest, RawRating
from ai_service.schemas.assessment import (
    AnswerStatus,
    Correctness,
    QuestionThread,
    RubricSource,
)
from ai_service.services.assessor import ThreadAssessor


def assessor(
    case: QaCase,
    rater: ScriptedRater,
    mode: PromptMode = PromptMode.DUAL,
    port: FakeChoicePort | None = None,
) -> ThreadAssessor:
    judge = Judge(port or FakeChoicePort(attempt=case.attempt), TOPICS, 0.6, 0.6)
    return ThreadAssessor(judge, TokenSetMatcher(BANK, 0.5), rater, mode)


def run(
    subject: ThreadAssessor, case: QaCase, cache: AssessmentCache | None = None, final: bool = True
) -> QuestionThread:
    threads = asyncio.run(subject.assess(cache or AssessmentCache(), build_sentences(case), final))
    assert len(threads) == 1
    return threads[0]


def both_styles(raw: RawRating) -> dict[PromptStyle, RawRating]:
    return {PromptStyle.CONCEPT_FIRST: raw, PromptStyle.TERMINOLOGY_AWARE: raw}


def test_plain_language_correct_answer_is_rated_right_and_not_flagged() -> None:
    ids = answer_ids(PLAIN_CORRECT)
    rater = ScriptedRater(both_styles(raw_assessment(ids)))

    thread = run(assessor(PLAIN_CORRECT, rater), PLAIN_CORRECT)

    assert thread.rating is not None
    assert thread.rating.correctness is Correctness.RIGHT
    assert thread.rating.evidence.correctness == ids
    assert not thread.needs_review
    assert thread.status is AnswerStatus.ANSWERED
    assert thread.topic == "algorithms"


def test_the_bank_supplies_the_expected_points_the_answer_is_judged_against() -> None:
    rater = ScriptedRater(both_styles(raw_assessment(["t2.s1"])))

    thread = run(assessor(PLAIN_CORRECT, rater), PLAIN_CORRECT)

    assert thread.rubric is not None
    assert thread.rubric.source is RubricSource.BANK
    assert thread.rubric.bank_id == "qb-binary-search"
    assert rater.requests[0][1].points == ["halves the search space", "needs sorted data"]
    assert rater.drafted == []


def test_jargon_with_shallow_reasoning_is_flagged_when_the_two_prompts_diverge() -> None:
    ids = answer_ids(JARGON_SHALLOW)
    rater = ScriptedRater(
        {
            # Concept-first sees through the vocabulary; terminology-aware is taken in by it.
            PromptStyle.CONCEPT_FIRST: raw_assessment(
                ids, ("not_covered", "not_covered"), "partly", wrong="yes"
            ),
            PromptStyle.TERMINOLOGY_AWARE: raw_assessment(ids),
        }
    )

    thread = run(assessor(JARGON_SHALLOW, rater), JARGON_SHALLOW)

    assert thread.needs_review
    assert thread.review_reason == DISAGREE_REASON
    assert thread.rating is not None
    assert thread.rating.correctness is Correctness.WRONG  # not rewarded for the buzzwords


def test_jargon_rated_low_by_both_prompts_is_wrong_without_needing_review() -> None:
    ids = answer_ids(JARGON_SHALLOW)
    rater = ScriptedRater(
        both_styles(raw_assessment(ids, ("not_covered", "not_covered"), "partly", wrong="yes"))
    )

    thread = run(assessor(JARGON_SHALLOW, rater), JARGON_SHALLOW)

    assert thread.rating is not None
    assert thread.rating.correctness is Correctness.WRONG
    assert not thread.needs_review


def test_i_dont_know_is_no_attempt_and_is_never_rated() -> None:
    rater = ScriptedRater(
        both_styles(raw_assessment(["t2.s1"], ("not_covered", "not_covered"), "off_topic"))
    )

    thread = run(assessor(DONT_KNOW, rater), DONT_KNOW)

    assert thread.status is AnswerStatus.NO_ATTEMPT
    assert thread.rating is None
    assert rater.requests == []


def test_both_prompts_are_sent_at_the_same_time() -> None:
    class Overlap(ScriptedRater):
        in_flight = 0
        peak = 0

        async def rate(self, request: RatingRequest, style: PromptStyle) -> RawRating:
            Overlap.in_flight += 1
            Overlap.peak = max(Overlap.peak, Overlap.in_flight)
            await asyncio.sleep(0.01)  # let the other prompt start before this one finishes
            Overlap.in_flight -= 1
            return await super().rate(request, style)

    rater = Overlap(both_styles(raw_assessment(["t2.s1"])))

    run(assessor(PLAIN_CORRECT, rater), PLAIN_CORRECT)

    assert Overlap.peak == 2


def test_one_prompt_failing_flags_the_thread_instead_of_losing_it() -> None:
    class SecondFails(ScriptedRater):
        async def rate(self, request: RatingRequest, style: PromptStyle) -> RawRating:
            if style is PromptStyle.TERMINOLOGY_AWARE:
                raise RatingError("down")
            return await super().rate(request, style)

    rater = SecondFails(both_styles(raw_assessment(["t2.s1"])))

    thread = run(assessor(PLAIN_CORRECT, rater), PLAIN_CORRECT)

    assert thread.rating is None
    assert "rating unavailable" in (thread.review_reason or "")


def test_single_mode_uses_only_the_concept_first_prompt() -> None:
    rater = ScriptedRater(both_styles(raw_assessment(["t2.s1"])))

    run(assessor(PLAIN_CORRECT, rater, PromptMode.SINGLE), PLAIN_CORRECT)

    assert [style for style, _ in rater.requests] == [PromptStyle.CONCEPT_FIRST]


def test_a_question_not_in_the_bank_gets_a_generated_rubric() -> None:
    case = QaCase("novel", "Why is the sky blue?", ("Scattering of light.",), "attempt")
    rater = ScriptedRater(both_styles(raw_assessment(["t2.s1"])))

    thread = run(assessor(case, rater), case)

    assert thread.rubric is not None
    assert thread.rubric.source is RubricSource.GENERATED
    assert thread.rubric.points == ["a drafted concept", "another drafted concept"]
    assert rater.drafted == ["Why is the sky blue?"]


def test_a_rating_citing_sentences_outside_the_answer_is_dropped_and_flagged() -> None:
    rater = ScriptedRater(both_styles(raw_assessment(["t1.s1"])))

    thread = run(assessor(PLAIN_CORRECT, rater), PLAIN_CORRECT)

    assert thread.rating is None
    assert thread.needs_review
    assert "outside the answer (t1.s1)" in (thread.review_reason or "")


def test_a_leading_question_marks_the_answer_as_primed() -> None:
    rater = ScriptedRater(both_styles(raw_assessment(["t2.s1"])))
    port = FakeChoicePort(leading="leading")

    thread = run(assessor(PLAIN_CORRECT, rater, port=port), PLAIN_CORRECT)

    assert thread.primed


def test_a_low_confidence_topic_is_flagged() -> None:
    class Unsure(FakeChoicePort):
        async def choose(self, *args: object) -> dict[str, object]:  # type: ignore[override]
            return {"choice": "algorithms", "confidence": 0.2}

    rater = ScriptedRater(both_styles(raw_assessment(["t2.s1"])))

    thread = run(assessor(PLAIN_CORRECT, rater, port=Unsure()), PLAIN_CORRECT)

    assert thread.needs_review
    assert "topic: low confidence" in (thread.review_reason or "")


def test_an_unreachable_rater_keeps_the_thread_flagged_and_retries_next_time() -> None:
    class Down(ScriptedRater):
        async def rate(self, request: RatingRequest, style: PromptStyle) -> RawRating:
            raise RatingError("down")

    cache = AssessmentCache()
    thread = run(assessor(PLAIN_CORRECT, Down({})), PLAIN_CORRECT, cache)

    assert thread.rating is None
    assert thread.needs_review
    assert "rating unavailable" in (thread.review_reason or "")
    assert cache.threads == {}  # not cached, so the next chunk tries again


def test_an_open_thread_is_tagged_but_not_rated_until_it_is_final() -> None:
    rater = ScriptedRater(both_styles(raw_assessment(["t2.s1"])))
    subject = assessor(PLAIN_CORRECT, rater)
    cache = AssessmentCache()

    live = run(subject, PLAIN_CORRECT, cache, final=False)
    assert (live.topic, live.rating, rater.requests) == ("algorithms", None, [])

    done = run(subject, PLAIN_CORRECT, cache, final=True)
    assert done.rating is not None


def test_a_finished_thread_is_rated_once_across_chunks() -> None:
    rater = ScriptedRater(both_styles(raw_assessment(["t2.s1"])))
    subject = assessor(PLAIN_CORRECT, rater, PromptMode.SINGLE)
    cache = AssessmentCache()

    first = run(subject, PLAIN_CORRECT, cache)
    second = run(subject, PLAIN_CORRECT, cache)

    assert first == second
    assert len(rater.requests) == 1


@pytest.mark.parametrize("case", [PLAIN_CORRECT, JARGON_SHALLOW, DONT_KNOW])
def test_answers_and_follow_ups_inherit_their_threads_topic(case: QaCase) -> None:
    rater = ScriptedRater(both_styles(raw_assessment(answer_ids(case))))

    thread = run(assessor(case, rater), case)

    assert thread.topic == "algorithms"
    assert thread.answer_ids == answer_ids(case)
