"""The three scripted cases against a real local model. Skipped unless Ollama is reachable.

Run with: uv run pytest -m ollama
Not run in CI or on machines without Ollama; see the README for what is verified.
"""

import asyncio

import httpx
import pytest
from qa_cases import (
    BANK,
    DONT_KNOW,
    JARGON_SHALLOW,
    PLAIN_CORRECT,
    TOPICS,
    FakeChoicePort,
    QaCase,
    build_sentences,
)

from ai_service.assessment.cache import AssessmentCache
from ai_service.assessment.judgements import Judge
from ai_service.assessment.matching import TokenSetMatcher
from ai_service.core.config import Settings
from ai_service.rating.consensus import PromptMode
from ai_service.rating.ollama import OllamaRater
from ai_service.schemas.assessment import AnswerStatus, Correctness
from ai_service.services.assessor import ThreadAssessor

pytestmark = pytest.mark.ollama

SETTINGS = Settings()


def _ollama_is_up() -> bool:
    try:
        return httpx.get(f"{SETTINGS.ollama_url}/api/tags", timeout=1).is_success
    except httpx.HTTPError:
        return False


@pytest.fixture(autouse=True)
def require_ollama() -> None:
    if not _ollama_is_up():
        pytest.skip("Ollama is not running")


def assess(case: QaCase):  # type: ignore[no-untyped-def]
    async def go():  # type: ignore[no-untyped-def]
        async with httpx.AsyncClient(timeout=SETTINGS.ollama_timeout_seconds) as client:
            rater = OllamaRater(client, SETTINGS.ollama_url, SETTINGS.ollama_model)
            judge = Judge(FakeChoicePort(attempt=case.attempt), TOPICS, 0.6, 0.6)
            subject = ThreadAssessor(judge, TokenSetMatcher(BANK, 0.5), rater, PromptMode.DUAL)
            return (await subject.assess(AssessmentCache(), build_sentences(case), True))[0]

    return asyncio.run(go())


RANK = {Correctness.WRONG: 0, Correctness.PARTLY_RIGHT: 1, Correctness.RIGHT: 2}


def test_plain_language_answer_is_not_marked_down_for_lacking_jargon() -> None:
    thread = assess(PLAIN_CORRECT)

    assert thread.rating is not None
    assert thread.rating.correctness is not Correctness.WRONG


def test_jargon_with_wrong_reasoning_is_rated_wrong_or_flagged() -> None:
    thread = assess(JARGON_SHALLOW)

    assert thread.needs_review or (
        thread.rating is not None and thread.rating.correctness is Correctness.WRONG
    )


def test_the_model_tells_the_two_answers_apart() -> None:
    plain = assess(PLAIN_CORRECT).rating
    jargon = assess(JARGON_SHALLOW).rating

    assert plain is not None
    assert jargon is not None
    assert RANK[plain.correctness] > RANK[jargon.correctness]


def test_i_dont_know_is_no_attempt() -> None:
    thread = assess(DONT_KNOW)

    assert thread.status is AnswerStatus.NO_ATTEMPT
    assert thread.rating is None
