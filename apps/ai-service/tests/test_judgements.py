import asyncio
from collections.abc import Mapping

from ai_service.assessment.catalog import Topic
from ai_service.assessment.judgements import Judge, check_choice
from ai_service.labelling.port import RawChoice, TaggingError

TOPICS = (Topic(id="algorithms", label="Algorithms", description="d"),)


class StubPort:
    name = "stub"

    def __init__(self, raw: RawChoice | None) -> None:
        self._raw = raw

    async def choose(
        self, state: str, question_name: str, instructions: str, criteria: Mapping[str, str]
    ) -> RawChoice:
        if self._raw is None:
            raise TaggingError("down")
        return self._raw


def judge(raw: RawChoice | None) -> Judge:
    return Judge(StubPort(raw), TOPICS, topic_threshold=0.6, confidence_threshold=0.6)


def test_a_listed_topic_with_good_confidence_is_accepted() -> None:
    result = asyncio.run(judge({"choice": "algorithms", "confidence": 0.9}).topic("q"))

    assert (result.choice, result.problem) == ("algorithms", None)


def test_a_topic_outside_the_list_is_rejected() -> None:
    result = asyncio.run(judge({"choice": "astrology", "confidence": 0.9}).topic("q"))

    assert result.choice is None
    assert result.problem == "choice 'astrology' is not allowed"


def test_a_low_confidence_topic_is_kept_but_flagged() -> None:
    result = asyncio.run(judge({"choice": "algorithms", "confidence": 0.3}).topic("q"))

    assert result.choice == "algorithms"
    assert result.problem == "low confidence"


def test_malformed_output_is_rejected() -> None:
    assert check_choice({"choice": "attempt", "confidence": 7}, {"attempt"}, 0.6).choice is None
    assert check_choice({"confidence": 0.9}, {"attempt"}, 0.6).choice is None


def test_an_unreachable_model_is_marked_unavailable_not_wrong() -> None:
    result = asyncio.run(judge(None).attempt("q", "a"))

    assert result.choice is None
    assert result.unavailable
