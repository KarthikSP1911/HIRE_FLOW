import asyncio
import json
from collections.abc import Callable

import httpx
import pytest
from qa_cases import raw_assessment

from ai_service.assessment.rating_verifier import check_rating
from ai_service.rating.consensus import agree
from ai_service.rating.ollama import RATING_SCHEMA, OllamaRater
from ai_service.rating.port import AnswerSentence, PromptStyle, RatingError, RatingRequest
from ai_service.schemas.assessment import Correctness, Depth, Evidence, Rating, Relevance, YesNo

ANSWER = {"a1", "a2"}
KNOWN = {"q1", "a1", "a2", "x9"}
POINTS = 2


def good(**changes: object) -> dict[str, object]:
    return {**raw_assessment(["a1"]), **changes}


def check(raw: dict[str, object]) -> tuple[Rating | None, list[str]]:
    return check_rating(raw, ANSWER, KNOWN, POINTS)


def test_a_valid_rating_citing_answer_sentences_is_accepted() -> None:
    rating, problems = check(good())

    assert rating is not None
    assert rating.correctness is Correctness.RIGHT
    assert [c.point for c in rating.coverage] == [1, 2]
    assert problems == []


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (good(relevance="excellent"), "rating failed schema validation"),
        (good(states_something_wrong=None), "rating failed schema validation"),
        (good(points=[{"point": 1, "coverage": "mostly", "evidence": []}]), "schema validation"),
        ({"relevance": "fully"}, "rating failed schema validation"),
        (raw_assessment(["nope"]), "unknown sentence id nope"),
        (raw_assessment(["q1"]), "outside the answer (q1)"),
        (raw_assessment(["a1", "x9"]), "outside the answer (x9)"),
    ],
)
def test_a_bad_rating_is_rejected_with_a_reason(raw: dict[str, object], reason: str) -> None:
    rating, problems = check(raw)

    assert rating is None
    assert reason in problems[0]


def test_a_covered_point_without_evidence_is_kept_but_flagged() -> None:
    rating, problems = check(raw_assessment([]))

    assert rating is not None
    assert problems == ["rating cites no evidence"]


def test_a_wrong_answer_with_no_evidence_is_not_flagged_for_that_alone() -> None:
    rating, problems = check(raw_assessment([], ("not_covered", "not_covered")))

    assert rating is not None
    assert rating.correctness is Correctness.WRONG
    assert problems == []


def test_example_claimed_without_a_supporting_sentence_is_flagged() -> None:
    raw = good(example_given="yes", example_evidence=[])

    _, problems = check(raw)

    assert problems == ["example_given has no supporting sentence"]


@pytest.mark.parametrize(
    ("points", "reason"),
    [
        ([{"point": 1, "coverage": "covered", "evidence": ["a1"]}], "skipped expected point 2"),
        (
            [
                {"point": 1, "coverage": "covered", "evidence": ["a1"]},
                {"point": 1, "coverage": "partly", "evidence": ["a1"]},
                {"point": 2, "coverage": "covered", "evidence": ["a1"]},
            ],
            "the same expected point twice",
        ),
        (
            [
                {"point": 1, "coverage": "covered", "evidence": ["a1"]},
                {"point": 2, "coverage": "covered", "evidence": ["a1"]},
                {"point": 3, "coverage": "covered", "evidence": ["a1"]},
            ],
            "expected point that does not exist",
        ),
    ],
)
def test_inconsistent_point_numbers_are_flagged(
    points: list[dict[str, object]], reason: str
) -> None:
    rating, problems = check(good(points=points))

    assert rating is not None
    assert any(reason in p for p in problems)


def test_a_skipped_point_counts_as_not_covered_so_omission_never_raises_a_rating() -> None:
    rating, _ = check(good(points=[{"point": 1, "coverage": "covered", "evidence": ["a1"]}]))

    assert rating is not None
    assert rating.correctness is Correctness.PARTLY_RIGHT  # 1 of 2 points, not RIGHT


def test_free_text_fields_are_dropped_not_kept() -> None:
    rating, _ = check(good(explanation="They were great"))

    assert rating is not None
    assert "explanation" not in rating.model_dump()


def rating(relevance: str, correctness: str, depth: str, example: str = "no") -> Rating:
    return Rating(
        relevance=Relevance(relevance),
        correctness=Correctness(correctness),
        depth=Depth(depth),
        example_given=YesNo(example),
        evidence=Evidence(),
        score=5,
    )


@pytest.mark.parametrize(
    ("second", "expected"),
    [
        (rating("fully", "right", "adequate"), True),  # identical
        (rating("partly", "partly_right", "surface"), True),  # one step apart on every scale
        (rating("fully", "wrong", "adequate"), False),  # right vs wrong
        (rating("off_topic", "right", "adequate"), False),  # fully vs off_topic
        (rating("fully", "right", "adequate", "yes"), False),  # example must match exactly
    ],
)
def test_prompts_agree_when_equal_or_adjacent(second: Rating, expected: bool) -> None:
    assert agree(rating("fully", "right", "adequate"), second) is expected


def ollama(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaRater:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaRater(client, "http://ollama:11434/", "qwen3.5:4b")


def reply(content: object) -> httpx.Response:
    return httpx.Response(200, json={"message": {"content": json.dumps(content)}})


REQUEST = RatingRequest("Q?", ["a point", "another point"], [AnswerSentence("a1", "text")])


def test_rate_sends_the_fixed_schema_at_temperature_zero_and_returns_the_parsed_json() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return reply(good())

    raw = asyncio.run(ollama(handler).rate(REQUEST, PromptStyle.CONCEPT_FIRST))

    body = seen[0]
    assert body["format"] == RATING_SCHEMA
    assert body["options"] == {"temperature": 0}
    assert body["stream"] is False
    assert raw["relevance"] == "fully"


def test_the_two_prompt_styles_send_different_instructions() -> None:
    systems: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        systems.append(json.loads(request.content)["messages"][0]["content"])
        return reply(good())

    rater = ollama(handler)
    asyncio.run(rater.rate(REQUEST, PromptStyle.CONCEPT_FIRST))
    asyncio.run(rater.rate(REQUEST, PromptStyle.TERMINOLOGY_AWARE))

    assert systems[0] != systems[1]
    assert all("plain everyday language" in s for s in systems)


def test_an_unreachable_server_raises_a_typed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(RatingError):
        asyncio.run(ollama(handler).rate(REQUEST, PromptStyle.CONCEPT_FIRST))


def test_non_json_content_raises_a_typed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "sure, here you go"}})

    with pytest.raises(RatingError):
        asyncio.run(ollama(handler).rate(REQUEST, PromptStyle.CONCEPT_FIRST))


def test_draft_points_are_returned_and_unusable_points_raise() -> None:
    assert asyncio.run(
        ollama(lambda r: reply({"points": ["one", " two "]})).draft_points("Q?")
    ) == [
        "one",
        "two",
    ]
    with pytest.raises(RatingError):
        asyncio.run(ollama(lambda r: reply({"points": [1, 2]})).draft_points("Q?"))
