"""Correctness and depth follow from per-point coverage, in plain code."""

import pytest
from qa_cases import raw_assessment

from ai_service.rating.derive import derive_rating
from ai_service.schemas.assessment import Correctness, Depth, ModelAssessment


def derive(
    coverages: tuple[str, ...], ids: list[str] | None = None, wrong: str = "no"
) -> tuple[Correctness, Depth]:
    raw = raw_assessment(ids if ids is not None else ["a1", "a2"], coverages, wrong=wrong)
    rating = derive_rating(ModelAssessment.model_validate(raw), expected_points=len(coverages))
    return rating.correctness, rating.depth


@pytest.mark.parametrize(
    ("coverages", "expected"),
    [
        (("covered", "covered"), Correctness.RIGHT),
        (("covered", "covered", "partly"), Correctness.RIGHT),  # 2.5 / 3 = 0.83
        (("covered", "not_covered"), Correctness.PARTLY_RIGHT),
        (("partly", "partly"), Correctness.PARTLY_RIGHT),
        (("not_covered", "not_covered"), Correctness.WRONG),
        (("partly", "not_covered", "not_covered", "not_covered"), Correctness.WRONG),  # 0.125
    ],
)
def test_correctness_follows_the_share_of_points_covered(
    coverages: tuple[str, ...], expected: Correctness
) -> None:
    assert derive(coverages)[0] is expected


@pytest.mark.parametrize(
    ("coverages", "expected"),
    [
        (("covered", "covered"), Correctness.PARTLY_RIGHT),  # RIGHT drops one step
        (("covered", "not_covered"), Correctness.WRONG),
        (("not_covered", "not_covered"), Correctness.WRONG),  # cannot go below WRONG
    ],
)
def test_stating_something_false_costs_a_step(
    coverages: tuple[str, ...], expected: Correctness
) -> None:
    assert derive(coverages, wrong="yes")[0] is expected


def test_depth_is_surface_when_most_points_are_missing() -> None:
    assert derive(("covered", "not_covered", "not_covered"))[1] is Depth.SURFACE


def test_depth_is_detailed_only_when_well_covered_across_several_sentences() -> None:
    assert derive(("covered", "covered"), ids=["a1", "a2"])[1] is Depth.DETAILED
    assert derive(("covered", "covered"), ids=["a1"])[1] is Depth.ADEQUATE  # one sentence only


def score(
    coverages: tuple[str, ...],
    relevance: str = "fully",
    wrong: str = "no",
    expected: int | None = None,
) -> int:
    raw = raw_assessment(["a1"], coverages, relevance=relevance, wrong=wrong)
    rating = derive_rating(ModelAssessment.model_validate(raw), expected or len(coverages))
    return rating.score


@pytest.mark.parametrize(
    ("coverages", "expected"),
    [
        (("covered", "covered"), 10),
        (("covered", "covered", "partly"), 8),  # 2.5 of 3 = 0.83
        (("covered", "not_covered"), 5),
        (("covered", "partly", "not_covered", "not_covered"), 4),  # 1.5 of 4 = 0.375
        (("not_covered", "not_covered"), 0),
    ],
)
def test_score_is_the_share_of_expected_points_covered_out_of_ten(
    coverages: tuple[str, ...], expected: int
) -> None:
    assert score(coverages) == expected


def test_a_false_statement_costs_two_points_but_never_goes_below_zero() -> None:
    assert score(("covered", "covered"), wrong="yes") == 8
    assert score(("not_covered", "not_covered"), wrong="yes") == 0


def test_an_off_topic_answer_scores_zero_whatever_the_points_say() -> None:
    assert score(("covered", "covered"), relevance="off_topic") == 0


def test_points_the_model_skipped_lower_the_score() -> None:
    assert score(("covered",), expected=2) == 5


def test_evidence_lists_only_sentences_behind_covered_points_and_wrong_claims() -> None:
    raw = raw_assessment(["a1"], ("covered", "not_covered"), wrong="yes")
    raw["wrong_evidence"] = ["a2"]

    rating = derive_rating(ModelAssessment.model_validate(raw), expected_points=2)

    assert rating.evidence.relevance == ["a1"]
    assert rating.evidence.correctness == ["a1", "a2"]
