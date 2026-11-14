"""Turn the model's per-point coverage into correctness and depth. Plain code, no model.

A small model is poor at judging "how correct" in one step but can say whether a stated concept
point appears in the answer. So the model only reports coverage, and the scales follow from it.
"""

from ai_service.schemas.assessment import (
    CORRECTNESS_SCALE,
    Correctness,
    Coverage,
    Depth,
    Evidence,
    ModelAssessment,
    PointCoverage,
    Rating,
    Relevance,
    YesNo,
)

COVERAGE_WEIGHT = {Coverage.COVERED: 1.0, Coverage.PARTLY: 0.5, Coverage.NOT_COVERED: 0.0}
# Share of expected points covered (partly counts half) needed for each step.
RIGHT_AT = 0.75
PARTLY_RIGHT_AT = 0.25
SURFACE_BELOW = 0.5
DETAILED_MIN_SENTENCES = 2
MAX_SCORE = 10
# Stating something false costs this many points, on top of the points it failed to cover.
WRONG_STATEMENT_PENALTY = 2


def _score(points: list[PointCoverage], expected: int) -> float:
    """Points the model skipped count as not covered, so omissions never raise a rating."""
    seen: dict[int, Coverage] = {}
    for item in points:
        seen.setdefault(item.point, item.coverage)
    return sum(COVERAGE_WEIGHT[c] for p, c in seen.items() if p <= expected) / expected


def _dedupe(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def _points_out_of_ten(share: float, assessment: ModelAssessment) -> int:
    """Share of expected points covered, scaled to 10, with an off-topic answer scoring 0."""
    if assessment.relevance is Relevance.OFF_TOPIC:
        return 0
    points = int(share * MAX_SCORE + 0.5)  # round half up (Python's round() rounds 2.5 down)
    if assessment.states_something_wrong is YesNo.YES:
        points -= WRONG_STATEMENT_PENALTY
    return max(points, 0)


def derive_rating(assessment: ModelAssessment, expected_points: int) -> Rating:
    score = _score(assessment.points, expected_points)
    if score >= RIGHT_AT:
        correctness = Correctness.RIGHT
    elif score >= PARTLY_RIGHT_AT:
        correctness = Correctness.PARTLY_RIGHT
    else:
        correctness = Correctness.WRONG
    if assessment.states_something_wrong is YesNo.YES:
        # Stating something false costs a step, whatever else the answer got right.
        step = max(CORRECTNESS_SCALE.index(correctness) - 1, 0)
        correctness = CORRECTNESS_SCALE[step]

    supporting = _dedupe(
        [i for p in assessment.points if p.coverage is not Coverage.NOT_COVERED for i in p.evidence]
    )
    if score < SURFACE_BELOW:
        depth = Depth.SURFACE
    elif score >= RIGHT_AT and len(supporting) >= DETAILED_MIN_SENTENCES:
        depth = Depth.DETAILED
    else:
        depth = Depth.ADEQUATE

    wrong = _dedupe(assessment.wrong_evidence)
    return Rating(
        relevance=assessment.relevance,
        correctness=correctness,
        depth=depth,
        example_given=assessment.example_given,
        score=_points_out_of_ten(score, assessment),
        evidence=Evidence(
            relevance=supporting,
            correctness=_dedupe([*supporting, *wrong]),
            depth=supporting,
            example_given=_dedupe(assessment.example_evidence),
        ),
        coverage=sorted(assessment.points, key=lambda p: p.point),
    )
