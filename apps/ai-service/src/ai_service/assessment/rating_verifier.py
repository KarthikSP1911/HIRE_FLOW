"""Plain-code checks on a rater's output. A rating that fails a hard check is dropped."""

from pydantic import ValidationError

from ai_service.rating.derive import derive_rating
from ai_service.rating.port import RawRating
from ai_service.schemas.assessment import (
    Correctness,
    ModelAssessment,
    Rating,
    Relevance,
    YesNo,
)


def _cited(assessment: ModelAssessment) -> set[str]:
    ids = {i for p in assessment.points for i in p.evidence}
    return ids | set(assessment.wrong_evidence) | set(assessment.example_evidence)


def check_rating(
    raw: RawRating, answer_ids: set[str], known_ids: set[str], expected_points: int
) -> tuple[Rating | None, list[str]]:
    """Return the derived rating (None if the output is rejected) and the reasons to review it."""
    try:
        assessment = ModelAssessment.model_validate(raw)
    except ValidationError:
        return None, ["rating failed schema validation"]

    cited = _cited(assessment)
    unknown = sorted(cited - known_ids)
    if unknown:
        return None, [f"rating cites unknown sentence id {', '.join(unknown)}"]
    outside = sorted(cited - answer_ids)
    if outside:
        return None, [f"rating cites sentences outside the answer ({', '.join(outside)})"]

    problems: list[str] = []
    numbers = [p.point for p in assessment.points]
    if any(n > expected_points for n in numbers):
        problems.append("rating refers to an expected point that does not exist")
    if len(set(numbers)) != len(numbers):
        problems.append("rating covers the same expected point twice")
    missing = sorted(set(range(1, expected_points + 1)) - set(numbers))
    if missing:
        problems.append(f"rating skipped expected point {', '.join(map(str, missing))}")

    rating = derive_rating(assessment, expected_points)
    is_low = rating.correctness is Correctness.WRONG or rating.relevance is Relevance.OFF_TOPIC
    if not rating.evidence.all_ids() and not is_low:
        problems.append("rating cites no evidence")
    if rating.example_given is YesNo.YES and not rating.evidence.example_given:
        problems.append("example_given has no supporting sentence")
    return rating, problems
