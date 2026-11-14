"""Combine the two prompts' ratings: accept if they agree or are adjacent, else flag."""

from enum import StrEnum

from ai_service.schemas.assessment import (
    CORRECTNESS_SCALE,
    DEPTH_SCALE,
    RELEVANCE_SCALE,
    Rating,
)

DISAGREE_REASON = "rating prompts disagree"


class PromptMode(StrEnum):
    DUAL = "dual"
    SINGLE = "single"


def _adjacent(scale: tuple[StrEnum, ...], a: StrEnum, b: StrEnum) -> bool:
    return abs(scale.index(a) - scale.index(b)) <= 1


def agree(first: Rating, second: Rating) -> bool:
    """Equal or one step apart on each scale; example_given must match exactly."""
    return (
        _adjacent(RELEVANCE_SCALE, first.relevance, second.relevance)
        and _adjacent(CORRECTNESS_SCALE, first.correctness, second.correctness)
        and _adjacent(DEPTH_SCALE, first.depth, second.depth)
        and first.example_given == second.example_given
    )
