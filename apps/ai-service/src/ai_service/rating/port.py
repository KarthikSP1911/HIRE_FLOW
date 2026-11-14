"""The rating port: every answer-rating model sits behind this interface."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class PromptStyle(StrEnum):
    CONCEPT_FIRST = "concept_first"
    TERMINOLOGY_AWARE = "terminology_aware"


class RatingError(RuntimeError):
    """The rating model could not be reached or returned an unusable response."""


@dataclass(frozen=True)
class AnswerSentence:
    id: str
    text: str


@dataclass(frozen=True)
class RatingRequest:
    question: str
    points: Sequence[str]
    answers: Sequence[AnswerSentence]


# What a model hands back, untrusted. Expected shape: schemas.assessment.Rating.
RawRating = Mapping[str, object]


class AnswerRater(Protocol):
    name: str

    async def draft_points(self, question: str) -> list[str]:
        """Concept-level expected points for a question that is not in the bank."""
        ...

    async def rate(self, request: RatingRequest, style: PromptStyle) -> RawRating: ...
