"""The decision port: every labelling model sits behind this interface."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from ai_service.schemas.transcript import Sentence

# What a model hands back, untrusted: expected keys are sentence_id, label, confidence.
RawDecision = Mapping[str, object]


class DecisionError(RuntimeError):
    """The decision model could not be reached or returned an unusable response."""


class DecisionPort(Protocol):
    name: str

    async def decide(self, sentence: Sentence, context: Sequence[Sentence]) -> RawDecision: ...


# A single-choice answer, untrusted: expected keys are choice and confidence.
RawChoice = Mapping[str, object]


class ChoicePort(Protocol):
    """Ask a decision model any single-choice question (topic, attempt, leading, ...)."""

    name: str

    async def choose(
        self, state: str, question_name: str, instructions: str, criteria: Mapping[str, str]
    ) -> RawChoice: ...
