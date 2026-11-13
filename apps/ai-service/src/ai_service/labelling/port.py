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
