"""The tagging ports: every model that labels sentences or answers choices sits behind these."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from ai_service.schemas.transcript import Sentence

# What a model hands back, untrusted: expected keys are sentence_id, label, confidence.
RawDecision = Mapping[str, object]


class TaggingError(RuntimeError):
    """The tagging model could not be reached or returned an unusable response."""


class LabelConversation(Protocol):
    """Labels one recording's sentences batch by batch, remembering what came before.

    Each batch continues the same conversation, so earlier sentences are context for later
    ones and a model server can reuse the work it already did on them.
    """

    async def label(self, targets: Sequence[Sentence]) -> Sequence[RawDecision]:
        """Label every target in one call. On error nothing is remembered."""
        ...


class SentenceTagger(Protocol):
    name: str

    def start_conversation(self, context_size: int) -> LabelConversation:
        """`context_size`: earlier sentences to restate if the conversation has to restart."""
        ...


# A single-choice answer, untrusted: expected keys are choice and confidence.
RawChoice = Mapping[str, object]


class ChoicePort(Protocol):
    """Ask a tagging model any single-choice question (topic, attempt, leading, ...)."""

    name: str

    async def choose(
        self, state: str, question_name: str, instructions: str, criteria: Mapping[str, str]
    ) -> RawChoice: ...
