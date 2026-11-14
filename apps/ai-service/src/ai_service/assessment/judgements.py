"""Single-choice judgements through the decision port: topic, attempt and leading question.

Every answer is checked in plain code: the choice must be one of the offered values and the
confidence a number in [0, 1]. Anything else is flagged, never silently accepted.
"""

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_service.assessment.catalog import Topic
from ai_service.labelling.port import ChoicePort, DecisionError, RawChoice

NO_TOPIC: Final = "none"
ATTEMPT: Final = "attempt"
NO_ATTEMPT: Final = "no_attempt"
LEADING: Final = "leading"
NEUTRAL: Final = "neutral"


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    choice: str
    confidence: float = Field(ge=0, le=1)


@dataclass(frozen=True)
class Judgement:
    """A checked choice. `choice` is None when the model output was unusable."""

    choice: str | None
    confidence: float | None
    problem: str | None
    # The model could not be reached: worth retrying later, unlike a bad answer.
    unavailable: bool = False


def check_choice(raw: RawChoice, allowed: set[str], low_confidence: float) -> Judgement:
    try:
        parsed = _Choice.model_validate(raw)
    except ValidationError:
        return Judgement(None, None, "model output failed schema validation")
    if parsed.choice not in allowed:
        return Judgement(None, parsed.confidence, f"choice '{parsed.choice}' is not allowed")
    problem = "low confidence" if parsed.confidence < low_confidence else None
    return Judgement(parsed.choice, parsed.confidence, problem)


class Judge:
    def __init__(
        self,
        port: ChoicePort,
        topics: tuple[Topic, ...],
        topic_threshold: float,
        confidence_threshold: float,
    ) -> None:
        self._port = port
        self._topics = topics
        self._topic_threshold = topic_threshold
        self._confidence_threshold = confidence_threshold

    async def topic(self, question: str) -> Judgement:
        criteria = {t.id: f"{t.label}: {t.description}" for t in self._topics}
        criteria[NO_TOPIC] = "the question fits none of the other topics"
        return await self._ask(
            f"Interview question:\n{question}",
            "topic",
            "Which topic is this interview question about?",
            criteria,
            self._topic_threshold,
        )

    async def attempt(self, question: str, answer: str) -> Judgement:
        return await self._ask(
            f"Interview question:\n{question}\n\nCandidate answer:\n{answer}",
            "attempt",
            "Did the candidate try to answer the question?",
            {
                ATTEMPT: "the candidate tries to answer, even if the answer is wrong or partial",
                NO_ATTEMPT: "the candidate says they do not know, or declines the question",
            },
            self._confidence_threshold,
        )

    async def leading(self, question: str) -> Judgement:
        return await self._ask(
            f"Interviewer question:\n{question}",
            "leading",
            "Is this question leading?",
            {
                LEADING: "the question hints at, contains or pushes toward the expected answer",
                NEUTRAL: "an open question that does not hint at the answer",
            },
            self._confidence_threshold,
        )

    async def _ask(
        self, state: str, name: str, instructions: str, criteria: dict[str, str], threshold: float
    ) -> Judgement:
        try:
            raw = await self._port.choose(state, name, instructions, criteria)
        except DecisionError as exc:
            return Judgement(None, None, f"{name} check unavailable ({exc})", unavailable=True)
        return check_choice(raw, set(criteria), threshold)
