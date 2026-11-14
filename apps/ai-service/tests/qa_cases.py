"""Three scripted interview exchanges with known correct outcomes, plus fakes to drive them."""

from collections.abc import Mapping
from dataclasses import dataclass

from ai_service.assessment.catalog import BankEntry, Topic
from ai_service.assessment.judgements import ATTEMPT, NEUTRAL
from ai_service.labelling.port import RawChoice
from ai_service.rating.port import PromptStyle, RatingRequest, RawRating
from ai_service.schemas.transcript import Label, Sentence, Speaker

TOPICS = (Topic(id="algorithms", label="Algorithms", description="d"),)
BANK = [
    BankEntry(
        id="qb-binary-search",
        question="Explain how binary search works and why it is fast.",
        topic="algorithms",
        expected_points=["halves the search space", "needs sorted data"],
        difficulty="easy",
    ),
    BankEntry(
        id="qb-index",
        question="What is a database index and what is the trade-off of adding one?",
        topic="algorithms",
        expected_points=["finds rows without a full scan", "slows writes and costs space"],
        difficulty="medium",
    ),
]


@dataclass(frozen=True)
class QaCase:
    name: str
    question: str
    answer: tuple[str, ...]
    attempt: str  # what a correct attempt check says about this answer


# A correct answer in plain everyday words: must not be marked down for lacking jargon.
PLAIN_CORRECT = QaCase(
    name="plain_correct",
    question="Can you explain how binary search works and why it is fast?",
    answer=(
        "You keep the list in order and look at the middle one.",
        "If your number is smaller you throw away the right half, otherwise the left half.",
        "Every look cuts the pile in half, so even a huge list needs only a few looks.",
    ),
    attempt=ATTEMPT,
)

# Confident buzzwords around wrong reasoning: must not be rewarded for vocabulary.
JARGON_SHALLOW = QaCase(
    name="jargon_shallow",
    question="What is a database index and what is the trade-off of adding one?",
    answer=(
        "It is a B-plus-tree-backed, ACID-compliant denormalised sharding layer.",
        "It leverages polymorphic hashing to optimise throughput, so there is no downside.",
    ),
    attempt=ATTEMPT,
)

# Declining to answer: tracked as no attempt, never rated as wrong.
DONT_KNOW = QaCase(
    name="dont_know",
    question="What is a race condition and how can you prevent one?",
    answer=("Sorry, I do not know that one.",),
    attempt="no_attempt",
)

ALL_CASES = (PLAIN_CORRECT, JARGON_SHALLOW, DONT_KNOW)


def build_sentences(case: QaCase) -> list[Sentence]:
    """The interviewer's question as t1.s1, then the candidate's answer as t2.s1, t2.s2, ..."""

    def sentence(sid: str, speaker: Speaker, label: Label, text: str) -> Sentence:
        return Sentence(
            id=sid,
            speaker=speaker,
            start=0,
            end=1,
            text=text,
            transcription_confidence=0.95,
            label=label,
            label_confidence=0.95,
            labeller="fake",
        )

    return [
        sentence("t1.s1", Speaker.INTERVIEWER, Label.QUESTION, case.question),
        *[
            sentence(f"t2.s{i}", Speaker.CANDIDATE, Label.ANSWER, text)
            for i, text in enumerate(case.answer, start=1)
        ],
    ]


def answer_ids(case: QaCase) -> list[str]:
    return [f"t2.s{i}" for i in range(1, len(case.answer) + 1)]


def raw_assessment(
    ids: list[str],
    coverages: tuple[str, ...] = ("covered", "covered"),
    relevance: str = "fully",
    wrong: str = "no",
    example: str = "no",
) -> dict[str, object]:
    """What a rater model returns: coverage per expected point, citing the given sentence ids."""
    return {
        "relevance": relevance,
        "points": [
            {
                "point": n,
                "coverage": coverage,
                "evidence": ids if coverage != "not_covered" else [],
            }
            for n, coverage in enumerate(coverages, start=1)
        ],
        "states_something_wrong": wrong,
        "wrong_evidence": ids if wrong == "yes" else [],
        "example_given": example,
        "example_evidence": ids if example == "yes" else [],
    }


class FakeChoicePort:
    """Answers topic/attempt/leading the way a correct model would for the given case."""

    name = "fake-choice"

    def __init__(self, attempt: str = ATTEMPT, topic: str = "algorithms", leading: str = NEUTRAL):
        self.calls: list[str] = []
        self._answers: Mapping[str, str] = {
            "topic": topic,
            "attempt": attempt,
            "leading": leading,
        }

    async def choose(
        self, state: str, question_name: str, instructions: str, criteria: Mapping[str, str]
    ) -> RawChoice:
        self.calls.append(question_name)
        return {"choice": self._answers[question_name], "confidence": 0.9}


class ScriptedRater:
    """Returns a prepared rating per prompt style, and records every request."""

    name = "scripted"

    def __init__(
        self, ratings: Mapping[PromptStyle, RawRating], points: list[str] | None = None
    ) -> None:
        self._ratings = ratings
        self._points = points or ["a drafted concept", "another drafted concept"]
        self.requests: list[tuple[PromptStyle, RatingRequest]] = []
        self.drafted: list[str] = []

    async def draft_points(self, question: str) -> list[str]:
        self.drafted.append(question)
        return self._points

    async def rate(self, request: RatingRequest, style: PromptStyle) -> RawRating:
        self.requests.append((style, request))
        return self._ratings[style]
