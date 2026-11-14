"""Question threads: a question, its answers, its topic and how the answers were rated."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Relevance(StrEnum):
    OFF_TOPIC = "off_topic"
    PARTLY = "partly"
    FULLY = "fully"


class Correctness(StrEnum):
    WRONG = "wrong"
    PARTLY_RIGHT = "partly_right"
    RIGHT = "right"


class Depth(StrEnum):
    SURFACE = "surface"
    ADEQUATE = "adequate"
    DETAILED = "detailed"


class YesNo(StrEnum):
    YES = "yes"
    NO = "no"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    UNANSWERED = "unanswered"
    # The candidate declined or said they do not know. Tracked apart from a wrong answer.
    NO_ATTEMPT = "no_attempt"
    # Only the interviewer spoke after the question.
    SELF_ANSWERED = "self_answered"


class RubricSource(StrEnum):
    BANK = "bank"
    GENERATED = "generated"


# Ordinal order of each rating scale, lowest first. Used to tell "adjacent" from "diverging".
RELEVANCE_SCALE = (Relevance.OFF_TOPIC, Relevance.PARTLY, Relevance.FULLY)
CORRECTNESS_SCALE = (Correctness.WRONG, Correctness.PARTLY_RIGHT, Correctness.RIGHT)
DEPTH_SCALE = (Depth.SURFACE, Depth.ADEQUATE, Depth.DETAILED)


class Evidence(BaseModel):
    """Ids of the answer sentences that support each rating."""

    model_config = ConfigDict(extra="ignore")

    relevance: list[str] = Field(default_factory=list)
    correctness: list[str] = Field(default_factory=list)
    depth: list[str] = Field(default_factory=list)
    example_given: list[str] = Field(default_factory=list)

    def all_ids(self) -> set[str]:
        return {*self.relevance, *self.correctness, *self.depth, *self.example_given}


class Coverage(StrEnum):
    COVERED = "covered"
    PARTLY = "partly"
    NOT_COVERED = "not_covered"


class PointCoverage(BaseModel):
    """Whether the answer covers one expected point (numbered from 1), and where it shows."""

    model_config = ConfigDict(extra="ignore")

    point: int = Field(ge=1)
    coverage: Coverage
    evidence: list[str] = Field(default_factory=list)


class ModelAssessment(BaseModel):
    """The only shape a rater model may return: allowed values plus sentence ids, no free text.

    Correctness and depth are not asked of the model. They are derived in code from `points`.
    """

    model_config = ConfigDict(extra="ignore")

    relevance: Relevance
    points: list[PointCoverage]
    states_something_wrong: YesNo
    wrong_evidence: list[str] = Field(default_factory=list)
    example_given: YesNo
    example_evidence: list[str] = Field(default_factory=list)


class Rating(BaseModel):
    """The checked rating shown to the user, derived from a validated `ModelAssessment`."""

    model_config = ConfigDict(extra="ignore")

    relevance: Relevance
    correctness: Correctness
    depth: Depth
    example_given: YesNo
    evidence: Evidence
    # 0-10, derived in code from point coverage (see rating/derive.py). Not asked of the model.
    score: int = Field(ge=0, le=10)
    # Per expected point, in rubric order, so the page can show what each rating rests on.
    coverage: list[PointCoverage] = Field(default_factory=list)


class Rubric(BaseModel):
    points: list[str]
    source: RubricSource
    bank_id: str | None = None


class QuestionThread(BaseModel):
    id: str
    question_id: str
    topic: str | None = None
    topic_confidence: float | None = None
    answer_ids: list[str] = Field(default_factory=list)
    follow_up_ids: list[str] = Field(default_factory=list)
    # Questions the candidate asked back. Never rated and never a thread of their own.
    clarification_ids: list[str] = Field(default_factory=list)
    # The interviewer answering their own question.
    self_answer_ids: list[str] = Field(default_factory=list)
    status: AnswerStatus = AnswerStatus.UNANSWERED
    # A thread is closed once the next question arrives or the recording ends.
    closed: bool = False
    # The question was leading, so the answer is not clean evidence.
    primed: bool = False
    rubric: Rubric | None = None
    rating: Rating | None = None
    needs_review: bool = False
    review_reason: str | None = None
