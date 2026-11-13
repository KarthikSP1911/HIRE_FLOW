from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Speaker(StrEnum):
    INTERVIEWER = "interviewer"
    CANDIDATE = "candidate"


class Label(StrEnum):
    QUESTION = "question"
    ANSWER = "answer"
    FOLLOW_UP = "follow_up"
    OTHER = "other"


class SpeakerChange(BaseModel):
    """The given speaker took over `at` seconds after the recording started."""

    speaker: Speaker
    at: float = Field(ge=0)


class Word(BaseModel):
    text: str
    start: float
    end: float
    probability: float


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word]


class Transcription(BaseModel):
    language: str
    language_probability: float
    segments: list[Segment]


class Sentence(BaseModel):
    id: str
    speaker: Speaker
    start: float
    end: float
    text: str
    transcription_confidence: float
    label: Label | None = None
    label_confidence: float | None = None
    labeller: str | None = None
    needs_review: bool = False
    review_reason: str | None = None


class Turn(BaseModel):
    id: str
    speaker: Speaker
    start: float
    end: float
    sentences: list[Sentence]


class ProcessResponse(BaseModel):
    language: str
    language_probability: float
    labeller: str
    turns: list[Turn]


class Decision(BaseModel):
    """The shape every labeller output must satisfy before it is looked at further."""

    model_config = ConfigDict(extra="ignore")

    sentence_id: str
    label: str
    confidence: float = Field(ge=0, le=1)
