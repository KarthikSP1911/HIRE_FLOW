"""Per-session results, so a finished question is judged and rated once, not on every chunk."""

from dataclasses import dataclass, field

from ai_service.assessment.judgements import Judgement
from ai_service.schemas.assessment import QuestionThread

# Keyed by text as well as id: ids are positional and can shift when a sentence grows.
JudgementKey = tuple[str, str, str]  # (kind, question id, text judged)
ThreadKey = tuple[str, str, tuple[tuple[str, str], ...]]  # (question id, question, members)


@dataclass
class AssessmentCache:
    judgements: dict[JudgementKey, Judgement] = field(default_factory=dict)
    drafts: dict[str, list[str]] = field(default_factory=dict)
    threads: dict[ThreadKey, QuestionThread] = field(default_factory=dict)
