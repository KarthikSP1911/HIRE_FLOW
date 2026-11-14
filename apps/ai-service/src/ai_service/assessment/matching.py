"""Match a spoken question to a question-bank entry by similarity."""

import re
from collections.abc import Sequence
from typing import Protocol

from ai_service.assessment.catalog import BankEntry

# Words that carry no topic: spoken phrasing around the actual question.
_FILLER = frozenset(
    "a an the you your can could would please me us we i is are was do does did of to and or "
    "so now tell what how why about with for in on it that this let lets next".split()
)


class QuestionMatcher(Protocol):
    def match(self, question: str) -> BankEntry | None: ...


def _tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", text.lower())) - _FILLER


class TokenSetMatcher:
    """Lexical Jaccard similarity. Swap in an embedding matcher behind the same Protocol."""

    def __init__(self, bank: Sequence[BankEntry], threshold: float) -> None:
        self._entries = [(entry, _tokens(entry.question)) for entry in bank]
        self._threshold = threshold

    def match(self, question: str) -> BankEntry | None:
        spoken = _tokens(question)
        best: BankEntry | None = None
        best_score = 0.0
        for entry, tokens in self._entries:
            union = spoken | tokens
            score = len(spoken & tokens) / len(union) if union else 0.0
            if score > best_score:
                best, best_score = entry, score
        return best if best_score >= self._threshold else None
