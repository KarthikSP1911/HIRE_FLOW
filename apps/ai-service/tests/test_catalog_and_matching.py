import json
from pathlib import Path

import pytest

from ai_service.assessment.catalog import BankEntry, load_catalog
from ai_service.assessment.matching import TokenSetMatcher
from ai_service.core.config import CONFIG_DIR

TOPICS = [{"id": "algorithms", "label": "Algorithms", "description": "d"}]


def entry(question: str) -> BankEntry:
    return BankEntry(
        id="qb-1",
        question=question,
        topic="algorithms",
        expected_points=["one concept", "another concept"],
        difficulty="easy",
    )


def test_shipped_config_loads_with_eight_topics() -> None:
    catalog = load_catalog(CONFIG_DIR / "topics.json", CONFIG_DIR / "question_bank.json")

    assert len(catalog.topics) == 8
    assert catalog.bank


def test_bank_entry_with_an_unknown_topic_fails_fast(tmp_path: Path) -> None:
    topics = tmp_path / "topics.json"
    bank = tmp_path / "bank.json"
    topics.write_text(json.dumps(TOPICS))
    bank.write_text(
        json.dumps(
            [
                {
                    "id": "x",
                    "question": "q",
                    "topic": "astrology",
                    "expected_points": ["a", "b"],
                    "difficulty": "easy",
                }
            ]
        )
    )

    with pytest.raises(ValueError, match="astrology"):
        load_catalog(topics, bank)


def test_bank_entry_needs_two_to_four_points() -> None:
    with pytest.raises(ValueError):
        BankEntry(id="x", question="q", topic="t", expected_points=["only one"], difficulty="easy")


def test_spoken_phrasing_still_matches_the_bank_entry() -> None:
    matcher = TokenSetMatcher([entry("Explain how binary search works.")], threshold=0.5)

    match = matcher.match("Okay, can you explain how binary search works?")

    assert match is not None
    assert match.id == "qb-1"


def test_an_unrelated_question_matches_nothing() -> None:
    matcher = TokenSetMatcher([entry("Explain how binary search works.")], threshold=0.5)

    assert matcher.match("Tell me about your last team") is None
