import math

import pytest

from ai_service.clients.ollama import TokenChoice
from ai_service.labelling.confidence import UNMEASURED, choice_confidence, value_offsets

ALLOWED = ["question", "answer", "follow_up", "other"]


def tokens(*pieces: tuple[str, dict[str, float]]) -> list[TokenChoice]:
    return [TokenChoice(text, {t: math.log(p) for t, p in alts.items()}) for text, alts in pieces]


def test_value_offsets_point_at_each_value_in_generated_order() -> None:
    text = '{"t1.s1": "question", "t1.s2": "other"}'
    offsets = value_offsets(text)
    assert list(offsets) == ["t1.s1", "t1.s2"]
    assert text[offsets["t1.s1"] :].startswith("question")
    assert text[offsets["t1.s2"] :].startswith("other")


def test_probability_is_renormalised_over_allowed_values_only() -> None:
    text = '{"a": "question"}'
    # "What" and "int" are not allowed values, so their mass is dropped; casing still counts.
    toks = tokens(
        ('{"a": "', {}),
        ("question", {"What": 0.4, "question": 0.3, "Question": 0.1, "other": 0.1, "int": 0.1}),
        ('"}', {}),
    )
    offset = value_offsets(text)["a"]
    assert choice_confidence(toks, text, offset, "question", ALLOWED) == pytest.approx(0.8)


def test_a_partial_token_counts_toward_the_value_it_begins() -> None:
    text = '{"a": "follow_up"}'
    toks = tokens(('{"a": "', {}), ("follow", {"follow": 0.5, "question": 0.5}), ('_up"}', {}))
    offset = value_offsets(text)["a"]
    assert choice_confidence(toks, text, offset, "follow_up", ALLOWED) == pytest.approx(0.5)


def test_a_token_that_includes_the_opening_quote_is_handled() -> None:
    text = '{"a": "other"}'
    toks = tokens(('{"a": ', {}), ('"other', {'"other': 0.6, '"answer': 0.2, "x": 0.2}), ('"}', {}))
    offset = value_offsets(text)["a"]
    assert choice_confidence(toks, text, offset, "other", ALLOWED) == pytest.approx(0.75)


def test_values_sharing_a_first_token_are_told_apart_by_later_tokens() -> None:
    topics = ["databases", "data-structures", "networking"]
    text = '{"choice": "data-structures"}'
    toks = tokens(
        ('{"choice": "', {}),
        ("data", {"data": 0.8, "network": 0.2}),  # 0.8 shared by both "data..." topics
        ("-structures", {"-structures": 0.75, "bases": 0.25}),
        ('"}', {}),
    )
    offset = value_offsets(text)["choice"]
    confidence = choice_confidence(toks, text, offset, "data-structures", topics)
    assert confidence == pytest.approx(0.8 * 0.75)


def test_no_allowed_alternatives_means_unmeasured() -> None:
    text = '{"a": "other"}'
    toks = tokens(('{"a": "', {}), ("other", {"Sure": 1.0}), ('"}', {}))
    offset = value_offsets(text)["a"]
    assert choice_confidence(toks, text, offset, "other", ALLOWED) == UNMEASURED
