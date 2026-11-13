import pytest

from ai_service.schemas.transcript import Label, Sentence, Speaker
from ai_service.services.verifier import Verifier

verifier = Verifier(confidence_threshold=0.6, stt_threshold=0.5)
KNOWN = {"t1.s1", "t2.s1"}


def sentence(speaker: Speaker = Speaker.INTERVIEWER, stt: float = 0.9) -> Sentence:
    return Sentence(
        id="t1.s1", speaker=speaker, start=0, end=1, text="hi", transcription_confidence=stt
    )


def raw(
    label: object = "question", confidence: object = 0.9, sid: object = "t1.s1"
) -> dict[str, object]:
    return {"sentence_id": sid, "label": label, "confidence": confidence}


def check(s: Sentence, r: dict[str, object]) -> Sentence:
    return verifier.apply(s, r, KNOWN, "fake")


def test_clean_output_passes() -> None:
    result = check(sentence(), raw())
    assert result.label == Label.QUESTION
    assert not result.needs_review


def test_unknown_sentence_id_is_flagged_and_label_dropped() -> None:
    result = check(sentence(), raw(sid="t9.s9"))
    assert result.needs_review and result.label is None
    assert "unknown sentence id" in (result.review_reason or "")


def test_label_outside_allowed_four_is_flagged() -> None:
    result = check(sentence(), raw(label="skill"))
    assert result.needs_review and result.label is None
    assert "not allowed" in (result.review_reason or "")


def test_schema_failure_is_flagged() -> None:
    result = check(sentence(), {"label": "question"})
    assert result.needs_review and "schema" in (result.review_reason or "")


def test_out_of_range_confidence_fails_schema() -> None:
    assert check(sentence(), raw(confidence=1.7)).needs_review


def test_low_label_confidence_is_flagged() -> None:
    result = check(sentence(), raw(confidence=0.3))
    assert result.needs_review and "low label confidence" in (result.review_reason or "")


def test_low_transcription_confidence_is_flagged() -> None:
    result = check(sentence(stt=0.2), raw())
    assert result.needs_review and "transcription" in (result.review_reason or "")


@pytest.mark.parametrize("label", ["question", "follow_up"])
def test_candidate_asking_is_flagged(label: str) -> None:
    assert check(sentence(Speaker.CANDIDATE), raw(label=label)).needs_review


def test_interviewer_answering_is_flagged() -> None:
    assert check(sentence(Speaker.INTERVIEWER), raw(label="answer")).needs_review


def test_candidate_answer_is_fine() -> None:
    assert not check(sentence(Speaker.CANDIDATE), raw(label="answer")).needs_review


def test_garbled_text_is_flagged() -> None:
    garbled = sentence().model_copy(update={"text": "My name \ufffd\ufffd"})
    result = check(garbled, raw())
    assert result.needs_review and "garbled" in (result.review_reason or "")
