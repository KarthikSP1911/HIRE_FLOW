"""Plain-code checks on model output. Nothing a model says is trusted unchecked."""

from collections.abc import Mapping

from pydantic import ValidationError

from ai_service.schemas.transcript import Decision, Label, Sentence, Speaker

NOT_ALLOWED_FOR = {
    Speaker.CANDIDATE: {Label.QUESTION, Label.FOLLOW_UP},
    Speaker.INTERVIEWER: {Label.ANSWER},
}


class Verifier:
    def __init__(self, confidence_threshold: float, stt_threshold: float) -> None:
        self._confidence_threshold = confidence_threshold
        self._stt_threshold = stt_threshold

    def apply(
        self, sentence: Sentence, raw: Mapping[str, object], known_ids: set[str], labeller: str
    ) -> Sentence:
        label, confidence, problems = self._check(sentence, raw, known_ids)
        return sentence.model_copy(
            update={
                "label": label,
                "label_confidence": confidence,
                "labeller": labeller,
                "needs_review": bool(problems),
                "review_reason": "; ".join(problems) or None,
            }
        )

    def _check(
        self, sentence: Sentence, raw: Mapping[str, object], known_ids: set[str]
    ) -> tuple[Label | None, float | None, list[str]]:
        problems: list[str] = []
        if "\ufffd" in sentence.text:
            problems.append("garbled transcription")
        if sentence.transcription_confidence < self._stt_threshold:
            problems.append("low transcription confidence")

        try:
            decision = Decision.model_validate(raw)
        except ValidationError:
            return None, None, [*problems, "model output failed schema validation"]

        if decision.sentence_id not in known_ids:
            return None, None, [*problems, f"unknown sentence id {decision.sentence_id}"]
        if decision.sentence_id != sentence.id:
            return (
                None,
                None,
                [*problems, f"label cites a different sentence ({decision.sentence_id})"],
            )

        try:
            label = Label(decision.label)
        except ValueError:
            return (
                None,
                decision.confidence,
                [*problems, f"label '{decision.label}' is not allowed"],
            )

        if decision.confidence < self._confidence_threshold:
            problems.append("low label confidence")
        if label in NOT_ALLOWED_FOR[sentence.speaker]:
            problems.append(f"{sentence.speaker} sentence labelled {label}")
        return label, decision.confidence, problems
