"""Prompts for the tagging model.

Kept short: on CPU a prompt token costs ~50 ms to read unless Ollama can reuse it from the
previous call. With hybrid models such as Qwen3.5 that only works when the new prompt extends
the previous one exactly, so live labelling is one growing conversation (see ollama.py).
"""

from collections.abc import Mapping, Sequence

from ai_service.schemas.transcript import Label, Sentence

LABEL_CRITERIA: Mapping[Label, str] = {
    Label.QUESTION: "the interviewer asks something new",
    Label.ANSWER: "the candidate responds to a question",
    Label.FOLLOW_UP: "the interviewer asks something building on the candidate's previous answer",
    Label.OTHER: "filler or acknowledgement such as okay, thanks, let me think",
}

_JSON_ONLY = "Reply with JSON only."


def _criteria_lines(criteria: Mapping[str, str]) -> str:
    return "\n".join(f"- {key}: {text}" for key, text in criteria.items())


LABEL_SYSTEM = (
    "Label the role each numbered sentence plays in this interview, keyed by its number.\n"
    f"{_criteria_lines({label.value: text for label, text in LABEL_CRITERIA.items()})}\n"
    "Each message brings the next sentences of the same interview; earlier messages are "
    f"context.\n{_JSON_ONLY}"
)


def label_keys(targets: Sequence[Sentence]) -> list[str]:
    """Short keys ("1", "2", ...) instead of sentence ids: every generated token costs ~0.1 s."""
    return [str(n) for n in range(1, len(targets) + 1)]


def label_turn(targets: Sequence[Sentence], earlier: Sequence[Sentence] = ()) -> str:
    """One user message. `earlier` restates context only when a conversation restarts."""
    to_label = "\n".join(
        f"[{key}] {s.speaker}: {s.text}"
        for key, s in zip(label_keys(targets), targets, strict=True)
    )
    if not earlier:
        return to_label
    context = "\n".join(f"{s.speaker}: {s.text}" for s in earlier)
    return f"Earlier:\n{context}\n\nSentences to label:\n{to_label}"


def choice_prompt(state: str, instructions: str, criteria: Mapping[str, str]) -> str:
    criteria_lines = _criteria_lines(criteria)
    return f"{instructions} Pick one choice.\n{criteria_lines}\n{_JSON_ONLY}\n\n{state}"
