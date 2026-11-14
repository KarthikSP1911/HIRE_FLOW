"""The two differently framed rating prompts. Both share the same ground rules."""

from ai_service.rating.port import PromptStyle, RatingRequest

# Kept short on purpose: on CPU every prompt token costs time, and the rules are re-read per call.
_RULES = """Check an interview answer against numbered expected points. Reply with JSON only.
- Per point: covered, partly or not_covered, with the ids of the sentences that show it. Judge
  the idea, not the wording: plain everyday language can cover a point, technical terms without
  correct reasoning cannot.
- states_something_wrong: yes if the answer asserts anything factually incorrect, even
  confidently (cite ids in wrong_evidence).
- relevance: fully, partly or off_topic.
- example_given: yes only for a concrete example (cite ids in example_evidence).
- Cite only the listed ids."""

_FRAMING = {
    PromptStyle.CONCEPT_FIRST: "Ignore wording: ask if the idea itself is present and correct.",
    PromptStyle.TERMINOLOGY_AWARE: (
        "Notice the technical terms used and check each is used correctly, but stay fair to "
        "answers that reach the right idea without them."
    ),
}

DRAFT_POINTS_PROMPT = (
    "List 2 to 4 short expected concept points a good answer to this interview question should "
    "cover. State concepts, not keywords. Reply only with the requested JSON.\n\nQuestion: "
)


def system_prompt(style: PromptStyle) -> str:
    return f"{_RULES}\n\n{_FRAMING[style]}"


def user_prompt(request: RatingRequest) -> str:
    points = "\n".join(f"{n}. {point}" for n, point in enumerate(request.points, start=1))
    answer = "\n".join(f"[{sentence.id}] {sentence.text}" for sentence in request.answers)
    return (
        f"Question:\n{request.question}\n\nExpected points:\n{points}\n\n"
        f"Candidate answer, by sentence id:\n{answer}"
    )
