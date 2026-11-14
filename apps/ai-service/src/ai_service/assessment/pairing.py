"""Link answers to the question they respond to. Plain code over labelled sentences."""

from collections.abc import Sequence

from ai_service.schemas.assessment import AnswerStatus, QuestionThread
from ai_service.schemas.transcript import Label, Sentence, Speaker


def pair_threads(sentences: Sequence[Sentence], final: bool) -> list[QuestionThread]:
    """Build one thread per interviewer question, in order.

    A follow-up joins the open thread. A candidate's question is a clarification inside the
    thread, not a new one. Interviewer sentences labelled answer are the interviewer answering
    themselves. Sentences without a label are skipped and flagged on the open thread.
    """
    threads: list[QuestionThread] = []
    for sentence in sentences:
        open_thread = threads[-1] if threads else None
        if sentence.label is None:
            if open_thread is not None:
                open_thread.needs_review = True
                open_thread.review_reason = _join(
                    open_thread.review_reason, f"unlabelled sentence {sentence.id}"
                )
            continue
        if sentence.label is Label.QUESTION and sentence.speaker is Speaker.INTERVIEWER:
            threads.append(QuestionThread(id=f"q{len(threads) + 1}", question_id=sentence.id))
        elif open_thread is None:
            continue
        elif sentence.label is Label.QUESTION:
            open_thread.clarification_ids.append(sentence.id)
        elif sentence.label is Label.FOLLOW_UP and sentence.speaker is Speaker.INTERVIEWER:
            open_thread.follow_up_ids.append(sentence.id)
        elif sentence.label is Label.ANSWER and sentence.speaker is Speaker.CANDIDATE:
            open_thread.answer_ids.append(sentence.id)
        elif sentence.label is Label.ANSWER:
            open_thread.self_answer_ids.append(sentence.id)

    for index, thread in enumerate(threads):
        thread.closed = final or index < len(threads) - 1
        thread.status = _status(thread)
    return threads


def _status(thread: QuestionThread) -> AnswerStatus:
    if thread.answer_ids:
        return AnswerStatus.ANSWERED
    if thread.self_answer_ids:
        return AnswerStatus.SELF_ANSWERED
    return AnswerStatus.UNANSWERED


def _join(existing: str | None, reason: str) -> str:
    return f"{existing}; {reason}" if existing else reason
