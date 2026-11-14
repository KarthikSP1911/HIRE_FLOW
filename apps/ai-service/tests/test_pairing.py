from ai_service.assessment.pairing import pair_threads
from ai_service.schemas.assessment import AnswerStatus
from ai_service.schemas.transcript import Label, Sentence, Speaker

INT, CAND = Speaker.INTERVIEWER, Speaker.CANDIDATE


def sentences(*spec: tuple[Speaker, Label | None]) -> list[Sentence]:
    return [
        Sentence(
            id=f"s{n}",
            speaker=speaker,
            start=n,
            end=n + 1,
            text=f"text {n}",
            transcription_confidence=0.9,
            label=label,
        )
        for n, (speaker, label) in enumerate(spec, start=1)
    ]


def test_answers_attach_to_the_question_they_follow() -> None:
    threads = pair_threads(
        sentences((INT, Label.QUESTION), (CAND, Label.ANSWER), (CAND, Label.ANSWER)), final=True
    )

    assert len(threads) == 1
    assert threads[0].question_id == "s1"
    assert threads[0].answer_ids == ["s2", "s3"]
    assert threads[0].status is AnswerStatus.ANSWERED


def test_a_question_without_an_answer_is_unanswered() -> None:
    threads = pair_threads(sentences((INT, Label.QUESTION), (INT, Label.QUESTION)), final=True)

    assert [t.status for t in threads] == [AnswerStatus.UNANSWERED] * 2


def test_interviewer_answering_their_own_question_is_self_answered() -> None:
    threads = pair_threads(sentences((INT, Label.QUESTION), (INT, Label.ANSWER)), final=True)

    assert threads[0].status is AnswerStatus.SELF_ANSWERED
    assert threads[0].self_answer_ids == ["s2"]
    assert threads[0].answer_ids == []


def test_candidate_clarifying_question_stays_in_the_thread_and_is_not_an_answer() -> None:
    threads = pair_threads(
        sentences(
            (INT, Label.QUESTION),
            (CAND, Label.QUESTION),
            (INT, Label.OTHER),
            (CAND, Label.ANSWER),
        ),
        final=True,
    )

    assert len(threads) == 1
    assert threads[0].clarification_ids == ["s2"]
    assert threads[0].answer_ids == ["s4"]


def test_follow_up_belongs_to_the_thread_it_builds_on() -> None:
    threads = pair_threads(
        sentences(
            (INT, Label.QUESTION),
            (CAND, Label.ANSWER),
            (INT, Label.FOLLOW_UP),
            (CAND, Label.ANSWER),
        ),
        final=True,
    )

    assert len(threads) == 1
    assert threads[0].follow_up_ids == ["s3"]
    assert threads[0].answer_ids == ["s2", "s4"]


def test_only_the_last_thread_stays_open_until_final() -> None:
    spec = [
        (INT, Label.QUESTION),
        (CAND, Label.ANSWER),
        (INT, Label.QUESTION),
        (CAND, Label.ANSWER),
    ]

    live = pair_threads(sentences(*spec), final=False)
    done = pair_threads(sentences(*spec), final=True)

    assert [t.closed for t in live] == [True, False]
    assert [t.closed for t in done] == [True, True]


def test_an_unlabelled_sentence_flags_the_open_thread_for_review() -> None:
    threads = pair_threads(sentences((INT, Label.QUESTION), (CAND, None)), final=True)

    assert threads[0].needs_review
    assert threads[0].review_reason == "unlabelled sentence s2"


def test_sentences_before_any_question_are_ignored() -> None:
    assert pair_threads(sentences((CAND, Label.ANSWER), (INT, Label.FOLLOW_UP)), final=True) == []
