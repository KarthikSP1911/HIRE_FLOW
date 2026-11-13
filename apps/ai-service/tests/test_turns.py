from ai_service.schemas.transcript import Segment, Speaker, SpeakerChange, Word
from ai_service.services.turns import build_turns


def seg(start: float, texts: list[str]) -> Segment:
    words = [
        Word(text=t, start=start + i, end=start + i + 1, probability=0.9)
        for i, t in enumerate(texts)
    ]
    return Segment(start=start, end=words[-1].end, text=" ".join(texts), words=words)


LOG = [
    SpeakerChange(speaker=Speaker.INTERVIEWER, at=0),
    SpeakerChange(speaker=Speaker.CANDIDATE, at=10),
]


def test_segments_are_assigned_by_speaker_log_and_merged() -> None:
    turns = build_turns([seg(0, ["Hi."]), seg(3, ["Why?"]), seg(10, ["Because."])], LOG)
    assert [(t.id, t.speaker) for t in turns] == [
        ("t1", Speaker.INTERVIEWER),
        ("t2", Speaker.CANDIDATE),
    ]
    assert [s.id for s in turns[0].sentences] == ["t1.s1", "t1.s2"]


def test_turn_is_split_into_sentences_with_timings() -> None:
    turn = build_turns([seg(0, [" Hello", "there.", " How", "are", "you?"])], LOG)[0]
    assert [s.text for s in turn.sentences] == ["Hello there.", "How are you?"]
    assert (turn.sentences[1].start, turn.sentences[1].end) == (2, 5)


def test_hindi_danda_ends_a_sentence() -> None:
    turn = build_turns([seg(0, ["namaste", "।", "kaise", "ho?"])], LOG)[0]
    assert len(turn.sentences) == 2


def test_no_segments_gives_no_turns() -> None:
    assert build_turns([], LOG) == []


def test_garbled_word_pieces_fall_back_to_clean_segment_text() -> None:
    garbled = Segment(
        start=0,
        end=2,
        text="मेरा नाम कार्तिक है।",
        words=[Word(text="\ufffd\ufffd", start=0, end=1, probability=0.4)],
    )
    sentences = build_turns([garbled], LOG)[0].sentences
    assert [s.text for s in sentences] == ["मेरा नाम कार्तिक है।"]
    assert sentences[0].transcription_confidence == 0.4
