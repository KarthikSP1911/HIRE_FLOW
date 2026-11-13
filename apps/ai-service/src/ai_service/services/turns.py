"""Speaker turns and sentences from transcribed segments."""

from ai_service.schemas.transcript import (
    Segment,
    Sentence,
    Speaker,
    SpeakerChange,
    Turn,
    Word,
)

SENTENCE_ENDINGS = (".", "?", "!", "।")
REPLACEMENT_CHAR = "\ufffd"


def speaker_at(time: float, log: list[SpeakerChange]) -> Speaker:
    current = log[0].speaker if log else Speaker.INTERVIEWER
    for change in sorted(log, key=lambda c: c.at):
        if change.at <= time:
            current = change.speaker
    return current


def _words(segment: Segment) -> list[Word]:
    # Word pieces of Hindi/Kannada etc. can split a character's bytes and decode to U+FFFD,
    # so for those (or when no timings came back) use the segment's own clean text as one token.
    if segment.words and not any(REPLACEMENT_CHAR in w.text for w in segment.words):
        return segment.words
    confidence = (
        sum(w.probability for w in segment.words) / len(segment.words) if segment.words else 1.0
    )
    return [Word(text=segment.text, start=segment.start, end=segment.end, probability=confidence)]


def _split_sentences(turn_id: str, speaker: Speaker, words: list[Word]) -> list[Sentence]:
    sentences: list[Sentence] = []
    pending: list[Word] = []

    def flush() -> None:
        if not pending:
            return
        sentences.append(
            Sentence(
                id=f"{turn_id}.s{len(sentences) + 1}",
                speaker=speaker,
                start=pending[0].start,
                end=pending[-1].end,
                text=" ".join(w.text.strip() for w in pending),
                transcription_confidence=sum(w.probability for w in pending) / len(pending),
            )
        )
        pending.clear()

    for word in words:
        pending.append(word)
        if word.text.strip().endswith(SENTENCE_ENDINGS):
            flush()
    flush()
    return sentences


def build_turns(segments: list[Segment], log: list[SpeakerChange]) -> list[Turn]:
    grouped: list[tuple[Speaker, list[Segment]]] = []
    for segment in segments:
        speaker = speaker_at(segment.start, log)
        if grouped and grouped[-1][0] == speaker:
            grouped[-1][1].append(segment)
        else:
            grouped.append((speaker, [segment]))

    turns: list[Turn] = []
    for index, (speaker, group) in enumerate(grouped, start=1):
        turn_id = f"t{index}"
        words = [w for segment in group for w in _words(segment)]
        turns.append(
            Turn(
                id=turn_id,
                speaker=speaker,
                start=group[0].start,
                end=group[-1].end,
                sentences=_split_sentences(turn_id, speaker, words),
            )
        )
    return turns
