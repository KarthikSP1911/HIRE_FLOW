"""Speech-to-text. Self-contained so it can move to apps/capture later."""

import logging
import os
import tempfile
from typing import Protocol

from faster_whisper import WhisperModel

from ai_service.schemas.transcript import Segment, Transcription, Word

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Audio could not be transcribed."""


class Transcriber(Protocol):
    def transcribe(self, audio: bytes, suffix: str) -> Transcription: ...


def _pick_device() -> tuple[str, str]:
    import ctranslate2

    if ctranslate2.get_cuda_device_count() > 0:
        return "cuda", "float16"
    return "cpu", "int8"


class FasterWhisperTranscriber:
    def __init__(self, model_size: str, device: str = "auto", beam_size: int = 1) -> None:
        self._model_size = model_size
        self._beam_size = beam_size
        self._device = device
        self._model: WhisperModel | None = None

    def _load(self) -> WhisperModel:
        # Loaded lazily so the app starts fast and tests never download weights.
        if self._model is None:
            device, compute_type = (
                _pick_device() if self._device == "auto" else (self._device, "int8")
            )
            self._model = WhisperModel(self._model_size, device=device, compute_type=compute_type)
        return self._model

    def transcribe(self, audio: bytes, suffix: str) -> Transcription:
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(audio)
            segments, info = self._load().transcribe(
                path,
                word_timestamps=True,
                vad_filter=True,
                language=None,
                beam_size=self._beam_size,
            )
            return Transcription(
                language=info.language,
                language_probability=info.language_probability,
                segments=[
                    Segment(
                        start=s.start,
                        end=s.end,
                        text=s.text.strip(),
                        words=[
                            Word(text=w.word, start=w.start, end=w.end, probability=w.probability)
                            for w in (s.words or [])
                        ],
                    )
                    for s in segments
                ],
            )
        except Exception as exc:
            # Log the cause only, never the audio or transcript (PII).
            logger.exception("transcription failed")
            raise TranscriptionError("transcription failed") from exc
        finally:
            # Candidate audio is PII: never keep it, even when transcription fails.
            os.unlink(path)
