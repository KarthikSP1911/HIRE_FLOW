from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_service.rating.consensus import PromptMode

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class Settings(BaseSettings):
    """Application settings loaded from environment variables (and `.env` in local dev)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ai-service"
    environment: str = "local"
    log_level: str = "INFO"

    # Speech to text
    whisper_model: str = "small"
    whisper_device: str = "auto"
    # Greedy decoding (1) is several times faster than beam search and fine for short live slices.
    whisper_beam_size: int = 1
    max_audio_mb: int = 25

    # Tagging (sentence labels, topic, attempt, leading) runs on Ollama, see below.
    # Unset means OLLAMA_MODEL; a smaller model keeps live labelling fast.
    tagging_model: str | None = None
    context_sentences: int = 4

    # Verifier thresholds
    confidence_threshold: float = 0.6
    stt_low_confidence: float = 0.7

    # Topics, question bank and answer rating
    topics_path: Path = CONFIG_DIR / "topics.json"
    question_bank_path: Path = CONFIG_DIR / "question_bank.json"
    topic_confidence_threshold: float = 0.6
    bank_match_threshold: float = 0.5
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    # Generous: local models on CPU can take a long time per answer.
    ollama_timeout_seconds: float = 180.0
    rating_prompt_mode: PromptMode = PromptMode.DUAL

    @property
    def effective_tagging_model(self) -> str:
        return self.tagging_model or self.ollama_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
