from functools import lru_cache
from pathlib import Path
from typing import Literal

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

    # Decision model
    labeller: Literal["strands", "laya"] = "strands"
    decider_url: str = "http://localhost:8100"
    laya_url: str = "http://localhost:8888"
    laya_model: str = "laya-multilingual"
    laya_api_key: str | None = None
    decision_timeout_seconds: float = 30.0
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
