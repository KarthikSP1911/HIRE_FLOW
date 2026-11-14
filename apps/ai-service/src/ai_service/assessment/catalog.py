"""The topic list and question bank, loaded once from JSON config and validated up front."""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

MIN_POINTS = 2
MAX_POINTS = 4


class Topic(BaseModel, frozen=True):
    id: str
    label: str
    description: str


class BankEntry(BaseModel, frozen=True):
    id: str
    question: str
    topic: str
    expected_points: list[str] = Field(min_length=MIN_POINTS, max_length=MAX_POINTS)
    difficulty: str


@dataclass(frozen=True)
class Catalog:
    topics: tuple[Topic, ...]
    bank: tuple[BankEntry, ...]

    @property
    def topic_ids(self) -> set[str]:
        return {t.id for t in self.topics}


def load_catalog(topics_path: Path, bank_path: Path) -> Catalog:
    """Fail fast at startup if the config is malformed or the bank cites an unknown topic."""
    topics = tuple(TypeAdapter(list[Topic]).validate_python(json.loads(topics_path.read_text())))
    bank = tuple(TypeAdapter(list[BankEntry]).validate_python(json.loads(bank_path.read_text())))
    catalog = Catalog(topics, bank)
    unknown = {e.topic for e in bank} - catalog.topic_ids
    if unknown:
        raise ValueError(
            f"question bank uses topics missing from the topic list: {sorted(unknown)}"
        )
    return catalog
