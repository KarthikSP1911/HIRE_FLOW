"""Tag sentences and answer single-choice questions with a local model served by Ollama.

Output is kept as short as possible because, on CPU, every generated token costs ~0.1 s:
labels come back as `{"1": "<label>", "2": ...}`, numbered in prompt order and mapped back to
sentence ids in code, and choices as `{"choice": "<value>"}`. The JSON schema forces exactly
those keys and the offered values. Confidence is read from token probabilities (see
confidence.py), not written by the model. Everything is checked again in code.

Live labelling is one conversation per recording: each batch is a new message after the model's
earlier replies, so Ollama only reads the new sentences instead of the whole prompt again.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import Any, Final

import httpx

from ai_service.clients.ollama import ChatResult, OllamaChat, OllamaError
from ai_service.labelling.confidence import choice_confidence, value_offsets
from ai_service.labelling.port import RawChoice, RawDecision, TaggingError
from ai_service.labelling.prompts import LABEL_SYSTEM, choice_prompt, label_keys, label_turn
from ai_service.schemas.transcript import Label, Sentence

CHOICE_KEY: Final = "choice"
# Enough alternatives to cover every allowed value plus casing/spacing variants.
TOP_LOGPROBS: Final = 10
LABELS: Final = tuple(label.value for label in Label)
# About 2,000 tokens of history, well inside Ollama's default 4,096-token context. Past it the
# conversation restarts with a few earlier sentences restated (one slower call).
MAX_HISTORY_CHARS: Final = 8_000

Message = dict[str, str]


def _enum(values: Collection[str]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def labels_schema(targets: Sequence[Sentence]) -> dict[str, Any]:
    keys = label_keys(targets)
    return {
        "type": "object",
        "properties": {key: _enum(LABELS) for key in keys},
        "required": keys,
    }


def choice_schema(criteria: Mapping[str, str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {CHOICE_KEY: _enum(criteria)},
        "required": [CHOICE_KEY],
    }


def _confidence(result: ChatResult, key: str, chosen: object, allowed: Collection[str]) -> object:
    offset = value_offsets(result.text).get(key)
    if offset is None or not isinstance(chosen, str):
        return None  # the verifier rejects the missing number
    return choice_confidence(result.tokens, result.text, offset, chosen, allowed)


async def _ask(
    ollama: OllamaChat, name: str, messages: list[Message], schema: dict[str, Any]
) -> ChatResult:
    try:
        return await ollama.chat_with_logprobs(messages, schema, TOP_LOGPROBS)
    except OllamaError as exc:
        raise TaggingError(f"{name} tagging model unavailable or unusable") from exc


class OllamaLabelConversation:
    def __init__(self, ollama: OllamaChat, name: str, context_size: int) -> None:
        self._ollama = ollama
        self._name = name
        self._context_size = context_size
        self._turns: list[Message] = []
        self._seen: list[Sentence] = []

    async def label(self, targets: Sequence[Sentence]) -> Sequence[RawDecision]:
        # A restarted conversation restates the last few sentences so context is not lost.
        earlier = self._seen[-self._context_size :] if not self._turns else []
        user = {"role": "user", "content": label_turn(targets, earlier)}
        system = {"role": "system", "content": LABEL_SYSTEM}
        result = await _ask(
            self._ollama, self._name, [system, *self._turns, user], labels_schema(targets)
        )
        # Replay the reply exactly as generated, so the next call extends this one token for
        # token and Ollama can reuse its cache.
        self._turns += [user, {"role": "assistant", "content": result.text}]
        self._seen += targets
        if sum(len(turn["content"]) for turn in self._turns) > MAX_HISTORY_CHARS:
            self._turns = []

        ids = dict(zip(label_keys(targets), (s.id for s in targets), strict=True))
        return [
            {
                # An unknown key stays as is, so the pipeline sees no label for any target.
                "sentence_id": ids.get(key, key),
                "label": label,
                "confidence": _confidence(result, key, label, LABELS),
            }
            for key, label in result.content.items()
        ]


class OllamaTagger:
    """Serves both the SentenceTagger and the ChoicePort."""

    def __init__(self, client: httpx.AsyncClient, url: str, model: str) -> None:
        self.name = f"ollama:{model}"
        self._ollama = OllamaChat(client, url, model)

    def start_conversation(self, context_size: int) -> OllamaLabelConversation:
        return OllamaLabelConversation(self._ollama, self.name, context_size)

    async def choose(
        self, state: str, question_name: str, instructions: str, criteria: Mapping[str, str]
    ) -> RawChoice:
        prompt = choice_prompt(state, instructions, criteria)
        result = await _ask(
            self._ollama, self.name, [{"role": "user", "content": prompt}], choice_schema(criteria)
        )
        chosen = result.content.get(CHOICE_KEY)
        return {
            "choice": chosen,
            "confidence": _confidence(result, CHOICE_KEY, chosen, list(criteria)),
        }
