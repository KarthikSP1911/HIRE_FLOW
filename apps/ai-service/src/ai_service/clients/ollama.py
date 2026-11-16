"""The one place that talks to Ollama: a chat call constrained to a JSON schema."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx


class OllamaError(RuntimeError):
    """Ollama could not be reached or returned something that is not a JSON object."""


@dataclass(frozen=True)
class TokenChoice:
    """One generated token and the most likely alternatives the model considered there."""

    token: str
    # Alternative token text -> log probability, before the JSON schema narrowed the choice.
    alternatives: Mapping[str, float]


@dataclass(frozen=True)
class ChatResult:
    content: dict[str, Any]
    # The raw generated text; the tokens below join back into exactly this.
    text: str
    tokens: Sequence[TokenChoice]


class OllamaChat:
    def __init__(self, client: httpx.AsyncClient, url: str, model: str) -> None:
        self.model = model
        self._client = client
        self._url = url.rstrip("/") + "/api/chat"

    async def chat(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        return (await self._post(messages, schema, top_logprobs=0)).content

    async def chat_with_logprobs(
        self, messages: list[dict[str, str]], schema: dict[str, Any], top_logprobs: int
    ) -> ChatResult:
        result = await self._post(messages, schema, top_logprobs)
        if not result.tokens:
            raise OllamaError(f"ollama:{self.model} returned no logprobs (needs Ollama >= 0.12)")
        return result

    async def _post(
        self, messages: list[dict[str, str]], schema: dict[str, Any], top_logprobs: int
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0},
        }
        if top_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs
        try:
            response = await self._client.post(self._url, json=payload)
            response.raise_for_status()
            body = response.json()
            text = body["message"]["content"]
            parsed = json.loads(text)
            tokens = [
                TokenChoice(
                    item["token"],
                    {alt["token"]: alt["logprob"] for alt in item.get("top_logprobs") or []},
                )
                for item in body.get("logprobs") or []
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise OllamaError(f"ollama:{self.model} unavailable or unusable") from exc
        if not isinstance(parsed, dict):
            raise OllamaError(f"ollama:{self.model} returned a non-object response")
        return ChatResult(parsed, text, tokens)
