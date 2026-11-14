"""Rate answers with a local model served by Ollama, constrained to a fixed JSON schema."""

import json
from typing import Any

import httpx

from ai_service.rating.port import PromptStyle, RatingError, RatingRequest, RawRating
from ai_service.rating.prompts import DRAFT_POINTS_PROMPT, system_prompt, user_prompt
from ai_service.schemas.assessment import Coverage, Relevance, YesNo

MAX_DRAFT_POINTS = 4


def _enum(values: type[Relevance] | type[Coverage] | type[YesNo]) -> object:
    return {"type": "string", "enum": [v.value for v in values]}


_IDS = {"type": "array", "items": {"type": "string"}}

# Only allowed values, point numbers and sentence ids: there is no field for prose.
RATING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relevance": _enum(Relevance),
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "integer", "minimum": 1},
                    "coverage": _enum(Coverage),
                    "evidence": _IDS,
                },
                "required": ["point", "coverage", "evidence"],
            },
        },
        "states_something_wrong": _enum(YesNo),
        "wrong_evidence": _IDS,
        "example_given": _enum(YesNo),
        "example_evidence": _IDS,
    },
    "required": [
        "relevance",
        "points",
        "states_something_wrong",
        "wrong_evidence",
        "example_given",
        "example_evidence",
    ],
}

POINTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": MAX_DRAFT_POINTS,
        }
    },
    "required": ["points"],
}


class OllamaRater:
    def __init__(self, client: httpx.AsyncClient, url: str, model: str) -> None:
        self.name = f"ollama:{model}"
        self._client = client
        self._url = url.rstrip("/") + "/api/chat"
        self._model = model

    async def draft_points(self, question: str) -> list[str]:
        body = await self._chat(
            [{"role": "user", "content": DRAFT_POINTS_PROMPT + question}], POINTS_SCHEMA
        )
        points = body.get("points")
        if not isinstance(points, list) or not all(
            isinstance(p, str) and p.strip() for p in points
        ):
            raise RatingError(f"{self.name} returned unusable expected points")
        return [p.strip() for p in points][:MAX_DRAFT_POINTS]

    async def rate(self, request: RatingRequest, style: PromptStyle) -> RawRating:
        messages = [
            {"role": "system", "content": system_prompt(style)},
            {"role": "user", "content": user_prompt(request)},
        ]
        return await self._chat(messages, RATING_SCHEMA)

    async def _chat(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0},
        }
        try:
            response = await self._client.post(self._url, json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            parsed = json.loads(content)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise RatingError(f"{self.name} rating server unavailable or unusable") from exc
        if not isinstance(parsed, dict):
            raise RatingError(f"{self.name} returned a non-object response")
        return parsed
