"""Adapters for decision models that speak the `/v1/systemone` choice protocol."""

from collections.abc import Mapping, Sequence

import httpx

from ai_service.labelling.port import DecisionError, RawChoice, RawDecision
from ai_service.schemas.transcript import Label, Sentence

QUESTION_NAME = "sentence_type"
INSTRUCTIONS = "What role does the last sentence play in this interview?"
CRITERIA = {
    Label.QUESTION: "the interviewer asks something new",
    Label.ANSWER: "the candidate responds to a question",
    Label.FOLLOW_UP: "the interviewer asks something building on the candidate's previous answer",
    Label.OTHER: "filler or acknowledgement such as okay, thanks, let me think",
}


def build_state(sentence: Sentence, context: Sequence[Sentence]) -> str:
    history = "\n".join(f"[{s.speaker}] {s.text}" for s in context)
    return f"Earlier:\n{history}\n\nSentence to label:\n[{sentence.speaker}] {sentence.text}"


def _answer_node(body: Mapping[str, object], question_name: str) -> Mapping[str, object]:
    answers = body.get("answers")
    node = answers.get(question_name) if isinstance(answers, dict) else None
    return node if isinstance(node, dict) else body


def read_choice(body: Mapping[str, object], question_name: str) -> RawChoice:
    node = _answer_node(body, question_name)
    choice = node.get("choice")
    probabilities = node.get("probabilities")
    confidence = node.get("confidence")
    if isinstance(probabilities, dict) and isinstance(choice, str) and choice in probabilities:
        confidence = probabilities[choice]
    return {"choice": choice, "confidence": confidence}


def parse_choice(sentence_id: str, body: Mapping[str, object]) -> RawDecision:
    choice = read_choice(body, QUESTION_NAME)
    return {
        "sentence_id": sentence_id,
        "label": choice["choice"],
        "confidence": choice["confidence"],
    }


class SystemOneLabeller:
    def __init__(
        self,
        name: str,
        client: httpx.AsyncClient,
        url: str,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.name = name
        self._client = client
        self._url = url.rstrip("/") + "/v1/systemone"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def decide(self, sentence: Sentence, context: Sequence[Sentence]) -> RawDecision:
        body = await self._ask(
            build_state(sentence, context),
            QUESTION_NAME,
            INSTRUCTIONS,
            {label.value: text for label, text in CRITERIA.items()},
        )
        return parse_choice(sentence.id, body)

    async def choose(
        self, state: str, question_name: str, instructions: str, criteria: Mapping[str, str]
    ) -> RawChoice:
        return read_choice(
            await self._ask(state, question_name, instructions, criteria), question_name
        )

    async def _ask(
        self, state: str, question_name: str, instructions: str, criteria: Mapping[str, str]
    ) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "state": state,
            "questions": {
                question_name: {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": dict(criteria),
                }
            },
        }
        if self._model:
            payload["model"] = self._model
        try:
            response = await self._client.post(self._url, json=payload, headers=self._headers)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DecisionError(f"{self.name} decision server unavailable") from exc
        if not isinstance(body, dict):
            raise DecisionError(f"{self.name} returned a non-object response")
        return body
