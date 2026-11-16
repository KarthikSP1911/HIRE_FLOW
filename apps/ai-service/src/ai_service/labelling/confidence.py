"""Confidence from token probabilities instead of a number the model writes about itself.

At the token where a JSON string value starts, Ollama reports the model's most likely next
tokens before the schema narrowed them. Keeping only the alternatives that begin one of the
allowed values and renormalising gives the probability the model put on its choice among them.
Writing the value costs no extra generated tokens.
"""

import math
import re
from collections.abc import Collection, Sequence

from ai_service.clients.ollama import TokenChoice

# A `"key": "value"` pair in the generated JSON. Keys and values never contain quotes here:
# both are sentence ids or fixed choice names.
_PAIR = re.compile(r'"(?P<key>[^"]+)"\s*:\s*"(?P<value>[^"]*)"')

# The model could not be measured: low enough that the verifier always flags it.
UNMEASURED = 0.0


def value_offsets(text: str) -> dict[str, int]:
    """Character offset where each key's string value starts, in generated order."""
    return {m["key"]: m.start("value") for m in _PAIR.finditer(text)}


def _tokens_from(tokens: Sequence[TokenChoice], offset: int) -> list[tuple[TokenChoice, int]]:
    """The token covering `offset` and every token after it, each with its start offset."""
    found: list[tuple[TokenChoice, int]] = []
    position = 0
    for token in tokens:
        end = position + len(token.token)
        if end > offset:
            found.append((token, position))
        position = end
    return found


def _fits(piece: str, remaining: str) -> bool:
    return remaining.startswith(piece) or piece.startswith(remaining)


def _step(token: TokenChoice, lead: str, chosen_rest: str, rests: list[str]) -> float | None:
    """Share of one token's probability that continues the chosen value.

    `lead` is text the token carries before the value's next character (such as the opening
    quote); `rests` are the not-yet-generated parts of every still-possible value. Each
    alternative counts once, even when it fits several values. None if nothing fits.
    """
    chosen_mass = total = 0.0
    for alternative, logprob in token.alternatives.items():
        if not alternative.startswith(lead):
            continue
        piece = alternative[len(lead) :].lower()
        if not piece or not any(_fits(piece, rest) for rest in rests):
            continue
        total += math.exp(logprob)
        if _fits(piece, chosen_rest):
            chosen_mass += math.exp(logprob)
    return chosen_mass / total if total else None


def choice_confidence(
    tokens: Sequence[TokenChoice],
    text: str,
    offset: int,
    chosen: str,
    allowed: Collection[str],
) -> float:
    """P(chosen | the model picks one of `allowed`), token by token.

    Values can share their first tokens ("data" begins both "databases" and "data-structures"),
    so keep multiplying the step probabilities until only the chosen value still fits.
    """
    if chosen not in allowed:
        return UNMEASURED
    confidence = 1.0
    done = 0  # characters of the chosen value already generated
    for token, start in _tokens_from(tokens, offset):
        still_possible = [v for v in allowed if v.startswith(chosen[:done])]
        if len(still_possible) <= 1 or done >= len(chosen):
            break
        lead = text[start : offset + done] if start < offset + done else ""
        share = _step(token, lead, chosen[done:], [v[done:] for v in still_possible])
        if share is None:
            return UNMEASURED
        confidence *= share
        done = start + len(token.token) - offset
    return confidence
