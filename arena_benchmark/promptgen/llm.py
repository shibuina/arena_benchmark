"""One model call per step, cached by content hash (`new_plan_2.md` §9).

The model runs at temperature 0 and its raw answer is cached by
`sha256(system prompt, user prompt)`, so regenerating with an unchanged cache is
byte-identical and free. The cache is a directory of JSON files, one per call, which is also
the audit trail: every rejected answer is kept beside the accepted one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import attrs

from ..propose.client import DEFAULT_MODEL, Unavailable, ask, prompt_sha256

#: Where answers are cached unless told otherwise.
DEFAULT_CACHE = Path(os.environ.get("ARENA_DATA_DIR", "data")) / "promptgen" / "llm_cache"

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class AnswerError(ValueError):
    """The model's answer was not JSON."""


@attrs.frozen
class Answer:
    text: str
    sha256: str
    model: str
    cached: bool

    def json(self) -> dict[str, Any]:
        text = _FENCE.sub("", self.text.strip()).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise AnswerError(f"no JSON object in the answer: {text[:200]!r}")
        try:
            doc = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise AnswerError(f"answer is not valid JSON ({exc}): {text[:200]!r}") from None
        if not isinstance(doc, dict):
            raise AnswerError("answer must be a JSON object")
        return doc


class Oracle:
    """A stand-in for the model: answers come from a mapping keyed by the user prompt's
    sha256 prefix, or from a callable. Used for tests and for `--oracle` runs, where the
    prompt set's own template slots are the answer."""

    def __init__(self, answer: object) -> None:
        self._answer = answer

    def __call__(self, system: str, user: str) -> str:
        value = self._answer(system, user) if callable(self._answer) else self._answer
        return value if isinstance(value, str) else json.dumps(value)


def call(system: str, user: str, *, cache_dir: Path | None = None, model: str | None = None, oracle: Oracle | None = None) -> Answer:
    """One completion, cached. Raises `Unavailable` when no model is configured and no oracle given."""
    name = model or DEFAULT_MODEL
    key = prompt_sha256(name, system, user)
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    path = root / f"{key}.json"
    if oracle is None and path.is_file():
        doc = json.loads(path.read_text())
        return Answer(text=str(doc["answer"]), sha256=hashlib.sha256(str(doc["answer"]).encode()).hexdigest()[:16], model=str(doc.get("model", name)), cached=True)
    if oracle is not None:
        text = oracle(system, user)
        return Answer(text=text, sha256=hashlib.sha256(text.encode()).hexdigest()[:16], model="oracle", cached=False)
    text = ask(system, user, model=name, temperature=0.0)
    from ..propose import client as _client  # noqa: PLC0415 - the usage side channel of the call above
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": name, "system_sha256": prompt_sha256(system), "user": user, "answer": text, "usage": _client.LAST_USAGE}, indent=1))
    return Answer(text=text, sha256=hashlib.sha256(text.encode()).hexdigest()[:16], model=name, cached=False)


__all__ = ["DEFAULT_CACHE", "Answer", "AnswerError", "Oracle", "Unavailable", "call"]
