"""One thin call into the project's existing LLM client.

`task_generator.utils.gpt` already carries the genai shim, honours `LLM_API_ENDPOINT` for a
local endpoint, and is what `TM_Prompt` uses. This adds no second provider and no second client;
it adds a synchronous, offline-safe entry point, because generation runs from a CLI rather than
from inside a running node.

**Nothing in the generation path calls this.** The classifier writes a reviewed mapping to disk
and `populate` reads that file, so scenario generation stays deterministic, offline and free.
A model in the generation loop would make every regeneration a fresh experiment.
"""

from __future__ import annotations

import hashlib
import os

#: Same environment variable `prompt_utils` reads, so one setting steers both paths.
DEFAULT_MODEL = os.environ.get("ARENA_PROMPT_LLM", "gemini-3.5-flash")


class Unavailable(RuntimeError):
    """No usable client - the SDK is missing, or no endpoint and no API key is configured."""


def prompt_sha256(*parts: str) -> str:
    """Content hash of everything that steers a response.

    Recorded beside every answer: a changed prompt is a changed experiment, and a mapping whose
    provenance cannot distinguish the two is a mapping nobody can re-derive.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def available() -> bool:
    """Whether a call could be made at all, without making one."""
    try:
        _client()
    except Unavailable:
        return False
    return True


def _client() -> object:
    if not os.environ.get("LLM_API_ENDPOINT") and not (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    ):
        raise Unavailable(
            "no LLM configured: set GEMINI_API_KEY (or GOOGLE_API_KEY) for the hosted API, "
            "or LLM_API_ENDPOINT to point at a local one"
        )
    try:
        from task_generator.utils.gpt import genai  # noqa: PLC0415 - optional, CLI-only
    except Exception as exc:  # noqa: BLE001 - any import failure means the same thing here
        raise Unavailable(f"the genai client is not importable: {exc}") from exc
    return genai


#: One client for the process. Held rather than built per call: a `Client()` that goes out of
#: scope takes its httpx pool with it, and the in-flight request dies as
#: "Cannot send a request, as the client has been closed."
_CLIENT: object | None = None


def ask(system: str, user: str, *, model: str | None = None, temperature: float = 0.0) -> str:
    """One completion, as raw text. Raises `Unavailable` rather than returning a guess.

    `temperature: 0.0` because the task is classification, not invention. The one thing worse
    than a model disagreeing with a human reviewer is a model disagreeing with itself between
    two runs over the same world.
    """
    global _CLIENT  # noqa: PLW0603 - one client per process is the point
    genai = _client()
    if _CLIENT is None:
        _CLIENT = genai.Client()
    try:
        response = _CLIENT.models.generate_content(
            model=model or DEFAULT_MODEL,
            contents=[user],
            config=genai.types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - the SDK raises many types; the caller needs one
        raise Unavailable(f"the model call failed: {type(exc).__name__}: {exc}") from exc
    text = getattr(response, "text", None)
    if not text:
        raise Unavailable("the model returned an empty response")
    return text.strip()


def strip_fence(text: str) -> str:
    """Drop a ``` fence if the model added one, the way `TM_Prompt` does."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    return text[:-3].strip() if text.endswith("```") else text.strip()
