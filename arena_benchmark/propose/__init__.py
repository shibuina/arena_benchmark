"""Where a model earns its place: the zone names no rule can classify.

`llm_proposal_plan.md` narrows the model's job twice. The catalogue, not the model, is the
source of cases; what remains and is worth building first is part 1's question - which zones are
work zones and which are break zones - for the tail `zones.py`'s substring rules do not reach.
"""

from .client import DEFAULT_MODEL, Unavailable, available, prompt_sha256
from .zone_roles import (
    STORE,
    Proposal,
    build_prompt,
    classify_tail,
    load_overrides,
    merge,
    parse,
    unknown_names,
)

__all__ = [
    "DEFAULT_MODEL",
    "STORE",
    "Proposal",
    "Unavailable",
    "available",
    "build_prompt",
    "classify_tail",
    "load_overrides",
    "merge",
    "parse",
    "prompt_sha256",
    "unknown_names",
]
