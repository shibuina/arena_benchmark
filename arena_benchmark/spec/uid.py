"""Stable identities for a cell and an episode.

Two ids, because the benchmark asks two different questions of the same rows:

``cell_uid``
    Everything about the situation *except* which planner met it. Rows sharing a cell_uid
    are the arms of one comparison, so this is the group key for every paired statistic. It
    is also the identity of a suite entry, since a suite entry is a situation, not a result.

    Note what is **not** in it: the cell's ``id`` and the spec's name. Both are labels. If
    they were hashed, renaming a cell would invalidate every row already measured for it,
    and two specs describing the same situation could never be recognised as describing the
    same situation. Leaving them out also gives the expansion something to check - two cells
    that hash alike really are the same construction under two names, which is a spec bug
    that no downstream join could have disambiguated.

``episode_uid``
    One run: a cell, an arm, a seed. The join key end to end - it is written into the
    generated sweep plan as the case ``label``, which the runner copies verbatim into
    ``trials.jsonl``, so a manifest row and the trial that ran it can always be matched
    without inferring anything from knob/level/seed collisions.

Both are content hashes, so they are stable across spec reordering, comments and formatting,
and they change when - and only when - something that affects the episode changes. A spec
edited to add a cell leaves every other uid untouched, which is what makes an incremental
re-run possible instead of a full rebuild.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

#: Hex characters kept from the digest. 16 gives a collision probability under 1e-9 for the
#: dataset sizes here (~1e4 rows) while staying short enough to read in a filename.
UID_LEN = 16


def _canonical(payload: object) -> bytes:
    """Bytes that depend on content and nothing else.

    ``sort_keys`` so a mapping's insertion order cannot change the hash, and floats are
    normalised through ``repr`` by json itself so 0.5 and 0.50 agree.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()[:UID_LEN]


def cell_uid(
    *,
    world: str,
    scenario: str,
    knob: str,
    level: float,
    params: Mapping[str, Any],
    context: Mapping[str, Any],
) -> str:
    """Identity of a situation: world, scenario, construction, and the run context.

    ``knob`` and ``level`` are hashed alongside ``params`` because they are part of the
    construction and not always reachable through it - a grid expands into cells whose
    parameters are identical and whose knob and level are the only thing that differs.

    ``context`` carries the spec-level settings that change what the episode *is* rather than
    who runs it: simulator, human backend, task modes, episode duration. The arm is
    deliberately absent - that is the whole point of this id.
    """
    return _digest(
        {
            "v": 1,
            "world": world,
            "scenario": scenario,
            "knob": knob,
            "level": float(level),
            "params": dict(params),
            "context": dict(context),
        }
    )


def episode_uid(*, cell: str, arm_id: str, arm: Mapping[str, Any], seed: int) -> str:
    """Identity of one run: a cell, an arm, a seed.

    The arm's settings are hashed alongside its id so that editing an arm in place - a
    different robot, a different tick rate - produces new episodes rather than silently
    reusing rows measured under the old configuration.
    """
    return _digest({"v": 1, "cell": cell, "arm_id": arm_id, "arm": dict(arm), "seed": int(seed)})
