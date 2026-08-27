"""Classifying a world's zones into the roles the base scenarios need.

The five base scenarios differ mainly in *where* people are: `normal` puts them where work
happens, `break` puts them where people go when they are not working. Neither is derivable from
geometry — it is a question about what a room is *for*, and worlds answer it only by naming the
zone.

There are 110 distinct zone names across the 77 annotated worlds and the distribution is long
tailed, so this is a two-stage classifier: substring rules for the head, and an explicit
`unknown` for everything else. `unknown` is a real answer, not a failure — it routes a name to a
human or to the LLM classifier rather than guessing, because guessing `storage` is a work zone
silently puts a crowd in a cupboard.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from typing import Any


class Role(enum.StrEnum):
    """What a zone is for, from the point of view of populating it."""

    WORK = "work"
    """Where people are when they are working: desks, stations, treatment rooms."""
    BREAK = "break"
    """Where people go when they are not: kitchens, waiting areas, bathrooms, lounges."""
    TRANSIT = "transit"
    """Corridors, junctions, lobbies. People pass through; nobody dwells."""
    PRIVATE = "private"
    """Bedrooms and similar. Populated in neither `normal` nor `break` - a robot has no
    business routing through them and a crowd there is not a navigation scenario."""
    UNKNOWN = "unknown"
    """Not classifiable by rule. Left for review rather than guessed."""


#: Substring rules, first match wins, checked in this order. Order matters: `nurse_resting_room`
#: must reach the BREAK rule (`resting`) before the WORK rule (`nurse`), and `wc_waiting` must
#: reach BREAK before anything else claims it.
_RULES: tuple[tuple[Role, tuple[str, ...]], ...] = (
    (Role.BREAK, (
        "waiting", "break_room", "resting", "lounge", "cafeteria", "dining",
        "kitchen", "kitchenette", "staff_room", "bathroom", "ensuite", "wc_",
        "restroom", "toilet", "balcony", "living_room", "phone_booth",
    )),
    (Role.PRIVATE, ("bedroom", "master_bed")),
    (Role.WORK, (
        "work", "office", "nurse_station", "reception", "patient_room", "patient_ward",
        "exam", "consultation", "operating", "procedure", "triage", "lab",
        "meeting_room", "conference", "pharmacy", "file_room", "utility", "storage",
        "workspace",
    )),
    (Role.TRANSIT, (
        "hallway", "corridor", "junction", "arm_", "entrance", "entry", "link_",
        "lobby", "destination", "empty_zone",
    )),
)


def classify(name: str) -> Role:
    """The role of one zone name.

    Pure and case-insensitive. A name matching no rule is `UNKNOWN`, deliberately.
    """
    lowered = name.strip().lower()
    for role, needles in _RULES:
        if any(n in lowered for n in needles):
            return role
    return Role.UNKNOWN


def classify_all(names: Sequence[str], overrides: Mapping[str, str] | None = None) -> dict[str, Role]:
    """Roles for a world's zones, with a review layer on top.

    `overrides` is the reviewed mapping - from a human or from the LLM classifier - and always
    wins. That is the whole point of the split: the rules cover the head cheaply and the
    override file carries the judgements that had to be made once.
    """
    resolved = {n: classify(n) for n in names}
    for name, role in (overrides or {}).items():
        if name in resolved:
            resolved[name] = Role(role)
    return resolved


def zones_by_role(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> dict[Role, list[str]]:
    """Zone names grouped by role, from a world's parsed `zones` list."""
    names = [str(z["name"]) for z in zones if isinstance(z, Mapping) and z.get("name")]
    out: dict[Role, list[str]] = {r: [] for r in Role}
    for name, role in classify_all(names, overrides).items():
        out[role].append(name)
    return {r: sorted(v) for r, v in out.items()}
