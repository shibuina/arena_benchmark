"""Classifying the zone names the substring rules cannot: the model's job in part 1.

`catalogue_plan.md` §4 puts one question to part 1 per world - which zones are working zones and
which are break zones - and answers it in three layers: substring rules for the head, the zone's
own `semantics:` where a world declares any, and a model for the tail. This is the tail.

**Only genuinely unknown names are sent.** `classify` covers 99 of the 116 distinct zone names in
the tree for free and deterministically, and asking a model to confirm that `kitchen` is a break
zone is the expensive random number generator `llm_proposal_plan.md` §0 warns about. What reaches
the model is what `Role.UNKNOWN` marks, which is the point of `UNKNOWN` being a real answer
rather than a failure.

**The answer is written to disk and reviewed, not consulted at generation time.** `populate`
reads `zone_roles.yaml`; it never calls a model. So regeneration stays deterministic, offline and
free, a human can overrule any entry by editing one line, and a reviewed entry is never asked
again. `source: human` always wins and is never overwritten.

What the model gets is the world's whole vocabulary and each unknown zone's geometry and
contents, not the bare name. A name like `top_room_3` carries nothing on its own; what it holds
and what it adjoins is the only evidence there is - and when there is none, `unknown` is a
permitted answer and the better one.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs
import yaml

from ..scenario.zones import Role, classify
from .client import DEFAULT_MODEL, Unavailable, ask, prompt_sha256, strip_fence

#: The reviewed mapping, shipped with the package. Keyed by zone name rather than by world: the
#: 116 names span 78 worlds, `zones_by_role` takes a flat name->role map, and a name reviewed
#: once should not need reviewing again in the next world that uses it.
STORE = Path(__file__).resolve().parents[1] / "scenario" / "zone_roles.yaml"

SYSTEM = """\
You classify rooms in a robot navigation simulator by what people use them for.

Answer with one role per zone, from exactly this set:

  work     - where people are while working: desks, stations, treatment rooms, labs, meeting rooms
  break    - where people go when they are not working: kitchens, waiting areas, lounges, toilets
  transit  - corridors, junctions, lobbies, entrances. People pass through; nobody stays
  private  - bedrooms and similar. A robot has no business routing through them
  unknown  - the evidence does not support any of the above

`unknown` is a real answer and the right one when a zone's name is a placeholder and it holds
nothing that says what it is for. A wrong confident answer puts a crowd in a cupboard; an honest
`unknown` sends the zone to a human. Do not guess from the room number.

Reply with JSON only, no prose and no code fence:
{"<zone name>": {"role": "<role>", "why": "<at most 12 words>"}}
"""


@attrs.frozen(kw_only=True)
class Proposal:
    """One zone's proposed role, with enough provenance to re-derive or overrule it."""

    zone: str
    role: Role
    why: str = ""
    model: str = ""
    prompt_sha: str = ""

    @property
    def confident(self) -> bool:
        return self.role is not Role.UNKNOWN


def load_overrides(store: Path | None = None) -> dict[str, str]:
    """The reviewed mapping, flattened to what `zones_by_role` takes. Empty when absent."""
    path = store or STORE
    if not path.is_file():
        return {}
    doc = yaml.safe_load(path.read_text()) or {}
    entries = doc.get("zones") or {}
    return {
        name: str(entry["role"])
        for name, entry in entries.items()
        if isinstance(entry, Mapping) and entry.get("role")
    }


def unknown_names(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> list[str]:
    """Zone names no rule and no reviewed entry accounts for, sorted and deduplicated."""
    reviewed = set(overrides or {})
    return sorted({
        str(z["name"]) for z in zones
        if isinstance(z, Mapping) and z.get("name")
        and str(z["name"]) not in reviewed
        and classify(str(z["name"])) is Role.UNKNOWN
    })


def _evidence(zone: Mapping[str, Any]) -> dict[str, Any]:
    """What is actually known about one zone: its extent and what stands in it.

    Corners become an extent rather than a corner list - a model asked to classify a room does
    not need its vertices, and a long thin zone reads as a corridor where a square one does not.
    """
    corners = [(float(c["x"]), float(c["y"])) for c in zone.get("corners", []) if isinstance(c, Mapping)]
    statics = (zone.get("entities") or {}).get("static") or []
    contents = sorted({
        str(e.get("model", "")).split("/")[-1]
        for e in statics if isinstance(e, Mapping) and e.get("model")
    })
    out: dict[str, Any] = {"contents": contents or "nothing"}
    if corners:
        width = max(x for x, _ in corners) - min(x for x, _ in corners)
        height = max(y for _, y in corners) - min(y for _, y in corners)
        out["size_m"] = [round(width, 1), round(height, 1)]
    if doors := zone.get("doors"):
        out["doors"] = len(doors)
    return out


def build_prompt(zones: Sequence[Mapping[str, Any]], names: Sequence[str], *, world: str) -> str:
    """The user half of the request: the world's vocabulary, then the zones in question.

    The full vocabulary is included even though only some names are asked about, because the
    rest is the evidence: `top_room_3` beside `main_hallway` and `reception` is an office, and
    the same name beside `kitchen` and `bedroom_1` is a flat.
    """
    by_name = {str(z["name"]): z for z in zones if isinstance(z, Mapping) and z.get("name")}
    vocabulary = {
        name: classify(name).value
        for name in sorted(by_name)
    }
    asked = {name: _evidence(by_name[name]) for name in names if name in by_name}
    return (
        f"World: {world}\n\n"
        f"Every zone this world declares, with the role already known where a rule covers it:\n"
        f"{json.dumps(vocabulary, indent=2)}\n\n"
        f"Classify only these, using what they contain and what the rest of the world suggests:\n"
        f"{json.dumps(asked, indent=2)}\n"
    )


def parse(text: str, asked: Sequence[str], *, model: str, sha: str) -> tuple[list[Proposal], list[str]]:
    """Proposals and complaints. Anything unasked-for or unrecognised is dropped, never coerced."""
    try:
        answer = json.loads(strip_fence(text))
    except json.JSONDecodeError as exc:
        return [], [f"response was not JSON: {exc}"]
    if not isinstance(answer, dict):
        return [], [f"response was {type(answer).__name__}, expected an object"]

    proposals: list[Proposal] = []
    complaints: list[str] = []
    wanted = set(asked)
    for name, entry in answer.items():
        if name not in wanted:
            complaints.append(f"{name}: not one of the zones asked about")
            continue
        role_name = entry.get("role") if isinstance(entry, Mapping) else entry
        try:
            role = Role(str(role_name))
        except ValueError:
            complaints.append(f"{name}: {role_name!r} is not a role")
            continue
        why = str(entry.get("why", "")).strip() if isinstance(entry, Mapping) else ""
        proposals.append(Proposal(zone=name, role=role, why=why, model=model, prompt_sha=sha))

    for missing in sorted(wanted - {p.zone for p in proposals}):
        complaints.append(f"{missing}: no answer returned")
    return proposals, complaints


def classify_tail(
    zones: Sequence[Mapping[str, Any]],
    *,
    world: str,
    names: Sequence[str] | None = None,
    model: str | None = None,
) -> tuple[list[Proposal], list[str]]:
    """Ask the model about this world's unclassifiable zones. One call, or none if none remain."""
    asked = list(names if names is not None else unknown_names(zones, load_overrides()))
    if not asked:
        return [], []
    user = build_prompt(zones, asked, world=world)
    sha = prompt_sha256(SYSTEM, user)
    return parse(ask(SYSTEM, user, model=model), asked, model=model or DEFAULT_MODEL, sha=sha)


def merge(proposals: Sequence[Proposal], *, store: Path | None = None) -> tuple[int, list[str]]:
    """Write confident proposals into the store. Returns how many landed, and what was kept.

    Two entries are never overwritten: one a human reviewed, and one already holding the same
    role. The first is the whole point of the store being a reviewable file; the second keeps
    the diff of a re-run empty when nothing changed.
    """
    path = store or STORE
    doc = yaml.safe_load(path.read_text()) if path.is_file() else None
    doc = doc if isinstance(doc, dict) else {}
    entries: dict[str, Any] = doc.get("zones") or {}

    written = 0
    kept: list[str] = []
    for proposal in proposals:
        if not proposal.confident:
            kept.append(f"{proposal.zone}: model returned `unknown` - left for a human")
            continue
        existing = entries.get(proposal.zone)
        if isinstance(existing, Mapping) and existing.get("source") == "human":
            kept.append(f"{proposal.zone}: reviewed by a human as {existing.get('role')!r}")
            continue
        if isinstance(existing, Mapping) and existing.get("role") == proposal.role.value:
            continue
        entries[proposal.zone] = {
            "role": proposal.role.value,
            "source": "llm",
            "model": proposal.model,
            "prompt_sha": proposal.prompt_sha,
            "why": proposal.why,
        }
        written += 1

    if written:
        doc["zones"] = dict(sorted(entries.items()))
        doc["updated"] = datetime.date.today().isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Zone name -> role, for the names `zones.py`'s substring rules do not cover.\n"
            "# `source: llm` entries were proposed by a model and may be overruled: change the\n"
            "# role and set `source: human`, and nothing will overwrite it again.\n"
            "# Read by `populate` through `load_overrides`; no model is called at generation time.\n\n"
            + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
        )
    return written, kept


__all__ = [
    "STORE",
    "SYSTEM",
    "Proposal",
    "Unavailable",
    "build_prompt",
    "classify_tail",
    "load_overrides",
    "merge",
    "parse",
    "unknown_names",
]
