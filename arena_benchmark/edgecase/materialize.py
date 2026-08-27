"""Writing a scenario directory: base population plus an `edge_case:` block.

Part 2 of `generation_plan.md` is a transform — the output *is* the input with a block appended
and a provenance header. Arena's loader accepts and ignores an unknown top-level key, so an
annotated scenario runs anywhere; only `arena_benchmark` reads the block.

Four properties, from `generation_plan.md` §2.2:

* **pure** — output depends only on (base document, block). Regeneration is byte-identical.
* **valid either way** — the annotated scenario runs under plain `task.obstacles:=scenario`.
* **one case per scenario** — re-annotating replaces; blocks never stack.
* **provenance in the file** — the header names the base and the generator.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

#: The key Arena ignores and this package reads.
BLOCK = "edge_case"


#: Marker written into every generated scenario's header, and the test for whether replacing
#: one is safe.
GENERATED_MARKER = "# Generated - do not hand-edit; regenerate instead."


def _is_generated(path: Path) -> bool:
    """Whether a scenario file was written by this tool and is therefore safe to replace."""
    try:
        return GENERATED_MARKER in path.read_text()[:1200]
    except OSError:
        return False


def case_scenario_id(base: str, case_id: str) -> str:
    """`<base>__<case>` — the base is in the name because two cases differing only in base are
    different situations, and a suite listing should not need opening files to tell them apart."""
    return f"{base}__{case_id}"


def annotate(base_doc: dict[str, Any], block: dict[str, Any], *, base: str) -> dict[str, Any]:
    """The base document with the block attached, replacing any block already there."""
    doc = {k: v for k, v in base_doc.items() if k != BLOCK}
    doc[BLOCK] = {"base": base, **block}
    return doc


def _header(base: str, block: dict[str, Any], generator: str) -> str:
    return "\n".join((
        f"# Case scenario: {block.get('note') or block.get('id', '')}",
        f"# base: {base} · generator: {generator} · case: {block.get('id', '')}",
        "#",
        "# Generated - do not hand-edit; regenerate instead.",
        "# The edge_case block is a REQUEST. Geometry is solved at run time against the robot's",
        "# planned route and the ACHIEVED values are recorded in cases.jsonl - they will often",
        "# differ. Arena's scenario loader ignores this block.",
        "", "",
    ))


def write(
    base_doc: dict[str, Any],
    block: dict[str, Any],
    *,
    world_root: str | os.PathLike[str],
    base: str,
    generator: str = "catalogue",
    extra_dynamic: Sequence[dict[str, Any]] = (),
) -> Path:
    """Write `scenarios/<base>__<case>/scenario.yaml`. Returns the directory.

    `extra_dynamic` are agents the case itself authors - a doorway blocker - appended after the
    base population so the base stays byte-identical up to where the case begins.
    """
    case_id = str(block.get("id") or "case")
    out = Path(world_root) / "scenarios" / case_scenario_id(base, case_id)
    out.mkdir(parents=True, exist_ok=True)
    doc = annotate(base_doc, block, base=base)
    if extra_dynamic:
        doc["dynamic"] = [*(doc.get("dynamic") or []), *extra_dynamic]

    # The case scenario is a copy of the base, so it inherits the base's `./name.yaml`
    # references and needs the same file beside it.
    for agent in doc.get("dynamic") or []:
        kind = str((agent.get("agent") or {}).get("agent_type", "")) if isinstance(agent, dict) else ""
        if kind.startswith("./"):
            src = Path(world_root) / "scenarios" / base / kind[2:]
            if src.is_file():
                shutil.copy2(src, out / src.name)
    (out / "scenario.yaml").write_text(
        _header(base, block, generator) + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None)
    )
    return out


class WouldOverwrite(Exception):
    """A generated scenario would replace one that already exists."""


def write_base(
    doc: dict[str, Any],
    *,
    world_root: str | os.PathLike[str],
    profile: str,
    generator: str = "populate",
    force: bool = False,
    note: str = "",
) -> Path:
    """Write one base scenario, `scenarios/<profile>/scenario.yaml`.

    `profile` is the directory name - a base *variant* id such as `normal_b` - and `note` is
    the one line that says what makes it that variant.

    **Refuses to replace a scenario that already exists.** The catalogue's profile names are
    ordinary words and several shipped worlds already use them - `hospital_1` ships a
    hand-authored 14-agent `normal`, which an earlier version of this function silently
    overwrote. A generated scenario is regenerable; a hand-authored one is not.
    """
    out = Path(world_root) / "scenarios" / profile
    existing = out / "scenario.yaml"
    if existing.is_file() and not force and not _is_generated(existing):
        raise WouldOverwrite(
            f"{existing} already exists and was not generated by this tool. "
            f"Pass force=True to replace it, or generate under a different name."
        )
    out.mkdir(parents=True, exist_ok=True)

    # Copy in any agent type the population references by relative path. `resolve_agent_type_path`
    # resolves `./name.yaml` against the scenario's own directory, so the file has to live beside
    # the scenario that uses it - a reference alone is not enough.
    from ..scenario.populate import bt_type_source  # noqa: PLC0415 - avoids a package cycle

    if any(str((a.get("agent") or {}).get("agent_type", "")).startswith("./")
           for a in (doc.get("dynamic") or []) if isinstance(a, dict)):
        src = bt_type_source()
        if src.is_file():
            shutil.copy2(src, out / src.name)
    n = len(doc.get("dynamic") or [])
    header = "\n".join((
        f"# Base scenario: {profile} ({n} pedestrian{'s' if n != 1 else ''})",
        f"# generator: {generator}",
        *([f"# variant: {note}"] if note else []),
        "#",
        "# Generated - do not hand-edit; regenerate instead. Every agent is placed clear of the",
        "# robot's route: an agent on the route makes the controller refuse to move, which reads",
        "# as a frozen robot rather than a bad scenario.",
        "", "",
    ))
    (out / "scenario.yaml").write_text(header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))
    return out


def prune_generated(world_root: str | os.PathLike[str], keep: Sequence[str]) -> list[str]:
    """Remove generated scenario directories this run did not write. Returns what went.

    Only directories whose `scenario.yaml` carries `GENERATED_MARKER` are touched - a
    hand-authored scenario is never a candidate - and only those not in `keep`. This is what
    keeps a renamed layout (`normal` -> `normal_a`) from leaving the old one beside the new as
    a second, stale copy of the same population.
    """
    root = Path(world_root) / "scenarios"
    wanted = set(keep)
    gone: list[str] = []
    for path in sorted(root.glob("*/scenario.yaml")):
        name = path.parent.name
        if name in wanted or not _is_generated(path):
            continue
        shutil.rmtree(path.parent)
        gone.append(name)
    return gone
