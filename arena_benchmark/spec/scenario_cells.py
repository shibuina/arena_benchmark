"""Cells authored inside scenario files.

A scenario directory is a complete, self-describing situation: its population, and an
`edge_case:` block listing the runtime effects the task mode applies. A spec that lists
`scenarios:` runs each one as its own cell (and, since `scenario` is a launch-invariant key,
its own stack).

Arena's own loader ignores the block; only the task mode and this module read it. Validation
here mirrors `task_generator`'s block reader on the key set, so a spec refuses at expand time
what the mode would refuse at reset.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .schema import Cell, SpecError

BLOCK = "edge_case"

#: Provenance fields; carried, never acted on.
LABELS = frozenset({"id", "note", "base", "prompt_id", "steps", "provenance", "decisions", "owned", "formation"})
#: Fields that do something at run time.
ACTIVE = frozenset({"effects", "objects"})
BLOCK_KEYS = LABELS | ACTIVE

EFFECT_TYPES = frozenset({"intercept", "rally", "shuffle", "scatter", "track_robot", "hold"})


def scenario_path(worlds_root: str | os.PathLike[str], world: str, scenario: str) -> Path:
    return Path(worlds_root) / world / "scenarios" / scenario / "scenario.yaml"


def read_block(path: Path) -> dict[str, Any] | None:
    """The `edge_case:` block of one scenario file, or None when it declares none."""
    if not path.is_file():
        raise SpecError(f"scenario not found: {path}")
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise SpecError(f"{path}: not valid YAML: {exc}") from None
    if not isinstance(doc, dict):
        raise SpecError(f"{path}: expected a mapping at the top level")
    block = doc.get(BLOCK)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise SpecError(f"{path}: '{BLOCK}' must be a mapping, got {type(block).__name__}")
    unknown = set(block) - BLOCK_KEYS
    if unknown:
        raise SpecError(f"{path}: '{BLOCK}' has unknown key(s) {sorted(unknown)}; allowed: {sorted(BLOCK_KEYS)}")
    effects = block.get("effects") or []
    if not isinstance(effects, list):
        raise SpecError(f"{path}: '{BLOCK}.effects' must be a list")
    for i, effect in enumerate(effects):
        if not isinstance(effect, dict) or str(effect.get("type", "")) not in EFFECT_TYPES:
            raise SpecError(f"{path}: effects[{i}] must be a mapping with a type in {sorted(EFFECT_TYPES)}")
    return dict(block)


def effects_digest(block: dict[str, Any] | None) -> str:
    """Content hash of what the block does; labels are left out so a renamed case is the same cell."""
    active = {k: block.get(k) for k in sorted(ACTIVE)} if block else {}
    return hashlib.sha256(json.dumps(active, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cell_from_scenario(worlds_root: str | os.PathLike[str], world: str, scenario: str) -> Cell:
    """One cell, read from the scenario that declares it. A scenario with no block is the
    control cell: the base population alone."""
    path = scenario_path(worlds_root, world, scenario)
    block = read_block(path)
    if block is None:
        return Cell(id=f"{scenario}__base", note=f"{world}/{scenario} base population, no edge case declared", scenario=scenario, world=world, params={})
    return Cell(
        id=str(block.get("id") or scenario),
        note=str(block.get("note") or ""),
        scenario=scenario,
        world=world,
        # The digest is what separates two scenarios whose names collide but whose effects
        # differ; it rides in `params` so the existing uid machinery hashes it.
        params={"edge_case.effects_digest": effects_digest(block)},
    )
