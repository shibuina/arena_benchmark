"""Cells authored inside scenario files rather than in the spec.

A spec that lists `cells:` keeps the construction and the scenario in different files: the
spec says "inject at 90 degrees" and the world's `scenarios/1/scenario.yaml` says who else is
in the corridor. Authoring the case *in* the scenario puts both in one place, so a scenario
directory is a complete, self-describing situation.

Arena's own loader ignores the block: an unknown top-level key in `scenario.yaml` is accepted
and dropped (verified against `arena_simulation_setup`'s loader, which returns the scenario
with its robots and agents intact and no `edge_case` attribute). So the block is inert to
everything except this module - a scenario carrying one still runs anywhere.

```yaml
# worlds/acoustics_bend/scenarios/1/scenario.yaml
robots:  [...]
dynamic: [...]

edge_case:                    # <- ignored by Arena, read here
  id: c_cross90
  note: crossing from the robot's right at the bend
  inject: 1
  place: intercept
  approach_angle: 90.0
  encounter_at: 0.5
```

**The cost, stated plainly.** `scenario` is a launch-invariant key: changing
`task.scenario.file` mid-session leaks flow regions across resets, so each scenario must be
its own stack. Cells in a spec share one launch; cells in scenario files do not. Ten cells
authored this way cost ten boots at 60-90 s each, where ten spec cells cost one. That is the
trade for self-describing scenarios, and it is the reason `cells:` is not being removed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .schema import Cell, SpecError

#: The top-level key a scenario uses to declare its own edge case.
BLOCK = "edge_case"

#: Fields the block may carry. `id` and `note` are labels; the rest map onto
#: `task.edge_case.*` node parameters and are validated against that namespace downstream.
BLOCK_KEYS = frozenset({
    "id", "note", "base", "knob", "level", "catalogue",
    "inject", "place", "approach_angle", "encounter_at", "co_arrival_offset",
    "inject_type", "inject_model", "waypoint_mode", "ped_speed", "lead_out",
    "angle_search", "objects", "timing", "mode", "select",
    "scope",
    # Seconds a time-driven mechanism waits before firing. The mode refuses it on a block that
    # sets neither `objects` nor `waypoints`, so writing it elsewhere fails at load, not silently.
    "delay",
    # Nested plans rather than scalars - see `_STRUCTURED`.
    "waypoints", "trigger",
})

#: Block fields whose machinery is planned but not built. A block using one is refused, because
#: emitting `task.edge_case.<field>` for it would set a parameter no node subscribes to - a
#: silent no-op, which is the one failure this pipeline exists to prevent. Empty now that crowd
#: scope, runtime waypoints and proximity triggers have all landed; kept because it is the
#: guard the next unbuilt field needs.
_UNIMPLEMENTED: dict[str, str] = {}

#: Block fields that are nested plans, not scalars. They have no `task.edge_case.*` parameter:
#: the mode reads them straight out of the scenario file at reset. Emitting one as a parameter
#: would be exactly the silent no-op above, so they are excluded from `params` - and they still
#: separate two cells, because the case scenario each lives in is part of the cell's identity.
_STRUCTURED = frozenset({"waypoints", "trigger"})

#: Block fields that are labels rather than parameters.
_LABELS = frozenset({"id", "note", "base", "knob", "level", "catalogue"})


def scenario_path(worlds_root: str | os.PathLike[str], world: str, scenario: str) -> Path:
    return Path(worlds_root) / world / "scenarios" / scenario / "scenario.yaml"


def read_block(path: Path) -> dict[str, Any] | None:
    """The `edge_case:` block of one scenario file, or None when it declares none.

    A scenario without a block is not an error - it is a perfectly good base population, and
    is exactly what a control arm wants.
    """
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
        raise SpecError(
            f"{path}: '{BLOCK}' has unknown key(s) {sorted(unknown)}; allowed: {sorted(BLOCK_KEYS)}"
        )
    return dict(block)


def cell_from_scenario(
    worlds_root: str | os.PathLike[str],
    world: str,
    scenario: str,
) -> Cell:
    """One cell, read from the scenario that declares it.

    A scenario with no block becomes an unperturbed control cell rather than being skipped:
    the spec asked for it, so it runs, and `inject: 0` is what "the base population alone"
    means everywhere else in this pipeline.
    """
    path = scenario_path(worlds_root, world, scenario)
    block = read_block(path)

    if block is None:
        return Cell(
            id=f"{scenario}__base",
            note=f"{world}/{scenario} base population, no edge case declared",
            scenario=scenario,
            world=world,
            params={"task.edge_case.inject": 0},
        )

    unbuilt = sorted(k for k in block if k in _UNIMPLEMENTED)
    if unbuilt:
        detail = "; ".join(f"{k!r} needs {_UNIMPLEMENTED[k]}" for k in unbuilt)
        raise SpecError(f"{path}: block requires machinery that is not built - {detail}")

    params = {
        f"task.edge_case.{k}": v
        for k, v in block.items()
        if k not in _LABELS and k not in _STRUCTURED
    }
    level = block.get("level")
    knob = str(block.get("knob") or "")
    if knob and level is None:
        raise SpecError(f"{path}: '{BLOCK}.knob' is set but 'level' is not")

    return Cell(
        id=str(block.get("id") or scenario),
        note=str(block.get("note") or ""),
        scenario=scenario,
        world=world,
        knob=knob,
        level=0.0 if level is None else float(level),
        catalogue=str(block.get("catalogue") or ""),
        params=params,
    )
