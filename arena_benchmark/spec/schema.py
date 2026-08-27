"""The benchmark spec: what to run, against which planners, how many times.

A sweep plan (``task_generator.utils.edge_case_plan``) describes trials inside one already
running stack. It cannot describe a benchmark, because the thing a benchmark varies most -
the planner - is bound when the stack launches, not by a parameter. A spec is the layer
above: it names the **arms** as well as the cells, and expanding it produces episodes that
span many launches.

```yaml
version: 1
name: screen_v1

sim: dummy
human: arena
tm_robots: edge_case
tm_obstacles: edge_case
base_mode: scenario
scenario: default
episode_duration: 60

worlds: [arena_arena_002, arena_classic_002]

arms:
  - {id: nav2_diff, robot: jackal, mobile: nav2, drive: differential}
  - {id: rlrvo,     robot: jackal, mobile: drl, planner: rlrvo, drive: differential}

seeds: {start: 1000, count: 3}

cells:
  - id: base
    params: {task.edge_case.inject: 0}
  - id: c_headon_mid
    params:
      task.edge_case.inject: 1
      task.edge_case.place: intercept
      task.edge_case.approach_angle: 180.0
      task.edge_case.encounter_at: 0.5
```

Validation is strict in the same way and for the same reason as the sweep plan's: an unknown
key is an error, never a warning. A spec that silently ignored ``aproach_angle`` would report
a swept factor that was never swept.

Two rules here are not typo-catching but methodology, enforced because getting them wrong
produces a dataset that looks fine and answers nothing:

* **Every drive family that contains a learned planner must also contain a baseline.**
  ``robot.mobile:=drl`` picks a chassis compatible with the planner's ``action_type``, so a
  differential planner and an omnidirectional one are not comparable to each other or to a
  single shared control. Each family needs its own.
* **Cell parameters must be node parameters.** Anything that has to be decided at launch
  belongs on the arm, not in ``params``, because the orchestrator can only set the latter
  between episodes.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs
import yaml


def default_worlds_root() -> str:
    """Where worlds live, for a spec that reads cells out of scenario files.

    Resolved from the package's own location rather than the working directory, so
    `arena_bench` behaves the same from the workspace root and from inside the package.

    **The source tree wins over the installed share directory**, which is the opposite of
    `spec/locate.py`. Generated scenarios - the five base scenarios and the case scenarios
    written on top of them - are authored into `src/`, because that is the version-controlled
    source of truth; `install/` is build output that a rebuild overwrites. Preferring the share
    directory made the container read a copy that did not have them yet, which surfaced as
    "scenario not found" for scenarios sitting on disk.
    """
    src = Path(__file__).resolve().parents[3] / "arena_simulation_setup" / "worlds"
    if src.is_dir():
        return str(src)
    try:
        from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415

        share = Path(get_package_share_directory("arena_simulation_setup")) / "worlds"
        if share.is_dir():
            return str(share)
    except Exception:
        pass
    return str(src)

#: The only spec version this module understands. Bumped when a change would make an older
#: spec mean something different, never for additions.
SPEC_VERSION = 1

#: Drive families. An arm must declare one, and comparisons only ever happen within a family.
DRIVES = frozenset({"differential", "omnidirectional"})

#: Mobile adapters that take a `planner` name. Everything else must not carry one.
PLANNER_ADAPTERS = frozenset({"drl", "rosnav_rl"})

SPEC_KEYS = frozenset(
    {
        "version", "name", "sim", "human", "tm_robots", "tm_obstacles", "base_mode",
        "scenario", "catalogue", "select", "timing", "driver", "episode_duration",
        "worlds", "arms", "cells", "grid", "seeds", "scenarios", "worlds_root",
    }
)

ARM_KEYS = frozenset({"id", "robot", "mobile", "planner", "global_planner", "drive", "rate_hz", "extra"})

CELL_KEYS = frozenset({"id", "knob", "level", "catalogue", "params", "note"})

GRID_KEYS = frozenset({"knobs", "levels", "catalogue"})

SEED_KEYS = frozenset({"start", "count"})

_DEFAULTS: dict[str, Any] = {
    "sim": "dummy",
    "human": "arena",
    "tm_robots": "edge_case",
    "tm_obstacles": "edge_case",
    "base_mode": "scenario",
    "scenario": "default",
    "catalogue": "level_d",
    "select": "nearest_approach",
    "timing": "off",
    "driver": "reset",
    "episode_duration": 60.0,
}


class SpecError(Exception):
    """A spec could not be read, or says something that cannot be executed."""


@attrs.frozen(kw_only=True)
class Arm:
    """One planner configuration, and everything about it that is fixed at launch."""

    id: str
    robot: str
    mobile: str
    drive: str
    planner: str = ""
    global_planner: str = ""
    rate_hz: float | None = None
    extra: Mapping[str, str] = attrs.field(factory=dict)

    @property
    def is_baseline(self) -> bool:
        """A classical stack, i.e. the control for its drive family."""
        return self.mobile not in PLANNER_ADAPTERS

    def launch_args(self) -> dict[str, str]:
        """The ``k:=v`` arguments that select this arm.

        ``rate_hz`` is emitted even when it matches the default, because the default comes
        from the robot's own ``caps/mobile.yaml`` and therefore differs between arms unless
        somebody pins it.
        """
        args: dict[str, str] = {"robot": self.robot, "robot.mobile": self.mobile}
        if self.planner:
            args["robot.mobile.planner"] = self.planner
        if self.global_planner:
            args["robot.mobile.global_planner"] = self.global_planner
        if self.rate_hz is not None:
            args["robot.mobile.rate"] = f"{self.rate_hz:g}"
        args.update(self.extra)
        return args

    def identity(self) -> dict[str, Any]:
        """What makes this arm itself, for hashing into an episode uid."""
        return {
            "robot": self.robot,
            "mobile": self.mobile,
            "planner": self.planner,
            "global_planner": self.global_planner,
            "drive": self.drive,
            "rate_hz": self.rate_hz,
            "extra": dict(self.extra),
        }


@attrs.frozen(kw_only=True)
class Cell:
    """One construction under test: a knob, a geometry, an object timeline, or nothing."""

    id: str
    knob: str = ""
    level: float = 0.0
    catalogue: str = ""
    params: Mapping[str, Any] = attrs.field(factory=dict)
    note: str = ""
    #: Base scenario this cell runs against. Empty means the spec-wide `scenario`. Set when the
    #: cell was authored inside a scenario file, where the two are inseparable by construction.
    scenario: str = ""
    #: World this cell is bound to, when it was authored inside that world's scenario file. A
    #: case written in `acoustics_bend/scenarios/1/` describes *that* corridor; replicating it
    #: onto another world would silently mean something else. Empty = applies to every world,
    #: which is what a spec-authored cell does.
    world: str = ""


@attrs.frozen(kw_only=True)
class Spec:
    """A validated benchmark spec. Expand it with :func:`arena_benchmark.spec.expand`."""

    name: str
    version: int
    sim: str
    human: str
    tm_robots: str
    tm_obstacles: str
    base_mode: str
    scenario: str
    catalogue: str
    select: str
    timing: str
    driver: str
    episode_duration: float
    worlds: tuple[str, ...]
    arms: tuple[Arm, ...]
    cells: tuple[Cell, ...]
    seed_start: int
    seed_count: int

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_start, self.seed_start + self.seed_count))

    @property
    def n_episodes(self) -> int:
        """Episodes this spec expands to.

        Not `worlds x cells x arms x seeds`: a cell authored inside a world's scenario file is
        bound to that world and runs only there, so it contributes one world's worth rather
        than the full cross product.
        """
        per_cell = sum(1 if c.world else len(self.worlds) for c in self.cells)
        return per_cell * len(self.arms) * self.seed_count

    def context(self) -> dict[str, Any]:
        """Spec-level settings that change what an episode *is*, for the cell uid.

        Deliberately excludes ``name``, ``driver`` and anything cosmetic: renaming a spec or
        switching how episodes are triggered must not invalidate rows already measured.
        """
        return {
            "sim": self.sim,
            "human": self.human,
            "tm_robots": self.tm_robots,
            "tm_obstacles": self.tm_obstacles,
            "base_mode": self.base_mode,
            "episode_duration": self.episode_duration,
            "select": self.select,
            "timing": self.timing,
        }

    def families(self) -> dict[str, tuple[Arm, ...]]:
        """Arms grouped by drive family - the only groups within which a comparison holds."""
        out: dict[str, list[Arm]] = {}
        for arm in self.arms:
            out.setdefault(arm.drive, []).append(arm)
        return {k: tuple(v) for k, v in sorted(out.items())}


# Parsing
# -------


def _mapping(raw: object, where: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise SpecError(f"{where}: expected a mapping, got {type(raw).__name__}")
    return raw


def _reject_unknown(raw: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


def _str_list(raw: object, where: str) -> tuple[str, ...]:
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise SpecError(f"{where}: expected a list, got {type(raw).__name__}")
    values = tuple(str(v) for v in raw)
    if not values:
        raise SpecError(f"{where}: must not be empty")
    dupes = sorted({v for v in values if values.count(v) > 1})
    if dupes:
        raise SpecError(f"{where}: duplicate entries {dupes}")
    return values


def _float(raw: object, where: str) -> float:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise SpecError(f"{where}: expected a number, got {raw!r}") from None


def _parse_arm(raw: object, index: int) -> Arm:
    where = f"arms[{index}]"
    raw = _mapping(raw, where)
    _reject_unknown(raw, ARM_KEYS, where)

    for required in ("id", "robot", "mobile", "drive"):
        if not raw.get(required):
            raise SpecError(f"{where}: '{required}' is required")

    drive = str(raw["drive"])
    if drive not in DRIVES:
        raise SpecError(f"{where}: drive={drive!r} is not one of {sorted(DRIVES)}")

    mobile = str(raw["mobile"])
    planner = str(raw.get("planner") or "")
    if mobile in PLANNER_ADAPTERS and not planner:
        raise SpecError(f"{where}: mobile={mobile!r} needs a 'planner'")
    if mobile not in PLANNER_ADAPTERS and planner:
        raise SpecError(
            f"{where}: mobile={mobile!r} takes no 'planner' (got {planner!r}); "
            f"planner names apply to {sorted(PLANNER_ADAPTERS)}"
        )

    extra = raw.get("extra") or {}
    extra = _mapping(extra, f"{where}.extra")

    rate = raw.get("rate_hz")
    return Arm(
        id=str(raw["id"]),
        robot=str(raw["robot"]),
        mobile=mobile,
        drive=drive,
        planner=planner,
        global_planner=str(raw.get("global_planner") or ""),
        rate_hz=None if rate is None else _float(rate, f"{where}.rate_hz"),
        extra={str(k): str(v) for k, v in extra.items()},
    )


def _parse_cell(raw: object, index: int) -> Cell:
    where = f"cells[{index}]"
    raw = _mapping(raw, where)
    _reject_unknown(raw, CELL_KEYS, where)

    if not raw.get("id"):
        raise SpecError(f"{where}: 'id' is required - it names the cell in the suite and the summary")

    params = _mapping(raw.get("params") or {}, f"{where}.params")
    for key in params:
        if not str(key).startswith("task."):
            raise SpecError(
                f"{where}.params: {key!r} is not a node parameter. Cell parameters are set "
                f"between episodes, so anything decided at launch belongs on the arm instead."
            )

    knob = str(raw.get("knob") or "")
    level = raw.get("level")
    if knob and level is None:
        raise SpecError(f"{where}: knob={knob!r} needs a 'level'")

    return Cell(
        id=str(raw["id"]),
        knob=knob,
        level=0.0 if level is None else _float(level, f"{where}.level"),
        catalogue=str(raw.get("catalogue") or ""),
        params={str(k): v for k, v in params.items()},
        note=str(raw.get("note") or ""),
    )


def _expand_grid(raw: object) -> list[Cell]:
    """``grid: {knobs: [...], levels: [...]}`` as a shorthand for one-at-a-time cells."""
    where = "grid"
    raw = _mapping(raw, where)
    _reject_unknown(raw, GRID_KEYS, where)

    knobs = _str_list(raw.get("knobs"), f"{where}.knobs")
    levels_raw = raw.get("levels")
    if isinstance(levels_raw, str) or not isinstance(levels_raw, Sequence) or not levels_raw:
        raise SpecError(f"{where}.levels: expected a non-empty list")
    levels = [_float(v, f"{where}.levels") for v in levels_raw]
    catalogue = str(raw.get("catalogue") or "")

    return [
        Cell(
            id=f"{knob}@{level:g}",
            knob=knob,
            level=level,
            catalogue=catalogue,
            params={"task.edge_case.inject": 1},
        )
        for knob in knobs
        for level in levels
    ]


def _cells_from_scenarios(raw: Mapping[str, Any], scenarios: Sequence[str]) -> list[Cell]:
    """Cells authored inside scenario files - see `spec/scenario_cells.py`.

    Imported here rather than at module scope because `scenario_cells` imports `Cell` and
    `SpecError` from this module.
    """
    from .scenario_cells import cell_from_scenario  # noqa: PLC0415

    worlds = _str_list(raw.get("worlds"), "spec.worlds")
    worlds_root = str(raw.get("worlds_root") or default_worlds_root())

    out: list[Cell] = []
    for scenario in scenarios:
        # A cell is read once per (world, scenario): the same scenario name in two worlds is
        # two different situations, and each world's file may declare a different block.
        for world in worlds:
            out.append(cell_from_scenario(worlds_root, world, scenario))
    return out


def _parse_seeds(raw: object) -> tuple[int, int]:
    if raw is None:
        return 1000, 3
    if isinstance(raw, int) and not isinstance(raw, bool):
        return 1000, int(raw)
    raw = _mapping(raw, "seeds")
    _reject_unknown(raw, SEED_KEYS, "seeds")
    start = int(raw.get("start", 1000))
    count = int(raw.get("count", 3))
    if count < 1:
        raise SpecError(f"seeds.count: must be at least 1, got {count}")
    return start, count


def parse_spec(raw: object, *, name: str = "spec") -> Spec:
    """Validate a loaded document into a :class:`Spec`."""
    raw = _mapping(raw, "spec")
    _reject_unknown(raw, SPEC_KEYS, "spec")

    version = raw.get("version")
    if version is None:
        raise SpecError("spec: 'version' is required")
    if int(version) != SPEC_VERSION:
        raise SpecError(f"spec: version {version} is not supported (this build reads {SPEC_VERSION})")

    settings = {key: raw.get(key, default) for key, default in _DEFAULTS.items()}

    arms = tuple(_parse_arm(a, i) for i, a in enumerate(raw.get("arms") or []))
    if not arms:
        raise SpecError("spec: 'arms' must name at least one planner configuration")
    arm_ids = [a.id for a in arms]
    dupes = sorted({i for i in arm_ids if arm_ids.count(i) > 1})
    if dupes:
        raise SpecError(f"spec.arms: duplicate id(s) {dupes}")

    cells = [_parse_cell(c, i) for i, c in enumerate(raw.get("cells") or [])]
    if "grid" in raw:
        cells.extend(_expand_grid(raw["grid"]))
    if "scenarios" in raw:
        cells.extend(_cells_from_scenarios(raw, _str_list(raw.get("scenarios"), "spec.scenarios")))
    if not cells:
        raise SpecError("spec: needs 'cells', 'grid', or both")
    cell_keys = [(c.world, c.id) for c in cells]
    dupes = sorted({f"{w}/{i}" if w else i for (w, i) in cell_keys if cell_keys.count((w, i)) > 1})
    if dupes:
        raise SpecError(f"spec.cells: duplicate id(s) {dupes}")

    seed_start, seed_count = _parse_seeds(raw.get("seeds"))

    spec = Spec(
        name=str(raw.get("name") or name),
        version=SPEC_VERSION,
        sim=str(settings["sim"]),
        human=str(settings["human"]),
        tm_robots=str(settings["tm_robots"]),
        tm_obstacles=str(settings["tm_obstacles"]),
        base_mode=str(settings["base_mode"]),
        scenario=str(settings["scenario"]),
        catalogue=str(settings["catalogue"]),
        select=str(settings["select"]),
        timing=str(settings["timing"]),
        driver=str(settings["driver"]),
        episode_duration=_float(settings["episode_duration"], "spec.episode_duration"),
        worlds=_str_list(raw.get("worlds"), "spec.worlds"),
        arms=arms,
        cells=tuple(cells),
        seed_start=seed_start,
        seed_count=seed_count,
    )

    _check_baselines(spec)
    return spec


def _check_baselines(spec: Spec) -> None:
    """Every drive family with a learned planner needs its own classical control.

    ``robot.mobile:=drl`` selects a chassis matching the planner's ``action_type``, so a
    differential arm and an omnidirectional one differ by robot body as well as by policy.
    Comparing across families, or against one shared baseline, measures the chassis.
    """
    for drive, arms in spec.families().items():
        if any(a.is_baseline for a in arms):
            continue
        learned = sorted(a.id for a in arms if not a.is_baseline)
        raise SpecError(
            f"spec.arms: drive family {drive!r} has learned planner(s) {learned} and no baseline. "
            f"Add a classical arm on the same chassis (e.g. mobile: nav2), or the comparison "
            f"has no control."
        )


def load_spec(path: str | os.PathLike[str]) -> Spec:
    """Read and validate a spec file."""
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except FileNotFoundError:
        raise SpecError(f"spec: no such file {p}") from None
    except yaml.YAMLError as exc:
        raise SpecError(f"spec: {p} is not valid YAML: {exc}") from None
    return parse_spec(raw, name=p.stem)
