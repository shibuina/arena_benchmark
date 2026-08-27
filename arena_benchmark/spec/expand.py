"""Spec -> manifest: every episode the benchmark intends to run, named before it runs.

Expansion happens once, up front, and the manifest is the run's contract. Everything
downstream reads it rather than re-deriving: the orchestrator schedules from it, resume
diffs against it, and assembly joins to it. A row that is not in the manifest did not
belong in the run, and a manifest row with no measurement is a gap the summary has to
account for rather than quietly omit.

The product is ``worlds x cells x arms x seeds``, in that nesting order so that a manifest
read top to bottom groups the way a person thinks about it. The orchestrator regroups for
cost; that is its job, not this one's.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import attrs

from .schema import Arm, Cell, Spec
from .uid import cell_uid, episode_uid


@attrs.frozen(kw_only=True)
class Episode:
    """One intended run. Immutable, fully resolved - no defaults left to apply later."""

    episode_uid: str
    cell_uid: str

    spec_name: str
    world: str
    scenario: str
    sim: str
    human: str
    tm_robots: str
    tm_obstacles: str
    base_mode: str
    driver: str
    episode_duration: float
    select: str
    timing: str

    cell_id: str
    knob: str
    level: float
    catalogue: str
    params: Mapping[str, Any]
    note: str

    arm_id: str
    robot: str
    mobile: str
    planner: str
    global_planner: str
    drive: str
    rate_hz: float | None
    #: Arm-level launch overrides, as sorted pairs so the record stays hashable and can
    #: take part in the shard key.
    arm_extra: tuple[tuple[str, str], ...]

    seed: int

    def as_row(self) -> dict[str, Any]:
        """A flat JSON-safe mapping, for ``manifest.jsonl`` and the assembled table."""
        row = attrs.asdict(self)
        row["params"] = dict(self.params)
        row["arm_extra"] = dict(self.arm_extra)
        return row


def expand(spec: Spec) -> list[Episode]:
    """Every episode the spec describes.

    Raises on a uid collision. Two cells that resolve to identical parameters are the one
    failure mode a content hash cannot paper over: the rows would be indistinguishable in
    every downstream join, so the spec has to say which one it meant.
    """
    episodes = list(_iter_episodes(spec))

    seen: dict[str, Episode] = {}
    for ep in episodes:
        clash = seen.get(ep.episode_uid)
        if clash is not None:
            raise ValueError(
                f"spec {spec.name!r}: cells {clash.cell_id!r} and {ep.cell_id!r} produce an "
                f"identical episode in world {ep.world!r} on arm {ep.arm_id!r} - they differ "
                f"only by name. Give them different parameters or drop one."
            )
        seen[ep.episode_uid] = ep

    return episodes


def _iter_episodes(spec: Spec) -> Iterator[Episode]:
    context = spec.context()
    for world in spec.worlds:
        for cell in spec.cells:
            # A cell authored inside a world's scenario describes that world's geometry.
            if cell.world and cell.world != world:
                continue
            # A cell authored inside a scenario file carries its own; otherwise the
            # spec-wide one applies.
            scenario = cell.scenario or spec.scenario
            cuid = cell_uid(
                world=world,
                scenario=scenario,
                knob=cell.knob,
                level=cell.level,
                params=cell.params,
                context=context,
            )
            for arm in spec.arms:
                for seed in spec.seeds:
                    yield _episode(spec, world, cell, arm, seed, cuid, scenario)


def _episode(spec: Spec, world: str, cell: Cell, arm: Arm, seed: int, cuid: str, scenario: str) -> Episode:
    return Episode(
        episode_uid=episode_uid(cell=cuid, arm_id=arm.id, arm=arm.identity(), seed=seed),
        cell_uid=cuid,
        spec_name=spec.name,
        world=world,
        scenario=scenario,
        sim=spec.sim,
        human=spec.human,
        tm_robots=spec.tm_robots,
        tm_obstacles=spec.tm_obstacles,
        base_mode=spec.base_mode,
        driver=spec.driver,
        episode_duration=spec.episode_duration,
        select=spec.select,
        timing=spec.timing,
        cell_id=cell.id,
        knob=cell.knob,
        level=cell.level,
        # A cell may name its own catalogue, which is what lets level_a and level_d cells
        # share one spec; otherwise the run-wide default applies.
        catalogue=cell.catalogue or spec.catalogue,
        params=dict(cell.params),
        note=cell.note,
        arm_id=arm.id,
        robot=arm.robot,
        mobile=arm.mobile,
        planner=arm.planner,
        global_planner=arm.global_planner,
        drive=arm.drive,
        rate_hz=arm.rate_hz,
        arm_extra=tuple(sorted((str(k), str(v)) for k, v in arm.extra.items())),
        seed=seed,
    )
