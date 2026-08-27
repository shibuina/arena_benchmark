"""Grouping the manifest by what a relaunch actually costs.

Boot to ``EPISODE STARTED`` costs 60-90 s; an episode reset costs seconds. So the schedule
is decided by one question per field: can the orchestrator change this between episodes, or
does it need a new stack? Everything in :data:`LAUNCH_KEYS` needs a new stack. Everything
else - the whole ``task.edge_case.*`` surface, the seed - is a parameter set, and the
existing sweep runner already drives those inside one launch.

Grouping by that key first is the difference between a run that boots once per trial and one
that boots once per few dozen. On the reference confirm budget it is the difference between
roughly 170 hours and roughly 21.

``scenario`` is in the key even though ``task.scenario.file`` is a node parameter. Changing
it mid-session leaks flow regions across resets, so in practice it is launch-time whatever
the parameter surface says. That is an operational fact, not a schema one, and it belongs
here where the schedule is decided.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import attrs

from ..spec.expand import Episode

#: Episode fields that cannot change without a new stack. The order is the sort order of
#: shards, chosen so a run groups by world then by arm - which is also the order that keeps
#: a partially finished run interpretable.
LAUNCH_KEYS: tuple[str, ...] = (
    "sim",
    "world",
    "scenario",
    "human",
    "tm_robots",
    "tm_obstacles",
    "robot",
    "mobile",
    "planner",
    "global_planner",
    "rate_hz",
    "arm_extra",
)


@attrs.frozen(kw_only=True)
class Shard:
    """One stack's worth of work: the launch that starts it and the episodes it runs."""

    key: tuple[tuple[str, Any], ...]
    episodes: tuple[Episode, ...]

    @property
    def fields(self) -> dict[str, Any]:
        return dict(self.key)

    @property
    def world(self) -> str:
        return str(self.fields["world"])

    @property
    def arm_id(self) -> str:
        """Every episode in a shard shares an arm, so the first one names it."""
        return self.episodes[0].arm_id

    @property
    def label(self) -> str:
        """Short human-readable name, used for the run directory and progress output."""
        return f"{self.world}__{self.arm_id}"

    def launch_args(self) -> dict[str, str]:
        """``k:=v`` arguments for ``arena launch``.

        ``tm_obstacles`` and ``tm_robots`` are passed explicitly rather than left to the
        launch defaults: a default that changed underneath a half-finished dataset would
        make two shards of the same run mean different things.
        """
        f = self.fields
        # `task.robots` / `task.obstacles`, not `tm_robots` / `tm_obstacles`: the latter are
        # deprecated aliases in `arena_bringup.substitutions.DEPRECATED_LAUNCH_ARGS`. They
        # still resolve, with a warning per launch - which across a few hundred shards is
        # noise that hides a real one.
        args: dict[str, str] = {
            "sim": str(f["sim"]),
            "world": str(f["world"]),
            "human": str(f["human"]),
            "robot": str(f["robot"]),
            "robot.mobile": str(f["mobile"]),
            "task.robots": str(f["tm_robots"]),
            "task.obstacles": str(f["tm_obstacles"]),
        }
        if f.get("planner"):
            args["robot.mobile.planner"] = str(f["planner"])
        if f.get("global_planner"):
            args["robot.mobile.global_planner"] = str(f["global_planner"])
        if f.get("rate_hz") is not None:
            args["robot.mobile.rate"] = f"{float(f['rate_hz']):g}"
        # Arm-level escape hatch, applied last so a spec can override anything above it.
        args.update({k: str(v) for k, v in f.get("arm_extra") or ()})
        return args


def shard(episodes: Iterable[Episode]) -> list[Shard]:
    """Group episodes into stacks, deterministically.

    Within a shard, episodes keep their manifest order. Across shards the order is the tuple
    order of :data:`LAUNCH_KEYS`, so two expansions of the same spec schedule identically -
    which is what lets a resumed run be compared against an uninterrupted one.
    """
    groups: dict[tuple[tuple[str, Any], ...], list[Episode]] = {}
    for ep in episodes:
        key = tuple((field, getattr(ep, field)) for field in LAUNCH_KEYS)
        groups.setdefault(key, []).append(ep)

    return [
        Shard(key=key, episodes=tuple(eps))
        for key, eps in sorted(groups.items(), key=lambda kv: _sort_key(kv[0]))
    ]


def _sort_key(key: Sequence[tuple[str, Any]]) -> tuple[str, ...]:
    """Sortable form of a shard key. Values are heterogeneous (rate_hz is a float or None),
    so they are compared as strings rather than natively."""
    return tuple(f"{name}={value!r}" for name, value in key)


def summarise(shards: Sequence[Shard]) -> dict[str, Any]:
    """Counts worth printing before a run starts, and worth recording with its output."""
    return {
        "shards": len(shards),
        "episodes": sum(len(s.episodes) for s in shards),
        "worlds": sorted({s.world for s in shards}),
        "arms": sorted({s.arm_id for s in shards}),
        "episodes_per_shard": {s.label: len(s.episodes) for s in shards},
    }
