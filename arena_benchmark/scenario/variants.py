"""Base-scenario variants: the same profile, structurally different situations.

`variant_plan.md` §2. A variant is a *named recipe* that differs from its siblings on a
categorical axis - never a re-seed. Seeds are the replication dimension inside a `cell_uid`;
variants are different cells. Three axes:

* **ROUTE** - `fwd` runs the world's shipped route as written; `rev` swaps its ends. The robot
  spec is still copied, never invented: reversal is the one transform that cannot produce a
  route the world does not already demonstrate is drivable.
* **FLOW** - which way the source→sink stream runs relative to the route: `cross`, `counter`,
  `co`, or `none` for no stream at all.
* **CAST** - `dispersed` spreads dwellers over every target zone; `clustered` packs them into
  the one or two zones nearest the route; `stream_led` halves them and raises the flow.

**Sibling recipes must differ on at least two of the three.** `check_table` enforces it over
the table below, and a test runs it, so a recipe that is a re-seed of its neighbour cannot be
added without someone saying so.

This module also owns the **robot route's length**. An episode ends when the robot completes
its phases, so a 12 m leg at 0.5 m/s is a 24 s episode - too short for a crowd to form, a
stream to ramp, or an edge case with a `delay` to fire before it is over. `robot_for` writes
phases that shuttle the shipped endpoints until the route takes at least `MIN_EPISODE_S` at
`ROBOT_SPEED_MPS`, which is the faster of the two controller defaults in
`arena_robots/config/nav2/controllers` (mppi/rpp/hateb 0.5; dwb 0.26). Sizing for the faster
one means the floor holds under either.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import attrs

from .populate import _centroid, _polygon

#: Shortest episode worth measuring, seconds. The robot's route is extended until it takes at
#: least this long at `ROBOT_SPEED_MPS`; pedestrians get routes at least this long to walk.
MIN_EPISODE_S = 30.0

#: The faster of nav2's controller defaults for `max_linear_vel`. Planning for the faster
#: controller is what makes the floor hold for the slower one too.
ROBOT_SPEED_MPS = 0.5

#: Margin on the route length, for the goal tolerance (1.0 m per leg) and the corners a
#: planner cuts. 1.2 turns a 30 s floor into a 36 s target at nominal speed.
ROUTE_MARGIN = 1.2

#: Never write a route with more legs than this. A world whose endpoints are 2 m apart would
#: otherwise get a 20-leg shuttle, which is a robot pacing a doorway rather than traversing a
#: building.
MAX_LEGS = 8

ROUTES = ("fwd", "rev")
FLOWS = ("cross", "counter", "co", "none")
CASTS = ("dispersed", "clustered", "stream_led")


@attrs.frozen(kw_only=True)
class BaseRecipe:
    """One base variant: a profile plus a point on each of the three axes."""

    id: str
    profile: str
    route: str
    flow: str
    cast: str
    note: str

    def __attrs_post_init__(self) -> None:
        if self.route not in ROUTES or self.flow not in FLOWS or self.cast not in CASTS:
            raise ValueError(f"recipe {self.id!r} is off the axes: {self.route}/{self.flow}/{self.cast}")
        if not self.id.startswith(f"{self.profile}_") and self.id != self.profile:
            raise ValueError(f"recipe {self.id!r} must be named after its profile {self.profile!r}")

    def axes(self) -> tuple[str, str, str]:
        return (self.route, self.flow, self.cast)


def distinct_axes(a: BaseRecipe, b: BaseRecipe) -> int:
    """How many of the three axes two recipes differ on."""
    return sum(x != y for x, y in zip(a.axes(), b.axes(), strict=True))


#: The table. `_a` is always today's base, so the dataset that existed before variants is the
#: first variant of each profile and nothing already measured moves.
BASE_RECIPES: dict[str, tuple[BaseRecipe, ...]] = {
    "normal": (
        BaseRecipe(id="normal_a", profile="normal", route="fwd", flow="cross", cast="dispersed",
                   note="people spread over the working zones, a stream crossing the route"),
        BaseRecipe(id="normal_b", profile="normal", route="rev", flow="counter", cast="clustered",
                   note="a knot of people beside the route, traffic coming head-on"),
        BaseRecipe(id="normal_c", profile="normal", route="fwd", flow="co", cast="stream_led",
                   note="a thin cast and a stream walking the robot's own way"),
    ),
    "break": (
        BaseRecipe(id="break_a", profile="break", route="fwd", flow="cross", cast="dispersed",
                   note="people spread over the break zones, a stream crossing the route"),
        BaseRecipe(id="break_b", profile="break", route="rev", flow="none", cast="clustered",
                   note="one crowded lounge and no stream - a doorway crowd"),
    ),
    "busy": (
        BaseRecipe(id="busy_a", profile="busy", route="fwd", flow="cross", cast="dispersed",
                   note="a full building, a stream crossing the route"),
        BaseRecipe(id="busy_b", profile="busy", route="fwd", flow="counter", cast="stream_led",
                   note="pushing against heavy oncoming traffic"),
        BaseRecipe(id="busy_c", profile="busy", route="rev", flow="co", cast="clustered",
                   note="a dense knot beside the route and a stream following the robot"),
    ),
    "night": (
        BaseRecipe(id="night_a", profile="night", route="fwd", flow="none", cast="dispersed",
                   note="sparse and quiet - the low-density control"),
        BaseRecipe(id="night_b", profile="night", route="rev", flow="counter", cast="stream_led",
                   note="a skeleton crew and a thin stream coming the other way"),
    ),
    "empty": (
        BaseRecipe(id="empty", profile="empty", route="fwd", flow="none", cast="dispersed",
                   note="nobody - the no-population control, and deliberately only one"),
    ),
}

BY_ID: dict[str, BaseRecipe] = {r.id: r for rs in BASE_RECIPES.values() for r in rs}


def check_table(table: Mapping[str, Sequence[BaseRecipe]] = BASE_RECIPES) -> None:
    """Refuse a table whose siblings are not distinct. Runs in a test; cheap enough to run always."""
    for profile, recipes in table.items():
        for i, a in enumerate(recipes):
            if a.profile != profile:
                raise ValueError(f"{a.id!r} is filed under {profile!r} but says {a.profile!r}")
            for b in recipes[i + 1:]:
                if distinct_axes(a, b) < 2:
                    raise ValueError(
                        f"{a.id!r} and {b.id!r} share {3 - distinct_axes(a, b)} of 3 axes - a variant "
                        f"must differ from its siblings on at least two"
                    )


# --- the robot's route -------------------------------------------------------------------

def _flip(pose: str | Sequence[float]) -> str | list[float]:
    """A pose facing the other way: a literal's yaw turns by pi; a zone name is unchanged."""
    if isinstance(pose, str):
        return pose
    if isinstance(pose, Sequence) and len(pose) >= 3 and not isinstance(pose, str):
        yaw = (float(pose[2]) + math.pi + math.pi) % (2 * math.pi) - math.pi
        return [float(pose[0]), float(pose[1]), round(yaw, 4)]
    return list(pose) if not isinstance(pose, str) else pose


def _endpoints(robot: Mapping[str, Any]) -> list[Any]:
    out: list[Any] = [robot["start"]]
    for phase in robot.get("phases") or []:
        if isinstance(phase, Mapping) and (goto := phase.get("goto")) is not None:
            out.append(goto)
    if (goal := robot.get("goal")) is not None:
        out.append(goal)
    return out


def _xy(pose: str | Sequence[float], centroids: Mapping[str, tuple[float, float]]) -> tuple[float, float] | None:
    if isinstance(pose, str):
        return centroids.get(pose)
    if isinstance(pose, Sequence) and len(pose) >= 2:
        return (float(pose[0]), float(pose[1]))
    return None


def route_length(endpoints: Sequence[Any], zones: Sequence[Mapping[str, Any]]) -> float:
    """Metres along a polyline of literals and zone names; names resolve to their centroids."""
    centroids = {
        str(z["name"]): _centroid(_polygon(z))
        for z in zones if isinstance(z, Mapping) and z.get("name") and len(_polygon(z)) >= 3
    }
    pts = [p for p in (_xy(e, centroids) for e in endpoints) if p is not None]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False))


def robot_for(
    recipe: BaseRecipe,
    robot: Mapping[str, Any],
    zones: Sequence[Mapping[str, Any]],
    *,
    min_seconds: float = MIN_EPISODE_S,
    speed: float = ROBOT_SPEED_MPS,
) -> dict[str, Any]:
    """The robot spec for one recipe: the shipped route, possibly reversed, always long enough.

    Written in the `phases` form regardless of how the world wrote it - `goal:` is deprecated
    by `RobotGoal.phase_list`. Cut to the shortest prefix that takes `min_seconds` at `speed`,
    or extended by shuttling between the shipped endpoints until it does. Both keep to points
    the world demonstrates are reachable from each other, which is the whole reason the robot
    is copied rather than planned.
    """
    ends = _endpoints(robot)
    if recipe.route == "rev":
        ends = [_flip(e) for e in reversed(ends)]

    need = min_seconds * speed * ROUTE_MARGIN
    # The shortest prefix of the shipped route that is long enough. A shipped route can be far
    # longer than an episode wants - the acoustics corridors ship a 200-leg shuttle authored
    # for a robot that paces indefinitely - and copying that verbatim is a 33-minute episode.
    # A prefix is still a route the world demonstrates is drivable, which is the whole reason
    # the robot is copied rather than planned.
    seq = [ends[0]]
    for point in ends[1:]:
        seq.append(point)
        if route_length(seq, zones) >= need or len(seq) - 1 >= MAX_LEGS:
            break

    if len(ends) >= 2:
        # The other direction: a shipped route too short for the floor is bounced between its
        # endpoints - back to the start, forward again - until it is long enough.
        back, forth = list(reversed(ends[:-1])), list(ends[1:])
        bounce = [back, forth]
        turn = 0
        while route_length(seq, zones) < need and len(seq) - 1 < MAX_LEGS:
            seq.extend(bounce[turn % 2])
            turn += 1
        del seq[MAX_LEGS + 1:]

    spec: dict[str, Any] = {"start": seq[0], "phases": [{"goto": p} for p in seq[1:]]}
    for key in ("start_floor", "goal_floor"):
        if robot.get(key):
            spec[key] = robot[key]
    return spec


def route_seconds(robot: Mapping[str, Any], zones: Sequence[Mapping[str, Any]], speed: float = ROBOT_SPEED_MPS) -> float:
    """How long the written route takes at nominal speed. For the CLI to report."""
    return route_length(_endpoints(robot), zones) / speed if speed > 0 else float("inf")


__all__ = [
    "BASE_RECIPES",
    "BY_ID",
    "CASTS",
    "FLOWS",
    "MAX_LEGS",
    "MIN_EPISODE_S",
    "ROBOT_SPEED_MPS",
    "ROUTES",
    "BaseRecipe",
    "check_table",
    "distinct_axes",
    "robot_for",
    "route_length",
    "route_seconds",
]
