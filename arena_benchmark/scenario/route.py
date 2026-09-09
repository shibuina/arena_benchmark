"""The robot's route in a base scenario: the world's shipped route, always long enough.

An episode ends when the robot completes its phases, so a 12 m leg at 0.5 m/s is a 24 s
episode - too short for a crowd to form or an edge case with an onset to fire before it is
over. `robot_for` writes phases in the `phases` form that shuttle the shipped endpoints until
the route takes at least `MIN_EPISODE_S` at `ROBOT_SPEED_MPS`, the faster of the two controller
defaults in `arena_robots/config/nav2/controllers` (mppi/rpp/hateb 0.5; dwb 0.26); sizing for
the faster one means the floor holds under either. The robot spec is copied, never invented:
every world ships a scenario with a working route. (Until 2026-08-28 this lived in
`variants.py` with a reversed-route variant; bases are one per profile now.)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .populate import _centroid, _polygon

#: Shortest episode worth measuring, seconds. The robot's route is extended until it takes at
#: least this long at `ROBOT_SPEED_MPS`; pedestrians get routes at least this long to walk.
MIN_EPISODE_S = 30.0

#: Hard episode budget, seconds: the planner queue dwells 60 sim-seconds per scenario
#: (2026-09-01, was 240), so travel plus the case's own delays must fit inside it. The budget
#: outranks the floor: a route is never extended past `MAX_ROUTE_M`.
MAX_EPISODE_S = 60.0
#: Seconds of the budget reserved for the case to play out: onsets and holds are capped at
#: 30 s (fractions of the budget) and partly overlap the drive.
CASE_RESERVE_S = 20.0

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

#: Metres of route the episode budget affords: travel time is what is left after the case's
#: reserve, at the faster controller's speed.
MAX_ROUTE_M = (MAX_EPISODE_S - CASE_RESERVE_S) * ROBOT_SPEED_MPS

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
    robot: Mapping[str, Any],
    zones: Sequence[Mapping[str, Any]],
    *,
    min_seconds: float = MIN_EPISODE_S,
    speed: float = ROBOT_SPEED_MPS,
) -> dict[str, Any]:
    """The robot spec for a base: the shipped route, always long enough.

    Written in the `phases` form regardless of how the world wrote it - `goal:` is deprecated
    by `RobotGoal.phase_list`. Cut to the shortest prefix that takes `min_seconds` at `speed`,
    or extended by shuttling between the shipped endpoints until it does. Both keep to points
    the world demonstrates are reachable from each other, which is the whole reason the robot
    is copied rather than planned.
    """
    ends = _endpoints(robot)

    need = min_seconds * speed * ROUTE_MARGIN
    # The shortest prefix of the shipped route that is long enough. A shipped route can be far
    # longer than an episode wants - the acoustics corridors ship a 200-leg shuttle authored
    # for a robot that paces indefinitely - and copying that verbatim is a 33-minute episode.
    # A prefix is still a route the world demonstrates is drivable, which is the whole reason
    # the robot is copied rather than planned.
    seq = [ends[0]]
    for point in ends[1:]:
        seq.append(point)
        if route_length(seq, zones) >= min(need, MAX_ROUTE_M) or len(seq) - 1 >= MAX_LEGS:
            break

    if len(ends) >= 2:
        # The other direction: a shipped route too short for the floor is bounced between its
        # endpoints - back to the start, forward again - until it is long enough. The episode
        # budget outranks the floor: never bounce past `MAX_ROUTE_M`.
        back, forth = list(reversed(ends[:-1])), list(ends[1:])
        bounce = [back, forth]
        turn = 0
        capped = False
        while not capped and route_length(seq, zones) < need and len(seq) - 1 < MAX_LEGS:
            added = False
            for point in bounce[turn % 2]:
                if len(seq) - 1 >= MAX_LEGS:
                    break
                if route_length([*seq, point], zones) > MAX_ROUTE_M:
                    capped = True
                    break
                seq.append(point)
                added = True
                if route_length(seq, zones) >= need:
                    break
            turn += 1
            if not added:
                break
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
    "CASE_RESERVE_S",
    "MAX_EPISODE_S",
    "MAX_LEGS",
    "MAX_ROUTE_M",
    "MIN_EPISODE_S",
    "ROBOT_SPEED_MPS",
    "ROUTE_MARGIN",
    "robot_for",
    "route_length",
    "route_seconds",
]
