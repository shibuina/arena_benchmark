"""Edge-case variants: the same case, a structurally different mechanism.

`variant_plan.md` §3. Each of the nine cases keeps its identity and gains two or three
variants, and the rule for siblings is the same as for base variants: they differ in a
*mechanism* a reviewer can name, never in a seed. A variant is a delta over the case's block -
a few keys replaced, sometimes a key dropped - and `check_table` refuses two siblings that
carry the same mechanism tag.

**What is not here.** No cross product. A case variant runs on one base variant, chosen by
`backdrop_for`: rotation through the profile's base variants unless the recipe pins a cast it
needs (`stampede_dense` wants the `stream_led` flow it raises). Every pair of sibling case
variants therefore differs in mechanism *and* backdrop, and every backdrop is a shipped base,
which is that entry's control.

**Delay.** Every variant whose block carries a `waypoints:` plan fires it `DELAY_S` seconds in.
That is the one delay the mode can honour - `scenario_block.DELAYABLE` is `objects` and
`waypoints`, and a `delay` on anything else is refused rather than ignored, because a
kinematic knob cannot be rewritten mid-episode. Injected encounters are delayed *spatially*
instead: `encounter_at: 0.5` on a route that now takes at least 30 s puts the meeting 15 s or
more into the episode. Ten seconds is long enough for a stream to ramp (`RAMP_AT` is 30 s, so
it is a third of the way up), for the robot to be under way, and for the base population to
have left its marks - and short enough that the perturbation still owns most of the episode.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import attrs

from ..scenario.populate import MODELS, _centroid, _polygon, clearance
from ..scenario.variants import BASE_RECIPES, BaseRecipe

#: Seconds into the episode a time-driven perturbation fires. See the module docstring.
DELAY_S = 10.0

#: A doorway blocker needs a door this close to the robot's route, metres, or the case is
#: refused for that world: a blocker in a doorway the robot never passes obstructs nothing.
DOOR_MAX_DIST_M = 3.0

#: Name of the agent a `doorway` variant authors into the scenario, and selects.
DOOR_BLOCKER = "door_blocker"


@attrs.frozen(kw_only=True)
class Variant:
    """One variant of one case."""

    id: str
    #: Categorical tag; siblings must differ on it.
    mechanism: str
    note: str
    #: Keys replaced in the case's block. A nested `waypoints` mapping is merged, not replaced.
    params: Mapping[str, Any] = attrs.Factory(dict)
    #: Case block keys removed before the variant's are applied.
    drop: frozenset[str] = frozenset()
    #: Pin the backdrop to the base variant with this cast, instead of rotating.
    backdrop: str | None = None
    #: Number of transit zones the world must declare (for a rally to a *different* exit).
    needs_exits: int = 0
    #: A Level B knob: the base must carry `regions:`.
    needs_regions: bool = False
    #: Authors a blocker at the nearest door and selects it.
    door: bool = False
    #: Whether to apply `DELAY_S` when the block turns out to be delayable.
    delay: bool = True


def _v(id: str, mechanism: str, note: str, **kw: Any) -> Variant:  # noqa: A002 - matches the field
    return Variant(id=id, mechanism=mechanism, note=note, **kw)


CASE_VARIANTS: dict[str, tuple[Variant, ...]] = {
    "blocking": (
        _v("stand", "stationary", "stands still mid-route"),
        _v("crawl", "slow_mover", "walks ahead at a crawl, in the robot's direction - to be overtaken, not stopped",
           params={"approach_angle": 0.0, "level": 0.2}),
        _v("doorway", "doorway", "stands in the doorway nearest the route",
           door=True,
           params={"mode": "perturb", "inject": 0, "select": DOOR_BLOCKER,
                   "knob": "desired_velocity", "level": 0.0, "catalogue": "level_a"},
           drop=frozenset({"place", "approach_angle", "encounter_at"})),
    ),
    "approach": (
        _v("head_on", "head_on", "walks straight at the robot"),
        _v("crossing", "crossing", "comes in from the robot's right", params={"approach_angle": 90.0}),
        _v("overtake", "overtake", "comes up fast from behind",
           params={"approach_angle": 0.0, "knob": "desired_velocity", "level": 0.95, "catalogue": "level_a"}),
    ),
    "denial": (
        _v("pair", "conversation", "two people talk for as long as it takes"),
        _v("huddle", "huddle", "a conversation wide enough to fill the corridor",
           params={"knob": "interaction_radius", "level": 2.0}),
    ),
    "stubborn": (
        _v("one", "single", "the one agent on the route does not yield"),
        _v("crowd", "crowd", "nobody yields", params={"scope": "all"}),
    ),
    "blackout": (
        _v("slow", "slow", "everyone slows to a shuffle"),
        _v("blind", "blind", "everyone sees a fraction as far",
           params={"knob": "vision_range", "level": 0.05}),
    ),
    "erratic": (
        _v("jitter", "jitter", "frequent, short reroutes",
           params={"waypoints": {"period": 2.0, "radius": 1.5}}),
        _v("wander", "wander", "rare, long reroutes",
           params={"waypoints": {"period": 6.0, "radius": 5.0}}),
    ),
    "evacuation": (
        _v("orderly", "orderly", "everyone leaves at a walking pace", params={"level": 0.5}),
        _v("panic", "panic", "everyone runs for the exit"),
        _v("far_exit", "far_exit", "everyone heads for the exit farthest from the robot", needs_exits=2),
    ),
    "stampede": (
        _v("push", "repulsion", "the crowd stops keeping its distance"),
        _v("rush", "speed", "the crowd hurries back",
           params={"knob": "desired_velocity", "level": 0.95, "catalogue": "level_a"}),
        _v("dense", "density", "more people arrive than can leave",
           needs_regions=True, backdrop="stream_led",
           params={"knob": "crowd_density", "level": 24, "catalogue": "level_b"},
           drop=frozenset({"scope"})),
    ),
    "formation_break": (
        _v("early", "early", "the queue scatters when the robot is still 5 m out",
           params={"trigger": {"robot_within": 5.0}}, delay=False),
        _v("late", "late", "the queue holds until the robot is 1.5 m away",
           params={"trigger": {"robot_within": 1.5}}, delay=False),
    ),
}


def check_table(table: Mapping[str, Sequence[Variant]] = CASE_VARIANTS) -> None:
    """Refuse siblings that share a mechanism, and cases with fewer than two variants."""
    for case_id, variants in table.items():
        if len(variants) < 2:
            raise ValueError(f"{case_id!r} has {len(variants)} variant(s); two is the floor")
        seen: dict[str, str] = {}
        for v in variants:
            if v.mechanism in seen:
                raise ValueError(
                    f"{case_id}: {v.id!r} and {seen[v.mechanism]!r} share mechanism {v.mechanism!r}"
                )
            seen[v.mechanism] = v.id


def variants_for(case_id: str) -> tuple[Variant, ...]:
    return CASE_VARIANTS.get(case_id, ())


def backdrop_for(profile: str, index: int, variant: Variant) -> BaseRecipe:
    """Which base variant this case variant runs on.

    Rotation by the variant's index, so siblings land on different backdrops; a pinned cast
    wins where the recipe declares one and the profile has a base of that cast.
    """
    recipes = BASE_RECIPES[profile]
    if variant.backdrop:
        for recipe in recipes:
            if recipe.cast == variant.backdrop:
                return recipe
    return recipes[index % len(recipes)]


def merge_params(case_params: Mapping[str, Any], variant: Variant) -> dict[str, Any]:
    """The case's block with the variant applied: drops first, then keys, `waypoints` merged."""
    out = {k: v for k, v in case_params.items() if k not in variant.drop}
    for key, value in variant.params.items():
        if key == "waypoints" and isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = {**out[key], **value}
        elif key == "trigger" and isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def delayable(block: Mapping[str, Any]) -> bool:
    """Mirror of `scenario_block.DELAYABLE`: only time-driven mechanisms can wait."""
    return bool(block.get("waypoints")) or bool(block.get("objects"))


# --- world-specific resolution ------------------------------------------------------------

def _doors(zones: Sequence[Mapping[str, Any]]) -> list[tuple[str, tuple[float, float], float]]:
    """Every door as (name, midpoint, yaw along the door line)."""
    out = []
    for zone in zones:
        if not isinstance(zone, Mapping):
            continue
        for door in zone.get("doors") or []:
            if not isinstance(door, Mapping) or not door.get("name"):
                continue
            s, e = door.get("start") or {}, door.get("end") or {}
            try:
                sx, sy, ex, ey = float(s["x"]), float(s["y"]), float(e["x"]), float(e["y"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append((str(door["name"]), ((sx + ex) / 2.0, (sy + ey) / 2.0), math.atan2(ey - sy, ex - sx)))
    return out


def nearest_door(
    zones: Sequence[Mapping[str, Any]],
    route: Sequence[tuple[float, float]],
    max_dist: float = DOOR_MAX_DIST_M,
) -> tuple[str, tuple[float, float], float] | None:
    """The door closest to the robot's route, if one is within `max_dist`. Ties break on name."""
    doors = _doors(zones)
    if not doors or len(route) < 1:
        return None
    best = min(doors, key=lambda d: (round(clearance(d[1], route), 3), d[0]))
    return best if clearance(best[1], route) <= max_dist else None


def door_blocker(zones: Sequence[Mapping[str, Any]], route: Sequence[tuple[float, float]]) -> dict[str, Any] | None:
    """The agent a `doorway` variant authors: standing in the nearest door, facing along it.

    A literal, not `pose: <door>`: a door polygon is a thin strip and the runtime resolver's
    0.4 m disc may fail in it, on which it warns and returns a blocked point. The midpoint of
    the opening is free by construction.
    """
    found = nearest_door(zones, route)
    if found is None:
        return None
    _, (x, y), yaw = found
    return {
        "name": DOOR_BLOCKER,
        "model": MODELS[0],
        "pose": [round(x, 3), round(y, 3), round(yaw, 3)],
        "agent": {"agent_type": "adult", "desired_velocity": 1.0},
        # Stands where it is put: this one agent is the obstruction, and `once` with a single
        # waypoint is how the backend spells "stay".
        "waypoints": [[round(x, 3), round(y, 3), 0.0]],
        "waypoint_mode": "once",
    }


def far_exit(
    zones: Sequence[Mapping[str, Any]],
    route: Sequence[tuple[float, float]],
    overrides: Mapping[str, str] | None = None,
) -> str | None:
    """The transit zone farthest from the robot's route, when the world has at least two.

    Two, because with one exit "the far exit" is "the exit", and the variant would be the
    `panic` case under another name - the thing the whole variant layer exists to refuse.
    """
    from ..scenario.zones import Role, zones_by_role  # noqa: PLC0415

    transit = zones_by_role(zones, overrides).get(Role.TRANSIT) or []
    polys = {str(z["name"]): _polygon(z) for z in zones if isinstance(z, Mapping) and z.get("name")}
    usable = [n for n in transit if len(polys.get(n) or []) >= 3]
    if len(usable) < 2:
        return None
    return max(usable, key=lambda n: (round(clearance(_centroid(polys[n]), route), 3), n))


__all__ = [
    "CASE_VARIANTS",
    "DELAY_S",
    "DOOR_BLOCKER",
    "DOOR_MAX_DIST_M",
    "Variant",
    "backdrop_for",
    "check_table",
    "delayable",
    "door_blocker",
    "far_exit",
    "merge_params",
    "nearest_door",
    "variants_for",
]
