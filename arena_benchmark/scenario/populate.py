"""Generating the four base scenarios for a world.

`normal`, `break`, `night`, `empty` — the population states of `new_plan_2.md` §1.2 (`busy`
is a Type A case now, injected by `promptgen.step_a`). All four are static: a population written into `scenario.yaml` at
authoring time, with no runtime machinery. `night`'s slower walking is a per-agent
`desired_velocity`, which is what keeps it a base scenario rather than an edge case.

**Every pedestrian is written in rooms, never in coordinates.** `pose: laboratory`,
`waypoints: [laboratory, central_hallway, patient_ward]`. Each name is resolved at run time by
`PointResolver` to a free point in that polygon, so a scenario says *who is where and where they
go* and the world's geometry decides the rest - regenerate the map, prune the furniture, and
the scenario still holds. Nothing here samples a coordinate.

What that gives up, stated once: the offline placement checks that used to keep a literal
pose 1.5 m off the robot's route are gone, because there is no pose to check. Two zone-level
rules stand in for them. An agent never *starts* in a zone the robot's own spec names, since
the robot resolves anywhere inside such a zone; and where `zone_screen` has found zones whose
every point clears the route, those are the homes. Everything else is a destination, which is
the point - a pedestrian who never crosses a corridor never meets the robot.

Three rules govern the population, and all three were learned by breaking them:

* **Everybody walks.** A behaviour-tree agent never follows its waypoints - `_advance_waypoints`
  skips `BehaviorTreeMovement` - and the shipped `talker.yaml` holds a conversation forever.
  Half of every generated population used to be talkers, so half of every population stood
  still. Now two agents per base are talkers, placed in the same room so they have each other
  to talk to, and every other agent is a builtin type that walks its rooms. Two, because a
  Level D case (`denial`) needs a conversation to lengthen and one person cannot hold one.
* **Everybody goes somewhere else.** A waypoint list holding only the agent's own room leaves
  it milling inside four walls. Every walker gets a route through one to four *other* zones,
  extended until it is at least `MIN_PED_ROUTE_M` long where the world allows.
* **The robot spec is copied, never invented.** Every world already ships a scenario with a
  working route. Deriving one from zone geometry would be a second, worse route planner.

Walkers are not all `adult`. Each is drawn from a **mix of kinematic types** (`KINEMATICS`,
`MIX`): the hurried, the distracted, the elderly, and - in residential worlds - children.
(Wheelchairs and carts were dropped 2026-08-31: Arena cannot embody them - the only pedestrian
meshes are bipedal adults, so the wide slow body the type promises never meets the robot's
sensors.) Every one is `mode: simple`, walks its
rooms, and carries every Level A field, so the knobs land on all of them.

A base scenario carries **no** `regions:` (sources/sinks): the population is exactly the cast
written here, so every pedestrian on the floor is one a case can name, own and score. (Streams of
anonymous `flow_*` agents used to be added for the knob-era Level B tail; removed 2026-08-28.)
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .zones import Role, zones_by_role

#: Behaviour-tree agent type shipped with this package and copied into generated scenarios.
#: Without it every agent is a builtin type (`mode: simple`) and every Level D case is
#: structurally inert - measured, `blocking` aborted 14 times in a row before this existed.
BT_AGENT_TYPE = "talker.yaml"

#: Behaviour-tree agents come in pairs, one pair per room: a conversation needs two people, and
#: a talker with nobody in range waits out its patience and stands there anyway. `TALKER_SHARE`
#: of the cast (rounded down to whole pairs, at least one pair) talks; the rest walks. A cast
#: too small for a pair plus two walkers has no talkers - a base where nobody walks is furniture.
TALKERS = 2
TALKER_SHARE = 0.3
MIN_WALKERS = 2

#: Dwellers per zone of the profile's role at the `normal` profile, before the world's crowd
#: factor. Was 1 until 2026-08-28: seven people in a thirteen-room hospital read as empty.
PEOPLE_PER_ZONE = 2
#: Fewest people a populated base carries, whatever the world's size.
MIN_CAST = 4

#: Metres of clearance between a pedestrian's home zone and the robot's route polyline, for
#: `zone_screen` to decide which zones may be homes. Nothing in this module measures it.
MIN_ROUTE_CLEARANCE = 1.5

#: How many places a walker visits besides its own, and how often. Capped per agent by the number
#: of distinct zones the world actually offers.
ROUTE_STOPS: tuple[tuple[int, float], ...] = ((1, 0.34), (2, 0.30), (3, 0.22), (4, 0.14))

#: Shortest one-way route, metres between zone centroids, where the world allows it: about 30 s
#: of walking there and back at a nominal 1 m/s. `stops` is a floor; `_route` keeps adding zones
#: until the route is at least this long or the world runs out of zones to add.
MIN_PED_ROUTE_M = 15.0

#: The kinematic (`mode: simple`) pedestrian types a walker may be, with each type's
#: `desired_velocity` distribution as (mean, std, clip_low, clip_high) - mirrored from
#: humansim's `config/agent_types/*.yaml`, and a test reads those files and refuses drift.
#: Mirrored rather than imported because this package generates without ROS on the path.
#:
#: The per-agent speed written into the scenario is drawn from the type's own distribution
#: and scaled by the profile and the world's tempo, so a `night` elder is slower than a `busy`
#: one and both are slower than any `hurried`. Radius is *not* written: the adapter now lets
#: the type's radius stand unless a scenario says otherwise.
KINEMATICS: dict[str, tuple[float, float, float, float]] = {
    "adult":      (1.1, 0.12, 0.5, 1.5),
    "elder":      (0.75, 0.12, 0.3, 1.1),
    "child":      (1.0, 0.25, 0.4, 1.8),
    "hurried":    (1.7, 0.15, 1.3, 2.2),
    "distracted": (0.95, 0.12, 0.5, 1.3),
}

#: Who is about, per profile: relative weights over `KINEMATICS`. A type a building has no
#: business hosting (`ONLY_IN`) is dropped before drawing, and its weight goes to the rest.
MIX: dict[str, dict[str, int]] = {
    "normal": {"adult": 55, "hurried": 15, "distracted": 15, "elder": 10, "child": 5},
    "break":  {"adult": 50, "distracted": 30, "elder": 10, "child": 10},
    "night":  {"adult": 50, "elder": 30, "distracted": 10, "hurried": 10},
}

#: Who arrives when a building gets `busy` - the Type A case, not a base profile any more.
BUSY_MIX: dict[str, int] = {"adult": 50, "hurried": 30, "distracted": 15, "elder": 5}

#: Types that only appear in a kind of building. A child on a hospital ward is the sort of
#: thing a reviewer flags before anything else.
ONLY_IN: dict[str, tuple[str, ...]] = {
    "child": ("residential",),
}

#: Zone-name fragments of rooms nobody is placed in or sent to: sanitary rooms and cupboards.
#: Measured 2026-08-28 on hospital_1: walkers spawned in `wc_waiting` never left - its 0.7 m
#: stall doors are sealed in HumanSim's inflated grid - and six of eleven walkers stood still.
NOT_FOR_PEOPLE = ("wc", "toilet", "bath", "restroom", "lavatory", "stall", "storage", "closet", "elevator")


def for_people(names: Sequence[str]) -> list[str]:
    """`names` minus the rooms nobody is placed in or sent to (`NOT_FOR_PEOPLE`)."""
    return [n for n in names if not any(h in str(n).lower() for h in NOT_FOR_PEOPLE)]


#: Zone-name fragments that make a building a hospital for the purpose of the mix.
_HOSPITAL_HINTS = ("patient", "nurse", "ward", "exam", "pharmacy", "operating", "triage", "consult")

#: Per-world multipliers, drawn once per world, so two worlds generated from the same `--seed`
#: are not the same crowd in different rooms.
TEMPO_RANGE = (0.92, 1.08)
CROWD_RANGE = (0.8, 1.3)

#: Population multipliers and speed scaling, relative to the world's nominal count.
PROFILES: dict[str, dict[str, float]] = {
    "normal": {"count": 1.0, "speed": 1.0},
    "break":  {"count": 1.0, "speed": 1.0},
    "night":  {"count": 0.5, "speed": 0.8},
    "empty":  {"count": 0.0, "speed": 1.0},
}

#: Which zone role each profile populates.
TARGET_ROLE: dict[str, Role] = {
    "normal": Role.WORK,
    "break": Role.BREAK,
    "night": Role.WORK,
    "empty": Role.WORK,
}

#: HumanSim's default `max_velocity` distribution mean (`config/agent_types`), and the headroom a
#: faster walker gets over its own desired speed.
MAX_VELOCITY_DEFAULT = 1.5
MAX_VELOCITY_HEADROOM = 0.3

#: Meshes in shipped use. Shuffled per world so populations are not all led by the same mesh -
#: an unresolved model logs a warning and silently falls back to a default.
MODELS: tuple[str, ...] = (
    "female_adult_medical_01", "male_adult_medical_01", "male_adult_construction_01",
    "female_adult_business_02", "male_adult_construction_02", "female_adult_police_01",
)


def _polygon(zone: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(float(c["x"]), float(c["y"])) for c in zone.get("corners", []) if isinstance(c, Mapping)]


def _centroid(poly: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def _dist_to_segment(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay, bx, by = *a, *b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def _endpoints(robot: Mapping[str, Any]) -> list[Any]:
    """Every endpoint the robot spec names, in order: start, phase gotos, goal."""
    out: list[Any] = []
    if (start := robot.get("start")) is not None:
        out.append(start)
    for phase in robot.get("phases") or []:
        if isinstance(phase, Mapping) and (goto := phase.get("goto")) is not None:
            out.append(goto)
    if (goal := robot.get("goal")) is not None:
        out.append(goal)
    return out


def robot_zones(robot: Mapping[str, Any]) -> frozenset[str]:
    """Zones the robot spec names instead of giving coordinates for.

    A zone-named endpoint means the robot may be resolved *anywhere* in that zone, so the whole
    polygon is potentially the robot's. Nobody starts there.
    """
    return frozenset(e for e in _endpoints(robot) if isinstance(e, str))


def route_points(
    robot: Mapping[str, Any],
    zones: Sequence[Mapping[str, Any]] | None = None,
) -> list[tuple[float, float]]:
    """The robot's route as a polyline: start, then every phase goal or the single goal.

    A zone-named endpoint resolves to that zone's centroid when `zones` is given. Without it the
    endpoint is dropped, and dropping every endpoint is how this once returned an empty route
    for the zone-named worlds - which made every route check vacuously pass. Pass `zones`.
    """
    centroids = {
        str(z["name"]): _centroid(_polygon(z))
        for z in (zones or [])
        if isinstance(z, Mapping) and z.get("name") and len(_polygon(z)) >= 3
    }
    pts: list[tuple[float, float]] = []
    for endpoint in _endpoints(robot):
        if isinstance(endpoint, str):
            if (centre := centroids.get(endpoint)) is not None:
                pts.append(centre)
        elif isinstance(endpoint, Sequence):
            pts.append((float(endpoint[0]), float(endpoint[1])))
    return pts


def clearance(pt: tuple[float, float], route: Sequence[tuple[float, float]]) -> float:
    """Distance from a point to the robot's route polyline. Infinite when there is no route."""
    if len(route) < 2:
        return math.dist(pt, route[0]) if route else float("inf")
    return min(_dist_to_segment(pt, route[i], route[i + 1]) for i in range(len(route) - 1))


def nominal_count(zones_of_role: Sequence[str]) -> int:
    """The world's nominal population: `PEOPLE_PER_ZONE` per zone of the target role."""
    return max(1, PEOPLE_PER_ZONE * len(zones_of_role))


def cast_split(n: int) -> tuple[int, int]:
    """(talkers, walkers) for a cast of `n`: `TALKER_SHARE` in whole pairs - at least one pair
    - never leaving fewer than `MIN_WALKERS` walkers; no pair at all only when the cast is too
    small for a pair and two walkers."""
    pairs = max(1, int(n * TALKER_SHARE) // TALKERS)
    while pairs > 0 and n - pairs * TALKERS < MIN_WALKERS:
        pairs -= 1
    talkers = pairs * TALKERS
    return talkers, n - talkers


def _targets(profile: str, by_role: dict[Role, list[str]]) -> tuple[list[str], str]:
    """Zones this profile populates, and the basis on which they were chosen.

    The five profiles were designed around a workplace: `normal` puts people where work
    happens, `break` where they go when they are not working. **A residential world has no work
    zones at all** - `arena_arena_001` is living_room, kitchen, bathroom and two bedrooms - so
    the workplace mapping yields an empty population for three of the five profiles.

    The residential reading keeps the distinction the profiles exist to draw, by shifting it one
    step: the living areas become where people are, and the private rooms become where they go
    otherwise. That is an assumption about intent, so the caller announces it rather than
    letting a home quietly produce a different dataset from an office.
    """
    work, brk, private = by_role[Role.WORK], by_role[Role.BREAK], by_role[Role.PRIVATE]
    if work:
        return (brk or work, "workplace") if profile == "break" else (work, "workplace")

    if brk or private:
        # Residential: living areas are the active zones; private rooms are the retreat.
        return (private or brk, "residential") if profile == "break" else (brk or private, "residential")

    return by_role[Role.TRANSIT], "transit-only"


def _route(
    home: str,
    pool: Sequence[str],
    index: int,
    stops: int,
    centroids: Mapping[str, tuple[float, float]],
) -> list[str]:
    """Up to `stops` zones to walk to, none of them `home`, more if the walk is still short.

    Zones are taken in round-robin order from this agent's index, so neighbouring agents get
    different itineraries and the population spreads over the world instead of every agent
    shuttling to the same room. A route never visits the same room twice running.

    A world with one zone gets an empty route, and the caller writes the home twice: each
    reference resolves independently, so the agent walks between two points of the room rather
    than standing on one - the only walk a one-room world affords.
    """
    others = [z for z in pool if z != home and z in centroids]
    out: list[str] = []
    length, last = 0.0, centroids.get(home)
    for k in range(len(others)):
        if k >= stops and length >= MIN_PED_ROUTE_M:
            break
        zone = others[(index + k) % len(others)]
        if out and zone == out[-1]:
            continue
        if last is not None:
            length += math.dist(last, centroids[zone])
        last = centroids[zone]
        out.append(zone)
    return out


def _waypoint_mode(legs: int, rng: random.Random) -> str:
    """How an agent traverses its route, chosen to suit the route's shape.

    Verified against `agent_manager`'s advance logic: `reverse` shuttles back and forth, the
    only sensible reading of a two-point route; `repeat` cycles and wraps, so three or more stops
    become a circuit; `random` jumps to a random *other* waypoint, a real wander at three points
    and `reverse` in disguise at two.

    `once` is deliberately never emitted. It advances to the last waypoint and stops, so the
    agent becomes furniture for the rest of the episode - the same failure as an agent that
    never leaves its room, arrived at from the other direction.
    """
    if legs < 3:
        return "reverse"
    return rng.choices(("repeat", "reverse", "random"), weights=(0.5, 0.3, 0.2))[0]


def build(
    profile: str,
    *,
    zones: Sequence[Mapping[str, Any]],
    robot: Mapping[str, Any],
    overrides: Mapping[str, str] | None = None,
    seed: int = 0,
    homes: frozenset[str] = frozenset(),
    world: str = "",
) -> dict[str, Any]:
    """One base scenario document, every pedestrian in rooms - one base per profile.

    Deterministic in its inputs - `world` among them, so two worlds at one seed are not one
    crowd - and free of coordinates: what varies between regenerations is who is where and where
    they go, never a number the world's geometry should have decided.

    `homes` are the zones `zone_screen` cleared as starting rooms - every point in them off the
    robot's route. Empty means "no screen"; the target zones minus the robot's own then serve,
    and the CLI says so.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}")

    doc: dict[str, Any] = {"robots": [dict(robot)], "dynamic": []}
    spec = PROFILES[profile]
    if spec["count"] == 0.0:
        return doc

    by_role = zones_by_role(zones, overrides)
    targets, _basis = _targets(profile, by_role)
    if not targets:
        return doc

    centroids = {
        str(z["name"]): _centroid(_polygon(z))
        for z in zones if isinstance(z, Mapping) and z.get("name") and len(_polygon(z)) >= 3
    }
    route = route_points(robot, zones)
    forbidden = robot_zones(robot)

    # Where people start. The screened zones of the profile's role where the screen found any;
    # when those are fewer than three rooms, every other screened, non-private room joins so
    # the population is spread over the building rather than stacked in one or two rooms
    # (hospital_1: only two of six work zones pass the screen). Otherwise the targets minus the
    # rooms the robot itself may resolve into.
    starts = for_people([z for z in targets if z in homes and z in centroids])
    if len(starts) < 3 and homes:
        starts += for_people([z for role in (Role.WORK, Role.BREAK, Role.TRANSIT, Role.UNKNOWN) for z in by_role[role]
                              if z in homes and z in centroids and z not in starts and z not in forbidden])
    starts = starts or \
        for_people([z for z in targets if z not in forbidden and z in centroids]) or \
        for_people([z for z in targets if z in centroids]) or \
        [z for z in targets if z in centroids]
    if not starts:
        return doc

    # Where people walk to: every zone but the private ones, so a route crosses corridors and
    # rooms alike. The robot's own zones are fine as destinations - the agent arrives later.
    destinations = for_people([
        z for role in (Role.WORK, Role.BREAK, Role.TRANSIT, Role.UNKNOWN)
        for z in by_role[role] if z in centroids
    ]) or [z for role in (Role.WORK, Role.BREAK, Role.TRANSIT, Role.UNKNOWN) for z in by_role[role] if z in centroids]

    rng = random.Random(f"{world}:{profile}:{seed}")
    world_rng = random.Random(f"{world}:{seed}")
    tempo = world_rng.uniform(*TEMPO_RANGE)
    crowd = world_rng.uniform(*CROWD_RANGE)
    models = list(MODELS)
    world_rng.shuffle(models)

    n = max(MIN_CAST, round(nominal_count(targets) * spec["count"] * crowd))
    # Talkers come in pairs, one pair per room, so each pair can find each other; the walkers
    # are guaranteed - a base where nobody walks is furniture.
    talkers, _walkers = cast_split(n)
    talker_rooms = list(starts)
    rng_rooms = random.Random(f"{world}:{profile}:{seed}:talkers")
    rng_rooms.shuffle(talker_rooms)

    kind_of_world = world_kind(zones, overrides)
    weights = mix_for(profile, kind_of_world)
    types, type_weights = list(weights), list(weights.values())

    agents: list[dict[str, Any]] = []
    for i in range(n):
        stops = rng.choices([s for s, _ in ROUTE_STOPS], weights=[w for _, w in ROUTE_STOPS])[0]

        if i < talkers:
            home = talker_rooms[(i // TALKERS) % len(talker_rooms)]
            kind = f"./{BT_AGENT_TYPE}"
            speed = speed_for("adult", rng, spec["speed"] * tempo)
            # A talker's waypoints are its own room: the behaviour tree drives it, and the list
            # only has to be well-formed. Two entries, because the backend appends the spawn
            # pose to a one-entry list on any mode but `once`.
            waypoints: list[str] = [home, home]
        else:
            home = starts[i % len(starts)]
            kind = rng.choices(types, weights=type_weights)[0]
            speed = speed_for(kind, rng, spec["speed"] * tempo)
            legs = _route(home, destinations, i, stops, centroids)
            waypoints = [home, *legs] if legs else [home, home]

        agent: dict[str, Any] = {"agent_type": kind, "desired_velocity": speed}
        if speed > MAX_VELOCITY_DEFAULT:
            # The engine clamps every step to the type's `max_velocity` (1.5 +- 0.1 by default),
            # so a hurried 1.9 m/s would walk at 1.5; lift the cap with the speed.
            agent["max_velocity"] = round(speed + MAX_VELOCITY_HEADROOM, 2)
        agents.append({
            "name": f"{profile}_{i + 1}",
            "model": models[i % len(models)],
            "pose": home,
            "agent": agent,
            "waypoints": waypoints,
            "waypoint_mode": _waypoint_mode(len(waypoints), rng),
        })

    # A profile-selecting Type B ("the elderly people...") must always have someone to tune:
    # 25 realised scenarios were dropped to `matched nobody` before this guarantee. `elder` is
    # the one profile every mix carries, so the last walker becomes one when the draw missed.
    walkers = [a for a in agents if not str(a["agent"]["agent_type"]).startswith("./")]
    if walkers and all(a["agent"]["agent_type"] != "elder" for a in walkers):
        last = walkers[-1]
        last["agent"] = {"agent_type": "elder", "desired_velocity": speed_for("elder", random.Random(f"{world}:{profile}:{seed}:elder"), spec["speed"] * tempo)}

    doc["dynamic"] = agents
    return doc


def bt_type_source() -> Path:
    """Path of the shipped behaviour-tree agent type, for the writer to copy in."""
    return Path(__file__).resolve().parent / "agent_types" / BT_AGENT_TYPE


def basis_for(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> str:
    """Which mapping `build` will use for this world: workplace, residential, or transit-only."""
    return _targets("normal", zones_by_role(zones, overrides))[1]


def work_zones(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> list[str]:
    """The zones where *work happens* in this world - the ones `normal` populates.

    Public because "back to work" has to mean the same thing to a case that sends a crowd there
    as it did to the generator that placed them away from it. A residential world has no work
    zones at all, so this follows the same basis shift `_targets` does: the living areas are the
    active zones. Defining it twice would let the two drift, and a stampede rallying into a
    bedroom is exactly the kind of quiet nonsense that costs a run to notice.
    """
    return list(_targets("normal", zones_by_role(zones, overrides))[0])


def world_kind(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> str:
    """`hospital`, `residential` or `office`, from the world's own vocabulary.

    Hospital wins over residential when both read (a care home has wards and bedrooms), and
    `office` is the default because it is the kind that excludes nothing but the specialised
    types.
    """
    names = [str(z["name"]).lower() for z in zones if isinstance(z, Mapping) and z.get("name")]
    if any(hint in n for n in names for hint in _HOSPITAL_HINTS):
        return "hospital"
    if basis_for(zones, overrides) == "residential":
        return "residential"
    return "office"


def mix_for(profile: str, kind: str) -> dict[str, int]:
    """The type weights a profile draws from in this kind of building."""
    weights = MIX.get(profile, {"adult": 1})
    allowed = {t: w for t, w in weights.items() if kind in ONLY_IN.get(t, (kind,))}
    return allowed or {"adult": 1}


def speed_for(kind: str, rng: random.Random, scale: float = 1.0) -> float:
    """One walker's speed: the type's distribution, scaled, clipped to the type's own range."""
    mean, std, lo, hi = KINEMATICS[kind]
    raw = rng.gauss(mean, std) * scale
    return round(min(max(raw, lo), hi), 2)


def extra_agents(
    n: int,
    *,
    zones: Sequence[Mapping[str, Any]],
    robot: Mapping[str, Any],
    homes: frozenset[str] = frozenset(),
    seed: int = 0,
    world: str = "",
    mix: Mapping[str, int] = BUSY_MIX,
    prefix: str = "busy",
    speed_scale: float = 1.0,
    overrides: Mapping[str, str] | None = None,
    existing: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """`n` more walkers for a scenario that already has a population: the `busy` injection.

    Same placement rules as `build` - homes off the robot's route, routes through other rooms,
    types drawn from `mix` - and deterministic in its inputs. Names continue from `prefix_1`,
    skipping any the scenario already uses.
    """
    by_role = zones_by_role(zones, overrides)
    targets, _basis = _targets("normal", by_role)
    centroids = {
        str(z["name"]): _centroid(_polygon(z))
        for z in zones if isinstance(z, Mapping) and z.get("name") and len(_polygon(z)) >= 3
    }
    forbidden = robot_zones(robot)
    starts = for_people([z for z in targets if z in homes and z in centroids]) or \
        [z for z in targets if z not in forbidden and z in centroids] or \
        [z for z in targets if z in centroids]
    if not starts or n <= 0:
        return []
    destinations = for_people([
        z for role in (Role.WORK, Role.BREAK, Role.TRANSIT, Role.UNKNOWN)
        for z in by_role[role] if z in centroids
    ])
    rng = random.Random(f"{world}:{prefix}:{seed}")
    models = list(MODELS)
    rng.shuffle(models)
    weights = mix_for("normal", world_kind(zones, overrides)) if not mix else {t: w for t, w in mix.items() if world_kind(zones, overrides) in ONLY_IN.get(t, (world_kind(zones, overrides),))}
    weights = weights or {"adult": 1}
    types, type_weights = list(weights), list(weights.values())
    taken = {str(a.get("name")) for a in ((existing or {}).get("dynamic") or []) if isinstance(a, Mapping)}
    agents: list[dict[str, Any]] = []
    i = 0
    while len(agents) < n:
        i += 1
        name = f"{prefix}_{i}"
        if name in taken:
            continue
        stops = rng.choices([s for s, _ in ROUTE_STOPS], weights=[w for _, w in ROUTE_STOPS])[0]
        home = starts[i % len(starts)]
        kind = rng.choices(types, weights=type_weights)[0]
        legs = _route(home, destinations, i, stops, centroids)
        waypoints = [home, *legs] if legs else [home, home]
        agents.append({
            "name": name,
            "model": models[i % len(models)],
            "pose": home,
            "agent": {"agent_type": kind, "desired_velocity": speed_for(kind, rng, speed_scale)},
            "waypoints": waypoints,
            "waypoint_mode": _waypoint_mode(len(waypoints), rng),
        })
    return agents
