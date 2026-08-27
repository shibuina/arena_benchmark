"""Flow sources and sinks for a generated base scenario.

`populate` places a cast; a region spawns a stream. The reason to generate regions is not
variety. `_perturb_regions` refuses a Level B knob when the episode has no regions -

    "... regions - there is no population to perturb. Use a scenario with a `regions:` block"

- so a base without them puts the whole crowd/global tail out of reach: `flow_rate`,
`crowd_density`, `crowd_volume`, `source_speed`, `source_radius`, `sink_capacity`,
`sink_absorption`. All seven are built, verified against the humansim adapter, and were
unreachable on all thirty generated base scenarios because none carried a region.

**Only the stream is reachable that way.** Level B rewrites region config, so it touches the
agents a source spawns and not the dwellers `populate` places. A base carrying both has a
population Level B half-perturbs, which is a real limitation and the reason the two halves are
sized so the stream is the smaller one: a dweller cast that a Level A or D knob can reach, plus
traffic that a Level B knob can.

Placement differs from `populate`'s in one respect, deliberately. A dweller must be clear of the
robot's whole route, because an agent standing on it freezes the controller. A source is a
doorway the crowd comes *through*; agents leave it immediately, and requiring the whole route to
be clear would reject every entrance - `hospital_1/default` puts its main-entrance source
directly on the robot's start. What must not happen is spawning on top of the robot at t=0, so a
region is held clear of the robot's **start** rather than its route.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from .zones import Role, zones_by_role

#: Metres a region polygon must keep from the robot's start pose. Same margin as a dweller's
#: route clearance - the failure it prevents is the same one, at t=0.
MIN_START_CLEARANCE = 1.5

#: Arrival profile and standing ceiling per base profile. `rate` is the two keyframes of a ramp,
#: agents per second; `max_concurrent` bounds how many of this source's agents exist at once.
#: `empty` is absent on purpose - it is the no-population control, and a stream would populate it.
REGION_PROFILES: dict[str, dict[str, Any]] = {
    "normal": {"rate": (0.05, 0.15), "max_concurrent": 3},
    "break":  {"rate": (0.03, 0.10), "max_concurrent": 2},
    "busy":   {"rate": (0.15, 0.40), "max_concurrent": 8},
    "night":  {"rate": (0.02, 0.05), "max_concurrent": 1},
}

#: Seconds at which the arrival rate reaches its second keyframe. Both keyframes carry a rate
#: because `flow_rate` perturbs every keyframe: a profile that ramps back to nominal would
#: quietly undo the tail.
RAMP_AT = 30.0

#: Per-world multiplier on both the arrival rate and the standing ceiling. Without it every
#: world of a given profile streamed at exactly the same rate, which made the four populated
#: bases of one building indistinguishable from another's.
INTENSITY_RANGE = (0.7, 1.4)

#: Multiplier a `stream_led` base variant applies on top: the cast it gave up comes back as flow.
STREAM_LED_INTENSITY = 1.6

#: How a stream may run relative to the robot's route. `any` takes the best pair on the other
#: criteria; the three named topologies are requested by base-variant recipes.
TOPOLOGIES = ("any", "cross", "counter", "co")

#: |cos| below this between the stream vector and the nearest route segment is a crossing;
#: above it, the sign says whether the stream runs with the robot or against it.
_TOPOLOGY_COS = 0.5

#: Written even though the adapter defaults them. `source_radius` and `sink_absorption` are
#: declared `create: true` precisely because scenarios omit these, and a knob that has to insert
#: its own field is a knob whose target nobody reviewed. Emitting them makes the tail's landing
#: site explicit and readable in the file.
DEFAULT_AGENT_RADIUS = 0.35
DEFAULT_ABSORPTION_RADIUS = 0.5


def _centroid(poly: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def _polygon(zone: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(float(c["x"]), float(c["y"])) for c in zone.get("corners", []) if isinstance(c, Mapping)]


def _segments_cross(a: tuple[float, float], b: tuple[float, float],
                    c: tuple[float, float], d: tuple[float, float]) -> bool:
    def side(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = side(c, d, a), side(c, d, b)
    d3, d4 = side(a, b, c), side(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _point_segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def _segment_distance(a: tuple[float, float], b: tuple[float, float],
                      c: tuple[float, float], d: tuple[float, float]) -> float:
    if _segments_cross(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b), _point_segment_distance(d, a, b),
    )


def _inside(pt: tuple[float, float], poly: Sequence[tuple[float, float]]) -> bool:
    x, y = pt
    hit = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            hit = not hit
    return hit


def _near(poly: Sequence[tuple[float, float]], pt: tuple[float, float], within: float) -> bool:
    """Whether a disc of `within` around `pt` touches the polygon at all.

    The polygon rather than its centroid: a source is a region an agent may spawn anywhere in,
    so a big zone whose centre is far from the robot can still have a corner on top of it.
    """
    if _inside(pt, poly):
        return True
    return any(
        _point_segment_distance(pt, poly[i], poly[(i + 1) % len(poly)]) < within
        for i in range(len(poly))
    )


def _meets_route(a: tuple[float, float], b: tuple[float, float],
                 route: Sequence[tuple[float, float]], within: float = MIN_START_CLEARANCE) -> bool:
    """Whether the line a stream takes shares space with the robot's route.

    Distance rather than intersection, because the realistic case is not a clean crossing: in a
    tee corridor the sink sits *on* the main corridor, so the stream ends on the robot's line
    instead of cutting it, and a strict intersection test scores that as no encounter at all. A
    stream the robot never meets is a stream that measures nothing, so this is the selection
    criterion rather than a filter applied afterwards.
    """
    if len(route) < 2:
        return not route or math.dist(a, route[0]) <= within or math.dist(b, route[0]) <= within
    return any(
        _segment_distance(a, b, route[i], route[i + 1]) <= within
        for i in range(len(route) - 1)
    )


def topology(a: tuple[float, float], b: tuple[float, float], route: Sequence[tuple[float, float]]) -> str:
    """How a stream from `a` to `b` runs relative to the route: `cross`, `counter` or `co`.

    Measured against the route segment nearest the stream's midpoint, oriented the way the robot
    drives it - so reversing the route turns a `co` stream into a `counter` one, which is what a
    reversed variant means.
    """
    if len(route) < 2:
        return "cross"
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    seg = min(
        ((route[i], route[i + 1]) for i in range(len(route) - 1)),
        key=lambda s: _point_segment_distance(mid, s[0], s[1]),
    )
    dx, dy = seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]
    vx, vy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(dx, dy) * math.hypot(vx, vy)
    if norm == 0:
        return "cross"
    cos = (dx * vx + dy * vy) / norm
    if cos > _TOPOLOGY_COS:
        return "co"
    if cos < -_TOPOLOGY_COS:
        return "counter"
    return "cross"


def pick_endpoints(
    zones: Sequence[Mapping[str, Any]],
    *,
    route: Sequence[tuple[float, float]],
    resolvable: Mapping[str, int],
    robot_zones: frozenset[str] = frozenset(),
    overrides: Mapping[str, str] | None = None,
    want: str = "any",
) -> tuple[str, str] | None:
    """The zones to use as source and sink, or None when the world affords no usable pair.

    Sources prefer transit zones. A corridor, entrance or lobby is where traffic enters a
    building, and it is the one role `populate` never puts a dweller in, so the two halves of the
    population do not compete for the same floor.

    **`resolvable` is not `ref_safe`.** A region is admitted on distance from the robot's *start*
    where a dweller is admitted on distance from its whole *route*, so gating regions on the
    dweller screen would throw away exactly the zones worth streaming through - the hallway the
    robot drives down is the one every useful stream has to cross. `resolvable` maps a zone name
    to how many points the runtime resolver could return in it; zero means the region would
    resolve to a blocked point, which is the only disqualification geometry imposes here.
    """
    by_name = {str(z["name"]): _polygon(z) for z in zones if isinstance(z, Mapping) and z.get("name")}
    by_role = zones_by_role(zones, overrides)
    start = route[0] if route else None

    def resolves(name: str) -> bool:
        return len(by_name.get(name) or []) >= 3 and resolvable.get(name, 0) > 0

    def usable_source(name: str) -> bool:
        # Start clearance is a *source* rule. A source spawns bodies, and spawning one on the
        # robot at t=0 is the failure. A sink only absorbs, so a sink in the zone the robot
        # starts in is harmless - and in a small corridor world it is often the only sink that
        # puts the stream anywhere near the robot at all.
        if not resolves(name) or name in robot_zones:
            # A zone the robot's spec names is refused outright rather than measured against the
            # centroid: the robot resolves to a random point anywhere inside it, so no part of
            # the polygon is reliably 1.5 m away from wherever the robot turns out to be.
            return False
        return start is None or not _near(by_name[name], start, MIN_START_CLEARANCE)

    transit = [n for n in by_role[Role.TRANSIT] if usable_source(n)]
    rooms = [n for n in (*by_role[Role.WORK], *by_role[Role.BREAK]) if usable_source(n)]
    sink_transit = [n for n in by_role[Role.TRANSIT] if resolves(n)]
    sink_rooms = [n for n in (*by_role[Role.WORK], *by_role[Role.BREAK]) if resolves(n)]
    # Traffic enters through transit and ends up in a room, so a transit source and a room sink
    # are preferred - but as a *ranking*, not a filter. The acoustics corridors are
    # transit-only, a flat is rooms-only, and a building whose one corridor runs east of the
    # route cannot stream *across* it from a corridor at all: office to meeting room is the
    # crossing there, and a requested topology outranks the tidier source.
    sources = [*transit, *rooms]
    sinks = [*sink_rooms, *sink_transit]
    if not sources or not sinks:
        return None

    if want not in TOPOLOGIES:
        raise ValueError(f"unknown topology {want!r}; expected one of {TOPOLOGIES}")
    is_transit = set(by_role[Role.TRANSIT])

    def rank(s: str, t: str) -> tuple[bool, bool, bool, int, float]:
        a, b = _centroid(by_name[s]), _centroid(by_name[t])
        # The requested topology first, where one was requested; then meeting the robot, since a
        # stream it never meets measures nothing; then a transit source, which is where traffic
        # comes from; then the smaller of the two zones, because a source or a sink squeezed
        # into a cupboard jams on its own geometry rather than on the crowd. Distance only breaks
        # ties.
        return (
            want == "any" or topology(a, b, route) == want,
            _meets_route(a, b, route),
            s in is_transit,
            min(resolvable.get(s, 0), resolvable.get(t, 0)),
            math.dist(a, b),
        )

    pairs = [(rank(s, t), s, t) for s in sources for t in sinks if s != t]
    if not pairs:
        return None
    # A pair that misses the robot is still kept - it exercises every Level B knob, it just does
    # not obstruct anything - so meeting sorts first rather than filtering. `meets_route` lets
    # the caller say which one it got.
    _, source, sink = max(pairs)
    return source, sink


def build_regions(
    profile: str,
    *,
    zones: Sequence[Mapping[str, Any]],
    route: Sequence[tuple[float, float]],
    resolvable: Mapping[str, int],
    robot_zones: frozenset[str] = frozenset(),
    overrides: Mapping[str, str] | None = None,
    speed_scale: float = 1.0,
    world: str = "",
    seed: int = 0,
    want: str = "any",
    cast: str = "dispersed",
) -> dict[str, Any]:
    """The `regions:` block for one base scenario. Empty when the world affords no pair.

    `want` asks for a stream topology; `none` means no stream, and returns empty on purpose - a
    `night_a` base is quiet because it has no flow, not because the world lacked a pair.

    Polygons are written as `ref: <zone>` rather than corner lists, which is the room-centric
    form: the region is the room, and a world whose geometry is edited does not leave a stale
    rectangle behind in every scenario that streamed through it.
    """
    spec = REGION_PROFILES.get(profile)
    if spec is None or want == "none":
        return {}
    picked = pick_endpoints(
        zones, route=route, resolvable=resolvable, robot_zones=robot_zones, overrides=overrides,
        want=want,
    )
    if picked is None:
        return {}
    source, sink = picked
    # Jittered per world so two buildings on the same profile do not stream at identical rates.
    # Keyed on the world alone, matching `populate`'s tempo: one building has one traffic
    # character across all four of its populated bases.
    rng = random.Random(f"{world}:{seed}:regions")
    intensity = rng.uniform(*INTENSITY_RANGE) * (STREAM_LED_INTENSITY if cast == "stream_led" else 1.0)
    lo, hi = (round(r * intensity, 3) for r in spec["rate"])
    concurrent = max(1, round(spec["max_concurrent"] * intensity))

    return {
        "inflow": {
            "type": "source",
            "ref": source,
            "config": {
                "rate_profile": [{"t": 0.0, "rate": lo}, {"t": RAMP_AT, "rate": hi}],
                "max_concurrent": concurrent,
                "max_total": -1,
                "agent": {
                    "agent_type": "adult",
                    "desired_velocity": {
                        "min": round(0.9 * speed_scale, 2),
                        "max": round(1.5 * speed_scale, 2),
                    },
                    "agent_radius": DEFAULT_AGENT_RADIUS,
                    "sink_affinity": [{"sink": "outflow", "weight": 1.0}],
                },
            },
        },
        "outflow": {
            "type": "sink",
            "ref": sink,
            "config": {
                "capacity": -1,
                "absorption_radius": DEFAULT_ABSORPTION_RADIUS,
            },
        },
    }


def stream_topology(
    block: Mapping[str, Any],
    zones: Sequence[Mapping[str, Any]],
    route: Sequence[tuple[float, float]],
) -> str:
    """The topology a written block actually got. For the CLI to report against what was asked."""
    by_name = {str(z["name"]): _polygon(z) for z in zones if isinstance(z, Mapping) and z.get("name")}
    try:
        source = by_name[str(block["inflow"]["ref"])]
        sink = by_name[str(block["outflow"]["ref"])]
    except (KeyError, TypeError):
        return "none"
    return topology(_centroid(source), _centroid(sink), route)


def meets_route(
    block: Mapping[str, Any],
    zones: Sequence[Mapping[str, Any]],
    route: Sequence[tuple[float, float]],
) -> bool:
    """Whether a generated block's stream actually meets the robot. For the CLI to report."""
    by_name = {str(z["name"]): _polygon(z) for z in zones if isinstance(z, Mapping) and z.get("name")}
    try:
        source = by_name[str(block["inflow"]["ref"])]
        sink = by_name[str(block["outflow"]["ref"])]
    except (KeyError, TypeError):
        return False
    return _meets_route(_centroid(source), _centroid(sink), route)
