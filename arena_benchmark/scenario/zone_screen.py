"""Deciding which zones may be referenced by name instead of by coordinate.

`pose: exam_room_1` is resolved at run time by `PointResolver.resolve`, which samples a random
point inside the named polygon and keeps it only if `is_valid` accepts. That predicate is
`make_is_valid(world_manager.map, Obstacles.SAFE_DIST)`.

**It is not the weak check it looks like.** `map.png` is rasterised from walls and doors alone,
but the grid behind `is_valid` is not `map.png`: `WorldManager.__init__` stamps every static
entity into `map.occupancy` through `obstacle_occupy`, sized by the asset's own
`annotation.yaml` half-diagonal and skipping only fixtures mounted above passage height. So the
runtime resolver *does* see furniture. It is the nav2 static layer, screened separately by
`screen_worlds`, that does not.

Two gaps used to remain. **One of them was closed rather than screened around.** `SAFE_DIST` is
0.15 m and, as its own docstring says, a *sampling* clearance: it is how far a pedestrian's
CENTRE must sit from occupied space, and it says nothing about the body. Validating a standing
pose with it accepted points 0.2 m from a wall for an agent of nominal radius 0.25 m, and not by
bad luck - `sample_point_in_polygon` retries only what `is_valid` rejects, so such a point is
accepted on the first draw and reported as success. Screening for it was hopeless: the 0.15-to-
0.5 m band is 30-60% of the floor of every room in the tree, furnished or bare, so no zone in
any world passed. `TM_Scenario.reset` now validates at `SAFE_DIST + PEDESTRIAN_BODY_RADIUS`,
which is what this module mirrors in `RUNTIME_SAFE_DIST`.

What is left is the gap no runtime check can close:

* **The resolver knows nothing about the robot's route.** That is the check that keeps
  `acoustics_bend`'s frozen-robot failure - a pedestrian 0.25 m from the robot's start,
  controller refusing to move for 43 s of 45 - from coming back. The scenario mode resolves
  poses before any robot exists to ask.

And when the resolver does exhaust its retries it warns and **returns the last invalid
candidate** rather than dropping the agent, so a zone it cannot satisfy fails quietly.

So a zone is referenceable only when every point the runtime could return for it is a point
`populate` would have accepted itself. That is a property of the zone rather than of any one
sample, it is checkable offline once per world, and it is what lets a room-centric scenario keep
the guarantees a literal one has.

Requires numpy and scipy. Neither is needed to generate literal scenarios, so both are imported
lazily and a world that cannot be screened generates literals rather than failing.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs

if typing.TYPE_CHECKING:
    import numpy as np

from .occupancy import AGENT_RADIUS, BREATHING_ROOM

#: What `TM_Scenario.reset` passes to `make_is_valid`: `Obstacles.SAFE_DIST` (0.15, a sampling
#: margin) plus `Obstacles.PEDESTRIAN_BODY_RADIUS` (0.25, the nominal adult body). Mirrored from
#: `task_generator/constants/runtime.py`, and it must stay mirrored: a screen that models a
#: kinder threshold than the resolver enforces clears zones the resolver will then fail in, and
#: one that models a stricter threshold rejects zones that were fine.
RUNTIME_SAFE_DIST = 0.15 + 0.25

#: What `populate` demands of a literal it places itself: 0.5 m, slightly wider than the
#: runtime's 0.4 m disc. It can afford to be - it resamples until it finds one and drops the
#: agent if it cannot, where the resolver has to return something.
BODY = AGENT_RADIUS + BREATHING_ROOM


@attrs.frozen(kw_only=True)
class Verdict:
    """Whether one zone may be written as a name, and why not when it may not."""

    zone: str
    ref_safe: bool
    reason: str = ""
    #: Cells the runtime resolver could return - inside the polygon and past `SAFE_DIST`.
    admissible: int = 0
    #: Of those, how many `populate` would have refused.
    rejected: int = 0

    def __bool__(self) -> bool:
        return self.ref_safe


def _load_grid(world_root: str | Path) -> tuple[np.ndarray, float, float, float] | None:
    """Wall occupancy as a boolean mask, plus resolution and origin. None when unreadable."""
    root = Path(world_root)
    meta_path = next(iter(sorted(root.glob("*/map.yaml"))), None)
    if meta_path is None:
        return None
    try:
        import numpy as np  # noqa: PLC0415 - optional, only needed to screen
        import yaml  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        meta = yaml.safe_load(meta_path.read_text()) or {}
        image = Image.open(meta_path.parent / meta.get("image", "map.png")).convert("L")
    except Exception:
        return None

    res = float(meta.get("resolution", 0.05))
    origin = meta.get("origin") or [0.0, 0.0, 0.0]
    thresh = float(meta.get("occupied_thresh", 0.65))
    negate = bool(meta.get("negate", 0))

    pixels = np.asarray(image, dtype=float) / 255.0
    shade = pixels if negate else 1.0 - pixels
    return shade > thresh, res, float(origin[0]), float(origin[1])


def _stamp_statics(occupied: np.ndarray, statics: Sequence[Mapping[str, Any]], res: float, ox: float, oy: float) -> None:
    """Add the world's furniture to the wall mask, the way `WorldManager` does at load.

    `obstacle_occupy` fills the axis-aligned *rectangle* around the footprint radius rather than
    the disc, so this does too. Matching the runtime matters more than being tighter than it: a
    screen that models a kinder grid than the resolver uses clears zones the resolver will then
    fail in.
    """
    height, width = occupied.shape
    for obj in statics:
        pose = obj.get("pose")
        if not isinstance(pose, (list, tuple)) or len(pose) < 2:
            continue
        radius = float(obj.get("radius") or obj.get("interaction_radius") or 0.5)
        x, y = float(pose[0]), float(pose[1])
        c0 = max(0, int((x - radius - ox) / res))
        c1 = min(width, int((x + radius - ox) / res) + 1)
        r0 = max(0, height - 1 - int((y + radius - oy) / res))
        r1 = min(height, height - int((y - radius - oy) / res))
        if r0 < r1 and c0 < c1:
            occupied[r0:r1, c0:c1] = True


def _rasterize(poly: Sequence[tuple[float, float]], shape: tuple[int, int], res: float, ox: float, oy: float) -> np.ndarray:
    """Boolean mask of the cells a polygon covers, in the grid's row/column frame."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    height, width = shape
    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(
        [(int((x - ox) / res), height - 1 - int((y - oy) / res)) for x, y in poly],
        fill=1,
    )
    return np.asarray(canvas, dtype=bool)


def _cell_centres(mask: np.ndarray, res: float, ox: float, oy: float) -> tuple[np.ndarray, np.ndarray]:
    """World-frame x and y of every set cell, matching `Occupancy._cell` in reverse."""
    import numpy as np  # noqa: PLC0415

    rows, cols = np.nonzero(mask)
    height = mask.shape[0]
    return cols * res + ox, (height - 1 - rows) * res + oy


def _route_clearance(xs: np.ndarray, ys: np.ndarray, route: Sequence[tuple[float, float]]) -> np.ndarray:
    """Distance from every cell to the robot's route polyline, vectorised."""
    import numpy as np  # noqa: PLC0415

    if not route:
        return np.full(xs.shape, np.inf)
    if len(route) == 1:
        return np.hypot(xs - route[0][0], ys - route[0][1])

    best = np.full(xs.shape, np.inf)
    for (ax, ay), (bx, by) in zip(route, route[1:], strict=False):
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom == 0:
            best = np.minimum(best, np.hypot(xs - ax, ys - ay))
            continue
        t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / denom, 0.0, 1.0)
        best = np.minimum(best, np.hypot(xs - (ax + t * dx), ys - (ay + t * dy)))
    return best


def screen_zones(
    zones: Sequence[Mapping[str, Any]],
    *,
    world_root: str | Path,
    route: Sequence[tuple[float, float]],
    statics: Sequence[Mapping[str, Any]] = (),
    min_route_clearance: float,
    safe_dist: float = RUNTIME_SAFE_DIST,
    robot_zones: frozenset[str] = frozenset(),
) -> dict[str, Verdict]:
    """A verdict per zone. An unscreenable world returns verdicts that all refuse.

    Refusing is the safe direction: a zone written as a literal is still correct, where a zone
    written as a name that should not have been is a pedestrian in a wall, discovered in
    simulation if at all.

    `robot_zones` are zones the robot's own spec names as an endpoint. Those are refused whole:
    the robot's start resolves anywhere inside such a zone, so no part of it is reliably clear of
    the robot, and the route polyline through its centroid understates that by a room's width.
    """
    verdicts: dict[str, Verdict] = {}
    names = [str(z["name"]) for z in zones if isinstance(z, Mapping) and z.get("name")]

    grid = _load_grid(world_root)
    if grid is None:
        return {n: Verdict(zone=n, ref_safe=False, reason="no occupancy grid to screen against") for n in names}
    occupied, res, ox, oy = grid

    import scipy.ndimage  # noqa: PLC0415

    _stamp_statics(occupied, statics, res, ox, oy)
    # One EDT answers every clearance query: `dist > RUNTIME_SAFE_DIST` is what the resolver
    # accepts, and `BODY` is the slightly wider bar `populate` holds its own literals to.
    # Distance is to the nearest occupied cell, so it runs over the free space.
    dist = scipy.ndimage.distance_transform_edt(~occupied) * res

    for zone in zones:
        if not isinstance(zone, Mapping) or not zone.get("name"):
            continue
        name = str(zone["name"])
        poly = [(float(c["x"]), float(c["y"])) for c in zone.get("corners", []) if isinstance(c, Mapping)]
        if len(poly) < 3:
            verdicts[name] = Verdict(zone=name, ref_safe=False, reason="fewer than three corners")
            continue

        mask = _rasterize(poly, dist.shape, res, ox, oy)
        if not mask.any():
            verdicts[name] = Verdict(zone=name, ref_safe=False, reason="polygon covers no grid cell")
            continue

        clearance = dist[mask]
        # The runtime's admissible set: what `is_valid` would let `sample_point_in_polygon`
        # return. Everything below is asked of these cells only - a cell the runtime would
        # never return cannot make a zone unsafe.
        admissible = clearance > safe_dist
        n_admissible = int(admissible.sum())
        if n_admissible == 0:
            verdicts[name] = Verdict(
                zone=name, ref_safe=False,
                reason="no point in the zone clears SAFE_DIST; the resolver would warn and return a blocked point",
            )
            continue

        if name in robot_zones:
            # Refused whole, and the admissible count is still reported: a caller choosing a
            # flow sink wants to know the zone resolves fine, only that a dweller must not
            # stand in it.
            verdicts[name] = Verdict(
                zone=name, ref_safe=False, admissible=n_admissible, rejected=n_admissible,
                reason="the robot's own spec names this zone as an endpoint, so it may start or "
                       "finish anywhere inside it",
            )
            continue

        xs, ys = _cell_centres(mask, res, ox, oy)
        xs, ys, clearance = xs[admissible], ys[admissible], clearance[admissible]

        # Only the route is left to check. Wall and furniture clearance is already carried by
        # `admissible`, because the runtime resolver now enforces a body rather than a centre -
        # a point it would return is a point an agent fits at. The robot's route is the one
        # thing it cannot know, and so the one thing this screen exists to add.
        route_bad = _route_clearance(xs, ys, route) < min_route_clearance
        n_rejected = int(route_bad.sum())

        if n_rejected == 0:
            verdicts[name] = Verdict(zone=name, ref_safe=True, admissible=n_admissible)
            continue

        verdicts[name] = Verdict(
            zone=name, ref_safe=False, admissible=n_admissible, rejected=n_rejected,
            reason=f"{n_rejected}/{n_admissible} resolvable points sit within "
                   f"{min_route_clearance} m of the robot's route",
        )

    return verdicts


def ref_safe_zones(verdicts: Mapping[str, Verdict]) -> frozenset[str]:
    """The names that may be written as zone references."""
    return frozenset(name for name, v in verdicts.items() if v.ref_safe)
