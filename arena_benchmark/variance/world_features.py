"""A world as a feature vector, and how far apart a set of worlds sits.

Features (all from the occupancy map with the furniture stamped in, plus `world.yaml` zones):

| feature            | what it measures                                                        |
|--------------------|-------------------------------------------------------------------------|
| `area_m2`          | walkable area (free cells with >= `ROBOT_CLEAR` clearance)              |
| `free_fraction`    | walkable area / area of the map's occupied bounding box                 |
| `clearance_p10`    | 10th percentile of wall/furniture clearance over walkable cells (m)     |
| `clearance_median` | median clearance (m) - corridor half-width, roughly                     |
| `narrow_fraction`  | share of walkable cells with < `NARROW_M` clearance (tight passages)    |
| `rooms`            | connected regions once every passage narrower than a door is cut        |
| `doorways`         | how many of those regions a door-wide opening joins (merges)            |
| `tortuosity`       | mean geodesic / straight-line distance between sampled free points      |
| `zones`            | named zones with a polygon                                              |
| `zone_area_median` | median zone area (m^2)                                                  |
| `aspect`           | bounding-box aspect ratio of the walkable area (>= 1)                   |
| `clutter`          | furniture footprint / raw free area                                     |

`variance_report` normalises each feature to [0, 1] over the worlds given and reports the
per-feature spread, the mean pairwise distance, each world's distinctiveness (distance to its
nearest neighbour) and a greedy farthest-point ordering: the first `k` worlds of that order are
the subset that covers the feature space best, which is the order to author and test them in.
"""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from ..promptgen.world import ROBOT_CLEAR, Grid
from ..scenario.occupancy import static_polygon, world_statics
from ..scenario.zone_screen import _load_grid, _stamp_statics

FEATURES: tuple[str, ...] = (
    "area_m2", "free_fraction", "clearance_p10", "clearance_median", "narrow_fraction",
    "rooms", "doorways", "tortuosity", "zones", "zone_area_median", "aspect", "clutter",
)
#: A passage narrower than twice this is "narrow" (a person and a robot cannot pass).
NARROW_M = 0.6
#: Cutting every opening narrower than twice this (a wide door is 1.5 m) splits the map into rooms.
DOOR_CLEAR_M = 0.8
#: Regions smaller than this after the cut are alcoves, not rooms.
MIN_ROOM_M2 = 2.0
#: Geodesic sampling: this many points, pairs at least this far apart.
TORTUOSITY_SAMPLES = 12
TORTUOSITY_MIN_M = 3.0
#: BFS lattice for the geodesic (the map is downsampled to this cell size).
GEODESIC_CELL_M = 0.2


class WorldFeatureError(RuntimeError):
    pass


def _polygon(zone: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(c["x"]), float(c["y"])) for c in zone.get("corners") or [] if isinstance(c, dict)]


def _area(poly: Sequence[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))) / 2


def load_zones(world_root: Path) -> list[dict[str, Any]]:
    level = next(iter(sorted(world_root.glob("*/world.yaml"))), None)
    if level is None:
        return []
    doc = yaml.safe_load(level.read_text()) or {}
    zones = doc if isinstance(doc, list) else doc.get("zones") or []
    return [z for z in zones if isinstance(z, dict) and z.get("name")]


def load_grids(world_root: Path) -> tuple[Grid, Grid, float]:
    """(walls-only grid, walls+furniture grid, furniture footprint in m^2)."""
    loaded = _load_grid(world_root)
    if loaded is None:
        raise WorldFeatureError(f"{world_root.name}: no map")
    occupied, res, ox, oy = loaded
    raw = Grid(occupied=occupied.copy(), res=res, ox=ox, oy=oy)
    stamped = occupied.copy()
    try:
        statics = world_statics(world_root)
    except (AttributeError, TypeError, KeyError, ValueError):  # a world.yaml that is a bare zone list, or no entities block
        statics = []
    _stamp_statics(stamped, statics, res, ox, oy)
    footprint = sum(_area(p) for p in (static_polygon(s) for s in statics) if p)
    return raw, Grid(occupied=stamped, res=res, ox=ox, oy=oy), footprint


def _geodesic_tortuosity(grid: Grid, *, seed: int = 0) -> float:
    """Mean (shortest walkable path / straight line) between sampled points of the main region.

    8-connected Dijkstra on a coarse lattice, so a straight corridor in any direction scores ~1
    and only real detours (around a wall, through a door) push the ratio up.
    """
    import numpy as np
    import scipy.ndimage

    k = max(1, int(round(GEODESIC_CELL_M / grid.res)))
    walkable = grid.dist >= ROBOT_CLEAR
    coarse = walkable[::k, ::k]
    labels, n = scipy.ndimage.label(coarse)
    if n == 0:
        return 1.0
    main = int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
    cells = np.argwhere(labels == main)
    if len(cells) < 2:
        return 1.0
    rng = random.Random(seed)
    picks = [tuple(int(v) for v in cells[i]) for i in rng.sample(range(len(cells)), min(TORTUOSITY_SAMPLES, len(cells)))]
    step = grid.res * k
    rows, cols = coarse.shape
    inside = labels == main
    moves = [(dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    ratios: list[float] = []
    for src in picks:
        dist = np.full(coarse.shape, np.inf)
        dist[src] = 0.0
        heap: list[tuple[float, int, int]] = [(0.0, src[0], src[1])]
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r, c]:
                continue
            for dr, dc, w in moves:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and inside[rr, cc] and d + w < dist[rr, cc]:
                    dist[rr, cc] = d + w
                    heapq.heappush(heap, (d + w, rr, cc))
        for dst in picks:
            euclid = math.dist(src, dst) * step
            if dst == src or euclid < TORTUOSITY_MIN_M or not np.isfinite(dist[dst]):
                continue
            ratios.append(float(dist[dst]) * step / euclid)
    return float(sum(ratios) / len(ratios)) if ratios else 1.0


def features(world_root: str | Path, *, seed: int = 0) -> dict[str, float]:
    import numpy as np
    import scipy.ndimage

    root = Path(world_root)
    raw, grid, footprint = load_grids(root)
    zones = load_zones(root)
    walkable = grid.dist >= ROBOT_CLEAR
    cell = grid.res * grid.res
    area = float(walkable.sum()) * cell
    if area <= 0:
        raise WorldFeatureError(f"{root.name}: no walkable cell")
    occ_rows, occ_cols = np.nonzero(grid.occupied)
    bbox_area = float((occ_rows.max() - occ_rows.min() + 1) * (occ_cols.max() - occ_cols.min() + 1)) * cell if len(occ_rows) else area
    clear = grid.dist[walkable]
    rooms_mask = grid.dist >= DOOR_CLEAR_M
    labels, _n = scipy.ndimage.label(rooms_mask)
    sizes = np.bincount(labels.ravel())[1:] * cell
    rooms = int((sizes >= MIN_ROOM_M2).sum())
    _labels_joined, joined = scipy.ndimage.label(walkable)
    big_joined = int((np.bincount(_labels_joined.ravel())[1:] * cell >= MIN_ROOM_M2).sum())
    doorways = max(0, rooms - big_joined)
    free_rows, free_cols = np.nonzero(walkable)
    h = float(free_rows.max() - free_rows.min() + 1)
    w = float(free_cols.max() - free_cols.min() + 1)
    aspect = max(h, w) / max(1.0, min(h, w))
    raw_free = float((~raw.occupied).sum()) * cell
    polys = [p for p in (_polygon(z) for z in zones) if len(p) >= 3]
    zone_areas = sorted(_area(p) for p in polys)
    return {
        "area_m2": round(area, 2),
        "free_fraction": round(area / max(bbox_area, cell), 4),
        "clearance_p10": round(float(np.percentile(clear, 10)), 3),
        "clearance_median": round(float(np.median(clear)), 3),
        "narrow_fraction": round(float((clear < NARROW_M).mean()), 4),
        "rooms": float(rooms),
        "doorways": float(doorways),
        "tortuosity": round(_geodesic_tortuosity(grid, seed=seed), 4),
        "zones": float(len(polys)),
        "zone_area_median": round(zone_areas[len(zone_areas) // 2], 2) if zone_areas else 0.0,
        "aspect": round(aspect, 3),
        "clutter": round(footprint / max(raw_free, cell), 4),
    }


# --- variance over a set of worlds ----------------------------------------------------------


def normalise(table: dict[str, dict[str, float]], keys: Sequence[str] = FEATURES) -> dict[str, list[float]]:
    """Each world as a vector in [0, 1]^d (min-max over the worlds given; a constant feature is 0)."""
    lo = {k: min(v[k] for v in table.values()) for k in keys}
    hi = {k: max(v[k] for v in table.values()) for k in keys}
    return {w: [((v[k] - lo[k]) / (hi[k] - lo[k])) if hi[k] > lo[k] else 0.0 for k in keys] for w, v in table.items()}


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def farthest_point_order(vectors: dict[str, list[float]], *, first: str | None = None) -> list[str]:
    """Greedy k-centre ordering: each next world is the one farthest from every world chosen so far."""
    names = list(vectors)
    if not names:
        return []
    if first is None:
        # start from the world farthest from the centroid, so the ordering is deterministic
        d = len(next(iter(vectors.values())))
        centroid = [sum(vectors[n][i] for n in names) / len(names) for i in range(d)]
        first = min(names, key=lambda n: (-_dist(vectors[n], centroid), n))
    order = [first]
    while len(order) < len(names):
        rest = [n for n in names if n not in order]
        order.append(min(rest, key=lambda n: (-min(_dist(vectors[n], vectors[o]) for o in order), n)))
    return order


def coverage(vectors: dict[str, list[float]], subset: Sequence[str]) -> float:
    """Mean distance from every world to its nearest member of `subset` (0 = the subset is everything)."""
    if not subset:
        return float("inf")
    return sum(min(_dist(vectors[w], vectors[s]) for s in subset) for w in vectors) / len(vectors)


def variance_report(table: dict[str, dict[str, float]], keys: Sequence[str] = FEATURES) -> dict[str, Any]:
    vectors = normalise(table, keys)
    names = list(vectors)
    d = len(keys)
    per_feature = {}
    for i, k in enumerate(keys):
        col = [vectors[n][i] for n in names]
        mean = sum(col) / len(col)
        per_feature[k] = round(math.sqrt(sum((x - mean) ** 2 for x in col) / len(col)), 4)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    spread = (sum(_dist(vectors[a], vectors[b]) for a, b in pairs) / len(pairs) / math.sqrt(d)) if pairs else 0.0
    distinct = {n: round(min((_dist(vectors[n], vectors[m]) for m in names if m != n), default=0.0) / math.sqrt(d), 4) for n in names}
    order = farthest_point_order(vectors)
    return {
        "features": list(keys),
        "per_feature_spread": per_feature,
        "spread": round(spread, 4),
        "distinctiveness": distinct,
        "order": order,
        "coverage_by_k": [round(coverage(vectors, order[:k]) / math.sqrt(d), 4) for k in range(1, len(order) + 1)],
    }


#: A world the generator can author on: a map, enough named zones to place a cast and name
#: endpoints, and enough floor for a robot leg (`MIN_LEG_M` is 15 m; a 40 m^2 world cannot hold one).
MIN_USABLE_ZONES = 3
MIN_USABLE_AREA_M2 = 40.0


def usable(feat: dict[str, float]) -> bool:
    return feat.get("zones", 0) >= MIN_USABLE_ZONES and feat.get("area_m2", 0) >= MIN_USABLE_AREA_M2


def world_table(worlds_root: str | Path, names: Sequence[str] = (), *, seed: int = 0,
                on_error: Callable[[str, Exception], None] | None = None) -> dict[str, dict[str, float]]:
    root = Path(worlds_root)
    picked = list(names) or sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    out: dict[str, dict[str, float]] = {}
    for name in picked:
        try:
            out[name] = features(root / name, seed=seed)
        except Exception as exc:  # noqa: BLE001 - a world without a map is reported, not fatal
            if on_error is not None:
                on_error(name, exc)
    return out


def format_table(table: dict[str, dict[str, float]], report: dict[str, Any] | None = None) -> str:
    cols = list(FEATURES)
    head = f"{'world':28s} " + " ".join(f"{c[:10]:>10s}" for c in cols) + ("   distinct" if report else "")
    lines = [head]
    for w, v in table.items():
        row = f"{w:28s} " + " ".join(f"{v[c]:10.3f}" if isinstance(v[c], float) and not float(v[c]).is_integer() else f"{v[c]:10.0f}" for c in cols)
        if report:
            row += f"   {report['distinctiveness'][w]:.3f}"
        lines.append(row)
    if report:
        lines.append("")
        lines.append("per-feature spread (std of the [0,1]-normalised value): " + ", ".join(f"{k}={v:.2f}" for k, v in report["per_feature_spread"].items()))
        lines.append(f"mean pairwise distance (normalised): {report['spread']:.3f}")
        lines.append("farthest-point order: " + " > ".join(report["order"]))
        lines.append("coverage gap after k worlds: " + ", ".join(f"{k + 1}:{c:.3f}" for k, c in enumerate(report["coverage_by_k"])))
    return "\n".join(lines)
