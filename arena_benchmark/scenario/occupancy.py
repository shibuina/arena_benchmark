"""Free-space checking against a world's occupancy grid.

Placing a pedestrian inside a wall, or close enough that its body overlaps one, produces a
scenario that looks fine in YAML and is wrong in simulation. Zone membership is not enough:
a zone polygon is a room's *extent*, and the room contains walls, doorframes and furniture that
the polygon says nothing about.

Measured on `arena_arena_001` before this existed: 3 of 13 generated agents sat within 0.35 m
of an occupied cell - none strictly inside one, which is exactly why the bug survived a visual
read of the YAML.

The grid is the authority, because it is what the planner and the collision checks use. This
module reads it directly rather than going through ROS, so scenario generation stays offline.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

#: Default body radius, metres. `ArenaHumanDynamicObstacle.agent_radius` defaults to 0.35.
AGENT_RADIUS = 0.35

#: Extra margin on top of the body, metres. A pedestrian flush against a wall is legal and
#: useless: it cannot step aside, so it behaves like furniture rather than a person.
BREATHING_ROOM = 0.15


class Occupancy:
    """A world's occupancy grid, queryable in world coordinates."""

    def __init__(self, meta: dict[str, Any], pixels: object, size: tuple[int, int]) -> None:
        self._px: Any = pixels
        self._w, self._h = size
        self._res = float(meta.get("resolution", 0.05))
        origin = meta.get("origin") or [0.0, 0.0, 0.0]
        self._ox, self._oy = float(origin[0]), float(origin[1])
        self._thresh = float(meta.get("occupied_thresh", 0.65))
        self._negate = bool(meta.get("negate", 0))

    @classmethod
    def load(cls, world_root: str | Path) -> Occupancy | None:
        """The grid for a world, or None when it cannot be read.

        None is a usable answer: placement falls back to zone-only and says so. Refusing to
        generate at all because a world ships no map would be worse than generating with a
        stated limitation.
        """
        root = Path(world_root)
        meta_path = next(iter(sorted(root.glob("*/map.yaml"))), None)
        if meta_path is None:
            return None
        try:
            meta = yaml.safe_load(meta_path.read_text()) or {}
            from PIL import Image  # noqa: PLC0415 - optional, and only needed to place

            image = Image.open(meta_path.parent / meta.get("image", "map.png")).convert("L")
        except Exception:
            return None
        return cls(meta, image.load(), image.size)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        col = int((x - self._ox) / self._res)
        # Image rows run top-down; map y runs bottom-up.
        row = self._h - 1 - int((y - self._oy) / self._res)
        return col, row

    def occupied(self, x: float, y: float) -> bool:
        """Whether one point is blocked. Off-grid counts as blocked."""
        col, row = self._cell(x, y)
        if not (0 <= col < self._w and 0 <= row < self._h):
            return True
        value = self._px[col, row]
        shade = value / 255.0 if self._negate else (255 - value) / 255.0
        return shade > self._thresh

    def free(self, x: float, y: float, radius: float = AGENT_RADIUS + BREATHING_ROOM) -> bool:
        """Whether a body of `radius` centred here fits in free space.

        Checks the disc rather than the centre: an agent is not a point, and the failure this
        exists to prevent is precisely a centre in free space with a body in a wall.
        """
        if self.occupied(x, y):
            return False
        steps = max(1, math.ceil(radius / self._res))
        for dc in range(-steps, steps + 1):
            for dr in range(-steps, steps + 1):
                if dc * dc + dr * dr > steps * steps:
                    continue
                if self.occupied(x + dc * self._res, y + dr * self._res):
                    return False
        return True


#: Half-extent used when an asset ships no annotation, metres. Chairs and small props dominate
#: the shipped furniture, so this is deliberately not generous.
DEFAULT_FOOTPRINT = 0.4


def asset_footprint(model: str, assets_root: Path) -> float:
    """Circumscribing radius of a model's footprint, from its `annotation.yaml`.

    `bounding_box` is `[[x_lo, x_hi], [y_lo, y_hi], [z_lo, z_hi]]`. A circle around the x-y
    extent is a conservative stand-in for a rotated rectangle: it over-blocks the corners
    slightly, which is the right direction to be wrong when the alternative is a pedestrian
    standing inside a desk.
    """
    ann = assets_root / model / "annotation.yaml"
    try:
        box = (yaml.safe_load(ann.read_text()) or {}).get("bounding_box")
        (x_lo, x_hi), (y_lo, y_hi) = box[0], box[1]
    except Exception:
        return DEFAULT_FOOTPRINT
    return math.hypot(max(abs(float(x_lo)), abs(float(x_hi))), max(abs(float(y_lo)), abs(float(y_hi))))


def world_statics(world_root: str | Path, assets_root: Path | None = None) -> list[dict[str, Any]]:
    """Every static furniture entity a world declares, with a footprint radius.

    **These are not in the occupancy grid.** Measured on `arena_arena_002`: 0 of 34 furniture
    positions register as occupied in `map.png`, which is rasterised from walls alone. A
    placement check that consults only the grid therefore puts pedestrians inside desks - which
    is exactly what it did.
    """
    root = Path(world_root)
    assets = assets_root or (root.parents[1] / "_assets" / "default")
    level = next(iter(sorted(root.glob("*/world.yaml"))), None)
    if level is None:
        return []
    try:
        doc = yaml.safe_load(level.read_text()) or {}
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for zone in doc.get("zones") or []:
        if not isinstance(zone, dict):
            continue
        for ent in ((zone.get("entities") or {}).get("static") or []):
            pos = ((ent or {}).get("pose") or {}).get("position") or {}
            if "x" not in pos or "y" not in pos:
                continue
            model = str(ent.get("model", ""))
            out.append({
                "name": ent.get("name", "?"),
                "model": model,
                "pose": [float(pos["x"]), float(pos["y"])],
                "radius": asset_footprint(model, assets) if model else DEFAULT_FOOTPRINT,
            })
    return out


def blocked_by_static(x: float, y: float, statics: list[dict[str, Any]], radius: float) -> bool:
    """Whether a point collides with a scenario's own `static:` objects.

    Those are placed by the scenario and so are absent from the world's grid, which is
    rasterised from the world description alone.
    """
    for obj in statics:
        pose = obj.get("pose")
        if not isinstance(pose, (list, tuple)) or len(pose) < 2:
            continue  # a zone-reference pose; nothing to measure against
        # `radius` is the asset's own footprint where known; `interaction_radius` covers the
        # scenario-declared objects, which are a different source with a different field.
        keep_out = float(obj.get("radius") or obj.get("interaction_radius") or 0.5)
        if math.dist((x, y), (float(pose[0]), float(pose[1]))) < radius + keep_out:
            return True
    return False
