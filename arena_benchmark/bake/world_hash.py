"""Content hash of a world, so a baked scenario can refuse a world that changed under it.

A baked scenario is a list of coordinates. Those coordinates are only meaningful against the
geometry they were solved on, and nothing in the tree versions a world — no hash, no manifest,
no revision. Edit a wall and every scenario baked against that world silently becomes wrong,
in a way that looks like a planner regression rather than a data problem.

So the hash covers what a coordinate depends on: the world description and the occupancy grid.
It deliberately does **not** cover `scenarios/`, because adding a scenario to a world does not
invalidate coordinates in the others - and a suite installs itself into that very directory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Files whose content a baked coordinate depends on, relative to the world root. Globs are
#: resolved in sorted order so the digest does not depend on filesystem ordering.
GEOMETRY_GLOBS: tuple[str, ...] = ("*/world.yaml", "*/map.yaml", "*/map.png", "graph.yaml")

HASH_LEN = 16


def world_files(world_root: Path) -> list[Path]:
    """Every geometry file of a world, sorted, deduplicated."""
    seen: dict[Path, None] = {}
    for pattern in GEOMETRY_GLOBS:
        for p in sorted(world_root.glob(pattern)):
            if p.is_file():
                seen.setdefault(p, None)
    return list(seen)


def world_hash(world_root: str | Path) -> str:
    """Digest of a world's geometry.

    Raises when the world has no geometry files at all: a hash of nothing would compare equal
    for every such world, which is worse than refusing.
    """
    root = Path(world_root)
    files = world_files(root)
    if not files:
        raise FileNotFoundError(f"{root}: no world geometry found (looked for {list(GEOMETRY_GLOBS)})")

    h = hashlib.sha256()
    for p in files:
        # The relative path is hashed too, so moving a file between levels changes the digest.
        h.update(str(p.relative_to(root)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()[:HASH_LEN]
