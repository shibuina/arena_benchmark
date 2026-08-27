"""Finding a shipped spec by name.

Same two-place lookup the edge_case catalogues use: the installed share directory first, so
a colcon-built workspace resolves what it installed, then the source tree, so a checkout
works without building. A path is always taken as a path.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Where shipped specs live inside the source tree, relative to this file.
_SOURCE_DIR = Path(__file__).resolve().parents[2] / "specs"


def spec_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415

        dirs.append(Path(get_package_share_directory("arena_benchmark")) / "specs")
    except Exception:
        # No ament index (a bare checkout, or a host without ROS). The source tree is the
        # answer, and it is the next candidate anyway.
        pass
    dirs.append(_SOURCE_DIR)
    return [d for d in dirs if d.is_dir()]


def available() -> list[str]:
    """Names of every shipped spec, deduplicated, in sorted order."""
    names = {p.stem for d in spec_dirs() for p in d.glob("*.yaml")}
    return sorted(names)


def resolve(name: str | os.PathLike[str]) -> Path:
    """A shipped spec's path, by name or by path.

    Raises with the list of what *is* available rather than a bare "not found" - a
    mistyped spec name should not cost a directory listing.
    """
    p = Path(name)
    if p.suffix in (".yaml", ".yml") or os.sep in str(name):
        return p
    for d in spec_dirs():
        candidate = d / f"{name}.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"unknown spec {name!r}; available: {available()}")
