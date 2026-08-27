"""Every robot × every compatible planner × every case scenario, as literal launch commands.

Derived from the workspace rather than typed in, so the file cannot drift from what ships:
a robot's drive family comes from the ros2_control controller its `arena_robots` entry
declares, a planner's from the `action_type` in its `planner.yaml`. Arms compare only within a
family (`dataset_generation.md` §2), so a differential planner is never paired with an
omnidirectional chassis - the pairing would boot and then not move.

The command shape is the one the 2026-08-26 roll-out used on every arm; `robot.mobile:=` is
the adapter (`nav2` or `drl`), `robot.mobile.planner:=` the learned planner under `drl`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

DIFFERENTIAL = "differential"
OMNIDIRECTIONAL = "omnidirectional"
#: Legged bases: `joint_trajectory_controller`, no wheeled adapter drives them. Listed, not paired.
LEGGED = "legged"

#: ros2_control controller type -> drive family. Anything else is unknown and listed as such.
_CONTROLLER_FAMILY: tuple[tuple[str, str], ...] = (
    ("DiffDriveController", DIFFERENTIAL),
    ("SwerveController", OMNIDIRECTIONAL),
    ("MecanumDriveController", OMNIDIRECTIONAL),
    ("OmniDriveController", OMNIDIRECTIONAL),
    ("JointTrajectoryController", LEGGED),
)

#: `planner.yaml` action_type -> drive family.
_ACTION_FAMILY: Mapping[str, str] = {
    "differential_drive": DIFFERENTIAL,
    "omnidirectional": OMNIDIRECTIONAL,
}

#: The classical baseline every family has. Not a learned planner, so not in `arena_planners`.
NAV2 = "nav2"


def robot_families(robots_root: Path) -> dict[str, str]:
    """`{robot: family}` for every robot directory under `robots_root`.

    A `<name>_plus` entry is a sensor pack on `<name>` (its yaml lists laserscan/pointcloud,
    no controllers) and inherits the parent's family. A robot whose yaml names no known
    controller is reported as `unknown` rather than dropped - silence here would look like
    "not a robot".
    """
    out: dict[str, str] = {}
    for d in sorted(p for p in robots_root.iterdir() if p.is_dir()):
        text = "\n".join(f.read_text(errors="replace") for f in d.glob("*.y*ml"))
        family = next((fam for needle, fam in _CONTROLLER_FAMILY if needle in text), None)
        out[d.name] = family or "unknown"
    for name, family in list(out.items()):
        if family == "unknown" and name.endswith("_plus") and name[:-5] in out:
            out[name] = out[name[:-5]]
    return out


def planner_families(planners_root: Path) -> dict[str, str]:
    """`{planner: family}` for every planner directory carrying a `planner.yaml`.

    Directories without one (`crowdnav`, `dsrnn`, ...) are not loadable by the `drl` adapter
    and are omitted - a command for them would fail at boot.
    """
    out: dict[str, str] = {}
    for d in sorted(p for p in planners_root.iterdir() if p.is_dir()):
        manifest = d / "planner.yaml"
        if not manifest.is_file():
            continue
        doc = yaml.safe_load(manifest.read_text()) or {}
        action = str(doc.get("action_type", "")).strip()
        out[d.name] = _ACTION_FAMILY.get(action, "unknown")
    return out


def scenarios(world_root: Path) -> tuple[list[str], list[str]]:
    """`(cases, bases)`: case scenarios carry `__`, bases do not. `default` is excluded."""
    names = sorted(p.name for p in (world_root / "scenarios").iterdir() if (p / "scenario.yaml").is_file())
    cases = [n for n in names if "__" in n]
    bases = [n for n in names if "__" not in n and n != "default"]
    return cases, bases


def command(*, scenario: str, robot: str, world: str, planner: str, human: str = "arena") -> str:
    """One launch line, in the shape every roll-out arm used."""
    mobile = f"robot.mobile:={NAV2}" if planner == NAV2 else f"robot.mobile:=drl robot.mobile.planner:={planner}"
    return (
        f"arena launch task.robots:=edge_case task.obstacles:=edge_case "
        f"task.scenario.file:={scenario} robot:={robot} human:={human} world:={world} {mobile}"
    )


def pairs(robots: Mapping[str, str], planners: Mapping[str, str]) -> list[tuple[str, str, str]]:
    """`(robot, planner, family)` for every compatible pairing; nav2 first in each family."""
    out: list[tuple[str, str, str]] = []
    for robot, family in robots.items():
        if family not in (DIFFERENTIAL, OMNIDIRECTIONAL):
            continue
        out.append((robot, NAV2, family))
        out.extend((robot, p, family) for p, pf in planners.items() if pf == family)
    return out


def render(
    *,
    world: str,
    robots: Mapping[str, str],
    planners: Mapping[str, str],
    cases: Sequence[str],
    bases: Sequence[str],
    generated_by: str,
) -> str:
    """The whole file. Matrix first, then one block per robot × planner."""
    all_pairs = pairs(robots, planners)
    by_family = {DIFFERENTIAL: [], OMNIDIRECTIONAL: []}
    for r, f in robots.items():
        if f in by_family:
            by_family[f].append(r)
    excluded = {r: f for r, f in robots.items() if f not in by_family}
    planners_of = {
        f: [NAV2] + [p for p, pf in planners.items() if pf == f] for f in by_family
    }

    lines: list[str] = []
    a = lines.append
    a(f"# AMBUSH roll-out commands — `{world}`")
    a("")
    a(f"Generated by `{generated_by}`. **Do not edit by hand** — regenerate, so the matrix tracks the")
    a("workspace: robot families come from each robot's ros2_control controller, planner families from")
    a("`planner.yaml` `action_type`, scenarios from the world's `scenarios/` directory.")
    a("")
    a("Run from the container. Once per shell:")
    a("")
    a("```bash")
    a("export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe")
    a("export GZ_IP=127.0.0.1 GZ_RELAY=127.0.0.1   # AMBUSH_TEST_COMMANDS.md §9 - WiFi roams kill gz-transport otherwise")
    a("```")
    a("")
    a("Every command below follows one shape; append `headless:=true` for unattended runs and")
    a("`task.auto_reset:=true` to keep cycling episodes. `sim:=gazebo` is the launch default and is")
    a("omitted. Wait for `EPISODE STARTED` before trusting anything (AMBUSH_TEST_COMMANDS.md §3).")
    a("")
    a("## 1. The matrix")
    a("")
    a("Arms compare only within a drive family, so each robot is paired with its family's planners only.")
    a("")
    a("| Family | Robots | Planners |")
    a("|---|---|---|")
    for f in (DIFFERENTIAL, OMNIDIRECTIONAL):
        a(f"| `{f}` | {', '.join(f'`{r}`' for r in by_family[f])} | {', '.join(f'`{p}`' for p in planners_of[f])} |")
    if excluded:
        a("")
        a("Listed, not paired: " + ", ".join(f"`{r}` ({f})" for r, f in excluded.items()) + ".")
    a("")
    a(f"{len(all_pairs)} robot × planner arms; {len(cases)} case scenarios and {len(bases)} base scenarios each; "
      f"**{len(all_pairs) * (len(cases) + len(bases))} commands**.")
    a("")
    a("Known as of 2026-08-26: `mpo700` shows 0.000 m displacement on every planner (chassis, not planner);")
    a("`sicnav` crashes on its first pedestrian step (`planners/sicnav/planner.py:105`). Commands are listed")
    a("regardless - the matrix is what ships, the caveats are what was measured.")
    a("")
    a("## 2. Scenarios")
    a("")
    a("Cases (`<base>__<case>`): " + ", ".join(f"`{c}`" for c in cases))
    a("")
    a("Bases (controls): " + ", ".join(f"`{b}`" for b in bases))
    a("")
    a("## 3. Commands, per robot × planner")
    a("")
    for robot, planner, family in all_pairs:
        a(f"### `{robot}` × `{planner}` ({family})")
        a("")
        a("```bash")
        a("# controls - the base scenarios, no edge case")
        for b in bases:
            a(command(scenario=b, robot=robot, world=world, planner=planner))
        a("# cases")
        for c in cases:
            a(command(scenario=c, robot=robot, world=world, planner=planner))
        a("```")
        a("")
    return "\n".join(lines).rstrip() + "\n"


def workspace_roots(worlds_root: Path) -> tuple[Path, Path]:
    """`(robots_root, planners_root)` relative to the worlds directory of a checkout."""
    src = worlds_root.resolve()
    while src.name != "src" and src.parent != src:
        src = src.parent
    return (
        src / "Arena" / "arena_robots" / "arena_robots" / "robots",
        src / "Arena" / "arena_planners" / "planners",
    )


_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def safe_world(world: str) -> str:
    return _SAFE.sub("_", world)
