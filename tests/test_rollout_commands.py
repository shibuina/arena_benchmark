"""The robot × planner × scenario command matrix, derived from a fake workspace."""

from __future__ import annotations

from pathlib import Path

import pytest
from arena_benchmark.rollout.commands import (
    DIFFERENTIAL,
    LEGGED,
    NAV2,
    OMNIDIRECTIONAL,
    command,
    pairs,
    planner_families,
    render,
    robot_families,
    scenarios,
)


def _robot(root: Path, name: str, controller: str | None):
    d = root / name
    d.mkdir(parents=True)
    body = f"controller:\n  type: {controller}\n" if controller else "sensor:\n  type: laserscan\n"
    (d / "control.yaml").write_text(body)


def _planner(root: Path, name: str, action: str | None):
    d = root / name
    d.mkdir(parents=True)
    if action:
        (d / "planner.yaml").write_text(f"name: {name}\naction_type: {action}\n")


@pytest.fixture()
def ws(tmp_path):
    robots = tmp_path / "robots"
    _robot(robots, "jackal", "diff_drive_controller/DiffDriveController")
    _robot(robots, "mpo700", "arena_swerve_controller/SwerveController")
    _robot(robots, "rbkairos", "mecanum_drive_controller/MecanumDriveController")
    _robot(robots, "rbkairos_plus", None)
    _robot(robots, "go1", "joint_trajectory_controller/JointTrajectoryController")
    _robot(robots, "mystery", None)
    planners = tmp_path / "planners"
    _planner(planners, "cadrl", "differential_drive")
    _planner(planners, "rlrvo", "omnidirectional")
    _planner(planners, "crowdnav", None)
    world = tmp_path / "worlds" / "w"
    for s in ("default", "normal", "busy", "normal__blocking", "busy__stampede"):
        (world / "scenarios" / s).mkdir(parents=True)
        (world / "scenarios" / s / "scenario.yaml").write_text("robots: []\n")
    return robots, planners, world


def test_robot_families_come_from_the_controller(ws):
    robots, _, _ = ws
    fam = robot_families(robots)
    assert fam["jackal"] == DIFFERENTIAL
    assert fam["mpo700"] == OMNIDIRECTIONAL
    assert fam["rbkairos"] == OMNIDIRECTIONAL
    assert fam["go1"] == LEGGED


def test_a_plus_variant_inherits_its_parent(ws):
    robots, _, _ = ws
    assert robot_families(robots)["rbkairos_plus"] == OMNIDIRECTIONAL


def test_an_unrecognised_robot_is_listed_not_dropped(ws):
    """Silence would look like 'not a robot'."""
    robots, _, _ = ws
    assert robot_families(robots)["mystery"] == "unknown"


def test_only_manifest_bearing_planners_are_loadable(ws):
    _, planners, _ = ws
    fam = planner_families(planners)
    assert fam == {"cadrl": DIFFERENTIAL, "rlrvo": OMNIDIRECTIONAL}


def test_scenarios_split_cases_from_bases_and_skip_default(ws):
    _, _, world = ws
    assert scenarios(world) == (["busy__stampede", "normal__blocking"], ["busy", "normal"])


def test_pairs_never_cross_a_drive_family(ws):
    robots, planners, _ = ws
    got = pairs(robot_families(robots), planner_families(planners))
    assert ("jackal", "cadrl", DIFFERENTIAL) in got
    assert ("mpo700", "rlrvo", OMNIDIRECTIONAL) in got
    assert not any(r == "jackal" and p == "rlrvo" for r, p, _ in got)
    assert not any(r == "mpo700" and p == "cadrl" for r, p, _ in got)
    assert not any(r == "go1" for r, _, _ in got)


def test_every_wheeled_robot_gets_nav2_first(ws):
    robots, planners, _ = ws
    got = pairs(robot_families(robots), planner_families(planners))
    assert [p for r, p, _ in got if r == "jackal"][0] == NAV2


def test_command_shape_matches_the_rollout():
    line = command(scenario="normal__approach", robot="jackal", world="arena_arena_002", planner="cadrl")
    assert line == (
        "arena launch task.robots:=edge_case task.obstacles:=edge_case "
        "task.scenario.file:=normal__approach robot:=jackal human:=arena world:=arena_arena_002 "
        "robot.mobile:=drl robot.mobile.planner:=cadrl"
    )
    assert command(scenario="normal", robot="mpo700", world="w", planner=NAV2).endswith("robot.mobile:=nav2")


def test_render_counts_every_command(ws):
    robots, planners, world = ws
    cases, bases = scenarios(world)
    text = render(world="w", robots=robot_families(robots), planners=planner_families(planners),
                  cases=cases, bases=bases, generated_by="test")
    n_pairs = len(pairs(robot_families(robots), planner_families(planners)))
    assert text.count("arena launch ") == n_pairs * (len(cases) + len(bases))
    assert "`go1` (legged)" in text
    assert "`mystery` (unknown)" in text


def test_the_cli_filters_to_the_flagship_chassis_by_default(ws, tmp_path, monkeypatch):
    """One reference robot per drive family is what the benchmark compares on; the full
    inventory is one `--robots ''` away."""
    from arena_benchmark import cli
    from arena_benchmark.rollout import commands as mod

    robots, planners, world = ws
    monkeypatch.setattr(mod, "workspace_roots", lambda _w: (robots, planners))
    out = tmp_path / "out.md"
    assert cli.main(["commands", "w", "--worlds", str(world.parent), "--out", str(out)]) == 0
    text = out.read_text()
    assert "### `jackal` ×" in text and "### `mpo700` ×" in text
    assert "### `rbkairos` ×" not in text
    assert cli.main(["commands", "w", "--worlds", str(world.parent), "--out", str(out), "--robots", "nope"]) == 1
