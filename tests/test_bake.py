"""Baking a solved case into a portable scenario.

The scenario is the product, so these are tests about an artifact rather than about a
pipeline: does it carry literal coordinates, does it declare what it inherited, and does it
refuse to be built from a case that was never solved.
"""

from __future__ import annotations

import json

import pytest
import yaml

from arena_benchmark.bake import BakeError, bake_case, is_bakeable, scenario_id

CASE = {
    "status": "ok",
    "world": "testworld",
    "run_seed": "abc123",
    "episode_id": 4,
    "seed": 11,
    "place": "intercept",
    "spawn_pose": [-2.0047, 1.1525],
    "ped_goal": [-1.3765, -3.5091],
    "encounter_point": [-1.4766, -2.7658],
    "robot_leg": [[-1.9807, -6.6874], [-0.9726, 1.1557]],
    "approach_angle_requested": 180.0,
    "approach_angle_achieved": 195.0,
    "encounter_at_requested": 0.5,
    "encounter_at_achieved": 0.5,
    "geometry_retries": 27,
    "designed_ttc": 3.5944,
    "ped_speed": 1.1,
    "waypoint_mode": "reverse",
    "inject_model": "female_adult_business_02",
    "inject_type": "adult",
    "knob": "",
}


@pytest.fixture
def world(tmp_path):
    """A minimal world: geometry files for the hash, plus a base scenario."""
    root = tmp_path / "worlds" / "testworld"
    (root / "0").mkdir(parents=True)
    (root / "0" / "world.yaml").write_text("- name: room\n  corners: [{x: 0, y: 0}]\n")
    (root / "0" / "map.yaml").write_text("resolution: 0.05\norigin: [-1, -1, 0]\n")
    (root / "0" / "map.png").write_bytes(b"\x89PNG-not-really")
    scen = root / "scenarios" / "base"
    scen.mkdir(parents=True)
    return root, scen / "scenario.yaml"


def write_base(path, agents):
    path.write_text(yaml.safe_dump({"robots": [{"start": [0, 0, 0]}], "dynamic": agents}))


def test_bakes_a_scenario_with_literal_coordinates(world, tmp_path):
    root, scen = world
    write_base(scen, [{"name": "p1", "model": "m", "pose": [1.0, 2.0, 0.0], "agent": {"agent_type": "adult"}}])

    out = bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "suite")
    doc = yaml.safe_load((out / "scenario.yaml").read_text())

    assert doc["robots"][0]["start"][:2] == [-1.9807, -6.6874]
    assert doc["robots"][0]["goal"][:2] == [-0.9726, 1.1557]

    edge = doc["dynamic"][-1]
    assert edge["name"] == "edge_0"
    assert edge["pose"][:2] == [-2.0047, 1.1525]
    assert edge["waypoints"][0][:2] == [-1.3765, -3.5091]
    assert len(edge["waypoints"][0]) == 3
    assert edge["waypoint_mode"] == "reverse"
    assert edge["agent"]["desired_velocity"] == 1.1


def test_injected_agent_faces_its_goal(world, tmp_path):
    """`spawn_pose` carries no orientation, and a pedestrian facing the wrong way walks a
    different first second than the one that was solved."""
    import math

    root, scen = world
    write_base(scen, [])
    out = bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "suite")
    edge = yaml.safe_load((out / "scenario.yaml").read_text())["dynamic"][-1]

    expected = math.atan2(-3.5091 - 1.1525, -1.3765 - (-2.0047))
    assert edge["pose"][2] == pytest.approx(expected, abs=1e-3)


def test_base_population_is_copied_verbatim(world, tmp_path):
    root, scen = world
    agents = [{"name": "p1", "pose": [1.0, 2.0, 0.0]}, {"name": "p2", "pose": [3.0, 4.0, 0.0]}]
    write_base(scen, agents)

    out = bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "suite")
    doc = yaml.safe_load((out / "scenario.yaml").read_text())
    assert [a["name"] for a in doc["dynamic"]] == ["p1", "p2", "edge_0"]


def test_inherited_zone_refs_are_declared(world, tmp_path):
    """A copied base population that uses zone references makes the entry no more portable
    than its source. The entry has to say so rather than look portable."""
    root, scen = world
    write_base(scen, [{"name": "z1", "pose": "open_work_area"}, {"name": "lit", "pose": [1.0, 2.0, 0.0]}])

    out = bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "suite")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["inherited_zone_refs"] == ["z1"]


def test_a_fully_literal_base_declares_no_inheritance(world, tmp_path):
    root, scen = world
    write_base(scen, [{"name": "lit", "pose": [1.0, 2.0, 0.0]}])
    out = bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "suite")
    assert json.loads((out / "meta.json").read_text())["inherited_zone_refs"] == []


def test_meta_records_requested_and_achieved(world, tmp_path):
    """A 195-degree case must never be citable as a head-on one."""
    root, scen = world
    write_base(scen, [])
    out = bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "suite")
    c = json.loads((out / "meta.json").read_text())["construction"]
    assert (c["approach_angle_requested"], c["approach_angle_achieved"]) == (180.0, 195.0)
    assert c["geometry_retries"] == 27


def test_world_hash_is_recorded_and_changes_with_the_world(world, tmp_path):
    """A world edit silently invalidates every coordinate baked against it."""
    root, scen = world
    write_base(scen, [])
    before = json.loads((bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "a") / "meta.json").read_text())
    (root / "0" / "map.png").write_bytes(b"\x89PNG-different")
    after = json.loads((bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "b") / "meta.json").read_text())
    assert before["world_hash"] != after["world_hash"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"status": "aborted"},
        {"spawn_pose": None},
        {"ped_goal": None},
        {"robot_leg": None},
    ],
)
def test_unsolved_cases_are_refused(world, tmp_path, mutation):
    root, scen = world
    write_base(scen, [])
    with pytest.raises(BakeError):
        bake_case({**CASE, **mutation}, world_root=root, source_scenario=scen, out_root=tmp_path / "suite")


def test_is_bakeable_matches_what_bake_accepts():
    assert is_bakeable(CASE)
    assert not is_bakeable({**CASE, "status": "baseline"})
    assert not is_bakeable({**CASE, "spawn_pose": None})


def test_scenario_id_encodes_the_construction():
    assert scenario_id(CASE) == "c_a180_e50__testworld"


def test_replay_script_names_the_installed_scenario(world, tmp_path):
    root, scen = world
    write_base(scen, [])
    out = bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "suite")
    replay = (out / "replay.sh").read_text()
    assert "task.obstacles:=scenario" in replay
    assert "c_a180_e50__testworld" in replay
    # The whole point: a consumer needs no edge_case mode and no solver.
    assert "edge_case" not in replay


def test_two_solves_of_one_cell_do_not_overwrite_each_other(world, tmp_path):
    """Different seeds solve to different geometry, so they are different scenarios. Naming
    them only by their construction silently collapsed six baked entries into three."""
    root, scen = world
    write_base(scen, [])
    out_root = tmp_path / "suite"

    a = bake_case(CASE, world_root=root, source_scenario=scen, out_root=out_root)
    b = bake_case({**CASE, "seed": 12, "spawn_pose": [1.5, 2.5]}, world_root=root, source_scenario=scen, out_root=out_root)

    assert a != b
    assert len([p for p in out_root.iterdir() if p.is_dir()]) == 2


def test_identical_content_bakes_to_the_same_identity(world, tmp_path):
    """Content addressing: the same artifact keeps its uid regardless of which run produced it."""
    root, scen = world
    write_base(scen, [])
    a = bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "a")
    b = bake_case({**CASE, "run_seed": "other", "episode_id": 99}, world_root=root, source_scenario=scen, out_root=tmp_path / "b")
    ua = json.loads((a / "meta.json").read_text())["scenario_uid"]
    ub = json.loads((b / "meta.json").read_text())["scenario_uid"]
    assert ua == ub


def test_no_staging_directory_is_left_behind(world, tmp_path):
    root, scen = world
    write_base(scen, [])
    out_root = tmp_path / "suite"
    bake_case(CASE, world_root=root, source_scenario=scen, out_root=out_root)
    assert not [p for p in out_root.iterdir() if p.name.startswith(".staging")]


def test_meta_and_replay_name_the_final_directory(world, tmp_path):
    """The installed directory name embeds the uid, and both files must agree with it."""
    root, scen = world
    write_base(scen, [])
    out = bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "suite")
    assert json.loads((out / "meta.json").read_text())["scenario_id"] == out.name
    assert out.name in (out / "replay.sh").read_text()


def test_waypoints_are_three_element(world, tmp_path):
    """A two-element waypoint makes the loader drop the agent silently - the scenario comes up
    with every other agent and no warning at all."""
    root, scen = world
    write_base(scen, [])
    out = bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "suite")
    edge = yaml.safe_load((out / "scenario.yaml").read_text())["dynamic"][-1]
    for wp in edge["waypoints"]:
        assert len(wp) == 3, f"waypoint {wp} must be [x, y, theta]"


def test_baked_agent_shape_matches_a_shipped_scenario(world, tmp_path):
    """Structural parity with the format the loader actually accepts."""
    root, scen = world
    write_base(scen, [{"name": "p1", "model": "m", "pose": [1.0, 2.0, 0.0],
                       "agent": {"agent_type": "adult"}, "waypoints": [[3.0, 4.0, 0.0]]}])
    out = bake_case(CASE, world_root=root, source_scenario=scen, out_root=tmp_path / "suite")
    doc = yaml.safe_load((out / "scenario.yaml").read_text())
    base, edge = doc["dynamic"][0], doc["dynamic"][-1]
    assert set(base) <= set(edge) | {"waypoint_mode"}, "baked agent is missing keys the base has"
    assert len(edge["pose"]) == len(base["pose"]) == 3
