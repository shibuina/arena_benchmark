"""Cells authored inside scenario files.

The point of the form is that a scenario directory is a complete, self-describing situation:
who is in the corridor and what is being done to the robot live in one file.
"""

from __future__ import annotations

import pytest
import yaml

from arena_benchmark.spec import SpecError, expand, parse_spec
from arena_benchmark.spec.scenario_cells import cell_from_scenario, read_block


@pytest.fixture
def worlds(tmp_path):
    """Two worlds, each with scenario `1` (declaring a case) and `2` (declaring none)."""
    root = tmp_path / "worlds"
    for world in ("w1", "w2"):
        base = {"robots": [{"start": [0, 0, 0]}], "dynamic": [{"name": "p", "pose": [1, 1, 0]}]}
        one = root / world / "scenarios" / "1"
        one.mkdir(parents=True)
        (one / "scenario.yaml").write_text(yaml.safe_dump({
            **base,
            "edge_case": {
                "id": "c_cross90", "note": "crossing at the bend",
                "inject": 1, "place": "intercept",
                "approach_angle": 90.0, "encounter_at": 0.5,
            },
        }))
        two = root / world / "scenarios" / "2"
        two.mkdir(parents=True)
        (two / "scenario.yaml").write_text(yaml.safe_dump(base))
    return root


def spec_for(root, scenarios):
    return parse_spec({
        "version": 1,
        "worlds": ["w1", "w2"],
        "worlds_root": str(root),
        "arms": [{"id": "nav2", "robot": "jackal", "mobile": "nav2", "drive": "differential"}],
        "scenarios": scenarios,
        "seeds": {"start": 1, "count": 1},
    })


def test_block_becomes_task_parameters(worlds):
    cell = cell_from_scenario(worlds, "w1", "1")
    assert cell.id == "c_cross90"
    assert cell.scenario == "1"
    assert cell.params == {
        "task.edge_case.inject": 1,
        "task.edge_case.place": "intercept",
        "task.edge_case.approach_angle": 90.0,
        "task.edge_case.encounter_at": 0.5,
    }


def test_labels_do_not_become_parameters(worlds):
    """`id` and `note` name the cell; they are not node parameters."""
    cell = cell_from_scenario(worlds, "w1", "1")
    assert not any(k.endswith((".id", ".note")) for k in cell.params)
    assert cell.note == "crossing at the bend"


def test_a_scenario_without_a_block_is_a_control_cell(worlds):
    """Not an error and not skipped - the spec asked for it, so it runs unperturbed."""
    cell = cell_from_scenario(worlds, "w1", "2")
    assert cell.params == {"task.edge_case.inject": 0}
    assert cell.scenario == "2"


def test_unknown_block_key_is_refused(worlds, tmp_path):
    p = worlds / "w1" / "scenarios" / "1" / "scenario.yaml"
    doc = yaml.safe_load(p.read_text())
    doc["edge_case"]["aproach_angle"] = 90.0
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(SpecError, match="unknown key"):
        read_block(p)


def test_knob_without_level_is_refused(worlds):
    p = worlds / "w1" / "scenarios" / "1" / "scenario.yaml"
    doc = yaml.safe_load(p.read_text())
    doc["edge_case"]["knob"] = "need_decay"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(SpecError, match="'level' is not"):
        cell_from_scenario(p.parents[3], "w1", "1")


def test_missing_scenario_is_refused(worlds):
    with pytest.raises(SpecError, match="scenario not found"):
        cell_from_scenario(worlds, "w1", "nope")


def test_spec_reads_one_cell_per_world_and_scenario(worlds):
    """The same scenario name in two worlds is two situations, not one."""
    spec = spec_for(worlds, ["1", "2"])
    assert len(spec.cells) == 4
    assert {c.scenario for c in spec.cells} == {"1", "2"}


def test_each_scenario_becomes_its_own_shard(worlds):
    """`scenario` is launch-invariant, so authoring cells this way costs one boot each."""
    from arena_benchmark.orchestrate import shard

    episodes = expand(spec_for(worlds, ["1", "2"]))
    shards = shard(episodes)
    # 2 worlds x 2 scenarios, one arm
    assert len(shards) == 4
    assert all(len(s.episodes) == 1 for s in shards)


def test_cell_uid_separates_scenarios(worlds):
    """Two scenarios in one world are different situations and must not collide."""
    episodes = expand(spec_for(worlds, ["1", "2"]))
    by_scenario = {e.scenario: e.cell_uid for e in episodes if e.world == "w1"}
    assert by_scenario["1"] != by_scenario["2"]


def test_a_world_bound_cell_runs_only_in_its_own_world(worlds):
    """A case written in `w1/scenarios/1/` describes w1's corridor; replicating it onto w2
    would silently mean something else."""
    episodes = expand(spec_for(worlds, ["1"]))
    for e in episodes:
        assert e.world == "w1" or e.world == "w2"
    # one cell per world, not the 2x2 cross product a spec-authored cell would give
    assert len(episodes) == 2
    assert {(e.world, e.cell_id) for e in episodes} == {("w1", "c_cross90"), ("w2", "c_cross90")}


def test_same_label_in_two_worlds_is_not_a_duplicate(worlds):
    """Both worlds declare `id: c_cross90`; they are two cells, not a spec error."""
    spec = spec_for(worlds, ["1"])
    assert [c.id for c in spec.cells] == ["c_cross90", "c_cross90"]
    assert [c.world for c in spec.cells] == ["w1", "w2"]


def test_spec_and_scenario_cells_can_coexist(worlds):
    """A spec-authored cell still applies to every world; a scenario-authored one does not."""
    spec = parse_spec({
        "version": 1,
        "worlds": ["w1", "w2"],
        "worlds_root": str(worlds),
        "arms": [{"id": "nav2", "robot": "jackal", "mobile": "nav2", "drive": "differential"}],
        "cells": [{"id": "spec_base", "params": {"task.edge_case.inject": 0}}],
        "scenarios": ["1"],
        "seeds": {"start": 1, "count": 1},
    })
    eps = expand(spec)
    by_cell = {}
    for e in eps:
        by_cell.setdefault(e.cell_id, set()).add(e.world)
    assert by_cell["spec_base"] == {"w1", "w2"}     # unbound: every world
    assert by_cell["c_cross90"] == {"w1", "w2"}     # bound, but declared in both


def test_a_nested_plan_never_becomes_a_parameter(worlds):
    """The mode reads `waypoints:` straight out of the scenario file. Emitting
    `task.edge_case.waypoints` as well would set a parameter no node subscribes to - a silent
    no-op, which is the one failure this pipeline exists to prevent."""
    p = worlds / "w1" / "scenarios" / "1" / "scenario.yaml"
    doc = yaml.safe_load(p.read_text())
    doc["edge_case"] = {
        "id": "c_erratic",
        "inject": 0,
        "waypoints": {"mode": "shuffle"},
        "trigger": {"robot_within": 3.0},
    }
    p.write_text(yaml.safe_dump(doc))

    cell = cell_from_scenario(worlds, "w1", "1")
    assert cell.id == "c_erratic"
    assert cell.params == {"task.edge_case.inject": 0}


def test_two_plans_in_different_scenarios_are_different_cells(worlds):
    """The plan is not in `params`, so what separates them is the scenario each lives in."""
    for name, period in (("1", 2.0), ("2", 9.0)):
        p = worlds / "w1" / "scenarios" / name / "scenario.yaml"
        doc = yaml.safe_load(p.read_text())
        doc["edge_case"] = {"id": "c_erratic", "inject": 0,
                            "waypoints": {"mode": "shuffle", "period": period}}
        p.write_text(yaml.safe_dump(doc))

    a = cell_from_scenario(worlds, "w1", "1")
    b = cell_from_scenario(worlds, "w1", "2")
    assert a.scenario != b.scenario


def test_base_is_a_label_not_a_parameter(worlds):
    p = worlds / "w1" / "scenarios" / "1" / "scenario.yaml"
    doc = yaml.safe_load(p.read_text())
    doc["edge_case"]["base"] = "normal"
    p.write_text(yaml.safe_dump(doc))
    cell = cell_from_scenario(worlds, "w1", "1")
    assert not any("base" in k for k in cell.params)
