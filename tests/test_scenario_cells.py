from __future__ import annotations

import pytest
import yaml
from arena_benchmark.spec import SpecError, expand, parse_spec
from arena_benchmark.spec.scenario_cells import cell_from_scenario, effects_digest, read_block

BLOCK = {
    "id": "u_001", "prompt_id": "u_001", "steps": ["C:approach"], "note": "a child crosses",
    "effects": [{"type": "intercept", "profile": "child", "angle": 90.0, "route_fraction": 0.5}],
}


@pytest.fixture
def worlds(tmp_path):
    root = tmp_path / "worlds"
    for world in ("w1", "w2"):
        base = {"robots": [{"start": [0, 0, 0]}], "dynamic": [{"name": "p", "pose": [1, 1, 0]}]}
        one = root / world / "scenarios" / "1"
        one.mkdir(parents=True)
        (one / "scenario.yaml").write_text(yaml.safe_dump({**base, "edge_case": BLOCK}))
        two = root / world / "scenarios" / "2"
        two.mkdir(parents=True)
        (two / "scenario.yaml").write_text(yaml.safe_dump(base))
    return root


def spec_for(root, scenarios):
    return parse_spec({
        "version": 1, "worlds": ["w1", "w2"], "worlds_root": str(root),
        "arms": [{"id": "nav2", "robot": "jackal", "mobile": "nav2", "drive": "differential"}],
        "scenarios": scenarios, "seeds": {"start": 1, "count": 1},
    })


def test_block_reads_back(worlds) -> None:
    block = read_block(worlds / "w1" / "scenarios" / "1" / "scenario.yaml")
    assert block is not None and block["effects"][0]["type"] == "intercept"


def test_a_scenario_without_a_block_is_a_control_cell(worlds) -> None:
    cell = cell_from_scenario(worlds, "w1", "2")
    assert cell.id == "2__base" and cell.params == {}


def test_unknown_block_key_is_refused(worlds) -> None:
    path = worlds / "w1" / "scenarios" / "1" / "scenario.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["edge_case"]["knob"] = "vision_fov"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(SpecError, match="unknown key"):
        read_block(path)


def test_unknown_effect_type_is_refused(worlds) -> None:
    path = worlds / "w1" / "scenarios" / "1" / "scenario.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["edge_case"]["effects"] = [{"type": "sprint"}]
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(SpecError, match="effects\\[0\\]"):
        read_block(path)


def test_missing_scenario_is_refused(worlds) -> None:
    with pytest.raises(SpecError, match="not found"):
        cell_from_scenario(worlds, "w1", "3")


def test_effects_digest_ignores_labels() -> None:
    a = {"id": "x", "effects": [{"type": "rally", "target": "exit"}]}
    b = {"id": "y", "note": "renamed", "effects": [{"type": "rally", "target": "exit"}]}
    c = {"id": "x", "effects": [{"type": "rally", "target": "lobby"}]}
    assert effects_digest(a) == effects_digest(b) != effects_digest(c)


def test_spec_reads_one_cell_per_world_and_scenario(worlds) -> None:
    episodes = list(expand(spec_for(worlds, ["1", "2"])))
    assert len(episodes) == 4
    assert {e.cell_id for e in episodes} == {"u_001", "2__base"}


def test_cell_uid_separates_scenarios(worlds) -> None:
    episodes = list(expand(spec_for(worlds, ["1", "2"])))
    uids = {(e.world, e.scenario): e.cell_uid for e in episodes}
    assert uids[("w1", "1")] != uids[("w1", "2")]


def test_same_effects_in_two_worlds_are_different_cells(worlds) -> None:
    episodes = list(expand(spec_for(worlds, ["1"])))
    assert len({e.cell_uid for e in episodes}) == 2


def test_different_effects_are_different_cells(worlds) -> None:
    path = worlds / "w2" / "scenarios" / "1" / "scenario.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["edge_case"]["effects"][0]["angle"] = 180.0
    path.write_text(yaml.safe_dump(doc))
    a = cell_from_scenario(worlds, "w1", "1").params["edge_case.effects_digest"]
    b = cell_from_scenario(worlds, "w2", "1").params["edge_case.effects_digest"]
    assert a != b
