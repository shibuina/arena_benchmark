"""Validation rules. Every one of these is a spec that would otherwise run and mislead."""

from __future__ import annotations

import pytest
from arena_benchmark.spec import SpecError, parse_spec

BASE = {
    "version": 1,
    "worlds": ["w1"],
    "arms": [{"id": "nav2", "robot": "jackal", "mobile": "nav2", "drive": "differential"}],
    "cells": [{"id": "base", "params": {"task.edge_case.inject": 0}}],
}


def spec(**overrides):
    return parse_spec({**BASE, **overrides})


def test_minimal_spec_parses():
    s = spec()
    assert s.n_episodes == 1 * 1 * 1 * 3
    assert s.seeds == (1000, 1001, 1002)


def test_unknown_top_level_key_is_an_error():
    with pytest.raises(SpecError, match="unknown key"):
        spec(scenaria=["oops"])


def test_unknown_cell_key_is_an_error():
    with pytest.raises(SpecError, match="unknown key"):
        spec(cells=[{"id": "c", "aproach_angle": 90}])


def test_unsupported_version_is_refused():
    with pytest.raises(SpecError, match="version"):
        spec(version=99)


def test_drl_arm_needs_a_planner():
    with pytest.raises(SpecError, match="needs a 'planner'"):
        spec(arms=[{"id": "a", "robot": "jackal", "mobile": "drl", "drive": "differential"}])


def test_classical_arm_rejects_a_planner_name():
    """A `planner:` on a nav2 arm is a typo that would otherwise be silently dropped."""
    with pytest.raises(SpecError, match="takes no 'planner'"):
        spec(arms=[{"id": "a", "robot": "jackal", "mobile": "nav2", "drive": "differential", "planner": "cadrl"}])


def test_unknown_drive_family_is_refused():
    with pytest.raises(SpecError, match="drive="):
        spec(arms=[{"id": "a", "robot": "jackal", "mobile": "nav2", "drive": "tracked"}])


def test_learned_family_without_a_baseline_is_refused():
    """The confound the whole arm design exists to close: no control on that chassis."""
    with pytest.raises(SpecError, match="no baseline"):
        spec(
            arms=[
                {"id": "nav2", "robot": "jackal", "mobile": "nav2", "drive": "differential"},
                {"id": "rlrvo", "robot": "mpo700", "mobile": "drl", "planner": "rlrvo", "drive": "omnidirectional"},
            ]
        )


def test_each_family_may_carry_its_own_baseline():
    s = spec(
        arms=[
            {"id": "nav2_diff", "robot": "jackal", "mobile": "nav2", "drive": "differential"},
            {"id": "cadrl", "robot": "jackal", "mobile": "drl", "planner": "cadrl", "drive": "differential"},
            {"id": "nav2_omni", "robot": "mpo700", "mobile": "nav2", "drive": "omnidirectional"},
            {"id": "rlrvo", "robot": "mpo700", "mobile": "drl", "planner": "rlrvo", "drive": "omnidirectional"},
        ]
    )
    assert sorted(s.families()) == ["differential", "omnidirectional"]


def test_launch_time_setting_in_cell_params_is_refused():
    """`robot:=` cannot be changed between episodes, so it must not look like it can."""
    with pytest.raises(SpecError, match="not a node parameter"):
        spec(cells=[{"id": "c", "params": {"robot": "husky"}}])


def test_duplicate_ids_are_refused():
    with pytest.raises(SpecError, match="duplicate id"):
        spec(cells=[{"id": "c"}, {"id": "c"}])
    with pytest.raises(SpecError, match="duplicate id"):
        spec(arms=[BASE["arms"][0], BASE["arms"][0]])


def test_knob_without_a_level_is_refused():
    with pytest.raises(SpecError, match="needs a 'level'"):
        spec(cells=[{"id": "c", "knob": "need_decay"}])


def test_grid_expands_to_one_cell_per_knob_level():
    s = spec(grid={"knobs": ["need_decay", "step_patience"], "levels": [0.95, 0.99]})
    ids = [c.id for c in s.cells]
    assert ids == ["base", "need_decay@0.95", "need_decay@0.99", "step_patience@0.95", "step_patience@0.99"]


def test_arm_launch_args_pin_the_rate():
    """The default tick comes from the robot's own caps file and so differs per arm."""
    s = spec(
        arms=[
            {"id": "nav2", "robot": "jackal", "mobile": "nav2", "drive": "differential"},
            {"id": "a", "robot": "jackal", "mobile": "drl", "planner": "cadrl", "drive": "differential", "rate_hz": 5},
        ]
    )
    args = s.arms[1].launch_args()
    assert args["robot.mobile"] == "drl"
    assert args["robot.mobile.planner"] == "cadrl"
    assert args["robot.mobile.rate"] == "5"


def test_every_shipped_spec_is_valid():
    """A shipped spec that does not parse is a broken release, not a broken run."""
    from arena_benchmark.spec import available, expand, load_spec, resolve

    names = available()
    assert names, "no specs shipped"
    for name in names:
        spec = load_spec(resolve(name))
        episodes = expand(spec)
        assert episodes, f"{name}: expands to nothing"
        assert len(episodes) == spec.n_episodes
