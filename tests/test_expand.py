"""Expansion: the manifest is the run's contract, so it has to be exact and repeatable."""

from __future__ import annotations

import pytest
from arena_benchmark.spec import expand, parse_spec

SPEC = {
    "version": 1,
    "worlds": ["w1", "w2"],
    "arms": [
        {"id": "nav2", "robot": "jackal", "mobile": "nav2", "drive": "differential"},
        {"id": "cadrl", "robot": "jackal", "mobile": "drl", "planner": "cadrl", "drive": "differential"},
    ],
    "cells": [
        {"id": "base", "params": {"task.edge_case.inject": 0}},
        {"id": "c_only", "params": {"task.edge_case.inject": 1}},
    ],
    "seeds": {"start": 1000, "count": 3},
}


def test_expansion_is_the_full_product():
    eps = expand(parse_spec(SPEC))
    assert len(eps) == 2 * 2 * 2 * 3 == parse_spec(SPEC).n_episodes


def test_expansion_is_deterministic():
    a = [e.episode_uid for e in expand(parse_spec(SPEC))]
    b = [e.episode_uid for e in expand(parse_spec(SPEC))]
    assert a == b


def test_every_episode_uid_is_unique():
    eps = expand(parse_spec(SPEC))
    assert len({e.episode_uid for e in eps}) == len(eps)


def test_arms_of_one_comparison_share_a_cell_uid():
    """The pairing key: same situation, different planner."""
    eps = expand(parse_spec(SPEC))
    paired = [e for e in eps if e.world == "w1" and e.cell_id == "base" and e.seed == 1000]
    assert len(paired) == 2
    assert len({e.cell_uid for e in paired}) == 1
    assert {e.arm_id for e in paired} == {"nav2", "cadrl"}


def test_worlds_do_not_share_a_cell_uid():
    eps = expand(parse_spec(SPEC))
    by_world = {e.world: e.cell_uid for e in eps if e.cell_id == "base"}
    assert by_world["w1"] != by_world["w2"]


def test_two_cells_that_differ_only_by_name_are_refused():
    """A content hash cannot distinguish them, so nothing downstream could either."""
    spec = parse_spec(
        {
            **SPEC,
            "cells": [
                {"id": "one", "params": {"task.edge_case.inject": 1}},
                {"id": "two", "params": {"task.edge_case.inject": 1}},
            ],
        }
    )
    with pytest.raises(ValueError, match="differ.*only by name"):
        expand(spec)


def test_cell_catalogue_overrides_the_run_wide_default():
    spec = parse_spec({**SPEC, "catalogue": "level_d", "cells": [{"id": "a", "knob": "vision_fov", "level": 0.0, "catalogue": "level_a"}]})
    assert {e.catalogue for e in expand(spec)} == {"level_a"}


def test_rows_are_json_safe():
    row = expand(parse_spec(SPEC))[0].as_row()
    assert isinstance(row["params"], dict)
    assert isinstance(row["arm_extra"], dict)
