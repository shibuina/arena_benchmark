"""Sharding: the grouping that decides whether a run takes 170 hours or 21."""

from __future__ import annotations

from arena_benchmark.orchestrate import shard, summarise
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


def episodes():
    return expand(parse_spec(SPEC))


def test_one_shard_per_world_and_arm():
    shards = shard(episodes())
    assert len(shards) == 2 * 2
    assert {s.label for s in shards} == {"w1__nav2", "w1__cadrl", "w2__nav2", "w2__cadrl"}


def test_cells_and_seeds_stay_inside_one_shard():
    """They are parameter sets, so they must not cost a relaunch."""
    for s in shard(episodes()):
        assert len(s.episodes) == 2 * 3


def test_sharding_is_deterministic():
    assert [s.label for s in shard(episodes())] == [s.label for s in shard(episodes())]


def test_launch_args_select_the_arm():
    by_label = {s.label: s.launch_args() for s in shard(episodes())}
    assert by_label["w1__cadrl"]["robot.mobile"] == "drl"
    assert by_label["w1__cadrl"]["robot.mobile.planner"] == "cadrl"
    assert "robot.mobile.planner" not in by_label["w1__nav2"]
    assert by_label["w1__nav2"]["world"] == "w1"


def test_launch_args_use_current_names_not_deprecated_aliases():
    """`tm_robots`/`tm_obstacles` still resolve, but warn once per launch - across a few
    hundred shards that noise hides a real warning."""
    args = shard(episodes())[0].launch_args()
    assert args["task.robots"] == "edge_case"
    assert args["task.obstacles"] == "edge_case"
    assert "tm_robots" not in args
    assert "tm_obstacles" not in args


def test_arm_extra_reaches_the_launch_and_splits_shards():
    spec = parse_spec(
        {
            **SPEC,
            "arms": [
                {"id": "nav2", "robot": "jackal", "mobile": "nav2", "drive": "differential"},
                {"id": "nav2_tuned", "robot": "jackal", "mobile": "nav2", "drive": "differential", "extra": {"headless": "true"}},
            ],
        }
    )
    shards = shard(expand(spec))
    assert len(shards) == 4
    tuned = next(s for s in shards if s.arm_id == "nav2_tuned")
    assert tuned.launch_args()["headless"] == "true"


def test_summary_counts_the_whole_run():
    info = summarise(shard(episodes()))
    assert info["shards"] == 4
    assert info["episodes"] == 24
    assert info["arms"] == ["cadrl", "nav2"]
