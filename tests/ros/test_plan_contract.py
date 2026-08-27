"""The contract with `edge_case_sweep`: a generated plan is a plan the runner accepts.

This module never re-implements trial execution, which makes the plan file the entire
interface between the two packages - and an interface nothing tests is one that breaks
silently. If the sweep plan format gains a required key or tightens a rule, this fails here
rather than four hours into a run.

Marked `ros` because importing `task_generator.utils.edge_case_plan` pulls in the package,
which needs rclpy on the path.
"""

from __future__ import annotations

import pytest

from arena_benchmark.orchestrate import shard, write_plan
from arena_benchmark.spec import available, expand, load_spec, resolve

pytestmark = pytest.mark.ros

load_plan = pytest.importorskip(
    "task_generator.utils.edge_case_plan", reason="needs a sourced ROS environment"
).load_plan


@pytest.mark.parametrize("name", available())
def test_every_shard_plan_is_accepted_by_the_runner(name, tmp_path):
    spec = load_spec(resolve(name))
    shards = shard(expand(spec))
    assert shards

    for s in shards:
        path = write_plan(s, tmp_path / s.label)
        _settings, trials = load_plan(str(path))

        assert len(trials) == len(s.episodes), f"{s.label}: {len(trials)} trials for {len(s.episodes)} episodes"

        labels = [t.get("label") for t in trials]
        assert all(labels), f"{s.label}: a trial reached the runner with no label to join on"
        assert len(set(labels)) == len(trials), f"{s.label}: duplicate labels"
        assert labels == [e.episode_uid for e in s.episodes], f"{s.label}: labels reordered"


def test_the_runners_own_baseline_arm_stays_off(tmp_path):
    """It would add trials that have no manifest row, which resume would read as a gap."""
    spec = load_spec(resolve("smoke_v1"))
    s = shard(expand(spec))[0]
    _settings, trials = load_plan(str(write_plan(s, tmp_path)))
    assert len(trials) == len(s.episodes)
