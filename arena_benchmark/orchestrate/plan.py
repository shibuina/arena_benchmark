"""Shard -> sweep plan: the bridge to the trial runner that already exists.

This module deliberately stops at generating a plan. ``edge_case_sweep`` already waits for
the stack *and* for the ``task.edge_case.*`` parameter surface to appear, drives each trial,
and records the outcome; a sweep that skipped that wait would run every trial against the
defaults and produce a file that looks like results. Re-implementing it here would mean
re-implementing that wait, which is the one part nobody should write twice.

Two details in the emitted plan are load-bearing:

``baseline: false``
    The runner can prepend its own unperturbed control arm per seed. Ours are explicit cells
    with ids and uids, so leaving the runner's on would add trials that are in the output and
    not in the manifest - the exact mismatch resume and assembly exist to detect.

``label: <episode_uid>``
    The runner copies the whole case dict into ``trials.jsonl``, so the uid arrives in the
    output verbatim. Without it the join back to the manifest would have to be inferred from
    knob/level/seed, which stops being unique the moment two cells share them and differ only
    in geometry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .shard import Shard


def sweep_plan(shard: Shard) -> dict[str, Any]:
    """The plan document for one shard's trials."""
    first = shard.episodes[0]

    cases: list[dict[str, Any]] = []
    for ep in shard.episodes:
        case: dict[str, Any] = {
            "label": ep.episode_uid,
            "knob": ep.knob,
            "level": ep.level,
            "seed": ep.seed,
        }
        if ep.catalogue:
            case["catalogue"] = ep.catalogue
        if ep.params:
            case["params"] = dict(ep.params)
        cases.append(case)

    return {
        "base_mode": first.base_mode,
        "scenario": first.scenario,
        "select": first.select,
        "timing": first.timing,
        "catalogue": first.catalogue,
        "driver": first.driver,
        "episode_duration": first.episode_duration,
        "baseline": False,
        "cases": cases,
    }


def write_plan(shard: Shard, directory: str | os.PathLike[str]) -> Path:
    """Write a shard's plan next to where its output will land.

    The plan is kept with the run rather than in a temp dir: it is the most direct record of
    what a shard was asked to do, and reading it beside ``trials.jsonl`` answers "was this
    trial skipped or never scheduled?" without re-expanding the spec.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plan.yaml"
    path.write_text(yaml.safe_dump(sweep_plan(shard), sort_keys=False, default_flow_style=False))
    return path
