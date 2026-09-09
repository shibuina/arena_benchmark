from __future__ import annotations

from pathlib import Path

import pytest
from arena_benchmark.promptgen import prompts as P
from arena_benchmark.variance.diversity import (
    EXPRESSIBLE_CASES,
    NEAR_DUPLICATE_JACCARD,
    prompt_report,
    scenario_report,
)

WORLDS = Path(__file__).resolve().parents[2] / "arena_simulation_setup" / "worlds"


def _shipped(name):
    path = P.shipped_dir() / f"{name}.yaml"
    if not path.is_file():
        pytest.skip(f"no shipped set {name}")
    return P.read(path)


def _shipped_names():
    return sorted(p.stem for p in P.shipped_dir().glob("*.yaml"))


@pytest.mark.parametrize("name", _shipped_names())
def test_every_shipped_set_reads_varied(name):
    prompts = _shipped(name)
    r = prompt_report(prompts)
    assert r["n"] == len(prompts)
    assert len(r["depth"]) >= 3, "never one depth only"
    assert r["text_distinct"] == r["n"]
    assert r["text_similarity_mean"] < 0.15
    # the frozen arena_arena_002 set carries two near-duplicate pairs (_003/_009, _006/_008): tolerated, recorded
    assert len(r["near_duplicates"]) <= max(1, r["n"] // 10)
    assert r["timings"] >= 3 and len(r["profiles"]) >= 4 and r["places"] >= 3
    # the frozen first sets score 0.76-0.85; a supplement is allowed to be lopsided because it
    # fills the first set's gaps (it is measured on the union below)
    assert r["score"] >= (0.6 if name.endswith("_2") else 0.7)


@pytest.mark.parametrize("world", ["", "hospital_1", "arena_arena_002", "acoustics_cross6_narrow"])
def test_the_sets_a_world_loads_cover_every_case(world):
    """universal + universal_2 (+ <world> + <world>_2): every expressible case at least three times,
    and the universal pair alone at least twice each - so a supplement cannot be skipped."""
    universal = [p for p in P.load_sets("") if p.scope == "universal"]
    r_u = prompt_report(universal)
    assert not [c for c in EXPRESSIBLE_CASES if r_u["case_counts"][c] < 2], r_u["case_counts"]
    assert r_u["case_entropy"] >= 0.9
    if world:
        own = [p for p in P.load_sets(world) if p.scope == "world"]
        if not own:
            pytest.skip(f"no shipped set for {world}")
        r_w = prompt_report(own)
        assert not r_w["missing_cases"], r_w["case_counts"]
        assert r_w["case_entropy"] >= 0.85
    r = prompt_report(P.load_sets(world))
    assert not [c for c in EXPRESSIBLE_CASES if r["case_counts"][c] < 3], r["case_counts"]
    assert r["case_entropy"] >= 0.9 and r["score"] >= 0.75


def test_balanced_generation_fills_the_gaps_of_the_set_it_extends():
    base = P.generate(12, seed=6)
    r0 = prompt_report(base)
    assert r0["missing_cases"], "a small random set leaves cases untouched"
    more = P.generate_balanced(12, seed=6, avoid=base, start=13)
    assert [p.id for p in more] == [f"u_{i:03d}" for i in range(13, 25)]
    r = prompt_report(list(base) + more)
    assert not r["missing_cases"]
    assert r["case_entropy"] > r0["case_entropy"]
    assert r["text_distinct"] == 24 and not r["near_duplicates"]


def test_a_degenerate_set_scores_low_and_lists_the_duplicates():
    one = P.generate(1, seed=3)[0]
    clones = [P.Prompt(id=f"x_{i}", text=one.text, targets=one.targets, oracle=one.oracle, scope="universal", world="") for i in range(6)]
    r = prompt_report(clones)
    assert r["text_distinct"] == 1
    assert len(r["near_duplicates"]) == 15
    assert r["case_entropy"] < 0.4
    assert r["score"] < 0.35
    assert r["missing_cases"]


def test_generated_sets_get_more_diverse_with_more_prompts():
    small = prompt_report(P.generate(8, seed=1))
    large = prompt_report(P.generate(60, seed=1))
    assert large["combos"] > small["combos"]
    assert large["case_entropy"] >= small["case_entropy"]
    assert not large["missing_cases"]


def test_near_duplicate_threshold_is_a_real_fraction():
    assert 0.0 < NEAR_DUPLICATE_JACCARD < 1.0


@pytest.mark.parametrize("world", ["hospital_1", "arena_arena_002"])
def test_realised_scenarios_spread_over_cases_bases_and_onsets(world):
    root = WORLDS / world
    if not any((root / "scenarios").glob("*__*/record.json")):
        pytest.skip(f"{world}: no generated scenarios")
    r = scenario_report(root)
    assert r["n"] >= 40  # the core set is 12 prompts x 4 bases per world, minus infeasible ones
    assert r["status"].get("ok", 0) > r["status"].get("partial", 0)
    assert set(r["bases"]) >= {"normal", "break", "night", "empty"}
    assert not r["missing_cases"], r["steps"]
    assert r["case_entropy"] >= 0.8
    assert r["distinct_decisions"] >= 0.9
    assert r["onset_s"]["distinct"] >= 4 and r["onset_s"]["std"] >= 5.0
    assert r["route_fraction"]["distinct"] >= 3
    # the 13-prompt core set carries two approach angles and one `after` kind; the archived big set had three of each
    assert len(r["approach_angles"]) >= 2
    assert set(r["after"]) >= {"continue"}
