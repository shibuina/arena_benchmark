"""How varied a prompt set - and the scenarios realised from it - actually is.

Prompt level (`prompt_report`): which taxonomy cases the set targets and how evenly
(`case_counts`, `case_entropy`), how deep the prompts are (`depth`), how many distinct timings,
profiles, places and step combinations it uses, and how alike its texts are (token-trigram
Jaccard: `text_similarity_mean`, `near_duplicates`). `score` folds the normalised parts into one
number in [0, 1]; a set of one prompt repeated scores ~0, the shipped sets score > 0.7.

Scenario level (`scenario_report`): over a world's generated `<base>__<id>/` directories - status
mix, distinct decision keys, per-step and per-base counts, the spread of onsets, route fractions
and approach angles the effects use, and how many agents the cases touch.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import yaml

from ..promptgen.prompts import Prompt
from ..promptgen.spec import A_CASES, B_CASES, C_CASES, D_CASES

ALL_CASES: tuple[str, ...] = tuple(f"{s}:{c}" for s, cases in (("A", A_CASES), ("B", B_CASES), ("C", C_CASES), ("D", D_CASES)) for c in cases)
#: `erratic_behaviour` has no template in the prompt generator; the sets are measured against the cases they can express.
EXPRESSIBLE_CASES: tuple[str, ...] = tuple(c for c in ALL_CASES if c != "B:erratic_behaviour")
NEAR_DUPLICATE_JACCARD = 0.6


def _entropy(counts: Sequence[int]) -> float:
    """Shannon entropy normalised by the maximum for that many bins (1 = perfectly even)."""
    total = sum(counts)
    bins = len(counts)
    if total <= 0 or bins <= 1:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts if c)
    return h / math.log(bins)


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    toks = re.findall(r"[a-z']+", text.lower())
    return {tuple(toks[i:i + 3]) for i in range(len(toks) - 2)} if len(toks) >= 3 else {tuple(toks)}  # type: ignore[misc]


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def _timing_keys(p: Prompt) -> set[str]:
    keys: set[str] = set()
    for step, spec in (p.oracle or {}).items():
        if not isinstance(spec, dict):
            continue
        when = spec.get("when")
        if isinstance(when, dict):
            keys.add(f"{step}:{json.dumps(when, sort_keys=True)}")
        if "duration" in spec:
            keys.add(f"{step}:dur={spec['duration']}")
        if step == "A":
            keys.add(f"A:{spec.get('when', 'none')}")
    return keys


def _profiles(p: Prompt) -> set[str]:
    out: set[str] = set()
    for spec in (p.oracle or {}).values():
        if not isinstance(spec, dict):
            continue
        if spec.get("profile"):
            out.add(str(spec["profile"]))
        for m in spec.get("members") or []:
            out.add(str(m))
        sel = spec.get("select")
        if isinstance(sel, dict) and sel.get("profile"):
            out.add(str(sel["profile"]))
    return out


def _places(p: Prompt) -> set[str]:
    out: set[str] = set()
    for spec in (p.oracle or {}).values():
        if not isinstance(spec, dict):
            continue
        if spec.get("route_fraction") is not None:
            out.add(f"rf={spec['route_fraction']}")
        place = spec.get("place")
        if isinstance(place, dict):
            out.add("place=" + json.dumps(place, sort_keys=True))
        if spec.get("zone"):
            out.add(f"zone={spec['zone']}")
        if spec.get("angle") is not None:
            out.add(f"angle={spec['angle']}")
    return out


def prompt_report(prompts: Sequence[Prompt]) -> dict[str, Any]:
    n = len(prompts)
    case_counts = Counter(t for p in prompts for t in p.targets)
    counts = [case_counts.get(c, 0) for c in EXPRESSIBLE_CASES]
    depth = Counter(len(p.targets) for p in prompts)
    combos = {tuple(sorted(p.targets)) for p in prompts}
    timings = set().union(*(_timing_keys(p) for p in prompts)) if prompts else set()
    profiles = set().union(*(_profiles(p) for p in prompts)) if prompts else set()
    places = set().union(*(_places(p) for p in prompts)) if prompts else set()
    grams = [_trigrams(p.text) for p in prompts]
    sims = [_jaccard(a, b) for a, b in combinations(grams, 2)]
    near = [(prompts[i].id, prompts[j].id) for (i, j), s in zip(combinations(range(n), 2), sims, strict=True) if s >= NEAR_DUPLICATE_JACCARD]
    sim_mean = sum(sims) / len(sims) if sims else 0.0
    texts = len({p.text.strip().lower() for p in prompts})
    parts = {
        "case_entropy": _entropy(counts),
        "case_min_share": (min(counts) / max(1, max(counts))) if counts else 0.0,
        "combo_distinct": len(combos) / n if n else 0.0,
        "text_distinct": texts / n if n else 0.0,
        "text_dissimilarity": 1.0 - sim_mean,
        "timing_variety": min(1.0, len(timings) / 10.0),
        "profile_variety": min(1.0, len(profiles) / 8.0),
    }
    return {
        "n": n,
        "case_counts": {c: case_counts.get(c, 0) for c in ALL_CASES},
        "missing_cases": [c for c in EXPRESSIBLE_CASES if not case_counts.get(c)],
        "case_entropy": round(parts["case_entropy"], 4),
        "depth": {int(k): v for k, v in sorted(depth.items())},
        "combos": len(combos),
        "timings": len(timings),
        "profiles": sorted(profiles),
        "places": len(places),
        "text_similarity_mean": round(sim_mean, 4),
        "near_duplicates": near,
        "text_distinct": texts,
        "score": round(sum(parts.values()) / len(parts), 4),
        "parts": {k: round(v, 4) for k, v in parts.items()},
    }


# --- realised scenarios -----------------------------------------------------------------------


def _decisions(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("decisions")
    if isinstance(raw, dict):
        return raw
    try:
        return dict(ast.literal_eval(str(raw)))
    except (ValueError, SyntaxError):
        return {}


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def scenario_report(world_root: str | Path, *, include_ladder: bool = False) -> dict[str, Any]:
    root = Path(world_root) / "scenarios"
    status = Counter()
    bases = Counter()
    steps = Counter()
    depth = Counter()
    keys: set[str] = set()
    onsets: list[float] = []
    fractions: list[float] = []
    angles: set[float] = set()
    afters: Counter = Counter()
    effect_types: Counter = Counter()
    cast: list[int] = []
    n = 0
    for rec in sorted(root.glob("*__*/record.json")):
        name = rec.parent.name
        if not include_ladder and re.search(r"_l\d+$", name):
            continue
        d = json.loads(rec.read_text())
        n += 1
        status[str(d.get("status"))] += 1
        bases[str(d.get("base"))] += 1
        keys.add(str(d.get("dedup_key") or name))
        st = list(d.get("steps") or [])
        depth[len(st)] += 1
        for s in st:
            steps[s] += 1
        dec = _decisions(d)
        c = dec.get("C") or {}
        if isinstance(c, dict):
            if c.get("angle") is not None:
                angles.add(round(float(c["angle"]), 0))
            if c.get("route_fraction") is not None:
                fractions.append(float(c["route_fraction"]))
            if c.get("after"):
                afters[str(c["after"])] += 1
        sc = rec.parent / "scenario.yaml"
        if sc.is_file():
            doc = yaml.safe_load(sc.read_text()) or {}
            block = doc.get("edge_case") or {}
            for e in block.get("effects") or []:
                effect_types[str(e.get("type"))] += 1
                when = e.get("when") or {}
                if isinstance(when, dict):
                    if when.get("at") is not None:
                        onsets.append(float(when["at"]))
                    if when.get("route_fraction") is not None:
                        fractions.append(float(when["route_fraction"]))
            cast.append(len(doc.get("dynamic") or []))
    counts = [steps.get(c, 0) for c in EXPRESSIBLE_CASES]
    return {
        "world": Path(world_root).name,
        "n": n,
        "status": dict(status),
        "bases": dict(bases),
        "steps": dict(steps.most_common()),
        "missing_cases": [c for c in EXPRESSIBLE_CASES if not steps.get(c)],
        "case_entropy": round(_entropy(counts), 4),
        "depth": {int(k): v for k, v in sorted(depth.items())},
        "distinct_decisions": round(len(keys) / n, 4) if n else 0.0,
        "effect_types": dict(effect_types),
        "onset_s": {"n": len(onsets), "mean": round(sum(onsets) / len(onsets), 1) if onsets else None, "std": round(_std(onsets), 1), "distinct": len(set(onsets))},
        "route_fraction": {"n": len(fractions), "std": round(_std(fractions), 3), "distinct": len(set(fractions))},
        "approach_angles": sorted(angles),
        "after": dict(afters),
        "cast_size": {"min": min(cast) if cast else 0, "max": max(cast) if cast else 0, "mean": round(sum(cast) / len(cast), 1) if cast else 0},
    }


def format_prompt_report(name: str, r: dict[str, Any]) -> str:
    lines = [f"== {name}: {r['n']} prompts, diversity score {r['score']:.2f}"]
    lines.append("   cases: " + ", ".join(f"{k.split(':')[1]}={v}" for k, v in r["case_counts"].items() if v) + (f"   MISSING {r['missing_cases']}" if r["missing_cases"] else ""))
    lines.append(f"   case entropy {r['case_entropy']:.2f}; depth {r['depth']}; {r['combos']} step combinations; {r['timings']} timings; {len(r['profiles'])} profiles; {r['places']} places")
    lines.append(f"   text: {r['text_distinct']} distinct, mean trigram similarity {r['text_similarity_mean']:.3f}, near-duplicates {r['near_duplicates'] or 'none'}")
    lines.append("   parts: " + ", ".join(f"{k}={v:.2f}" for k, v in r["parts"].items()))
    return "\n".join(lines)


def format_scenario_report(r: dict[str, Any]) -> str:
    lines = [f"== {r['world']}: {r['n']} scenarios {r['status']}, bases {r['bases']}"]
    lines.append(f"   steps: {r['steps']}" + (f"   MISSING {r['missing_cases']}" if r["missing_cases"] else ""))
    lines.append(f"   case entropy {r['case_entropy']:.2f}; depth {r['depth']}; distinct decisions {r['distinct_decisions']:.2f}")
    lines.append(f"   effects {r['effect_types']}; onsets {r['onset_s']}; route fractions {r['route_fraction']}; angles {r['approach_angles']}; after {r['after']}; cast {r['cast_size']}")
    return "\n".join(lines)
