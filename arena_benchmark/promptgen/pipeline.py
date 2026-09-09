"""prompt + world + base -> a scenario directory (`new_plan_2.md` §5, §7, §9).

Steps run in the plan's order - A by lexicon, D by model, B+C by model - each reading and
appending to one decision record, so no step re-interprets a sentence another already
realised. Ownership is enforced here: an agent belongs to at most one step.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import attrs
import yaml

from ..edgecase.materialize import GENERATED_MARKER
from .agents import TypeFiles
from .lexicon import match as lexicon_match
from .llm import Oracle, Unavailable
from .prompts import Prompt
from .spec import ADecision, Failure, Record, SchemaError, parse_b, parse_c, parse_d
from .step_a import StepError
from .step_a import apply as apply_a
from .step_bc import apply_b, apply_c
from .step_bc import decide as decide_bc
from .step_d import apply as apply_d
from .step_d import decide as decide_d
from .world import WorldContext, WorldError

#: Bases a Type A case may run on (§7). `busy` runs everywhere; the others need a crowd.
A_ON_EMPTY = frozenset({"busy"})
GENERATOR_VERSION = "promptgen-1"


def denied_on_empty(targets: tuple[str, ...]) -> list[str]:
    """The targets that cannot apply to an empty base: Type B selects from the crowd, and every
    Type A but `busy` moves it. Type C and D apply everywhere - their pedestrians are generated,
    not selected - and `busy` brings its own."""
    return [t for t in targets if t.startswith("B:") or (t.startswith("A:") and t.split(":", 1)[1] not in A_ON_EMPTY)]


@attrs.define
class Result:
    record: Record
    doc: dict[str, Any]
    files: TypeFiles
    effects: list[dict[str, Any]]
    scenario_id: str
    prompt: Prompt | None = None
    written: Path | None = None

    @property
    def block(self) -> dict[str, Any]:
        r = self.record
        block: dict[str, Any] = {
            "id": self.scenario_id, "base": r.base, "prompt_id": r.prompt_id, "steps": r.steps,
            "note": r.prompt,
            "effects": self.effects,
            "decisions": r.decisions(),
            "owned": {k: list(v) for k, v in r.owned.items()},
            **({"formation": dict(r.formation)} if r.formation else {}),
            "provenance": {**r.provenance, "generator": GENERATOR_VERSION, "status": r.status, "dedup_key": r.dedup_key()},
        }
        return block


def _fail(record: Record, step: str, exc: Exception) -> None:
    text = str(exc)
    kind = text.split(":", 1)[0] if ":" in text and text.split(":", 1)[0] in ("schema", "placement", "conflict", "applicability", "infeasible", "unavailable") else "schema" if isinstance(exc, SchemaError) else "placement" if isinstance(exc, WorldError) else "unavailable" if isinstance(exc, Unavailable) else "infeasible"
    record.failures.append(Failure(step=step, kind=kind, reason=text))


def _oracle_for(prompt: Prompt, key: str) -> Oracle:
    """The prompt's own slots as the model's answer."""
    if key == "D":
        return Oracle(prompt.oracle.get("D") or {"case": "none"})
    return Oracle({"B": prompt.oracle.get("B") or "none", "C": prompt.oracle.get("C") or "none"})


def pin_robot_route(doc: dict[str, Any], world: WorldContext, record: Record) -> None:
    """Write the robot's start and goal as the points the route was planned on.

    The base scenario names zones; at run time those resolve to random points of the zone,
    checked against walls only, so the robot's actual route differed from the one every
    placement, gate and interception here was designed against - and a goal drawn beside a
    desk left the robot "blocked" three metres short for a whole episode. The generator's
    endpoints are the zone centroids snapped into free space with the robot's clearance on a
    grid that has the furniture stamped in (`world.py`), so they are drivable and the runtime
    route is the planned one. The start faces along the route.
    """
    robots = doc.get("robots") or []
    if not robots or not isinstance(robots[0], dict) or world.grid is None or len(world.route) < 2:
        return
    (sx, sy), (gx, gy) = world.start, world.goal
    nx, ny = world.route[1]
    yaw = math.atan2(ny - sy, nx - sx)
    robots[0] = {**robots[0], "start": [round(sx, 2), round(sy, 2), round(yaw, 3)], "phases": [{"goto": [round(gx, 2), round(gy, 2)]}]}
    doc["robots"] = robots
    record.warnings.append(f"robot route pinned to the planned one: ({sx:.1f}, {sy:.1f}) -> ({gx:.1f}, {gy:.1f})")


def generate(
    prompt: Prompt,
    *,
    world: WorldContext,
    base_id: str,
    base_doc: Mapping[str, Any],
    base_dir: Path,
    seed: int = 0,
    cache_dir: Path | None = None,
    model: str | None = None,
    oracle: bool = False,
    a_case: str | None = None,
) -> Result:
    """`a_case` overrides Step 1: the case name to apply, or "" for none. Used by the ladder."""
    doc = copy.deepcopy(dict(base_doc))
    doc.pop("edge_case", None)
    files = TypeFiles(base_dir=Path(base_dir))
    record = Record(prompt_id=prompt.id, prompt=prompt.text, world=world.name, base=base_id, targets=tuple(prompt.targets))
    pin_robot_route(doc, world, record)
    effects: list[dict[str, Any]] = []
    owned: set[str] = set()
    intercepts = 0
    taken_fractions: list[float] = []
    populated = bool(doc.get("dynamic"))

    # An empty base is denied up front when nothing the prompt targets can apply to it: no
    # model is called and no realisation is attempted - the slot is a non-experiment, not a
    # failed one (371 noise failures + 100 infeasibles in the 08-31 oracle sweep before this).
    denied = denied_on_empty(prompt.targets)
    if not populated and denied and len(denied) == len(prompt.targets):
        record.failures.append(Failure(step="base", kind="denied", reason=f"empty base: {', '.join(denied)} need a crowd and nothing else is targeted"))
        return Result(record=record, doc=doc, files=files, effects=[], scenario_id=f"{base_id}__{prompt.id}", prompt=prompt)

    # Step 1 - A: the lexicon reads it exactly; in live mode a cheap model reads it by meaning
    # (typo/paraphrase tolerance) and extracts the `who` block the later steps share. The
    # lexicon stays the oracle: agreements keep the lexicon's decision, every disagreement is
    # recorded, and a silent lexicon defers to the model - that is the new capability.
    a = lexicon_match(prompt.text)
    who = ""
    if oracle is False and a_case is None:
        from . import step_a_llm  # noqa: PLC0415 - live mode only

        try:
            a_llm, who_block, notes = step_a_llm.decide(prompt.text, record, cache_dir=cache_dir)
            record.warnings.extend(notes)
            if who_block:
                record.provenance["who"] = who_block
                who = step_a_llm.who_line(who_block)
            if a is None and a_llm is not None:
                record.warnings.append(f"A read by the model (lexicon silent): {a_llm.case}")
                a = a_llm
            elif a is not None and a_llm is None:
                record.warnings.append(f"lexicon reads A:{a.case}, the model reads none; keeping the lexicon")
            elif a is not None and a_llm is not None and a_llm.case != a.case:
                record.warnings.append(f"A disagreement: lexicon {a.case}, model {a_llm.case}; taking the model")
                a = a_llm
        except (StepError, Unavailable) as exc:
            record.warnings.append(f"A model unavailable, lexicon only: {exc}")
    if a_case is not None:
        a = None if not a_case else (a if a is not None and a.case == a_case else ADecision(case=a_case))
    wanted_a = [t for t in prompt.targets if t.startswith("A:")]
    if a is None and wanted_a:
        record.failures.append(Failure(step="A", kind="schema", reason=f"prompt targets {wanted_a[0]} but the lexicon matched nothing"))
    if a is not None and wanted_a and f"A:{a.case}" != wanted_a[0]:
        record.warnings.append(f"lexicon matched {a.case}, prompt targets {wanted_a[0]}")
    if a is not None and not populated and a.case not in A_ON_EMPTY:
        record.failures.append(Failure(step="A", kind="denied", reason=f"{a.case} needs a crowd and the base is empty"))
        a = None
    if a is not None:
        try:
            new, names, notes = apply_a(doc, a, world, files, seed=seed)
            effects.extend(new)
            owned.update(names)
            record.owned["A"] = list(names)
            record.a = a
            record.warnings.extend(notes)
        except StepError as exc:
            _fail(record, "A", exc)

    # Step 2 - D
    try:
        d = decide_d(prompt.text, doc, world, record, cache_dir=cache_dir, model=model, oracle=_oracle_for(prompt, "D") if oracle else None, who=who)
    except (StepError, Unavailable) as exc:
        _fail(record, "D", exc)
        d = None
    if d is not None:
        try:
            new, names, notes, info = apply_d(doc, d, world, files, seed=seed, owned=owned, intercept_index=intercepts)
            intercepts += sum(1 for e in new if e["type"] == "intercept")
            effects.extend(new)
            owned.update(names)
            record.owned["D"] = list(names)
            record.d = d
            record.warnings.extend(notes)
            if info.get("formation"):
                record.formation = dict(info["formation"])
            if d.fraction is not None:
                taken_fractions.append(d.fraction)
        except StepError as exc:
            _fail(record, "D", exc)

    # Step 3 - B + C
    try:
        b, c = decide_bc(prompt.text, doc, world, record, taken_fractions=taken_fractions, cache_dir=cache_dir, model=model, oracle=_oracle_for(prompt, "BC") if oracle else None, who=who)
    except (StepError, Unavailable) as exc:
        _fail(record, "BC", exc)
        b = c = None
    if b is not None:
        # `populated` is the base's own crowd: on an empty base every agent present by now was
        # generated and owned by A or D, so B is structurally deniable, not circumstantially.
        if not populated:
            record.failures.append(Failure(step="B", kind="denied", reason="empty base: nobody to tune"))
        elif not [x for x in (doc.get("dynamic") or []) if isinstance(x, dict) and str(x.get("name")) not in owned]:
            record.failures.append(Failure(step="B", kind="applicability", reason="no unowned pedestrian to tune"))
        else:
            try:
                new, names, notes = apply_b(doc, b, world, files, owned=owned)
                effects.extend(new)
                owned.update(names)
                record.owned["B"] = list(names)
                record.b = b
                record.warnings.extend(notes)
            except StepError as exc:
                _fail(record, "B", exc)
    if c is not None:
        try:
            new, names, notes = apply_c(c, world, files, seed=seed, intercept_index=intercepts, taken_fractions=taken_fractions)
            intercepts += 1
            effects.extend(new)
            owned.update(names)
            record.owned["C"] = list(names)
            record.c = c
            record.warnings.extend(notes)
        except StepError as exc:
            _fail(record, "C", exc)

    # A rally over "everyone present at Step 1" must not reach agents later steps created.
    return Result(record=record, doc=doc, files=files, effects=effects, scenario_id=f"{base_id}__{prompt.id}", prompt=prompt)


def ladder(result: Result, *, world: WorldContext, base_doc: Mapping[str, Any], base_dir: Path, seed: int = 0) -> list[Result]:
    """The ablation ladder of a multi-step result (`new_plan_2.md` §0): the same decisions with
    the steps added one at a time, in the plan's order A, D, B, C. Each rung is realised from
    the record's own decisions, so no model call is made and every rung shares the full
    scenario's provenance. A one-step result has no ladder."""
    record = result.record
    order = [k for k in ("A", "D", "B", "C") if record.decisions().get(k)]
    if len(order) < 2:
        return []
    decided = record.decisions()
    rungs: list[Result] = []
    for depth in range(1, len(order)):
        keep = set(order[:depth])
        oracle_doc = {k: (decided[k] if k in keep else None) for k in ("A", "D", "B", "C")}
        rung = generate(
            attrs.evolve(result.prompt, id=f"{result.prompt.id}_l{depth}", oracle={k: v for k, v in oracle_doc.items() if v}),
            world=world, base_id=record.base, base_doc=base_doc, base_dir=base_dir, seed=seed, oracle=True,
            a_case=decided["A"]["case"] if "A" in keep and decided.get("A") else "",
        )
        rung.record.provenance["ladder_of"] = result.scenario_id
        rung.record.provenance["ladder_rung"] = depth
        rungs.append(rung)
    return rungs


def _header(result: Result) -> str:
    r = result.record
    return "\n".join((
        f"# Prompt scenario: {result.scenario_id}",
        f"# prompt: {r.prompt}",
        f"# base: {r.base} · steps: {', '.join(r.steps) or 'none'} · status: {r.status}",
        "#",
        GENERATED_MARKER,
        "# The edge_case block lists the runtime effects; everything else is baked into the population.",
        "", "",
    ))


def write_result(result: Result, world_root: Path, *, keep_partial: bool = True) -> Path | None:
    """Write `scenarios/<base>__<prompt_id>/`. Returns None when the record is infeasible or denied."""
    if result.record.status in ("infeasible", "denied") or (result.record.status == "partial" and not keep_partial):
        return None
    out = Path(world_root) / "scenarios" / result.scenario_id
    out.mkdir(parents=True, exist_ok=True)
    doc = {k: v for k, v in result.doc.items() if k != "edge_case"}
    doc["edge_case"] = result.block
    (out / "scenario.yaml").write_text(_header(result) + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, allow_unicode=True))
    result.files.write(out)
    (out / "record.json").write_text(json.dumps(result.record.as_dict(), indent=1, sort_keys=True, default=str))
    result.written = out
    return out


def replay_decisions(raw: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Typed decisions from a stored record, for tests and review."""
    a = raw.get("A")
    return (
        ADecision(case=str(a["case"]), k=float(a.get("k", 2.0))) if a else None,
        parse_d(raw.get("D")), parse_b(raw.get("B")), parse_c(raw.get("C")),
    )
