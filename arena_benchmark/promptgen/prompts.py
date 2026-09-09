"""Prompt sets, generated from the taxonomy and frozen (`new_plan_2.md` §1.1, §5.5, §8).

Every prompt is built from case templates with slots, so it carries `targets:` (the steps it
exercises) and an `oracle:` (the decisions its slots encode). The oracle is what `--oracle`
runs realise without a model, and what a model's answer is compared against in review.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import attrs
import yaml

from .spec import A_CASES

#: Per-scenario depth distribution (§5.5).
DEPTHS: tuple[tuple[int, float], ...] = ((1, 0.40), (2, 0.35), (3, 0.20), (4, 0.05))

WHO = ("an old man", "a child", "a nurse", "a visitor", "an absent-minded man", "a doctor", "a man in a hurry")
WHO_PROFILE = {"an old man": "elder", "a child": "child", "a nurse": "nurse", "a visitor": "adult", "an absent-minded man": "distracted", "a doctor": "doctor", "a man in a hurry": "hurried"}
GROUP = ("two nurses", "three colleagues", "a doctor and two visitors", "four people")
GROUP_MEMBERS = {"two nurses": ["nurse", "nurse"], "three colleagues": ["adult", "adult", "adult"], "a doctor and two visitors": ["doctor", "adult", "adult"], "four people": ["adult", "adult", "adult", "adult"]}
#: Only the `blocking` variant takes more than four members (spec allows 8): a wall for a wide
#: passage. The realiser re-sizes the line to the passage anyway; the wide texts make the
#: *described* scenario match wide corridors and exercise the 5-8 member range of the schema.
GROUP_WIDE = ("five colleagues", "six people")
GROUP_WIDE_MEMBERS = {"five colleagues": ["adult"] * 5, "six people": ["adult"] * 6}
PLACE_ROUTE = (("right in the robot's way", 0.5), ("just ahead of the robot", 0.3), ("further down the robot's route", 0.7), ("halfway along the robot's route", 0.5),
               ("right at the start of the robot's route", 0.15), ("just before the robot's goal", 0.85))
WHEN_D = (
    ("from the start", {"at": 0}), ("as the robot arrives", {"robot_within": 3.0}), ("after a while", {"at": 10}),
    ("twenty seconds in", {"at": 20}), ("after half a minute", {"at": 30}), ("once the robot is halfway", {"route_fraction": 0.5}),
    ("when the robot comes within a few metres", {"robot_within": 4.0}), ("when the robot is right there", {"robot_within": 1.5}),
)
#: Time constraints a Type A sentence may carry (the lexicon reads them back through `PHRASES`).
#: None = the case's own default onset, and it is the commonest, as in real prompts.
#: Everything is a fraction of the 60 s episode budget (`route.MAX_EPISODE_S`): "a minute in"
#: was dropped 2026-09-01 - an onset at the budget's edge is an edge case nobody witnesses.
WHEN_A = (None, None, "after a while", "twenty seconds in", "after half a minute", "as soon as the robot sets off", "once the robot is halfway")
#: Time constraints for a Type B sentence (its `when`). Capped at 40 s (2/3 of the budget).
WHEN_B = ((None, None), ("from the start", {"at": 0}), ("after a while", {"at": 10}), ("twenty seconds in", {"at": 20}), ("after half a minute", {"at": 30}), ("once the robot is halfway", {"route_fraction": 0.5}), ("forty seconds in", {"at": 40}))
#: Durations a Type C sentence may carry, seconds. "for a long time" means half the episode
#: now, not a full minute of a budget that only has one.
DUR_C = (("for a moment", 4), ("for a few seconds", 6), ("for a while", 12), ("for a good half minute", 30), ("for a long time", 30))

A_TEXT = {
    "blackout": ("The lights go out and everyone slows down, feeling their way.", "A power cut leaves the whole floor in the dark."),
    "evacuation": ("The fire alarm goes off and everyone heads for the exit.", "An evacuation is called; everybody makes for the exit."),
    "stampede": ("The break is over and the whole crowd rushes back to work all at once, pushing through.", "A shift change: everyone surges back to their posts, nobody yields."),
    # "busy" must not mix an intensifier with a quantity ("very busy ... twice as many"
    # realised k=3 while the text promised 2x): one k word per sentence.
    "busy": ("It is visiting hours and the place is packed.", "The floor is busy today, twice as many people as usual.", "The floor is slightly busier than usual."),
}
B_TEMPLATES = (
    # Profile selects only for profiles a base guarantees (`populate` guarantees an elder);
    # "The nurses are rushing" matched nobody in 48 realised scenarios and was retired.
    ("Half of the people here are visually impaired and cannot see the robot until it is close.", {"case": "intrinsic", "select": {"amount": "half"}, "params": {"vision_range": "low", "vision_fov": "low"}}),
    ("The elderly people are slow to react and do not get out of the way.", {"case": "intrinsic", "select": {"profile": "elder"}, "params": {"reaction": "high", "repulsion": "low"}}),
    ("A few people keep changing their minds about where they are going.", {"case": "erratic_route", "select": {"n": 3}}),
    ("Nobody is looking where they are walking.", {"case": "intrinsic", "select": {"amount": "all"}, "params": {"vision_fov": "low"}}),
    ("Half the people here are rushing.", {"case": "intrinsic", "select": {"amount": "half"}, "params": {"speed": "high"}}),
    ("Some people wander about with no settled destination.", {"case": "erratic_route", "select": {"amount": "half"}}),
)
C_TEMPLATES = (
    ("{who} walks straight at the robot {place} and keeps going.", {"case": "approach", "angle": 180, "after": "continue"}),
    ("{who} crosses the robot's path from the right, {place}.", {"case": "approach", "angle": 90, "after": "continue"}),
    ("{who} comes up fast from behind and overtakes the robot {place}.", {"case": "approach", "angle": 0, "speed": "high", "after": "continue"}),
    # stop_facing needs motion in the sentence or a model reads it as blocking/stand.
    ("{who} walks up to the robot {place}, stops right in front of it and stares at it for a while.", {"case": "approach", "angle": 180, "after": "stop_facing", "duration": 8}),
    ("{who} walks up to the robot {place} and then follows it around.", {"case": "approach", "angle": 180, "after": "follow", "duration": 15}),
    ("{who} stands still {place}, blocking it.", {"case": "blocking", "mode": "stand", "duration": 20}),
    ("{who} shuffles along slowly {place}, right in front of the robot.", {"case": "blocking", "mode": "lead"}),
)
D_TEMPLATES = (
    ("{group} stand talking {place} and will not stop.", {"case": "interaction_unbroken", "variant": "standing"}),
    ("{group} walk in from different rooms, meet {place} and talk forever.", {"case": "interaction_unbroken", "variant": "meet"}),
    ("{group} stand side by side across the corridor {place}, chatting, and do not move aside.", {"case": "interaction_unbroken", "variant": "blocking"}),
    ("{wide_group} stand shoulder to shoulder across the corridor {place} and do not move aside.", {"case": "interaction_unbroken", "variant": "blocking"}),
    ("{group} stand in a circle {place}; they scatter {when_d}.", {"case": "formation_break", "variant": "circle", "break": "scatter"}),
    ("{group} are queuing {place} and the queue breaks up {when_d}, everyone walking off.", {"case": "formation_break", "variant": "line", "break": "disperse"}),
    ("{who} walks towards the robot {place}, wants to keep going and refuses to give way.", {"case": "stubborn"}),
)


@attrs.frozen
class Prompt:
    id: str
    text: str
    targets: tuple[str, ...]
    oracle: dict[str, Any]
    scope: str = "universal"
    world: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "scope": self.scope, "world": self.world, "targets": list(self.targets), "text": self.text, "oracle": self.oracle}


def _fill(template: str, rng: random.Random, zones: Sequence[str], *, avoid: Sequence[float] = (), allow_zone: bool = True) -> tuple[str, dict[str, Any]]:
    """`allow_zone=False` for slots whose schema has no zone placement (Type C, D:stubborn) -
    a zone-located sentence there is unrealisable and its oracle silently lied at 0.5 before."""
    slots: dict[str, Any] = {}
    text = template
    if "{who}" in text:
        who = rng.choice(WHO)
        slots["profile"] = WHO_PROFILE[who]
        text = text.replace("{who}", who[0].upper() + who[1:] if text.startswith("{who}") else who)
    if "{group}" in text:
        group = rng.choice(GROUP)
        slots["members"] = GROUP_MEMBERS[group]
        text = text.replace("{group}", group[0].upper() + group[1:] if text.startswith("{group}") else group)
    if "{wide_group}" in text:
        group = rng.choice(GROUP_WIDE)
        slots["members"] = GROUP_WIDE_MEMBERS[group]
        text = text.replace("{wide_group}", group[0].upper() + group[1:] if text.startswith("{wide_group}") else group)
    if "{place}" in text:
        if allow_zone and zones and rng.random() < 0.5:
            zone = rng.choice(list(zones))
            slots["place"] = {"zone": zone}
            text = text.replace("{place}", f"in the {zone.replace('_', ' ')}")
        else:
            options = [pf for pf in PLACE_ROUTE if all(abs(pf[1] - f) >= 0.15 for f in avoid)] or list(PLACE_ROUTE)
            phrase, fraction = rng.choice(options)
            slots["place"] = {"route_fraction": fraction}
            text = text.replace("{place}", phrase)
    if "{when_d}" in text:
        phrase, when = rng.choice(WHEN_D)
        slots["when"] = when
        text = text.replace("{when_d}", phrase)
    return text, slots


def _c_oracle(slots: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    out = {"profile": slots.get("profile", "adult"), "route_fraction": (slots.get("place") or {}).get("route_fraction", 0.5), **base}
    return out


def _d_oracle(slots: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    if base["case"] == "stubborn":
        out.update({"profile": slots.get("profile", "adult"), "route_fraction": (slots.get("place") or {}).get("route_fraction", 0.5)})
        return out
    out["members"] = slots.get("members", ["adult", "adult"])
    out["place"] = slots.get("place", {"route_fraction": 0.5})
    if "when" in slots:
        out["when"] = slots["when"]
    return out


def generate(n: int, *, seed: int = 0, scope: str = "universal", world: str = "", zones: Sequence[str] = ()) -> list[Prompt]:
    """`n` prompts balanced over the depth distribution and the cases."""
    from ..scenario.populate import for_people  # noqa: PLC0415 - avoid a module cycle

    zones = for_people(zones)  # never place a prompt in a wc or a cupboard
    rng = random.Random(f"prompts:{scope}:{world}:{seed}")
    out: list[Prompt] = []
    a_cycle = list(A_CASES)
    for i in range(n):
        depth = rng.choices([d for d, _ in DEPTHS], weights=[w for _, w in DEPTHS])[0]
        steps = rng.sample(["A", "B", "C", "D"], depth)
        steps.sort(key="ADBC".index)
        sentences: list[str] = []
        targets: list[str] = []
        oracle: dict[str, Any] = {}
        for step in steps:
            if step == "A":
                case = a_cycle[i % len(a_cycle)]
                sentence = rng.choice(A_TEXT[case])
                when_a = rng.choice(WHEN_A) if case != "busy" else None
                if when_a:
                    sentence = f"{sentence[:-1].rstrip()}, {when_a}."
                sentences.append(sentence)
                targets.append(f"A:{case}")
                oracle["A"] = {"case": case}
            elif step == "B":
                text, base = rng.choice(B_TEMPLATES)
                phrase, when_b = rng.choice(WHEN_B)
                if phrase:
                    text = f"{text[:-1].rstrip()}, {phrase}."
                    base = {**base, "when": when_b}
                sentences.append(text)
                targets.append(f"B:{base['case']}")
                oracle["B"] = base
            elif step == "C":
                template, base = rng.choice(C_TEMPLATES)
                taken = [f for f in ((oracle.get("D") or {}).get("place") or {}, {"route_fraction": (oracle.get("D") or {}).get("route_fraction")}) for f in [f.get("route_fraction")] if f is not None]
                if "duration" in base:
                    phrase, seconds = rng.choice(DUR_C)
                    template = template.replace(" for a while", f" {phrase}") if " for a while" in template else f"{template[:-1].rstrip()} {phrase}."
                    base = {**base, "duration": seconds}
                text, slots = _fill(template, rng, zones, avoid=taken, allow_zone=False)
                sentences.append(text)
                targets.append(f"C:{base['case']}")
                oracle["C"] = _c_oracle(slots, base)
            else:
                template, base = rng.choice(D_TEMPLATES)
                text, slots = _fill(template, rng, zones, allow_zone=base["case"] != "stubborn")
                sentences.append(text)
                targets.append(f"D:{base['case']}")
                oracle["D"] = _d_oracle(slots, base)
        prefix = "u" if scope == "universal" else world
        out.append(Prompt(id=f"{prefix}_{i + 1:03d}", text=" ".join(sentences), targets=tuple(targets), oracle=oracle, scope=scope, world=world))
    return out


CASES_BY_STEP: dict[str, tuple[str, ...]] = {
    "A": tuple(A_CASES),
    "B": tuple(sorted({b["case"] for _t, b in B_TEMPLATES})),
    "C": tuple(sorted({b["case"] for _t, b in C_TEMPLATES})),
    "D": tuple(sorted({b["case"] for _t, b in D_TEMPLATES})),
}
#: Two prompts whose token trigrams overlap this much read as the same prompt.
NEAR_DUPLICATE = 0.6


def _grams(text: str) -> set[tuple[str, ...]]:
    toks = re.findall(r"[a-z']+", text.lower())
    return {tuple(toks[i:i + 3]) for i in range(len(toks) - 2)} if len(toks) >= 3 else {tuple(toks)}


def _similar(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def generate_balanced(n: int, *, seed: int = 0, scope: str = "universal", world: str = "", zones: Sequence[str] = (),
                      avoid: Sequence[Prompt] = (), start: int = 1, prefix: str | None = None) -> list[Prompt]:
    """`n` prompts that fill the taxonomy evenly and read differently from one another.

    Unlike `generate` (which samples steps and cases at random and is kept byte-stable for the
    frozen sets), every step and case is the least-covered one so far - counting the prompts in
    `avoid`, so a supplement set closes the gaps of the set it extends - and a text whose trigrams
    overlap an earlier prompt's by `NEAR_DUPLICATE` or more is re-drawn. Ids continue from `start`.
    """
    from ..scenario.populate import for_people  # noqa: PLC0415 - avoid a module cycle

    zones = for_people(zones)  # never place a prompt in a wc or a cupboard
    rng = random.Random(f"prompts-balanced:{scope}:{world}:{seed}")
    counts: dict[str, int] = {f"{st}:{c}": 0 for st, cs in CASES_BY_STEP.items() for c in cs}
    for p in avoid:
        for t in p.targets:
            counts[t] = counts.get(t, 0) + 1
    grams = [_grams(p.text) for p in avoid]
    prefix = prefix or ("u" if scope == "universal" else world)
    out: list[Prompt] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        depth = rng.choices([d for d, _ in DEPTHS], weights=[w for _, w in DEPTHS])[0]
        def least(step: str) -> int:
            return min(counts[f"{step}:{c}"] for c in CASES_BY_STEP[step])
        steps = sorted("ABCD", key=lambda st: (least(st), rng.random()))[:depth]
        steps.sort(key="ADBC".index)
        sentences: list[str] = []
        targets: list[str] = []
        oracle: dict[str, Any] = {}
        for step in steps:
            case = min(CASES_BY_STEP[step], key=lambda c: (counts[f"{step}:{c}"], rng.random()))
            if step == "A":
                sentence = rng.choice(A_TEXT[case])
                when_a = rng.choice(WHEN_A) if case != "busy" else None
                if when_a:
                    sentence = f"{sentence[:-1].rstrip()}, {when_a}."
                sentences.append(sentence)
                oracle["A"] = {"case": case}
            elif step == "B":
                text, base = rng.choice([t for t in B_TEMPLATES if t[1]["case"] == case])
                phrase, when_b = rng.choice(WHEN_B)
                if phrase:
                    text = f"{text[:-1].rstrip()}, {phrase}."
                    base = {**base, "when": when_b}
                sentences.append(text)
                oracle["B"] = base
            elif step == "C":
                template, base = rng.choice([t for t in C_TEMPLATES if t[1]["case"] == case])
                taken = [f for f in ((oracle.get("D") or {}).get("place") or {}, {"route_fraction": (oracle.get("D") or {}).get("route_fraction")}) for f in [f.get("route_fraction")] if f is not None]
                if "duration" in base:
                    phrase, seconds = rng.choice(DUR_C)
                    template = template.replace(" for a while", f" {phrase}") if " for a while" in template else f"{template[:-1].rstrip()} {phrase}."
                    base = {**base, "duration": seconds}
                text, slots = _fill(template, rng, zones, avoid=taken, allow_zone=False)
                sentences.append(text)
                oracle["C"] = _c_oracle(slots, base)
            else:
                template, base = rng.choice([t for t in D_TEMPLATES if t[1]["case"] == case])
                text, slots = _fill(template, rng, zones, allow_zone=base["case"] != "stubborn")
                sentences.append(text)
                oracle["D"] = _d_oracle(slots, base)
            targets.append(f"{step}:{case}")
        text = " ".join(sentences)
        g = _grams(text)
        if attempts < 20 * n and any(_similar(g, h) >= NEAR_DUPLICATE for h in grams):
            continue
        for t in targets:
            counts[t] += 1
        grams.append(g)
        out.append(Prompt(id=f"{prefix}_{start + len(out):03d}", text=text, targets=tuple(targets), oracle=oracle, scope=scope, world=world))
    return out


def write(prompts: Sequence[Prompt], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump([p.as_dict() for p in prompts], sort_keys=False, default_flow_style=None, allow_unicode=True))
    return path


def read(path: Path) -> list[Prompt]:
    raw = yaml.safe_load(path.read_text()) or []
    return [Prompt(id=str(p["id"]), text=str(p["text"]), targets=tuple(p.get("targets") or ()), oracle=dict(p.get("oracle") or {}), scope=str(p.get("scope", "universal")), world=str(p.get("world", ""))) for p in raw]


def shipped_dir() -> Path:
    return Path(__file__).with_name("prompts")


def load_sets(world: str, *, extra: Sequence[Path] = ()) -> list[Prompt]:
    """The frozen universal set plus this world's own, plus any files given."""
    paths = [*sorted(shipped_dir().glob("universal*.yaml")), *(sorted(shipped_dir().glob(f"{world}*.yaml")) if world else []), *extra]
    out: list[Prompt] = []
    for p in paths:
        if p.is_file():
            out.extend(read(p))
    return out
