"""Freeze a solved case into a portable scenario.

A Level C case does not exist as a file. It is solved at run time — A* over the occupancy
grid, an interception geometry against the robot's planned route, retried through a fallback
ladder — so shipping the *parameters* ships a recipe that re-solves, and re-solves differently:
a recorded case requested 180.0 deg head-on and achieved 195.0 after 27 retries. Two labs
running "the same" case would not be running the same case.

Everything needed to freeze it is already in `cases.jsonl`, and the scenario format already
accepts coordinate literals, so this is a transcription rather than new machinery:

===========================  ==================================================
`cases.jsonl`                `scenario.yaml`
===========================  ==================================================
`robot_leg`                  `robots[0].start` / `.goal`
`spawn_pose` + `ped_goal`    `dynamic[edge].pose` (yaw derived) / `.waypoints`
`ped_speed`                  `agent.desired_velocity`
`waypoint_mode`              `waypoint_mode`
`inject_model`               `model`
`inject_type` / `derived_agent_type_path`  a copied-in agent type, referenced `./name.yaml`
===========================  ==================================================

The base population is copied **verbatim** from the source scenario, so the entry is
self-contained: a consumer needs the world and nothing else. That copy is also where the one
portability caveat lives - see `bake_case`.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .world_hash import HASH_LEN, world_hash

#: Injected agents are named `edge_0..edge_N` by the mode; the baked entry keeps the name so a
#: scenario and the case it came from can be read side by side.
INJECT_PREFIX = "edge_"

#: Characters of `scenario_uid` appended to a directory name. Two solves of the same cell -
#: different seeds, or the same seed after a generator change - produce different geometry and
#: are different scenarios. Without a content discriminator in the name they overwrite each
#: other silently, which is the one failure a content-addressed identity must not permit.
UID_SUFFIX_LEN = 8


def scenario_uid(scenario_yaml: str, agent_types: Mapping[str, str], world_hash_value: str) -> str:
    """Content hash of the baked artifact: the scenario, its agent types, and the world.

    This is the primary identity of a suite entry. It changes exactly when the artifact
    changes, which is the correct trigger for re-certification, and never when it is merely
    renamed or moved.
    """
    # The *document*, not the file text. The header comment carries provenance - which run and
    # episode produced this solve - and that must not change the identity: re-baking the same
    # geometry from a different run is the same scenario, and a content-addressed id that said
    # otherwise would force a pointless re-certification.
    doc = yaml.safe_load(scenario_yaml) or {}
    h = hashlib.sha256()
    h.update(world_hash_value.encode())
    h.update(b"\0")
    h.update(yaml.safe_dump(doc, sort_keys=True, default_flow_style=False).encode())
    for name in sorted(agent_types):
        h.update(b"\0")
        h.update(name.encode())
        h.update(b"\0")
        h.update(agent_types[name].encode())
    return h.hexdigest()[:HASH_LEN]


class BakeError(Exception):
    """A case cannot be frozen into a scenario."""


def _xy(value: object, where: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise BakeError(f"{where}: expected [x, y], got {value!r}")
    return float(value[0]), float(value[1])


def _yaw(frm: tuple[float, float], to: tuple[float, float]) -> float:
    """Heading from one point to another.

    `spawn_pose` carries no orientation, but a pedestrian spawned facing the wrong way walks a
    different first second than the one that was solved. Facing its own goal is the only
    defensible reconstruction.
    """
    return round(math.atan2(to[1] - frm[1], to[0] - frm[0]), 4)


def is_bakeable(case: dict[str, Any]) -> bool:
    """Whether a case row carries a complete solved geometry.

    Aborted cases, baselines and `place: scenario` cases have no interception to freeze.
    """
    return (
        case.get("status") == "ok"
        and bool(case.get("spawn_pose"))
        and bool(case.get("ped_goal"))
        and bool(case.get("robot_leg"))
    )


def scenario_id(case: dict[str, Any]) -> str:
    """Readable directory name: what the case asked for, and where."""
    world = case.get("world") or "unknown"
    angle = case.get("approach_angle_requested")
    enc = case.get("encounter_at_requested")
    knob = case.get("knob") or ""
    parts = ["c"]
    if angle is not None:
        parts.append(f"a{int(round(float(angle))):03d}")
    if enc is not None:
        parts.append(f"e{int(round(float(enc) * 100)):02d}")
    if knob:
        parts.append(knob)
    return f"{'_'.join(parts)}__{world}"


def _agent_type_ref(case: dict[str, Any], out_dir: Path, source_dir: Path) -> str:
    """Resolve the injected agent's type, copying a file-backed one into the entry.

    A derived type (one the mode generated for a knob) takes precedence over the parent type:
    it is the thing that was actually run.
    """
    for key in ("derived_agent_type_path", "inject_type"):
        raw = str(case.get(key) or "")
        if not raw:
            continue
        if not raw.endswith((".yaml", ".yml")):
            # A builtin name like `adult`; nothing to copy.
            return raw
        src = Path(raw)
        if not src.is_file():
            # Recorded as an absolute container path; try the source scenario dir.
            src = source_dir / Path(raw).name
        if not src.is_file():
            raise BakeError(f"agent type {raw!r} not found (also tried {src})")
        shutil.copy2(src, out_dir / src.name)
        return f"./{src.name}"
    return "adult"


def _base_population(source_scenario: Path) -> list[dict[str, Any]]:
    """The source scenario's own agents, copied verbatim.

    **The portability caveat.** These entries are copied as written. If the source scenario
    specifies positions as zone references rather than literals, the baked entry inherits them
    and is no more portable than its source - it will re-resolve them, and on worlds where zone
    resolution is what fails it will not run at all. `bake_case` reports this rather than
    silently producing an entry that only appears portable.
    """
    doc = yaml.safe_load(source_scenario.read_text()) or {}
    return list(doc.get("dynamic") or [])


def _has_zone_refs(agents: list[dict[str, Any]]) -> list[str]:
    """Names of copied agents whose position is a zone reference rather than a literal."""
    named: list[str] = []
    for a in agents:
        pose = a.get("pose")
        if isinstance(pose, str):
            named.append(str(a.get("name") or "?"))
    return named


def bake_case(
    case: dict[str, Any],
    *,
    world_root: Path,
    source_scenario: Path,
    out_root: Path,
    suite_version: str = "0.1.0",
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write one baked scenario directory. Returns the directory."""
    if not is_bakeable(case):
        raise BakeError(f"case is not bakeable (status={case.get('status')!r}, missing solved geometry)")

    # Written to a staging directory first: the final name embeds the content hash of what is
    # written, which cannot be known until it exists.
    out_dir = out_root / f".staging-{case.get('run_seed', 'x')}-{case.get('episode_id', 0)}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    leg = case["robot_leg"]
    if not isinstance(leg, (list, tuple)) or len(leg) < 2:
        raise BakeError(f"robot_leg: expected two points, got {leg!r}")
    r_start, r_goal = _xy(leg[0], "robot_leg[0]"), _xy(leg[1], "robot_leg[1]")

    spawn = _xy(case["spawn_pose"], "spawn_pose")
    goal = _xy(case["ped_goal"], "ped_goal")

    agent_ref = _agent_type_ref(case, out_dir, source_scenario.parent)
    base = _base_population(source_scenario)
    inherited_zone_refs = _has_zone_refs(base)

    speed = case.get("ped_speed")
    injected: dict[str, Any] = {
        "name": f"{INJECT_PREFIX}0",
        "model": case.get("inject_model") or "female_adult_business_02",
        "pose": [round(spawn[0], 4), round(spawn[1], 4), _yaw(spawn, goal)],
        "agent": {"agent_type": agent_ref, **({"desired_velocity": round(float(speed), 3)} if speed else {})},
        # Three elements, not two. Every shipped scenario writes `[x, y, theta]`, and an agent
        # whose waypoint has only two is dropped **silently** at load: the scenario comes up
        # with every other agent present and no warning, which reads as "the injection did
        # nothing" rather than "the entry is malformed". Cost a full replay to find.
        "waypoints": [[round(goal[0], 4), round(goal[1], 4), _yaw(spawn, goal)]],
    }
    if case.get("waypoint_mode"):
        injected["waypoint_mode"] = case["waypoint_mode"]

    doc = {
        "robots": [{
            "start": [round(r_start[0], 4), round(r_start[1], 4), 0.0],
            "goal": [round(r_goal[0], 4), round(r_goal[1], 4), 0.0],
        }],
        "dynamic": [*base, injected],
    }

    header = _header(case, suite_version)
    (out_dir / "scenario.yaml").write_text(header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))

    agent_types = {p.name: p.read_text() for p in sorted(out_dir.glob("*.yaml")) if p.name != "scenario.yaml"}
    whash = world_hash(world_root)
    uid = scenario_uid((out_dir / "scenario.yaml").read_text(), agent_types, whash)

    meta = _meta(
        case,
        scenario_uid_value=uid,
        world_hash_value=whash,
        world_root=world_root,
        source_scenario=source_scenario,
        suite_version=suite_version,
        agent_ref=agent_ref,
        inherited_zone_refs=inherited_zone_refs,
        provenance=provenance or {},
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    (out_dir / "replay.sh").write_text(_replay(case, meta))
    (out_dir / "replay.sh").chmod(0o755)

    final = out_root / f"{scenario_id(case)}__{uid[:UID_SUFFIX_LEN]}"
    if final.exists():
        shutil.rmtree(final)
    out_dir.rename(final)
    # `replay.sh` and `meta.json` name the installed directory, which is only known now.
    (final / "meta.json").write_text(json.dumps({**meta, "scenario_id": final.name}, indent=2, sort_keys=True) + "\n")
    (final / "replay.sh").write_text(_replay(case, {**meta, "scenario_id": final.name}))
    (final / "replay.sh").chmod(0o755)
    return final


def _header(case: dict[str, Any], suite_version: str) -> str:
    req_a, ach_a = case.get("approach_angle_requested"), case.get("approach_angle_achieved")
    req_e, ach_e = case.get("encounter_at_requested"), case.get("encounter_at_achieved")
    retries = case.get("geometry_retries")
    lines = [
        f"# AMBUSH scenario suite v{suite_version} - baked, do not hand-edit",
        f"# source: run {case.get('run_seed')} episode {case.get('episode_id')} seed {case.get('seed')}",
    ]
    # Requested-vs-achieved is the whole reason this file exists rather than a parameter set.
    if req_a is not None:
        lines.append(f"# approach angle: requested {req_a}, achieved {ach_a}" + (f" after {retries} retries" if retries else ""))
    if req_e is not None:
        lines.append(f"# encounter_at:   requested {req_e}, achieved {ach_e}")
    if case.get("designed_ttc") is not None:
        lines.append(f"# designed_ttc:   {case['designed_ttc']:.3f} s (an input, not a result)")
    return "\n".join(lines) + "\n\n"


def _meta(
    case: dict[str, Any],
    *,
    scenario_uid_value: str,
    world_hash_value: str,
    world_root: Path,
    source_scenario: Path,
    suite_version: str,
    agent_ref: str,
    inherited_zone_refs: list[str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_version": suite_version,
        # Primary identity: content, not name. See `scenario_uid`.
        "scenario_uid": scenario_uid_value,
        "scenario_id": scenario_id(case),
        "world": case.get("world"),
        "world_hash": world_hash_value,
        "base_scenario": source_scenario.parent.name,
        "agent_type": agent_ref,
        "construction": {
            "level": "C",
            "knob": case.get("knob") or None,
            "knob_level": case.get("level"),
            "place": case.get("place"),
            "approach_angle_requested": case.get("approach_angle_requested"),
            "approach_angle_achieved": case.get("approach_angle_achieved"),
            "encounter_at_requested": case.get("encounter_at_requested"),
            "encounter_at_achieved": case.get("encounter_at_achieved"),
            "geometry_retries": case.get("geometry_retries"),
            "designed_ttc": case.get("designed_ttc"),
            "encounter_point": case.get("encounter_point"),
        },
        "realized_from": {
            "run_seed": case.get("run_seed"),
            "episode_id": case.get("episode_id"),
            "seed": case.get("seed"),
            "created_at": case.get("created_at"),
            **provenance,
        },
        # Empty is the portable case. A non-empty list means the copied base population still
        # carries zone references and the entry is only as portable as its source.
        "inherited_zone_refs": inherited_zone_refs,
        "admission": None,
    }


def _replay(case: dict[str, Any], meta: dict[str, Any]) -> str:
    return f"""#!/usr/bin/env bash
# Reproduce this scenario. Needs Arena and the world; nothing from arena_benchmark.
# Install first: copy this directory into
#   worlds/{meta['world']}/scenarios/{meta['scenario_id']}/
set -euo pipefail
arena launch \\
    sim:=gazebo world:={meta['world']} robot:=jackal human:=arena headless:=true \\
    task.obstacles:=scenario task.scenario:={meta['scenario_id']} \\
    "$@"
"""
