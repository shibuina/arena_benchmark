"""``arena_bench`` - the command line over the benchmark layers.

Only the verbs that are implemented appear here. A subcommand that printed a plausible
summary for work it had not done would be worse than no subcommand at all, which is the same
reason the task mode aborts rather than recording a silent no-op.

    arena_bench expand screen_v1 --out data/bench/screen
    arena_bench prompts --write && arena_bench prompts hospital_1 --write
    arena_bench promptgen hospital_1 --write
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .bake import bake_case, is_bakeable
from .orchestrate import shard, summarise
from .propose import (
    DEFAULT_MODEL,
    STORE,
    Unavailable,
    build_prompt,
    classify_tail,
    load_overrides,
    merge,
    unknown_names,
)
from .scenario import (
    MIN_EPISODE_S,
    MIN_ROUTE_CLEARANCE,
    PROFILES,
        ROBOT_SPEED_MPS,
    ref_safe_zones,
    robot_for,
    robot_zones,
    route_points,
    route_seconds,
    screen_zones,
    world_statics,
)
from .scenario import build as build_base
from .scenario.zones import zones_by_role
from .spec import Episode, Spec, SpecError, available, expand, load_spec, resolve


def _load(name: str) -> Spec:
    return load_spec(resolve(name))


def _write_manifest(episodes: Iterable[Episode], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.jsonl"
    with path.open("w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep.as_row(), sort_keys=True, default=str) + "\n")
    return path


def _cmd_expand(args: argparse.Namespace) -> int:
    spec = _load(args.spec)
    episodes = expand(spec)
    shards = shard(episodes)
    info = summarise(shards)

    print(f"spec     {spec.name}")
    print(f"worlds   {len(spec.worlds)}  cells {len(spec.cells)}  arms {len(spec.arms)}  seeds {spec.seed_count}")
    print(f"episodes {len(episodes)}")
    print(f"shards   {info['shards']}  ({len(episodes) // max(info['shards'], 1)} episodes each)")
    for family, arms in spec.families().items():
        names = ", ".join(f"{a.id}{'' if a.is_baseline else '*'}" for a in arms)
        print(f"  {family:16s} {names}")

    if args.out:
        path = _write_manifest(episodes, Path(args.out))
        print(f"wrote    {path}")
    return 0


def _cmd_bake(args: argparse.Namespace) -> int:
    """Freeze every solved case in a cases.jsonl into a portable scenario."""
    worlds = Path(args.worlds)
    out_root = Path(args.out)
    rows = [json.loads(l) for l in Path(args.cases).read_text().splitlines() if l.strip()]
    solved = [r for r in rows if is_bakeable(r)]
    print(f"cases    {len(rows)}  solved {len(solved)}")

    baked = skipped = 0
    for case in solved:
        world = case.get("world") or ""
        source = worlds / world / "scenarios" / args.base_scenario / "scenario.yaml"
        if not source.is_file():
            print(f"  skip {case.get('episode_id')}: no base scenario at {source}")
            skipped += 1
            continue
        try:
            out = bake_case(case, world_root=worlds / world, source_scenario=source, out_root=out_root)
        except Exception as exc:  # noqa: BLE001 - one bad case must not abandon the rest
            print(f"  skip {case.get('episode_id')}: {exc}")
            skipped += 1
            continue
        meta = json.loads((out / "meta.json").read_text())
        flag = "" if not meta["inherited_zone_refs"] else f"  (inherits {len(meta['inherited_zone_refs'])} zone refs)"
        print(f"  baked {out.name}{flag}")
        baked += 1

    print(f"wrote    {baked} scenario(s) to {out_root}" + (f", skipped {skipped}" if skipped else ""))
    return 0 if baked else 1


def _route_reachable(root: Path, zones: list, robot: dict) -> bool:
    """Whether the map (walls plus furniture) has a robot-wide path along the robot's route."""
    try:
        from .promptgen.world import Grid, WorldError, _endpoint
        from .scenario.occupancy import world_statics
        from .scenario.populate import route_points
        from .scenario.zone_screen import _load_grid, _stamp_statics

        pts = route_points(robot, zones)
        loaded = _load_grid(root)
        if loaded is None or len(pts) < 2:
            return True
        occupied, res, ox, oy = loaded
        occupied = occupied.copy()
        _stamp_statics(occupied, world_statics(root), res, ox, oy)
        grid = Grid(occupied=occupied, res=res, ox=ox, oy=oy)
        for a, b in zip(pts, pts[1:], strict=False):
            grid.plan(_endpoint(grid, a), _endpoint(grid, b))
        return True
    except WorldError:
        return False
    except Exception:  # noqa: BLE001 - reachability is a hint; the generator still validates the route
        return True


def _zones_connected(root: Path, zones: list, a_name: str, b_name: str) -> bool:
    """Whether the map (walls plus furniture) has a robot-wide path between the two zones' centroids."""
    try:
        from .promptgen.world import Grid, WorldError, _endpoint
        from .scenario.occupancy import world_statics
        from .scenario.zone_screen import _load_grid, _stamp_statics

        loaded = _load_grid(root)
        if loaded is None:
            return True  # nothing to check against
        occupied, res, ox, oy = loaded
        occupied = occupied.copy()
        _stamp_statics(occupied, world_statics(root), res, ox, oy)
        grid = Grid(occupied=occupied, res=res, ox=ox, oy=oy)

        def centroid(name: str) -> tuple[float, float]:
            z = next(z for z in zones if z["name"] == name)
            cs = z["corners"]
            return (sum(float(c["x"]) for c in cs) / len(cs), sum(float(c["y"]) for c in cs) / len(cs))

        grid.plan(_endpoint(grid, centroid(a_name)), _endpoint(grid, centroid(b_name)))
        return True
    except WorldError:
        return False
    except Exception:  # noqa: BLE001 - reachability is a hint; the populate step still validates the route
        return True


def _world_parts(worlds_root: Path, world: str) -> tuple[list, dict]:
    """A world's zones and a robot spec copied from one of its existing scenarios.

    The robot is never invented: every world ships a scenario with a working route, and
    deriving one from zone geometry would be a second, worse route planner.
    """
    import yaml as _yaml

    root = worlds_root / world
    level = next(iter(sorted(root.glob("*/world.yaml"))), None)
    if level is None:
        raise FileNotFoundError(f"{world}: no world.yaml")
    doc = _yaml.safe_load(level.read_text())
    zones = doc if isinstance(doc, list) else (doc.get("zones") or [])

    for cand in sorted((root / "scenarios").glob("*/scenario.yaml")):
        d = _yaml.safe_load(cand.read_text()) or {}
        robots = d.get("robots") or []
        if robots and isinstance(robots[0], dict) and robots[0].get("start"):
            robot = dict(robots[0])
            if _route_reachable(root, zones, robot):
                return zones, robot
            # a scenario copied from another world carries that world's coordinates
            # (acoustics_bend45_wide shipped arena_arena_032's route, 2026-08-30); fall back to zones
            print(f"  ! {world}: {cand.parent.name}'s robot route is not plannable on this map; ignoring it")

    # No scenario in this world declares a robot - several generated worlds ship only an empty
    # legacy `scenario.json`. Fall back to naming two zones as endpoints, which is what the
    # generated office worlds already do (`start: reception, goal: open_work_area`). That is
    # not route planning: the zone converter resolves each name to a free point and the robot's
    # planner does the rest. It is still a guess about intent, so it is announced, never silent.
    named = [z for z in zones if isinstance(z, dict) and z.get("name") and z.get("corners")]
    if len(named) >= 2:
        def centroid(z: dict[str, Any]) -> tuple[float, float]:
            cs = z["corners"]
            return (sum(float(c["x"]) for c in cs) / len(cs), sum(float(c["y"]) for c in cs) / len(cs))

        pairs = sorted(
            ((math.dist(centroid(a), centroid(b)), a["name"], b["name"]) for i, a in enumerate(named) for b in named[i + 1:]),
            reverse=True,
        )
        # the two most separated zones that the robot can actually travel between: the farthest pair
        # of a generated world can sit in disconnected rooms (arena_arena_032, _009, _065, _006, _007 -
        # "the robot cannot reach ..." on every base, 2026-08-30), so walk the pairs by distance and
        # keep the first one the map plans
        _, start, goal = pairs[0]
        for _d, a_name, b_name in pairs:
            if _zones_connected(root, named, a_name, b_name):
                start, goal = a_name, b_name
                break
        print(f"  ! {world}: no scenario declares a robot; using the two most separated reachable zones "
              f"as endpoints - start={start!r} goal={goal!r}")
        print("  ! override with --robot-start / --robot-goal if that is not the traverse you want")
        return zones, {"start": start, "goal": goal}

    raise FileNotFoundError(
        f"{world}: no existing scenario with a robot to copy, and fewer than two named zones "
        f"to fall back on. Author a scenario with a `robots:` block first."
    )


def _all_zones(worlds_root: Path, world: str) -> list:
    """Every zone a world declares, across all of its levels.

    `_world_parts` reads one level, which is right for generation - the robot lives on one
    floor. Classification is about the *vocabulary*, and a name that appears only on floor 3
    (`three_storied_residential`'s `3_study`) is invisible to a single-level read.
    """
    import yaml as _yaml

    zones: list = []
    seen: set[str] = set()
    for level in sorted((worlds_root / world).glob("*/world.yaml")):
        try:
            doc = _yaml.safe_load(level.read_text())
        except Exception:  # noqa: BLE001 - one unreadable level must not hide the others
            continue
        for zone in (doc if isinstance(doc, list) else (doc or {}).get("zones") or []):
            if isinstance(zone, dict) and zone.get("name") and str(zone["name"]) not in seen:
                seen.add(str(zone["name"]))
                zones.append(zone)
    return zones


def _cmd_zones(args: argparse.Namespace) -> int:
    worlds = Path(args.worlds)
    zones, _ = _world_parts(worlds, args.world)
    overrides = load_overrides()
    for role, names in zones_by_role(zones, overrides).items():
        if names:
            marked = ", ".join(f"{n}*" if n in overrides else n for n in names)
            print(f"{role.value:9s} {marked}")
    if any(n in overrides for names in zones_by_role(zones, overrides).values() for n in names):
        print("\n* from the reviewed mapping, not a substring rule")
    if unresolved := unknown_names(zones, overrides):
        print(f"\nunclassified: {', '.join(unresolved)}")
        print("  run `arena_bench classify` to have a model propose roles for these")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    """Ask a model for the zone roles no substring rule covers."""
    worlds = Path(args.worlds)
    # `.generated` is a scratch directory beside the worlds, not a world; a dotted name is
    # never one, and sweeping it asks the model about sixteen placeholder rooms nobody runs.
    names = sorted(args.world) if args.world else sorted(
        w for w in {p.parts[-3] for p in worlds.glob("*/*/world.yaml")} if not w.startswith(".")
    )
    overrides = load_overrides()

    pending: dict[str, tuple[list, list[str]]] = {}
    for world in dict.fromkeys(names):
        zones = _all_zones(worlds, world)
        if not zones:
            print(f"  ! {world}: no zones declared on any level")
            continue
        if todo := unknown_names(zones, overrides):
            pending[world] = (zones, todo)

    if not pending:
        print("every zone in every world is already classified by a rule or the reviewed mapping")
        return 0

    total = sum(len(todo) for _, todo in pending.values())
    print(f"{total} unclassified zone name(s) across {len(pending)} world(s)")
    if args.dry_run:
        for world, (zones, todo) in pending.items():
            print(f"\n### {world}  ({len(todo)})")
            print(build_prompt(zones, todo, world=world))
        print("(dry run - this is the prompt that would be sent; pass --write to ask and store)")
        return 0

    written = 0
    for world, (zones, todo) in pending.items():
        print(f"\n### {world}")
        try:
            proposals, complaints = classify_tail(zones, world=world, names=todo, model=args.model)
        except Unavailable as exc:
            print(f"  ! {exc}")
            return 1
        for proposal in proposals:
            mark = " " if proposal.confident else "?"
            print(f"  {mark} {proposal.zone:20s} {proposal.role.value:8s} {proposal.why}")
        for complaint in complaints:
            print(f"  ! {complaint}")
        landed, kept = merge(proposals)
        for note in kept:
            print(f"  = {note}")
        written += landed

    print(f"\n{written} entry(ies) written to {STORE}")
    print("review them there; set `source: human` on any you change and it will not be revisited")
    return 0


#: The one line a base's `note` says about it.
PROFILE_NOTES = {
    "normal": "working hours: the building's dwellers walk between rooms, pairs talk in rooms",
    "break": "break time: the same crowd in the break rooms",
    "night": "night: half the crowd, slower",
    "empty": "nobody - the no-population control",
}


def _cmd_populate(args: argparse.Namespace) -> int:
    worlds = Path(args.worlds)
    zones, robot = _world_parts(worlds, args.world)
    if getattr(args, "robot_start", None):
        robot = {"start": args.robot_start, "goal": args.robot_goal or robot.get("goal")}
        print(f"  robot endpoints overridden: start={robot['start']!r} goal={robot['goal']!r}")
    from .edgecase import prune_generated, write_base
    from .scenario.populate import basis_for

    overrides = load_overrides()
    if unresolved := unknown_names(zones, overrides):
        print(f"  ! {len(unresolved)} zone(s) no rule classifies: {', '.join(unresolved)}")
        print("  !   they are populated as neither work nor break; `arena_bench classify` proposes roles")
    basis = basis_for(zones, overrides)
    if basis != "workplace":
        print(f"  ! zone mapping: {basis} - this world has no work zones, so the active/retreat")
        print("  !   distinction is drawn between living areas and private rooms instead")

    # The screen decides which rooms may be *homes*: every point in them off the robot's route.
    # It is a property of the route's segments, and every base variant drives the same
    # segments, so it is computed once on the longest form the route takes. Furniture goes in
    # because the runtime resolver sees it too - it stamps the world's statics into its grid.
    statics = world_statics(worlds / args.world)
    longest = robot_for(robot, zones)
    route_all = route_points(longest, zones)
    verdicts = screen_zones(
        zones, world_root=worlds / args.world, route=route_all, statics=statics,
        min_route_clearance=MIN_ROUTE_CLEARANCE, robot_zones=robot_zones(robot),
    )
    homes = ref_safe_zones(verdicts)
    print(f"  homes: {len(homes)}/{len(verdicts)} zone(s) entirely clear of the route"
          + ("" if homes else " - falling back to every target zone but the robot's own"))
    for name, verdict in sorted(verdicts.items()):
        if not verdict.ref_safe and verdict.reason:
            print(f"    - {name}: {verdict.reason}")
    resolvable = {name: v.admissible for name, v in verdicts.items()}
    print(f"  route: {route_seconds(longest, zones):.0f} s at {ROBOT_SPEED_MPS} m/s "
          f"({len(longest['phases'])} leg(s); floor {MIN_EPISODE_S:.0f} s)")

    written: list[str] = []
    spec = longest
    for profile in PROFILES:
        doc = build_base(profile, zones=zones, robot=spec, seed=args.seed, homes=homes, overrides=overrides, world=args.world)
        agents = doc.get("dynamic") or []
        n = len(agents)
        if n == 0 and PROFILES[profile]["count"] > 0:
            print(f"  ! {profile}: no zone to populate - this base is measurable as `empty`,")
            print("  !   and every case needing a crowd will refuse it")
        talking = sum(1 for a in agents if "talker" in str(a["agent"].get("agent_type")))
        rooms = sorted({str(a["pose"]) for a in agents})
        line = f"  {profile:7s} {n:3d} agent(s): {talking} talking, {n - talking} walking, from {len(rooms)} room(s)"

        if args.write:
            from .edgecase.materialize import WouldOverwrite

            try:
                out = write_base(doc, world_root=worlds / args.world, profile=profile, force=args.force, note=PROFILE_NOTES[profile])
            except WouldOverwrite as exc:
                print(f"  {profile:7s} SKIPPED - {exc}")
                continue
            written.append(out.name)
        print(line)

    if args.write and args.prune:
        keep = [*written, *(d.name for d in (worlds / args.world / "scenarios").glob("*__*"))]
        for gone in prune_generated(worlds / args.world, keep):
            print(f"  - pruned stale generated base {gone}")
    if not args.write:
        print("(dry run - pass --write to materialise)")
    return 0


def _cmd_worlds(args: argparse.Namespace) -> int:
    """Per-world feature table, the spread of the set, and the farthest-point order to author/test in."""
    from .variance.world_features import format_table, usable, variance_report, world_table

    worlds = Path(args.worlds)
    errors: list[str] = []
    table = world_table(worlds, args.world, on_error=lambda name, exc: errors.append(f"{name}: {exc}"))
    if not args.all:
        table = {w: f for w, f in table.items() if usable(f)}
    if not table:
        print("no world measured" + (f" ({'; '.join(errors)})" if errors else ""))
        return 1
    report = variance_report(table)
    print(format_table(table, report))
    for e in errors:
        print(f"  ! {e}")
    if args.json:
        Path(args.json).write_text(json.dumps({"table": table, "report": report, "errors": errors}, indent=1))
    return 0


def _cmd_diversity(args: argparse.Namespace) -> int:
    """Diversity of the shipped prompt sets (plus any given) and of each world's realised scenarios."""
    from .promptgen import prompts as P
    from .variance.diversity import format_prompt_report, format_scenario_report, prompt_report, scenario_report

    worlds = Path(args.worlds)
    out: dict[str, Any] = {"prompts": {}, "scenarios": {}}
    sets = {p.stem: p for p in sorted(P.shipped_dir().glob("*.yaml"))}
    for extra in args.prompts or []:
        sets[Path(extra).stem] = Path(extra)
    for name, path in sets.items():
        r = prompt_report(P.read(path))
        out["prompts"][name] = r
        print(format_prompt_report(name, r))
    for world in args.world:
        r = scenario_report(worlds / world)
        out["scenarios"][world] = r
        print(format_scenario_report(r))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0


def _cmd_prompts(args: argparse.Namespace) -> int:
    """Generate a prompt set from the taxonomy and freeze it into the shipped prompt directory."""
    from .promptgen import prompts as P

    worlds = Path(args.worlds)
    out_dir = Path(args.out) if args.out else P.shipped_dir()
    scope = "world" if args.world else "universal"
    stem = args.world or "universal"
    names: list[str] = []
    if args.world:
        zones, _robot = _world_parts(worlds, args.world)
        names = sorted(str(z["name"]) for z in zones if isinstance(z, dict) and z.get("name"))
    if args.supplement:
        # a second set that closes the first one's gaps: least-covered cases first, no near-duplicate texts
        base_path = out_dir / f"{stem}.yaml"
        base = P.read(base_path) if base_path.is_file() else []
        prompts = P.generate_balanced(args.n or (10 if args.world else 40), seed=args.seed, scope=scope, world=args.world, zones=names,
                                      avoid=base, start=101 if args.world else 1, prefix=None if args.world else "v")
        path = out_dir / f"{stem}_2.yaml"
    elif args.balanced:
        prompts = P.generate_balanced(args.n or (20 if args.world else 40), seed=args.seed, scope=scope, world=args.world, zones=names)
        path = out_dir / f"{stem}.yaml"
    else:
        prompts = P.generate(args.n or (20 if args.world else 40), seed=args.seed, scope=scope, world=args.world, zones=names)
        path = out_dir / f"{stem}.yaml"
    from .promptgen.lexicon import match as lexicon_match

    misses = 0
    for prompt in prompts:
        wanted = [t for t in prompt.targets if t.startswith("A:")]
        got = lexicon_match(prompt.text)
        if wanted and (got is None or f"A:{got.case}" != wanted[0]):
            misses += 1
            print(f"  ! {prompt.id}: targets {wanted[0]}, lexicon {'matched nothing' if got is None else 'matched ' + got.case}")
    depth = {}
    for prompt in prompts:
        depth[len(prompt.targets)] = depth.get(len(prompt.targets), 0) + 1
    print(f"{len(prompts)} prompt(s); depth " + ", ".join(f"{k}:{v}" for k, v in sorted(depth.items())) + f"; lexicon round-trip misses: {misses}")
    if args.write:
        P.write(prompts, path)
        print(f"wrote {path}")
    else:
        for prompt in prompts[:8]:
            print(f"  {prompt.id} [{', '.join(prompt.targets)}] {prompt.text}")
        print("(dry run - pass --write to freeze)")
    return 1 if misses else 0


def _cmd_promptgen(args: argparse.Namespace) -> int:
    """prompt x base -> scenario directories for one world."""
    import yaml as _yaml

    from .promptgen import generate, write_result
    from .promptgen.prompts import load_sets
    from .promptgen.world import WorldContext, WorldError

    worlds = Path(args.worlds)
    root = worlds / args.world
    prompts = load_sets(args.world, extra=[Path(p) for p in (args.prompts or [])])
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        prompts = [p for p in prompts if p.id in wanted]
    if not prompts:
        print("no prompts: run `arena_bench prompts --write` (and `prompts <world> --write`) first")
        return 1
    bases = [b.strip() for b in args.bases.split(",") if b.strip()] if args.bases else list(PROFILES)
    overrides = load_overrides()
    cache = Path(args.cache) if args.cache else None
    seen: dict[str, str] = {}
    written = 0
    counts: dict[str, int] = {}
    for base_id in bases:
        base_path = root / "scenarios" / base_id / "scenario.yaml"
        if not base_path.is_file():
            print(f"  ! base {base_id}: not found - run `populate --write` first")
            continue
        base_doc = _yaml.safe_load(base_path.read_text()) or {}
        try:
            world = WorldContext.load(worlds, args.world, base_doc, overrides)
        except WorldError as exc:
            print(f"  ! {base_id}: {exc}")
            continue
        for prompt in prompts:
            try:
                result = generate(prompt, world=world, base_id=base_id, base_doc=base_doc, base_dir=base_path.parent,
                                  seed=args.seed, cache_dir=cache, model=args.model, oracle=args.oracle)
            except Exception as exc:  # noqa: BLE001 - one prompt must not stop the batch
                print(f"  ! {base_id}/{prompt.id}: {exc!r}")
                if args.verbose:
                    import traceback  # noqa: PLC0415

                    traceback.print_exc()
                counts["crashed"] = counts.get("crashed", 0) + 1
                continue
            status = result.record.status
            key = result.record.dedup_key()
            if status not in ("infeasible", "denied") and key in seen:
                status = f"duplicate of {seen[key]}"
            elif status not in ("infeasible", "denied"):
                seen[key] = result.scenario_id
            counts[status.split(" ")[0]] = counts.get(status.split(" ")[0], 0) + 1
            fails = "; ".join(f"{f.step}:{f.kind}" for f in result.record.failures)
            print(f"  {result.scenario_id:40s} {status:10s} [{', '.join(result.record.steps) or '-'}]" + (f"  failures: {fails}" if fails else ""))
            if args.verbose:
                for note in result.record.warnings:
                    print(f"      - {note}")
                for failure in result.record.failures:
                    print(f"      ! {failure.step}: {failure.reason}")
            if args.write and status in ("ok", "partial"):
                if write_result(result, root, keep_partial=not args.strict) is not None:
                    written += 1
            if args.ladder and status in ("ok", "partial"):
                from .promptgen.pipeline import ladder as _ladder

                for rung in _ladder(result, world=world, base_doc=base_doc, base_dir=base_path.parent, seed=args.seed):
                    print(f"    rung {rung.scenario_id:36s} {rung.record.status:10s} [{', '.join(rung.record.steps) or '-'}]")
                    if args.write and rung.record.status in ("ok", "partial") and write_result(rung, root, keep_partial=not args.strict) is not None:
                        written += 1
    print(f"{sum(counts.values())} generated: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + (f"; wrote {written}" if args.write else " (dry run - pass --write)"))
    if args.write:
        print("rebuild arena_simulation_setup before launching: new scenario directories are invisible to the runtime until then")
    return 0


def _cmd_commands(args: argparse.Namespace) -> int:
    """Write the robot x planner x scenario launch-command matrix for one world."""
    from .rollout.commands import planner_families, render, robot_families, scenarios, workspace_roots

    worlds = Path(args.worlds)  # main() has already resolved None to default_worlds_root()
    robots_root, planners_root = workspace_roots(worlds)
    for root, what in ((robots_root, "robots"), (planners_root, "planners")):
        if not root.is_dir():
            print(f"  ! no {what} directory at {root}")
            return 1
    cases, bases = scenarios(worlds / args.world)
    robots = robot_families(robots_root)
    if args.robots:
        wanted = [r.strip() for r in args.robots.split(",") if r.strip()]
        missing = sorted(set(wanted) - set(robots))
        if missing:
            print(f"  ! unknown robot(s) {missing}; known: {sorted(robots)}")
            return 1
        robots = {r: robots[r] for r in wanted}
    text = render(
        world=args.world,
        robots=robots,
        planners=planner_families(planners_root),
        cases=cases,
        bases=bases,
        generated_by=f"arena_bench commands {args.world}",
    )
    out = Path(args.out)
    out.write_text(text)
    print(f"  {out}: {text.count('arena launch ')} commands, {len(cases)} cases x {len(bases)} bases")
    return 0


def _cmd_specs(_: argparse.Namespace) -> int:
    for name in available():
        print(name)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arena_bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("expand", help="validate a spec and report what it would run")
    p.add_argument("spec", help="a shipped spec name, or a path to one")
    p.add_argument("--out", help="also write manifest.jsonl here")
    p.set_defaults(fn=_cmd_expand)

    p = sub.add_parser("bake", help="freeze solved cases into portable scenarios")
    p.add_argument("cases", help="path to a cases.jsonl")
    p.add_argument("--out", required=True, help="suite directory to write into")
    p.add_argument("--worlds", default="src/Arena/arena_simulation_setup/worlds", help="worlds root")
    p.add_argument("--base-scenario", default="default", help="scenario the cases were built on")
    p.set_defaults(fn=_cmd_bake)

    p = sub.add_parser("zones", help="classify a world's zones into roles")
    p.add_argument("world")
    p.add_argument("--worlds", default=None)
    p.set_defaults(fn=_cmd_zones)

    p = sub.add_parser("classify", help="ask a model for the zone roles no rule covers")
    p.add_argument("world", nargs="*", help="worlds to classify; default every world")
    p.add_argument("--worlds", default=None)
    p.add_argument("--model", default=None, help=f"override the model (default {DEFAULT_MODEL})")
    p.add_argument("--write", dest="dry_run", action="store_false",
                   help="actually call the model and store what it returns")
    p.set_defaults(fn=_cmd_classify, dry_run=True)

    p = sub.add_parser("populate", help="generate the four base scenarios for a world")
    p.add_argument("world")
    p.add_argument("--worlds", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write", action="store_true")
    p.add_argument("--robot-start", default=None, help="zone name or 'x,y,yaw' for the robot start")
    p.add_argument("--robot-goal", default=None, help="zone name or 'x,y,yaw' for the robot goal")
    p.add_argument("--prune", action="store_true",
                   help="remove generated base scenarios this run did not write (never hand-authored ones)")
    p.add_argument("--force", action="store_true",
                   help="replace scenarios this tool did not generate (they may be hand-authored)")
    p.set_defaults(fn=_cmd_populate)

    p = sub.add_parser("prompts", help="generate a prompt set from the taxonomy and freeze it")
    p.add_argument("world", nargs="?", default="", help="a world for its own set; omit for the universal set")
    p.add_argument("--worlds", default=None)
    p.add_argument("--n", type=int, default=0, help="how many prompts (default 40 universal, 20 per world)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="directory to write into (default: the shipped prompt directory)")
    p.add_argument("--balanced", action="store_true", help="fill the taxonomy evenly and reject near-duplicate texts (new sets)")
    p.add_argument("--supplement", action="store_true", help="write a `_2` set that closes the existing set's case gaps (ids continue)")
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=_cmd_prompts)

    p = sub.add_parser("promptgen", help="realise prompt x base scenarios for a world")
    p.add_argument("world")
    p.add_argument("--worlds", default=None)
    p.add_argument("--bases", default="", help="comma-separated base ids (default: every profile - normal, break, night, empty)")
    p.add_argument("--ids", default="", help="comma-separated prompt ids to restrict to")
    p.add_argument("--prompts", nargs="*", help="extra prompt files")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=None, help="model answer cache directory")
    p.add_argument("--model", default=None)
    p.add_argument("--oracle", action="store_true", help="use each prompt's own template slots instead of a model")
    p.add_argument("--strict", action="store_true", help="do not write partial scenarios")
    p.add_argument("--ladder", action="store_true", help="also write each multi-step scenario's ablation ladder (steps added one at a time)")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=_cmd_promptgen)

    p = sub.add_parser("commands", help="write every robot x planner x scenario launch command for a world")
    p.add_argument("world")
    p.add_argument("--worlds", default=None)
    p.add_argument("--out", default="AMBUSH_ROLLOUT_COMMANDS.md")
    p.add_argument("--robots", default="jackal,mpo700",
                   help="comma-separated chassis to include; '' for every wheeled robot in arena_robots")
    p.set_defaults(fn=_cmd_commands)

    p = sub.add_parser("worlds", help="feature table and variance of the worlds (map, furniture, zones)")
    p.add_argument("world", nargs="*", help="worlds to measure; default every world with a map")
    p.add_argument("--worlds", default=None)
    p.add_argument("--all", action="store_true", help="include worlds the generator cannot author on (fewer than 3 zones)")
    p.add_argument("--json", default=None, help="also write the table and report here")
    p.set_defaults(fn=_cmd_worlds)

    p = sub.add_parser("diversity", help="how varied the prompt sets and a world's realised scenarios are")
    p.add_argument("world", nargs="*", help="worlds whose scenario directories to measure")
    p.add_argument("--worlds", default=None)
    p.add_argument("--prompts", nargs="*", help="extra prompt files to measure")
    p.add_argument("--json", default=None)
    p.set_defaults(fn=_cmd_diversity)

    p = sub.add_parser("specs", help="list the shipped specs")
    p.set_defaults(fn=_cmd_specs)

    args = parser.parse_args(argv)
    if getattr(args, "worlds", "sentinel") is None:
        from .spec.schema import default_worlds_root

        args.worlds = default_worlds_root()
    try:
        return int(args.fn(args))
    except (SpecError, FileNotFoundError, ValueError) as exc:
        print(f"arena_bench: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
