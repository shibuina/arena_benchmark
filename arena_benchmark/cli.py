"""``arena_bench`` - the command line over the benchmark layers.

Only the verbs that are implemented appear here. A subcommand that printed a plausible
summary for work it had not done would be worse than no subcommand at all, which is the same
reason the task mode aborts rather than recording a silent no-op.

    arena_bench expand screen_v1 --out data/bench/screen
    arena_bench plan   screen_v1 --out data/bench/screen
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
from .edgecase import NotApplicable, backdrop_for, block_for, door_blocker, regions_capable, variants_for
from .edgecase import write as write_case
from .orchestrate import shard, summarise, write_plan
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
    BASE_RECIPES,
    MIN_EPISODE_S,
    MIN_ROUTE_CLEARANCE,
    PROFILES,
    REGION_PROFILES,
    ROBOT_SPEED_MPS,
    build_regions,
    meets_route,
    ref_safe_zones,
    robot_for,
    robot_zones,
    route_points,
    route_seconds,
    screen_zones,
    stream_topology,
    world_statics,
)
from .scenario import build as build_base
from .scenario.variants import BY_ID as BY_RECIPE
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


def _cmd_plan(args: argparse.Namespace) -> int:
    spec = _load(args.spec)
    shards = shard(expand(spec))
    root = Path(args.out)
    for s in shards:
        path = write_plan(s, root / s.label)
        print(f"{len(s.episodes):4d} trials  {path}")
    _write_manifest(expand(spec), root)
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
            return zones, dict(robots[0])

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

        pairs = [
            (math.dist(centroid(a), centroid(b)), a["name"], b["name"])
            for i, a in enumerate(named) for b in named[i + 1:]
        ]
        _, start, goal = max(pairs)
        print(f"  ! {world}: no scenario declares a robot; using the two most separated zones "
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
    longest = robot_for(BASE_RECIPES["normal"][0], robot, zones)
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
    for profile in PROFILES:
        for recipe in BASE_RECIPES[profile]:
            spec = robot_for(recipe, robot, zones)
            route = route_points(spec, zones)
            block = {} if args.no_regions else build_regions(
                profile, zones=zones, route=route, resolvable=resolvable,
                robot_zones=robot_zones(robot), overrides=overrides,
                speed_scale=PROFILES[profile]["speed"], world=args.world, seed=args.seed,
                want=recipe.flow, cast=recipe.cast,
            )
            doc = build_base(
                profile, zones=zones, robot=spec, seed=args.seed, homes=homes,
                regions=block, overrides=overrides, world=args.world, cast=recipe.cast,
            )
            n = len(doc.get("dynamic") or [])
            if n == 0 and PROFILES[profile]["count"] > 0:
                print(f"  ! {recipe.id}: no zone to populate - this base is measurable as `empty`,")
                print("  !   and every case needing a crowd will refuse it")

            flow = "no stream" if recipe.flow == "none" else "no stream - the world affords no pair"
            if block:
                got = stream_topology(block, zones, route)
                match = "" if got == recipe.flow or recipe.flow == "any" else f", asked {recipe.flow}"
                meets = "meets the route" if meets_route(block, zones, route) else "MISSES the route"
                flow = f"{block['inflow']['ref']} -> {block['outflow']['ref']} ({got}{match}; {meets})"
            line = f"  {recipe.id:9s} {n:3d} agent(s)  {recipe.route}/{recipe.cast:10s} flow {flow}"

            if args.write:
                from .edgecase.materialize import WouldOverwrite

                try:
                    out = write_base(
                        doc, world_root=worlds / args.world, profile=recipe.id, force=args.force,
                        note=f"{recipe.route}/{recipe.flow}/{recipe.cast} - {recipe.note}",
                    )
                except WouldOverwrite as exc:
                    print(f"  {recipe.id:9s} SKIPPED - {exc}")
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


def _cmd_catalogue(args: argparse.Namespace) -> int:
    import yaml as _yaml

    from .edgecase import CATALOGUE, prune_generated
    from .edgecase.variants import check_table as check_case_table
    from .scenario.variants import check_table as check_base_table

    check_base_table()
    check_case_table()

    worlds = Path(args.worlds)
    root = worlds / args.world
    zones, _ = _world_parts(worlds, args.world)
    made = refused = 0
    per_case: dict[str, int] = {}
    written: list[str] = []

    for profile in PROFILES:
        recipes = BASE_RECIPES[profile]
        docs: dict[str, dict] = {}
        for recipe in recipes:
            base = root / "scenarios" / recipe.id / "scenario.yaml"
            if base.is_file():
                docs[recipe.id] = _yaml.safe_load(base.read_text()) or {}
        if not docs:
            hint = " (found the pre-variant layout; re-run `populate --write --prune`)" \
                if (root / "scenarios" / profile / "scenario.yaml").is_file() else ""
            print(f"  {profile}: no base scenarios - run `populate --write` first{hint}")
            continue
        for recipe_id, doc in docs.items():
            if not regions_capable(doc) and profile in REGION_PROFILES and BY_RECIPE[recipe_id].flow != "none":
                print(f"  ! {recipe_id}: no `regions:` block - a Level B knob on this base would abort")

        for case in CATALOGUE:
            if profile not in case.bases:
                continue
            for k, variant in enumerate(variants_for(case.id)):
                recipe = backdrop_for(profile, k, variant)
                base_doc = docs.get(recipe.id)
                if base_doc is None:
                    print(f"  {recipe.id}/{case.id}_{variant.id}: SKIPPED - backdrop {recipe.id} was not generated")
                    refused += 1
                    continue
                robot = (base_doc.get("robots") or [{}])[0]
                route = route_points(robot, zones)
                try:
                    block = block_for(case.id, profile, base_doc, zones=zones, variant=variant, route=route)
                except NotApplicable as exc:
                    print(f"  {recipe.id}/{case.id}_{variant.id}: SKIPPED - {exc}")
                    refused += 1
                    continue
                extra = [door_blocker(zones, route)] if variant.door else []
                if args.write:
                    out = write_case(base_doc, block, world_root=root, base=recipe.id,
                                     extra_dynamic=[e for e in extra if e])
                    written.append(out.name)
                    print(f"  {out.name}")
                made += 1
                per_case[case.id] = per_case.get(case.id, 0) + 1

    ceiling = sum(len(variants_for(c.id)) * len(c.bases) for c in CATALOGUE)
    print(f"{made} case scenario(s){' written' if args.write else ' (dry run)'} of a {ceiling} ceiling"
          f"{f'; {refused} refused by this world' if refused else ''}")
    print("  " + "  ".join(f"{cid}={n}" for cid, n in sorted(per_case.items())))
    if args.write and args.prune:
        keep = [*written, *(r.id for rs in BASE_RECIPES.values() for r in rs)]
        for gone in prune_generated(root, keep):
            print(f"  - pruned stale generated scenario {gone}")
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

    p = sub.add_parser("plan", help="write one sweep plan per shard, plus the manifest")
    p.add_argument("spec")
    p.add_argument("--out", required=True, help="run directory to write into")
    p.set_defaults(fn=_cmd_plan)

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

    p = sub.add_parser("populate", help="generate the five base scenarios for a world")
    p.add_argument("world")
    p.add_argument("--worlds", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write", action="store_true")
    p.add_argument("--robot-start", default=None, help="zone name or 'x,y,yaw' for the robot start")
    p.add_argument("--robot-goal", default=None, help="zone name or 'x,y,yaw' for the robot goal")
    p.add_argument("--no-regions", action="store_true",
                   help="omit the source/sink block, leaving the Level B tail unreachable")
    p.add_argument("--prune", action="store_true",
                   help="remove generated base scenarios this run did not write (never hand-authored ones)")
    p.add_argument("--force", action="store_true",
                   help="replace scenarios this tool did not generate (they may be hand-authored)")
    p.set_defaults(fn=_cmd_populate)

    p = sub.add_parser("catalogue", help="write the applicable edge cases onto the base scenarios")
    p.add_argument("--prune", action="store_true",
                   help="remove generated case scenarios this run did not write (never hand-authored ones)")
    p.add_argument("world")
    p.add_argument("--worlds", default=None)
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=_cmd_catalogue)

    p = sub.add_parser("commands", help="write every robot x planner x scenario launch command for a world")
    p.add_argument("world")
    p.add_argument("--worlds", default=None)
    p.add_argument("--out", default="AMBUSH_ROLLOUT_COMMANDS.md")
    p.add_argument("--robots", default="jackal,mpo700",
                   help="comma-separated chassis to include; '' for every wheeled robot in arena_robots")
    p.set_defaults(fn=_cmd_commands)

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
