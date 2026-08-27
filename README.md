# `arena_benchmark`

Benchmark dataset generation for [`tm_obstacles:=edge_case`](https://github.com/shibuina/Arena/blob/feature/edge_case/task_generator/task_generator/tasks/obstacles/edge_case/README.md).

> **Task record:** `documents/dataset_generation.md` in the parent development workspace —
> the running record for ClickUp 86eypgfq7: findings, locked decisions, milestones and the
> update log. This README covers the module; that file covers the work.

The edge-case mode builds situations. This package decides **which of them belong in a
benchmark** — the ones that separate planners rather than the ones every planner survives —
runs them across arms at scale, and ships the result as a portable scenario suite plus the
evidence that certifies each entry.

---

## Why this is not part of `task_generator`

`edge_case_sweep` drives a **running** node by setting `task.edge_case.*` parameters between
episodes. The planner does not arrive that way: `robot.mobile:=drl robot.mobile.planner:=…`
is folded into the robot's launch description when it spawns, so changing arms means a new
stack. Owning stack lifetimes is what this package adds, and it is deliberately not something
a package imported by every running simulation should do.

Three more boundaries follow from that one:

| | `task_generator` | `arena_benchmark` |
|---|---|---|
| Unit of work | an episode inside somebody else's stack | a stack |
| Dependencies | must stay importable in a bare sim | may want pyarrow, scipy, pandas |
| Output lifecycle | `cases.jsonl`, reshaped whenever the mode is | a versioned dataset with a schema contract |
| Failure posture | abort loudly, never record a silent no-op | survive a segfault at trial 4000 and resume |

**What it does not do: re-implement trial execution.** It emits a sweep plan per shard and
hands it to `arena launch … sweep:=`, which already waits for the stack *and* for the
`task.edge_case.*` parameter surface to exist. Re-implementing that wait is how a run ends up
executing the defaults and producing a file that looks like results.

---

## Quick start

```bash
arena_bench specs                       # what is shipped
arena_bench expand screen_v1            # validate, and report what it would run
arena_bench plan   screen_v1 --out data/bench/screen
```

```
spec     screen_v1
worlds   4  cells 21  arms 2  seeds 3
episodes 504
shards   8  (63 episodes each)
  differential     nav2_diff, cadrl*
```

An asterisk marks a learned planner; the arm without one is its family's control.

---

## The shape of a run

```
spec  ──expand──▶  manifest  ──shard──▶  plans  ──arena launch──▶  run dirs  ──assemble──▶  dataset
```

**Expand.** `worlds × cells × arms × seeds`, each row carrying two ids. `cell_uid` is the
situation with the planner left out, so the arms of one comparison share it — that is the
group key for every paired statistic, and the identity of a suite entry. `episode_uid` adds
the arm and the seed, and rides into the run as the sweep plan's case `label`, which the
runner copies verbatim into `trials.jsonl`. That is the join key end to end.

Neither id hashes a name. Renaming a cell must not invalidate rows already measured for it,
and two cells that hash alike really are one construction under two names — a spec bug
[`expand`](arena_benchmark/spec/expand.py) refuses rather than passing on to a join that
could not have disambiguated them.

**Shard.** One question per field: can the orchestrator change this between episodes, or does
it need a new stack? Everything in `LAUNCH_KEYS` needs a new stack; the whole
`task.edge_case.*` surface and the seed do not. Boot costs 60–90 s and a reset costs seconds,
so grouping by that key first is the difference between booting once per trial and once per
few dozen.

`scenario` is in the key even though `task.scenario.file` is a node parameter, because
changing it mid-session leaks flow regions across resets. That is an operational fact rather
than a schema one, and it belongs where the schedule is decided.

---

## Specs

A [sweep plan](https://github.com/shibuina/Arena/blob/feature/edge_case/arena_bringup/configs/sweeps/README.md) describes trials inside one stack.
A **spec** is the layer above: it names the arms too, so expanding it produces episodes that
span many launches.

```yaml
version: 1
worlds: [arena_arena_002, arena_classic_002]
arms:
  - {id: nav2_diff, robot: jackal, mobile: nav2, drive: differential}
  - {id: cadrl, robot: jackal, mobile: drl, planner: cadrl, drive: differential}
seeds: {start: 1000, count: 3}
cells:
  - id: c_headon_mid
    params:
      task.edge_case.inject: 1
      task.edge_case.place: intercept
      task.edge_case.approach_angle: 180.0
```

Validation is strict for the same reason the sweep plan's is: an unknown key is an error, and
a spec that silently ignored `aproach_angle` would report a swept factor that was never swept.

Two rules are methodology rather than typo-catching, enforced because getting either wrong
produces a dataset that looks fine and answers nothing:

- **Every drive family with a learned planner needs its own baseline.** `robot.mobile:=drl`
  picks a chassis matching the planner's `action_type`, so a differential arm and an
  omnidirectional one differ by robot body as well as by policy. One shared control would be
  measuring the chassis.
- **Cell parameters must be node parameters.** Anything decided at launch belongs on the arm.
  The orchestrator can only set the other kind between episodes.

| Spec | Episodes | What it is |
|---|---|---|
| [`smoke_v1`](specs/smoke_v1.yaml) | 36 | Two worlds × three control cells × two arms × three seeds. Proves the machinery, measures nothing. |
| [`screen_v1`](specs/screen_v1.yaml) | 504 | Stage 0 screening: 21 one-at-a-time cells over four worlds. Ranks cells; is **not** the dataset. |

Screening seeds and dataset seeds are disjoint on purpose. Ranking cells by observed
separation and then reporting that same separation is selection on the outcome, and the
effect sizes come out inflated.

---

## Generated base scenarios

```bash
arena_bench zones    acoustics_bend                        # what the world's rooms are for
arena_bench classify                                      # ask a model about the ones no rule covers
arena_bench populate  acoustics_bend --write --prune      # the eleven base variants
arena_bench catalogue acoustics_bend --write --prune      # the case variants on them
```

`--prune` removes generated scenarios the run did not write and never touches a hand-authored
one, so a renamed layout does not leave its predecessor beside it.

**Every pedestrian is written in rooms, never in coordinates.** `pose: laboratory`,
`waypoints: [laboratory, central_hallway, patient_ward]` — each name resolved at run time by
`PointResolver` to a free point in that polygon. Nothing in `populate` samples a coordinate any
more. What decides where people *start* is [`zone_screen`](arena_benchmark/scenario/zone_screen.py):
a zone is a home only if every point in it is 1.5 m off the robot's route and it is not a zone
the robot's own spec names (the robot resolves anywhere inside such a zone — that is how a
generated agent ended up 1 m from the robot's start and blocked every episode on
`arena_arena_002`). Every other zone is a destination, which is the point: a pedestrian who
never crosses a corridor never meets the robot.

**Everybody walks.** This was the reason pedestrians barely moved: a behaviour-tree agent never
follows its waypoints (`_advance_waypoints` skips `BehaviorTreeMovement`), and the shipped
`talker.yaml` holds a conversation forever by design — and half of every population was a talker.
Now exactly two agents per populated base are talkers, put in the same room so they have each
other to talk to (a Level D case like `denial` needs a conversation to lengthen), and everyone
else is a builtin `adult` that walks a route through one to four *other* zones, extended to at
least 15 m one way where the world allows. A one-room world writes the room twice: each reference
resolves independently, so the agent walks between two points of it rather than standing.

**Walkers are a mix of pedestrian types.** `adult`, `elder`, `hurried`, `distracted`, and — only
in the kind of building that has them — `child` (residential), `wheelchair` and `cart`
(hospital). All are humansim builtins (`config/agent_types/`), `mode: simple` and derived from
`adult`, so every one walks its rooms and every Level A knob lands on it. Weights per profile
live in `populate.MIX`; `busy` is a third hurried, `night` is elders and gurneys. Each agent's
speed is drawn from its *type's* distribution and scaled by profile and world tempo; radius is
left to the type. That needed an adapter fix: `ArenaHumanDynamicObstacle` used to default
`desired_velocity` to 1.0–1.5 m/s and `radius` to 0.35 for every agent, so a type's speed and
body never applied — and a Level A speed or radius knob on a perturbed agent was undone at
spawn. Now only what the scenario declares overrides the type, and the perturbation engine
drops a scalar that would shadow the field it just rewrote.

`waypoint_mode` follows the route's shape, checked against `agent_manager`'s advance logic:
`reverse` shuttles a two-point route, `repeat` turns three or more into a circuit, `random`
jumps between them. **`once` is never emitted** — it advances to the last waypoint and stops,
leaving the agent as furniture for the rest of the episode.

**Worlds differ from each other.** The RNG was keyed on profile and seed alone, so every world
generated at `--seed 0` drew the same numbers: every world's first pedestrian walked at exactly
1.24 m/s and the model list started at the same mesh. The world name is now part of the key, and
three multipliers drawn once per world — tempo, crowd size, and flow intensity — give each
building a character its four populated bases share and its neighbours do not. Measured across
the seven generated worlds: speeds now span 0.9–1.5 with `office_1` slow and `hospital_1` fast,
source rates range 0.106–0.168, and standing ceilings 6–9. Same world and seed still reproduce
byte-identically.

**The runtime resolver had to be fixed first.** `TM_Scenario.reset` validated resolved poses
with `Obstacles.SAFE_DIST` alone — 0.15 m, and by its own docstring a *sampling* clearance that
"says nothing about the pedestrian's own extent". A zone reference could put a 0.25 m agent
0.2 m from a wall on the first draw, with no warning, since `sample_point_in_polygon` retries only
what `is_valid` rejects. The mode now validates at `SAFE_DIST + PEDESTRIAN_BODY_RADIUS`, and
`zone_screen` mirrors the same number.

**Every populated base also carries a source and a sink.** Not for variety:
[`_perturb_regions`](https://github.com/shibuina/Arena/blob/feature/edge_case/task_generator/task_generator/tasks/obstacles/edge_case/impl.py) aborts
a Level B knob when the episode has no regions, so a base without them puts the entire
crowd/global tail - `flow_rate`, `crowd_density`, `crowd_volume`, `source_speed`,
`source_radius`, `sink_capacity`, `sink_absorption` - out of reach. Thirty generated bases had
none. Regions are written as `ref: <zone>`, so the region is the room rather than a rectangle
that goes stale when the world is edited.

The pair is chosen to **meet the robot**: a stream it never encounters exercises the knobs and
measures nothing, so the picker ranks candidate pairs by whether the line between them comes
within 1.5 m of the robot's route. Sources additionally keep 1.5 m from the robot's *start* -
spawning a body on the robot at t=0 is the failure - while sinks need not, since a sink only
absorbs. `populate` reports which pair it got and whether it meets the route.

No catalogue case sets a Level B knob yet, so nothing perturbs these regions; what they change
today is the world the other nine cases run in, which is now continuously replenished rather
than a fixed cast.

---

## Variants

The parent workspace's `documents/variant_plan.md` defines the variant plan. Every profile has 2–3 base variants
and every case has 2–3 variants, and the rule for both is the same: **a variant is a named
recipe that differs from its siblings on a categorical axis; a seed is not a variant.** Seeds
replicate inside a `cell_uid`; variants are different cells.

**Base variants** ([`scenario/variants.py`](arena_benchmark/scenario/variants.py)) sit on
three axes — ROUTE `fwd`/`rev`, FLOW `cross`/`counter`/`co`/`none`, CAST
`dispersed`/`clustered`/`stream_led` — and siblings must differ on at least two. `_a` is always
the base that existed before variants. The table is checked by `check_table`, and by a test.

**Case variants** ([`edgecase/variants.py`](arena_benchmark/edgecase/variants.py)) are deltas
over a case's block, each tagged with a mechanism siblings may not share: `blocking` is
`stand`/`crawl`/`doorway`, `approach` is `head_on`/`crossing`/`overtake`, `stampede` is
`push`/`rush`/`dense` — the three single-knob projections of `catalogue_plan.md`'s own
definition. Names carry the mechanism: `normal_b__blocking_crawl`.

**No cross product.** A case variant runs on one backdrop, rotated through the profile's base
variants (`backdrop_for`), unless it pins a cast it needs — `stampede_dense` raises the flow, so
it runs on `stream_led`. Siblings therefore differ in mechanism *and* backdrop, and every
backdrop is a shipped base: its control. 11 bases and up to 84 cases per world.

**Every episode lasts at least 30 s.** An episode ends when the robot completes its phases, so
`robot_for` shuttles the shipped endpoints until the route takes `MIN_EPISODE_S` at
`ROBOT_SPEED_MPS` — 0.5 m/s, the faster of nav2's controller defaults, so the floor holds under
the slower one too. Pedestrians get routes of at least 15 m one way where the world allows.

**Time-driven variants wait 10 s.** `DELAY_S` goes on every block that carries a `waypoints:`
plan, which is the one mechanism the mode can postpone (`scenario_block.DELAYABLE`); a knob
cannot be rewritten mid-episode, and the mode refuses a `delay` it cannot honour rather than
ignoring it. Injected encounters are delayed spatially instead: `encounter_at: 0.5` on a 30 s
route is 15 s in. Ten seconds lets the stream ramp a third of the way and the robot get under
way, and leaves the perturbation most of the episode.

---

## Where the model fits

The parent workspace's `documents/llm_proposal_plan.md` narrows the model's job twice.
The catalogue, not the model, is the source of cases. What is left, and what is built, is part
1's question: **which zones are work zones and which are break zones**, for the tail
[`zones.py`](arena_benchmark/scenario/zones.py)'s substring rules do not reach.

```bash
arena_bench classify                     # dry run: prints the prompt it would send
arena_bench classify --write             # asks, and merges what comes back
arena_bench classify hospital_1 --write  # one world
```

Three properties make this cheap and reviewable rather than a token round-trip per scenario:

- **Only the tail is sent.** The rules cover 99 of the tree's 116 distinct zone names for free
  and deterministically. Asking a model to confirm that `kitchen` is a break zone is exactly the
  expensive random number generator §0 of the plan warns about, so `Role.UNKNOWN` — a real
  answer, not a failure — is what selects the input.
- **The answer is a file, not a call.** Results land in
  [`scenario/zone_roles.yaml`](arena_benchmark/scenario/zone_roles.yaml) with the model id, a
  `prompt_sha` and the model's own one-line reason. `populate` reads that file and never calls a
  model, so regeneration stays deterministic, offline and free. Edit any row and set
  `source: human` and nothing will overwrite it again.
- **`unknown` is a permitted answer.** The prompt says so, and the model uses it: on
  `hallway_large`'s twelve `top_room_N` / `bottom_room_N` it returned `unknown` twelve times —
  *"placeholder name and contains no objects to indicate its purpose"* — rather than guessing.
  A wrong confident answer puts a crowd in a cupboard; an honest one routes the zone to a human.

**Measured yield on the current tree: one name.** 17 of 116 names are unclassified by rule, and
16 of those are auto-generated placeholders in `hallway_large` and `.generated` with no
meaningful name, no furniture and nothing to classify by. The one real answer was
`3_study → work`. The value here is in new worlds, not this one — and it correctly reports how
little there is to do rather than manufacturing work.

---

## Layout

| Path | Role | State |
|---|---|---|
| [`scenario/zones.py`](arena_benchmark/scenario/zones.py) | zone name → role | built |
| [`scenario/occupancy.py`](arena_benchmark/scenario/occupancy.py) | the furniture the map omits, for the screen | built |
| [`scenario/zone_screen.py`](arena_benchmark/scenario/zone_screen.py) | which zones may be named rather than numbered | built |
| [`scenario/populate.py`](arena_benchmark/scenario/populate.py) | the five base scenarios | built |
| [`scenario/regions.py`](arena_benchmark/scenario/regions.py) | the source/sink pair each base carries, by topology | built |
| [`scenario/variants.py`](arena_benchmark/scenario/variants.py) | base-variant recipes; the robot's route length | built |
| [`edgecase/variants.py`](arena_benchmark/edgecase/variants.py) | case-variant recipes, backdrop assignment, delay | built |
| [`scenario/zone_roles.yaml`](arena_benchmark/scenario/zone_roles.yaml) | the reviewed name → role mapping | built |
| [`propose/zone_roles.py`](arena_benchmark/propose/zone_roles.py) | model classification of the tail | built |
| [`propose/client.py`](arena_benchmark/propose/client.py) | one call into `task_generator.utils.gpt` | built |
| [`edgecase/catalogue.py`](arena_benchmark/edgecase/catalogue.py) | the nine cases and their applicability | built |
| [`edgecase/materialize.py`](arena_benchmark/edgecase/materialize.py) | (base, block) → a case scenario directory | built |
| [`spec/schema.py`](arena_benchmark/spec/schema.py) | spec validation | built |
| [`spec/expand.py`](arena_benchmark/spec/expand.py) | spec → manifest | built |
| [`spec/uid.py`](arena_benchmark/spec/uid.py) | `cell_uid`, `episode_uid` | built |
| [`orchestrate/shard.py`](arena_benchmark/orchestrate/shard.py) | group by launch cost | built |
| [`orchestrate/plan.py`](arena_benchmark/orchestrate/plan.py) | shard → sweep plan | built |
| `orchestrate/launcher.py`, `pool.py`, `resume.py` | run stacks, k at a time, resumably | not built |
| `record/nav_metrics.py` | time-to-goal, path length, SPL, jerk | not built |
| `assemble/` | join four streams; bake solved cases into portable scenarios | not built |
| `analysis/` | saturation guard, arm separation, power | not built |
| `feedback/` | aborted, inert, and unmodelable cases | not built |

Tests: `tests/` runs without ROS; `tests/ros/` needs a sourced environment and holds the two
contracts with other packages. [`test_plan_contract`](tests/ros/test_plan_contract.py): every
generated plan is one [`edge_case_plan`](https://github.com/shibuina/Arena/blob/feature/edge_case/task_generator/task_generator/utils/edge_case_plan.py)
accepts, labels intact. [`test_room_centric_load`](tests/ros/test_room_centric_load.py): every
form `populate` writes is one `ScenarioView.load` reads back — zone references, waypoint lists
mixing a name with a triple, and `ref:` on a region, which was implemented, used by zero shipped
scenarios and covered nowhere. Each of those fails by dropping an agent rather than by raising,
so the tests assert on what came back, not that it parsed.

```bash
# inside the container, after `source /opt/arena_ws/source`
cd /opt/arena_ws/src/Arena/arena_benchmark && python3 -m pytest tests -q
```

---

## What is owed

**A measured episode.** No `scores.jsonl` row exists anywhere in the tree — the criticality
scorer postdates every recorded run, so the outcome half of the pipeline has never emitted a
value. Nothing here can be validated end to end until it does.

**A second arm that works.** Every DRL planner run so far fails the same two ways: the nav2
global planner cannot produce a plan because the robot spawns outside the costmap bounds, and
`arena_planners` drops the `costmap` observation as non-serializable, so learned planners
receive degraded input without erroring.
