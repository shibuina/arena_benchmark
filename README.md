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

A running node can switch scenarios between episodes (`task.scenario.file` + `reset_episode`).
The planner does not arrive that way: `robot.mobile:=drl robot.mobile.planner:=…`
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
changing it mid-session leaks a scenario's regions across resets. That is an operational fact rather
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

**Bases carry no sources or sinks** (removed 2026-08-28). They used to, so the knob-era Level B
tail (`flow_rate`, `crowd_density`, ...) had something to act on; that tail is gone, and a stream of
anonymous `flow_*` agents is population no case can name, own or score. The cast written in
`dynamic:` is the whole population. (`regions.py` and the `--no-regions` flag went with it.)

---

## Bases

One base per profile - `normal`, `break`, `night`, `empty` - written by `arena_bench populate
<world> --write --prune` into `worlds/<world>/scenarios/<profile>/`. (Until 2026-08-28 each
profile had two or three *variants* on route/flow/cast axes; with sources and sinks gone the
axes no longer distinguished anything, and the sparse casts had nobody walking, so they were
retired. `scenario/route.py` keeps the one thing that mattered: the shipped route, made long
enough for a 30 s episode.)

A base is `PEOPLE_PER_ZONE` (2) people per zone of the profile's role, times a per-world crowd
factor, at least four: `TALKER_SHARE` (30 %) of them in behaviour-tree pairs, one pair per room,
the rest walkers with ≥ 15 m itineraries - and at least two walkers whatever the size. People
start spread over the screened rooms (when fewer than three rooms of the profile's role pass the
screen, every other screened non-private room joins) - never in, or bound for, a sanitary room or a
cupboard (`populate.NOT_FOR_PEOPLE`: a toilet stall's 0.7 m door is sealed in HumanSim's grid). A walker drawn faster than HumanSim's
1.5 m/s default cap is a `hurried` type, whose own 2.4 m/s cap lets it actually hurry.

## Prompt-driven generation

`new_plan_2.md` (workspace root) is the design. A prompt, a world and a base scenario go in; a
scenario directory comes out. Package: [`promptgen/`](arena_benchmark/promptgen/__init__.py).

```
prompt + world + base
   -> Step 1  Type A by lexicon          lexicon.yaml, step_a.py   (blackout, evacuation, stampede, busy)
   -> Step 2  Type D by model            step_d.py, system_d.md    (stubborn, interaction_unbroken, formation_break)
   -> Step 3  Type B + C by model        step_bc.py, system_bc.md  (tune some pedestrians; one pedestrian vs the robot)
   -> scenarios/<base>__<prompt_id>/     pipeline.py
        scenario.yaml   population + derived profiles + `edge_case: {effects: [...]}`
        types/*.yaml    derived profiles (profiles.py: library + tiers; a model emits tiers, never numbers)
        record.json     decisions, ownership, failures (schema/placement/applicability/conflict/infeasible), dedup key, provenance
```

* **Prompt sets** are generated from the taxonomy with slots (who, group, place, and time
  constraints: onsets, gates and durations, `prompts.WHEN_A/WHEN_B/WHEN_D/DUR_C`) and frozen
  (`prompts/universal.yaml`, `prompts/<world>.yaml`; `arena_bench prompts [world] --write`).
  Each prompt carries `targets` (the steps it exercises) and `oracle` (the decisions its slots
  encode); the Type A targets are a round-trip test of the lexicon.
* **The world** ([`world.py`](arena_benchmark/promptgen/world.py)): zones with roles, the
  occupancy grid with furniture stamped in, the robot's first leg planned offline down the
  middle of the corridor, placement rules (clearance, reachability, 1.5 m from the start,
  1.0 m from the goal, 1.5 m from the route unless the case owns it), formation fitting.
* **Normal until the onset**: the population walks the base scenario's routes; every case is an
  effect with a `when`. A crowd-wide or per-agent parameter change is a `retune` (the agents are
  respawned where they stand on a derived profile) at the onset - blackout and Type B at 10 s,
  evacuation/stampede at 8 s - never baked into the population. Step D's members are ordinary
  walkers too: a `rally` brings them to their formation spots at `formation.at` (8 s, or the
  prompt's gate), a `retune` stops an unbroken group yielding, and a `formation_break` fires
  against the formation's centre afterwards. `record.json` / the block carry `formation`
  (`centre`, `radius`, `variant`, `at`) for the success criterion.
* **The robot's route is pinned**: the written scenario's `robots:` start and goal are the exact
  points the route was planned on (zone centroids snapped into free space on the furniture-aware
  grid), not zone names the runtime would resolve to random, walls-only-checked points - so the
  run drives the route the placements were designed against, and never a goal drawn beside a desk.
* **Ownership**: an agent belongs to at most one step; Type A's rally names the agents present
  at Step 1; B tunes only unowned agents; C is placed ≥ 2 m from a D placement (the model is
  asked again with the reason; in `--oracle` mode C is moved along the route).
* **Model**: Gemini through [`propose/client.py`](arena_benchmark/propose/client.py) at
  temperature 0, every answer cached by content hash under `data/promptgen/llm_cache`
  ([`llm.py`](arena_benchmark/promptgen/llm.py)), so regeneration is byte-identical and free.
  `--oracle` needs no model.

```bash
arena_bench prompts --write && arena_bench prompts hospital_1 --write
arena_bench promptgen hospital_1 --oracle -v                      # dry run, no model (every base: normal, break, night, empty)
arena_bench promptgen hospital_1 --write                          # model, writes
colcon build --packages-select arena_simulation_setup --symlink-install                 # then launch
```

The runtime side is the `edge_case` task mode
([README](../task_generator/task_generator/tasks/obstacles/edge_case/README.md)).

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
| [`scenario/populate.py`](arena_benchmark/scenario/populate.py) | the four base scenarios (one per profile) | built |
| [`scenario/route.py`](arena_benchmark/scenario/route.py) | the robot's shipped route, made long enough for a 30 s episode | built |
| [`scenario/zone_roles.yaml`](arena_benchmark/scenario/zone_roles.yaml) | the reviewed name → role mapping | built |
| [`propose/zone_roles.py`](arena_benchmark/propose/zone_roles.py) | model classification of the tail | built |
| [`propose/client.py`](arena_benchmark/propose/client.py) | one call into `task_generator.utils.gpt` | built |
| [`edgecase/materialize.py`](arena_benchmark/edgecase/materialize.py) | base scenario writer and the generated-marker pruning | built |
| [`promptgen/`](arena_benchmark/promptgen/__init__.py) | prompt → scenario: lexicon (A), model steps (D, B+C), profiles and tiers, placement, prompt sets | built |
| [`spec/schema.py`](arena_benchmark/spec/schema.py) | spec validation | built |
| [`spec/expand.py`](arena_benchmark/spec/expand.py) | spec → manifest | built |
| [`spec/uid.py`](arena_benchmark/spec/uid.py) | `cell_uid`, `episode_uid` | built |
| [`orchestrate/shard.py`](arena_benchmark/orchestrate/shard.py) | group by launch cost | built |
| `orchestrate/launcher.py`, `pool.py`, `resume.py` | run stacks, k at a time, resumably | not built |
| `record/nav_metrics.py` | time-to-goal, path length, SPL, jerk | not built |
| `assemble/` | join four streams; bake solved cases into portable scenarios | not built |
| `analysis/` | saturation guard, arm separation, power | not built |
| `feedback/` | aborted, inert, and unmodelable cases | not built |

Tests: `tests/` runs without ROS; `tests/ros/` needs a sourced environment and holds the
contract with the loader. [`test_room_centric_load`](tests/ros/test_room_centric_load.py): every
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
