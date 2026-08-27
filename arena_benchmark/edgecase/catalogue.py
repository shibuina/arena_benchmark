"""The nine edge cases, and which base scenarios can carry them.

`catalogue_plan.md` §1.2 and §1.3. Each case is a parameterised `edge_case:` block; the
applicability matrix says which base scenarios it is meaningful on. Both are data, so the
generator is a lookup rather than a pile of branches.

**Applicability is a gate, not a convention.** A case needs a population to act on: evacuating
an empty world produces an episode that aborts or measures nothing, and costs a boot to find
out. `applicable` refuses with the reason from the table.

Four of the nine run on capability that exists today. The other five are declared here with
`requires`, so the generator can refuse them until the capability lands rather than emitting a
block the mode will ignore.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from typing import Any

import attrs

from .variants import DELAY_S, DOOR_MAX_DIST_M, Variant, delayable, door_blocker, far_exit, merge_params


class Capability(enum.StrEnum):
    """Machinery a case needs beyond what the `edge_case` mode has.

    Every capability the catalogue once waited on is now built - crowd-wide perturbation
    (`catalogue_plan.md` §3.2), runtime route rewriting via `SetWaypoints` (§3.1), and
    robot-proximity triggers (§3.3) - so every entry currently requires `NONE`. The enum and
    the gate stay because they are what keeps the *next* unbuilt case from shipping as a
    scenario the mode silently ignores, which is the failure this whole pipeline is built to
    refuse.
    """

    NONE = "none"
    CROWD_SCOPE = "crowd_scope"
    """Perturb every placed agent, not one selected agent - `catalogue_plan.md` §3.2."""
    WAYPOINTS = "waypoints"
    """Rewrite agent routes at run time via SetWaypoints - §3.1."""
    TRIGGERS = "triggers"
    """Fire on robot proximity rather than a clock - §3.3."""


@attrs.frozen(kw_only=True)
class Case:
    """One catalogue entry."""

    id: str
    note: str
    params: Mapping[str, Any]
    requires: Capability = Capability.NONE
    #: Base profiles this case is meaningful on.
    bases: frozenset[str]
    #: Why it is excluded elsewhere - quoted back when the gate refuses.
    needs: str
    #: Where a `rally` plan sends the crowd, when the case carries one and names no explicit
    #: `target`. Resolved per world by `block_for`, because a zone name is not portable:
    #: `exit` is the way out, `work` is where work happens. Ignored by cases with no rally.
    rally_to: str = "exit"
    #: True when the knob is behavioural (Level D). Those live on behaviour-tree fields, and
    #: every builtin agent type (`adult`, `elder`, `robot`) is `mode: simple`, so the knob is
    #: structurally inert on a population made of them. Measured: `blocking` with
    #: `step_patience` aborted 14 times in a row on a generated `adult` population.
    needs_bt: bool = False


#: Profiles that have a crowd at all.
_POPULATED = frozenset({"normal", "break", "busy", "night"})
#: Profiles dense enough for density-dependent cases.
_DENSE = frozenset({"normal", "busy"})
#: Every profile, including `empty` - a case that injects its own agent needs nothing present.
_ALL = frozenset({"normal", "break", "busy", "night", "empty"})

#: What an injected agent is made of when there is nobody to copy. The mode's default is to
#: adopt a type and a mesh from the base population - experimentally the right choice, since
#: the tail is then the only thing separating the agent from its neighbours - and it refuses,
#: loudly, to invent one. Measured on `empty`: 233 aborted episodes and zero `ok` in two
#: windows, `auto_reset` spinning every 7 s. `adult` is the builtin kinematic type every Level A
#: knob lands on; `arenian` is the mesh the human simulator itself falls back to, shipped in
#: `_assets/_local`, so it resolves in every world without a network fetch.
EMPTY_INJECT_TYPE = "adult"
EMPTY_INJECT_MODEL = "arenian"


CATALOGUE: tuple[Case, ...] = (
    Case(
        id="blocking", note="an agent placed to obstruct the robot, and holding position",
        bases=_ALL, needs="injects its own agent",
        # `desired_velocity` (Level A), not `step_patience` (Level D). Both express "hold
        # position", but the behavioural one is inert on the builtin agent types a generated
        # population is made of, and aborts the case rather than obstructing anything.
        params={"inject": 1, "place": "intercept", "approach_angle": 180.0,
                "encounter_at": 0.5, "knob": "desired_velocity", "level": 0.0,
                "catalogue": "level_a"},
    ),
    Case(
        id="approach", note="an agent walks at the robot",
        bases=_ALL, needs="injects its own agent",
        # `scope: injected`, not `all`: the case is *one* agent walking at the robot. Over the
        # whole population it would be an evacuation toward the robot, which is a different
        # case and not one anybody authored here.
        params={"inject": 1, "place": "intercept", "approach_angle": 180.0,
                "encounter_at": 0.5,
                "waypoints": {"mode": "track_robot", "scope": "injected", "period": 2.0}},
    ),
    Case(
        id="denial", needs_bt=True, note="a conversation long enough that the robot must route around it",
        bases=_POPULATED, needs="two agents to converse",
        params={"mode": "perturb", "inject": 0, "knob": "interaction_duration", "level": 0.99},
    ),
    Case(
        id="stubborn", note="agents do not yield to the robot",
        bases=_POPULATED, needs="a crowd to ignore the robot",
        params={"mode": "perturb", "inject": 0, "knob": "repulsion_strength", "level": 0.0, "catalogue": "level_a"},
    ),
    Case(
        id="blackout", note="no lights: everyone slower, perception shorter",
        bases=_POPULATED, needs="a crowd to slow",
        params={"mode": "perturb", "inject": 0, "knob": "desired_velocity", "level": 0.05,
                "catalogue": "level_a", "scope": "all"},
    ),
    Case(
        id="erratic", note="waypoints change frequently; no settled flow to predict",
        bases=_POPULATED, needs="a crowd to reroute",
        params={"inject": 0, "waypoints": {"mode": "shuffle", "scope": "all", "period": 4.0,
                                           "radius": 3.0}},
    ),
    Case(
        id="evacuation", note="every agent rushes for one exit",
        bases=frozenset({"normal", "break", "busy"}), needs="a crowd to evacuate",
        # `mode: perturb` with `scope: all`, not `inject: 0` under the default inject mode: a
        # knob set under `mode: inject` with nothing injected has no target, and the mode
        # applies it to nobody without aborting. The rally is what makes it an evacuation;
        # the crowd-wide speed knob is what makes it a *rush*.
        # `waypoints.target` is filled in per world by `block_for`, because an exit is a zone
        # name and a zone name is not portable.
        params={"mode": "perturb", "inject": 0, "scope": "all",
                "knob": "desired_velocity", "level": 0.95, "catalogue": "level_a",
                "waypoints": {"mode": "rally", "scope": "all", "at": 8.0}},
    ),
    Case(
        id="stampede", note="a crowd surging back to work, dense and no longer yielding",
        # `break` included: a break-room population *is* the crowd a return-to-work surge is
        # made of, and it is the base where the case reads most cleanly. It stays out of
        # `night` for the usual reason - four agents is not a surge.
        bases=_DENSE | frozenset({"break"}), needs="density to stampede",
        # The rally is what makes this a stampede rather than a dense standstill. Low
        # `repulsion_strength` alone gives a crowd that overlaps in place; a crowd needs a
        # *shared destination* to surge, and `back to work` is the one every profile here has
        # in common - people on break return to it, people in transit are heading for it, and
        # people already at work converge on the main one.
        rally_to="work",
        params={"mode": "perturb", "inject": 0, "knob": "repulsion_strength", "level": 0.02,
                "catalogue": "level_a", "scope": "all",
                "waypoints": {"mode": "rally", "scope": "all", "at": 4.0}},
    ),
    Case(
        id="formation_break", note="a queue that breaks as the robot approaches",
        bases=_DENSE, needs="a queue to break",
        # Expressed as a triggered scatter rather than as a triggered *knob*. A trigger can
        # only gate what is time-driven: the pedestrian backend allocates a fresh agent id on
        # every spawn and has no in-place type replacement, so an agent's kinematics cannot be
        # rewritten mid-episode. A queue whose members are thrown to their own neighbourhood
        # the moment the robot closes in *is* the formation breaking, and unlike a knob it
        # actually fires. No behaviour tree needed, which also widens the bases it can run on.
        params={"inject": 0,
                "waypoints": {"mode": "shuffle", "scope": "all", "period": 3.0, "radius": 2.0},
                "trigger": {"robot_within": 3.0}},
    ),
)

BY_ID: dict[str, Case] = {c.id: c for c in CATALOGUE}


class NotApplicable(Exception):
    """A case cannot be built on this base scenario."""


def applicable(case_id: str, profile: str) -> bool:
    return profile in BY_ID[case_id].bases


def bt_capable(base_doc: dict[str, Any] | None) -> bool:
    """Whether a base scenario's population can carry a behavioural (Level D) knob.

    True when any agent references a scenario-local agent-type file (`./doctor.yaml`). The
    builtin types are all `mode: simple`, so a Level D knob on them is inert - the mode aborts
    the case rather than running an unperturbed episode, which is correct and still wasted a
    stack to discover.
    """
    if not base_doc:
        return False
    for agent in base_doc.get("dynamic") or []:
        if not isinstance(agent, dict):
            continue
        kind = str((agent.get("agent") or {}).get("agent_type", ""))
        if kind.endswith((".yaml", ".yml")):
            return True
    return False


def regions_capable(base_doc: dict[str, Any] | None) -> bool:
    """Whether a base scenario carries the flow regions a Level B knob would address.

    No catalogue case sets a Level B knob today, so nothing gates on this yet - it is what the
    generator reports, so an author can see which bases could carry one. `_perturb_regions`
    aborts a Level B case when the episode has no regions, naming the fix in the error itself:
    "Use a scenario with a `regions:` block". `populate` writes one wherever the world affords a
    source and a sink, and not every world does.
    """
    return bool((base_doc or {}).get("regions"))


def cases_for(
    profile: str,
    *,
    available: frozenset[Capability] | None = None,
    base_doc: dict[str, Any] | None = None,
) -> list[Case]:
    """Every case meaningful on this base profile, optionally filtered by capability.

    Two gates, and neither is a preference: the profile must be one the case is defined on, and
    the population must be able to carry a behavioural knob if the case uses one. Each refusal
    here is one that would otherwise cost a stack boot to discover.
    """
    have = {Capability.NONE} if available is None else set(available) | {Capability.NONE}
    bt = bt_capable(base_doc)
    return [
        c for c in CATALOGUE
        if profile in c.bases and c.requires in have and (bt or not c.needs_bt)
    ]


#: Name fragments that mark a zone people leave a building *through*, best first. Used when a
#: world declares no transit zone, which is common: an apartment has no corridor, and an office
#: floor plan often names its exit `reception` rather than `lobby`.
_EGRESS_HINTS: tuple[str, ...] = (
    "exit", "entrance", "entry", "lobby", "reception", "hallway", "corridor", "foyer", "living",
)


def _area(zone: Mapping[str, Any]) -> float:
    """Shoelace area of a zone polygon. Zero for anything that is not one."""
    corners = [c for c in (zone.get("corners") or []) if isinstance(c, Mapping)]
    if len(corners) < 3:
        return 0.0
    pts = [(float(c.get("x", 0.0)), float(c.get("y", 0.0))) for c in corners]
    return abs(sum(
        pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
        for i in range(len(pts))
    )) / 2.0


def work_target(zones: Sequence[Mapping[str, Any]] | None) -> str | None:
    """The zone a crowd surges *back to*, or None if this world declares none.

    "Working space" means exactly what it meant when the base scenarios were populated - the
    zones `normal` fills - so this defers to `populate.work_zones` rather than re-deriving it.
    Defining it twice would let the two drift, and on a residential world, where the mapping
    shifts so living areas become the active zones, a second definition would send the surge
    into a bedroom.

    The largest of them, because a crowd converges on the main work area rather than on
    whichever one sorts first. Ties break on name, so the pick is reproducible.
    """
    named = [z for z in (zones or []) if isinstance(z, Mapping) and z.get("name")]
    if not named:
        return None

    from ..scenario.populate import work_zones  # noqa: PLC0415

    wanted = set(work_zones(zones))
    candidates = [z for z in named if str(z["name"]) in wanted]
    if not candidates:
        return None
    return str(max(candidates, key=lambda z: (_area(z), str(z["name"])))["name"])


def rally_target(zones: Sequence[Mapping[str, Any]] | None) -> str | None:
    """The zone an evacuation rallies on, or None if this world declares none.

    Three rules, in order, because no single one covers the worlds in the tree:

    1. **A transit zone.** A corridor or lobby is what a building empties through, and it is
       the only role that means egress rather than occupancy.
    2. **A name that reads like a way out.** Most worlds declare no transit zone at all - an
       apartment has no corridor, and an office often calls its exit `reception`.
    3. **The largest zone.** Where a crowd with nowhere named to go actually converges. A weak
       rule, and last for that reason, but far better than rallying a building into a bathroom,
       which is what picking any old break zone did.

    Ties inside a rule break on sorted name, so the choice is reproducible across
    regenerations rather than dependent on the order `world.yaml` happens to list zones in.
    """
    named = [z for z in (zones or []) if isinstance(z, Mapping) and z.get("name")]
    if not named:
        return None

    from ..scenario.zones import Role, zones_by_role  # noqa: PLC0415

    transit = sorted(zones_by_role(zones).get(Role.TRANSIT) or [])
    if transit:
        return transit[0]

    for hint in _EGRESS_HINTS:
        matched = sorted(str(z["name"]) for z in named if hint in str(z["name"]).lower())
        if matched:
            return matched[0]

    return str(max(named, key=lambda z: (_area(z), str(z["name"])))["name"])


def block_for(
    case_id: str,
    profile: str,
    base_doc: dict[str, Any] | None = None,
    zones: Sequence[Mapping[str, Any]] | None = None,
    variant: Variant | None = None,
    route: Sequence[tuple[float, float]] = (),
) -> dict[str, Any]:
    """The `edge_case:` block for one case on one base scenario.

    `zones` resolves anything world-specific - the exit an evacuation rallies on, the door a
    blocker stands in. A case that needs one and cannot get one is refused rather than written
    without it: a rally with no target is a plan the mode will abort on at run time, which costs
    a whole stack to discover something knowable here.

    `variant` applies a `Variant` over the case's block; the block's `id` becomes
    `<case>_<variant>` so the scenario name and the cell carry the mechanism. `route` is the
    robot's route as a polyline, for the variants that resolve against it.
    """
    case = BY_ID.get(case_id)
    if case is None:
        raise NotApplicable(f"unknown case {case_id!r}; known: {sorted(BY_ID)}")
    if profile not in case.bases:
        raise NotApplicable(
            f"case {case_id!r} is not meaningful on base {profile!r}: it {case.needs}"
        )
    if case.needs_bt and base_doc is not None and not bt_capable(base_doc):
        raise NotApplicable(
            f"case {case_id!r} uses a behavioural knob ({case.params.get('knob')!r}), which is "
            f"structurally inert on the builtin agent types this base population is made of. "
            f"It needs a base whose agents reference a behaviour-tree type (./name.yaml)."
        )

    params = dict(case.params) if variant is None else merge_params(case.params, variant)
    label = case.id if variant is None else f"{case.id}_{variant.id}"
    note = case.note if variant is None else f"{case.note} - {variant.note}"
    block: dict[str, Any] = {"id": label, "note": note, **params}

    if variant is not None:
        if variant.needs_regions and base_doc is not None and not regions_capable(base_doc):
            raise NotApplicable(
                f"{label!r} perturbs flow regions ({params.get('knob')!r}) and this base declares "
                f"none; `_perturb_regions` would abort it. It needs a base with a `regions:` block."
            )
        if variant.door and door_blocker(list(zones or []), route) is None:
            raise NotApplicable(
                f"{label!r} needs a door within {DOOR_MAX_DIST_M} m of the robot's route, and this "
                f"world has none there."
            )
        if variant.needs_exits:
            target = far_exit(list(zones or []), route)
            if target is None:
                raise NotApplicable(
                    f"{label!r} rallies on a second exit, and this world declares fewer than "
                    f"{variant.needs_exits} transit zones - it would be `panic` under another name."
                )
            plan = block.get("waypoints")
            if isinstance(plan, Mapping):
                block["waypoints"] = {**plan, "target": target}
        if variant.delay and delayable(block):
            block["delay"] = DELAY_S

    injects = int(block.get("inject", 0) or 0) >= 1
    population_empty = profile == "empty" or (base_doc is not None and not (base_doc.get("dynamic") or []))
    if injects and population_empty:
        block.setdefault("inject_type", EMPTY_INJECT_TYPE)
        block.setdefault("inject_model", EMPTY_INJECT_MODEL)

    plan = block.get("waypoints")
    if isinstance(plan, Mapping) and plan.get("mode") == "rally" and not plan.get("target"):
        wants_work = case.rally_to == "work"
        target = work_target(zones) if wants_work else rally_target(zones)
        if target is None:
            where = "a working space" if wants_work else "an exit"
            hint = ("annotate the world with a work zone"
                    if wants_work else "annotate the world with a transit zone")
            raise NotApplicable(
                f"case {case_id!r} rallies on {where}, and this world declares none. Give the "
                f"block an explicit `waypoints.target: [x, y]`, or {hint}."
            )
        block["waypoints"] = {**plan, "target": target}

    return block
