"""The catalogue: nine cases, five base scenarios, and the applicability gate between them."""

from __future__ import annotations

import pytest
import yaml

from arena_benchmark.edgecase import (
    BY_ID, CATALOGUE, Capability, NotApplicable, annotate, block_for, case_scenario_id, cases_for,
)
from arena_benchmark.edgecase.catalogue import rally_target, work_target
from arena_benchmark.scenario import PROFILES, Role, build, classify, clearance, route_points


# --- zone classification (C0) -------------------------------------------------------------

@pytest.mark.parametrize(("name", "role"), [
    ("open_work_area", Role.WORK), ("nurse_station", Role.WORK), ("patient_room_3", Role.WORK),
    ("waiting_area", Role.BREAK), ("kitchen", Role.BREAK), ("bathroom", Role.BREAK),
    ("arm_w", Role.TRANSIT), ("junction", Role.TRANSIT), ("central_hallway", Role.TRANSIT),
    ("bedroom_1", Role.PRIVATE),
    ("top_room_3", Role.UNKNOWN),
])
def test_classification(name, role):
    assert classify(name) is role


@pytest.mark.parametrize(("name", "role"), [
    ("nurse_resting_room", Role.BREAK),   # 'resting' must beat 'nurse'
    ("wc_waiting", Role.BREAK),           # 'waiting' must beat nothing else claiming it
    ("doctors_resting_room", Role.BREAK),
])
def test_rule_order_traps(name, role):
    """Names where two rules could match; the break rule must win."""
    assert classify(name) is role


def test_unknown_is_an_answer_not_a_failure():
    """Guessing that `storage_3` is a work zone silently puts a crowd in a cupboard."""
    assert classify("zzz_unnamed_9") is Role.UNKNOWN


# --- base scenarios (C1) ------------------------------------------------------------------

ZONES = [
    {"name": "open_work_area", "corners": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}, {"x": 0, "y": 10}]},
    {"name": "waiting_area", "corners": [{"x": 20, "y": 0}, {"x": 30, "y": 0}, {"x": 30, "y": 10}, {"x": 20, "y": 10}]},
]
ROBOT = {"start": [-5.0, 5.0, 0.0], "goal": [-1.0, 5.0, 0.0]}


def test_every_profile_builds():
    for profile in PROFILES:
        doc = build(profile, zones=ZONES, robot=ROBOT, seed=1)
        assert "robots" in doc and "dynamic" in doc


def test_population_ordering():
    n = {p: len(build(p, zones=ZONES, robot=ROBOT, seed=1)["dynamic"]) for p in PROFILES}
    assert n["empty"] == 0
    assert n["night"] <= n["normal"] <= n["busy"]


def test_night_walks_slower():
    def speeds(p):
        return [a["agent"]["desired_velocity"] for a in build(p, zones=ZONES, robot=ROBOT, seed=1)["dynamic"]]
    assert max(speeds("night")) < max(speeds("normal"))


def test_generation_is_deterministic():
    a = build("busy", zones=ZONES, robot=ROBOT, seed=7)
    b = build("busy", zones=ZONES, robot=ROBOT, seed=7)
    assert yaml.safe_dump(a) == yaml.safe_dump(b)


def test_a_different_seed_gives_a_different_population():
    a = build("normal", zones=ZONES, robot=ROBOT, seed=1)
    b = build("normal", zones=ZONES, robot=ROBOT, seed=2)
    assert yaml.safe_dump(a) != yaml.safe_dump(b)


def test_robot_is_copied_verbatim():
    assert build("normal", zones=ZONES, robot=ROBOT, seed=1)["robots"][0] == ROBOT


def test_normal_and_break_use_different_rooms():
    def homes(p):
        return {a["pose"] for a in build(p, zones=ZONES, robot=ROBOT, seed=1)["dynamic"]}
    assert homes("normal") == {"open_work_area"}
    assert homes("break") == {"waiting_area"}


def test_every_pose_and_waypoint_is_a_zone_name():
    for a in build("busy", zones=ZONES, robot=ROBOT, seed=1)["dynamic"]:
        assert isinstance(a["pose"], str)
        assert all(isinstance(w, str) for w in a["waypoints"])


# --- the catalogue and its gate (C2) ------------------------------------------------------

def test_catalogue_size():
    assert len(CATALOGUE) == 9


def test_applicability_totals_34_per_world():
    """45 would be the full product; a case needs a population to act on."""
    total = sum(len([c for c in CATALOGUE if p in c.bases]) for p in PROFILES)
    assert total == 34


def test_empty_carries_only_self_injecting_cases():
    ids = {c.id for c in CATALOGUE if "empty" in c.bases}
    assert ids == {"blocking", "approach"}


def test_density_cases_are_excluded_from_sparse_bases():
    """`stampede` is the exception, and only on `break`: a break-room population *is* the
    crowd a return-to-work surge is made of. It stays out of `night` and `empty`."""
    assert not any("formation_break" == c.id and b in c.bases
                   for c in CATALOGUE for b in ("break", "night", "empty"))
    assert not any("stampede" == c.id and b in c.bases
                   for c in CATALOGUE for b in ("night", "empty"))


def test_inapplicable_case_is_refused_with_a_reason():
    with pytest.raises(NotApplicable, match="a crowd to evacuate"):
        block_for("evacuation", "empty")


def test_unknown_case_is_refused():
    with pytest.raises(NotApplicable, match="unknown case"):
        block_for("nope", "normal")


BT_BASE = {"dynamic": [{"name": "d", "agent": {"agent_type": "./doctor.yaml"}}]}
SIMPLE_BASE = {"dynamic": [{"name": "a", "agent": {"agent_type": "adult"}}]}


def test_every_case_runs_on_built_machinery():
    """Crowd scope, runtime waypoints and proximity triggers all landed, so nothing is
    withheld any more."""
    today = {c.id for c in cases_for("normal", base_doc=BT_BASE)}
    assert today == {c.id for c in CATALOGUE if "normal" in c.bases}
    assert {c.requires for c in CATALOGUE} == {Capability.NONE}


def test_the_capability_gate_still_withholds():
    """Exercised with a case pinned to a capability that is not on offer, since none is
    unbuilt now. The gate is what keeps the *next* unbuilt case from shipping as a scenario
    the mode silently ignores."""
    import attrs

    import arena_benchmark.edgecase.catalogue as cat

    pinned = attrs.evolve(BY_ID["stubborn"], requires=Capability.TRIGGERS)
    original = cat.CATALOGUE
    cat.CATALOGUE = tuple(pinned if c.id == "stubborn" else c for c in original)
    try:
        assert "stubborn" not in {c.id for c in cases_for("normal", base_doc=BT_BASE)}
        withall = {c.id for c in cases_for("normal", available=frozenset(Capability), base_doc=BT_BASE)}
        assert "stubborn" in withall
    finally:
        cat.CATALOGUE = original


def test_behavioural_cases_are_withheld_from_a_simple_population():
    """Level D knobs live on behaviour-tree fields; every builtin type is `mode: simple`, so the
    knob is inert and the mode aborts. `blocking` aborted 14 times in a row before this gate."""
    simple = {c.id for c in cases_for("normal", base_doc=SIMPLE_BASE)}
    assert "denial" not in simple
    assert "blocking" in simple, "blocking is Level A now and must survive a simple population"


def test_a_behavioural_case_on_a_simple_base_is_refused_with_a_reason():
    with pytest.raises(NotApplicable, match="structurally inert"):
        block_for("denial", "normal", SIMPLE_BASE)


def test_perturbing_cases_use_perturb_mode():
    """`mode: inject` with `inject: 0` leaves the knob with no target - it silently does
    nothing, which is exactly what `denial` and `stubborn` did in a real rollout."""
    for case_id in ("denial", "stubborn", "blackout", "stampede", "formation_break"):
        params = BY_ID[case_id].params
        if params.get("inject") == 0 and params.get("knob"):
            assert params.get("mode") == "perturb", f"{case_id} perturbs but does not say so"


def test_block_carries_id_and_note():
    b = block_for("blocking", "normal")
    assert b["id"] == "blocking" and b["note"]


# --- the writer ---------------------------------------------------------------------------

def test_annotate_replaces_never_stacks():
    """One case per scenario is the one-at-a-time discipline enforced by the file format."""
    base = {"robots": [], "dynamic": [], "edge_case": {"id": "old"}}
    out = annotate(base, {"id": "new"}, base="1")
    assert out["edge_case"]["id"] == "new"
    assert out["edge_case"]["base"] == "1"


def test_annotate_leaves_the_base_intact():
    base = {"robots": [{"start": [1, 2, 0]}], "dynamic": [{"name": "p"}]}
    out = annotate(base, {"id": "c"}, base="normal")
    assert out["robots"] == base["robots"] and out["dynamic"] == base["dynamic"]


def test_case_scenario_id_names_the_base():
    assert case_scenario_id("normal", "blocking") == "normal__blocking"


# --- free space (occupancy) ---------------------------------------------------------------

class FakeGrid:
    """A 10x10 m world, free except a 2x2 m block at (4..6, 4..6)."""

    def occupied(self, x, y):
        return not (0 <= x <= 10 and 0 <= y <= 10) or (4 <= x <= 6 and 4 <= y <= 6)

    def free(self, x, y, radius=0.5):
        import math
        steps = max(1, math.ceil(radius / 0.05))
        for dc in range(-steps, steps + 1):
            for dr in range(-steps, steps + 1):
                if dc * dc + dr * dr > steps * steps:
                    continue
                if self.occupied(x + dc * 0.05, y + dr * 0.05):
                    return False
        return True


ROOM = [{"name": "open_work_area",
         "corners": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}, {"x": 0, "y": 10}]}]
FAR_ROBOT = {"start": [-5.0, -5.0, 0.0], "goal": [-4.0, -5.0, 0.0]}


def test_static_objects_block_placement():
    from arena_benchmark.scenario.occupancy import blocked_by_static
    desk = [{"name": "desk", "pose": [5.0, 5.0, 0.0], "interaction_radius": 1.0}]
    assert blocked_by_static(5.2, 5.0, desk, 0.5)
    assert not blocked_by_static(9.0, 9.0, desk, 0.5)


def test_a_zone_ref_static_is_skipped_not_crashed_on():
    from arena_benchmark.scenario.occupancy import blocked_by_static
    assert not blocked_by_static(1.0, 1.0, [{"name": "desk", "pose": "reception"}], 0.5)


def test_write_base_refuses_to_replace_a_hand_authored_scenario(tmp_path):
    """`populate --write` once overwrote hospital_1's shipped 14-agent `normal`. A generated
    scenario is regenerable; a hand-authored one is not."""
    from arena_benchmark.edgecase.materialize import WouldOverwrite, write_base

    world = tmp_path / "w"
    (world / "scenarios" / "normal").mkdir(parents=True)
    (world / "scenarios" / "normal" / "scenario.yaml").write_text("robots: []\ndynamic: []\n")
    with pytest.raises(WouldOverwrite, match="not generated by this tool"):
        write_base({"robots": [], "dynamic": []}, world_root=world, profile="normal")


def test_write_base_replaces_its_own_output(tmp_path):
    """Regenerating must stay idempotent - the marker is what makes that safe."""
    from arena_benchmark.edgecase.materialize import write_base

    world = tmp_path / "w"
    write_base({"robots": [], "dynamic": []}, world_root=world, profile="normal")
    write_base({"robots": [], "dynamic": []}, world_root=world, profile="normal")


def test_missing_annotation_falls_back_to_a_default_footprint(tmp_path):
    from arena_benchmark.scenario.occupancy import DEFAULT_FOOTPRINT, asset_footprint
    assert asset_footprint("No/Such/Model", tmp_path) == DEFAULT_FOOTPRINT


# --- rally targets -------------------------------------------------------------------------

def _zone(name, size=1.0):
    return {"name": name, "corners": [{"x": 0.0, "y": 0.0}, {"x": size, "y": 0.0},
                                      {"x": size, "y": size}, {"x": 0.0, "y": size}]}


def test_a_transit_zone_wins_the_rally():
    """The only role that means egress rather than occupancy."""
    zones = [_zone("kitchen", 9.0), _zone("central_hallway", 1.0)]
    assert rally_target(zones) == "central_hallway"


def test_an_egress_name_is_used_when_no_zone_is_transit():
    """Most worlds declare no transit zone: an apartment has no corridor, and an office often
    calls its exit `reception`."""
    assert rally_target([_zone("open_work_area", 9.0), _zone("reception")]) == "reception"


def test_the_largest_zone_is_the_last_resort():
    """Better than rallying a building into a bathroom, which picking any break zone did."""
    assert rally_target([_zone("bathroom", 1.0), _zone("bedroom_1", 5.0)]) == "bedroom_1"


def test_the_pick_is_reproducible_across_zone_ordering():
    zones = [_zone("hallway_a"), _zone("hallway_b")]
    assert rally_target(zones) == rally_target(list(reversed(zones)))


def test_a_world_with_no_zones_has_no_rally_target():
    assert rally_target([]) is None
    assert rally_target(None) is None


def test_an_evacuation_block_carries_a_resolved_exit():
    block = block_for("evacuation", "normal", BT_BASE, zones=[_zone("central_hallway")])
    assert block["waypoints"]["target"] == "central_hallway"


def test_an_evacuation_is_refused_rather_than_written_without_an_exit():
    """A rally with no target aborts at run time - a whole stack spent to discover something
    knowable here."""
    with pytest.raises(NotApplicable, match="rallies on an exit, and this world declares none"):
        block_for("evacuation", "normal", BT_BASE, zones=[])


def test_approach_drives_only_the_agent_it_injected():
    """Over the whole population it would be an evacuation toward the robot."""
    assert block_for("approach", "normal", BT_BASE)["waypoints"]["scope"] == "injected"


def test_formation_break_is_a_triggered_scatter_not_a_triggered_knob():
    """A trigger can only gate what is time-driven: the backend allocates a new agent id on
    every spawn, so kinematics cannot be rewritten mid-episode."""
    block = block_for("formation_break", "normal", BT_BASE)
    assert block["trigger"] == {"robot_within": 3.0}
    assert block["waypoints"]["mode"] == "shuffle"
    assert "knob" not in block


def test_every_trigger_in_the_catalogue_has_a_plan_to_fire():
    """The task generator refuses a trigger with nothing to gate; the catalogue must not emit
    one."""
    for case in CATALOGUE:
        if "trigger" in case.params:
            assert "waypoints" in case.params, case.id


def test_a_stampede_surges_back_to_work_not_out_the_exit():
    """The rally is what makes it a stampede rather than a dense standstill: a crowd needs a
    shared destination to surge."""
    zones = [_zone("central_hallway"), _zone("open_work_area", 4.0), _zone("meeting_room_1")]
    block = block_for("stampede", "break", BT_BASE, zones=zones)
    assert block["waypoints"]["target"] == "open_work_area"
    assert block["waypoints"]["mode"] == "rally"


def test_an_evacuation_and_a_stampede_head_opposite_ways():
    zones = [_zone("central_hallway"), _zone("open_work_area", 4.0)]
    out = block_for("evacuation", "break", BT_BASE, zones=zones)["waypoints"]["target"]
    back = block_for("stampede", "break", BT_BASE, zones=zones)["waypoints"]["target"]
    assert (out, back) == ("central_hallway", "open_work_area")


def test_the_work_target_is_the_largest_work_zone():
    """A crowd converges on the main work area, not on whichever one sorts first."""
    zones = [_zone("meeting_room_1", 1.0), _zone("open_work_area", 6.0)]
    assert work_target(zones) == "open_work_area"


def test_the_work_target_never_lands_in_a_break_zone():
    """`work_zones` is the same definition the generator used to place people *away* from
    here, so a surge back to work cannot pick the room they were on break in."""
    zones = [_zone("open_work_area", 1.0), _zone("kitchen", 9.0), _zone("waiting_area", 9.0)]
    assert work_target(zones) == "open_work_area"


def test_a_residential_world_surges_to_its_living_area():
    """No work zones at all, so the basis shifts and the living areas are the active ones -
    the same shift `populate` makes, rather than a second definition that could send the
    surge into a bedroom."""
    zones = [_zone("bedroom_1", 4.0), _zone("living_room", 2.0), _zone("kitchen", 1.0)]
    assert work_target(zones) == "living_room"


def test_a_stampede_is_refused_when_a_world_has_nowhere_to_work():
    with pytest.raises(NotApplicable, match="rallies on a working space"):
        block_for("stampede", "break", BT_BASE, zones=[])


def test_stampede_now_runs_on_break():
    assert "break" in BY_ID["stampede"].bases
    assert {c.id for c in cases_for("break", base_doc=BT_BASE)} >= {"stampede"}


def test_stampede_stays_off_night():
    """Four agents is not a surge."""
    assert "night" not in BY_ID["stampede"].bases


def test_only_rally_cases_declare_a_destination_kind():
    for case in CATALOGUE:
        if case.rally_to != "exit":
            plan = case.params.get("waypoints") or {}
            assert plan.get("mode") == "rally", case.id



# --- empty base --------------------------------------------------------------------------------

def test_an_injecting_case_on_an_empty_base_names_its_own_type_and_mesh():
    """The mode refuses to invent either and aborts every episode - 233 aborts, zero ok,
    measured on `empty` before this."""
    block = block_for("blocking", "empty", {"robots": [{"start": "x"}], "dynamic": []})
    assert (block["inject_type"], block["inject_model"]) == ("adult", "arenian")


def test_a_populated_base_still_adopts_from_its_crowd():
    """Copying a neighbour is the experimentally right default; only emptiness overrides it."""
    block = block_for("blocking", "normal", BT_BASE)
    assert "inject_type" not in block and "inject_model" not in block


def test_a_non_injecting_case_never_gets_an_inject_type():
    block = block_for("stubborn", "normal", {"dynamic": []})
    assert "inject_type" not in block
