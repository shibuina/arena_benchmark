from __future__ import annotations

import math

import pytest

from arena_benchmark.edgecase import CATALOGUE, NotApplicable, block_for
from arena_benchmark.edgecase.variants import (
    CASE_VARIANTS,
    DELAY_S,
    DOOR_BLOCKER,
    Variant,
    backdrop_for,
    check_table as check_case_table,
    door_blocker,
    far_exit,
    merge_params,
    variants_for,
)
from arena_benchmark.scenario import (
    BASE_RECIPES,
    MIN_EPISODE_S,
    ROBOT_SPEED_MPS,
    build,
    build_regions,
    robot_for,
    route_seconds,
    topology,
)
from arena_benchmark.scenario.populate import MIN_PED_ROUTE_M
from arena_benchmark.scenario.variants import MAX_LEGS, BaseRecipe, check_table, distinct_axes
from arena_benchmark.spec.scenario_cells import read_block


def zone(name, x0, y0, x1, y1, doors=()):
    return {
        "name": name,
        "corners": [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}],
        "doors": list(doors),
    }


def door(name, x0, y0, x1, y1):
    return {"name": name, "start": {"x": x0, "y": y0, "z": 0}, "end": {"x": x1, "y": y1, "z": 0}}


# A building: a long east-west corridor with rooms above and below it, and a second corridor.
ZONES = [
    zone("main_hallway", 0.0, 14.0, 40.0, 18.0, doors=[door("office_1_door", 10.0, 14.0, 11.5, 14.0)]),
    zone("side_hallway", 42.0, 0.0, 46.0, 30.0),
    zone("office_1", 4.0, 4.0, 14.0, 12.0),
    zone("office_2", 18.0, 4.0, 28.0, 12.0),
    zone("meeting_room_1", 4.0, 20.0, 14.0, 28.0),
    zone("kitchen", 18.0, 20.0, 28.0, 28.0),
    zone("reception", 30.0, 4.0, 40.0, 12.0),
]
ROBOT = {"start": [2.0, 16.0, 0.0], "goal": [38.0, 16.0, 0.0]}
ROUTE = [(2.0, 16.0), (38.0, 16.0)]
RESOLVABLE = {z["name"]: 5000 for z in ZONES}
HOMES = frozenset(z["name"] for z in ZONES if z["name"] not in ("main_hallway", "side_hallway"))


# --- base recipes ----------------------------------------------------------------------

def test_the_base_table_is_distinct():
    check_table()


def test_every_profile_has_two_or_three_variants_except_empty():
    for profile, recipes in BASE_RECIPES.items():
        assert (len(recipes) == 1) if profile == "empty" else (2 <= len(recipes) <= 3), profile


def test_eleven_bases_per_world():
    assert sum(len(r) for r in BASE_RECIPES.values()) == 11


def test_the_first_variant_of_each_profile_is_todays_base():
    for profile, recipes in BASE_RECIPES.items():
        assert recipes[0].route == "fwd" and recipes[0].cast == "dispersed", profile


def test_siblings_sharing_two_axes_are_refused():
    a = BaseRecipe(id="x_a", profile="x", route="fwd", flow="cross", cast="dispersed", note="")
    b = BaseRecipe(id="x_b", profile="x", route="rev", flow="cross", cast="dispersed", note="")
    assert distinct_axes(a, b) == 1
    with pytest.raises(ValueError, match="at least two"):
        check_table({"x": (a, b)})


# --- the robot's route ---------------------------------------------------------------

def test_rev_swaps_the_ends_and_turns_the_robot_around():
    fwd = robot_for(BASE_RECIPES["normal"][0], ROBOT, ZONES)
    rev = robot_for(BASE_RECIPES["normal"][1], ROBOT, ZONES)
    assert fwd["start"][:2] == [2.0, 16.0]
    assert rev["start"][:2] == [38.0, 16.0]
    assert math.isclose(abs(rev["start"][2]), math.pi, abs_tol=1e-3)
    assert "goal" not in fwd and "goal" not in rev


def test_a_zone_named_robot_reverses_too():
    rev = robot_for(BASE_RECIPES["normal"][1], {"start": "reception", "goal": "office_1"}, ZONES)
    assert rev["start"] == "office_1"
    assert rev["phases"][0]["goto"] == "reception"


def test_a_short_route_is_shuttled_until_it_lasts_thirty_seconds():
    short = {"start": [2.0, 16.0, 0.0], "goal": [8.0, 16.0, 0.0]}
    spec = robot_for(BASE_RECIPES["normal"][0], short, ZONES)
    assert len(spec["phases"]) > 1
    assert route_seconds(spec, ZONES) >= MIN_EPISODE_S
    assert len(spec["phases"]) <= MAX_LEGS


def test_a_long_route_is_left_alone():
    spec = robot_for(BASE_RECIPES["normal"][0], ROBOT, ZONES)
    assert len(spec["phases"]) == 1
    assert route_seconds(spec, ZONES) >= MIN_EPISODE_S


def test_a_route_between_adjacent_points_stops_at_the_leg_cap():
    tiny = {"start": [2.0, 16.0, 0.0], "goal": [2.5, 16.0, 0.0]}
    spec = robot_for(BASE_RECIPES["normal"][0], tiny, ZONES)
    assert len(spec["phases"]) == MAX_LEGS


def test_a_shipped_marathon_is_cut_to_the_shortest_prefix_that_lasts():
    # The acoustics corridors ship a 200-leg hand-authored shuttle; copied verbatim that is a
    # 33-minute episode.
    legs = [[6.5, 1.5, 0.0], [1.5, 1.5, 3.142]] * 100
    marathon = {"start": [4.0, 1.5, 0.0], "phases": [{"goto": p} for p in legs]}
    spec = robot_for(BASE_RECIPES["normal"][0], marathon, ZONES)
    assert len(spec["phases"]) <= MAX_LEGS
    assert MIN_EPISODE_S <= route_seconds(spec, ZONES) < 120
    rev = robot_for(BASE_RECIPES["normal"][1], marathon, ZONES)
    assert rev["start"][:2] == [1.5, 1.5] and len(rev["phases"]) <= MAX_LEGS


def test_the_floor_holds_for_the_faster_controller():
    # Sized for the faster controller default, so the slower one only lengthens the episode.
    assert ROBOT_SPEED_MPS == 0.5


# --- cast and flow -------------------------------------------------------------------

def _pop(cast, profile="busy"):
    return build(profile, zones=ZONES, robot=ROBOT, seed=1, homes=HOMES, world="w", cast=cast)["dynamic"]


def test_clustered_packs_the_cast_into_two_zones():
    # `meeting_room_1` is a work zone by rule, and sits as close to the corridor as `office_1`.
    homes = {a["pose"] for a in _pop("clustered")}
    assert len(homes) <= 2
    assert homes <= {"office_1", "office_2", "reception", "meeting_room_1"}
    assert len({a["pose"] for a in _pop("dispersed")}) > 2


def test_stream_led_halves_the_cast():
    assert len(_pop("stream_led")) < len(_pop("dispersed"))


def test_topology_is_read_against_the_route_direction():
    assert topology((2.0, 16.0), (38.0, 16.0), ROUTE) == "co"
    assert topology((38.0, 16.0), (2.0, 16.0), ROUTE) == "counter"
    assert topology((10.0, 4.0), (10.0, 28.0), ROUTE) == "cross"
    assert topology((2.0, 16.0), (38.0, 16.0), list(reversed(ROUTE))) == "counter"


def test_a_requested_topology_is_honoured_where_the_world_allows():
    for want in ("cross", "counter", "co"):
        block = build_regions("busy", zones=ZONES, route=ROUTE, resolvable=RESOLVABLE, want=want)
        assert block, want
        from arena_benchmark.scenario import stream_topology

        assert stream_topology(block, ZONES, ROUTE) == want


def test_flow_none_means_no_stream_on_purpose():
    assert build_regions("night", zones=ZONES, route=ROUTE, resolvable=RESOLVABLE, want="none") == {}


def test_stream_led_streams_harder():
    plain = build_regions("busy", zones=ZONES, route=ROUTE, resolvable=RESOLVABLE)
    led = build_regions("busy", zones=ZONES, route=ROUTE, resolvable=RESOLVABLE, cast="stream_led")
    assert led["inflow"]["config"]["max_concurrent"] > plain["inflow"]["config"]["max_concurrent"]


def test_pedestrians_walk_at_least_the_route_floor_where_the_world_allows():
    def length(agent):
        centroid = {z["name"]: ((z["corners"][0]["x"] + z["corners"][2]["x"]) / 2,
                                (z["corners"][0]["y"] + z["corners"][2]["y"]) / 2) for z in ZONES}
        pts = [centroid[w] if isinstance(w, str) else (w[0], w[1]) for w in agent["waypoints"]]
        return sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False))

    # Talkers stand in their room by design; the floor is a walker's.
    walks = [length(a) for a in _pop("dispersed", "normal") if a["agent"]["agent_type"] == "adult"]
    assert walks
    assert min(walks) >= MIN_PED_ROUTE_M * 0.8


# --- case variants -------------------------------------------------------------------

def test_the_case_table_is_distinct():
    check_case_table()


def test_every_case_has_two_or_three_variants():
    assert set(CASE_VARIANTS) == {c.id for c in CATALOGUE}
    for case_id, variants in CASE_VARIANTS.items():
        assert 2 <= len(variants) <= 3, case_id


def test_the_ceiling_is_eighty_four_per_world():
    assert sum(len(variants_for(c.id)) * len(c.bases) for c in CATALOGUE) == 84


def test_siblings_sharing_a_mechanism_are_refused():
    twins = (Variant(id="a", mechanism="m", note=""), Variant(id="b", mechanism="m", note=""))
    with pytest.raises(ValueError, match="share mechanism"):
        check_case_table({"x": twins})


def test_backdrops_rotate_unless_pinned():
    stand, crawl, doorway = variants_for("blocking")
    assert [backdrop_for("normal", k, v).id for k, v in enumerate((stand, crawl, doorway))] == \
        ["normal_a", "normal_b", "normal_c"]
    assert backdrop_for("break", 2, doorway).id == "break_a"
    dense = variants_for("stampede")[2]
    assert backdrop_for("busy", 0, dense).cast == "stream_led"


def test_merge_keeps_nested_plans_and_honours_drops():
    base = {"inject": 1, "place": "intercept", "waypoints": {"mode": "shuffle", "period": 4.0, "radius": 3.0}}
    v = Variant(id="v", mechanism="m", note="", params={"waypoints": {"period": 2.0}}, drop=frozenset({"place"}))
    out = merge_params(base, v)
    assert out["waypoints"] == {"mode": "shuffle", "period": 2.0, "radius": 3.0}
    assert "place" not in out and out["inject"] == 1


BT_BASE = {"dynamic": [{"name": "d", "agent": {"agent_type": "./talker.yaml"}}],
           "regions": {"inflow": {"type": "source"}, "outflow": {"type": "sink"}}}
NO_REGIONS = {"dynamic": [{"name": "d", "agent": {"agent_type": "./talker.yaml"}}]}


def test_a_variant_block_carries_the_mechanism_in_its_id():
    crawl = variants_for("blocking")[1]
    block = block_for("blocking", "normal", BT_BASE, zones=ZONES, variant=crawl, route=ROUTE)
    assert block["id"] == "blocking_crawl"
    assert block["approach_angle"] == 0.0 and block["level"] == 0.2
    assert "crawl" in block["note"]


def test_time_driven_variants_are_delayed_and_others_are_not():
    jitter = variants_for("erratic")[0]
    assert block_for("erratic", "normal", BT_BASE, zones=ZONES, variant=jitter, route=ROUTE)["delay"] == DELAY_S
    stand = variants_for("blocking")[0]
    assert "delay" not in block_for("blocking", "normal", BT_BASE, zones=ZONES, variant=stand, route=ROUTE)
    early = variants_for("formation_break")[0]
    assert "delay" not in block_for("formation_break", "normal", BT_BASE, zones=ZONES, variant=early, route=ROUTE)


def test_the_delay_key_passes_the_benchmark_whitelist(tmp_path):
    import yaml

    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump({"robots": [], "edge_case": {"id": "x", "inject": 0, "delay": 10.0,
                                                               "waypoints": {"mode": "shuffle"}}}))
    assert read_block(path)["delay"] == 10.0


def test_the_doorway_blocker_stands_in_the_door_nearest_the_route():
    agent = door_blocker(ZONES, ROUTE)
    assert agent is not None
    assert agent["name"] == DOOR_BLOCKER
    assert agent["pose"][:2] == [10.75, 14.0]
    assert agent["waypoint_mode"] == "once"
    doorway = variants_for("blocking")[2]
    block = block_for("blocking", "normal", BT_BASE, zones=ZONES, variant=doorway, route=ROUTE)
    assert block["select"] == DOOR_BLOCKER and block["mode"] == "perturb"
    assert "place" not in block and "encounter_at" not in block


def test_no_door_near_the_route_refuses_the_doorway_variant():
    doorway = variants_for("blocking")[2]
    far = [(2.0, 40.0), (38.0, 40.0)]
    with pytest.raises(NotApplicable, match="door"):
        block_for("blocking", "normal", BT_BASE, zones=ZONES, variant=doorway, route=far)


def test_far_exit_needs_a_second_exit_and_takes_the_farther_one():
    assert far_exit(ZONES, ROUTE) == "side_hallway"
    one_exit = [z for z in ZONES if z["name"] != "side_hallway"]
    assert far_exit(one_exit, ROUTE) is None
    variant = variants_for("evacuation")[2]
    block = block_for("evacuation", "normal", BT_BASE, zones=ZONES, variant=variant, route=ROUTE)
    assert block["waypoints"]["target"] == "side_hallway"
    with pytest.raises(NotApplicable, match="second exit"):
        block_for("evacuation", "normal", BT_BASE, zones=one_exit, variant=variant, route=ROUTE)


def test_a_level_b_variant_needs_regions():
    dense = variants_for("stampede")[2]
    block = block_for("stampede", "busy", BT_BASE, zones=ZONES, variant=dense, route=ROUTE)
    assert block["knob"] == "crowd_density" and block["catalogue"] == "level_b"
    with pytest.raises(NotApplicable, match="regions"):
        block_for("stampede", "busy", NO_REGIONS, zones=ZONES, variant=dense, route=ROUTE)


def test_without_a_variant_block_for_is_unchanged():
    assert block_for("blocking", "normal", BT_BASE)["id"] == "blocking"
