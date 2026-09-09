"""The contract with Arena's scenario loader: what `populate` writes, `ScenarioView.load` reads.

Three forms this generator emits had no shipped user and therefore no coverage, and each fails
quietly rather than loudly if the loader stops accepting it:

* a **zone reference** as a pose (`pose: laboratory`) - every generated pedestrian now;
* **zone references as waypoints**, several names in one list, resolved independently;
* **`ref:` on a region**, which resolves a source or sink polygon from a zone name. Implemented
  in `World.zone_converter`, used by zero shipped scenarios and tested nowhere until now.

A malformed document cannot degrade silently - `_MODERN_ONLY_KEYS` makes the loader
raise rather than fall back to the legacy path - but a *dropped agent* can, which is what §5.2 of
`SCENARIO_AUTHORING.md` is about. So these assert on what came back, not merely that it parsed.

Marked `ros` because it needs `arena_simulation_setup` and its cattrs stack on the path.
"""

from __future__ import annotations

import shutil

import pytest

pytestmark = pytest.mark.ros

WorldIdentifier = pytest.importorskip(
    "arena_simulation_setup.tree.World", reason="needs a sourced environment"
).WorldIdentifier

WORLD = "hospital_1"


@pytest.fixture
def loaded(worlds_root, generated_scenario):
    """The generated `busy` scenario for `hospital_1`, round-tripped through the real loader."""
    import numpy as np

    view = WorldIdentifier(WORLD).resolve_sync()
    world = view.load()
    origins = view.level_origins() or {level_id: (0.0, 0.0) for level_id in world.levels}
    converter = world.compact_world(origins).zone_converter(np.random.default_rng(0))
    return view.scenario(generated_scenario).resolve_sync().load(converter=converter)


@pytest.fixture
def generated_scenario(worlds_root):
    """Write the generated document into the world, and take it away again."""
    import yaml
    from arena_benchmark.cli import _world_parts
    from arena_benchmark.scenario import (
        MIN_ROUTE_CLEARANCE,
        build,
        ref_safe_zones,
        robot_zones,
        route_points,
        screen_zones,
        world_statics,
    )
    from arena_benchmark.scenario.populate import bt_type_source

    root = worlds_root / WORLD
    zones, robot = _world_parts(worlds_root, WORLD)
    statics = world_statics(root)
    route = route_points(robot, zones)
    verdicts = screen_zones(
        zones, world_root=root, route=route, statics=statics,
        min_route_clearance=MIN_ROUTE_CLEARANCE, robot_zones=robot_zones(robot),
    )
    doc = build(
        "normal", zones=zones, robot=robot, seed=0,
        homes=ref_safe_zones(verdicts),
    )

    name = "pytest_room_centric"
    out = root / "scenarios" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "scenario.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))
    if bt_type_source().is_file():
        shutil.copy(bt_type_source(), out / bt_type_source().name)
    try:
        yield name
    finally:
        shutil.rmtree(out, ignore_errors=True)


@pytest.fixture
def worlds_root():
    """Where `WorldIdentifier` will actually look.

    Not derived from the package's own `__file__`: the module lives under `build/` while worlds
    resolve under `install/.../share`, and generating into the wrong one writes a scenario the
    loader never sees.
    """
    from pathlib import Path

    root = Path(WorldIdentifier(WORLD).resolve_sync().path).parent
    if not root.is_dir():
        pytest.skip("worlds not on this install")
    return root


def test_every_generated_agent_survives_the_load(loaded, generated_scenario, worlds_root):
    import yaml

    written = yaml.safe_load((worlds_root / WORLD / "scenarios" / generated_scenario / "scenario.yaml").read_text())
    # A two-element waypoint drops the agent silently, so a count mismatch is the symptom to
    # catch here rather than an exception.
    assert len(loaded.dynamic) == len(written["dynamic"])
    assert {a.name for a in loaded.dynamic} == {a["name"] for a in written["dynamic"]}


def test_a_zone_reference_resolves_to_a_point_inside_that_zone(loaded, generated_scenario, worlds_root):
    import yaml

    written = yaml.safe_load((worlds_root / WORLD / "scenarios" / generated_scenario / "scenario.yaml").read_text())
    referenced = {a["name"]: a["pose"] for a in written["dynamic"] if isinstance(a["pose"], str)}
    if not referenced:
        pytest.skip("no zone screened clean in this world")

    world = WorldIdentifier(WORLD).resolve_sync().load()
    polygons = {z.name: [(c.x, c.y) for c in z.corners] for level in world.levels.values() for z in level.zones}
    for agent in loaded.dynamic:
        zone = referenced.get(agent.name)
        if zone is None:
            continue
        assert _inside((agent.pose.position.x, agent.pose.position.y), polygons[zone]), (
            f"{agent.name} resolved outside {zone}"
        )


def test_every_waypoint_reference_resolves_to_its_own_zone(loaded, generated_scenario, worlds_root):
    import yaml

    written = yaml.safe_load((worlds_root / WORLD / "scenarios" / generated_scenario / "scenario.yaml").read_text())
    by_name = {a["name"]: a for a in written["dynamic"]}
    world = WorldIdentifier(WORLD).resolve_sync().load()
    polygons = {z.name: [(c.x, c.y) for c in z.corners] for level in world.levels.values() for z in level.zones}
    for agent in loaded.dynamic:
        names = by_name[agent.name]["waypoints"]
        assert len(agent.waypoints) == len(names)
        for waypoint, zone in zip(agent.waypoints, names, strict=True):
            assert isinstance(zone, str), f"{agent.name} carries a coordinate waypoint"
            assert _inside((waypoint.x, waypoint.y), polygons[zone]), f"{agent.name}: {zone} resolved outside"



def _inside(pt, poly):
    x, y = pt
    hit = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            hit = not hit
    return hit
