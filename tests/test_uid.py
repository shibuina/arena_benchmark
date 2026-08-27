"""Identity rules. These are what make a re-run incremental instead of a rebuild."""

from __future__ import annotations

from arena_benchmark.spec import cell_uid, episode_uid

CELL = dict(
    world="w",
    scenario="default",
    knob="",
    level=0.0,
    params={"task.edge_case.inject": 1, "task.edge_case.approach_angle": 180.0},
    context={"sim": "dummy"},
)


def test_cell_uid_is_stable_across_key_order():
    reordered = dict(CELL, params={"task.edge_case.approach_angle": 180.0, "task.edge_case.inject": 1})
    assert cell_uid(**CELL) == cell_uid(**reordered)


def test_cell_uid_changes_when_a_parameter_changes():
    other = dict(CELL, params={**CELL["params"], "task.edge_case.approach_angle": 90.0})
    assert cell_uid(**CELL) != cell_uid(**other)


def test_cell_uid_separates_grid_cells_that_share_parameters():
    """A grid expands into cells whose params are identical; only knob and level differ."""
    a = cell_uid(**dict(CELL, knob="need_decay", level=0.95))
    b = cell_uid(**dict(CELL, knob="need_decay", level=0.99))
    c = cell_uid(**dict(CELL, knob="step_patience", level=0.95))
    assert len({a, b, c}) == 3


def test_cell_uid_changes_with_the_run_context():
    """A cell measured under gazebo is not the same cell measured under dummy."""
    assert cell_uid(**CELL) != cell_uid(**dict(CELL, context={"sim": "gazebo"}))


def test_episode_uid_separates_arms_and_seeds():
    cuid = cell_uid(**CELL)
    a = episode_uid(cell=cuid, arm_id="nav2", arm={"robot": "jackal"}, seed=1)
    b = episode_uid(cell=cuid, arm_id="cadrl", arm={"robot": "jackal"}, seed=1)
    c = episode_uid(cell=cuid, arm_id="nav2", arm={"robot": "jackal"}, seed=2)
    assert len({a, b, c}) == 3


def test_editing_an_arm_in_place_invalidates_its_episodes():
    """Otherwise a rows measured on one chassis would be reused for another."""
    cuid = cell_uid(**CELL)
    before = episode_uid(cell=cuid, arm_id="a", arm={"robot": "jackal"}, seed=1)
    after = episode_uid(cell=cuid, arm_id="a", arm={"robot": "husky"}, seed=1)
    assert before != after
