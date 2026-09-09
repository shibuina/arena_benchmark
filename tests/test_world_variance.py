from __future__ import annotations

import pytest
import yaml
from arena_benchmark.variance.world_features import (
    FEATURES,
    coverage,
    farthest_point_order,
    features,
    normalise,
    usable,
    variance_report,
)

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("PIL")

RES = 0.05


def _zone(name, x0, y0, x1, y1):
    return {"name": name, "corners": [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}], "walls": [], "doors": [], "entities": {"static": [], "dynamic": []}}


def _write_world(root, occupied, zones):
    from PIL import Image

    (root / "0").mkdir(parents=True)
    img = np.where(occupied, 0, 254).astype(np.uint8)
    Image.fromarray(img, mode="L").save(root / "0" / "map.png")
    (root / "0" / "map.yaml").write_text(yaml.safe_dump({"image": "map.png", "resolution": RES, "origin": [0.0, 0.0, 0.0], "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.196}))
    (root / "0" / "world.yaml").write_text(yaml.safe_dump({"zones": zones}))
    return root


def _open_hall(tmp_path):
    """A 20 x 20 m empty hall."""
    n = int(20 / RES)
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return _write_world(tmp_path / "hall", occ, [_zone("hall", 1, 1, 19, 19), _zone("a", 1, 1, 5, 5), _zone("b", 15, 15, 19, 19)])


def _maze(tmp_path):
    """The same 20 x 20 m, cut into four rooms by two walls with one 1.0 m door each."""
    n = int(20 / RES)
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    mid = n // 2
    occ[mid - 1:mid + 1, :] = True
    occ[:, mid - 1:mid + 1] = True
    door = int(1.0 / RES)
    occ[mid - 1:mid + 1, 40:40 + door] = False  # a door in the horizontal wall, west half
    occ[mid - 1:mid + 1, mid + 60:mid + 60 + door] = False  # and one in the east half
    occ[40:40 + door, mid - 1:mid + 1] = False  # a door in the vertical wall, north half
    zones = [_zone("nw", 1, 1, 9, 9), _zone("ne", 11, 1, 19, 9), _zone("sw", 1, 11, 9, 19), _zone("se", 11, 11, 19, 19)]
    return _write_world(tmp_path / "maze", occ, zones)


def test_rooms_narrowness_and_tortuosity_separate_a_hall_from_a_maze(tmp_path):
    hall = features(_open_hall(tmp_path))
    maze = features(_maze(tmp_path))
    assert set(hall) == set(FEATURES)
    assert hall["rooms"] == 1 and hall["doorways"] == 0
    assert maze["rooms"] == 4 and maze["doorways"] >= 2
    assert maze["narrow_fraction"] > hall["narrow_fraction"]
    assert maze["tortuosity"] > 1.15 > hall["tortuosity"] > 0.99
    assert hall["area_m2"] > maze["area_m2"] > 200
    assert hall["clearance_median"] > maze["clearance_median"]
    assert hall["free_fraction"] > 0.9
    assert usable(hall) and usable(maze)


def test_variance_report_orders_the_outlier_first_and_coverage_falls_to_zero():
    table = {
        "a": dict.fromkeys(FEATURES, 0.0),
        "b": dict.fromkeys(FEATURES, 0.0),
        "c": dict.fromkeys(FEATURES, 1.0),
    }
    table["b"]["area_m2"] = 0.1
    vectors = normalise(table)
    assert vectors["c"] == [1.0] * len(FEATURES)
    order = farthest_point_order(vectors)
    assert order[0] == "c" and set(order) == {"a", "b", "c"}
    assert coverage(vectors, order[:1]) > coverage(vectors, order[:2]) > coverage(vectors, order) == 0.0
    report = variance_report(table)
    assert report["order"] == order
    assert report["coverage_by_k"][-1] == 0.0
    assert report["distinctiveness"]["c"] > report["distinctiveness"]["a"]
    assert report["per_feature_spread"]["rooms"] > 0 and report["spread"] > 0


def test_a_constant_feature_contributes_nothing():
    table = {"a": dict.fromkeys(FEATURES, 2.0), "b": dict.fromkeys(FEATURES, 2.0)}
    report = variance_report(table)
    assert report["spread"] == 0.0
    assert all(v == 0.0 for v in report["per_feature_spread"].values())
    assert report["order"] == ["a", "b"]


def test_a_world_without_zones_is_not_usable():
    assert not usable({"zones": 1, "area_m2": 50.0})
    assert not usable({"zones": 5, "area_m2": 12.0})
    assert usable({"zones": 3, "area_m2": 50.0})
