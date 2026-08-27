"""The edge-case catalogue and the writer that materialises it."""

from .catalogue import (
    BY_ID,
    CATALOGUE,
    Capability,
    Case,
    NotApplicable,
    applicable,
    block_for,
    bt_capable,
    cases_for,
    rally_target,
    regions_capable,
)
from .materialize import annotate, case_scenario_id, prune_generated, write, write_base
from .variants import CASE_VARIANTS, DELAY_S, Variant, backdrop_for, door_blocker, variants_for

__all__ = [
    "BY_ID",
    "CASE_VARIANTS",
    "CATALOGUE",
    "DELAY_S",
    "Capability",
    "Case",
    "NotApplicable",
    "Variant",
    "annotate",
    "applicable",
    "backdrop_for",
    "block_for",
    "bt_capable",
    "case_scenario_id",
    "cases_for",
    "door_blocker",
    "prune_generated",
    "rally_target",
    "regions_capable",
    "variants_for",
    "write",
    "write_base",
]
