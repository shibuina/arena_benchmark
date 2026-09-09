"""Prompt-driven edge-case scenario generation (`new_plan_2.md`).

    prompt + world + base scenario
        -> Step 1  Type A by lexicon          (lexicon.py, step_a.py)
        -> Step 2  Type D by model            (step_d.py)
        -> Step 3  Type B + C by model        (step_bc.py)
        -> scenario directory + decision record (pipeline.py)

Everything static is written into the scenario file; everything that depends on the live
robot becomes an `effects:` entry the `edge_case` task mode runs.
"""

from .pipeline import Result, generate, ladder, write_result
from .profiles import PROFILES, TIERS, Profile, derive, profile, tier_value
from .spec import Record, SchemaError

__all__ = ["PROFILES", "TIERS", "Profile", "Record", "Result", "SchemaError", "derive", "generate", "ladder", "profile", "tier_value", "write_result"]
