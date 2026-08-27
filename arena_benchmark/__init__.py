"""Benchmark dataset generation for Arena edge-case scenarios.

Turns the `edge_case` task mode into a versioned benchmark: a portable scenario suite plus
the per-episode evidence that certifies each entry as one that separates planners.

The layers, in the order data moves through them:

``spec``
    A benchmark spec (cells x worlds x arms x seeds) validated and expanded into a manifest
    of episodes, each with a stable uid.
``orchestrate``
    Manifest rows grouped by what costs a relaunch, then run as sweep plans handed to
    ``arena launch ... sweep:=``. This module never re-implements trial execution.
``record``
    The navigation metrics the existing pipeline does not measure.
``assemble``
    The four output streams joined into one row per episode, and solved cases frozen into
    portable scenario files.
``analysis``
    Which cells actually separate the arms, and by how much.
``feedback``
    What the generator still cannot express.
"""

__version__ = "0.1.0"
