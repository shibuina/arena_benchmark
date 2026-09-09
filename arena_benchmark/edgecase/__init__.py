"""Writing scenario directories: base scenarios and the pruning of generated ones."""

from .materialize import GENERATED_MARKER, WouldOverwrite, prune_generated, write_base

__all__ = ["GENERATED_MARKER", "WouldOverwrite", "prune_generated", "write_base"]
