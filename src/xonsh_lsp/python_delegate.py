"""
Python language features delegate - backwards compatibility shim.

This module re-exports from jedi_backend and preprocessing for backwards
compatibility. New code should import from those modules directly.
"""

from __future__ import annotations

# Re-export JediBackend as PythonDelegate for backwards compatibility
from xonsh_lsp.jedi_backend import JEDI_AVAILABLE, JediBackend

# Re-export preprocessing utilities
from xonsh_lsp.preprocessing import (
    PreprocessResult,
    map_position_from_processed,
    map_position_to_processed,
    preprocess_source,
    preprocess_with_mapping,
)

# Backwards-compatible alias
PythonDelegate = JediBackend

__all__ = [
    "PythonDelegate",
    "JediBackend",
    "JEDI_AVAILABLE",
    "PreprocessResult",
    "preprocess_source",
    "preprocess_with_mapping",
    "map_position_to_processed",
    "map_position_from_processed",
]
