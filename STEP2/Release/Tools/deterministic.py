"""
Backward-compatible imports for determinism utilities.

The main implementation lives in `Release/Common_Utils/deterministic.py`.
"""

from __future__ import annotations

from Common_Utils.deterministic import make_dataloader_seed, set_deterministic

__all__ = ["set_deterministic", "make_dataloader_seed"]

