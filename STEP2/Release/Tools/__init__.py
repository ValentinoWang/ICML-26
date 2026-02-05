"""
Compatibility package.

Some experiment scripts in the full workspace import `Tools.deterministic`.
The compact ICML `Release/` package keeps utilities under `Common_Utils/`,
so we provide a tiny shim to avoid import errors without duplicating logic.
"""

