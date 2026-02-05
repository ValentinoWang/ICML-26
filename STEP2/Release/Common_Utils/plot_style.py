"""
Centralized paper figure styling helpers.

Motivation: keep a *single* method→(color/linestyle/marker) mapping across all plots
so readers don't have to relearn legends from figure to figure (especially for:
No Filter / Pointwise / Dispersion / Set-Aware).
"""

from __future__ import annotations

from typing import Dict

# Canonical method keys used across the repo.
CANONICAL_METHOD_ORDER = ["no_filter", "pointwise", "dispersion", "set_aware"]

# --- Method → display name ---
METHOD_LABEL: Dict[str, str] = {
    "no_filter": "No Filter",
    "pointwise": "Pointwise",
    "dispersion": "Dispersion",
    "set_aware": "Set-Aware",
}

# --- Method → color ---
# Keep the common red/green/blue/gray palette, but fix the mapping:
#   No Filter  → gray
#   Pointwise  → red
#   Dispersion → green
#   Set-Aware  → blue
METHOD_COLOR: Dict[str, str] = {
    "no_filter": "#7f8c8d",  # gray
    "pointwise": "#d62728",  # red
    "dispersion": "#2ca02c",  # green
    "set_aware": "#1f77b4",  # blue
}

# --- Method → line style ---
# Make styles distinct even in grayscale print.
METHOD_LINESTYLE: Dict[str, object] = {
    "no_filter": "--",
    "pointwise": ":",
    "dispersion": "-.",
    "set_aware": "-",
}

# --- Method → marker ---
METHOD_MARKER: Dict[str, str] = {
    "no_filter": "o",
    "pointwise": "^",
    "dispersion": "s",
    "set_aware": "D",
}

# --- Aliases (various experiment keys → canonical key) ---
_ALIASES: Dict[str, str] = {
    # No filter
    "no_filter": "no_filter",
    "nofilter": "no_filter",
    "baseline": "no_filter",
    # Pointwise
    "pointwise": "pointwise",
    "standard_filter": "pointwise",
    "mlp_filter": "pointwise",
    "mlp": "pointwise",
    # Dispersion / diversity-only geometry
    "dispersion": "dispersion",
    # Set-aware / ours
    "set_aware": "set_aware",
    "set-aware": "set_aware",
    "setaware": "set_aware",
    "ours": "set_aware",
    "ours_full": "set_aware",
    "ours_small": "set_aware",
}


def canonical_method(method: str) -> str:
    key = method.strip().lower().replace(" ", "_")
    return _ALIASES.get(key, key)


def method_label(method: str, default: str | None = None) -> str:
    canon = canonical_method(method)
    return METHOD_LABEL.get(canon, default or method)


def method_style(method: str, *, label: str | None = None) -> Dict[str, object]:
    """
    Style dict with Matplotlib-friendly keys: {label, color, linestyle, marker}.
    """
    canon = canonical_method(method)
    if canon not in METHOD_COLOR:
        return {"label": label or method}
    return {
        "label": label or METHOD_LABEL[canon],
        "color": METHOD_COLOR[canon],
        "linestyle": METHOD_LINESTYLE[canon],
        "marker": METHOD_MARKER[canon],
    }


def method_style_short(method: str, *, label: str | None = None) -> Dict[str, object]:
    """
    Style dict for scripts that store short keys: {label, c, ls, mk}.
    """
    style = method_style(method, label=label)
    return {
        "label": style.get("label", method),
        "c": style.get("color", "#333333"),
        "ls": style.get("linestyle", "-"),
        "mk": style.get("marker", "o"),
    }
