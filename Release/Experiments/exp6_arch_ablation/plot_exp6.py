import pathlib
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER_TEX_DIR = PROJECT_ROOT / "Paper" / "tex"
if str(PAPER_TEX_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_TEX_DIR))

from plot_style import apply_icml_style

METHODS = ["only_weight", "only_correction", "mlp_correction", "ours_full"]
METHOD_LABELS = {
    "only_weight": "Weight-only",
    "only_correction": "Correction-only",
    "mlp_correction": "Corr-MLP (no set interaction)",
    "ours_full": "Full (Set-aware)",
}
METHOD_STYLES = {
    "only_weight": {"color": "#d62728", "linestyle": "--", "marker": "^"},
    "only_correction": {"color": "#2ca02c", "linestyle": ":", "marker": "s"},
    "mlp_correction": {"color": "#7f8c8d", "linestyle": "--", "marker": "o"},
    "ours_full": {"color": "#1f77b4", "linestyle": "-", "marker": "D"},
}
SCENARIO_TITLES = {
    "bayes": "Bayes Prior (n=5)",
    "ridge": "Ridge Regression",
    "complex": "Structural Bias",
    "complex_n32": "Structural Bias (N=32)",
}
LEGEND_KWARGS = {
    "default": {"loc": "best", "frameon": False},
    "complex": {"loc": "upper right", "bbox_to_anchor": (0.98, 0.98), "frameon": False, "borderaxespad": 0.4},
    "complex_n32": {"loc": "upper right", "bbox_to_anchor": (0.98, 0.98), "frameon": False, "borderaxespad": 0.4},
}


def plot_trajectories(all_series: Dict[str, Dict[str, Dict[str, List[float]]]], out_path: pathlib.Path, n_seeds: int) -> None:
    apply_icml_style(base_fontsize=11)
    preferred_order = ["bayes", "ridge", "complex", "complex_n32"]
    scenarios = [s for s in preferred_order if s in all_series] + [s for s in all_series.keys() if s not in preferred_order]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.6 * len(scenarios), 3.4), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]

    use_global_legend = len(scenarios) > 1
    panel_specs = [(ax, name, SCENARIO_TITLES.get(name, name), name == scenarios[-1]) for ax, name in zip(axes, scenarios)]

    for ax, name, title, show_legend in panel_specs:
        series = all_series[name]
        gens = np.arange(1, len(next(iter(series.values()))["mean"]) + 1)

        for m in METHODS:
            mean = np.array(series[m]["mean"])
            std = np.array(series[m]["std"])
            ci = 1.96 * std / np.sqrt(n_seeds)
            lower = mean - ci
            upper = mean + ci
            style = METHOD_STYLES.get(m, {})
            markevery = 20
            plot_style = dict(style)
            if "marker" in plot_style:
                plot_style.update(
                    {
                        "markevery": markevery,
                        "markersize": 4,
                        "markeredgewidth": 0.6,
                    }
                )
            ax.plot(gens, mean, label=METHOD_LABELS[m], linewidth=2, **plot_style)
            if m != "mlp_correction":
                ax.fill_between(gens, lower, upper, alpha=0.15, color=style.get("color"))

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$")
        ax.grid(alpha=0.3)

        if show_legend and not use_global_legend:
            legend_cfg = LEGEND_KWARGS.get(name, LEGEND_KWARGS["default"])
            ax.legend(**legend_cfg)

    for idx, ax in enumerate(axes):
        if idx != 0:
            ax.set_ylabel("")
    if use_global_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(handles),
                frameon=False,
                bbox_to_anchor=(0.5, 1.02),
                columnspacing=1.2,
                handletextpad=0.6,
            )
        fig.tight_layout(rect=[0, 0, 1, 0.88])
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_tail_bar(tails: Dict[str, Dict[str, Dict[str, float]]], out_path: pathlib.Path, n_seeds: int) -> None:
    scenarios = list(tails.keys())
    x = np.arange(len(METHODS))
    width = 0.35
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 4), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, name in zip(axes, scenarios):
        means = [tails[name][m]["mean"] for m in METHODS]
        stds = [tails[name][m]["std"] for m in METHODS]
        ci = 1.96 * np.array(stds) / np.sqrt(n_seeds)
        colors = [METHOD_STYLES.get(m, {}).get("color") for m in METHODS]
        ax.bar(x, means, yerr=ci, capsize=4, tick_label=[METHOD_LABELS[m] for m in METHODS], color=colors)
        ax.set_ylabel("Tail error (mean of last window)")
        ax.set_title(SCENARIO_TITLES.get(name, name))
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_set_size_trajs(
    set_size_series: Dict[int, Dict[str, Dict[str, List[float]]]],
    out_path: pathlib.Path,
    n_seeds: int,
    scenario: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for set_size in sorted(set_size_series.keys()):
        series = set_size_series[set_size]
        gens = np.arange(1, len(series["ours_full"]["mean"]) + 1)
        mean = np.array(series["ours_full"]["mean"])
        std = np.array(series["ours_full"]["std"])
        ci = 1.96 * std / np.sqrt(n_seeds)
        ax.plot(gens, mean, label=f"N={set_size}", linewidth=2)
        ax.fill_between(gens, mean - ci, mean + ci, alpha=0.15)
    title = f"Set Size Ablation ({SCENARIO_TITLES.get(scenario, scenario)})"
    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
