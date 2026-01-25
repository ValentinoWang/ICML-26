import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "Times New Roman"})

METHODS = ["only_weight", "only_correction", "mlp_correction", "ours_full"]
SCENARIO_TITLES = {
    "bayes": "Bayes Prior (n=5)",
    "ridge": "Ridge Regression",
    "complex": "Structural Bias",
}
LEGEND_KWARGS = {
    "default": {"loc": "best", "framealpha": 0.9},
    # Keep the complex scenario legend inside the axis to avoid crowding margins.
    "complex": {"loc": "upper right", "bbox_to_anchor": (0.98, 0.98), "framealpha": 0.9, "borderaxespad": 0.4},
}


def plot_trajectories(all_series: Dict[str, Dict[str, Dict[str, List[float]]]], out_path: pathlib.Path, n_seeds: int) -> None:
    scenarios = list(all_series.keys())
    if len(scenarios) == 3:
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        # Top row: bayes, ridge; Bottom row: complex spans both columns.
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :]),
        ]
    else:
        fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 4), sharey=False)
        if len(scenarios) == 1:
            axes = [axes]
    for ax, name in zip(axes, scenarios):
        series = all_series[name]
        gens = np.arange(1, len(next(iter(series.values()))["mean"]) + 1)
        for m in METHODS:
            mean = np.array(series[m]["mean"])
            std = np.array(series[m]["std"])
            ci = 1.96 * std / np.sqrt(n_seeds)
            ax.plot(gens, mean, label=m.replace("_", " ").title(), linewidth=2)
            ax.fill_between(gens, mean - ci, mean + ci, alpha=0.15)
        title = SCENARIO_TITLES.get(name, name)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$")
        ax.grid(alpha=0.3)
        if name == "complex":
            legend_cfg = LEGEND_KWARGS.get(name, LEGEND_KWARGS["default"])
            ax.legend(**legend_cfg)
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
        ax.bar(x, means, yerr=ci, capsize=4, tick_label=[m.replace("_", " ") for m in METHODS])
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
