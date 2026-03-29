import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.figsize": (6, 5.5),
    "figure.dpi": 300,
})

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = ROOT_DIR / "Total_results" / "Tables" / "exp11_gpt2_model" / "Results_streaming"
DEFAULT_OUT = ROOT_DIR / "Paper" / "LaTEX" / "icml2025" / "Figures" / "exp11_gpt2_ppl_annot.png"


def load_mean_ppl(results_root: pathlib.Path, methods: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    per_method = {m: {} for m in methods}
    max_gen = 0
    for seed_dir in sorted(results_root.iterdir()):
        if not seed_dir.is_dir():
            continue
        metrics_path = seed_dir / "metrics_diversity_ppl.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open() as f:
            data = json.load(f)
        history = data.get("history", {})
        for method in methods:
            for rec in history.get(method, []):
                gen = int(rec.get("generation", 0))
                max_gen = max(max_gen, gen)
                per_method[method].setdefault(gen, []).append(float(rec.get("val_ppl", np.nan)))

    if max_gen == 0 and all(not per_method[m] for m in methods):
        raise RuntimeError(f"No metrics found in {results_root}")

    generations = np.arange(0, max_gen + 1)
    means = {}
    for method in methods:
        means[method] = np.array([
            float(np.mean(per_method[method].get(gen, [np.nan])))
            for gen in range(max_gen + 1)
        ])
    return generations, means


def plot_right_panel_evidence(results_root: pathlib.Path, out_path: pathlib.Path) -> None:
    generations, means = load_mean_ppl(results_root, ("pointwise", "set_aware"))

    ppl_pointwise = means["pointwise"]
    ppl_ours = means["set_aware"]

    fig, ax = plt.subplots()

    all_ppl = np.concatenate([ppl_pointwise, ppl_ours])
    min_ppl = float(np.nanmin(all_ppl))
    max_ppl = float(np.nanmax(all_ppl))
    span = max_ppl - min_ppl
    if np.isfinite(span) and span > 0:
        stable_max = min_ppl + 0.25 * span
        collapse_min = min_ppl + 0.65 * span

        ax.axhspan(min_ppl, stable_max, color="#0173B2", alpha=0.08, linewidth=0)
        ax.axhspan(collapse_min, max_ppl, color="#D55E00", alpha=0.08, linewidth=0)
        ax.text(
            0.98,
            stable_max,
            "Stable regime",
            color="#0173B2",
            fontsize=11,
            fontweight="bold",
            ha="right",
            va="bottom",
            transform=ax.get_yaxis_transform(),
        )
        ax.text(
            0.02,
            collapse_min,
            "Collapse regime",
            color="#D55E00",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
            transform=ax.get_yaxis_transform(),
        )

    ax.plot(
        generations,
        ppl_pointwise,
        color="#d62728",
        linestyle=(0, (2, 2)),
        linewidth=3.2,
        marker="^",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="#d62728",
        markeredgewidth=1.4,
        label="Pointwise Filter",
    )
    ax.plot(
        generations,
        ppl_ours,
        color="#1f77b4",
        linestyle="-",
        linewidth=3.6,
        marker="D",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="#1f77b4",
        markeredgewidth=1.4,
        label="Set-Aware (Ours)",
    )

    ax.set_xlabel("Generation", fontsize=14, fontname="Times New Roman")
    ax.set_ylabel("Validation PPL (streaming)", fontsize=14, fontname="Times New Roman")

    ax.set_xticks(generations)
    ax.set_ylim(35, 240)

    # Use shaded regimes instead of hand-written annotations.

    ax.grid(True, which="both")

    ax.set_title("Quality (PPL) vs Generation", fontsize=14, fontname="Times New Roman", pad=6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    plot_right_panel_evidence(args.results_root, args.out)
