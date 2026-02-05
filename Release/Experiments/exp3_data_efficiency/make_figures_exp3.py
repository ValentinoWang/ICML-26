import csv
import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
BASE_FIGURES_DIR = ROOT / "Total_results" / "Figures" / SCRIPT_DIR.name
TABLES_DIR = BASE_TABLES_DIR / "results"
FIGURES_DIR = BASE_FIGURES_DIR / "results"

# Use Times-like serif fonts (TeX Gyre/CMU) for CVPR/ICCV style.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)


def read_csv(path: pathlib.Path) -> Dict[str, List[float]]:
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        cols: Dict[str, List[float]] = {h: [] for h in header}
        for row in reader:
            for h, v in zip(header, row):
                cols[h].append(float(v))
    return cols


def plot_breaking_floor(const_data: Dict[str, List[float]], out_path: pathlib.Path) -> None:
    gens = np.array(const_data["generation"])
    ours = np.array(const_data["ours_small_mean"])
    no_filter = np.array(const_data["small_no_filter_mean"])
    baseline = np.array(const_data["baseline_big_mean"])

    plt.figure(figsize=(6, 4))
    plt.plot(gens, no_filter, label="No Filter (small N)", color="red", linewidth=2, linestyle="--")
    plt.plot(gens, baseline, label="No Filter (big N)", color="orange", linewidth=2, linestyle="-.")
    plt.plot(gens, ours, label="Ours (small N)", color="green", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel(r"$\|\theta_t - \theta^*\|_2$")
    plt.title("Breaking the Bias Floor (hard bias |b|=0.5)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_data_efficiency(ridge_data: Dict[str, List[float]], out_path: pathlib.Path, n_seeds: int = 8) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    _plot_data_efficiency_ax(ax, ridge_data, n_seeds=n_seeds)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_data_efficiency_ax(ax, ridge_data: Dict[str, List[float]], n_seeds: int = 8) -> None:
    ours_tail = ridge_data["ours_small_mean"][-1]
    no_small_tail = ridge_data["small_no_filter_mean"][-1]
    big_tail = ridge_data["baseline_big_mean"][-1]
    ours_err = ridge_data["ours_small_std"][-1] / np.sqrt(n_seeds)
    no_small_err = ridge_data["small_no_filter_std"][-1] / np.sqrt(n_seeds)
    big_err = ridge_data["baseline_big_std"][-1] / np.sqrt(n_seeds)
    labels = ["Small No Filter", "Small Ours", "Big No Filter"]
    values = [no_small_tail, ours_tail, big_tail]
    colors = ["red", "green", "gray"]
    errors = [no_small_err, ours_err, big_err]

    bars = ax.bar(labels, values, color=colors, yerr=errors, error_kw={"capsize": 4, "elinewidth": 1.2})
    max_height = max(v + e for v, e in zip(values, errors))
    label_margin = max_height * 0.08
    headroom = max_height * 0.3
    for b, v, e in zip(bars, values, errors):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + e + label_margin,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max_height + headroom)
    ax.set_ylabel(r"$\|\theta_T - \theta^*\|_2$")
    ax.set_title("Data Efficiency (Ridge, n=100 vs n=10k)")
    ax.grid(axis="y", alpha=0.3)


def plot_robustness_cost(
    bayes_n5: Dict[str, List[float]],
    bayes_n100: Dict[str, List[float]],
    out_path: pathlib.Path,
    n_seeds: int = 8,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_robustness_cost_ax(ax, bayes_n5, bayes_n100, n_seeds=n_seeds)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_robustness_cost_ax(
    ax,
    bayes_n5: Dict[str, List[float]],
    bayes_n100: Dict[str, List[float]],
    n_seeds: int = 8,
) -> None:
    sizes = [5, 100]
    ours = [bayes_n5["ours_small_mean"][-1], bayes_n100["ours_small_mean"][-1]]
    no_filter = [bayes_n5["small_no_filter_mean"][-1], bayes_n100["small_no_filter_mean"][-1]]
    ours_err = [bayes_n5["ours_small_std"][-1] / np.sqrt(n_seeds), bayes_n100["ours_small_std"][-1] / np.sqrt(n_seeds)]
    no_err = [
        bayes_n5["small_no_filter_std"][-1] / np.sqrt(n_seeds),
        bayes_n100["small_no_filter_std"][-1] / np.sqrt(n_seeds),
    ]

    x = np.arange(len(sizes))
    width = 0.35

    bars_no = ax.bar(
        x - width / 2, no_filter, width, label="No Filter", color="red", yerr=no_err, error_kw={"capsize": 4, "elinewidth": 1.2}
    )
    bars_ours = ax.bar(
        x + width / 2, ours, width, label="Ours", color="green", yerr=ours_err, error_kw={"capsize": 4, "elinewidth": 1.2}
    )

    max_height = max([v + e for v, e in zip(ours, ours_err)] + [v + e for v, e in zip(no_filter, no_err)])
    label_margin = max_height * 0.08
    annot_margin = max_height * 0.2
    for bars, errs, vals in zip([bars_no, bars_ours], [no_err, ours_err], [no_filter, ours]):
        for b, e, v in zip(bars, errs, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + e + label_margin,
                f"{b.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    top = max_height + annot_margin
    ratio = no_filter[0] / ours[0] if ours[0] > 0 else np.inf
    diff = ours[1] - no_filter[1]
    ax.set_ylim(0, top + annot_margin * 0.6)
    ax.text(
        x[0],
        max(ours[0] + ours_err[0], no_filter[0] + no_err[0]) + annot_margin,
        f"~{ratio:.1f}x lower at n=5",
        ha="center",
        fontsize=9,
        color="black",
    )
    ax.text(
        x[1],
        max(ours[1] + ours_err[1], no_filter[1] + no_err[1]) + annot_margin,
        f"+{diff:.3f} at n=100",
        ha="center",
        fontsize=9,
        color="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={s}" for s in sizes])
    ax.set_ylabel(r"$\|\theta_T - \theta^*\|_2$")
    ax.set_title("Robustness vs. Cost (Bayes)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()


def plot_data_efficiency_and_robustness(
    ridge_data: Dict[str, List[float]],
    bayes_n5: Dict[str, List[float]],
    bayes_n100: Dict[str, List[float]],
    out_path: pathlib.Path,
    n_seeds: int = 8,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _plot_data_efficiency_ax(axes[0], ridge_data, n_seeds=n_seeds)
    _plot_robustness_cost_ax(axes[1], bayes_n5, bayes_n100, n_seeds=n_seeds)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_results(
    all_series: Dict[str, Dict[str, Dict[str, List[float]]]],
    out_path: pathlib.Path,
    n_seeds: int,
    titles: Dict[str, str],
) -> None:
    scenarios = list(all_series.keys())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 4), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, name in zip(axes, scenarios):
        series = all_series[name]
        gens = np.arange(1, len(next(iter(series.values()))["mean"]) + 1)
        for key, style in [
            ("baseline_big", {"linestyle": "--"}),
            ("small_no_filter", {"linestyle": "-."}),
            ("ours_small", {"linestyle": "-"}),
        ]:
            mean = np.array(series[key]["mean"])
            std = np.array(series[key]["std"])
            ci = 1.96 * std / np.sqrt(n_seeds)
            ax.plot(gens, mean, label=f"{key.replace('_', ' ')}", linewidth=2, **style)
            ax.fill_between(gens, mean - ci, mean + ci, alpha=0.15)
        ax.set_title(titles.get(name, name))
        ax.set_xlabel("Generation")
        ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    const_data = read_csv(TABLES_DIR / "exp3_data_eff_const.csv")
    ridge_data = read_csv(TABLES_DIR / "exp3_data_eff_ridge.csv")
    bayes_n5 = read_csv(TABLES_DIR / "exp3_data_eff_bayes_n5.csv")
    bayes_n100 = read_csv(TABLES_DIR / "exp3_data_eff_bayes.csv")

    plot_breaking_floor(const_data, FIGURES_DIR / "exp3_fig_A_breaking_floor.png")
    plot_data_efficiency(ridge_data, FIGURES_DIR / "exp3_fig_B_data_efficiency.png")
    plot_robustness_cost(bayes_n5, bayes_n100, FIGURES_DIR / "exp3_fig_C_robustness_cost.png")
    plot_data_efficiency_and_robustness(
        ridge_data,
        bayes_n5,
        bayes_n100,
        FIGURES_DIR / "exp3_fig_BC_combined.png",
    )


if __name__ == "__main__":
    main()
