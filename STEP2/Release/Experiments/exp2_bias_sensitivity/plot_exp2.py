import argparse
import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
BASE_TABLES_DIR = ROOT / "Total_results" / "Tables" / SCRIPT_DIR.name
BASE_FIGURES_DIR = ROOT / "Total_results" / "Figures" / SCRIPT_DIR.name
TABLES_DIR = BASE_TABLES_DIR / "results"
FIGURES_DIR = BASE_FIGURES_DIR / "results"

# Serif fonts for paper-quality figs (prefer Times)
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
}
plt.rcParams.update(PAPER_STYLE)

LABEL_MAP = {
    "no_filter": "No Filter",
    "standard_filter": "Pointwise MLP",
    "mlp_filter": "Pointwise MLP",
    "mlp": "Pointwise MLP",
    "dst": "DST (Pointwise)",
    "k_center": "k-Center (Coreset)",
    "ours": "Ours",
}
METHOD_STYLE = {
    "no_filter": {"color": "#949494", "linestyle": (0, (4, 2)), "marker": "o", "linewidth": 1.6, "alpha": 0.8, "fill_alpha": 0.08},
    "standard_filter": {"color": "#D55E00", "linestyle": (0, (1, 2)), "marker": "s", "linewidth": 1.8, "alpha": 0.9, "fill_alpha": 0.08},
    "mlp_filter": {"color": "#D55E00", "linestyle": (0, (1, 2)), "marker": "s", "linewidth": 1.8, "alpha": 0.9, "fill_alpha": 0.08},
    "mlp": {"color": "#D55E00", "linestyle": (0, (1, 2)), "marker": "s", "linewidth": 1.8, "alpha": 0.9, "fill_alpha": 0.08},
    "dst": {"color": "#C21E56", "linestyle": (0, (4, 1, 1, 1)), "marker": "^", "linewidth": 1.8, "alpha": 0.9, "fill_alpha": 0.08},
    "k_center": {"color": "#2A9D8F", "linestyle": (0, (3, 2)), "marker": "v", "linewidth": 1.6, "alpha": 0.85, "fill_alpha": 0.08},
    "ours": {"color": "#0072B2", "linestyle": "-", "marker": "D", "linewidth": 2.5, "alpha": 1.0, "fill_alpha": 0.2},
}

DEFAULT_STYLE = {
    "color": "#333333",
    "linestyle": "-",
    "marker": "o",
    "linewidth": 1.6,
    "alpha": 0.8,
    "fill_alpha": 0.08,
}

METHODS = ["no_filter", "standard_filter", "dst", "k_center", "ours"]


def _style_for(method: str) -> Dict[str, object]:
    style = DEFAULT_STYLE.copy()
    style.update(METHOD_STYLE.get(method, {}))
    return style


def display_label(name: str) -> str:
    return LABEL_MAP.get(name, name.replace("_", " ").title())


def plot_param_curve(
    xs: List[float],
    finals: Dict[str, Dict[str, List[float]]],
    xlabel: str,
    title: str,
    out_path: pathlib.Path,
    log_x: bool = False,
    n_seeds: int = 1,
) -> None:
    # Align aspect ratio with Exp1 single-plot style for visual consistency.
    plt.figure(figsize=(5.5, 4.5))
    for label, stats in finals.items():
        style = _style_for(label)
        ci = 1.96 * np.array(stats["std"]) / np.sqrt(n_seeds)
        plt.errorbar(
            xs,
            stats["mean"],
            yerr=ci,
            marker=style["marker"],
            linewidth=style["linewidth"],
            capsize=4,
            label=display_label(label),
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=style["alpha"],
        )
    plt.xlabel(xlabel)
    plt.ylabel("Final error (mean of last window)")
    if log_x:
        plt.xscale("log")
    # Remove explicit title for compact paper layout
    # plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_trajectories_grid(
    all_series: Dict[float, Dict[str, Dict[str, List[float]]]],
    param_label: str,
    out_path: pathlib.Path,
    suptitle: str | None = None,
    n_seeds: int = 1,
) -> None:
    values = list(all_series.keys())
    n = len(values)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), sharex=True, sharey=False)
    axes = axes.flatten()
    for idx, val in enumerate(values):
        ax = axes[idx]
        series = all_series[val]
        gens = np.arange(1, len(next(iter(series.values()))["mean"]) + 1)
        for m in METHODS:
            style = _style_for(m)
            mean = np.array(series[m]["mean"])
            std = np.array(series[m]["std"])
            ci = 1.96 * std / np.sqrt(n_seeds)
            ax.plot(
                gens,
                mean,
                label=display_label(m),
                linewidth=style["linewidth"],
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
            )
            ax.fill_between(gens, mean - ci, mean + ci, alpha=style["fill_alpha"], color=style["color"])
        # Drop subplot title for compact layout
        # ax.set_title(f"{param_label}={val}")
        ax.set_xlabel("Generation")
        ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$")
        ax.grid(alpha=0.3)
        ax.legend()
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    if suptitle:
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_by_method(
    all_series: Dict[float, Dict[str, Dict[str, List[float]]]],
    param_list: List[float],
    param_label: str,
    out_path: pathlib.Path,
    n_seeds: int = 1,
) -> None:
    n_methods = len(METHODS)
    fig, axes = plt.subplots(1, n_methods, figsize=(3.5 * n_methods, 4), sharex=True, sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for idx, m in enumerate(METHODS):
        ax = axes[idx]
        for val in param_list:
            stats = all_series[val][m]
            gens = np.arange(1, len(stats["mean"]) + 1)
            mean = np.array(stats["mean"])
            std = np.array(stats["std"])
            ci = 1.96 * std / np.sqrt(n_seeds)
            style = _style_for(m)
            ax.plot(
                gens,
                mean,
                label=f"{param_label}={val}",
                linewidth=style["linewidth"],
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=style["alpha"],
            )
            ax.fill_between(gens, mean - ci, mean + ci, alpha=style["fill_alpha"], color=style["color"])
        # Drop subplot title for compact layout
        # ax.set_title(display_label(m))
        ax.set_xlabel("Generation")
        ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _read_summary(path: pathlib.Path) -> Tuple[List[float], Dict[str, Dict[str, List[float]]], str]:
    rows = path.read_text().strip().splitlines()
    header = rows[0].split(",")
    param_name = header[0]
    finals: Dict[str, Dict[str, List[float]]] = {}
    params: List[float] = []
    for line in rows[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        params.append(float(parts[0]))
        for name, val in zip(header[1:], parts[1:]):
            method, stat = name.rsplit("_", 1)
            finals.setdefault(method, {"mean": [], "std": []})[stat].append(float(val))
    return params, finals, param_name


def _read_series_csv(path: pathlib.Path) -> Dict[str, Dict[str, List[float]]]:
    rows = path.read_text().strip().splitlines()
    header = rows[0].split(",")
    methods = []
    for h in header[1:]:
        base, stat = h.rsplit("_", 1)
        if stat == "mean":
            methods.append(base)
    series: Dict[str, Dict[str, List[float]]] = {m: {"mean": [], "std": []} for m in methods}
    for line in rows[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        for m in methods:
            series[m]["mean"].append(float(parts[header.index(f"{m}_mean")]))
            series[m]["std"].append(float(parts[header.index(f"{m}_std")]))
    return series


def _read_exp1_const(path: pathlib.Path) -> tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    rows = path.read_text().strip().splitlines()
    header = rows[0].split(",")
    gens = np.array([float(r.split(",")[0]) for r in rows[1:]])

    def _col(name: str) -> np.ndarray:
        idx = header.index(name)
        return np.array([float(r.split(",")[idx]) for r in rows[1:]], dtype=float)

    series = {
        "no_filter": {"mean": _col("no_filter_b0.5_mean"), "std": _col("no_filter_b0.5_std")},
        "standard_filter": {
            "mean": _col("standard_filter_b0.5_mean"),
            "std": _col("standard_filter_b0.5_std"),
        },
        "ours": {"mean": _col("ours_b0.5_mean"), "std": _col("ours_b0.5_std")},
    }
    return gens, series


def plot_theory_validation_triptych(
    exp1_const_csv: pathlib.Path,
    bias_summary_csv: pathlib.Path,
    ridge_summary_csv: pathlib.Path,
    out_path: pathlib.Path,
    n_seeds: int = 8,
) -> None:
    gens, exp1_series = _read_exp1_const(exp1_const_csv)
    bias_list, bias_finals, _ = _read_summary(bias_summary_csv)
    alpha_list, ridge_finals, _ = _read_summary(ridge_summary_csv)

    style_map = {
        "no_filter": {"color": "#7f8c8d", "linestyle": (0, (6, 2)), "marker": "o", "linewidth": 2.6},
        "standard_filter": {"color": "#d62728", "linestyle": (0, (2, 2)), "marker": "^", "linewidth": 2.6},
        "ours": {"color": "#1f77b4", "linestyle": "-", "marker": "D", "linewidth": 2.8},
    }
    label_map = {
        "no_filter": "No Filter",
        "standard_filter": "Pointwise Filter",
        "ours": "Set-Aware (Ours)",
    }

    # This figure is included as a full-width (two-column) triptych in the paper.
    # Use a canvas size close to the final LaTeX size to avoid downscaling that
    # makes ticks/labels unreadably small.
    triptych_style = dict(PAPER_STYLE)
    triptych_style.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    with plt.rc_context(triptych_style):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))

        # (a) Fixed bias trajectories
        ax = axes[0]
        for key in ["no_filter", "standard_filter", "ours"]:
            style = style_map[key]
            mean = exp1_series[key]["mean"]
            std = exp1_series[key]["std"]
            ci = 1.96 * std / np.sqrt(n_seeds)
            ax.plot(
                gens,
                mean,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=label_map[key],
            )
            ax.fill_between(gens, mean - ci, mean + ci, color=style["color"], alpha=0.12)
        ax.set_title(r"(a) Fixed Bias ($\beta=0.5$)", pad=3)
        ax.set_xlabel("Generation")
        ax.set_ylabel(r"$\|\theta_t - \theta^*\|_2$")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        ax.tick_params(axis="both", which="major", pad=2)

        # (b) Ridge sensitivity (final error)
        ax = axes[1]
        for key in ["no_filter", "standard_filter", "ours"]:
            style = style_map[key]
            stats = ridge_finals[key]
            ci = 1.96 * np.array(stats["std"]) / np.sqrt(n_seeds)
            ax.errorbar(
                alpha_list,
                stats["mean"],
                yerr=ci,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                capsize=3,
                label=label_map[key],
            )
        ax.set_title(r"(b) Ridge Sensitivity ($\alpha$)", pad=3)
        ax.set_xlabel(r"$\alpha$ (log-scale)")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        ax.tick_params(axis="both", which="major", pad=2)

        # (c) Additive bias sensitivity (final error)
        ax = axes[2]
        for key in ["no_filter", "standard_filter", "ours"]:
            style = style_map[key]
            stats = bias_finals[key]
            ci = 1.96 * np.array(stats["std"]) / np.sqrt(n_seeds)
            ax.errorbar(
                bias_list,
                stats["mean"],
                yerr=ci,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                capsize=3,
                label=label_map[key],
            )
        ax.set_title(r"(c) Additive Bias Sensitivity ($\beta$)", pad=3)
        ax.set_xlabel(r"$\beta$ (linear)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        ax.tick_params(axis="both", which="major", pad=2)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.18),
            handlelength=2.6,
            columnspacing=1.6,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.82])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _load_param_series(result_dir: pathlib.Path, prefix: str) -> Dict[float, Dict[str, Dict[str, List[float]]]]:
    series: Dict[float, Dict[str, Dict[str, List[float]]]] = {}
    for path in sorted(result_dir.glob(f"{prefix}*")):
        if not path.is_dir():
            continue
        suffix = path.name[len(prefix) :]
        try:
            key = float(suffix)
        except ValueError:
            continue
        traj = path / "trajectories.csv"
        if traj.exists():
            series[key] = _read_series_csv(traj)
    return dict(sorted(series.items()))


def _load_exp23_series(result_dir: pathlib.Path) -> Dict[int, Dict[float, Dict[str, Dict[str, List[float]]]]]:
    out: Dict[int, Dict[float, Dict[str, Dict[str, List[float]]]]] = {}
    for path in sorted(result_dir.glob("exp2_2.3_n*_delta*")):
        if not path.is_dir():
            continue
        name = path.name[len("exp2_2.3_n") :]
        if "_delta" not in name:
            continue
        n_str, delta_str = name.split("_delta", 1)
        try:
            n_val = int(n_str)
            delta_val = float(delta_str)
        except ValueError:
            continue
        traj = path / "trajectories.csv"
        if traj.exists():
            out.setdefault(n_val, {})[delta_val] = _read_series_csv(traj)
    return out


def regenerate_from_results(
    result_dir: pathlib.Path | None = None,
    figure_dir: pathlib.Path | None = None,
    n_seeds: int = 8,
) -> None:
    result_dir = result_dir or TABLES_DIR
    figure_dir = figure_dir or FIGURES_DIR
    # Exp 2.1
    bias_path = result_dir / "exp2_2.1_bias_summary.csv"
    if bias_path.exists():
        bias_list, exp21_finals, _ = _read_summary(bias_path)
        exp21_series = _load_param_series(result_dir, "exp2_2.1_bias_")
        plot_param_curve(
            bias_list,
            exp21_finals,
            xlabel=r"$\|\mathbf{b}_{const}\|_2$",
            title="Exp 2.1: Bias magnitude sensitivity",
            out_path=figure_dir / "exp2_2.1_bias_vs_error.png",
            n_seeds=n_seeds,
        )
        if exp21_series:
            plot_trajectories_grid(
                exp21_series,
                param_label="||b||",
                out_path=figure_dir / "exp2_2.1_trajs_by_bias.png",
                n_seeds=n_seeds,
            )
            plot_by_method(
                exp21_series,
                bias_list,
                param_label="||b||",
                out_path=figure_dir / "exp2_2.1_trajs_by_method.png",
                n_seeds=n_seeds,
            )

    # Exp 2.2
    ridge_path = result_dir / "exp2_2.2_ridge_summary.csv"
    if ridge_path.exists():
        alpha_list, exp22_finals, _ = _read_summary(ridge_path)
        exp22_series = _load_param_series(result_dir, "exp2_2.2_alpha_")
        plot_param_curve(
            alpha_list,
            exp22_finals,
            xlabel=r"$\alpha$ (ridge)",
            title="Exp 2.2: Ridge regularization sensitivity",
            out_path=figure_dir / "exp2_2.2_alpha_vs_error.png",
            log_x=True,
            n_seeds=n_seeds,
        )
        if exp22_series:
            plot_trajectories_grid(
                exp22_series,
                param_label="alpha",
                out_path=figure_dir / "exp2_2.2_trajs_by_alpha.png",
                n_seeds=n_seeds,
            )
            plot_by_method(
                exp22_series,
                alpha_list,
                param_label="alpha",
                out_path=figure_dir / "exp2_2.2_trajs_by_method.png",
                n_seeds=n_seeds,
            )

    # Exp 2.3
    prior_summaries = sorted(result_dir.glob("exp2_2.3_prior_summary_n*.csv"), key=lambda p: int(p.stem.split("_n")[1]))
    finals_by_n: Dict[int, Tuple[List[float], Dict[str, Dict[str, List[float]]]]] = {}
    sample_sizes: List[int] = []
    for path in prior_summaries:
        n_val = int(path.stem.split("_n")[1])
        deltas, finals, _ = _read_summary(path)
        finals_by_n[n_val] = (deltas, finals)
        sample_sizes.append(n_val)
    if finals_by_n:
        fig, axes = plt.subplots(1, len(sample_sizes), figsize=(6 * len(sample_sizes), 4), sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        for ax, n_val in zip(axes, sample_sizes):
            deltas, finals = finals_by_n[n_val]
            for m, stats in finals.items():
                style = _style_for(m)
                ci = 1.96 * np.array(stats["std"]) / np.sqrt(n_seeds)
                ax.errorbar(
                    deltas,
                    stats["mean"],
                    yerr=ci,
                    marker=style["marker"],
                    linewidth=style["linewidth"],
                    capsize=4,
                    label=display_label(m),
                    color=style["color"],
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                )
            # Drop subplot title for compact layout
            # ax.set_title(f"n={n_val}")
            ax.set_xlabel(r"$\|\delta\|$")
            ax.set_ylabel("Final error (mean tail)")
            ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
            ax.legend()
        # Remove suptitle for compact layout
        fig.tight_layout()
        fig.savefig(figure_dir / "exp2_2.3_prior_offset_curves.png", dpi=200)
        for ax in axes:
            ax.set_yscale("log")
        fig.savefig(figure_dir / "exp2_2.3_delta_log.png", dpi=200)
        plt.close(fig)

    exp23_series = _load_exp23_series(result_dir)
    for n_val, series in exp23_series.items():
        plot_trajectories_grid(
            series,
            param_label=r"$\delta$",
            out_path=figure_dir / f"exp2_2.3_trajs_n{n_val}.png",
            suptitle=f"Exp 2.3 trajectories (n={n_val})",
            n_seeds=n_seeds,
        )

    exp1_const = ROOT / "Total_results" / "Tables" / "exp1_bias_sources" / "results" / "exp1_1.1_const.csv"
    bias_summary = result_dir / "exp2_2.1_bias_summary.csv"
    ridge_summary = result_dir / "exp2_2.2_ridge_summary.csv"
    if exp1_const.exists() and bias_summary.exists() and ridge_summary.exists():
        out_path = ROOT.parent / "Paper" / "tex" / "Figures" / "exp3_theory_validation.png"
        plot_theory_validation_triptych(exp1_const, bias_summary, ridge_summary, out_path, n_seeds=n_seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Exp2 figures from existing results.")
    parser.add_argument("--results", type=pathlib.Path, default=TABLES_DIR, help="Result directory containing CSVs.")
    parser.add_argument("--fig-dir", type=pathlib.Path, default=FIGURES_DIR, help="Output directory for figures.")
    parser.add_argument("--n-seeds", type=int, default=8, help="Number of seeds used (for CI scaling).")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate all exp2 figures from CSV results.")
    args = parser.parse_args()
    if args.regenerate:
        regenerate_from_results(args.results, args.fig_dir, n_seeds=args.n_seeds)


if __name__ == "__main__":
    main()
